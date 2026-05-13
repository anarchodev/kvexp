//! Multi-tenant manifest over LMDB with per-tenant chains of nested
//! transactions.
//!
//! Architecture:
//!
//!   Manifest
//!   ├─ LMDB env (durable state)
//!   │    ├─ "_meta"   (raft_apply_idx watermark)
//!   │    ├─ "_stores" (directory of tenant ids)
//!   │    └─ "s_<hex>" sub-DBI per tenant
//!   │
//!   └─ per-tenant TenantState (in memory, lazy)
//!        ├─ main_overlay  ← confirmed writes pending durabilize
//!        └─ chain of open top-level Txns (in raft propose order)
//!              each Txn has:
//!              ├─ overlay   (this txn's writes)
//!              └─ open_child  ← LIFO stack: at most one savepoint open
//!                                (the savepoint can itself have one open, etc.)
//!
//! Writes go through Txns, which are opened from a `StoreLease`.
//! `Manifest.acquire(tenant_id)` takes the per-tenant dispatch lock
//! and returns a `StoreLease`; `lease.beginTxn()` opens a new top-
//! level Txn at the tail of that tenant's chain. `Txn.put` /
//! `Txn.delete` land in the txn's own overlay. `Txn.savepoint()`
//! pushes a child Txn whose parent is this one — savepoints stack
//! LIFO (single `open_child` slot). The lease may be released as soon
//! as dispatch is done; the Txn lives in the chain until its own
//! commit / rollback.
//!
//! Reads walk inside-out: savepoint stack → enclosing top-level Txn's
//! chain backward → tenant's main_overlay → LMDB read txn. Cross-
//! tenant in-flight state is never visible (each tenant has its own
//! chain and overlay; lease-based reads and `Snapshot` see only
//! main_overlay plus LMDB).
//!
//! Commit:
//!   - Top-level Txn: must be the chain head (raft-oldest); merges
//!     into tenant.main_overlay; unlinks from chain.
//!   - Savepoint: merges into parent.overlay; pops the parent's
//!     open_child slot.
//!
//! Rollback:
//!   - Top-level Txn: drops self AND every successor in the chain.
//!     Matches raft's tail-rejection semantics; on leadership loss
//!     the integration calls rollback on the oldest pending txn and
//!     everything past it falls.
//!   - Savepoint: drops self and any nested sub-savepoint.
//!
//! Durabilize:
//!   - Snapshots pending createStore/dropStore, swaps each tenant's
//!     main_overlay into a local buffer (open txn overlays are NOT
//!     touched — they live in the chain), opens one LMDB write txn,
//!     applies all of it + the raft watermark in a single commit.
//!   - Open txns are unaffected; their writes will land in main_overlay
//!     when they commit, and be drained at the next durabilize.

const std = @import("std");
const lmdb = @import("lmdb.zig");
const overlay_mod = @import("overlay.zig");
const Overlay = overlay_mod.Overlay;
const OverlayEntry = overlay_mod.OverlayEntry;

pub const InitOptions = struct {
    /// Maximum number of stores (LMDB sub-DBIs). Plus 2 reserved for
    /// `_meta` and `_stores`. Fixed at env open.
    max_stores: u32 = 65534,
    /// LMDB mmap size in bytes. Sparse — only touched pages cost RAM.
    max_map_size: usize = 16 * 1024 * 1024 * 1024,
    no_meta_sync: bool = false,
    no_sync: bool = false,
    /// Per-store cap on the *combined* size, in bytes, of the durable
    /// store's `main_overlay` plus the current writer's `Txn.overlay`.
    /// 0 = unlimited (default; pre-cap behavior).
    ///
    /// Enforced at `Txn.put`: if adding this key+value would push the
    /// (main_overlay.bytes + this_txn.overlay.bytes) sum over the cap,
    /// the put returns `error.OverlayCapExceeded` *without* mutating
    /// state. Host policy then chooses the response — typically
    /// rollback this Txn, run a durabilize to drain main_overlay, and
    /// retry. Backpressure becomes the application's call.
    ///
    /// Conservative: doesn't account for in-flight *sibling* Txns on
    /// the same tenant (the chain), and doesn't account for dedup
    /// against existing entries (overwrites are counted as additions
    /// against the cap). Tighten in a follow-up if needed.
    max_overlay_bytes_per_store: usize = 0,
};

pub const SpecificError = error{
    StoreAlreadyExists,
    StoreNotFound,
    ManifestPoisoned,
    InvalidSnapshotFormat,
    UnsupportedSnapshotVersion,
    /// Tried to commit a top-level Txn that wasn't the chain head.
    NotChainHead,
    /// Tried to commit/rollback a Txn that has an open savepoint
    /// child. Close the inner one first (LIFO).
    SavepointStillOpen,
    /// Put would push this tenant's in-memory overlay over the
    /// configured `max_overlay_bytes_per_store`. Caller should
    /// rollback + durabilize + retry, or drop the request.
    OverlayCapExceeded,
};

const META_DBI_NAME: [:0]const u8 = "_meta";
const STORES_DBI_NAME: [:0]const u8 = "_stores";
const META_RAFT_APPLY_KEY: []const u8 = "raft_apply_idx";

fn storeDbiName(id: u64, buf: *[19]u8) [:0]const u8 {
    return std.fmt.bufPrintZ(buf, "s_{x:0>16}", .{id}) catch unreachable;
}

fn encodeStoreIdKey(id: u64, buf: *[8]u8) []const u8 {
    std.mem.writeInt(u64, buf, id, .big);
    return buf;
}

fn decodeStoreIdKey(bytes: []const u8) u64 {
    std.debug.assert(bytes.len == 8);
    return std.mem.readInt(u64, bytes[0..8], .big);
}

// ─── Metrics ────────────────────────────────────────────────────────

/// Lock-free duration histogram. Bucket boundaries match Prometheus's
/// default `prometheus.DefBuckets` (5 ms .. 10 s, in seconds) translated
/// to nanoseconds. Counts are stored non-cumulatively for cheap
/// observation; `snapshot()` returns cumulative counts so the result
/// drops directly into Prometheus's `_bucket{le="..."}` lines.
pub const Histogram = struct {
    /// Upper bounds of each non-+Inf bucket, in nanoseconds. The
    /// implicit +Inf bucket at index `bucket_bounds_nanos.len` catches
    /// every observation that exceeds the largest bound.
    pub const bucket_bounds_nanos = [_]u64{
        5_000_000, //    5 ms
        10_000_000, //   10 ms
        25_000_000, //   25 ms
        50_000_000, //   50 ms
        100_000_000, //  100 ms
        250_000_000, //  250 ms
        500_000_000, //  500 ms
        1_000_000_000, //  1 s
        2_500_000_000, //  2.5 s
        5_000_000_000, //  5 s
        10_000_000_000, // 10 s
    };
    pub const bucket_count = bucket_bounds_nanos.len + 1; // includes +Inf

    buckets: [bucket_count]std.atomic.Value(u64) = @splat(std.atomic.Value(u64).init(0)),
    sum_nanos: std.atomic.Value(u64) = .init(0),

    pub fn observe(self: *Histogram, nanos: u64) void {
        var bucket_idx: usize = bucket_bounds_nanos.len; // +Inf default
        var i: usize = 0;
        while (i < bucket_bounds_nanos.len) : (i += 1) {
            if (nanos <= bucket_bounds_nanos[i]) {
                bucket_idx = i;
                break;
            }
        }
        _ = self.buckets[bucket_idx].fetchAdd(1, .monotonic);
        _ = self.sum_nanos.fetchAdd(nanos, .monotonic);
    }

    fn snapshot(self: *const Histogram) HistogramSnapshot {
        var snap: HistogramSnapshot = .{
            .buckets = undefined,
            .sum_nanos = self.sum_nanos.load(.acquire),
            .count = 0,
        };
        var cum: u64 = 0;
        var i: usize = 0;
        while (i < bucket_count) : (i += 1) {
            cum += self.buckets[i].load(.acquire);
            snap.buckets[i] = cum;
        }
        snap.count = cum;
        return snap;
    }
};

/// Point-in-time copy of a Histogram. `buckets` is cumulative (matches
/// Prometheus's bucket convention): `buckets[i]` is the number of
/// observations whose value was ≤ `Histogram.bucket_bounds_nanos[i]`,
/// and `buckets[bucket_count - 1]` is the +Inf bucket = total count.
pub const HistogramSnapshot = struct {
    buckets: [Histogram.bucket_count]u64,
    sum_nanos: u64,
    count: u64,
};

/// Per-thread shard for the hottest counters. Each shard occupies a
/// full cache line so adjacent shards never share one — and threads
/// map to shards via TLS-cached `gettid() & (HOT_SHARDS - 1)`, so a
/// given thread always hits the same shard. Snapshot sums across all
/// shards.
///
/// Sharded counters here are the ones called per-API-op
/// (put / delete / get / bytes_put). Less-frequent counters (txn
/// commit/rollback, lifecycle, durabilize, gauges) live unsharded on
/// `Metrics` directly — sharding them would just waste memory.
///
/// Note: `active_leases` and `active_snapshots` aren't stored as
/// gauges; they're derived from `leases_acquired - leases_released`
/// (and similarly for snapshots) at snapshot time. Two monotonic
/// counters shard cleanly; a fetchSub-style gauge does not.
pub const HotShard = extern struct {
    put: std.atomic.Value(u64) align(std.atomic.cache_line) = .init(0),
    bytes_put: std.atomic.Value(u64) = .init(0),
    get: std.atomic.Value(u64) = .init(0),
    delete: std.atomic.Value(u64) = .init(0),
    acquire: std.atomic.Value(u64) = .init(0),
    try_acquire: std.atomic.Value(u64) = .init(0),
    try_acquire_contended: std.atomic.Value(u64) = .init(0),
    leases_acquired: std.atomic.Value(u64) = .init(0),
    leases_released: std.atomic.Value(u64) = .init(0),
    txn_commit: std.atomic.Value(u64) = .init(0),
    txn_rollback: std.atomic.Value(u64) = .init(0),
    savepoint_commit: std.atomic.Value(u64) = .init(0),
    savepoint_rollback: std.atomic.Value(u64) = .init(0),
    _pad: [hot_shard_padding]u8 = @splat(0),

    const counter_bytes = 13 * @sizeOf(u64);
    const total_size = std.mem.alignForward(usize, counter_bytes, std.atomic.cache_line);
    const hot_shard_padding = total_size - counter_bytes;
};

comptime {
    if (@sizeOf(HotShard) != HotShard.total_size) {
        @compileError("HotShard size mismatch");
    }
    if (@sizeOf(HotShard) % std.atomic.cache_line != 0) {
        @compileError("HotShard must be a multiple of cache_line");
    }
}

/// Power-of-two count; thread→shard mapping uses `tid & (HOT_SHARDS - 1)`.
/// 64 is generous for any thread pool size we'd encounter and costs
/// ~4 KiB per Manifest, which is negligible.
pub const HOT_SHARDS: usize = 64;

// Per-thread cached shard index. Computed once per thread on first
// metric increment; subsequent ops are a TLS load + array offset.
// Shared across all Manifests in the same process (different Manifests
// have their own shard arrays, but the *index* mapping per thread is
// the same — that's fine).
threadlocal var tls_hot_shard_idx: usize = 0;
threadlocal var tls_hot_shard_mapped: bool = false;

inline fn currentHotShardIdx() usize {
    if (!tls_hot_shard_mapped) {
        const tid: u64 = @intCast(std.Thread.getCurrentId());
        tls_hot_shard_idx = @intCast(tid & (HOT_SHARDS - 1));
        tls_hot_shard_mapped = true;
    }
    return tls_hot_shard_idx;
}

/// Atomic counters and gauges for production observability. Each
/// counter is monotonically increasing; gauges go up and down. Reading
/// individual fields is lock-free; reading the full snapshot via
/// `Manifest.metricsSnapshot()` is point-in-time per-field but not
/// linearized across fields (which is fine for monitoring).
///
/// Counter naming follows Prometheus convention: `*_total` for monotonic
/// counters, no suffix for gauges.
pub const Metrics = struct {
    // Hot (sharded across cache lines): per-API-op and per-lease-cycle
    // counters. See HotShard for the full set.
    hot_shards: [HOT_SHARDS]HotShard align(std.atomic.cache_line) = @splat(.{}),

    // Cold (single atomic): lifecycle, durability, snapshot opens,
    // poison. These fire at "every checkpoint" frequency or rarer.
    create_store_total: std.atomic.Value(u64) = .init(0),
    drop_store_total: std.atomic.Value(u64) = .init(0),
    durabilize_total: std.atomic.Value(u64) = .init(0),
    durabilize_failed_total: std.atomic.Value(u64) = .init(0),
    snapshot_open_total: std.atomic.Value(u64) = .init(0),
    snapshot_close_total: std.atomic.Value(u64) = .init(0),
    poison_total: std.atomic.Value(u64) = .init(0),

    // Duration distributions. durabilize is the main one (one fsync
    // boundary per checkpoint); snapshot_open captures every
    // openSnapshot, useful for tracking state-transfer overhead.
    durabilize_duration: Histogram = .{},
    snapshot_open_duration: Histogram = .{},

    // ── hot-path helpers (sharded) ─────────────────────────────────

    inline fn currentShard(self: *Metrics) *HotShard {
        return &self.hot_shards[currentHotShardIdx()];
    }

    pub inline fn recordPut(self: *Metrics, bytes: u64) void {
        const s = self.currentShard();
        _ = s.put.fetchAdd(1, .monotonic);
        _ = s.bytes_put.fetchAdd(bytes, .monotonic);
    }

    pub inline fn recordDelete(self: *Metrics) void {
        _ = self.currentShard().delete.fetchAdd(1, .monotonic);
    }

    pub inline fn recordGet(self: *Metrics) void {
        _ = self.currentShard().get.fetchAdd(1, .monotonic);
    }

    pub inline fn recordAcquire(self: *Metrics) void {
        const s = self.currentShard();
        _ = s.acquire.fetchAdd(1, .monotonic);
        _ = s.leases_acquired.fetchAdd(1, .monotonic);
    }

    pub inline fn recordTryAcquire(self: *Metrics) void {
        const s = self.currentShard();
        _ = s.try_acquire.fetchAdd(1, .monotonic);
        _ = s.leases_acquired.fetchAdd(1, .monotonic);
    }

    pub inline fn recordTryAcquireContended(self: *Metrics) void {
        _ = self.currentShard().try_acquire_contended.fetchAdd(1, .monotonic);
    }

    pub inline fn recordRelease(self: *Metrics) void {
        _ = self.currentShard().leases_released.fetchAdd(1, .monotonic);
    }

    pub inline fn recordTxnCommit(self: *Metrics) void {
        _ = self.currentShard().txn_commit.fetchAdd(1, .monotonic);
    }

    pub inline fn recordTxnRollback(self: *Metrics) void {
        _ = self.currentShard().txn_rollback.fetchAdd(1, .monotonic);
    }

    pub inline fn recordSavepointCommit(self: *Metrics) void {
        _ = self.currentShard().savepoint_commit.fetchAdd(1, .monotonic);
    }

    pub inline fn recordSavepointRollback(self: *Metrics) void {
        _ = self.currentShard().savepoint_rollback.fetchAdd(1, .monotonic);
    }

    fn snapshot(self: *const Metrics) MetricsSnapshot {
        var put_total: u64 = 0;
        var bytes_put_total: u64 = 0;
        var get_total: u64 = 0;
        var delete_total: u64 = 0;
        var acquire_total: u64 = 0;
        var try_acquire_total: u64 = 0;
        var try_acquire_contended_total: u64 = 0;
        var leases_acquired: u64 = 0;
        var leases_released: u64 = 0;
        var txn_commit_total: u64 = 0;
        var txn_rollback_total: u64 = 0;
        var savepoint_commit_total: u64 = 0;
        var savepoint_rollback_total: u64 = 0;
        for (&self.hot_shards) |*s| {
            put_total += s.put.load(.monotonic);
            bytes_put_total += s.bytes_put.load(.monotonic);
            get_total += s.get.load(.monotonic);
            delete_total += s.delete.load(.monotonic);
            acquire_total += s.acquire.load(.monotonic);
            try_acquire_total += s.try_acquire.load(.monotonic);
            try_acquire_contended_total += s.try_acquire_contended.load(.monotonic);
            leases_acquired += s.leases_acquired.load(.monotonic);
            leases_released += s.leases_released.load(.monotonic);
            txn_commit_total += s.txn_commit.load(.monotonic);
            txn_rollback_total += s.txn_rollback.load(.monotonic);
            savepoint_commit_total += s.savepoint_commit.load(.monotonic);
            savepoint_rollback_total += s.savepoint_rollback.load(.monotonic);
        }
        const open = self.snapshot_open_total.load(.monotonic);
        const closed = self.snapshot_close_total.load(.monotonic);
        return .{
            .create_store_total = self.create_store_total.load(.monotonic),
            .drop_store_total = self.drop_store_total.load(.monotonic),
            .acquire_total = acquire_total,
            .try_acquire_total = try_acquire_total,
            .try_acquire_contended_total = try_acquire_contended_total,
            .txn_commit_total = txn_commit_total,
            .txn_rollback_total = txn_rollback_total,
            .savepoint_commit_total = savepoint_commit_total,
            .savepoint_rollback_total = savepoint_rollback_total,
            .put_total = put_total,
            .delete_total = delete_total,
            .get_total = get_total,
            .bytes_put_total = bytes_put_total,
            .durabilize_total = self.durabilize_total.load(.monotonic),
            .durabilize_failed_total = self.durabilize_failed_total.load(.monotonic),
            .snapshot_open_total = open,
            .poison_total = self.poison_total.load(.monotonic),
            // Gauges derived from monotonic counters: subtract released
            // from acquired. Atomic per-shard reads aren't linearized,
            // so under heavy contention this can momentarily go
            // negative for a few ns. Acceptable for monitoring.
            .active_leases = @as(i64, @intCast(leases_acquired)) -
                @as(i64, @intCast(leases_released)),
            .active_snapshots = @as(i64, @intCast(open)) -
                @as(i64, @intCast(closed)),
            .durabilize_duration = self.durabilize_duration.snapshot(),
            .snapshot_open_duration = self.snapshot_open_duration.snapshot(),
        };
    }
};

/// A point-in-time copy of `Metrics`. Plain values, safe to print, log,
/// or ship to a metrics backend.
pub const MetricsSnapshot = struct {
    create_store_total: u64,
    drop_store_total: u64,
    acquire_total: u64,
    try_acquire_total: u64,
    try_acquire_contended_total: u64,
    txn_commit_total: u64,
    txn_rollback_total: u64,
    savepoint_commit_total: u64,
    savepoint_rollback_total: u64,
    put_total: u64,
    delete_total: u64,
    get_total: u64,
    bytes_put_total: u64,
    durabilize_total: u64,
    durabilize_failed_total: u64,
    snapshot_open_total: u64,
    poison_total: u64,
    active_leases: i64,
    active_snapshots: i64,
    durabilize_duration: HistogramSnapshot,
    snapshot_open_duration: HistogramSnapshot,
};

// ─── Manifest ───────────────────────────────────────────────────────

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    env: lmdb.Env,
    meta_dbi: lmdb.Dbi,
    stores_dbi: lmdb.Dbi,

    /// store_id → DBI handle for durably-registered tenants.
    dbis: std.AutoHashMapUnmanaged(u64, lmdb.Dbi) = .empty,
    pending_creates: std.AutoHashMapUnmanaged(u64, void) = .empty,
    pending_drops: std.AutoHashMapUnmanaged(u64, void) = .empty,
    dbis_lock: std.Thread.Mutex = .{},

    /// store_id → per-tenant in-memory state (overlay + chain of open
    /// Txns). Lazily created on first acquire / lease.beginTxn / write.
    tenants: std.AutoHashMapUnmanaged(u64, *TenantState) = .empty,
    tenants_lock: std.Thread.Mutex = .{},

    poisoned: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Single-caller serializer for durabilize / openSnapshot.
    durabilize_lock: std.Thread.Mutex = .{},

    /// Atomic counters/gauges. Read via `metricsSnapshot()` for a
    /// consistent point-in-time view; the underlying fields are atomic
    /// but reading them individually is not coordinated across fields.
    metrics: Metrics = .{},

    /// See `InitOptions.max_overlay_bytes_per_store`. 0 = unlimited.
    max_overlay_bytes_per_store: usize = 0,

    pub fn init(
        self: *Manifest,
        allocator: std.mem.Allocator,
        path: [:0]const u8,
        options: InitOptions,
    ) !void {
        self.* = .{
            .allocator = allocator,
            .env = undefined,
            .meta_dbi = undefined,
            .stores_dbi = undefined,
            .max_overlay_bytes_per_store = options.max_overlay_bytes_per_store,
        };
        self.env = try lmdb.Env.open(path, .{
            .max_dbs = options.max_stores + 2,
            .max_map_size = options.max_map_size,
            .no_meta_sync = options.no_meta_sync,
            .no_sync = options.no_sync,
        });
        errdefer self.env.close();

        // Bootstrap _meta and _stores.
        {
            var txn = try lmdb.Txn.beginWrite(&self.env);
            errdefer txn.abort();
            self.meta_dbi = try txn.openDbi(META_DBI_NAME, true);
            self.stores_dbi = try txn.openDbi(STORES_DBI_NAME, true);
            try txn.commit();
        }
        errdefer self.deinitMaps();

        // Populate `dbis` from `_stores`.
        {
            var txn = try lmdb.Txn.beginWrite(&self.env);
            errdefer txn.abort();
            self.stores_dbi = try txn.openDbi(STORES_DBI_NAME, false);
            self.meta_dbi = try txn.openDbi(META_DBI_NAME, false);

            var cur = try txn.openCursor(self.stores_dbi);
            defer cur.close();
            var pair = try cur.first();
            while (pair) |p| : (pair = try cur.next()) {
                const id = decodeStoreIdKey(p.key);
                var name_buf: [19]u8 = undefined;
                const name = storeDbiName(id, &name_buf);
                const dbi = try txn.openDbi(name, false);
                self.dbis.put(self.allocator, id, dbi) catch
                    @panic("OOM populating dbis map at init");
            }
            try txn.commit();
        }
    }

    fn deinitMaps(self: *Manifest) void {
        self.dbis.deinit(self.allocator);
        self.pending_creates.deinit(self.allocator);
        self.pending_drops.deinit(self.allocator);
        // Drop every tenant's chain (Txns) and then drop the map's
        // reference to each TenantState. If a caller leaked a lease
        // past deinit, the TenantState lingers (refcount > 1) — that's
        // a caller bug; we prefer the leak to a UAF.
        var it = self.tenants.iterator();
        while (it.next()) |entry| {
            const ts = entry.value_ptr.*;
            var cur = ts.chain_head;
            while (cur) |c| {
                const next = c.chain_next;
                c.freeSubtreeLocked();
                cur = next;
            }
            ts.chain_head = null;
            ts.chain_tail = null;
            ts.releaseRef(self.allocator);
        }
        self.tenants.deinit(self.allocator);
    }

    pub fn deinit(self: *Manifest) void {
        self.deinitMaps();
        self.env.close();
        self.* = undefined;
    }

    // ── metrics ─────────────────────────────────────────────────────

    pub fn metricsSnapshot(self: *const Manifest) MetricsSnapshot {
        return self.metrics.snapshot();
    }

    // ── poison ──────────────────────────────────────────────────────

    pub fn isPoisoned(self: *const Manifest) bool {
        return self.poisoned.load(.monotonic);
    }

    fn poison(self: *Manifest) void {
        self.poisoned.store(true, .monotonic);
        _ = self.metrics.poison_total.fetchAdd(1, .monotonic);
    }

    pub fn _testPoison(self: *Manifest) void {
        self.poison();
    }

    inline fn checkAlive(self: *const Manifest) !void {
        if (self.isPoisoned()) return error.ManifestPoisoned;
    }

    // ── stores (durable lifecycle, buffered) ────────────────────────

    pub fn createStore(self: *Manifest, id: u64) !void {
        try self.checkAlive();
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (self.hasStoreLocked(id)) return error.StoreAlreadyExists;
        self.pending_creates.put(self.allocator, id, {}) catch
            @panic("OOM in createStore.pending_creates.put");
        _ = self.metrics.create_store_total.fetchAdd(1, .monotonic);
    }

    pub fn dropStore(self: *Manifest, id: u64) !bool {
        try self.checkAlive();

        // Remove the tenants-map entry up front so concurrent acquires
        // start fresh TenantStates (and so we can free this one once
        // every outstanding lease releases). We hold a transient
        // reference to the removed ts via `ts_for_drop`; the defer at
        // the bottom drops that reference, and if no lease holder is
        // outstanding, the TenantState is freed there.
        const ts_for_drop: ?*TenantState = blk: {
            self.tenants_lock.lock();
            defer self.tenants_lock.unlock();
            const kv = self.tenants.fetchRemove(id) orelse break :blk null;
            break :blk kv.value;
        };
        defer if (ts_for_drop) |ts| ts.releaseRef(self.allocator);

        // Wait for any active lease holder. After this lock, no other
        // dispatch path can be operating on this ts.
        //
        // Callers must not hold a lease on `id` from the same thread
        // (would self-deadlock). Per-thread caveat documented in the
        // README.
        if (ts_for_drop) |ts| ts.dispatch_lock.lock();
        defer if (ts_for_drop) |ts| ts.dispatch_lock.unlock();

        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (!self.hasStoreLocked(id)) return false;

        // Cancel a pending_create that hasn't been durabilized yet.
        if (self.pending_creates.remove(id)) {
            // If the store is also durable (a drop-recreate-drop cycle
            // on the same id), the underlying durable DBI also needs
            // emptying at next durabilize.
            if (self.dbis.contains(id)) {
                self.pending_drops.put(self.allocator, id, {}) catch
                    @panic("OOM in dropStore.pending_drops.put");
            }
        } else {
            // Durable store: queue the drop.
            self.pending_drops.put(self.allocator, id, {}) catch
                @panic("OOM in dropStore.pending_drops.put");
        }

        // Tear down the chain + clear main_overlay. Callers holding
        // pointers to these Txns now hold dangling pointers — the
        // explicit-drop-from-under-you case documented in the README.
        if (ts_for_drop) |ts| {
            ts.lock.lock();
            defer ts.lock.unlock();
            var cur = ts.chain_head;
            while (cur) |c| {
                const next = c.chain_next;
                c.freeSubtreeLocked();
                cur = next;
            }
            ts.chain_head = null;
            ts.chain_tail = null;
            ts.main_overlay.clear();
        }
        _ = self.metrics.drop_store_total.fetchAdd(1, .monotonic);
        return true;
    }

    pub fn hasStore(self: *Manifest, id: u64) !bool {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        return self.hasStoreLocked(id);
    }

    fn hasStoreLocked(self: *const Manifest, id: u64) bool {
        if (self.pending_creates.contains(id)) return true;
        if (self.pending_drops.contains(id)) return false;
        return self.dbis.contains(id);
    }

    pub fn listStores(self: *Manifest, allocator: std.mem.Allocator) ![]u64 {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        var it = self.dbis.keyIterator();
        while (it.next()) |id_ptr| {
            if (self.pending_drops.contains(id_ptr.*)) continue;
            try list.append(allocator, id_ptr.*);
        }
        var pc_it = self.pending_creates.keyIterator();
        while (pc_it.next()) |id_ptr| try list.append(allocator, id_ptr.*);
        std.mem.sort(u64, list.items, {}, std.sort.asc(u64));
        return try list.toOwnedSlice(allocator);
    }

    fn lookupDbi(self: *Manifest, id: u64) ?lmdb.Dbi {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (self.pending_drops.contains(id)) return null;
        return self.dbis.get(id);
    }

    // ── tenant state ────────────────────────────────────────────────

    fn getOrCreateTenantState(self: *Manifest, id: u64) !*TenantState {
        self.tenants_lock.lock();
        defer self.tenants_lock.unlock();
        if (self.tenants.get(id)) |ts| {
            ts.retain();
            return ts;
        }
        // Allocate the TenantState before inserting the map entry so a
        // mid-create OOM can't leave the map with a slot whose
        // value_ptr is undefined. (And we crash hard on OOM here
        // anyway — see CLAUDE.md: kvexp sits under raft, so OOM in
        // internal bookkeeping means the process is gone, full stop.)
        const ts = self.allocator.create(TenantState) catch
            @panic("OOM allocating TenantState");
        ts.* = .{ .main_overlay = Overlay.init(self.allocator) };
        // refcount initialized to 1 (the map's reference); retain once
        // more for the caller.
        ts.retain();
        self.tenants.put(self.allocator, id, ts) catch
            @panic("OOM inserting TenantState into tenants map");
        return ts;
    }

    // ── acquire / tryAcquire ────────────────────────────────────────

    /// Take exclusive dispatch ownership of `tenant_id`. Blocks until
    /// any current lease holder releases. The returned `StoreLease` is
    /// the only handle through which `beginTxn`, `get`, or `scanPrefix`
    /// can be issued for this tenant.
    ///
    /// Returns `error.StoreNotFound` if the tenant doesn't exist.
    ///
    /// The lease holder must call `release` to unlock. Other operations
    /// that touch the tenant (durabilize, openSnapshot, lockless cross-
    /// tenant reads) run concurrently — the lease only serializes
    /// dispatch.
    pub fn acquire(self: *Manifest, tenant_id: u64) !StoreLease {
        try self.checkAlive();
        if (!try self.hasStore(tenant_id)) return error.StoreNotFound;
        const ts = try self.getOrCreateTenantState(tenant_id);
        // ts is retained for us; if we exit without producing a lease,
        // we must drop that reference.
        errdefer ts.releaseRef(self.allocator);
        ts.dispatch_lock.lock();
        self.metrics.recordAcquire();
        return .{
            .manifest = self,
            .tenant_id = tenant_id,
            .tenant_state = ts,
        };
    }

    /// Non-blocking variant of `acquire`. Returns null if another
    /// holder already owns the lease.
    pub fn tryAcquire(self: *Manifest, tenant_id: u64) !?StoreLease {
        try self.checkAlive();
        if (!try self.hasStore(tenant_id)) return error.StoreNotFound;
        const ts = try self.getOrCreateTenantState(tenant_id);
        errdefer ts.releaseRef(self.allocator);
        if (!ts.dispatch_lock.tryLock()) {
            ts.releaseRef(self.allocator);
            self.metrics.recordTryAcquireContended();
            return null;
        }
        self.metrics.recordTryAcquire();
        return StoreLease{
            .manifest = self,
            .tenant_id = tenant_id,
            .tenant_state = ts,
        };
    }

    // ── durabilize ──────────────────────────────────────────────────

    pub fn durabilize(self: *Manifest, raft_idx: u64) !void {
        try self.checkAlive();
        self.durabilize_lock.lock();
        defer self.durabilize_lock.unlock();
        errdefer {
            _ = self.metrics.durabilize_failed_total.fetchAdd(1, .monotonic);
            self.poison();
        }
        const start = std.time.Instant.now() catch null;

        // 1. Snapshot pending creates/drops.
        var creates_to_apply: std.ArrayListUnmanaged(u64) = .empty;
        defer creates_to_apply.deinit(self.allocator);
        var drops_to_apply: std.ArrayListUnmanaged(struct { id: u64, dbi: lmdb.Dbi }) = .empty;
        defer drops_to_apply.deinit(self.allocator);
        {
            self.dbis_lock.lock();
            defer self.dbis_lock.unlock();
            var pc_it = self.pending_creates.keyIterator();
            while (pc_it.next()) |id_ptr| creates_to_apply.append(self.allocator, id_ptr.*) catch
                @panic("OOM building creates_to_apply in durabilize");
            var pd_it = self.pending_drops.keyIterator();
            while (pd_it.next()) |id_ptr| {
                const dbi = self.dbis.get(id_ptr.*) orelse continue;
                drops_to_apply.append(self.allocator, .{ .id = id_ptr.*, .dbi = dbi }) catch
                    @panic("OOM building drops_to_apply in durabilize");
            }
        }

        // 2. Enumerate every tenant.
        const TenantPair = struct { id: u64, ts: *TenantState };
        var tenant_pairs: std.ArrayListUnmanaged(TenantPair) = .empty;
        defer tenant_pairs.deinit(self.allocator);
        {
            self.tenants_lock.lock();
            defer self.tenants_lock.unlock();
            var it = self.tenants.iterator();
            while (it.next()) |entry| {
                tenant_pairs.append(self.allocator, .{
                    .id = entry.key_ptr.*,
                    .ts = entry.value_ptr.*,
                }) catch @panic("OOM building tenant_pairs in durabilize");
            }
        }

        // 3. Swap each tenant's main_overlay into a local buffer.
        //    Workers committing top-level Txns during durabilize land
        //    in a fresh main_overlay that we leave behind; their data
        //    will be in the next durabilize.
        var swapped_list: std.ArrayListUnmanaged(SwappedEntry) = .empty;
        defer {
            for (swapped_list.items) |*s| {
                var it = s.entries.iterator();
                while (it.next()) |e| {
                    self.allocator.free(e.key_ptr.*);
                    switch (e.value_ptr.*) {
                        .value => |v| self.allocator.free(v),
                        .tombstone => {},
                    }
                }
                s.entries.deinit(self.allocator);
            }
            swapped_list.deinit(self.allocator);
        }
        for (tenant_pairs.items) |t| {
            t.ts.lock.lock();
            const empty = t.ts.main_overlay.entries.count() == 0;
            if (empty) {
                t.ts.lock.unlock();
                continue;
            }
            const taken = t.ts.main_overlay.entries;
            t.ts.main_overlay.entries = .empty;
            t.ts.main_overlay.bytes = 0; // bytes lived in `entries`, now in `taken`
            t.ts.lock.unlock();
            swapped_list.append(self.allocator, .{
                .id = t.id,
                .entries = taken,
            }) catch @panic("OOM building swapped_list in durabilize");
        }

        // 4. Apply everything in one LMDB write txn.
        var txn = try lmdb.Txn.beginWrite(&self.env);
        errdefer txn.abort();

        for (drops_to_apply.items) |d| {
            try txn.dropDbi(d.dbi, false);
            var id_key_buf: [8]u8 = undefined;
            _ = try txn.del(self.stores_dbi, encodeStoreIdKey(d.id, &id_key_buf));
        }

        var new_dbis: std.ArrayListUnmanaged(NewDbi) = .empty;
        defer new_dbis.deinit(self.allocator);
        for (creates_to_apply.items) |id| {
            var name_buf: [19]u8 = undefined;
            const name = storeDbiName(id, &name_buf);
            const dbi = try txn.openDbi(name, true);
            var id_key_buf: [8]u8 = undefined;
            try txn.put(self.stores_dbi, encodeStoreIdKey(id, &id_key_buf), &.{});
            new_dbis.append(self.allocator, .{ .id = id, .dbi = dbi }) catch
                @panic("OOM building new_dbis in durabilize");
        }

        for (swapped_list.items) |s| {
            const dbi = self.lookupDbiForApply(s.id, new_dbis.items) orelse continue;
            var it = s.entries.iterator();
            while (it.next()) |e| {
                switch (e.value_ptr.*) {
                    .value => |v| try txn.put(dbi, e.key_ptr.*, v),
                    .tombstone => _ = try txn.del(dbi, e.key_ptr.*),
                }
            }
        }

        if (raft_idx != 0) {
            var idx_buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &idx_buf, raft_idx, .little);
            try txn.put(self.meta_dbi, META_RAFT_APPLY_KEY, &idx_buf);
        }

        try txn.commit();

        // 5. Bookkeeping.
        {
            self.dbis_lock.lock();
            defer self.dbis_lock.unlock();
            for (drops_to_apply.items) |d| {
                _ = self.dbis.remove(d.id);
                _ = self.pending_drops.remove(d.id);
            }
            for (new_dbis.items) |n| {
                self.dbis.put(self.allocator, n.id, n.dbi) catch
                    @panic("OOM in durabilize bookkeeping dbis.put");
                _ = self.pending_creates.remove(n.id);
            }
        }
        if (start) |s| {
            if (std.time.Instant.now()) |now| {
                self.metrics.durabilize_duration.observe(now.since(s));
            } else |_| {}
        }
        _ = self.metrics.durabilize_total.fetchAdd(1, .monotonic);
    }

    /// Convenience: durabilize without disturbing the raft watermark.
    /// Named `flush` (rather than `commit`) to avoid collision with
    /// `Txn.commit`, which means something completely different.
    pub fn flush(self: *Manifest) !void {
        return self.durabilize(0);
    }

    /// Read the durable raft watermark from LMDB. Returns 0 if no
    /// durabilize has ever written one.
    pub fn durableRaftIdx(self: *Manifest) !u64 {
        var txn = try lmdb.Txn.beginRead(&self.env);
        defer txn.abort();
        const bytes = (try txn.get(self.meta_dbi, META_RAFT_APPLY_KEY)) orelse return 0;
        if (bytes.len != 8) return 0;
        return std.mem.readInt(u64, bytes[0..8], .little);
    }

    const NewDbi = struct { id: u64, dbi: lmdb.Dbi };
    const SwappedEntry = struct {
        id: u64,
        entries: std.StringHashMapUnmanaged(OverlayEntry),
    };

    fn lookupDbiForApply(self: *Manifest, id: u64, new_dbis: []const NewDbi) ?lmdb.Dbi {
        for (new_dbis) |n| if (n.id == id) return n.dbi;
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        return self.dbis.get(id);
    }

    // ── snapshot ────────────────────────────────────────────────────

    pub fn openSnapshot(self: *Manifest) !Snapshot {
        // Serialize against durabilize so we get a coherent moment.
        self.durabilize_lock.lock();
        defer self.durabilize_lock.unlock();
        const start = std.time.Instant.now() catch null;

        var read_txn = try lmdb.Txn.beginRead(&self.env);
        errdefer read_txn.abort();

        // Capture every tenant's main_overlay (open txn state is NOT
        // captured — snapshots see committed-but-not-durabilized
        // state, never speculation).
        var ov_capture: std.AutoHashMapUnmanaged(u64, []SnapshotEntry) = .empty;
        errdefer freeSnapshotOverlay(self.allocator, &ov_capture);

        const TenantPair = struct { id: u64, ts: *TenantState };
        var pairs: std.ArrayListUnmanaged(TenantPair) = .empty;
        defer pairs.deinit(self.allocator);
        {
            self.tenants_lock.lock();
            defer self.tenants_lock.unlock();
            var it = self.tenants.iterator();
            while (it.next()) |entry| {
                pairs.append(self.allocator, .{
                    .id = entry.key_ptr.*,
                    .ts = entry.value_ptr.*,
                }) catch @panic("OOM building tenant pairs in openSnapshot");
            }
        }
        for (pairs.items) |p| {
            var entries: std.ArrayListUnmanaged(SnapshotEntry) = .empty;
            errdefer {
                for (entries.items) |e| {
                    self.allocator.free(e.key);
                    if (e.value) |v| self.allocator.free(v);
                }
                entries.deinit(self.allocator);
            }
            {
                p.ts.lock.lock();
                defer p.ts.lock.unlock();
                if (p.ts.main_overlay.entries.count() == 0) continue;
                var oe_it = p.ts.main_overlay.entries.iterator();
                while (oe_it.next()) |oe| {
                    const k = self.allocator.dupe(u8, oe.key_ptr.*) catch
                        @panic("OOM duping overlay key in openSnapshot");
                    const v: ?[]u8 = switch (oe.value_ptr.*) {
                        .value => |val| self.allocator.dupe(u8, val) catch
                            @panic("OOM duping overlay value in openSnapshot"),
                        .tombstone => null,
                    };
                    entries.append(self.allocator, .{ .key = k, .value = v }) catch
                        @panic("OOM appending overlay entry in openSnapshot");
                }
            }
            std.mem.sort(SnapshotEntry, entries.items, {}, SnapshotEntry.lessThan);
            const slice = entries.toOwnedSlice(self.allocator) catch
                @panic("OOM materializing overlay slice in openSnapshot");
            ov_capture.put(self.allocator, p.id, slice) catch
                @panic("OOM storing overlay slice in openSnapshot");
        }

        if (start) |s| {
            if (std.time.Instant.now()) |now| {
                self.metrics.snapshot_open_duration.observe(now.since(s));
            } else |_| {}
        }
        _ = self.metrics.snapshot_open_total.fetchAdd(1, .monotonic);
        return .{
            .manifest = self,
            .read_txn = read_txn,
            .overlays = ov_capture,
        };
    }

};

// ─── TenantState ────────────────────────────────────────────────────

const TenantState = struct {
    /// Internal serialization for main_overlay + chain mutations
    /// (taken briefly inside individual API calls).
    lock: std.Thread.Mutex = .{},
    /// Application-level dispatch lock: held by a StoreLease holder
    /// across many API calls. Top of the lock stack; never taken from
    /// inside kvexp's internal lock paths.
    dispatch_lock: std.Thread.Mutex = .{},
    main_overlay: Overlay,
    chain_head: ?*Txn = null,
    chain_tail: ?*Txn = null,

    /// Refcount = (1 if still in Manifest.tenants) + (1 per outstanding
    /// StoreLease). dropStore removes the map entry and drops the
    /// map's reference; the last lease holder's release() then frees.
    /// This is what lets dropStore reclaim memory without UAFing
    /// concurrent acquires that are between getOrCreateTenantState
    /// and dispatch_lock.lock().
    refcount: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),

    fn retain(self: *TenantState) void {
        // Relaxed: subsequent ops on the TenantState are synchronized
        // by tenant_state.lock or dispatch_lock; the refcount itself
        // doesn't publish data.
        _ = self.refcount.fetchAdd(1, .monotonic);
    }

    /// Drop a reference. If we were the last, free the TenantState.
    /// Caller must not be holding any of this TenantState's locks.
    fn releaseRef(self: *TenantState, allocator: std.mem.Allocator) void {
        // acq_rel: the acquire half ensures any prior releaseRef's
        // writes (free'ing values on rolled-back Txns, clearing the
        // overlay during dropStore, etc.) are visible to us before
        // we tear the TenantState down at refcount == 1.
        if (self.refcount.fetchSub(1, .acq_rel) == 1) {
            self.main_overlay.deinit();
            allocator.destroy(self);
        }
    }
};

// ─── Txn ────────────────────────────────────────────────────────────

pub const Txn = struct {
    manifest: *Manifest,
    tenant_id: u64,
    tenant_state: *TenantState,
    /// null for top-level (chain entry); non-null for savepoint.
    parent: ?*Txn,
    /// Linked-list pointers for the per-tenant chain. Used only on
    /// top-level Txns (parent == null).
    chain_prev: ?*Txn,
    chain_next: ?*Txn,
    /// This Txn's own writes.
    overlay: Overlay,
    /// The currently-open savepoint child, if any. LIFO: at most one.
    open_child: ?*Txn,

    // ── writes ──────────────────────────────────────────────────────

    pub fn put(self: *Txn, key: []const u8, value: []const u8) !void {
        try self.manifest.checkAlive();
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        // Per-store memory cap. Conservative — counts main_overlay +
        // this Txn's overlay + the new bytes, no dedup credit. The
        // check has to happen *before* putLocked mutates the overlay
        // so the caller's state is untouched on rejection.
        const cap = self.manifest.max_overlay_bytes_per_store;
        if (cap != 0) {
            const projected = self.tenant_state.main_overlay.bytes +
                self.overlay.bytes + key.len + value.len;
            if (projected > cap) return error.OverlayCapExceeded;
        }
        self.overlay.putLocked(key, value);
        self.manifest.metrics.recordPut(key.len + value.len);
    }

    pub fn delete(self: *Txn, key: []const u8) !bool {
        try self.manifest.checkAlive();
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        const existed = try self.keyExistsLocked(key);
        self.overlay.tombstoneLocked(key);
        self.manifest.metrics.recordDelete();
        return existed;
    }

    // ── reads ───────────────────────────────────────────────────────

    pub fn get(self: *Txn, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        self.manifest.metrics.recordGet();
        return self.getLocked(allocator, key);
    }

    fn getLocked(self: *Txn, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        // Walk savepoint stack: self → parent → parent.parent → ...
        // After top-level, walk chain backward.
        var cur: ?*Txn = self;
        while (cur) |t| {
            if (t.overlay.entries.get(key)) |entry| {
                switch (entry) {
                    .value => |v| return try allocator.dupe(u8, v),
                    .tombstone => return null,
                }
            }
            cur = if (t.parent) |p| p else t.chain_prev;
        }
        // main_overlay.
        if (self.tenant_state.main_overlay.entries.get(key)) |entry| {
            switch (entry) {
                .value => |v| return try allocator.dupe(u8, v),
                .tombstone => return null,
            }
        }
        // LMDB.
        const dbi = self.manifest.lookupDbi(self.tenant_id) orelse return null;
        var read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
        defer read_txn.abort();
        if (try read_txn.get(dbi, key)) |bytes| {
            return try allocator.dupe(u8, bytes);
        }
        return null;
    }

    /// Like getLocked but only returns whether the key resolves to a
    /// value (true) or null/tombstone (false). Used by delete() to
    /// compute the return bool without allocating.
    fn keyExistsLocked(self: *Txn, key: []const u8) !bool {
        var cur: ?*Txn = self;
        while (cur) |t| {
            if (t.overlay.entries.get(key)) |entry| {
                return switch (entry) {
                    .value => true,
                    .tombstone => false,
                };
            }
            cur = if (t.parent) |p| p else t.chain_prev;
        }
        if (self.tenant_state.main_overlay.entries.get(key)) |entry| {
            return switch (entry) {
                .value => true,
                .tombstone => false,
            };
        }
        const dbi = self.manifest.lookupDbi(self.tenant_id) orelse return false;
        var read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
        defer read_txn.abort();
        return (try read_txn.get(dbi, key)) != null;
    }

    pub fn scanPrefix(self: *Txn, prefix: []const u8) !TxnPrefixCursor {
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();

        // Flatten the layered view into a sorted slice, with closer-
        // to-self layers winning collisions. Tombstones are preserved
        // as null values — needed during the LMDB merge to shadow
        // entries that have been deleted in any overlay layer.
        var dedup: std.StringHashMapUnmanaged(?[]u8) = .empty;
        defer dedup.deinit(self.manifest.allocator);
        // Track key ownership separately so we can transfer to the
        // cursor's final slice without re-cloning.
        var ordered_keys: std.ArrayListUnmanaged([]u8) = .empty;
        defer ordered_keys.deinit(self.manifest.allocator);

        var cur: ?*Txn = self;
        while (cur) |t| {
            absorbOverlayLocked(self.manifest.allocator, &t.overlay, prefix, &dedup, &ordered_keys);
            cur = if (t.parent) |p| p else t.chain_prev;
        }
        absorbOverlayLocked(self.manifest.allocator, &self.tenant_state.main_overlay, prefix, &dedup, &ordered_keys);

        // Materialize sorted slice.
        var entries: std.ArrayListUnmanaged(TxnPrefixCursor.Entry) = .empty;
        errdefer {
            for (entries.items) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            entries.deinit(self.manifest.allocator);
        }
        entries.ensureTotalCapacity(self.manifest.allocator, ordered_keys.items.len) catch
            @panic("OOM reserving cursor entries");
        for (ordered_keys.items) |k| {
            const v = dedup.get(k).?;
            entries.appendAssumeCapacity(.{ .key = k, .value = v });
        }
        std.mem.sort(TxnPrefixCursor.Entry, entries.items, {}, TxnPrefixCursor.Entry.lessThan);
        const overlay_slice = entries.toOwnedSlice(self.manifest.allocator) catch
            @panic("OOM materializing cursor overlay slice");
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        // LMDB cursor (if the tenant has a durable DBI).
        var read_txn: lmdb.Txn = .{ .ptr = null };
        var lmdb_cur: lmdb.Cursor = .{ .ptr = null };
        var has_lmdb = false;
        if (self.manifest.lookupDbi(self.tenant_id)) |dbi| {
            read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
            errdefer read_txn.abort();
            lmdb_cur = try read_txn.openCursor(dbi);
            has_lmdb = true;
        }
        errdefer if (has_lmdb) {
            lmdb_cur.close();
            read_txn.abort();
        };

        var cursor: TxnPrefixCursor = .{
            .allocator = self.manifest.allocator,
            .overlay = overlay_slice,
            .ov_idx = 0,
            .read_txn = read_txn,
            .lmdb_cur = lmdb_cur,
            .has_lmdb = has_lmdb,
            .prefix = self.manifest.allocator.dupe(u8, prefix) catch
                @panic("OOM duping cursor prefix"),
            .have_tree = false,
            .tree_key = &.{},
            .tree_val = &.{},
            .out_key = &.{},
            .out_val = &.{},
        };
        try cursor.primeTree();
        return cursor;
    }

    // ── savepoint / commit / rollback ──────────────────────────────

    /// Push a new savepoint onto this Txn (or onto an existing chain
    /// of savepoints if there's already one open and you want to nest
    /// — but you'd call savepoint on the deepest open one, not on
    /// this Txn). Returns the child handle.
    pub fn savepoint(self: *Txn) !*Txn {
        try self.manifest.checkAlive();
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        const child = self.manifest.allocator.create(Txn) catch
            @panic("OOM allocating savepoint Txn");
        child.* = .{
            .manifest = self.manifest,
            .tenant_id = self.tenant_id,
            .tenant_state = self.tenant_state,
            .parent = self,
            .chain_prev = null,
            .chain_next = null,
            .overlay = Overlay.init(self.manifest.allocator),
            .open_child = null,
        };
        self.open_child = child;
        return child;
    }

    pub fn commit(self: *Txn) !void {
        try self.manifest.checkAlive();
        // Capture tenant_state up front so the deferred unlock doesn't
        // dereference `self` after destroy(self) has freed it — under
        // multi-threaded contention the allocator can reuse that slot
        // before the defer runs.
        const ts = self.tenant_state;
        const manifest = self.manifest;
        ts.lock.lock();
        defer ts.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        const is_savepoint = self.parent != null;
        if (self.parent) |parent| {
            // Savepoint commit.
            Overlay.moveInto(&self.overlay, &parent.overlay);
            parent.open_child = null;
        } else {
            // Top-level commit: must be chain head.
            if (ts.chain_head != self) return error.NotChainHead;
            Overlay.moveInto(&self.overlay, &ts.main_overlay);
            if (self.chain_next) |next| {
                next.chain_prev = null;
                ts.chain_head = next;
            } else {
                ts.chain_head = null;
                ts.chain_tail = null;
            }
        }
        self.overlay.deinit();
        manifest.allocator.destroy(self);
        if (is_savepoint) {
            manifest.metrics.recordSavepointCommit();
        } else {
            manifest.metrics.recordTxnCommit();
        }
    }

    pub fn rollback(self: *Txn) void {
        // Same hazard as commit: capture tenant_state before freeing self.
        const ts = self.tenant_state;
        const manifest = self.manifest;
        ts.lock.lock();
        defer ts.lock.unlock();
        const is_savepoint = self.parent != null;
        if (self.parent == null) {
            // Top-level: detach the tail of the chain at self, then
            // free self + all successors.
            if (self.chain_prev) |prev| {
                prev.chain_next = null;
                ts.chain_tail = prev;
            } else {
                ts.chain_head = null;
                ts.chain_tail = null;
            }
            var cur: ?*Txn = self;
            while (cur) |c| {
                const next = c.chain_next;
                c.freeSubtreeLocked();
                cur = next;
            }
        } else {
            // Savepoint: detach from parent + free subtree.
            self.parent.?.open_child = null;
            self.freeSubtreeLocked();
        }
        if (is_savepoint) {
            manifest.metrics.recordSavepointRollback();
        } else {
            manifest.metrics.recordTxnRollback();
        }
    }

    /// Recursively free self + any open_child subtree. Caller holds
    /// tenant_state.lock. Does not touch the chain (caller has
    /// already detached this Txn from the chain if needed).
    fn freeSubtreeLocked(self: *Txn) void {
        if (self.open_child) |child| child.freeSubtreeLocked();
        self.overlay.deinit();
        self.manifest.allocator.destroy(self);
    }
};

/// Add prefix-matching entries from `ov` into `dedup`, transferring
/// ownership of newly-cloned key bytes into `ordered_keys`. Caller
/// owns the dedup map and the ordered_keys list; both are freed by
/// the caller. Closer-to-cursor layers should be called first; entries
/// already in `dedup` are skipped.
fn absorbOverlayLocked(
    allocator: std.mem.Allocator,
    ov: *Overlay,
    prefix: []const u8,
    dedup: *std.StringHashMapUnmanaged(?[]u8),
    ordered_keys: *std.ArrayListUnmanaged([]u8),
) void {
    // Internal bookkeeping: panic on OOM rather than propagate. The
    // caller's `defer dedup.deinit(...)` only frees map storage, not
    // duped values — and an error path with N-1 successful dupes plus
    // 1 failure would leak those values. Sidestep the whole class of
    // problems: kvexp's recovery story is "process dies → raft
    // replays from durable watermark," so OOM in scan setup means we
    // crash hard and the host restarts.
    var it = ov.entries.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.startsWith(u8, k, prefix)) continue;
        const gop = dedup.getOrPut(allocator, k) catch
            @panic("OOM in absorbOverlayLocked.getOrPut");
        if (gop.found_existing) continue;
        const key_copy = allocator.dupe(u8, k) catch
            @panic("OOM duping key in absorbOverlayLocked");
        const val_copy: ?[]u8 = switch (e.value_ptr.*) {
            .value => |v| allocator.dupe(u8, v) catch
                @panic("OOM duping value in absorbOverlayLocked"),
            .tombstone => null,
        };
        gop.key_ptr.* = key_copy;
        gop.value_ptr.* = val_copy;
        ordered_keys.append(allocator, key_copy) catch
            @panic("OOM appending to ordered_keys in absorbOverlayLocked");
    }
}

// ─── TxnPrefixCursor ────────────────────────────────────────────────

pub const TxnPrefixCursor = struct {
    allocator: std.mem.Allocator,
    read_txn: lmdb.Txn,
    lmdb_cur: lmdb.Cursor,
    has_lmdb: bool,
    prefix: []u8,
    overlay: []Entry,
    ov_idx: usize,
    have_tree: bool,
    tree_key: []const u8,
    tree_val: []const u8,
    out_key: []const u8,
    out_val: []const u8,

    pub const Entry = struct {
        key: []u8,
        value: ?[]u8,
        pub fn lessThan(_: void, a: Entry, b: Entry) bool {
            return std.mem.lessThan(u8, a.key, b.key);
        }
    };

    fn primeTree(self: *TxnPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = if (self.prefix.len == 0)
            try self.lmdb_cur.first()
        else
            try self.lmdb_cur.seekGe(self.prefix);
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    fn advanceTree(self: *TxnPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = try self.lmdb_cur.next();
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    pub fn next(self: *TxnPrefixCursor) !bool {
        while (true) {
            const have_ov = self.ov_idx < self.overlay.len;
            if (!self.have_tree and !have_ov) return false;
            if (self.have_tree and have_ov) {
                const ord = std.mem.order(u8, self.tree_key, self.overlay[self.ov_idx].key);
                switch (ord) {
                    .lt => {
                        self.out_key = self.tree_key;
                        self.out_val = self.tree_val;
                        try self.advanceTree();
                        return true;
                    },
                    .gt => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                    .eq => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        try self.advanceTree();
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                }
            } else if (self.have_tree) {
                self.out_key = self.tree_key;
                self.out_val = self.tree_val;
                try self.advanceTree();
                return true;
            } else {
                const e = self.overlay[self.ov_idx];
                self.ov_idx += 1;
                if (e.value) |v| {
                    self.out_key = e.key;
                    self.out_val = v;
                    return true;
                }
                continue;
            }
        }
    }

    pub fn key(self: *const TxnPrefixCursor) []const u8 {
        return self.out_key;
    }
    pub fn value(self: *const TxnPrefixCursor) []const u8 {
        return self.out_val;
    }

    pub fn deinit(self: *TxnPrefixCursor) void {
        if (self.has_lmdb) {
            self.lmdb_cur.close();
            self.read_txn.abort();
        }
        for (self.overlay) |e| {
            self.allocator.free(e.key);
            if (e.value) |v| self.allocator.free(v);
        }
        self.allocator.free(self.overlay);
        self.allocator.free(self.prefix);
        self.* = undefined;
    }
};

// ─── StoreLease ─────────────────────────────────────────────────────

/// Exclusive dispatch handle for a tenant. While held, no other thread
/// can begin Txns for this tenant. Reads through the lease see only
/// `main_overlay` + LMDB — never an in-flight chain (those are this
/// lease holder's own; read through the Txn handle for that).
///
/// Release is mandatory; the lease drops the dispatch_lock on
/// `release`. A Txn obtained via `beginTxn` outlives the lease — the
/// speculative-apply recipe releases the lease as soon as dispatch is
/// done, while the Txn stays in the chain until raft commits or
/// rejects.
pub const StoreLease = struct {
    manifest: *Manifest,
    tenant_id: u64,
    tenant_state: *TenantState,

    /// Open a new top-level Txn at the tail of this tenant's chain.
    /// Reads through the Txn see its own writes + older chain entries
    /// + the tenant's main_overlay + LMDB.
    pub fn beginTxn(self: *StoreLease) !*Txn {
        try self.manifest.checkAlive();
        const ts = self.tenant_state;
        ts.lock.lock();
        defer ts.lock.unlock();

        const txn = self.manifest.allocator.create(Txn) catch
            @panic("OOM allocating Txn in beginTxn");
        txn.* = .{
            .manifest = self.manifest,
            .tenant_id = self.tenant_id,
            .tenant_state = ts,
            .parent = null,
            .chain_prev = ts.chain_tail,
            .chain_next = null,
            .overlay = Overlay.init(self.manifest.allocator),
            .open_child = null,
        };
        if (ts.chain_tail) |tail| {
            tail.chain_next = txn;
        } else {
            ts.chain_head = txn;
        }
        ts.chain_tail = txn;
        return txn;
    }

    /// Point-read against committed state (main_overlay + LMDB). Does
    /// NOT see this lease's in-flight Txns.
    pub fn get(self: *StoreLease, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        const ts = self.tenant_state;
        ts.lock.lock();
        defer ts.lock.unlock();
        self.manifest.metrics.recordGet();
        if (ts.main_overlay.entries.get(key)) |entry| {
            switch (entry) {
                .value => |v| return try allocator.dupe(u8, v),
                .tombstone => return null,
            }
        }
        const dbi = self.manifest.lookupDbi(self.tenant_id) orelse return null;
        var txn = try lmdb.Txn.beginRead(&self.manifest.env);
        defer txn.abort();
        if (try txn.get(dbi, key)) |bytes| return try allocator.dupe(u8, bytes);
        return null;
    }

    /// Prefix scan over committed state (main_overlay + LMDB).
    pub fn scanPrefix(self: *StoreLease, prefix: []const u8) !TxnPrefixCursor {
        var entries: std.ArrayListUnmanaged(TxnPrefixCursor.Entry) = .empty;
        errdefer {
            for (entries.items) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            entries.deinit(self.manifest.allocator);
        }
        {
            const ts = self.tenant_state;
            ts.lock.lock();
            defer ts.lock.unlock();
            var it = ts.main_overlay.entries.iterator();
            while (it.next()) |e| {
                const k = e.key_ptr.*;
                if (!std.mem.startsWith(u8, k, prefix)) continue;
                const k_copy = self.manifest.allocator.dupe(u8, k) catch
                    @panic("OOM duping key in StoreLease.scanPrefix");
                const v_copy: ?[]u8 = switch (e.value_ptr.*) {
                    .value => |v| self.manifest.allocator.dupe(u8, v) catch
                        @panic("OOM duping value in StoreLease.scanPrefix"),
                    .tombstone => null,
                };
                entries.append(self.manifest.allocator, .{ .key = k_copy, .value = v_copy }) catch
                    @panic("OOM appending entry in StoreLease.scanPrefix");
            }
        }
        std.mem.sort(TxnPrefixCursor.Entry, entries.items, {}, TxnPrefixCursor.Entry.lessThan);
        const overlay_slice = entries.toOwnedSlice(self.manifest.allocator) catch
            @panic("OOM materializing slice in StoreLease.scanPrefix");
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        var read_txn: lmdb.Txn = .{ .ptr = null };
        var lmdb_cur: lmdb.Cursor = .{ .ptr = null };
        var has_lmdb = false;
        if (self.manifest.lookupDbi(self.tenant_id)) |dbi| {
            read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
            errdefer read_txn.abort();
            lmdb_cur = try read_txn.openCursor(dbi);
            has_lmdb = true;
        }
        errdefer if (has_lmdb) {
            lmdb_cur.close();
            read_txn.abort();
        };

        var cur: TxnPrefixCursor = .{
            .allocator = self.manifest.allocator,
            .read_txn = read_txn,
            .lmdb_cur = lmdb_cur,
            .has_lmdb = has_lmdb,
            .prefix = self.manifest.allocator.dupe(u8, prefix) catch
                @panic("OOM duping prefix in StoreLease.scanPrefix"),
            .overlay = overlay_slice,
            .ov_idx = 0,
            .have_tree = false,
            .tree_key = &.{},
            .tree_val = &.{},
            .out_key = &.{},
            .out_val = &.{},
        };
        try cur.primeTree();
        return cur;
    }

    /// Drop the dispatch lock. Required. Any Txns this lease opened
    /// remain in the chain until their own commit / rollback.
    pub fn release(self: *StoreLease) void {
        const ts = self.tenant_state;
        const manifest = self.manifest;
        ts.dispatch_lock.unlock();
        manifest.metrics.recordRelease();
        // Drop the reference we took in acquire / tryAcquire. If this
        // tenant was dropped (so the tenants map no longer holds a
        // reference) and we were the last lease holder, this frees
        // the TenantState.
        ts.releaseRef(manifest.allocator);
        self.* = undefined;
    }
};

// ─── Snapshot ───────────────────────────────────────────────────────

const SnapshotEntry = struct {
    key: []u8,
    value: ?[]u8,
    fn lessThan(_: void, a: SnapshotEntry, b: SnapshotEntry) bool {
        return std.mem.lessThan(u8, a.key, b.key);
    }
};

fn freeSnapshotOverlay(
    allocator: std.mem.Allocator,
    map: *std.AutoHashMapUnmanaged(u64, []SnapshotEntry),
) void {
    var it = map.iterator();
    while (it.next()) |entry| {
        for (entry.value_ptr.*) |kv| {
            allocator.free(kv.key);
            if (kv.value) |v| allocator.free(v);
        }
        allocator.free(entry.value_ptr.*);
    }
    map.deinit(allocator);
}

pub const Snapshot = struct {
    manifest: *Manifest,
    read_txn: lmdb.Txn,
    overlays: std.AutoHashMapUnmanaged(u64, []SnapshotEntry),

    pub fn close(self: *Snapshot) void {
        _ = self.manifest.metrics.snapshot_close_total.fetchAdd(1, .monotonic);
        freeSnapshotOverlay(self.manifest.allocator, &self.overlays);
        self.read_txn.abort();
        self.* = undefined;
    }

    pub fn get(
        self: *Snapshot,
        allocator: std.mem.Allocator,
        store_id: u64,
        key: []const u8,
    ) !?[]u8 {
        self.manifest.metrics.recordGet();
        if (self.overlays.get(store_id)) |entries| {
            if (binarySearchEntry(entries, key)) |idx| {
                const e = entries[idx];
                if (e.value) |v| return try allocator.dupe(u8, v);
                return null;
            }
        }
        const dbi = self.manifest.lookupDbi(store_id) orelse return null;
        if (try self.read_txn.get(dbi, key)) |bytes| return try allocator.dupe(u8, bytes);
        return null;
    }

    pub fn scanPrefix(self: *Snapshot, store_id: u64, prefix: []const u8) !SnapshotPrefixCursor {
        var entries: std.ArrayListUnmanaged(TxnPrefixCursor.Entry) = .empty;
        errdefer {
            for (entries.items) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            entries.deinit(self.manifest.allocator);
        }
        if (self.overlays.get(store_id)) |captured| {
            for (captured) |e| {
                if (!std.mem.startsWith(u8, e.key, prefix)) continue;
                const k_copy = self.manifest.allocator.dupe(u8, e.key) catch
                    @panic("OOM duping key in Snapshot.scanPrefix");
                const v_copy: ?[]u8 = if (e.value) |v|
                    (self.manifest.allocator.dupe(u8, v) catch
                        @panic("OOM duping value in Snapshot.scanPrefix"))
                else
                    null;
                entries.append(self.manifest.allocator, .{ .key = k_copy, .value = v_copy }) catch
                    @panic("OOM appending entry in Snapshot.scanPrefix");
            }
        }
        const overlay_slice = entries.toOwnedSlice(self.manifest.allocator) catch
            @panic("OOM materializing slice in Snapshot.scanPrefix");
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        var lmdb_cur: lmdb.Cursor = .{ .ptr = null };
        var has_lmdb = false;
        if (self.manifest.lookupDbi(store_id)) |dbi| {
            lmdb_cur = try self.read_txn.openCursor(dbi);
            has_lmdb = true;
        }
        errdefer if (has_lmdb) lmdb_cur.close();

        var cur: SnapshotPrefixCursor = .{
            .allocator = self.manifest.allocator,
            .lmdb_cur = lmdb_cur,
            .has_lmdb = has_lmdb,
            .prefix = self.manifest.allocator.dupe(u8, prefix) catch
                @panic("OOM duping prefix in Snapshot.scanPrefix"),
            .overlay = overlay_slice,
            .ov_idx = 0,
            .have_tree = false,
            .tree_key = &.{},
            .tree_val = &.{},
            .out_key = &.{},
            .out_val = &.{},
        };
        try cur.primeTree();
        return cur;
    }

    pub fn listStores(self: *Snapshot, allocator: std.mem.Allocator) ![]u64 {
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        var cur = try self.read_txn.openCursor(self.manifest.stores_dbi);
        defer cur.close();
        var pair = try cur.first();
        while (pair) |p| : (pair = try cur.next()) {
            try list.append(allocator, decodeStoreIdKey(p.key));
        }
        return try list.toOwnedSlice(allocator);
    }
};

/// SnapshotPrefixCursor uses TxnPrefixCursor's merge logic; the only
/// difference is it doesn't own the read_txn (the Snapshot does).
pub const SnapshotPrefixCursor = struct {
    allocator: std.mem.Allocator,
    lmdb_cur: lmdb.Cursor,
    has_lmdb: bool,
    prefix: []u8,
    overlay: []TxnPrefixCursor.Entry,
    ov_idx: usize,
    have_tree: bool,
    tree_key: []const u8,
    tree_val: []const u8,
    out_key: []const u8,
    out_val: []const u8,

    fn primeTree(self: *SnapshotPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = if (self.prefix.len == 0)
            try self.lmdb_cur.first()
        else
            try self.lmdb_cur.seekGe(self.prefix);
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    fn advanceTree(self: *SnapshotPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = try self.lmdb_cur.next();
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    pub fn next(self: *SnapshotPrefixCursor) !bool {
        while (true) {
            const have_ov = self.ov_idx < self.overlay.len;
            if (!self.have_tree and !have_ov) return false;
            if (self.have_tree and have_ov) {
                const ord = std.mem.order(u8, self.tree_key, self.overlay[self.ov_idx].key);
                switch (ord) {
                    .lt => {
                        self.out_key = self.tree_key;
                        self.out_val = self.tree_val;
                        try self.advanceTree();
                        return true;
                    },
                    .gt => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                    .eq => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        try self.advanceTree();
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                }
            } else if (self.have_tree) {
                self.out_key = self.tree_key;
                self.out_val = self.tree_val;
                try self.advanceTree();
                return true;
            } else {
                const e = self.overlay[self.ov_idx];
                self.ov_idx += 1;
                if (e.value) |v| {
                    self.out_key = e.key;
                    self.out_val = v;
                    return true;
                }
                continue;
            }
        }
    }

    pub fn key(self: *const SnapshotPrefixCursor) []const u8 {
        return self.out_key;
    }
    pub fn value(self: *const SnapshotPrefixCursor) []const u8 {
        return self.out_val;
    }

    pub fn deinit(self: *SnapshotPrefixCursor) void {
        if (self.has_lmdb) self.lmdb_cur.close();
        for (self.overlay) |e| {
            self.allocator.free(e.key);
            if (e.value) |v| self.allocator.free(v);
        }
        self.allocator.free(self.overlay);
        self.allocator.free(self.prefix);
        self.* = undefined;
    }
};

fn binarySearchEntry(entries: []const SnapshotEntry, key: []const u8) ?usize {
    var lo: usize = 0;
    var hi: usize = entries.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        switch (std.mem.order(u8, entries[mid].key, key)) {
            .lt => lo = mid + 1,
            .gt => hi = mid,
            .eq => return mid,
        }
    }
    return null;
}

// ─── State transfer dump/load ──────────────────────────────────────

pub const SNAPSHOT_MAGIC: u32 = 0x4B565853;
pub const SNAPSHOT_VERSION: u8 = 1;
pub const SNAP_TAG_KV: u8 = 1;
pub const SNAP_TAG_END: u8 = 2;

pub fn dumpSnapshot(snap: *Snapshot, writer: anytype) !void {
    try writer.writeInt(u32, SNAPSHOT_MAGIC, .little);
    try writer.writeByte(SNAPSHOT_VERSION);
    try writer.writeInt(u64, 0, .little); // reserved
    try writer.writeInt(u64, try snap.manifest.durableRaftIdx(), .little);

    const stores = try snap.listStores(snap.manifest.allocator);
    defer snap.manifest.allocator.free(stores);

    for (stores) |id| {
        var cur = try snap.scanPrefix(id, "");
        defer cur.deinit();
        while (try cur.next()) {
            const k = cur.key();
            const v = cur.value();
            try writer.writeByte(SNAP_TAG_KV);
            try writer.writeInt(u64, id, .little);
            try writer.writeInt(u16, @intCast(k.len), .little);
            try writer.writeInt(u16, @intCast(v.len), .little);
            try writer.writeAll(k);
            try writer.writeAll(v);
        }
    }
    try writer.writeByte(SNAP_TAG_END);
}

pub fn loadSnapshot(manifest: *Manifest, reader: anytype) !u64 {
    const magic = try reader.readInt(u32, .little);
    if (magic != SNAPSHOT_MAGIC) return error.InvalidSnapshotFormat;
    const version = try reader.readByte();
    if (version != SNAPSHOT_VERSION) return error.UnsupportedSnapshotVersion;
    _ = try reader.readInt(u64, .little);
    const last_applied = try reader.readInt(u64, .little);

    const KMAX = 256;
    const VMAX = 1 << 20;
    var key_buf: [KMAX]u8 = undefined;
    var val_buf: [VMAX]u8 = undefined;

    // One lease + one Txn per tenant. Leases are released at the end;
    // Txns commit at the end. Both maps share the same key set.
    var leases: std.AutoHashMapUnmanaged(u64, StoreLease) = .empty;
    defer {
        var lit = leases.iterator();
        while (lit.next()) |entry| entry.value_ptr.release();
        leases.deinit(manifest.allocator);
    }
    var txns: std.AutoHashMapUnmanaged(u64, *Txn) = .empty;
    defer txns.deinit(manifest.allocator);
    errdefer {
        var it = txns.iterator();
        while (it.next()) |entry| entry.value_ptr.*.rollback();
    }

    while (true) {
        const tag = try reader.readByte();
        switch (tag) {
            SNAP_TAG_END => break,
            SNAP_TAG_KV => {
                const id = try reader.readInt(u64, .little);
                const klen = try reader.readInt(u16, .little);
                const vlen = try reader.readInt(u16, .little);
                if (klen > key_buf.len or vlen > val_buf.len) return error.InvalidSnapshotFormat;
                try reader.readNoEof(key_buf[0..klen]);
                try reader.readNoEof(val_buf[0..vlen]);
                if (!(try manifest.hasStore(id))) try manifest.createStore(id);
                const lease_gop = leases.getOrPut(manifest.allocator, id) catch
                    @panic("OOM in loadSnapshot.leases.getOrPut");
                if (!lease_gop.found_existing) {
                    lease_gop.value_ptr.* = try manifest.acquire(id);
                }
                const txn_gop = txns.getOrPut(manifest.allocator, id) catch
                    @panic("OOM in loadSnapshot.txns.getOrPut");
                if (!txn_gop.found_existing) {
                    txn_gop.value_ptr.* = try lease_gop.value_ptr.beginTxn();
                }
                try txn_gop.value_ptr.*.put(key_buf[0..klen], val_buf[0..vlen]);
            },
            else => return error.InvalidSnapshotFormat,
        }
    }
    // Commit all txns. Each tenant's chain has exactly one entry (we
    // created it at first put for that tenant), so each commit is the
    // chain head. After this, the txns map holds dangling pointers
    // — clear it so the errdefer above doesn't double-free.
    var it = txns.iterator();
    while (it.next()) |entry| try entry.value_ptr.*.commit();
    txns.clearRetainingCapacity();
    return last_applied;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

const Harness = struct {
    tmp: std.testing.TmpDir,
    // Heap-allocated so the path pointer is stable across Harness moves.
    // Earlier versions stored path as a slice into an in-struct array;
    // since `init()` returns by value, the slice's pointer kept pointing
    // at init's stack-local copy of that array — silently corrupted by
    // any later caller whose stack happened to overwrite that slot.
    path: [:0]u8,
    manifest: *Manifest,

    fn init() !Harness {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &dir_buf);
        const tmp_path = try std.fmt.allocPrint(testing.allocator, "{s}/kvexp-test.mdb", .{dir_path});
        defer testing.allocator.free(tmp_path);
        const path = try testing.allocator.dupeZ(u8, tmp_path);
        errdefer testing.allocator.free(path);
        const manifest = try testing.allocator.create(Manifest);
        errdefer testing.allocator.destroy(manifest);
        try manifest.init(testing.allocator, path, .{
            .max_map_size = 32 * 1024 * 1024,
            .max_stores = 256,
        });
        return .{ .tmp = tmp, .path = path, .manifest = manifest };
    }

    fn deinit(self: *Harness) void {
        self.manifest.deinit();
        testing.allocator.destroy(self.manifest);
        testing.allocator.free(self.path);
        self.tmp.cleanup();
    }

    fn cycle(self: *Harness) !void {
        self.manifest.deinit();
        try self.manifest.init(testing.allocator, self.path, .{
            .max_map_size = 32 * 1024 * 1024,
            .max_stores = 256,
        });
    }

    /// Test helper: acquire the lease, begin a Txn, release the lease.
    /// Mirrors the speculative-apply pattern where the lease is held
    /// only for dispatch and the Txn outlives it. Caller is responsible
    /// for committing or rolling back the returned Txn.
    fn quickTxn(self: *Harness, tenant_id: u64) !*Txn {
        var lease = try self.manifest.acquire(tenant_id);
        defer lease.release();
        return try lease.beginTxn();
    }
};

test "Manifest: create + reopen survives" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(7);
    try h.manifest.flush();
    try h.cycle();
    try testing.expect(try h.manifest.hasStore(7));
}

test "Txn: simple put then commit lands in main_overlay, reads see it" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var txn = try h.quickTxn(1);
    try txn.put("k", "v");
    try txn.commit();

    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Txn: rollback drops writes; main_overlay untouched" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var txn = try h.quickTxn(1);
    try txn.put("k", "v");
    txn.rollback();

    var s = try h.manifest.acquire(1);
    defer s.release();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Txn: reads see its own writes before commit" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var txn = try h.quickTxn(1);
    defer txn.rollback();
    try txn.put("k", "v");
    const got = (try txn.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Txn: chain reads see earlier txns' writes" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    try t1.put("a", "1");
    var t2 = try h.quickTxn(1);
    // t2 should see t1's writes (chain reads backward).
    const got = (try t2.get(testing.allocator, "a")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("1", got);

    t2.rollback();
    t1.rollback();
}

test "Txn: top-level rollback cascades to successors" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    try t1.put("a", "1");
    var t2 = try h.quickTxn(1);
    try t2.put("b", "2");
    var t3 = try h.quickTxn(1);
    try t3.put("c", "3");

    // Rollback t2 → t2 and t3 are dropped.
    t2.rollback();

    // Chain head is t1; only t1 remains open.
    try testing.expect(h.manifest.tenants.get(1).?.chain_head == t1);
    try testing.expect(h.manifest.tenants.get(1).?.chain_tail == t1);

    try t1.commit();

    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("1", got);
    try testing.expect((try s.get(testing.allocator, "b")) == null);
    try testing.expect((try s.get(testing.allocator, "c")) == null);
}

test "Txn: commit must be chain head" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    var t2 = try h.quickTxn(1);
    try testing.expectError(error.NotChainHead, t2.commit());
    t2.rollback();
    try t1.commit();
}

test "Savepoint: commit merges into parent overlay" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    try t.put("a", "1");
    var sp = try t.savepoint();
    try sp.put("b", "2");
    try sp.commit();
    // After sp.commit(), b is in t.overlay.
    const got = (try t.get(testing.allocator, "b")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("2", got);
}

test "Savepoint: rollback drops only the savepoint's writes" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    try t.put("a", "1");
    var sp = try t.savepoint();
    try sp.put("b", "2");
    sp.rollback();
    // a still present, b gone.
    const ga = (try t.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
    try testing.expect((try t.get(testing.allocator, "b")) == null);
}

test "Savepoint: nested savepoints LIFO" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    var sp1 = try t.savepoint();
    try sp1.put("a", "1");
    var sp2 = try sp1.savepoint();
    try sp2.put("b", "2");
    try sp2.commit();      // b merges into sp1
    try sp1.commit();      // a, b merge into t

    const ga = (try t.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
    const gb = (try t.get(testing.allocator, "b")).?;
    defer testing.allocator.free(gb);
    try testing.expectEqualStrings("2", gb);
}

test "Savepoint: commit/rollback while inner savepoint open errors" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    var sp = try t.savepoint();
    try testing.expectError(error.SavepointStillOpen, t.commit());
    try testing.expectError(error.SavepointStillOpen, t.put("x", "y"));
    sp.rollback();
    try t.put("x", "y"); // works after closing the savepoint
}

test "durabilize: drains main_overlay; in-flight txn stays" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    try t1.put("durable", "yes");
    try t1.commit(); // → main_overlay

    var t2 = try h.quickTxn(1);
    try t2.put("inflight", "maybe");

    try h.manifest.flush();    // drains main_overlay; t2 untouched

    // After commit, t2 still sees its own write.
    const got = (try t2.get(testing.allocator, "inflight")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("maybe", got);

    // A fresh Store sees the durable value, not t2's in-flight one.
    var s = try h.manifest.acquire(1);
    defer s.release();
    const dg = (try s.get(testing.allocator, "durable")).?;
    defer testing.allocator.free(dg);
    try testing.expectEqualStrings("yes", dg);
    try testing.expect((try s.get(testing.allocator, "inflight")) == null);

    t2.rollback();
}

test "Txn.scanPrefix: merges chain + main + LMDB in order" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("a", "lmdb-a");
        try t.put("c", "lmdb-c");
        try t.commit();
    }
    try h.manifest.flush(); // a, c are in LMDB now

    var t1 = try h.quickTxn(1);
    defer t1.rollback();
    try t1.put("b", "t1-b");
    var t2 = try h.quickTxn(1);
    defer t2.rollback();
    try t2.put("d", "t2-d");
    try t2.put("a", "t2-a"); // override LMDB's a

    var cur = try t2.scanPrefix("");
    defer cur.deinit();
    var collected: [4]struct { k: []const u8, v: []const u8 } = undefined;
    var n: usize = 0;
    while (try cur.next()) : (n += 1) {
        collected[n] = .{ .k = cur.key(), .v = cur.value() };
    }
    try testing.expectEqual(@as(usize, 4), n);
    try testing.expectEqualStrings("a", collected[0].k);
    try testing.expectEqualStrings("t2-a", collected[0].v);
    try testing.expectEqualStrings("b", collected[1].k);
    try testing.expectEqualStrings("t1-b", collected[1].v);
    try testing.expectEqualStrings("c", collected[2].k);
    try testing.expectEqualStrings("lmdb-c", collected[2].v);
    try testing.expectEqualStrings("d", collected[3].k);
    try testing.expectEqualStrings("t2-d", collected[3].v);
}

test "Txn: delete tombstone shadows everything below" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "lmdb");
        try t.commit();
    }
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    try testing.expect(try t.delete("k"));
    try testing.expect((try t.get(testing.allocator, "k")) == null);
}

test "durableRaftIdx: durabilize(idx) stamps; survives reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try testing.expectEqual(@as(u64, 0), try h.manifest.durableRaftIdx());
    try h.manifest.durabilize(42);
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());
    try h.cycle();
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());
    // commit() preserves the watermark.
    try h.manifest.flush();
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());
}

test "poison blocks writes; reopen clears" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    h.manifest._testPoison();
    try testing.expect(h.manifest.isPoisoned());
    try testing.expectError(error.ManifestPoisoned, h.manifest.durabilize(0));
    try testing.expectError(error.ManifestPoisoned, h.manifest.createStore(2));
    try testing.expectError(error.ManifestPoisoned, h.quickTxn(1));

    try h.cycle();
    try testing.expect(!h.manifest.isPoisoned());
    try testing.expect(try h.manifest.hasStore(1));
}

test "Snapshot: sees main_overlay + LMDB; not in-flight Txn" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("durable", "yes");
        try t.commit();
    }
    try h.manifest.flush();

    // Now a committed-but-not-durabilized write:
    {
        var t = try h.quickTxn(1);
        try t.put("commitnotdurable", "yes");
        try t.commit();
    }
    // And an open Txn:
    var inflight = try h.quickTxn(1);
    defer inflight.rollback();
    try inflight.put("inflight", "no");

    var snap = try h.manifest.openSnapshot();
    defer snap.close();

    const a = (try snap.get(testing.allocator, 1, "durable")).?;
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("yes", a);
    const b = (try snap.get(testing.allocator, 1, "commitnotdurable")).?;
    defer testing.allocator.free(b);
    try testing.expectEqualStrings("yes", b);
    try testing.expect((try snap.get(testing.allocator, 1, "inflight")) == null);
}

test "dumpSnapshot / loadSnapshot round-trip" {
    var src = try Harness.init();
    defer src.deinit();
    try src.manifest.createStore(10);
    {
        var t = try src.quickTxn(10);
        try t.put("a", "1");
        try t.put("b", "2");
        try t.commit();
    }
    try src.manifest.durabilize(99);

    var snap = try src.manifest.openSnapshot();
    defer snap.close();

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    var writer_state = buf.writer(testing.allocator);
    try dumpSnapshot(&snap, &writer_state);

    var dst = try Harness.init();
    defer dst.deinit();
    var reader_state = std.io.fixedBufferStream(buf.items);
    const last = try loadSnapshot(dst.manifest, reader_state.reader());
    try testing.expectEqual(@as(u64, 99), last);
    try dst.manifest.durabilize(last);
    try testing.expectEqual(@as(u64, 99), try dst.manifest.durableRaftIdx());

    var s = try dst.manifest.acquire(10);
    defer s.release();
    const ga = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
}

test "dropStore: while txns open, drops everything cleanly" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    // Open a txn, then drop the store. The txn's memory is freed.
    _ = try h.quickTxn(1);
    // Don't keep a handle past dropStore — that's the "explicit
    // drop-from-under-you" caveat.
    _ = try h.manifest.dropStore(1);
    try h.manifest.flush();
    try testing.expect(!try h.manifest.hasStore(1));
}

test "Property: random workload + simulated crash recovers durable prefix" {
    const runs: u64 = 30;
    var seed: u64 = 1;
    while (seed <= runs) : (seed += 1) {
        propRunOne(seed) catch |err| {
            std.debug.print("\n*** property test failed at seed {} : {} ***\n", .{ seed, err });
            return err;
        };
    }
}

const PROP_NUM_STORES: usize = 4;
const PROP_NUM_KEYS: usize = 10;

const PropModel = struct {
    store_exists: [PROP_NUM_STORES]bool = @splat(false),
    key_present: [PROP_NUM_STORES][PROP_NUM_KEYS]bool = @splat(@splat(false)),
    values: [PROP_NUM_STORES][PROP_NUM_KEYS]u8 = @splat(@splat(0)),
};

fn keyBuf(buf: *[3]u8, i: usize) []const u8 {
    buf[0] = 'k';
    buf[1] = '0' + @as(u8, @intCast(i / 10));
    buf[2] = '0' + @as(u8, @intCast(i % 10));
    return buf[0..3];
}

fn verifyAgainstModel(manifest: *Manifest, model: *const PropModel) !void {
    for (0..PROP_NUM_STORES) |sid_usize| {
        const sid: u64 = @intCast(sid_usize);
        const exists = try manifest.hasStore(sid);
        if (exists != model.store_exists[sid_usize]) return error.StoreExistenceMismatch;
        if (!exists) continue;
        var s = try manifest.acquire(sid);
        defer s.release();
        for (0..PROP_NUM_KEYS) |kid| {
            var kb: [3]u8 = undefined;
            const k = keyBuf(&kb, kid);
            const got_opt = try s.get(testing.allocator, k);
            defer if (got_opt) |g| testing.allocator.free(g);
            const expected_present = model.key_present[sid_usize][kid];
            if (expected_present and got_opt == null) return error.KeyMissing;
            if (!expected_present and got_opt != null) return error.KeyUnexpected;
            if (expected_present) {
                if (got_opt.?.len != 1) return error.ValueLenMismatch;
                if (got_opt.?[0] != model.values[sid_usize][kid]) return error.ValueMismatch;
            }
        }
    }
}

fn propRunOne(seed: u64) !void {
    var rng_state = std.Random.DefaultPrng.init(seed);
    const rng = rng_state.random();
    var h = try Harness.init();
    defer h.deinit();

    var model: PropModel = .{};
    var durable_model: PropModel = .{};
    const op_count: u32 = rng.intRangeAtMost(u32, 30, 80);
    const crash_at: u32 = rng.intRangeLessThan(u32, 0, op_count);

    var i: u32 = 0;
    while (i < op_count) : (i += 1) {
        if (i == crash_at) {
            try h.cycle();
            try verifyAgainstModel(h.manifest, &durable_model);
            return;
        }
        const choice = rng.intRangeLessThan(u32, 0, 100);
        if (choice < 55) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const kid = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
            const v = rng.int(u8);
            var t = try h.quickTxn(@intCast(sid_usize));
            var kb: [3]u8 = undefined;
            var vb: [1]u8 = .{v};
            try t.put(keyBuf(&kb, kid), vb[0..1]);
            try t.commit();
            model.key_present[sid_usize][kid] = true;
            model.values[sid_usize][kid] = v;
        } else if (choice < 70) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const kid = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
            var t = try h.quickTxn(@intCast(sid_usize));
            var kb: [3]u8 = undefined;
            const existed = try t.delete(keyBuf(&kb, kid));
            try t.commit();
            if (existed != model.key_present[sid_usize][kid]) return error.DeleteMismatch;
            model.key_present[sid_usize][kid] = false;
        } else if (choice < 82) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (model.store_exists[sid_usize]) continue;
            try h.manifest.createStore(@intCast(sid_usize));
            model.store_exists[sid_usize] = true;
        } else if (choice < 87) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const dropped = try h.manifest.dropStore(@intCast(sid_usize));
            if (!dropped) return error.DropReturnedFalse;
            model.store_exists[sid_usize] = false;
            for (0..PROP_NUM_KEYS) |k| model.key_present[sid_usize][k] = false;
        } else {
            try h.manifest.flush();
            durable_model = model;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Recovery contract: what survives a crash is exactly what the last
// successful durabilize wrote. h.cycle() simulates a crash by tearing down
// the in-memory manifest and reopening the same LMDB file.
// ───────────────────────────────────────────────────────────────────────────

test "Recovery: committed-but-not-flushed write is lost on crash" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush(); // store is durable; no keys yet

    {
        var t = try h.quickTxn(1);
        try t.put("k", "v"); // committed → main_overlay; NOT yet durable
        try t.commit();
    }

    try h.cycle();

    try testing.expect(try h.manifest.hasStore(1));
    var s = try h.manifest.acquire(1);
    defer s.release();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Recovery: in-flight Txn writes are lost on crash; earlier durable writes survive" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("durable", "yes");
        try t.commit();
    }
    try h.manifest.flush(); // "durable" is now in LMDB

    // Open a Txn that never commits or rolls back before the crash.
    var inflight = try h.quickTxn(1);
    try inflight.put("inflight", "no");

    try h.cycle(); // deinit tears down the open Txn; nothing reached LMDB

    var s = try h.manifest.acquire(1);
    defer s.release();
    const a = (try s.get(testing.allocator, "durable")).?;
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("yes", a);
    try testing.expect((try s.get(testing.allocator, "inflight")) == null);
}

test "Recovery: createStore without flush does not survive crash" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(42);
    try testing.expect(try h.manifest.hasStore(42)); // pending; visible in-memory

    try h.cycle();

    try testing.expect(!try h.manifest.hasStore(42));
}

test "Recovery: dropStore without flush does not survive crash; store comes back with data" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush(); // store + key durable

    _ = try h.manifest.dropStore(1); // pending drop, not flushed
    try testing.expect(!try h.manifest.hasStore(1));

    try h.cycle();

    try testing.expect(try h.manifest.hasStore(1));
    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Recovery: flushed dropStore removes the tenant on reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(7);
    {
        var t = try h.quickTxn(7);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush();
    _ = try h.manifest.dropStore(7);
    try h.manifest.flush(); // drop is now durable

    try h.cycle();

    try testing.expect(!try h.manifest.hasStore(7));
}

test "Recovery: flushed delete tombstone survives crash" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush(); // k=v durable

    {
        var t = try h.quickTxn(1);
        try testing.expect(try t.delete("k"));
        try t.commit();
    }
    try h.manifest.flush(); // tombstone durable

    try h.cycle();

    var s = try h.manifest.acquire(1);
    defer s.release();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Recovery: durabilize across multiple tenants is atomic on reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.createStore(2);
    try h.manifest.createStore(3);
    inline for (.{ .{ 1, "1-a" }, .{ 2, "2-a" }, .{ 3, "3-a" } }) |c| {
        var t = try h.quickTxn(c[0]);
        try t.put("a", c[1]);
        try t.commit();
    }
    try h.manifest.durabilize(100); // one atomic LMDB commit

    try h.cycle();

    try testing.expectEqual(@as(u64, 100), try h.manifest.durableRaftIdx());
    inline for (.{ .{ 1, "1-a" }, .{ 2, "2-a" }, .{ 3, "3-a" } }) |c| {
        try testing.expect(try h.manifest.hasStore(c[0]));
        var s = try h.manifest.acquire(c[0]);
        defer s.release();
        const got = (try s.get(testing.allocator, "a")).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(c[1], got);
    }
}

test "Recovery: successive durabilizes — last state wins after reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "v1");
        try t.commit();
    }
    try h.manifest.durabilize(10);

    {
        var t = try h.quickTxn(1);
        try t.put("k", "v2");
        try t.commit();
    }
    try h.manifest.durabilize(20);

    try h.cycle();

    try testing.expectEqual(@as(u64, 20), try h.manifest.durableRaftIdx());
    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v2", got);
}

test "Recovery: flush() persists data without touching the watermark" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize(50); // watermark=50

    {
        var t = try h.quickTxn(1);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush(); // data durable; watermark unchanged

    try h.cycle();

    try testing.expectEqual(@as(u64, 50), try h.manifest.durableRaftIdx());
    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Recovery: only committed Txns durabilize; rolled-back siblings don't" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);

    var t1 = try h.quickTxn(1);
    try t1.put("a", "1");
    var t2 = try h.quickTxn(1);
    try t2.put("b", "2");
    t2.rollback(); // drops t2
    try t1.commit(); // t1 → main_overlay
    try h.manifest.flush();

    try h.cycle();

    var s = try h.manifest.acquire(1);
    defer s.release();
    const ga = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
    try testing.expect((try s.get(testing.allocator, "b")) == null);
}

test "Recovery: snapshot install becomes durable through durabilize + reopen" {
    // Build the source state and serialize a snapshot, then tear src down
    // entirely so dst doesn't share an open LMDB env with anything else.
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    {
        var src = try Harness.init();
        defer src.deinit();
        try src.manifest.createStore(5);
        {
            var t = try src.quickTxn(5);
            try t.put("k", "v");
            try t.commit();
        }
        try src.manifest.durabilize(77);

        var snap = try src.manifest.openSnapshot();
        defer snap.close();
        var w = buf.writer(testing.allocator);
        try dumpSnapshot(&snap, &w);
    }

    var dst = try Harness.init();
    defer dst.deinit();
    var stream = std.io.fixedBufferStream(buf.items);
    const last = try loadSnapshot(dst.manifest, stream.reader());
    try dst.manifest.durabilize(last);

    try dst.cycle();

    try testing.expectEqual(@as(u64, 77), try dst.manifest.durableRaftIdx());
    try testing.expect(try dst.manifest.hasStore(5));
    var s = try dst.manifest.acquire(5);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Metrics: counters and gauges reflect hot-path activity" {
    var h = try Harness.init();
    defer h.deinit();

    // Baseline: nothing happened yet.
    {
        const m = h.manifest.metricsSnapshot();
        try testing.expectEqual(@as(u64, 0), m.create_store_total);
        try testing.expectEqual(@as(u64, 0), m.acquire_total);
        try testing.expectEqual(@as(u64, 0), m.put_total);
        try testing.expectEqual(@as(u64, 0), m.txn_commit_total);
        try testing.expectEqual(@as(u64, 0), m.durabilize_total);
        try testing.expectEqual(@as(i64, 0), m.active_leases);
        try testing.expectEqual(@as(i64, 0), m.active_snapshots);
    }

    try h.manifest.createStore(1);
    try h.manifest.createStore(2);
    try h.manifest.flush(); // counts as one durabilize

    // Two stores created, one durabilize, no leases yet.
    {
        const m = h.manifest.metricsSnapshot();
        try testing.expectEqual(@as(u64, 2), m.create_store_total);
        try testing.expectEqual(@as(u64, 1), m.durabilize_total);
        try testing.expectEqual(@as(u64, 0), m.durabilize_failed_total);
    }

    {
        var lease = try h.manifest.acquire(1);
        defer lease.release();

        // active_leases gauge tracks the held lease.
        try testing.expectEqual(@as(i64, 1), h.manifest.metricsSnapshot().active_leases);

        var t = try lease.beginTxn();
        try t.put("k", "abc");
        _ = try t.delete("nonexistent");
        try t.commit();
    }

    // After release: active_leases back to 0; counters reflect ops.
    {
        const m = h.manifest.metricsSnapshot();
        try testing.expectEqual(@as(u64, 1), m.acquire_total);
        try testing.expectEqual(@as(u64, 1), m.put_total);
        try testing.expectEqual(@as(u64, 1), m.delete_total);
        try testing.expectEqual(@as(u64, 1), m.txn_commit_total);
        // "k"=1 byte + "abc"=3 bytes = 4 bytes counted
        try testing.expectEqual(@as(u64, 4), m.bytes_put_total);
        try testing.expectEqual(@as(i64, 0), m.active_leases);
    }

    // Rollback path + savepoint counters.
    {
        var lease = try h.manifest.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        var sp = try t.savepoint();
        try sp.put("x", "y");
        sp.rollback();
        t.rollback();
    }
    {
        const m = h.manifest.metricsSnapshot();
        try testing.expectEqual(@as(u64, 1), m.txn_rollback_total);
        try testing.expectEqual(@as(u64, 1), m.savepoint_rollback_total);
    }

    // Snapshot open + close updates gauge.
    {
        var snap = try h.manifest.openSnapshot();
        try testing.expectEqual(@as(i64, 1), h.manifest.metricsSnapshot().active_snapshots);
        snap.close();
        try testing.expectEqual(@as(i64, 0), h.manifest.metricsSnapshot().active_snapshots);
    }
    try testing.expectEqual(@as(u64, 1), h.manifest.metricsSnapshot().snapshot_open_total);

    // Poison increments its counter.
    h.manifest._testPoison();
    try testing.expectEqual(@as(u64, 1), h.manifest.metricsSnapshot().poison_total);
}

test "Metrics: duration histograms record durabilize and snapshot timing" {
    var h = try Harness.init();
    defer h.deinit();

    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "v");
        try t.commit();
    }
    // One real durabilize + one openSnapshot/close pair.
    try h.manifest.durabilize(7);
    var snap = try h.manifest.openSnapshot();
    snap.close();

    const m = h.manifest.metricsSnapshot();

    // Durabilize histogram: one observation, bounded by +Inf, sum > 0.
    try testing.expectEqual(@as(u64, 1), m.durabilize_duration.count);
    try testing.expect(m.durabilize_duration.sum_nanos > 0);
    // The +Inf bucket (cumulative) sees every observation.
    try testing.expectEqual(
        @as(u64, 1),
        m.durabilize_duration.buckets[Histogram.bucket_count - 1],
    );
    // Cumulative: each bucket count is >= the previous.
    var i: usize = 1;
    while (i < Histogram.bucket_count) : (i += 1) {
        try testing.expect(m.durabilize_duration.buckets[i] >= m.durabilize_duration.buckets[i - 1]);
    }

    // Snapshot-open histogram: same shape.
    try testing.expectEqual(@as(u64, 1), m.snapshot_open_duration.count);
    try testing.expect(m.snapshot_open_duration.sum_nanos > 0);
    try testing.expectEqual(
        @as(u64, 1),
        m.snapshot_open_duration.buckets[Histogram.bucket_count - 1],
    );
}

test "Histogram.observe puts values in the right bucket" {
    var hist: Histogram = .{};
    // Force a known set of observations: one in each boundary range.
    hist.observe(1_000_000); //   1 ms → bucket 0 (≤ 5ms)
    hist.observe(7_000_000); //   7 ms → bucket 1 (≤ 10ms)
    hist.observe(20_000_000); //  20 ms → bucket 2 (≤ 25ms)
    hist.observe(20_000_000_000); // 20 s → +Inf bucket

    const snap = hist.snapshot();
    // Cumulative: ≤5ms = 1, ≤10ms = 2, ≤25ms = 3, all the way through
    // and the +Inf bucket picks up the 20s one for a total of 4.
    try testing.expectEqual(@as(u64, 1), snap.buckets[0]);
    try testing.expectEqual(@as(u64, 2), snap.buckets[1]);
    try testing.expectEqual(@as(u64, 3), snap.buckets[2]);
    try testing.expectEqual(@as(u64, 3), snap.buckets[Histogram.bucket_count - 2]); // last finite bound
    try testing.expectEqual(@as(u64, 4), snap.buckets[Histogram.bucket_count - 1]); // +Inf
    try testing.expectEqual(@as(u64, 4), snap.count);
    try testing.expectEqual(
        @as(u64, 1_000_000 + 7_000_000 + 20_000_000 + 20_000_000_000),
        snap.sum_nanos,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Failure injection: trigger a real LMDB error mid-durabilize (MapFull,
// from a deliberately tiny mmap), verify poison fires and recovery works.
// This exercises the `errdefer self.poison()` path with an actual LMDB
// error rather than _testPoison's fake one.
// ───────────────────────────────────────────────────────────────────────────

test "Failure: durabilize MapFull poisons manifest; reopen recovers durable prefix" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try tmp.dir.realpath(".", &dir_buf);
    const tmp_path = try std.fmt.allocPrint(testing.allocator, "{s}/fail.mdb", .{dir_path});
    defer testing.allocator.free(tmp_path);
    const path = try testing.allocator.dupeZ(u8, tmp_path);
    defer testing.allocator.free(path);

    // 256 KiB map. Big enough for the durable anchor below, too small
    // to swallow ~200 KiB of additional 1 KiB values plus B-tree
    // overhead — so the second durabilize commits will be rejected
    // with MapFull.
    const tiny_map: usize = 256 * 1024;

    var mf: Manifest = undefined;
    try mf.init(testing.allocator, path, .{ .max_map_size = tiny_map, .max_stores = 16 });
    var deinit_owed = true;
    defer if (deinit_owed) mf.deinit();

    try mf.createStore(1);

    // Land an anchor in durable storage at raft_idx = 10.
    {
        var lease = try mf.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        try t.put("anchor", "durable");
        try t.commit();
    }
    try mf.durabilize(10);
    try testing.expectEqual(@as(u64, 10), try mf.durableRaftIdx());

    // Now stuff main_overlay with more bytes than the LMDB map can fit.
    {
        var lease = try mf.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        const big_val = [_]u8{0xAB} ** 1024;
        var key_buf: [16]u8 = undefined;
        var i: u32 = 0;
        while (i < 400) : (i += 1) {
            const key = try std.fmt.bufPrint(&key_buf, "k{d:0>14}", .{i});
            try t.put(key, &big_val);
        }
        try t.commit();
    }

    // durabilize fails. Don't be picky about the exact error code —
    // just confirm it errored, poisoned the manifest, and incremented
    // the failure counter.
    const result = mf.durabilize(20);
    try testing.expect(std.meta.isError(result));
    try testing.expect(mf.isPoisoned());
    try testing.expectEqual(@as(u64, 1), mf.metricsSnapshot().durabilize_failed_total);

    // Every mutating API call now returns ManifestPoisoned.
    try testing.expectError(error.ManifestPoisoned, mf.createStore(2));
    try testing.expectError(error.ManifestPoisoned, mf.durabilize(30));
    try testing.expectError(error.ManifestPoisoned, mf.acquire(1));

    // Reopen recovers to the last successful durabilize. The 400-key
    // burst is gone (it was in main_overlay, never reached LMDB); the
    // anchor is intact; raft_idx is still 10, not 20.
    mf.deinit();
    deinit_owed = false;
    try mf.init(testing.allocator, path, .{ .max_map_size = tiny_map, .max_stores = 16 });
    deinit_owed = true;

    try testing.expect(!mf.isPoisoned());
    try testing.expectEqual(@as(u64, 10), try mf.durableRaftIdx());

    var lease = try mf.acquire(1);
    defer lease.release();
    const got = (try lease.get(testing.allocator, "anchor")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("durable", got);

    // None of the post-anchor keys made it.
    try testing.expect((try lease.get(testing.allocator, "k00000000000000")) == null);
}

// ───────────────────────────────────────────────────────────────────────────
// Fuzz tests for loadSnapshot. Snapshots arrive over the network from
// other raft nodes, so the parser is in untrusted-input territory.
// Goal: any byte sequence must either parse cleanly or return a
// well-defined error — never panic, never UAF, never infinite-loop.
// ───────────────────────────────────────────────────────────────────────────

test "Fuzz: loadSnapshot handles random bytes without panicking" {
    var rng_state = std.Random.DefaultPrng.init(0xfa11f00d);
    const rng = rng_state.random();
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var h = try Harness.init();
        defer h.deinit();
        const len = rng.intRangeAtMost(usize, 0, 1024);
        const data = try testing.allocator.alloc(u8, len);
        defer testing.allocator.free(data);
        rng.bytes(data);
        var stream = std.io.fixedBufferStream(data);
        _ = loadSnapshot(h.manifest, stream.reader()) catch {};
    }
}

test "Fuzz: loadSnapshot handles perturbed valid snapshots without panicking" {
    // Build one valid snapshot, then mutate small numbers of random
    // bytes in copies of it. This is far more likely to get past the
    // magic + version check and exercise the inner per-KV parsing
    // paths than fully random bytes.
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    {
        var src = try Harness.init();
        defer src.deinit();
        try src.manifest.createStore(1);
        try src.manifest.createStore(2);
        {
            var t = try src.quickTxn(1);
            try t.put("alpha", "one");
            try t.put("bravo", "two");
            try t.commit();
        }
        {
            var t = try src.quickTxn(2);
            try t.put("xray", "X");
            try t.commit();
        }
        try src.manifest.durabilize(99);
        var snap = try src.manifest.openSnapshot();
        defer snap.close();
        var w = buf.writer(testing.allocator);
        try dumpSnapshot(&snap, &w);
    }

    var rng_state = std.Random.DefaultPrng.init(0xb16b00b5);
    const rng = rng_state.random();
    var iter: usize = 0;
    while (iter < 300) : (iter += 1) {
        const data = try testing.allocator.dupe(u8, buf.items);
        defer testing.allocator.free(data);
        // Flip 1–8 random bytes.
        const n_perturb = rng.intRangeAtMost(usize, 1, 8);
        var p: usize = 0;
        while (p < n_perturb) : (p += 1) {
            const idx = rng.intRangeLessThan(usize, 0, data.len);
            data[idx] = rng.int(u8);
        }
        var h = try Harness.init();
        defer h.deinit();
        var stream = std.io.fixedBufferStream(data);
        _ = loadSnapshot(h.manifest, stream.reader()) catch {};
    }
}

test "Fuzz: loadSnapshot handles truncated snapshots without panicking" {
    // Build a valid snapshot, then feed every possible prefix length.
    // Catches off-by-one mistakes in the parser's bounds checking
    // (e.g. reading klen but not the key bytes).
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    {
        var src = try Harness.init();
        defer src.deinit();
        try src.manifest.createStore(1);
        {
            var t = try src.quickTxn(1);
            try t.put("k", "v");
            try t.put("key2", "longer-value");
            try t.commit();
        }
        try src.manifest.durabilize(7);
        var snap = try src.manifest.openSnapshot();
        defer snap.close();
        var w = buf.writer(testing.allocator);
        try dumpSnapshot(&snap, &w);
    }

    var trunc: usize = 0;
    while (trunc <= buf.items.len) : (trunc += 1) {
        var h = try Harness.init();
        defer h.deinit();
        var stream = std.io.fixedBufferStream(buf.items[0..trunc]);
        _ = loadSnapshot(h.manifest, stream.reader()) catch {};
    }
}

test "OverlayCap: put rejected past the cap; durabilize drains and unblocks" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try tmp.dir.realpath(".", &dir_buf);
    const tmp_path = try std.fmt.allocPrint(testing.allocator, "{s}/cap.mdb", .{dir_path});
    defer testing.allocator.free(tmp_path);
    const path = try testing.allocator.dupeZ(u8, tmp_path);
    defer testing.allocator.free(path);

    // Tight per-store cap. With ~100 B values + 4 B keys, that's
    // headroom for ~9 entries before the 10th is rejected.
    const cap: usize = 1024;

    var mf: Manifest = undefined;
    try mf.init(testing.allocator, path, .{
        .max_map_size = 4 * 1024 * 1024,
        .max_stores = 16,
        .max_overlay_bytes_per_store = cap,
    });
    defer mf.deinit();
    try mf.createStore(1);
    try mf.flush();

    // Fill up close to but under the cap; commits land in main_overlay.
    {
        var lease = try mf.acquire(1);
        defer lease.release();
        const value = [_]u8{0xAB} ** 100;
        var key_buf: [4]u8 = undefined;
        var i: u32 = 0;
        // 9 entries × (4 + 100) = 936 bytes ≤ 1024 cap.
        while (i < 9) : (i += 1) {
            var t = try lease.beginTxn();
            const key = try std.fmt.bufPrint(&key_buf, "{x:0>4}", .{i});
            try t.put(key, &value);
            try t.commit();
        }
    }

    // Next put would push us to 1040 > 1024. Expect rejection.
    {
        var lease = try mf.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        defer t.rollback();
        const value = [_]u8{0xAB} ** 100;
        const result = t.put("ffff", &value);
        try testing.expectError(error.OverlayCapExceeded, result);
    }

    // Durabilize drains main_overlay → 0 bytes used. Next put succeeds.
    try mf.flush();
    {
        var lease = try mf.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        const value = [_]u8{0xAB} ** 100;
        try t.put("ffff", &value);
        try t.commit();
    }
}

test "OverlayCap: cap counts main_overlay + this txn's overlay together" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try tmp.dir.realpath(".", &dir_buf);
    const tmp_path = try std.fmt.allocPrint(testing.allocator, "{s}/cap.mdb", .{dir_path});
    defer testing.allocator.free(tmp_path);
    const path = try testing.allocator.dupeZ(u8, tmp_path);
    defer testing.allocator.free(path);

    var mf: Manifest = undefined;
    try mf.init(testing.allocator, path, .{
        .max_map_size = 4 * 1024 * 1024,
        .max_stores = 16,
        .max_overlay_bytes_per_store = 512,
    });
    defer mf.deinit();
    try mf.createStore(1);
    try mf.flush();

    // Land 400 bytes in main_overlay (4 keys × 100B values + 4B keys = 416B).
    {
        var lease = try mf.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        const value = [_]u8{0xAB} ** 100;
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            var key_buf: [4]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "{x:0>4}", .{i});
            try t.put(key, &value);
        }
        try t.commit();
    }

    // Now open a Txn. main_overlay = 416 B, cap = 512 B → 96 B headroom.
    // One more 104-byte put would put us at 520 > 512. Reject.
    var lease = try mf.acquire(1);
    defer lease.release();
    var t = try lease.beginTxn();
    defer t.rollback();
    const value = [_]u8{0xAB} ** 100;
    try testing.expectError(error.OverlayCapExceeded, t.put("0099", &value));

    // A much smaller put fits: 4B key + 1B value = 5B. 416 + 5 = 421 ≤ 512.
    try t.put("xx99", "z");
}
// Multi-threaded property test: N workers race on M pre-created tenants
// while a checkpointer thread periodically durabilizes. Each commit (and
// the checkpointer's "snapshot model" step) is coordinated by an RwLock so
// the captured durable_model accurately reflects what survives a cycle.
// After workers finish: one final durabilize + cycle + verify.
// ───────────────────────────────────────────────────────────────────────────

const MT_NUM_WORKERS: usize = 4;
const MT_NUM_TENANTS: usize = 4;
const MT_OPS_PER_WORKER: u32 = 200;
const MT_CHECKPOINT_SLEEP_NS: u64 = 500 * std.time.ns_per_us; // 0.5 ms

const MtCtx = struct {
    manifest: *Manifest,
    rwlock: *std.Thread.RwLock,
    model_lock: *std.Thread.Mutex,
    model: *PropModel,
    seed: u64,
    err: *std.atomic.Value(usize),
};

const MtCheckpointerCtx = struct {
    manifest: *Manifest,
    rwlock: *std.Thread.RwLock,
    model_lock: *std.Thread.Mutex,
    model: *PropModel,
    durable: *PropModel,
    stop: *std.atomic.Value(bool),
    err: *std.atomic.Value(usize),
    durabilize_count: *std.atomic.Value(u32),
};

fn mtWorker(ctx: *MtCtx) void {
    mtWorkerInner(ctx) catch |err| {
        _ = ctx.err.cmpxchgStrong(0, @intFromError(err), .acq_rel, .acquire);
    };
}

fn mtWorkerInner(ctx: *MtCtx) !void {
    var rng_state = std.Random.DefaultPrng.init(ctx.seed);
    const rng = rng_state.random();
    var i: u32 = 0;
    while (i < MT_OPS_PER_WORKER) : (i += 1) {
        const tenant_usize = rng.intRangeLessThan(usize, 0, MT_NUM_TENANTS);
        const tenant: u64 = @intCast(tenant_usize);
        const key_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
        const choice = rng.intRangeLessThan(u32, 0, 100);

        ctx.rwlock.lockShared();
        defer ctx.rwlock.unlockShared();

        var lease = try ctx.manifest.acquire(tenant);
        defer lease.release();
        var t = try lease.beginTxn();

        var kb: [3]u8 = undefined;
        const key = keyBuf(&kb, key_usize);

        if (choice < 70) {
            const v: u8 = rng.int(u8);
            const vb = [_]u8{v};
            t.put(key, vb[0..1]) catch |e| {
                t.rollback();
                return e;
            };
            try t.commit();
            ctx.model_lock.lock();
            defer ctx.model_lock.unlock();
            ctx.model.key_present[tenant_usize][key_usize] = true;
            ctx.model.values[tenant_usize][key_usize] = v;
        } else {
            _ = t.delete(key) catch |e| {
                t.rollback();
                return e;
            };
            try t.commit();
            ctx.model_lock.lock();
            defer ctx.model_lock.unlock();
            ctx.model.key_present[tenant_usize][key_usize] = false;
        }
    }
}

fn mtCheckpointer(ctx: *MtCheckpointerCtx) void {
    while (!ctx.stop.load(.acquire)) {
        std.Thread.sleep(MT_CHECKPOINT_SLEEP_NS);
        ctx.rwlock.lock();
        defer ctx.rwlock.unlock();
        ctx.manifest.flush() catch |err| {
            _ = ctx.err.cmpxchgStrong(0, @intFromError(err), .acq_rel, .acquire);
            return;
        };
        ctx.model_lock.lock();
        defer ctx.model_lock.unlock();
        ctx.durable.* = ctx.model.*;
        _ = ctx.durabilize_count.fetchAdd(1, .monotonic);
    }
}

fn mtRunOne(seed: u64) !void {
    var h = try Harness.init();
    defer h.deinit();

    // Pre-create all tenants and flush.
    for (0..MT_NUM_TENANTS) |i| try h.manifest.createStore(@intCast(i));
    try h.manifest.flush();

    var model: PropModel = .{};
    for (0..MT_NUM_TENANTS) |i| model.store_exists[i] = true;
    var durable: PropModel = model;

    var rwlock: std.Thread.RwLock = .{};
    var model_lock: std.Thread.Mutex = .{};
    var stop = std.atomic.Value(bool).init(false);
    var err = std.atomic.Value(usize).init(0);
    var durabilize_count = std.atomic.Value(u32).init(0);

    var ctxs: [MT_NUM_WORKERS]MtCtx = undefined;
    var threads: [MT_NUM_WORKERS]std.Thread = undefined;
    for (0..MT_NUM_WORKERS) |i| {
        ctxs[i] = .{
            .manifest = h.manifest,
            .rwlock = &rwlock,
            .model_lock = &model_lock,
            .model = &model,
            .seed = seed *% 31 +% @as(u64, i),
            .err = &err,
        };
        threads[i] = try std.Thread.spawn(.{}, mtWorker, .{&ctxs[i]});
    }

    var ckpt_ctx: MtCheckpointerCtx = .{
        .manifest = h.manifest,
        .rwlock = &rwlock,
        .model_lock = &model_lock,
        .model = &model,
        .durable = &durable,
        .stop = &stop,
        .err = &err,
        .durabilize_count = &durabilize_count,
    };
    const ckpt = try std.Thread.spawn(.{}, mtCheckpointer, .{&ckpt_ctx});

    for (threads) |th| th.join();
    stop.store(true, .release);
    ckpt.join();

    const code = err.load(.acquire);
    if (code != 0) {
        std.debug.print("MT worker/ckpt errored: code {}\n", .{code});
        return error.MtWorkerErrored;
    }

    // Final durabilize + snapshot.
    rwlock.lock();
    try h.manifest.flush();
    durable = model;
    rwlock.unlock();

    // Crash + verify.
    try h.cycle();
    try verifyAgainstModel(h.manifest, &durable);

    // Sanity: the checkpointer should have run at least once during the
    // workers. If this fires, the test isn't actually exercising the
    // concurrent durabilize path.
    if (durabilize_count.load(.acquire) == 0) return error.CheckpointerStarved;
}

test "MT property: workers + checkpointer + crash recovers durable prefix" {
    const seeds: u64 = 3;
    var seed: u64 = 1;
    while (seed <= seeds) : (seed += 1) {
        mtRunOne(seed) catch |e| {
            std.debug.print("\n*** MT property failed at seed {}: {}\n", .{ seed, e });
            return e;
        };
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Speculative-apply property test: workers release the lease *before*
// committing — the production raft pattern. Txns accumulate in per-tenant
// chains until a single committer thread drains them in raft order (which
// is queue insertion order, which — because we push while still holding the
// lease — equals per-tenant chain order). Exercises chain length > 1, the
// chain-head commit gate, and cross-thread Txn handoff.
// ───────────────────────────────────────────────────────────────────────────

const SPEC_NUM_WORKERS: usize = 4;
const SPEC_OPS_PER_WORKER: u32 = 150;

const SpecPending = struct {
    txn: *Txn,
    tenant_usize: usize,
    key_usize: usize,
    value: u8,
};

const SpecQueue = struct {
    mu: std.Thread.Mutex = .{},
    items: std.ArrayListUnmanaged(SpecPending) = .empty,
    head: usize = 0,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) SpecQueue {
        return .{ .allocator = allocator };
    }
    fn deinit(self: *SpecQueue) void {
        self.items.deinit(self.allocator);
    }
    fn push(self: *SpecQueue, p: SpecPending) !void {
        self.mu.lock();
        defer self.mu.unlock();
        try self.items.append(self.allocator, p);
    }
    fn pop(self: *SpecQueue) ?SpecPending {
        self.mu.lock();
        defer self.mu.unlock();
        if (self.head >= self.items.items.len) return null;
        const item = self.items.items[self.head];
        self.head += 1;
        return item;
    }
};

const SpecWorkerCtx = struct {
    manifest: *Manifest,
    queue: *SpecQueue,
    seed: u64,
    err: *std.atomic.Value(usize),
};

const SpecCommitterCtx = struct {
    queue: *SpecQueue,
    stop: *std.atomic.Value(bool),
    model_lock: *std.Thread.Mutex,
    model: *PropModel,
    err: *std.atomic.Value(usize),
};

fn specWorker(ctx: *SpecWorkerCtx) void {
    specWorkerInner(ctx) catch |err| {
        _ = ctx.err.cmpxchgStrong(0, @intFromError(err), .acq_rel, .acquire);
    };
}

fn specWorkerInner(ctx: *SpecWorkerCtx) !void {
    var rng_state = std.Random.DefaultPrng.init(ctx.seed);
    const rng = rng_state.random();
    var i: u32 = 0;
    while (i < SPEC_OPS_PER_WORKER) : (i += 1) {
        const tenant_usize = rng.intRangeLessThan(usize, 0, MT_NUM_TENANTS);
        const tenant: u64 = @intCast(tenant_usize);
        const key_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
        const v: u8 = rng.int(u8);

        var lease = try ctx.manifest.acquire(tenant);
        // Hold the lease across beginTxn + put + push: per-tenant push
        // order must equal per-tenant chain order or the committer will
        // hit NotChainHead.
        var t = try lease.beginTxn();
        var kb: [3]u8 = undefined;
        const vb = [_]u8{v};
        t.put(keyBuf(&kb, key_usize), vb[0..1]) catch |e| {
            t.rollback();
            lease.release();
            return e;
        };
        try ctx.queue.push(.{
            .txn = t,
            .tenant_usize = tenant_usize,
            .key_usize = key_usize,
            .value = v,
        });
        lease.release();
    }
}

fn specCommitter(ctx: *SpecCommitterCtx) void {
    while (true) {
        if (ctx.queue.pop()) |pending| {
            pending.txn.commit() catch |e| {
                _ = ctx.err.cmpxchgStrong(0, @intFromError(e), .acq_rel, .acquire);
                return;
            };
            ctx.model_lock.lock();
            defer ctx.model_lock.unlock();
            ctx.model.key_present[pending.tenant_usize][pending.key_usize] = true;
            ctx.model.values[pending.tenant_usize][pending.key_usize] = pending.value;
        } else {
            if (ctx.stop.load(.acquire)) return;
            std.Thread.yield() catch {};
        }
    }
}

fn specRunOne(seed: u64) !void {
    var h = try Harness.init();
    defer h.deinit();

    for (0..MT_NUM_TENANTS) |i| try h.manifest.createStore(@intCast(i));
    try h.manifest.flush();

    var model: PropModel = .{};
    for (0..MT_NUM_TENANTS) |i| model.store_exists[i] = true;

    var queue = SpecQueue.init(testing.allocator);
    defer queue.deinit();
    var model_lock: std.Thread.Mutex = .{};
    var stop = std.atomic.Value(bool).init(false);
    var err = std.atomic.Value(usize).init(0);

    var ctxs: [SPEC_NUM_WORKERS]SpecWorkerCtx = undefined;
    var threads: [SPEC_NUM_WORKERS]std.Thread = undefined;
    for (0..SPEC_NUM_WORKERS) |i| {
        ctxs[i] = .{
            .manifest = h.manifest,
            .queue = &queue,
            .seed = seed *% 37 +% @as(u64, i),
            .err = &err,
        };
        threads[i] = try std.Thread.spawn(.{}, specWorker, .{&ctxs[i]});
    }

    var committer_ctx: SpecCommitterCtx = .{
        .queue = &queue,
        .stop = &stop,
        .model_lock = &model_lock,
        .model = &model,
        .err = &err,
    };
    const committer = try std.Thread.spawn(.{}, specCommitter, .{&committer_ctx});

    for (threads) |th| th.join();
    stop.store(true, .release);
    committer.join();

    const code = err.load(.acquire);
    if (code != 0) {
        std.debug.print("Spec worker/committer errored: code {}\n", .{code});
        return error.SpecErrored;
    }

    try h.manifest.flush();
    try h.cycle();
    try verifyAgainstModel(h.manifest, &model);
}

test "Spec property: lease-released-before-commit + committer + recovery" {
    const seeds: u64 = 3;
    var seed: u64 = 1;
    while (seed <= seeds) : (seed += 1) {
        specRunOne(seed) catch |e| {
            std.debug.print("\n*** Spec property failed at seed {}: {}\n", .{ seed, e });
            return e;
        };
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Concurrent-snapshot property test: workers do random puts/deletes, a
// snapper thread continuously opens snapshots and reads every (tenant, key)
// without taking the worker-side rwlock — verifying the README claim that
// openSnapshot does not contend with lease holders. Each returned value
// must be a valid 1-byte payload; nulls are allowed (the key may legitimately
// have been deleted or never written). Final state verified after cycle.
// ───────────────────────────────────────────────────────────────────────────

const SNAP_NUM_WORKERS: usize = 4;
const SNAP_OPS_PER_WORKER: u32 = 150;

const SnapCtx = struct {
    manifest: *Manifest,
    model_lock: *std.Thread.Mutex,
    model: *PropModel,
    seed: u64,
    err: *std.atomic.Value(usize),
};

const SnapperCtx = struct {
    manifest: *Manifest,
    stop: *std.atomic.Value(bool),
    err: *std.atomic.Value(usize),
    snapshot_count: *std.atomic.Value(u32),
};

fn snapWorker(ctx: *SnapCtx) void {
    snapWorkerInner(ctx) catch |err| {
        _ = ctx.err.cmpxchgStrong(0, @intFromError(err), .acq_rel, .acquire);
    };
}

fn snapWorkerInner(ctx: *SnapCtx) !void {
    var rng_state = std.Random.DefaultPrng.init(ctx.seed);
    const rng = rng_state.random();
    var i: u32 = 0;
    while (i < SNAP_OPS_PER_WORKER) : (i += 1) {
        const tenant_usize = rng.intRangeLessThan(usize, 0, MT_NUM_TENANTS);
        const tenant: u64 = @intCast(tenant_usize);
        const key_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
        const choice = rng.intRangeLessThan(u32, 0, 100);

        var lease = try ctx.manifest.acquire(tenant);
        defer lease.release();
        var t = try lease.beginTxn();

        var kb: [3]u8 = undefined;
        const key = keyBuf(&kb, key_usize);

        if (choice < 70) {
            const v: u8 = rng.int(u8);
            const vb = [_]u8{v};
            t.put(key, vb[0..1]) catch |e| {
                t.rollback();
                return e;
            };
            try t.commit();
            ctx.model_lock.lock();
            defer ctx.model_lock.unlock();
            ctx.model.key_present[tenant_usize][key_usize] = true;
            ctx.model.values[tenant_usize][key_usize] = v;
        } else {
            _ = t.delete(key) catch |e| {
                t.rollback();
                return e;
            };
            try t.commit();
            ctx.model_lock.lock();
            defer ctx.model_lock.unlock();
            ctx.model.key_present[tenant_usize][key_usize] = false;
        }
    }
}

fn snapper(ctx: *SnapperCtx) void {
    snapperInner(ctx) catch |err| {
        _ = ctx.err.cmpxchgStrong(0, @intFromError(err), .acq_rel, .acquire);
    };
}

fn snapperInner(ctx: *SnapperCtx) !void {
    while (!ctx.stop.load(.acquire)) {
        var snap = try ctx.manifest.openSnapshot();
        defer snap.close();
        var tid: usize = 0;
        while (tid < MT_NUM_TENANTS) : (tid += 1) {
            var kid: usize = 0;
            while (kid < PROP_NUM_KEYS) : (kid += 1) {
                var kb: [3]u8 = undefined;
                const got_opt = try snap.get(testing.allocator, @intCast(tid), keyBuf(&kb, kid));
                defer if (got_opt) |g| testing.allocator.free(g);
                if (got_opt) |g| {
                    if (g.len != 1) return error.InvalidValueLen;
                }
            }
        }
        _ = ctx.snapshot_count.fetchAdd(1, .monotonic);
    }
}

fn snapRunOne(seed: u64) !void {
    var h = try Harness.init();
    defer h.deinit();

    for (0..MT_NUM_TENANTS) |i| try h.manifest.createStore(@intCast(i));
    try h.manifest.flush();

    var model: PropModel = .{};
    for (0..MT_NUM_TENANTS) |i| model.store_exists[i] = true;

    var model_lock: std.Thread.Mutex = .{};
    var stop = std.atomic.Value(bool).init(false);
    var err = std.atomic.Value(usize).init(0);
    var snapshot_count = std.atomic.Value(u32).init(0);

    var ctxs: [SNAP_NUM_WORKERS]SnapCtx = undefined;
    var threads: [SNAP_NUM_WORKERS]std.Thread = undefined;
    for (0..SNAP_NUM_WORKERS) |i| {
        ctxs[i] = .{
            .manifest = h.manifest,
            .model_lock = &model_lock,
            .model = &model,
            .seed = seed *% 41 +% @as(u64, i),
            .err = &err,
        };
        threads[i] = try std.Thread.spawn(.{}, snapWorker, .{&ctxs[i]});
    }

    var snapper_ctx: SnapperCtx = .{
        .manifest = h.manifest,
        .stop = &stop,
        .err = &err,
        .snapshot_count = &snapshot_count,
    };
    const snapper_thread = try std.Thread.spawn(.{}, snapper, .{&snapper_ctx});

    for (threads) |th| th.join();
    stop.store(true, .release);
    snapper_thread.join();

    const code = err.load(.acquire);
    if (code != 0) {
        std.debug.print("Snap worker/snapper errored: code {}\n", .{code});
        return error.SnapErrored;
    }

    if (snapshot_count.load(.acquire) == 0) return error.SnapperStarved;

    try h.manifest.flush();
    try h.cycle();
    try verifyAgainstModel(h.manifest, &model);
}

test "Snap property: workers + concurrent snapper + crash recovers state" {
    const seeds: u64 = 3;
    var seed: u64 = 1;
    while (seed <= seeds) : (seed += 1) {
        snapRunOne(seed) catch |e| {
            std.debug.print("\n*** Snap property failed at seed {}: {}\n", .{ seed, e });
            return e;
        };
    }
}
