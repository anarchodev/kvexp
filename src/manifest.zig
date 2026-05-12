//! Multi-tenant manifest over LMDB + an in-memory write buffer (memtable).
//!
//! Architecture: each `Store` is an LMDB sub-DBI (`MDB_CREATE`).
//! `Store.put` / `Store.delete` write to an in-memory per-store
//! Overlay (the memtable). `durabilize()` opens a single LMDB write
//! transaction, drains every overlay into its sub-DBI, also writes
//! the `last_applied_raft_idx` watermark into a `_meta` sub-DBI, then
//! commits — that commit is the atomic durability point.
//!
//! Recovery on `init`: open the env, read `last_applied_raft_idx`
//! from `_meta`, populate the per-store DBI handle map from the
//! `_stores` sub-DBI (which lists every live store_id). The raft
//! layer replays log entries with idx > last_applied_raft_idx.
//!
//! Concurrency:
//!   * Multiple workers may put/delete/get concurrently on **distinct**
//!     stores with no shared lock (per-store overlay locks).
//!   * Workers on the **same** store serialize via `storeLock(id)`
//!     (caller-style; Store.put/delete take it internally).
//!   * `durabilize` is single-caller (`durabilize_lock`) and is the
//!     only place that opens an LMDB write txn. LMDB's single-writer
//!     model is acceptable here because writes happen in batched
//!     bursts at durabilize, not per-put.
//!
//! Reads (`Store.get`, `Store.scanPrefix`, `Snapshot.get/scanPrefix`)
//! open LMDB read txns. Read txns are cheap; LMDB's reader table is
//! lock-free for concurrent readers.

const std = @import("std");
const lmdb = @import("lmdb.zig");
const overlay_mod = @import("overlay.zig");
const Overlay = overlay_mod.Overlay;
const OverlayEntry = overlay_mod.OverlayEntry;

pub const InitOptions = struct {
    /// Maximum number of stores (LMDB sub-DBIs). Plus 2 reserved for
    /// kvexp's internal `_meta` and `_stores`. Fixed at env open.
    max_stores: u32 = 65534,
    /// LMDB mmap size in bytes. Sparse — only touched pages cost
    /// memory. Default 16 GiB.
    max_map_size: usize = 16 * 1024 * 1024 * 1024,
    /// Skip metadata-page fsync on commit. Saves ~30% commit latency
    /// at the cost of a small window where the last commit may roll
    /// back on crash. Default off; raft replays the missing entries
    /// after such a rollback, so enabling this is sometimes the right
    /// call — but we leave the conservative choice as default.
    no_meta_sync: bool = false,
    /// Disable ALL fsync. Tests-only territory; production data loss
    /// on the slightest crash.
    no_sync: bool = false,
};

pub const SpecificError = error{
    StoreAlreadyExists,
    StoreNotFound,
    ManifestPoisoned,
    InvalidSnapshotFormat,
    UnsupportedSnapshotVersion,
};

const META_DBI_NAME: [:0]const u8 = "_meta";
const STORES_DBI_NAME: [:0]const u8 = "_stores";
const META_RAFT_APPLY_KEY: []const u8 = "raft_apply_idx";

/// LMDB DBI name for a per-store sub-DBI. "s_" + 16 hex chars + null = 19 bytes.
fn storeDbiName(id: u64, buf: *[19]u8) [:0]const u8 {
    return std.fmt.bufPrintZ(buf, "s_{x:0>16}", .{id}) catch unreachable;
}

/// 8-byte big-endian key for the `_stores` directory listing.
fn encodeStoreIdKey(id: u64, buf: *[8]u8) []const u8 {
    std.mem.writeInt(u64, buf, id, .big);
    return buf;
}

fn decodeStoreIdKey(bytes: []const u8) u64 {
    std.debug.assert(bytes.len == 8);
    return std.mem.readInt(u64, bytes[0..8], .big);
}

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    env: lmdb.Env,
    meta_dbi: lmdb.Dbi,
    stores_dbi: lmdb.Dbi,

    /// store_id → DBI handle for stores already DURABLE in LMDB. The
    /// existence-test (`hasStore`) layers `pending_creates` on top
    /// and `pending_drops` underneath this set. Populated at init
    /// from `_stores`; mutated only by durabilize.
    dbis: std.AutoHashMapUnmanaged(u64, lmdb.Dbi) = .empty,
    /// Store IDs that have been `createStore`-d in memory since the
    /// last durabilize. No DBI exists yet; writes to such a store
    /// land in the overlay only. Drained at durabilize (DBI created,
    /// listing entry written, moved into `dbis`).
    pending_creates: std.AutoHashMapUnmanaged(u64, void) = .empty,
    /// Store IDs that have been `dropStore`-d in memory since the
    /// last durabilize. The DBI may still be in `dbis`; drained at
    /// durabilize (DBI emptied, listing entry deleted, removed from
    /// `dbis`).
    pending_drops: std.AutoHashMapUnmanaged(u64, void) = .empty,
    /// store_id → overlay. Lazily created on first write.
    overlays: std.AutoHashMapUnmanaged(u64, *Overlay) = .empty,
    /// Per-store writer mutex for serializing concurrent puts/deletes
    /// on the same store. Lazily allocated.
    store_locks: std.AutoHashMapUnmanaged(u64, *std.Thread.Mutex) = .empty,

    /// Guards `dbis`. Short critical sections; createStore/dropStore
    /// are rare.
    dbis_lock: std.Thread.Mutex = .{},
    /// Guards `overlays`. Brief.
    overlays_lock: std.Thread.Mutex = .{},
    /// Guards `store_locks`. Brief.
    store_locks_lock: std.Thread.Mutex = .{},
    /// Single-caller serialization for durabilize. Held outermost.
    durabilize_lock: std.Thread.Mutex = .{},

    poisoned: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

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
        };
        self.env = try lmdb.Env.open(path, .{
            .max_dbs = options.max_stores + 2,
            .max_map_size = options.max_map_size,
            .no_meta_sync = options.no_meta_sync,
            .no_sync = options.no_sync,
            // We always run single-process; the no_lock flag skips
            // LMDB's reader-table file lock, which costs us nothing.
            .no_lock = true,
        });
        errdefer self.env.close();

        // Bootstrap: open `_meta` and `_stores` and load the watermark.
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
            // Need to re-open the DBIs in this txn to use them.
            const stores_dbi = try txn.openDbi(STORES_DBI_NAME, false);
            self.stores_dbi = stores_dbi;
            self.meta_dbi = try txn.openDbi(META_DBI_NAME, false);

            var cur = try txn.openCursor(stores_dbi);
            defer cur.close();
            var pair = try cur.first();
            while (pair) |p| : (pair = try cur.next()) {
                const id = decodeStoreIdKey(p.key);
                var name_buf: [19]u8 = undefined;
                const name = storeDbiName(id, &name_buf);
                const dbi = try txn.openDbi(name, false);
                try self.dbis.put(self.allocator, id, dbi);
            }
            try txn.commit();
        }
    }

    fn deinitMaps(self: *Manifest) void {
        self.dbis.deinit(self.allocator);
        self.pending_creates.deinit(self.allocator);
        self.pending_drops.deinit(self.allocator);
        var ov_it = self.overlays.iterator();
        while (ov_it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.overlays.deinit(self.allocator);
        var sl_it = self.store_locks.valueIterator();
        while (sl_it.next()) |m| self.allocator.destroy(m.*);
        self.store_locks.deinit(self.allocator);
    }

    pub fn deinit(self: *Manifest) void {
        self.deinitMaps();
        self.env.close();
        self.* = undefined;
    }

    // ── raft watermark ──────────────────────────────────────────────

    /// The raft index at the last successful `durabilize`. Reads from
    /// LMDB (one short read txn). Returns 0 if no durabilize has
    /// written a watermark yet.
    pub fn durableRaftIdx(self: *Manifest) !u64 {
        var txn = try lmdb.Txn.beginRead(&self.env);
        defer txn.abort();
        const bytes = (try txn.get(self.meta_dbi, META_RAFT_APPLY_KEY)) orelse return 0;
        if (bytes.len != 8) return 0;
        return std.mem.readInt(u64, bytes[0..8], .little);
    }

    /// Drop every per-store overlay's contents in-memory. Does NOT
    /// touch LMDB. Used on leadership loss: the integration calls
    /// this, then replays raft entries from `durableRaftIdx() + 1`
    /// onward, which repopulates the overlays with the confirmed
    /// (raft-committed) prefix. Speculative-but-uncommitted writes
    /// vanish along with the overlay — those proposals never made it
    /// into the raft log, so they never had a durability guarantee.
    pub fn discardOverlays(self: *Manifest) void {
        self.overlays_lock.lock();
        defer self.overlays_lock.unlock();
        var it = self.overlays.iterator();
        while (it.next()) |entry| {
            const ov = entry.value_ptr.*;
            ov.lock.lock();
            defer ov.lock.unlock();
            ov.clear();
        }
    }

    // ── poison ──────────────────────────────────────────────────────

    pub fn isPoisoned(self: *const Manifest) bool {
        return self.poisoned.load(.monotonic);
    }

    fn poison(self: *Manifest) void {
        self.poisoned.store(true, .monotonic);
    }

    pub fn _testPoison(self: *Manifest) void {
        self.poison();
    }

    inline fn checkAlive(self: *const Manifest) !void {
        if (self.isPoisoned()) return error.ManifestPoisoned;
    }

    // ── per-store locks ─────────────────────────────────────────────

    pub fn storeLock(self: *Manifest, id: u64) !*std.Thread.Mutex {
        self.store_locks_lock.lock();
        defer self.store_locks_lock.unlock();
        const gop = try self.store_locks.getOrPut(self.allocator, id);
        if (!gop.found_existing) {
            const m = try self.allocator.create(std.Thread.Mutex);
            m.* = .{};
            gop.value_ptr.* = m;
        }
        return gop.value_ptr.*;
    }

    // ── stores: create / drop / has / list ──────────────────────────

    pub fn createStore(self: *Manifest, id: u64) !void {
        try self.checkAlive();
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (self.hasStoreLocked(id)) return error.StoreAlreadyExists;
        // We may be in one of:
        //   (false, false, false) → just add pending_create
        //   (true, false, true)   → durable + pending drop. Add pending_create
        //                           too. At durabilize, drop empties the DBI
        //                           and create re-registers it: effectively a
        //                           wipe-and-rebind which is what callers
        //                           expect from drop-then-create.
        try self.pending_creates.put(self.allocator, id, {});
    }

    pub fn dropStore(self: *Manifest, id: u64) !bool {
        try self.checkAlive();
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (!self.hasStoreLocked(id)) return false;
        // Cancel a pending_create that hasn't been durabilized yet.
        if (self.pending_creates.remove(id)) {
            self.clearOverlayForLocked(id);
            // If the store is also durable (the drop-and-recreate
            // case), we still need to queue the drop so durabilize
            // empties LMDB.
            if (self.dbis.contains(id)) {
                try self.pending_drops.put(self.allocator, id, {});
            }
            return true;
        }
        // Durable store: queue the drop.
        try self.pending_drops.put(self.allocator, id, {});
        self.clearOverlayForLocked(id);
        return true;
    }

    /// hasStore variant that assumes the caller already holds dbis_lock.
    fn hasStoreLocked(self: *const Manifest, id: u64) bool {
        if (self.pending_creates.contains(id)) return true;
        if (self.pending_drops.contains(id)) return false;
        return self.dbis.contains(id);
    }

    /// Caller must hold dbis_lock (so the overlay pointer doesn't get
    /// freed mid-clear).
    fn clearOverlayForLocked(self: *Manifest, id: u64) void {
        self.overlays_lock.lock();
        defer self.overlays_lock.unlock();
        if (self.overlays.get(id)) |ov| {
            ov.lock.lock();
            defer ov.lock.unlock();
            ov.clear();
        }
    }

    pub fn hasStore(self: *Manifest, id: u64) !bool {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        return self.hasStoreLocked(id);
    }

    /// Compatibility: returns Some(id) if the store exists, None
    /// otherwise. The original CoW model returned the tree's root
    /// page number — LMDB hides that; we use the store_id itself as
    /// an opaque "exists" token.
    pub fn storeRoot(self: *Manifest, id: u64) !?u64 {
        if (try self.hasStore(id)) return id;
        return null;
    }

    pub fn listStores(self: *Manifest, allocator: std.mem.Allocator) ![]u64 {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        // Union of durable (minus pending drops) and pending creates.
        var it = self.dbis.keyIterator();
        while (it.next()) |id_ptr| {
            if (self.pending_drops.contains(id_ptr.*)) continue;
            try list.append(allocator, id_ptr.*);
        }
        var pc_it = self.pending_creates.keyIterator();
        while (pc_it.next()) |id_ptr| {
            try list.append(allocator, id_ptr.*);
        }
        std.mem.sort(u64, list.items, {}, std.sort.asc(u64));
        return try list.toOwnedSlice(allocator);
    }

    /// Returns the durable DBI handle if the store has been
    /// durabilized; null if pending-create (DBI doesn't exist yet) or
    /// dropped (caller should treat as not-exists if needed). Callers
    /// that need to know "does the store logically exist" should use
    /// `hasStore`.
    fn lookupDbi(self: *Manifest, id: u64) ?lmdb.Dbi {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (self.pending_drops.contains(id)) return null;
        return self.dbis.get(id);
    }

    // ── overlay management ──────────────────────────────────────────

    pub fn overlayFor(self: *Manifest, id: u64) !*Overlay {
        self.overlays_lock.lock();
        defer self.overlays_lock.unlock();
        const gop = try self.overlays.getOrPut(self.allocator, id);
        if (!gop.found_existing) {
            const ov = try self.allocator.create(Overlay);
            ov.* = Overlay.init(self.allocator);
            gop.value_ptr.* = ov;
        }
        return gop.value_ptr.*;
    }

    pub fn maybeOverlayFor(self: *Manifest, id: u64) ?*Overlay {
        self.overlays_lock.lock();
        defer self.overlays_lock.unlock();
        return self.overlays.get(id);
    }

    // ── durabilize ──────────────────────────────────────────────────

    /// Commit every pending overlay + create + drop to LMDB in one
    /// atomic write transaction, stamping the `raft_idx` watermark
    /// into the `_meta` sub-DBI as part of the same commit.
    ///
    /// If `raft_idx == 0`, the watermark is NOT written (the previous
    /// durable value is preserved). This lets test code and
    /// integrations that don't track a raft idx call `commit()` (an
    /// alias) without disturbing the watermark.
    ///
    /// If `raft_idx > 0`, the value is written verbatim — kvexp does
    /// not enforce monotonicity. Callers should pass a value that
    /// monotonically advances; passing a smaller value is technically
    /// allowed (think: recovery scenarios) but typically a bug.
    pub fn durabilize(self: *Manifest, raft_idx: u64) !void {
        try self.checkAlive();
        self.durabilize_lock.lock();
        defer self.durabilize_lock.unlock();
        errdefer self.poison();

        // Snapshot the pending creates / drops list. Under dbis_lock
        // we take a stable copy; subsequent createStore/dropStore
        // calls may add to the live sets — those will be drained at
        // the next durabilize.
        var creates_to_apply: std.ArrayListUnmanaged(u64) = .empty;
        defer creates_to_apply.deinit(self.allocator);
        var drops_to_apply: std.ArrayListUnmanaged(struct { id: u64, dbi: lmdb.Dbi }) = .empty;
        defer drops_to_apply.deinit(self.allocator);
        {
            self.dbis_lock.lock();
            defer self.dbis_lock.unlock();
            var pc_it = self.pending_creates.keyIterator();
            while (pc_it.next()) |id_ptr| try creates_to_apply.append(self.allocator, id_ptr.*);
            var pd_it = self.pending_drops.keyIterator();
            while (pd_it.next()) |id_ptr| {
                const dbi = self.dbis.get(id_ptr.*) orelse continue;
                try drops_to_apply.append(self.allocator, .{ .id = id_ptr.*, .dbi = dbi });
            }
        }

        // Snapshot the (id, overlay) pairs to drain.
        var pairs: std.ArrayListUnmanaged(struct { id: u64, ov: *Overlay }) = .empty;
        defer pairs.deinit(self.allocator);
        {
            self.overlays_lock.lock();
            defer self.overlays_lock.unlock();
            var it = self.overlays.iterator();
            while (it.next()) |entry| {
                try pairs.append(self.allocator, .{
                    .id = entry.key_ptr.*,
                    .ov = entry.value_ptr.*,
                });
            }
        }

        // SWAP each non-empty overlay's entries into a local map.
        // Workers writing during durabilize land in the fresh empty
        // map; we own the swapped entries free-and-clear. DBI resolution
        // is deferred to apply-time because pending-create stores
        // don't have a durable DBI yet — it's created earlier in this
        // same write txn.
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

        for (pairs.items) |p| {
            p.ov.lock.lock();
            const empty = p.ov.isEmptyLocked();
            if (empty) {
                p.ov.lock.unlock();
                continue;
            }
            const taken = p.ov.entries;
            p.ov.entries = .empty;
            p.ov.lock.unlock();
            try swapped_list.append(self.allocator, .{
                .id = p.id,
                .entries = taken,
            });
        }

        var txn = try lmdb.Txn.beginWrite(&self.env);
        errdefer txn.abort();

        // Apply pending drops first (empties their DBIs).
        for (drops_to_apply.items) |d| {
            try txn.dropDbi(d.dbi, false);
            var id_key_buf: [8]u8 = undefined;
            _ = try txn.del(self.stores_dbi, encodeStoreIdKey(d.id, &id_key_buf));
        }

        // Apply pending creates. Collect (id, new_dbi) for later
        // commit into `self.dbis`.
        var new_dbis: std.ArrayListUnmanaged(NewDbi) = .empty;
        defer new_dbis.deinit(self.allocator);
        for (creates_to_apply.items) |id| {
            var name_buf: [19]u8 = undefined;
            const name = storeDbiName(id, &name_buf);
            const dbi = try txn.openDbi(name, true);
            var id_key_buf: [8]u8 = undefined;
            try txn.put(self.stores_dbi, encodeStoreIdKey(id, &id_key_buf), &.{});
            try new_dbis.append(self.allocator, .{ .id = id, .dbi = dbi });
        }

        // Resolve each overlay's DBI for this txn. Freshly-created
        // stores' DBIs are in `new_dbis`; everything else is in
        // `self.dbis`. Drops have already emptied the underlying
        // tree, so any overlay entries for a dropped (and not
        // re-created) store would have no live DBI — skip.
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

        // Only write the watermark when the caller supplied a non-zero
        // value. `raft_idx == 0` means "don't disturb whatever's
        // already durable" — used by tests and by `commit()`.
        if (raft_idx != 0) {
            var idx_buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &idx_buf, raft_idx, .little);
            try txn.put(self.meta_dbi, META_RAFT_APPLY_KEY, &idx_buf);
        }

        try txn.commit();

        // Commit succeeded: bookkeeping. Move new DBIs into self.dbis,
        // remove dropped, clear pending sets (only the entries we
        // applied — concurrent callers may have added more).
        {
            self.dbis_lock.lock();
            defer self.dbis_lock.unlock();
            for (drops_to_apply.items) |d| {
                _ = self.dbis.remove(d.id);
                _ = self.pending_drops.remove(d.id);
            }
            for (new_dbis.items) |n| {
                try self.dbis.put(self.allocator, n.id, n.dbi);
                _ = self.pending_creates.remove(n.id);
            }
        }
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

    /// Convenience: durabilize without disturbing the raft watermark.
    /// Used by tests and by callers that don't track a raft idx.
    /// Equivalent to `durabilize(0)`.
    pub fn commit(self: *Manifest) !void {
        return self.durabilize(0);
    }

    // ── snapshots ───────────────────────────────────────────────────

    pub fn openSnapshot(self: *Manifest) !Snapshot {
        // Serialize against durabilize: openSnapshot needs an atomic
        // moment in which overlays + LMDB state agree.
        self.durabilize_lock.lock();
        defer self.durabilize_lock.unlock();

        // LMDB read txn — provides the immutable point-in-time view
        // of every store.
        var read_txn = try lmdb.Txn.beginRead(&self.env);
        errdefer read_txn.abort();

        // Capture overlays.
        var ov_capture: std.AutoHashMapUnmanaged(u64, []StorePrefixCursor.Entry) = .empty;
        errdefer freeOverlayCapture(self.allocator, &ov_capture);

        // List overlays to copy under the overlays_lock; release before
        // capturing each (we still hold the per-overlay lock during
        // its copy).
        const Pair = struct { id: u64, ov: *Overlay };
        var pairs: std.ArrayListUnmanaged(Pair) = .empty;
        defer pairs.deinit(self.allocator);
        {
            self.overlays_lock.lock();
            defer self.overlays_lock.unlock();
            var it = self.overlays.iterator();
            while (it.next()) |entry| {
                try pairs.append(self.allocator, .{
                    .id = entry.key_ptr.*,
                    .ov = entry.value_ptr.*,
                });
            }
        }
        for (pairs.items) |p| {
            var entries: std.ArrayListUnmanaged(StorePrefixCursor.Entry) = .empty;
            errdefer freeOverlayEntries(self.allocator, entries.items, entries.items.len);
            errdefer entries.deinit(self.allocator);
            {
                p.ov.lock.lock();
                defer p.ov.lock.unlock();
                if (p.ov.isEmptyLocked()) continue;
                var oe_it = p.ov.entries.iterator();
                while (oe_it.next()) |oe| {
                    const k = try self.allocator.dupe(u8, oe.key_ptr.*);
                    errdefer self.allocator.free(k);
                    const v: ?[]u8 = switch (oe.value_ptr.*) {
                        .value => |val| try self.allocator.dupe(u8, val),
                        .tombstone => null,
                    };
                    try entries.append(self.allocator, .{ .key = k, .value = v });
                }
            }
            std.mem.sort(StorePrefixCursor.Entry, entries.items, {}, StorePrefixCursor.entryLessThan);
            const slice = try entries.toOwnedSlice(self.allocator);
            try ov_capture.put(self.allocator, p.id, slice);
        }

        return .{
            .manifest = self,
            .read_txn = read_txn,
            .overlays = ov_capture,
        };
    }

    // ── verify ──────────────────────────────────────────────────────

    pub const VerifyReport = struct {
        store_count: u64,
        durable_raft_idx: u64,
    };

    pub fn verify(self: *Manifest, allocator: std.mem.Allocator) !VerifyReport {
        _ = allocator;
        const idx = try self.durableRaftIdx();
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        return .{
            .store_count = self.dbis.count(),
            .durable_raft_idx = idx,
        };
    }
};

fn freeOverlayEntries(
    allocator: std.mem.Allocator,
    entries: []StorePrefixCursor.Entry,
    upto: usize,
) void {
    for (entries[0..upto]) |e| {
        allocator.free(e.key);
        if (e.value) |v| allocator.free(v);
    }
}

fn freeOverlayCapture(
    allocator: std.mem.Allocator,
    map: *std.AutoHashMapUnmanaged(u64, []StorePrefixCursor.Entry),
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

// ── Store ───────────────────────────────────────────────────────────

pub const Store = struct {
    manifest: *Manifest,
    id: u64,
    /// Null iff the store is pending-create (no durable DBI yet).
    /// Writes still work via the overlay; reads that miss the overlay
    /// return null (the store has no durable data yet).
    dbi: ?lmdb.Dbi,

    pub fn open(manifest: *Manifest, id: u64) !Store {
        if (!try manifest.hasStore(id)) return error.StoreNotFound;
        return .{ .manifest = manifest, .id = id, .dbi = manifest.lookupDbi(id) };
    }

    pub fn deinit(self: *Store) void {
        _ = self;
    }

    pub fn get(self: *Store, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        // Overlay first.
        if (self.manifest.maybeOverlayFor(self.id)) |ov| {
            ov.lock.lock();
            defer ov.lock.unlock();
            if (ov.getLocked(key)) |entry| switch (entry) {
                .value => |v| return try allocator.dupe(u8, v),
                .tombstone => return null,
            };
        }
        // No DBI yet (pending create) → no durable data to fall through to.
        const dbi = self.dbi orelse return null;
        var txn = try lmdb.Txn.beginRead(&self.manifest.env);
        defer txn.abort();
        const got = (try txn.get(dbi, key)) orelse return null;
        return try allocator.dupe(u8, got);
    }

    pub fn put(self: *Store, key: []const u8, value: []const u8) !void {
        try self.manifest.checkAlive();
        const sl = try self.manifest.storeLock(self.id);
        sl.lock();
        defer sl.unlock();
        const ov = try self.manifest.overlayFor(self.id);
        ov.lock.lock();
        defer ov.lock.unlock();
        try ov.putLocked(key, value);
    }

    pub fn delete(self: *Store, key: []const u8) !bool {
        try self.manifest.checkAlive();
        const sl = try self.manifest.storeLock(self.id);
        sl.lock();
        defer sl.unlock();

        const ov = try self.manifest.overlayFor(self.id);
        {
            ov.lock.lock();
            defer ov.lock.unlock();
            if (ov.getLocked(key)) |entry| switch (entry) {
                .value => {
                    try ov.tombstoneLocked(key);
                    return true;
                },
                .tombstone => return false,
            };
        }

        // Overlay miss — check LMDB to determine return bool. If no
        // durable DBI exists (pending create), nothing to delete.
        const dbi = self.dbi orelse return false;
        const existed_in_tree = blk: {
            var txn = try lmdb.Txn.beginRead(&self.manifest.env);
            defer txn.abort();
            const got = try txn.get(dbi, key);
            break :blk got != null;
        };
        if (!existed_in_tree) return false;

        ov.lock.lock();
        defer ov.lock.unlock();
        try ov.tombstoneLocked(key);
        return true;
    }

    pub fn scanPrefix(self: *Store, prefix: []const u8) !StorePrefixCursor {
        // Snapshot the overlay's prefix-matching entries.
        var entries: std.ArrayListUnmanaged(StorePrefixCursor.Entry) = .empty;
        errdefer {
            for (entries.items) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            entries.deinit(self.manifest.allocator);
        }
        if (self.manifest.maybeOverlayFor(self.id)) |ov| {
            ov.lock.lock();
            defer ov.lock.unlock();
            var it = ov.entries.iterator();
            while (it.next()) |entry| {
                const k = entry.key_ptr.*;
                if (!std.mem.startsWith(u8, k, prefix)) continue;
                const key_copy = try self.manifest.allocator.dupe(u8, k);
                errdefer self.manifest.allocator.free(key_copy);
                const val_copy: ?[]u8 = switch (entry.value_ptr.*) {
                    .value => |v| try self.manifest.allocator.dupe(u8, v),
                    .tombstone => null,
                };
                try entries.append(self.manifest.allocator, .{ .key = key_copy, .value = val_copy });
            }
        }
        std.mem.sort(StorePrefixCursor.Entry, entries.items, {}, StorePrefixCursor.entryLessThan);
        const overlay_slice = try entries.toOwnedSlice(self.manifest.allocator);
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        // If the store has no durable DBI yet (pending create), skip
        // the LMDB cursor entirely — the merge becomes overlay-only.
        // We still need valid (empty) Txn/Cursor structs because the
        // cursor field types aren't optional; default-constructed
        // structs with null ptrs are no-ops on close/abort.
        var read_txn: lmdb.Txn = .{ .ptr = null };
        var lmdb_cur: lmdb.Cursor = .{ .ptr = null };
        if (self.dbi) |dbi| {
            read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
            errdefer read_txn.abort();
            lmdb_cur = try read_txn.openCursor(dbi);
        }
        errdefer if (lmdb_cur.ptr != null) lmdb_cur.close();
        errdefer if (read_txn.ptr != null) read_txn.abort();

        var cur: StorePrefixCursor = .{
            .allocator = self.manifest.allocator,
            .read_txn = read_txn,
            .lmdb_cur = lmdb_cur,
            .has_lmdb = self.dbi != null,
            .prefix = try self.manifest.allocator.dupe(u8, prefix),
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
};

// ── Merge cursor over LMDB cursor + captured overlay ────────────────

pub const StorePrefixCursor = struct {
    allocator: std.mem.Allocator,
    read_txn: lmdb.Txn,
    lmdb_cur: lmdb.Cursor,
    /// False for pending-create stores (no durable DBI). `primeTree`
    /// / `advanceTree` short-circuit and leave `have_tree` false.
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
        value: ?[]u8, // null = tombstone
    };

    pub fn entryLessThan(_: void, a: Entry, b: Entry) bool {
        return std.mem.lessThan(u8, a.key, b.key);
    }

    fn primeTree(self: *StorePrefixCursor) !void {
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

    fn advanceTree(self: *StorePrefixCursor) !void {
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

    pub fn next(self: *StorePrefixCursor) !bool {
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
            } else { // overlay only
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

    pub fn key(self: *const StorePrefixCursor) []const u8 {
        return self.out_key;
    }

    pub fn value(self: *const StorePrefixCursor) []const u8 {
        return self.out_val;
    }

    pub fn deinit(self: *StorePrefixCursor) void {
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

// ── Snapshot ────────────────────────────────────────────────────────

pub const Snapshot = struct {
    manifest: *Manifest,
    read_txn: lmdb.Txn,
    overlays: std.AutoHashMapUnmanaged(u64, []StorePrefixCursor.Entry),

    pub fn close(self: *Snapshot) void {
        var it = self.overlays.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.*) |kv| {
                self.manifest.allocator.free(kv.key);
                if (kv.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(entry.value_ptr.*);
        }
        self.overlays.deinit(self.manifest.allocator);
        self.read_txn.abort();
        self.* = undefined;
    }

    pub fn storeRoot(self: *Snapshot, store_id: u64) !?u64 {
        if (self.manifest.lookupDbi(store_id) == null) return null;
        return store_id;
    }

    pub fn get(
        self: *Snapshot,
        allocator: std.mem.Allocator,
        store_id: u64,
        key: []const u8,
    ) !?[]u8 {
        if (self.overlays.get(store_id)) |entries| {
            if (binarySearchEntry(entries, key)) |idx| {
                const e = entries[idx];
                if (e.value) |v| return try allocator.dupe(u8, v);
                return null;
            }
        }
        const dbi = self.manifest.lookupDbi(store_id) orelse return error.StoreNotFound;
        const got = (try self.read_txn.get(dbi, key)) orelse return null;
        return try allocator.dupe(u8, got);
    }

    pub fn scanPrefix(self: *Snapshot, store_id: u64, prefix: []const u8) !SnapshotPrefixCursor {
        const dbi = self.manifest.lookupDbi(store_id) orelse return error.StoreNotFound;

        // Filter captured overlay for prefix matches.
        var entries: std.ArrayListUnmanaged(StorePrefixCursor.Entry) = .empty;
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
                const k = try self.manifest.allocator.dupe(u8, e.key);
                errdefer self.manifest.allocator.free(k);
                const v: ?[]u8 = if (e.value) |val| try self.manifest.allocator.dupe(u8, val) else null;
                try entries.append(self.manifest.allocator, .{ .key = k, .value = v });
            }
        }
        const overlay_slice = try entries.toOwnedSlice(self.manifest.allocator);
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        var lmdb_cur = try self.read_txn.openCursor(dbi);
        errdefer lmdb_cur.close();

        var cur: SnapshotPrefixCursor = .{
            .allocator = self.manifest.allocator,
            .lmdb_cur = lmdb_cur,
            .prefix = try self.manifest.allocator.dupe(u8, prefix),
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

/// Same shape as StorePrefixCursor but borrowing the snapshot's read
/// txn (snapshot owns it). Separate type so deinit doesn't double-
/// abort the txn.
pub const SnapshotPrefixCursor = struct {
    allocator: std.mem.Allocator,
    lmdb_cur: lmdb.Cursor,
    prefix: []u8,
    overlay: []StorePrefixCursor.Entry,
    ov_idx: usize,
    have_tree: bool,
    tree_key: []const u8,
    tree_val: []const u8,
    out_key: []const u8,
    out_val: []const u8,

    fn primeTree(self: *SnapshotPrefixCursor) !void {
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
        self.lmdb_cur.close();
        // Don't abort the txn — owned by the Snapshot.
        for (self.overlay) |e| {
            self.allocator.free(e.key);
            if (e.value) |v| self.allocator.free(v);
        }
        self.allocator.free(self.overlay);
        self.allocator.free(self.prefix);
        self.* = undefined;
    }
};

fn binarySearchEntry(entries: []const StorePrefixCursor.Entry, key: []const u8) ?usize {
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

// ── State-transfer dump / restore ──────────────────────────────────

pub const SNAPSHOT_MAGIC: u32 = 0x4B565853; // 'KVXS' little-endian
pub const SNAPSHOT_VERSION: u8 = 1;
pub const SNAP_TAG_KV: u8 = 1;
pub const SNAP_TAG_END: u8 = 2;

pub fn dumpSnapshot(snap: *Snapshot, writer: anytype) !void {
    try writer.writeInt(u32, SNAPSHOT_MAGIC, .little);
    try writer.writeByte(SNAPSHOT_VERSION);
    // Reserved field (was apply_seq pre-LMDB). Kept as a 0 in the
    // wire format so future tooling can re-purpose it without bumping
    // the version. loadSnapshot discards it.
    try writer.writeInt(u64, 0, .little);
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
                if (!(try manifest.hasStore(id))) {
                    try manifest.createStore(id);
                }
                var s = try Store.open(manifest, id);
                defer s.deinit();
                try s.put(key_buf[0..klen], val_buf[0..vlen]);
            },
            else => return error.InvalidSnapshotFormat,
        }
    }
    // Caller passes `last_applied` to `durabilize` to commit it
    // alongside the loaded data in a single LMDB txn.
    return last_applied;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

const Harness = struct {
    tmp: std.testing.TmpDir,
    path_buf: [std.fs.max_path_bytes:0]u8,
    path: [:0]const u8,
    manifest: *Manifest,

    fn init() !Harness {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var self: Harness = .{
            .tmp = tmp,
            .path_buf = undefined,
            .path = undefined,
            .manifest = undefined,
        };
        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &dir_buf);
        const p = try std.fmt.bufPrintZ(&self.path_buf, "{s}/kvexp-test.mdb", .{dir_path});
        self.path = p;
        self.manifest = try testing.allocator.create(Manifest);
        errdefer testing.allocator.destroy(self.manifest);
        try self.manifest.init(testing.allocator, self.path, .{
            .max_map_size = 32 * 1024 * 1024,
            .max_stores = 256,
        });
        return self;
    }

    fn deinit(self: *Harness) void {
        self.manifest.deinit();
        testing.allocator.destroy(self.manifest);
        self.tmp.cleanup();
    }

    /// Close + reopen the manifest. Used for crash-recovery tests.
    fn cycle(self: *Harness) !void {
        self.manifest.deinit();
        try self.manifest.init(testing.allocator, self.path, .{
            .max_map_size = 32 * 1024 * 1024,
            .max_stores = 256,
        });
    }
};

test "Manifest: create + reopen survives" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(7);
    try h.manifest.createStore(42);
    try h.manifest.commit();
    try h.cycle();
    try testing.expect(try h.manifest.hasStore(7));
    try testing.expect(try h.manifest.hasStore(42));
    try testing.expect(!try h.manifest.hasStore(99));
}

test "Manifest: createStore rejects duplicate" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try testing.expectError(error.StoreAlreadyExists, h.manifest.createStore(1));
}

test "Manifest: dropStore removes the entry; recreate works" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(10);
    try h.manifest.commit();
    try testing.expect(try h.manifest.hasStore(10));
    try testing.expect(try h.manifest.dropStore(10));
    try testing.expect(!try h.manifest.dropStore(10));
    try h.manifest.commit();
    try h.cycle();
    try testing.expect(!try h.manifest.hasStore(10));
}

test "Store: put/get/delete in one store" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try s.put("k", "v");
    {
        const got = (try s.get(testing.allocator, "k")).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings("v", got);
    }
    try testing.expect(try s.delete("k"));
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Store: durabilize then reopen sees the writes" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("a", "1");
        try s.put("b", "2");
    }
    try h.manifest.commit();
    try h.cycle();
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    const a = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("1", a);
    const b = (try s.get(testing.allocator, "b")).?;
    defer testing.allocator.free(b);
    try testing.expectEqualStrings("2", b);
}

test "Store: writes lost on close before durabilize" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.commit();
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v");
    }
    // No durabilize.
    try h.cycle();
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Store.scanPrefix: merges overlay + lmdb in sorted order" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("a", "tree-a");
        try s.put("c", "tree-c");
    }
    try h.manifest.commit();
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try s.put("b", "ov-b");
    try s.put("c", "ov-c"); // override

    var cur = try s.scanPrefix("");
    defer cur.deinit();
    var collected: [3]struct { k: []const u8, v: []const u8 } = undefined;
    var n: usize = 0;
    while (try cur.next()) : (n += 1) {
        collected[n] = .{ .k = cur.key(), .v = cur.value() };
    }
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqualStrings("a", collected[0].k);
    try testing.expectEqualStrings("tree-a", collected[0].v);
    try testing.expectEqualStrings("b", collected[1].k);
    try testing.expectEqualStrings("ov-b", collected[1].v);
    try testing.expectEqualStrings("c", collected[2].k);
    try testing.expectEqualStrings("ov-c", collected[2].v);
}

test "Store.scanPrefix: tombstone shadows tree entry" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("a", "1");
        try s.put("b", "2");
        try s.put("c", "3");
    }
    try h.manifest.commit();
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    _ = try s.delete("b");

    var cur = try s.scanPrefix("");
    defer cur.deinit();
    var keys: [2]u8 = undefined;
    var n: usize = 0;
    while (try cur.next()) : (n += 1) {
        keys[n] = cur.key()[0];
    }
    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqual(@as(u8, 'a'), keys[0]);
    try testing.expectEqual(@as(u8, 'c'), keys[1]);
}

test "Manifest: durableRaftIdx round-trips via durabilize(idx)" {
    var h = try Harness.init();
    defer h.deinit();
    try testing.expectEqual(@as(u64, 0), try h.manifest.durableRaftIdx());

    try h.manifest.durabilize(42);
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());

    try h.cycle();
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());

    // `commit()` (alias for durabilize(0)) preserves the watermark.
    try h.manifest.commit();
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());

    // Advancing.
    try h.manifest.durabilize(100);
    try testing.expectEqual(@as(u64, 100), try h.manifest.durableRaftIdx());
}

test "discardOverlays: drops in-flight writes; durable state intact" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("durable", "yes");
    }
    try h.manifest.commit(); // "durable" → "yes" lives in LMDB.

    // Stage speculative writes that won't be raft-committed.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("speculative", "maybe");
        try s.put("durable", "overridden");
    }

    h.manifest.discardOverlays();

    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    // "speculative" vanished.
    try testing.expect((try s.get(testing.allocator, "speculative")) == null);
    // "durable" reverted to its LMDB value.
    const got = (try s.get(testing.allocator, "durable")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("yes", got);
}

test "Manifest: poison blocks writes; reopen clears it" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.commit();
    h.manifest._testPoison();
    try testing.expect(h.manifest.isPoisoned());
    try testing.expectError(error.ManifestPoisoned, h.manifest.durabilize(0));
    try testing.expectError(error.ManifestPoisoned, h.manifest.createStore(2));
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try testing.expectError(error.ManifestPoisoned, s.put("k", "v"));

    try h.cycle();
    try testing.expect(!h.manifest.isPoisoned());
    try testing.expect(try h.manifest.hasStore(1));
}

test "Snapshot: captures uncommitted overlay puts" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("a", "tree-a");
    }
    try h.manifest.commit();
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("b", "ov-b");
    }
    var snap = try h.manifest.openSnapshot();
    defer snap.close();
    const ga = (try snap.get(testing.allocator, 1, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("tree-a", ga);
    const gb = (try snap.get(testing.allocator, 1, "b")).?;
    defer testing.allocator.free(gb);
    try testing.expectEqualStrings("ov-b", gb);
}

test "dumpSnapshot / loadSnapshot: full round-trip" {
    var src = try Harness.init();
    defer src.deinit();
    try src.manifest.createStore(10);
    try src.manifest.createStore(20);
    {
        var s = try Store.open(src.manifest, 10);
        defer s.deinit();
        try s.put("a", "1");
        try s.put("b", "2");
    }
    {
        var s = try Store.open(src.manifest, 20);
        defer s.deinit();
        try s.put("z", "9");
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
    const reader = reader_state.reader();
    const last = try loadSnapshot(dst.manifest, reader);
    try testing.expectEqual(@as(u64, 99), last);
    try dst.manifest.durabilize(last);

    try testing.expectEqual(@as(u64, 99), try dst.manifest.durableRaftIdx());
    try testing.expect(try dst.manifest.hasStore(10));
    try testing.expect(try dst.manifest.hasStore(20));
    var s10 = try Store.open(dst.manifest, 10);
    defer s10.deinit();
    const ga = (try s10.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
}

test "Manifest: listStores returns ids in ascending order" {
    var h = try Harness.init();
    defer h.deinit();
    const ids = [_]u64{ 5, 1, 9, 3, 100 };
    for (ids) |id| try h.manifest.createStore(id);
    const got = try h.manifest.listStores(testing.allocator);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(u64, &.{ 1, 3, 5, 9, 100 }, got);
}

test "verify: reports store count + raft watermark" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.createStore(2);
    try h.manifest.durabilize(7);
    const report = try h.manifest.verify(testing.allocator);
    try testing.expectEqual(@as(u64, 2), report.store_count);
    try testing.expectEqual(@as(u64, 7), report.durable_raft_idx);
}

// -----------------------------------------------------------------------------
// Property test: random workload + simulated crash (close+reopen without
// intervening durabilize). After recovery, the durable prefix must exactly
// match the model snapshot taken at the last successful durabilize.
// -----------------------------------------------------------------------------

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
        var s = try Store.open(manifest, sid);
        defer s.deinit();
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

    const op_count: u32 = rng.intRangeAtMost(u32, 40, 120);
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
            var s = try Store.open(h.manifest, @intCast(sid_usize));
            defer s.deinit();
            var kb: [3]u8 = undefined;
            var vb: [1]u8 = .{v};
            try s.put(keyBuf(&kb, kid), vb[0..1]);
            model.key_present[sid_usize][kid] = true;
            model.values[sid_usize][kid] = v;
        } else if (choice < 70) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const kid = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
            var s = try Store.open(h.manifest, @intCast(sid_usize));
            defer s.deinit();
            var kb: [3]u8 = undefined;
            const existed = try s.delete(keyBuf(&kb, kid));
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
            try h.manifest.commit();
            durable_model = model;
        }
    }
}

test "property: random workload + crash recovers durable prefix" {
    const runs: u64 = 50;
    var seed: u64 = 1;
    while (seed <= runs) : (seed += 1) {
        propRunOne(seed) catch |err| {
            std.debug.print("\n*** property test failed at seed {} : {} ***\n", .{ seed, err });
            return err;
        };
    }
}
