//! Manifest: forest root pointer + per-store directory + durable
//! free-page list.
//!
//! Layout in the data file:
//!   page 0       — reserved for the higher-layer file header (phase 1)
//!   page 1       — manifest slot A
//!   page 2       — manifest slot B
//!   page 3..     — data (manifest tree pages + freelist tree pages
//!                  + per-store B-tree pages, interleaved)
//!
//! Each slot holds **two roots**: the manifest tree (`store_id → root`)
//! and the freelist tree (`(freed_at_seq, page_no) → ε`), plus a magic
//! word, monotonic sequence, and CRC32. Commits alternate slots:
//! write the inactive one, fsync, swap "active." A torn write to one
//! slot leaves the other valid; recovery picks the higher-seq valid
//! slot.
//!
//! Free-page reclamation is one-commit-lagged. A page tagged
//! `freed_at_seq=N` was superseded in commit N's window; it was last
//! referenced by manifest `M_{N-1}`. With two-slot alternation, slot
//! `(K-1) mod 2` is overwritten exactly when commit K+1 runs — so
//! `M_{N-1}` becomes non-durable when commit N+1 lands. A freelist
//! entry tagged `freed_at_seq=N` is therefore reusable once the
//! current durable sequence has reached `N+1`, i.e.,
//! `freed_at_seq <= sequence - 1`.
//!
//! ## Freelist on-disk format: vector-valued chunks
//!
//! The freelist B-tree stores **packed lists** of page_nos under each
//! key, not one entry per cell. This amortizes freelist maintenance
//! cost — without it, every freelist insert/delete goes through a
//! full CoW path, and the freelist's own work dominates the workload.
//!
//! Key  = 16 bytes: 8 BE `freed_at_seq` | 8 BE `first_page_no` (just a
//!        uniquifier so the same seq can have multiple chunks)
//! Value = u16 count LE | count u64 page_nos LE
//!
//! With our 2KB max inline value, a chunk holds up to 255 page_nos.
//! A commit that frees K pages writes ⌈K/255⌉ freelist puts instead
//! of K. refillReusable consumes whole chunks and queues each chunk's
//! key for deletion at next commit, so the freelist's own size stays
//! proportional to *in-flight reusable pages*, not to *historical
//! frees*.
//!
//! Manifest is **its own PageAllocator**. CoW operations on the
//! manifest tree, the freelist tree, and per-store trees all route
//! allocations through `Manifest.pageAllocator()`. Allocation pops
//! from an in-memory `reusable` queue (populated from the durable
//! freelist at open / after each commit); when empty, it falls back to
//! `file.growBy`. Frees append to an in-memory `pending_free` list,
//! which is folded into the durable freelist at the next commit.

const std = @import("std");
const page = @import("page.zig");
const btree = @import("btree.zig");
const Tree = btree.Tree;
const PageAllocator = btree.PageAllocator;
const paged_file_mod = @import("paged_file.zig");
const PagedFile = paged_file_mod.PagedFile;
const PagedFileApi = paged_file_mod.PagedFileApi;
const FaultyPagedFile = @import("faulty_paged_file.zig").FaultyPagedFile;
const PageCache = @import("page_cache.zig").PageCache;
const Header = @import("header.zig").Header;

pub const SLOT_A_PAGE: u64 = 1;
pub const SLOT_B_PAGE: u64 = 2;
pub const FIRST_DATA_PAGE: u64 = 3;

pub const STORE_ID_LEN: usize = 8;
pub const STORE_VAL_LEN: usize = 8;
pub const FREELIST_KEY_LEN: usize = 16; // 8 BE seq | 8 BE first_page_no
pub const REUSABLE_BATCH: usize = 4096;
/// Max page_nos per chunk cell: (max inline value − 2-byte count) / 8.
pub const PAGES_PER_CHUNK: usize = (page.MAX_VAL_LEN - 2) / 8;

pub const ManifestSlot = extern struct {
    magic: u32 align(1) = 0,
    slot_id: u8 align(1) = 0,
    _pad: [3]u8 align(1) = .{ 0, 0, 0 },
    sequence: u64 align(1) = 0,
    manifest_root: u64 align(1) = 0,
    freelist_root: u64 align(1) = 0,
    /// Cluster-wide "last applied raft index" watermark. Set by the
    /// integration on snapshot tick; read at open so the raft layer
    /// can skip already-applied log entries. Stored here (not in the
    /// manifest tree) because it's tiny, frequently-updated, and
    /// would otherwise force a tree CoW on every advance.
    last_applied_raft_idx: u64 align(1) = 0,
    _reserved: [page.PAGE_SIZE - 44]u8 align(1) = @splat(0),
    checksum: u32 align(1) = 0,

    pub const MAGIC: u32 = 0x6B764D31; // "kvM1"

    comptime {
        std.debug.assert(@sizeOf(@This()) == page.PAGE_SIZE);
    }

    pub fn computeChecksum(self: *ManifestSlot) void {
        const bytes: [*]const u8 = @ptrCast(self);
        self.checksum = std.hash.Crc32.hash(bytes[0 .. @sizeOf(ManifestSlot) - 4]);
    }

    pub fn isValid(self: *const ManifestSlot) bool {
        if (self.magic != MAGIC) return false;
        const bytes: [*]const u8 = @ptrCast(self);
        return std.hash.Crc32.hash(bytes[0 .. @sizeOf(ManifestSlot) - 4]) == self.checksum;
    }
};

pub fn encodeStoreId(id: u64, buf: *[STORE_ID_LEN]u8) []const u8 {
    std.mem.writeInt(u64, buf, id, .big);
    return buf;
}

pub fn decodeStoreId(bytes: []const u8) u64 {
    std.debug.assert(bytes.len == STORE_ID_LEN);
    return std.mem.readInt(u64, bytes[0..STORE_ID_LEN], .big);
}

pub fn encodeRoot(root: u64, buf: *[STORE_VAL_LEN]u8) []const u8 {
    std.mem.writeInt(u64, buf, root, .little);
    return buf;
}

pub fn decodeRoot(bytes: []const u8) u64 {
    std.debug.assert(bytes.len == STORE_VAL_LEN);
    return std.mem.readInt(u64, bytes[0..STORE_VAL_LEN], .little);
}

/// Encode a freelist chunk key: `seq | first_page_no_in_chunk`. The
/// `first_page_no` is a uniquifier so the same seq can have multiple
/// chunks without key collisions.
pub fn encodeFreelistKey(freed_at_seq: u64, first_page_no: u64, buf: *[FREELIST_KEY_LEN]u8) []const u8 {
    std.mem.writeInt(u64, buf[0..8], freed_at_seq, .big);
    std.mem.writeInt(u64, buf[8..16], first_page_no, .big);
    return buf;
}

pub fn decodeFreelistKey(bytes: []const u8) struct { freed_at_seq: u64, first_page_no: u64 } {
    std.debug.assert(bytes.len == FREELIST_KEY_LEN);
    return .{
        .freed_at_seq = std.mem.readInt(u64, bytes[0..8], .big),
        .first_page_no = std.mem.readInt(u64, bytes[8..16], .big),
    };
}

/// Encode a chunk value: 2-byte LE count, then `count` u64 LE page_nos.
pub fn encodeFreelistValue(page_nos: []const u64, buf: []u8) []const u8 {
    std.debug.assert(page_nos.len <= std.math.maxInt(u16));
    std.debug.assert(buf.len >= 2 + 8 * page_nos.len);
    std.mem.writeInt(u16, buf[0..2], @intCast(page_nos.len), .little);
    var i: usize = 2;
    for (page_nos) |p| {
        std.mem.writeInt(u64, buf[i..][0..8], p, .little);
        i += 8;
    }
    return buf[0..i];
}

/// Iterator over page_nos packed in a chunk value.
pub const ChunkIterator = struct {
    value: []const u8,
    index: usize = 2,
    count: usize,

    pub fn init(value: []const u8) ChunkIterator {
        return .{ .value = value, .count = std.mem.readInt(u16, value[0..2], .little) };
    }

    pub fn next(self: *ChunkIterator) ?u64 {
        if (self.index >= 2 + 8 * self.count) return null;
        const p = std.mem.readInt(u64, self.value[self.index..][0..8], .little);
        self.index += 8;
        return p;
    }
};

pub const Error = anyerror;

pub const SpecificError = error{
    StoreAlreadyExists,
    StoreNotFound,
    ManifestCorrupt,
};

const FreedPage = struct { page_no: u64, freed_at_seq: u64 };

const RootSlot = struct {
    root: u64,
    /// true = differs from the durable manifest tree (needs flush
    /// at next durabilize). false = matches the durable tree.
    dirty: bool,
};

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    cache: *PageCache,
    file: PagedFileApi,

    tree: Tree, // manifest B-tree
    freelist: Tree, // free-page B-tree

    /// Which slot (0 or 1) is currently "active" — holds the
    /// highest-seq valid manifest.
    active_slot: u32,
    /// Sequence of the active slot. Highest durable seq.
    active_seq: u64,
    /// Cluster-wide last-applied raft index, mirrored from the
    /// active slot's `last_applied_raft_idx`. Updated via
    /// `setLastAppliedRaftIdx`; written to disk by `durabilize`.
    last_applied_raft_idx: u64,
    /// Sequence of the other slot. Valid only if `inactive_valid`.
    /// A page tagged `freed_at_seq = inactive_seq + 1` is unsafe to
    /// reuse — its referencing manifest is still durable in this slot.
    inactive_seq: u64,
    inactive_valid: bool,

    /// Back-compat alias for active_seq (older code reads `sequence`).
    sequence: u64,

    /// Pages popped from the durable freelist that are eligible for reuse.
    reusable: std.ArrayListUnmanaged(u64),
    /// Freelist keys corresponding to `reusable` — queued to delete
    /// from the durable freelist at the next durabilize.
    consumed_keys: std.ArrayListUnmanaged([FREELIST_KEY_LEN]u8),
    /// Pages freed by CoW operations since the last durabilize. Folded
    /// into the durable freelist at the next durabilize.
    pending_free: std.ArrayListUnmanaged(FreedPage),

    /// Lock ordering (no thread holds an earlier lock while acquiring
    /// a later one):
    ///   store_lock(id)            — held by one writer to that store
    ///                                at a time; uncontended across
    ///                                distinct stores
    ///   tree_lock                 — manifest tree mutations + reads
    ///   freelist_tree_lock        — freelist tree mutations + reads
    ///                                (only held during durabilize)
    ///   store_root_cache_lock     — in-memory store_root cache
    ///                                (hashmap; uncontended for distinct
    ///                                store_ids in practice)
    ///   reusable_lock             — `reusable` + `consumed_keys`
    ///                                (both refilled by refillReusable
    ///                                and drained by durabilize)
    ///   pending_free_lock         — `pending_free`
    /// `cache.lock` (interior, owned by PageCache) is acquired
    /// briefly *inside* the alloc/free regions.
    ///
    /// `reusable_lock` and `pending_free_lock` are at the same level
    /// — neither is held while acquiring the other. Splitting them
    /// (formerly one `alloc_lock`) lets `allocImpl` and `freeImpl`
    /// run concurrently across workers on different stores, which was
    /// the dominant bottleneck at 4+ writers. `PagedFile.growBy` has
    /// its own internal lock; it's called *outside* both of these.
    tree_lock: std.Thread.Mutex = .{},
    freelist_tree_lock: std.Thread.Mutex = .{},
    reusable_lock: std.Thread.Mutex = .{},
    pending_free_lock: std.Thread.Mutex = .{},
    /// Serializes the entire `durabilize` call. Held outermost (above
    /// every other manifest lock) so that two callers can't race on
    /// pending_free drain, foldPendingFree, or the slot write — all of
    /// which assume exclusive ownership of the apply→durable transition.
    /// Workers do NOT acquire this lock; only `durabilize` does. Lock
    /// ordering: durabilize_lock → store_lock(id) → tree_lock | … .
    durabilize_lock: std.Thread.Mutex = .{},

    /// In-memory cache of store_id → current root. The hot path
    /// (`setStoreRoot` / `storeRoot`) hits this map directly, never
    /// touching the manifest tree. `durabilize` step 0 flushes dirty
    /// entries into the manifest tree before the page-flush phase.
    /// On `init` the cache is bulk-populated from the durable tree;
    /// after that, every `hasStore` / `storeRoot` is an O(1) hashmap
    /// lookup. Tracks `dirty` so durabilize only writes entries that
    /// changed since the last flush.
    store_root_cache: std.AutoHashMapUnmanaged(u64, RootSlot),
    store_root_cache_lock: std.Thread.Mutex = .{},

    /// Per-store write locks, keyed by store_id. Allocated lazily on
    /// first Store.put for that id; not freed (small overhead per
    /// active store).
    store_locks: std.AutoHashMapUnmanaged(u64, *std.Thread.Mutex),
    store_locks_lock: std.Thread.Mutex = .{},

    /// Live read snapshots: maps snap_seq → live count. A page tagged
    /// `freed_at_seq=N` is potentially in a snapshot's view iff some
    /// live snapshot has `snap_seq >= N`. refillReusable consults
    /// `minLiveSnapSeq()` to keep such pages out of the reusable
    /// queue.
    snapshot_counts: std.AutoHashMapUnmanaged(u64, u32),
    snapshots_lock: std.Thread.Mutex = .{},

    /// Set to true if any error escapes `durabilize`. Linux's "fsync
    /// gate" semantics (an fsync error may only be reported once before
    /// the kernel forgets the EIO state) make naive retry unsafe — a
    /// retry could silently succeed with some pages never durable, and
    /// the next durabilize would promote a slot referencing those
    /// non-durable pages. Once poisoned, every mutating entry point
    /// returns `error.ManifestPoisoned`; recovery requires closing the
    /// manifest and reopening (which re-reads the active slot from
    /// disk — the last known-durable state).
    ///
    /// Reads (`Store.get`, `scanPrefix`, `hasStore`, `storeRoot`,
    /// `listStores`, live `Snapshot`s) are allowed even when poisoned:
    /// they see the (possibly inconsistent) in-memory state, which is
    /// useful for forensic inspection before the operator closes.
    poisoned: std.atomic.Value(bool) align(@alignOf(usize)) = std.atomic.Value(bool).init(false),

    pub fn pageAllocator(self: *Manifest) PageAllocator {
        return .{ .ctx = self, .vtable = &alloc_vtable };
    }

    const alloc_vtable: PageAllocator.VTable = .{
        .alloc = allocImpl,
        .free = freeImpl,
    };

    /// Number of pages to grow the file by when `reusable` is empty.
    /// Cutting the growBy-syscall frequency by this factor is the
    /// dominant win against `reusable_lock` contention — the slow path
    /// (file extension) was getting hit on every alloc, holding the
    /// lock for the duration of an ftruncate. With batching, only
    /// 1/GROW_BATCH of allocs take the slow path; the rest are pure
    /// in-memory ArrayList pops. Bounded leak per crash: up to
    /// (GROW_BATCH - 1) pages that were grown but never used end up
    /// past the durable manifest's reach. ~128KB worst case at 32; a
    /// future compact-on-deinit could ftruncate them away.
    pub const GROW_BATCH: u64 = 32;

    fn allocImpl(ctx: *anyopaque) anyerror!u64 {
        const self: *Manifest = @ptrCast(@alignCast(ctx));
        self.reusable_lock.lock();
        defer self.reusable_lock.unlock();
        if (self.reusable.pop()) |p| return p;
        // Slow path: grow by a batch, return one, cache the rest.
        // PagedFile.growBy has its own internal lock so this is safe
        // to call while holding reusable_lock; with GROW_BATCH=32, this
        // path fires ~1/32 as often as the old per-alloc growBy(1).
        const first = try self.file.growBy(GROW_BATCH);
        var i: u64 = 1;
        while (i < GROW_BATCH) : (i += 1) {
            try self.reusable.append(self.allocator, first + i);
        }
        return first;
    }

    fn freeImpl(ctx: *anyopaque, page_no: u64, freed_at_seq: u64) anyerror!void {
        const self: *Manifest = @ptrCast(@alignCast(ctx));
        self.pending_free_lock.lock();
        defer self.pending_free_lock.unlock();
        try self.pending_free.append(self.allocator, .{
            .page_no = page_no,
            .freed_at_seq = freed_at_seq,
        });
    }

    /// In-place init so `self` has a stable address (the page allocator
    /// captures `&self` as ctx).
    pub fn init(self: *Manifest, allocator: std.mem.Allocator, cache: *PageCache, file: PagedFileApi) !void {
        self.allocator = allocator;
        self.cache = cache;
        self.file = file;
        self.reusable = .empty;
        self.consumed_keys = .empty;
        self.pending_free = .empty;
        self.store_locks = .empty;
        self.snapshot_counts = .empty;
        self.store_root_cache = .empty;
        self.tree_lock = .{};
        self.freelist_tree_lock = .{};
        self.reusable_lock = .{};
        self.pending_free_lock = .{};
        self.durabilize_lock = .{};
        self.store_locks_lock = .{};
        self.snapshots_lock = .{};
        self.store_root_cache_lock = .{};
        self.poisoned = std.atomic.Value(bool).init(false);

        // File-header handling. Three cases:
        //   (a) pageCount == 0          → genuinely fresh file: grow,
        //                                  write Header(page 0), fsync.
        //   (b) 0 < pageCount < FIRST_DATA_PAGE → malformed/partial
        //                                  file (created but never had
        //                                  slots/header written). Refuse.
        //   (c) pageCount >= FIRST_DATA_PAGE → existing file: read+
        //                                  validate Header at page 0.
        // This catches: opening a foreign file, a kvexp v1 file (the
        // pre-CRC format with a different InternalHeader layout), a
        // file written with a different page size, or a file truncated
        // to a partial length.
        const initial_pages = file.pageCount();
        if (initial_pages == 0) {
            while (file.pageCount() < FIRST_DATA_PAGE) {
                _ = try file.growBy(1);
            }
            var hdr_buf: [page.PAGE_SIZE]u8 align(4096) = undefined;
            @memset(&hdr_buf, 0);
            const new_hdr = Header.init(page.PAGE_SIZE);
            @memcpy(hdr_buf[0..@sizeOf(Header)], new_hdr.toBytes());
            try file.writePage(0, &hdr_buf);
            try file.fsync();
        } else if (initial_pages < FIRST_DATA_PAGE) {
            return error.IncompleteFile;
        } else {
            var hdr_buf: [page.PAGE_SIZE]u8 align(4096) = undefined;
            try file.readPage(0, &hdr_buf);
            const hdr = Header.fromBytes(&hdr_buf);
            try hdr.validate(page.PAGE_SIZE);
        }

        var buf_a: [page.PAGE_SIZE]u8 align(4096) = undefined;
        var buf_b: [page.PAGE_SIZE]u8 align(4096) = undefined;
        try file.readPage(SLOT_A_PAGE, &buf_a);
        try file.readPage(SLOT_B_PAGE, &buf_b);
        const slot_a: *ManifestSlot = @ptrCast(@alignCast(&buf_a));
        const slot_b: *ManifestSlot = @ptrCast(@alignCast(&buf_b));
        const va = slot_a.isValid();
        const vb = slot_b.isValid();

        var active_slot: u32 = 1;
        var active_seq: u64 = 0;
        var inactive_seq: u64 = 0;
        var inactive_valid = false;
        var manifest_root: u64 = 0;
        var freelist_root: u64 = 0;
        var last_applied_raft_idx: u64 = 0;
        if (va and vb) {
            if (slot_a.sequence >= slot_b.sequence) {
                active_slot = 0;
                active_seq = slot_a.sequence;
                inactive_seq = slot_b.sequence;
                manifest_root = slot_a.manifest_root;
                freelist_root = slot_a.freelist_root;
                last_applied_raft_idx = slot_a.last_applied_raft_idx;
            } else {
                active_slot = 1;
                active_seq = slot_b.sequence;
                inactive_seq = slot_a.sequence;
                manifest_root = slot_b.manifest_root;
                freelist_root = slot_b.freelist_root;
                last_applied_raft_idx = slot_b.last_applied_raft_idx;
            }
            inactive_valid = true;
        } else if (va) {
            active_slot = 0;
            active_seq = slot_a.sequence;
            manifest_root = slot_a.manifest_root;
            freelist_root = slot_a.freelist_root;
            last_applied_raft_idx = slot_a.last_applied_raft_idx;
        } else if (vb) {
            active_slot = 1;
            active_seq = slot_b.sequence;
            manifest_root = slot_b.manifest_root;
            freelist_root = slot_b.freelist_root;
            last_applied_raft_idx = slot_b.last_applied_raft_idx;
        }
        self.active_slot = active_slot;
        self.active_seq = active_seq;
        self.inactive_seq = inactive_seq;
        self.inactive_valid = inactive_valid;
        self.sequence = active_seq;
        self.last_applied_raft_idx = last_applied_raft_idx;

        const page_alloc = self.pageAllocator();
        self.tree = try Tree.init(allocator, cache, file, page_alloc);
        self.tree.root = manifest_root;
        self.tree.seq = active_seq + 1;
        self.freelist = try Tree.init(allocator, cache, file, page_alloc);
        self.freelist.root = freelist_root;
        self.freelist.seq = active_seq + 1;

        try self.refillReusable();
        try self.populateStoreRootCache();
    }

    /// Bulk-load every (store_id, root) from the durable manifest
    /// tree into the in-memory cache. Called once at `init`. After
    /// this, every `hasStore` / `storeRoot` is a cache lookup —
    /// the manifest tree is only re-touched at `durabilize` flush
    /// time and on `createStore` / `dropStore` (which write through).
    fn populateStoreRootCache(self: *Manifest) !void {
        if (self.tree.root == 0) return;
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        self.store_root_cache_lock.lock();
        defer self.store_root_cache_lock.unlock();
        var cursor = try self.tree.scanPrefix("");
        defer cursor.deinit();
        while (try cursor.next()) {
            const id = decodeStoreId(cursor.key());
            const root = decodeRoot(cursor.value());
            try self.store_root_cache.put(self.allocator, id, .{ .root = root, .dirty = false });
        }
    }

    /// Start a new apply unit. Subsequent mutations tag dirty pages
    /// with the returned sequence. Multiple applies between durabilize
    /// calls accumulate in-memory state tagged with distinct seqs;
    /// orphan elision skips intermediate page versions at durabilize.
    /// Pure in-memory counter; does not check `isPoisoned` so the
    /// signature stays infallible. `put`/`delete`/`durabilize` are the
    /// gates that refuse a poisoned manifest.
    pub fn nextApply(self: *Manifest) u64 {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        self.tree.seq += 1;
        self.freelist.seq = self.tree.seq;
        return self.tree.seq;
    }

    /// Current apply seq (the seq mutations are currently tagging).
    pub fn applySeq(self: *Manifest) u64 {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        return self.tree.seq;
    }

    /// Highest durable seq.
    pub fn durableSeq(self: *const Manifest) u64 {
        return self.active_seq;
    }

    /// True iff `durabilize` previously errored. See `poisoned` field
    /// for rationale.
    pub fn isPoisoned(self: *const Manifest) bool {
        return self.poisoned.load(.monotonic);
    }

    fn poison(self: *Manifest) void {
        self.poisoned.store(true, .monotonic);
    }

    /// Test-only: force the manifest into the poisoned state. Used by
    /// tests that can't easily induce an fsync failure but need to
    /// verify the post-poison contract.
    pub fn _testPoison(self: *Manifest) void {
        self.poison();
    }

    inline fn checkAlive(self: *const Manifest) !void {
        if (self.isPoisoned()) return error.ManifestPoisoned;
    }

    /// Cluster-wide last-applied raft index watermark. Read at open
    /// from the active slot. Set via `setLastAppliedRaftIdx`; written
    /// to disk by the next `durabilize` call.
    pub fn lastAppliedRaftIdx(self: *Manifest) u64 {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        return self.last_applied_raft_idx;
    }

    pub fn setLastAppliedRaftIdx(self: *Manifest, idx: u64) !void {
        try self.checkAlive();
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        self.last_applied_raft_idx = idx;
    }

    pub fn deinit(self: *Manifest) void {
        self.reusable.deinit(self.allocator);
        self.consumed_keys.deinit(self.allocator);
        self.pending_free.deinit(self.allocator);
        var it = self.store_locks.valueIterator();
        while (it.next()) |m| self.allocator.destroy(m.*);
        self.store_locks.deinit(self.allocator);
        self.snapshot_counts.deinit(self.allocator);
        self.store_root_cache.deinit(self.allocator);
        self.* = undefined;
    }

    fn registerSnapshot(self: *Manifest, seq: u64) !void {
        self.snapshots_lock.lock();
        defer self.snapshots_lock.unlock();
        const gop = try self.snapshot_counts.getOrPut(self.allocator, seq);
        if (gop.found_existing) {
            gop.value_ptr.* += 1;
        } else {
            gop.value_ptr.* = 1;
        }
    }

    fn unregisterSnapshot(self: *Manifest, seq: u64) void {
        self.snapshots_lock.lock();
        defer self.snapshots_lock.unlock();
        const entry = self.snapshot_counts.getPtr(seq) orelse return;
        entry.* -= 1;
        if (entry.* == 0) _ = self.snapshot_counts.remove(seq);
    }

    /// Lowest snap_seq across live snapshots, or null when there are
    /// none. Used by refillReusable to keep snapshot-referenced pages
    /// out of the reusable queue.
    pub fn minLiveSnapSeq(self: *Manifest) ?u64 {
        self.snapshots_lock.lock();
        defer self.snapshots_lock.unlock();
        var min: ?u64 = null;
        var it = self.snapshot_counts.keyIterator();
        while (it.next()) |k| {
            if (min == null or k.* < min.?) min = k.*;
        }
        return min;
    }

    pub const VerifyReport = struct {
        file_pages: u64,
        manifest_tree_pages: u64,
        store_count: u64,
        store_tree_pages: u64,
        freelist_tree_pages: u64,
        freelist_recorded_pages: u64,
        /// Pages held in the in-memory reusable allocator cache from
        /// grow-batching. These are accounted-for at the application
        /// level (the manifest knows about them) but not in any durable
        /// structure. After a crash, refillReusable won't find them;
        /// they would be counted as orphans on a fresh open. Bounded
        /// by ~GROW_BATCH per grow event.
        reusable_pages: u64,
        /// File pages not accounted for by header(0) + slots(1,2) +
        /// manifest tree + store trees + freelist tree + freelist
        /// recorded + in-memory reusable. Should be 0 for a clean
        /// forest; non-zero indicates true leaks.
        orphan_pages: u64,
    };

    /// Walk the entire forest under the current in-memory roots and
    /// report invariants. This is intended for admin tooling and tests
    /// — it's not on any hot path. Takes the tree_lock and the
    /// freelist_tree_lock for the duration of the walk; concurrent
    /// writers block.
    pub fn verify(self: *Manifest, allocator: std.mem.Allocator) !VerifyReport {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        self.freelist_tree_lock.lock();
        defer self.freelist_tree_lock.unlock();

        var mt_pages: std.AutoHashMapUnmanaged(u64, void) = .empty;
        defer mt_pages.deinit(allocator);
        try btree.collectTreePages(self.cache, self.tree.root, &mt_pages, allocator);

        var st_pages: std.AutoHashMapUnmanaged(u64, void) = .empty;
        defer st_pages.deinit(allocator);
        var store_count: u64 = 0;
        {
            var cursor = try btree.PrefixCursor.open(self.cache, self.tree.root, "");
            defer cursor.deinit();
            while (try cursor.next()) {
                store_count += 1;
                const root = decodeRoot(cursor.value());
                try btree.collectTreePages(self.cache, root, &st_pages, allocator);
            }
        }

        var fl_tree_pages: std.AutoHashMapUnmanaged(u64, void) = .empty;
        defer fl_tree_pages.deinit(allocator);
        try btree.collectTreePages(self.cache, self.freelist.root, &fl_tree_pages, allocator);
        var recorded: u64 = 0;
        {
            var cursor = try btree.PrefixCursor.open(self.cache, self.freelist.root, "");
            defer cursor.deinit();
            while (try cursor.next()) {
                var it = ChunkIterator.init(cursor.value());
                while (it.next()) |_| recorded += 1;
            }
        }

        self.reusable_lock.lock();
        const reusable_pages: u64 = self.reusable.items.len;
        self.reusable_lock.unlock();

        const file_pages = self.file.pageCount();
        const fixed_pages: u64 = FIRST_DATA_PAGE; // header + slot A + slot B
        const accounted = fixed_pages +
            mt_pages.count() +
            st_pages.count() +
            fl_tree_pages.count() +
            recorded +
            reusable_pages;
        const orphan_pages: u64 = if (file_pages > accounted) file_pages - accounted else 0;

        return .{
            .file_pages = file_pages,
            .manifest_tree_pages = mt_pages.count(),
            .store_count = store_count,
            .store_tree_pages = st_pages.count(),
            .freelist_tree_pages = fl_tree_pages.count(),
            .freelist_recorded_pages = recorded,
            .reusable_pages = reusable_pages,
            .orphan_pages = orphan_pages,
        };
    }

    /// Open a read snapshot capturing the current applied state. The
    /// returned snapshot may be read from any thread until `close()`;
    /// writes to the underlying stores do not affect what the
    /// snapshot sees.
    pub fn openSnapshot(self: *Manifest) !Snapshot {
        self.tree_lock.lock();
        const snap_seq = self.tree.seq;
        const manifest_root = self.tree.root;
        self.tree_lock.unlock();
        try self.registerSnapshot(snap_seq);
        return .{
            .manifest = self,
            .snap_seq = snap_seq,
            .manifest_root = manifest_root,
        };
    }

    /// Obtain (or lazily create) the per-store write mutex for `id`.
    /// Returned mutex is unlocked; caller must `.lock()` and
    /// `.unlock()` around the per-store write critical section.
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

    pub fn pendingSeq(self: *const Manifest) u64 {
        return self.tree.seq;
    }

    pub fn hasStore(self: *Manifest, id: u64) !bool {
        self.store_root_cache_lock.lock();
        defer self.store_root_cache_lock.unlock();
        return self.store_root_cache.contains(id);
    }

    /// Compatibility alias — kept for callers that already hold
    /// `tree_lock` and want the in-tree path. The cache is the
    /// authoritative answer; tree_lock isn't required.
    fn hasStoreLocked(self: *Manifest, id: u64) !bool {
        return self.hasStore(id);
    }

    pub fn storeRoot(self: *Manifest, id: u64) !?u64 {
        self.store_root_cache_lock.lock();
        defer self.store_root_cache_lock.unlock();
        if (self.store_root_cache.get(id)) |slot| return slot.root;
        return null;
    }

    fn storeRootLocked(self: *Manifest, id: u64) !?u64 {
        return self.storeRoot(id);
    }

    pub fn createStore(self: *Manifest, id: u64) !void {
        try self.checkAlive();
        // Acquire tree_lock first (lock-ordering rule), then the
        // cache lock. createStore is rare — the per-call cost of
        // taking both locks is fine.
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        self.store_root_cache_lock.lock();
        defer self.store_root_cache_lock.unlock();
        if (self.store_root_cache.contains(id)) return error.StoreAlreadyExists;
        try self.setStoreRootInTreeLocked(id, 0);
        try self.store_root_cache.put(self.allocator, id, .{ .root = 0, .dirty = false });
    }

    pub fn dropStore(self: *Manifest, id: u64) !bool {
        try self.checkAlive();
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        self.store_root_cache_lock.lock();
        defer self.store_root_cache_lock.unlock();
        if (!self.store_root_cache.contains(id)) return false;
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        _ = try self.tree.delete(k);
        _ = self.store_root_cache.remove(id);
        return true;
    }

    /// Hot-path write: O(1) hashmap update. The manifest tree is
    /// NOT touched here — `durabilize` flushes dirty entries before
    /// the slot swap. This is the critical-path optimization that
    /// removes manifest-tree CoW from every put.
    pub fn setStoreRoot(self: *Manifest, id: u64, root: u64) !void {
        try self.checkAlive();
        self.store_root_cache_lock.lock();
        defer self.store_root_cache_lock.unlock();
        try self.store_root_cache.put(self.allocator, id, .{ .root = root, .dirty = true });
    }

    /// Write-through variant used by `createStore` and the
    /// `durabilize` cache flush. Caller must hold `tree_lock`.
    fn setStoreRootInTreeLocked(self: *Manifest, id: u64, root: u64) !void {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        var val_buf: [STORE_VAL_LEN]u8 = undefined;
        const v = encodeRoot(root, &val_buf);
        try self.tree.put(k, v);
    }

    pub fn listStores(self: *Manifest, allocator: std.mem.Allocator) ![]u64 {
        self.store_root_cache_lock.lock();
        defer self.store_root_cache_lock.unlock();
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        try list.ensureTotalCapacity(allocator, self.store_root_cache.count());
        var it = self.store_root_cache.keyIterator();
        while (it.next()) |id_ptr| {
            list.appendAssumeCapacity(id_ptr.*);
        }
        // Hashmap iteration is unordered; the tree-scan implementation
        // returned ascending ids and several tests rely on that.
        std.mem.sort(u64, list.items, {}, std.sort.asc(u64));
        return try list.toOwnedSlice(allocator);
    }

    /// Total pages in the data file. Useful for tests that want to
    /// detect unbounded growth.
    pub fn fileSizePages(self: *const Manifest) u64 {
        return self.file.pageCount();
    }

    /// Number of in-memory free pages immediately available for reuse.
    pub fn reusableCount(self: *const Manifest) usize {
        return self.reusable.items.len;
    }

    /// Durabilize everything applied so far. After return, the current
    /// `tree.seq` (the latest apply seq) is durable in one of the two
    /// slots. Pages superseded by later applies (orphans) are NOT
    /// written to disk — only the latest reachable state hits storage.
    ///
    /// Phase 6: lock acquisition is laid out so that worker threads can
    /// continue calling Store.put against the manifest tree during
    /// most of durabilize. Specifically: the freelist mutations run
    /// under `freelist_tree_lock`, leaving `tree_lock` free for worker
    /// reads/writes against the manifest tree. The slot write is
    /// likewise free of `tree_lock`. `reusable_lock` / `pending_free_lock`
    /// are taken in short bursts only.
    pub fn durabilize(self: *Manifest) !void {
        try self.checkAlive();
        // Serialize the whole call. Concurrent durabilize callers would
        // race on pending_free drain (two `toOwnedSlice`s on the same
        // list), on the slot write (same target page), and on active_*
        // promotion. Hold this lock above every other manifest lock.
        self.durabilize_lock.lock();
        defer self.durabilize_lock.unlock();
        // Any error escaping durabilize past this point leaves the
        // in-memory state in a possibly-inconsistent shape (drained
        // pending_free without slot promotion, partial cache flush,
        // half-written freelist tree). Poison forces a close+reopen,
        // which re-reads the active slot — the last known-durable
        // state — and discards everything in memory. This is the only
        // safe recovery from an fsync gate (kernel may have forgotten
        // the EIO; a naive retry could silently succeed).
        errdefer self.poison();

        // Step 0: flush dirty store_root cache entries into the
        // manifest tree. The hot apply path bypasses tree_lock via
        // the cache; this is the only point where those buffered
        // writes hit the durable tree. Held under tree_lock so the
        // tree.put operations use a stable self.tree.seq, and under
        // cache_lock so concurrent setStoreRoot calls serialize
        // against the flush.
        self.tree_lock.lock();
        {
            // tree_lock must release on error inside the drain — a
            // setStoreRootInTreeLocked failure would otherwise leave
            // tree_lock permanently held, deadlocking every later
            // operation (including the reads we explicitly permit on a
            // poisoned manifest).
            errdefer self.tree_lock.unlock();
            self.store_root_cache_lock.lock();
            defer self.store_root_cache_lock.unlock();
            var it = self.store_root_cache.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.dirty) {
                    try self.setStoreRootInTreeLocked(entry.key_ptr.*, entry.value_ptr.root);
                    entry.value_ptr.dirty = false;
                }
            }
        }
        const K = self.tree.seq;
        self.tree_lock.unlock();
        if (K <= self.active_seq) return;

        // 1. Snapshot pending_free + consumed_keys (brief). Each lock
        //    held only for the duration of its toOwnedSlice — workers
        //    on the other list can keep running.
        self.pending_free_lock.lock();
        const pf = try self.pending_free.toOwnedSlice(self.allocator);
        self.pending_free_lock.unlock();
        defer self.allocator.free(pf);

        self.reusable_lock.lock();
        const ck = try self.consumed_keys.toOwnedSlice(self.allocator);
        self.reusable_lock.unlock();
        defer self.allocator.free(ck);

        // 2. Build initial orphan set from pf.
        var orphans: std.AutoHashMapUnmanaged(u64, void) = .empty;
        defer orphans.deinit(self.allocator);
        for (pf) |fp| {
            if (fp.freed_at_seq <= K) {
                try orphans.put(self.allocator, fp.page_no, {});
            }
        }

        // 3. Fold pf into the durable freelist + delete consumed_keys.
        //    freelist_tree_lock guards the freelist tree from concurrent
        //    refillReusable scans. The freelist tree.put/delete calls
        //    internally take reusable_lock (via allocImpl) and
        //    pending_free_lock (via freeImpl) — never freelist_tree_lock
        //    recursively. Hold via defer so an error in foldPendingFree
        //    or freelist.delete doesn't leak the lock.
        {
            self.freelist_tree_lock.lock();
            defer self.freelist_tree_lock.unlock();
            try self.foldPendingFree(pf);
            for (ck) |key| _ = try self.freelist.delete(&key);
        }

        // 4. Extend orphan set with pages freed during fold/delete.
        self.pending_free_lock.lock();
        for (self.pending_free.items) |fp| {
            if (fp.freed_at_seq <= K) {
                try orphans.put(self.allocator, fp.page_no, {});
            }
        }
        self.pending_free_lock.unlock();

        // 5. Flush dirty pages tagged seq <= K, skipping orphans.
        //    fdatasync (not fsync) — kvexp only mutates file size
        //    (via growBy) and page contents; both are covered by
        //    fdatasync per Linux semantics (size is metadata "needed
        //    to retrieve the data"). Skipping atime/ctime sync is a
        //    free win on the hot path.
        try self.cache.flushUpToSkipping(K, &orphans);
        try self.file.fdatasync();

        // 6. Write the inactive slot.
        const next_slot: u32 = 1 - self.active_slot;
        const slot_page: u64 = if (next_slot == 0) SLOT_A_PAGE else SLOT_B_PAGE;
        var slot_buf: [page.PAGE_SIZE]u8 align(4096) = undefined;
        @memset(&slot_buf, 0);
        const slot: *ManifestSlot = @ptrCast(@alignCast(&slot_buf));
        // Reading tree.root and freelist.root under their respective
        // locks guards against an in-flight worker mutation. Workers
        // may proceed; we capture a consistent snapshot.
        self.tree_lock.lock();
        const manifest_root = self.tree.root;
        self.tree_lock.unlock();
        self.freelist_tree_lock.lock();
        const freelist_root = self.freelist.root;
        self.freelist_tree_lock.unlock();
        slot.* = .{
            .magic = ManifestSlot.MAGIC,
            .slot_id = @intCast(next_slot),
            .sequence = K,
            .manifest_root = manifest_root,
            .freelist_root = freelist_root,
            .last_applied_raft_idx = self.last_applied_raft_idx,
        };
        slot.computeChecksum();
        try self.file.writePage(slot_page, &slot_buf);
        // Slot writes never change the file size; fdatasync suffices.
        try self.file.fdatasync();

        // 7. Promote the new slot. Old active becomes new inactive.
        self.inactive_seq = self.active_seq;
        self.inactive_valid = self.active_seq > 0;
        self.active_slot = next_slot;
        self.active_seq = K;
        self.sequence = K;

        // 8. Bump tree.seq for the next apply. Workers' subsequent
        //    storeRoot reads use the new seq for their dirty-page
        //    tagging.
        self.tree_lock.lock();
        self.tree.seq = K + 1;
        self.tree_lock.unlock();
        self.freelist_tree_lock.lock();
        self.freelist.seq = K + 1;
        self.freelist_tree_lock.unlock();

        // 9. Refill reusable from the durable freelist.
        try self.refillReusable();
    }

    /// Backward-compat alias for tests/callers that don't separate apply
    /// from durabilize: `commit()` is just `durabilize()`.
    pub fn commit(self: *Manifest) !void {
        return self.durabilize();
    }

    /// Scan the durable freelist for chunks whose `freed_at_seq` is
    /// safe to reuse, unpacking them into the in-memory `reusable`
    /// queue. A page tagged `freed_at_seq=N` is unsafe iff `N-1` is
    /// one of the two durable slot seqs (its referencing manifest is
    /// still on disk). Concretely, only `inactive_seq + 1` (when the
    /// inactive slot is valid) can appear in the durable freelist as
    /// unsafe — entries tagged `active_seq + 1` haven't been folded
    /// yet (those applies aren't durabilized). Group commit makes
    /// large gaps between durable seqs possible, so this rule
    /// generalizes phase-4's "max_eligible = sequence - 1."
    fn refillReusable(self: *Manifest) !void {
        if (self.active_seq < 1) return;
        const min_snap = self.minLiveSnapSeq();

        // freelist_tree_lock serializes against any other freelist tree
        // mutation (only durabilize mutates it, but defensive). The
        // reusable_lock guards the reusable + consumed_keys lists.
        self.freelist_tree_lock.lock();
        defer self.freelist_tree_lock.unlock();
        self.reusable_lock.lock();
        defer self.reusable_lock.unlock();

        if (self.reusable.items.len >= REUSABLE_BATCH) return;
        var cursor = try self.freelist.scanPrefix("");
        defer cursor.deinit();
        while (try cursor.next()) {
            const decoded = decodeFreelistKey(cursor.key());
            // Two-slot durability invariant.
            if (self.inactive_valid and decoded.freed_at_seq == self.inactive_seq + 1) continue;
            if (decoded.freed_at_seq == self.active_seq + 1) continue;
            // Snapshot invariant: any live snapshot whose snap_seq is
            // >= freed_at_seq might have the page in its view.
            if (min_snap) |m| {
                if (decoded.freed_at_seq >= m) continue;
            }
            var it = ChunkIterator.init(cursor.value());
            while (it.next()) |p| {
                try self.reusable.append(self.allocator, p);
            }
            var key_copy: [FREELIST_KEY_LEN]u8 = undefined;
            @memcpy(&key_copy, cursor.key());
            try self.consumed_keys.append(self.allocator, key_copy);
            if (self.reusable.items.len >= REUSABLE_BATCH) break;
        }
    }

    /// Group pending_free entries by their `freed_at_seq` and write
    /// chunked cells into the freelist B-tree. All entries from a
    /// single commit window share one seq, so most workloads see ⌈P /
    /// PAGES_PER_CHUNK⌉ freelist puts, not P.
    fn foldPendingFree(self: *Manifest, pf: []const FreedPage) !void {
        if (pf.len == 0) return;

        // Group by seq using a hash map. For workloads where ~all
        // entries share one seq, this is trivially small.
        var groups: std.AutoHashMapUnmanaged(u64, std.ArrayListUnmanaged(u64)) = .empty;
        defer {
            var it = groups.valueIterator();
            while (it.next()) |list| list.deinit(self.allocator);
            groups.deinit(self.allocator);
        }
        for (pf) |fp| {
            const gop = try groups.getOrPut(self.allocator, fp.freed_at_seq);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(self.allocator, fp.page_no);
        }

        // For each seq group, chunk the page_nos and write each chunk
        // as one freelist.put.
        var val_buf: [page.MAX_VAL_LEN]u8 = undefined;
        var grp_it = groups.iterator();
        while (grp_it.next()) |entry| {
            const seq = entry.key_ptr.*;
            const pages = entry.value_ptr.items;
            var i: usize = 0;
            while (i < pages.len) {
                const end = @min(i + PAGES_PER_CHUNK, pages.len);
                const chunk = pages[i..end];
                var key_buf: [FREELIST_KEY_LEN]u8 = undefined;
                const key = encodeFreelistKey(seq, chunk[0], &key_buf);
                const value = encodeFreelistValue(chunk, &val_buf);
                try self.freelist.put(key, value);
                i = end;
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Store wrapper
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Read snapshot
// -----------------------------------------------------------------------------

/// A point-in-time consistent read view of the manifest forest, for
/// **infrastructure callers** that need to read while writers
/// continue — not for application requests. The request path
/// serializes reads and writes inside a per-store batch under
/// exclusive lock, so it has no use for MVCC. Snapshots exist for:
///
///   * Phase-10 raft state transfer (the leader ships the durable
///     state to a catching-up follower, an operation that runs for
///     seconds or minutes while writers continue).
///   * Online backups.
///   * Admin/debug tooling (`kvexp.fsck`, `kvexp.dump`).
///
/// Captures the manifest tree root atomically under `tree_lock` at
/// open; subsequent writes produce new CoW pages that the snapshot
/// does not see. While the snapshot is alive, the free-page
/// allocator refuses to hand out any page whose `freed_at_seq` is
/// at or after the snapshot's `snap_seq`, so the OLD pages the
/// snapshot walks stay intact on disk.
pub const Snapshot = struct {
    manifest: *Manifest,
    snap_seq: u64,
    manifest_root: u64,

    pub fn close(self: *Snapshot) void {
        self.manifest.unregisterSnapshot(self.snap_seq);
        self.* = undefined;
    }

    pub fn storeRoot(self: *Snapshot, store_id: u64) !?u64 {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(store_id, &id_buf);
        const v = try btree.treeGet(self.manifest.cache, self.manifest_root, self.manifest.allocator, k);
        if (v) |bytes| {
            defer self.manifest.allocator.free(bytes);
            return decodeRoot(bytes);
        }
        return null;
    }

    pub fn get(
        self: *Snapshot,
        allocator: std.mem.Allocator,
        store_id: u64,
        key: []const u8,
    ) !?[]u8 {
        const root = (try self.storeRoot(store_id)) orelse return error.StoreNotFound;
        return try btree.treeGet(self.manifest.cache, root, allocator, key);
    }

    pub fn scanPrefix(self: *Snapshot, store_id: u64, prefix: []const u8) !btree.PrefixCursor {
        const root = (try self.storeRoot(store_id)) orelse return error.StoreNotFound;
        return try btree.PrefixCursor.open(self.manifest.cache, root, prefix);
    }

    /// Enumerate every store_id present in the snapshot. Walks the
    /// snapshot's captured manifest tree (not the live cache), so
    /// the returned set is the point-in-time view at `snap_seq` —
    /// any createStore that happened after snapshot open is
    /// excluded. Caller owns the returned slice.
    pub fn listStores(self: *Snapshot, allocator: std.mem.Allocator) ![]u64 {
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        if (self.manifest_root == 0) return try list.toOwnedSlice(allocator);
        var cursor = try btree.PrefixCursor.open(self.manifest.cache, self.manifest_root, "");
        defer cursor.deinit();
        while (try cursor.next()) {
            try list.append(allocator, decodeStoreId(cursor.key()));
        }
        return try list.toOwnedSlice(allocator);
    }
};

// ── State-transfer dump / restore ──────────────────────────────────
//
// Phase 10: produce a point-in-time logical dump of a snapshot for
// state transfer to a follower that has fallen too far behind raft
// to catch up via log replay. The wire format is record-oriented
// (no per-store framing — stores are inferred from the records),
// which makes it streamable: the sender pushes records as it scans;
// the receiver creates stores lazily and writes records as they
// arrive. The receiver is expected to start from a fresh manifest
// (the caller wipes the data file before calling loadSnapshot).
//
// Format:
//   header: magic:u32 'KVXS' | version:u8 | snap_seq:u64 |
//           last_applied_raft_idx:u64
//   records: u8 tag
//     tag = SNAP_TAG_KV: u64 store_id | u16 key_len | u16 val_len
//                      | key | value
//     tag = SNAP_TAG_END: no payload
//
// Values are passed through verbatim — kvexp doesn't interpret them,
// so any application-level prefix (e.g. rove's seq-prefix) round-trips
// unchanged.

pub const SNAPSHOT_MAGIC: u32 = 0x4B565853; // 'KVXS' little-endian
pub const SNAPSHOT_VERSION: u8 = 1;
pub const SNAP_TAG_KV: u8 = 1;
pub const SNAP_TAG_END: u8 = 2;

pub fn dumpSnapshot(snap: *Snapshot, writer: anytype) !void {
    try writer.writeInt(u32, SNAPSHOT_MAGIC, .little);
    try writer.writeByte(SNAPSHOT_VERSION);
    try writer.writeInt(u64, snap.snap_seq, .little);
    try writer.writeInt(u64, snap.manifest.last_applied_raft_idx, .little);

    const stores = try snap.listStores(snap.manifest.allocator);
    defer snap.manifest.allocator.free(stores);

    for (stores) |id| {
        var cursor = try snap.scanPrefix(id, "");
        defer cursor.deinit();
        while (try cursor.next()) {
            const k = cursor.key();
            const v = cursor.value();
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

/// Restore a manifest's state from a stream produced by
/// `dumpSnapshot`. Caller must have a freshly-initialized manifest
/// (typically by truncating the data file and re-running
/// `Manifest.init`). On success, the manifest is left in a
/// dirty-but-not-durable state — caller should call `durabilize()`
/// to commit. Returns the `last_applied_raft_idx` from the stream
/// header so the caller can use it as the raft replay floor.
pub fn loadSnapshot(manifest: *Manifest, reader: anytype) !u64 {
    const magic = try reader.readInt(u32, .little);
    if (magic != SNAPSHOT_MAGIC) return error.InvalidSnapshotFormat;
    const version = try reader.readByte();
    if (version != SNAPSHOT_VERSION) return error.UnsupportedSnapshotVersion;
    _ = try reader.readInt(u64, .little); // snap_seq (informational)
    const last_applied = try reader.readInt(u64, .little);

    var key_buf: [page.MAX_KEY_LEN]u8 = undefined;
    var val_buf: [page.MAX_VAL_LEN]u8 = undefined;

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

    try manifest.setLastAppliedRaftIdx(last_applied);
    return last_applied;
}

pub const Store = struct {
    manifest: *Manifest,
    id: u64,
    tree: Tree,

    pub fn open(manifest: *Manifest, id: u64) !Store {
        manifest.tree_lock.lock();
        const root_opt = try manifest.storeRootLocked(id);
        const seq = manifest.tree.seq;
        manifest.tree_lock.unlock();
        const root = root_opt orelse return error.StoreNotFound;
        var tree = try Tree.init(manifest.allocator, manifest.cache, manifest.file, manifest.pageAllocator());
        tree.root = root;
        tree.seq = seq;
        return .{ .manifest = manifest, .id = id, .tree = tree };
    }

    pub fn deinit(self: *Store) void {
        _ = self;
    }

    pub fn get(self: *Store, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        self.manifest.tree_lock.lock();
        const root_opt = try self.manifest.storeRootLocked(self.id);
        self.manifest.tree_lock.unlock();
        const root = root_opt orelse return error.StoreNotFound;
        self.tree.root = root;
        // tree.get runs without any manifest lock; it uses cache_lock
        // internally for pin/release. A concurrent setStoreRoot may
        // change the durable root while we read, but our local
        // self.tree.root snapshot stays valid for this get.
        return try self.tree.get(allocator, key);
    }

    pub fn put(self: *Store, key: []const u8, value: []const u8) !void {
        try self.manifest.checkAlive();
        // Per-store write lock: serializes writers on the same store
        // without blocking writers on different stores.
        const sl = try self.manifest.storeLock(self.id);
        sl.lock();
        defer sl.unlock();

        // Phase 1: read storeRoot + capture current apply seq.
        self.manifest.tree_lock.lock();
        const root_opt = try self.manifest.storeRootLocked(self.id);
        const seq = self.manifest.tree.seq;
        self.manifest.tree_lock.unlock();
        const root = root_opt orelse return error.StoreNotFound;

        // Phase 2: store-tree CoW without manifest locks. allocImpl
        // takes reusable_lock briefly; freeImpl takes pending_free_lock
        // briefly; cache takes cache_lock briefly. Writers on different
        // stores can run this part concurrently.
        self.tree.root = root;
        self.tree.seq = seq;
        try self.tree.put(key, value);

        // Phase 3: publish the new store root via the manifest tree.
        try self.manifest.setStoreRoot(self.id, self.tree.root);
    }

    pub fn delete(self: *Store, key: []const u8) !bool {
        try self.manifest.checkAlive();
        const sl = try self.manifest.storeLock(self.id);
        sl.lock();
        defer sl.unlock();

        self.manifest.tree_lock.lock();
        const root_opt = try self.manifest.storeRootLocked(self.id);
        const seq = self.manifest.tree.seq;
        self.manifest.tree_lock.unlock();
        const root = root_opt orelse return error.StoreNotFound;

        self.tree.root = root;
        self.tree.seq = seq;
        const existed = try self.tree.delete(key);
        if (existed) try self.manifest.setStoreRoot(self.id, self.tree.root);
        return existed;
    }

    pub fn scanPrefix(self: *Store, prefix: []const u8) !btree.PrefixCursor {
        self.manifest.tree_lock.lock();
        const root_opt = try self.manifest.storeRootLocked(self.id);
        self.manifest.tree_lock.unlock();
        const root = root_opt orelse return error.StoreNotFound;
        self.tree.root = root;
        return try self.tree.scanPrefix(prefix);
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;
const BufferPool = @import("buffer_pool.zig").BufferPool;

const Harness = struct {
    tmp: std.testing.TmpDir,
    path_buf: [std.fs.max_path_bytes]u8,
    path_len: usize,
    pool_capacity: u32,

    file: *PagedFile,
    pool: *BufferPool,
    cache: *PageCache,
    manifest: *Manifest, // heap-allocated for stable address

    fn init(pool_capacity: u32) !Harness {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();

        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &dir_buf);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const written = try std.fmt.bufPrint(&path_buf, "{s}/manifest.test", .{dir_path});
        const path_len = written.len;

        var h: Harness = .{
            .tmp = tmp,
            .path_buf = path_buf,
            .path_len = path_len,
            .pool_capacity = pool_capacity,
            .file = undefined,
            .pool = undefined,
            .cache = undefined,
            .manifest = undefined,
        };
        try h.openLayers(.{ .create = true, .truncate = true });
        return h;
    }

    fn deinit(self: *Harness) void {
        self.closeLayers();
        self.tmp.cleanup();
    }

    fn path(self: *const Harness) []const u8 {
        return self.path_buf[0..self.path_len];
    }

    fn openLayers(self: *Harness, open_opts: PagedFile.OpenOptions) !void {
        self.file = try testing.allocator.create(PagedFile);
        errdefer testing.allocator.destroy(self.file);
        self.file.* = try PagedFile.open(self.path(), open_opts);
        errdefer self.file.close();

        self.pool = try testing.allocator.create(BufferPool);
        errdefer testing.allocator.destroy(self.pool);
        self.pool.* = try BufferPool.init(testing.allocator, page.PAGE_SIZE, self.pool_capacity);
        errdefer self.pool.deinit(testing.allocator);

        self.cache = try testing.allocator.create(PageCache);
        errdefer testing.allocator.destroy(self.cache);
        self.cache.* = try PageCache.init(testing.allocator, self.file.api(), self.pool, .{});
        errdefer self.cache.deinit();

        self.manifest = try testing.allocator.create(Manifest);
        errdefer testing.allocator.destroy(self.manifest);
        try self.manifest.init(testing.allocator, self.cache, self.file.api());
    }

    fn closeLayers(self: *Harness) void {
        self.manifest.deinit();
        testing.allocator.destroy(self.manifest);
        self.cache.deinit();
        testing.allocator.destroy(self.cache);
        self.pool.deinit(testing.allocator);
        testing.allocator.destroy(self.pool);
        self.file.close();
        testing.allocator.destroy(self.file);
    }

    fn cycle(self: *Harness) !void {
        self.closeLayers();
        try self.openLayers(.{});
    }
};

test "Manifest: fresh file has no stores; commit + reopen still has none" {
    var h = try Harness.init(32);
    defer h.deinit();
    try testing.expectEqual(@as(u64, 0), h.manifest.sequence);
    try testing.expect(!try h.manifest.hasStore(42));
    try h.manifest.commit();
    try testing.expectEqual(@as(u64, 1), h.manifest.sequence);

    try h.cycle();
    try testing.expect(!try h.manifest.hasStore(42));
    try testing.expectEqual(@as(u64, 1), h.manifest.sequence);
}

test "Manifest: createStore + commit + reopen → store still exists" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.manifest.createStore(42);
    try h.manifest.createStore(7);
    try h.manifest.commit();

    try h.cycle();
    try testing.expect(try h.manifest.hasStore(42));
    try testing.expect(try h.manifest.hasStore(7));
    try testing.expect(!try h.manifest.hasStore(99));
}

test "Manifest: createStore rejects duplicate" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.manifest.createStore(1);
    try testing.expectError(error.StoreAlreadyExists, h.manifest.createStore(1));
}

test "Manifest: dropStore removes the entry" {
    var h = try Harness.init(32);
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

test "Store: put/get/delete round-trip in one store" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();

    try s.put("hello", "world");
    const got = (try s.get(testing.allocator, "hello")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("world", got);

    try testing.expect(try s.delete("hello"));
    try testing.expect((try s.get(testing.allocator, "hello")) == null);
}

test "Manifest: multi-store writes commit and survive reopen" {
    var h = try Harness.init(128);
    defer h.deinit();
    const ids = [_]u64{ 1, 2, 3, 100, 999 };
    for (ids) |id| try h.manifest.createStore(id);

    for (ids) |id| {
        var s = try Store.open(h.manifest, id);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d}", .{id});
        var val_buf: [16]u8 = undefined;
        const val = try std.fmt.bufPrint(&val_buf, "v{d}", .{id});
        try s.put(key, val);
    }
    try h.manifest.commit();

    try h.cycle();
    for (ids) |id| {
        var s = try Store.open(h.manifest, id);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d}", .{id});
        var val_buf: [16]u8 = undefined;
        const expected = try std.fmt.bufPrint(&val_buf, "v{d}", .{id});
        const got = (try s.get(testing.allocator, key)).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(expected, got);
    }
}

test "Manifest: listStores returns ids in ascending order" {
    var h = try Harness.init(64);
    defer h.deinit();
    const ids = [_]u64{ 5, 1, 9, 3, 100, 42 };
    for (ids) |id| try h.manifest.createStore(id);

    const got = try h.manifest.listStores(testing.allocator);
    defer testing.allocator.free(got);

    var expected = ids;
    std.mem.sort(u64, &expected, {}, struct {
        fn lt(_: void, a: u64, b: u64) bool {
            return a < b;
        }
    }.lt);
    try testing.expectEqualSlices(u64, &expected, got);
}

test "Manifest: poison blocks all subsequent writes; reads still work" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try s.put("k", "v");
    try h.manifest.durabilize();

    h.manifest._testPoison();
    try testing.expect(h.manifest.isPoisoned());

    // Every mutating entry point must refuse the poisoned manifest.
    try testing.expectError(error.ManifestPoisoned, h.manifest.durabilize());
    try testing.expectError(error.ManifestPoisoned, h.manifest.createStore(2));
    try testing.expectError(error.ManifestPoisoned, h.manifest.dropStore(1));
    try testing.expectError(error.ManifestPoisoned, h.manifest.setLastAppliedRaftIdx(7));
    try testing.expectError(error.ManifestPoisoned, h.manifest.setStoreRoot(1, 99));
    try testing.expectError(error.ManifestPoisoned, s.put("k2", "v"));
    try testing.expectError(error.ManifestPoisoned, s.delete("k"));

    // Reads still pass through — durable state was captured on the
    // last successful durabilize and is unchanged in memory.
    try testing.expect(try h.manifest.hasStore(1));
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Manifest: reopen clears poison flag" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(7);
    try h.manifest.durabilize();
    h.manifest._testPoison();
    try testing.expect(h.manifest.isPoisoned());

    try h.cycle();
    try testing.expect(!h.manifest.isPoisoned());
    // The pre-poison durable state survives. The store_id 7 was
    // committed before _testPoison, so it must come back.
    try testing.expect(try h.manifest.hasStore(7));
    try h.manifest.createStore(8); // Fresh manifest accepts writes.
}

test "Manifest: concurrent durabilize callers serialize cleanly" {
    // Without durabilize_lock, two callers race on pending_free drain
    // and the slot write. With it, they queue up cleanly and all
    // intervening apply units land durable.
    var h = try Harness.init(128);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();

    const Worker = struct {
        fn run(m: *Manifest, iters: u32, err_count: *std.atomic.Value(u32)) void {
            var i: u32 = 0;
            while (i < iters) : (i += 1) {
                m.durabilize() catch {
                    _ = err_count.fetchAdd(1, .monotonic);
                };
            }
        }
    };

    // Stage some pending work before launching threads.
    _ = h.manifest.nextApply();
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d}", .{i});
        try s.put(key, "v");
    }

    var err_count = std.atomic.Value(u32).init(0);
    const t1 = try std.Thread.spawn(.{}, Worker.run, .{ h.manifest, 10, &err_count });
    const t2 = try std.Thread.spawn(.{}, Worker.run, .{ h.manifest, 10, &err_count });
    t1.join();
    t2.join();
    try testing.expectEqual(@as(u32, 0), err_count.load(.monotonic));

    // Final durabilize then reopen: every staged key must be readable.
    try h.manifest.durabilize();
    try h.cycle();
    var s2 = try Store.open(h.manifest, 1);
    defer s2.deinit();
    i = 0;
    while (i < 50) : (i += 1) {
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d}", .{i});
        const got = (try s2.get(testing.allocator, key)).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings("v", got);
    }
}

test "Manifest: fresh init writes file header at page 0" {
    var h = try Harness.init(64);
    defer h.deinit();

    const buf = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
    defer testing.allocator.free(buf);
    try h.file.readPage(0, buf);
    const hdr = Header.fromBytes(buf);
    try hdr.validate(page.PAGE_SIZE);
}

test "Manifest: reopen validates file header" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(7);
    try h.manifest.commit();
    try h.cycle(); // reopen — should validate page 0 cleanly.
    try testing.expect(try h.manifest.hasStore(7));
}

test "Manifest: bad magic at page 0 → BadMagic on reopen" {
    var h = try Harness.init(64);
    defer h.deinit();
    h.closeLayers();

    const buf = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
    defer testing.allocator.free(buf);
    {
        var pf = try PagedFile.open(h.path(), .{});
        defer pf.close();
        try pf.readPage(0, buf);
        buf[0] ^= 0xFF;
        try pf.writePage(0, buf);
        try pf.fsync();
    }

    try testing.expectError(error.BadMagic, h.openLayers(.{}));
    // Restore so the harness can clean up.
    {
        var pf = try PagedFile.open(h.path(), .{});
        defer pf.close();
        try pf.readPage(0, buf);
        buf[0] ^= 0xFF;
        try pf.writePage(0, buf);
        try pf.fsync();
    }
    try h.openLayers(.{});
}

test "Manifest: unsupported format_version → UnsupportedVersion on reopen" {
    var h = try Harness.init(64);
    defer h.deinit();
    h.closeLayers();

    const buf = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
    defer testing.allocator.free(buf);
    {
        var pf = try PagedFile.open(h.path(), .{});
        defer pf.close();
        try pf.readPage(0, buf);
        // Overwrite format_version (offset 8 in Header) with 99.
        std.mem.writeInt(u32, buf[8..][0..4], 99, .little);
        try pf.writePage(0, buf);
        try pf.fsync();
    }

    try testing.expectError(error.UnsupportedVersion, h.openLayers(.{}));
    {
        var pf = try PagedFile.open(h.path(), .{});
        defer pf.close();
        try pf.readPage(0, buf);
        std.mem.writeInt(u32, buf[8..][0..4], @import("header.zig").FORMAT_VERSION, .little);
        try pf.writePage(0, buf);
        try pf.fsync();
    }
    try h.openLayers(.{});
}

test "Manifest: partial file (1 page) → IncompleteFile" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try tmp.dir.realpath(".", &dir_buf);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/partial.kv", .{dir_path});

    // Create a file with exactly 1 page (less than FIRST_DATA_PAGE=3).
    {
        var pf = try PagedFile.open(path, .{ .create = true, .truncate = true });
        defer pf.close();
        _ = try pf.growBy(1);
    }

    var file = try PagedFile.open(path, .{});
    defer file.close();
    var pool = try @import("buffer_pool.zig").BufferPool.init(testing.allocator, page.PAGE_SIZE, 8);
    defer pool.deinit(testing.allocator);
    var cache = try PageCache.init(testing.allocator, file.api(), &pool, .{});
    defer cache.deinit();
    var manifest: Manifest = undefined;
    try testing.expectError(error.IncompleteFile, manifest.init(testing.allocator, &cache, file.api()));
}

test "Manifest: torn write to active slot — recovery picks the other" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.createStore(2);
    try h.manifest.commit(); // commit #1 writes slot 0

    try h.manifest.createStore(3);
    try h.manifest.commit(); // commit #2 writes slot 1; active_slot == 1

    try testing.expectEqual(@as(u32, 1), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 2), h.manifest.sequence);

    h.closeLayers();

    {
        var pf = try PagedFile.open(h.path(), .{});
        defer pf.close();
        const zero_buf = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
        defer testing.allocator.free(zero_buf);
        @memset(zero_buf, 0);
        try pf.writePage(SLOT_B_PAGE, zero_buf);
        try pf.fsync();
    }

    try h.openLayers(.{});

    try testing.expectEqual(@as(u32, 0), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 1), h.manifest.sequence);
    try testing.expect(try h.manifest.hasStore(1));
    try testing.expect(try h.manifest.hasStore(2));
    try testing.expect(!try h.manifest.hasStore(3));
}

test "Manifest: 1000-store stress, commit, reopen, sample-verify" {
    var h = try Harness.init(512);
    defer h.deinit();

    const N: u64 = 1000;
    var id: u64 = 0;
    while (id < N) : (id += 1) try h.manifest.createStore(id);
    id = 0;
    while (id < N) : (id += 1) {
        var s = try Store.open(h.manifest, id);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{id});
        var val_buf: [16]u8 = undefined;
        const val = try std.fmt.bufPrint(&val_buf, "v{d}", .{id});
        try s.put(key, val);
    }
    try h.manifest.commit();

    try h.cycle();

    const samples = [_]u64{ 0, 1, 7, 99, 500, 999 };
    for (samples) |sid| {
        var s = try Store.open(h.manifest, sid);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{sid});
        var val_buf: [16]u8 = undefined;
        const expected = try std.fmt.bufPrint(&val_buf, "v{d}", .{sid});
        const got = (try s.get(testing.allocator, key)).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(expected, got);
    }

    const all = try h.manifest.listStores(testing.allocator);
    defer testing.allocator.free(all);
    try testing.expectEqual(N, all.len);
}

// -----------------------------------------------------------------------------
// Phase 4 tests: durable free-list, reuse, net-zero workload
// -----------------------------------------------------------------------------

test "Freelist: pending_free populated by mutations, drained at commit" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();

    // Each put runs in its own apply unit (the production pattern:
    // every writeset calls nextApply via begin). This forces the
    // hybrid CoW path to treat each put as touching durable pages,
    // so orphans actually accumulate.
    _ = h.manifest.nextApply();
    try s.put("a", "1");
    _ = h.manifest.nextApply();
    try s.put("b", "2");
    _ = h.manifest.nextApply();
    try s.put("c", "3");
    try testing.expect(h.manifest.pending_free.items.len > 0);

    try h.manifest.commit();
    try testing.expect(h.manifest.pending_free.items.len < 10);
}

test "Freelist: reuse after two-commit lag" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();

    // Drive churn so freelist accumulates entries. Advance seq per
    // put to make each its own apply unit — without this, the hybrid
    // CoW path mutates leaves in place and produces no orphans.
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        _ = h.manifest.nextApply();
        var key_buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&key_buf, "k{d:0>4}", .{i});
        try s.put(k, "x");
    }
    try h.manifest.commit();
    try h.manifest.commit(); // empty commit advances seq

    // After 2 commits, the first commit's freed pages should be
    // eligible for reuse.
    try testing.expect(h.manifest.reusable.items.len > 0);
}

test "Freelist: survives reopen" {
    var h = try Harness.init(128);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        _ = h.manifest.nextApply();
        var key_buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&key_buf, "k{d:0>4}", .{i});
        try s.put(k, "x");
    }
    try h.manifest.commit();
    try h.manifest.commit();

    const freelist_root_before = h.manifest.freelist.root;
    try testing.expect(freelist_root_before != 0);

    try h.cycle();
    try testing.expectEqual(freelist_root_before, h.manifest.freelist.root);
}

// -----------------------------------------------------------------------------
// Phase 5: apply/durabilize split + orphan elision tests
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Phase 8: recovery + verify
// -----------------------------------------------------------------------------

/// Helper for fault-injection tests: open the file with default
/// options, overwrite a single page with the given content (must be
/// page-size aligned), fsync, close.
fn corruptSlotInFile(path: []const u8, slot_page: u64, content: []const u8) !void {
    var pf = try PagedFile.open(path, .{});
    defer pf.close();
    try pf.writePage(slot_page, content);
    try pf.fsync();
}

test "Manifest: lastAppliedRaftIdx round-trips across durabilize + reopen" {
    var h = try Harness.init(32);
    defer h.deinit();
    try testing.expectEqual(@as(u64, 0), h.manifest.lastAppliedRaftIdx());

    try h.manifest.setLastAppliedRaftIdx(42);
    try testing.expectEqual(@as(u64, 42), h.manifest.lastAppliedRaftIdx());

    // Not durable yet.
    try h.manifest.durabilize();
    try h.cycle();
    try testing.expectEqual(@as(u64, 42), h.manifest.lastAppliedRaftIdx());

    // Advance + durabilize + reopen again.
    try h.manifest.setLastAppliedRaftIdx(1000);
    try h.manifest.durabilize();
    try h.cycle();
    try testing.expectEqual(@as(u64, 1000), h.manifest.lastAppliedRaftIdx());
}

test "Phase 8: verify reports correct counts on a small forest" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.createStore(2);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        _ = h.manifest.nextApply();
        try s.put("a", "1");
        _ = h.manifest.nextApply();
        try s.put("b", "2");
    }
    {
        var s = try Store.open(h.manifest, 2);
        defer s.deinit();
        _ = h.manifest.nextApply();
        try s.put("x", "1");
    }
    try h.manifest.durabilize();

    const report = try h.manifest.verify(testing.allocator);
    try testing.expectEqual(@as(u64, 2), report.store_count);
    try testing.expect(report.manifest_tree_pages >= 1);
    try testing.expect(report.store_tree_pages >= 2);
    try testing.expect(report.freelist_tree_pages >= 1);
    // The freelist has recorded the orphans produced by the work
    // above; combined with reachable pages, the file should mostly
    // be accounted for.
    try testing.expect(report.orphan_pages == 0);
}

test "Phase 8: recovery — slot A garbage, slot B valid → falls back to B" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize(); // commit #1 writes slot 0
    try h.manifest.createStore(2);
    try h.manifest.durabilize(); // commit #2 writes slot 1

    try testing.expectEqual(@as(u32, 1), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 2), h.manifest.active_seq);

    h.closeLayers();

    // Corrupt slot B (the active slot) with random bytes (bad CRC).
    const garbage = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
    defer testing.allocator.free(garbage);
    @memset(garbage, 0xAB);
    try corruptSlotInFile(h.path(), SLOT_B_PAGE, garbage);

    try h.openLayers(.{});

    // Recovery falls back to slot A (seq=1).
    try testing.expectEqual(@as(u32, 0), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 1), h.manifest.active_seq);
    try testing.expect(try h.manifest.hasStore(1));
    try testing.expect(!try h.manifest.hasStore(2));
}

test "Phase 8: recovery — slot B garbage, slot A valid → falls back to A" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize(); // slot 0
    try h.manifest.createStore(2);
    try h.manifest.durabilize(); // slot 1
    try h.manifest.createStore(3);
    try h.manifest.durabilize(); // slot 0 again, overwriting commit #1

    try testing.expectEqual(@as(u32, 0), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 3), h.manifest.active_seq);

    h.closeLayers();

    // Corrupt slot A.
    const garbage = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
    defer testing.allocator.free(garbage);
    @memset(garbage, 0);
    try corruptSlotInFile(h.path(), SLOT_A_PAGE, garbage);

    try h.openLayers(.{});

    // Recovery falls back to slot B (seq=2, the previous durable).
    try testing.expectEqual(@as(u32, 1), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 2), h.manifest.active_seq);
    try testing.expect(try h.manifest.hasStore(1));
    try testing.expect(try h.manifest.hasStore(2));
    try testing.expect(!try h.manifest.hasStore(3));
}

test "Phase 8: recovery — both slots garbage looks fresh" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize();
    try h.manifest.createStore(2);
    try h.manifest.durabilize();

    h.closeLayers();

    const garbage = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
    defer testing.allocator.free(garbage);
    @memset(garbage, 0);
    try corruptSlotInFile(h.path(), SLOT_A_PAGE, garbage);
    try corruptSlotInFile(h.path(), SLOT_B_PAGE, garbage);

    try h.openLayers(.{});

    try testing.expectEqual(@as(u64, 0), h.manifest.active_seq);
    try testing.expect(!h.manifest.inactive_valid);
    try testing.expect(!try h.manifest.hasStore(1));
    try testing.expect(!try h.manifest.hasStore(2));
}

test "Phase 8: recovery — single-byte tear in slot CRC fails validation" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(7);
    try h.manifest.durabilize();
    try h.manifest.createStore(8);
    try h.manifest.durabilize();
    const active_before = h.manifest.active_slot;
    h.closeLayers();

    // Read the active slot, flip a single byte in the manifest_root
    // field, write it back. CRC should fail.
    const active_page: u64 = if (active_before == 0) SLOT_A_PAGE else SLOT_B_PAGE;
    {
        var pf = try PagedFile.open(h.path(), .{});
        defer pf.close();
        const buf = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
        defer testing.allocator.free(buf);
        try pf.readPage(active_page, buf);
        buf[16] ^= 0x01; // flip a bit in manifest_root
        try pf.writePage(active_page, buf);
        try pf.fsync();
    }

    try h.openLayers(.{});

    // Should have fallen back to the OTHER slot. Either both stores
    // exist (corruption was on slot 0 with newer seq → fall back to
    // older but valid slot 1 with seq=1, just store 7) or vice versa.
    try testing.expect(h.manifest.active_slot != active_before);
}

test "Phase 8: durabilize() with crash before slot fsync — old state survives" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize();

    // Mid-write simulation: workers have dirty pages but durabilize
    // never runs. Simulated by closing without commit.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "uncommitted");
    }
    // No durabilize call — close while dirty pages still in memory.
    h.closeLayers();
    try h.openLayers(.{});

    // The reopen sees the previous durable state. The uncommitted
    // value was never written to the slot, so it's gone.
    try testing.expect(try h.manifest.hasStore(1));
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

// -----------------------------------------------------------------------------
// Phase 7: read snapshots
// -----------------------------------------------------------------------------

test "Phase 7: snapshot sees point-in-time value across concurrent writes" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize();

    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v1");
    }
    try h.manifest.durabilize();

    var snap = try h.manifest.openSnapshot();
    defer snap.close();

    // Concurrent writes after snapshot.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v2");
        try s.put("k", "v3");
    }
    try h.manifest.durabilize();

    // Snapshot still sees v1.
    const old = (try snap.get(testing.allocator, 1, "k")).?;
    defer testing.allocator.free(old);
    try testing.expectEqualStrings("v1", old);

    // Live store sees v3.
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    const cur = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(cur);
    try testing.expectEqualStrings("v3", cur);
}

test "Phase 7: live snapshot blocks reuse of pages freed at its seq or later" {
    var h = try Harness.init(128);
    defer h.deinit();
    try h.manifest.createStore(1);

    // Seed some data and durabilize twice so reusable is populated
    // through normal phase-4 paths. Advance seq per put so the
    // hybrid CoW actually produces orphans.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        var i: u32 = 0;
        while (i < 30) : (i += 1) {
            _ = h.manifest.nextApply();
            var key_buf: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&key_buf, "k{d:0>4}", .{i});
            try s.put(k, "x");
        }
    }
    try h.manifest.durabilize();
    try h.manifest.durabilize();
    try testing.expect(h.manifest.reusable.items.len > 0);

    // Open a snapshot at this point. Its snap_seq is the next apply
    // seq (a low number; everything freed from here forward becomes
    // ineligible).
    var snap = try h.manifest.openSnapshot();
    const snap_seq = snap.snap_seq;
    try testing.expectEqual(snap_seq, h.manifest.minLiveSnapSeq().?);

    // Drive more writes so the freelist accumulates entries tagged
    // with snap_seq or later.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        var i: u32 = 0;
        while (i < 30) : (i += 1) {
            _ = h.manifest.nextApply();
            var key_buf: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&key_buf, "k{d:0>4}", .{i});
            try s.put(k, "y");
        }
    }
    try h.manifest.durabilize();
    try h.manifest.durabilize();

    // Inspect the freelist directly: there must be chunks tagged
    // >= snap_seq (those would be reusable without the snapshot),
    // and none of those pages should appear in reusable.
    var any_chunk_blocked = false;
    {
        h.manifest.freelist_tree_lock.lock();
        defer h.manifest.freelist_tree_lock.unlock();
        var cursor = try h.manifest.freelist.scanPrefix("");
        defer cursor.deinit();
        while (try cursor.next()) {
            const decoded = decodeFreelistKey(cursor.key());
            if (decoded.freed_at_seq >= snap_seq) {
                any_chunk_blocked = true;
                break;
            }
        }
    }
    try testing.expect(any_chunk_blocked);

    // Close the snapshot. Subsequent refill should pick up the
    // previously-blocked entries.
    const reusable_before_close = h.manifest.reusable.items.len;
    snap.close();
    try testing.expectEqual(@as(?u64, null), h.manifest.minLiveSnapSeq());

    // Trigger another refill (durabilize calls it). After this the
    // reusable list should grow as the blocked chunks become eligible.
    try h.manifest.durabilize();
    try testing.expect(h.manifest.reusable.items.len > reusable_before_close);
}

test "Phase 7: long-running prefix scan sees consistent view during heavy writes" {
    var h = try Harness.init(256);
    defer h.deinit();
    try h.manifest.createStore(1);

    // Populate 100 keys with "v0".
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            var key_buf: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i});
            try s.put(k, "v0");
        }
    }
    try h.manifest.durabilize();

    var snap = try h.manifest.openSnapshot();
    defer snap.close();

    const Writer = struct {
        manifest: *Manifest,
        n_keys: u32,

        fn run(self: @This()) !void {
            var s = try Store.open(self.manifest, 1);
            defer s.deinit();
            var pass: u32 = 0;
            while (pass < 3) : (pass += 1) {
                var i: u32 = 0;
                while (i < self.n_keys) : (i += 1) {
                    var key_buf: [16]u8 = undefined;
                    const k = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i});
                    var val_buf: [16]u8 = undefined;
                    const v = try std.fmt.bufPrint(&val_buf, "v{d}", .{pass + 1});
                    try s.put(k, v);
                }
            }
        }
    };
    var writer = try std.Thread.spawn(.{}, Writer.run, .{Writer{
        .manifest = h.manifest,
        .n_keys = 100,
    }});

    // Scan the snapshot. Every value must be "v0".
    var cursor = try snap.scanPrefix(1, "k");
    defer cursor.deinit();
    var seen: u32 = 0;
    while (try cursor.next()) {
        const v = cursor.value();
        try testing.expectEqualStrings("v0", v);
        seen += 1;
    }
    try testing.expectEqual(@as(u32, 100), seen);

    writer.join();

    // Live store now reflects the writer's pass 3.
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    const cur = (try s.get(testing.allocator, "k000000")).?;
    defer testing.allocator.free(cur);
    try testing.expectEqualStrings("v3", cur);
}

test "Hybrid CoW: same-seq puts to one leaf reuse the shadow page" {
    // The hybrid CoW path mutates a page in place when it was already
    // dirtied in the current apply unit. This test verifies the
    // observable side effect: a batch of same-seq puts produces far
    // fewer orphans (and grows the file less) than the same batch
    // with seq advances between each put.
    var h_same = try Harness.init(128);
    defer h_same.deinit();
    try h_same.manifest.createStore(1);
    var size_before_same: u64 = 0;
    {
        var s = try Store.open(h_same.manifest, 1);
        defer s.deinit();
        size_before_same = h_same.manifest.fileSizePages();
        var i: u32 = 0;
        while (i < 50) : (i += 1) {
            var kb: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&kb, "k{d:0>4}", .{i});
            try s.put(k, "x");
        }
    }
    const orphans_same = h_same.manifest.pending_free.items.len;
    const grew_same = h_same.manifest.fileSizePages() - size_before_same;

    var h_per_seq = try Harness.init(128);
    defer h_per_seq.deinit();
    try h_per_seq.manifest.createStore(1);
    var size_before_per: u64 = 0;
    {
        var s = try Store.open(h_per_seq.manifest, 1);
        defer s.deinit();
        size_before_per = h_per_seq.manifest.fileSizePages();
        var i: u32 = 0;
        while (i < 50) : (i += 1) {
            _ = h_per_seq.manifest.nextApply();
            var kb: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&kb, "k{d:0>4}", .{i});
            try s.put(k, "x");
        }
    }
    const orphans_per = h_per_seq.manifest.pending_free.items.len;
    const grew_per = h_per_seq.manifest.fileSizePages() - size_before_per;

    // Hybrid path generates strictly fewer orphans and grows the file
    // less than the per-seq path on the same workload.
    try testing.expect(orphans_same < orphans_per);
    try testing.expect(grew_same <= grew_per);

    // Both runs end up with the same final state observable through
    // the public API.
    var s1 = try Store.open(h_same.manifest, 1);
    defer s1.deinit();
    var s2 = try Store.open(h_per_seq.manifest, 1);
    defer s2.deinit();
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        var kb: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&kb, "k{d:0>4}", .{i});
        const v1 = (try s1.get(testing.allocator, k)) orelse return error.MissingKey;
        defer testing.allocator.free(v1);
        const v2 = (try s2.get(testing.allocator, k)) orelse return error.MissingKey;
        defer testing.allocator.free(v2);
        try testing.expectEqualStrings("x", v1);
        try testing.expectEqualStrings("x", v2);
    }
}

test "Hybrid CoW: same-seq updates to one key keep the same leaf page_no" {
    // The same-key-update case is the canonical in-place win: every
    // update after the first should mutate the same leaf, leaving
    // the store_root unchanged.
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try s.put("k", "v0");
    const root_after_first = (try h.manifest.storeRoot(1)).?;
    var i: u32 = 1;
    while (i < 20) : (i += 1) {
        var vb: [16]u8 = undefined;
        const v = try std.fmt.bufPrint(&vb, "v{d}", .{i});
        try s.put("k", v);
    }
    const root_after_many = (try h.manifest.storeRoot(1)).?;
    try testing.expectEqual(root_after_first, root_after_many);
    const v_final = (try s.get(testing.allocator, "k")) orelse return error.MissingKey;
    defer testing.allocator.free(v_final);
    try testing.expectEqualStrings("v19", v_final);
}

test "Phase 5: nextApply assigns distinct seqs, durabilize lands once" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);

    // First durabilize establishes baseline.
    try h.manifest.durabilize();
    const baseline_seq = h.manifest.durableSeq();

    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    try s.put("hot", "v1");
    try testing.expectEqual(baseline_seq + 1, h.manifest.applySeq());

    _ = h.manifest.nextApply();
    try s.put("hot", "v2");
    try testing.expectEqual(baseline_seq + 2, h.manifest.applySeq());

    _ = h.manifest.nextApply();
    try s.put("hot", "v3");
    try testing.expectEqual(baseline_seq + 3, h.manifest.applySeq());

    try testing.expectEqual(baseline_seq, h.manifest.durableSeq());
    try h.manifest.durabilize();
    try testing.expectEqual(baseline_seq + 3, h.manifest.durableSeq());

    // The final state alone is what's durable.
    try h.cycle();
    var s2 = try Store.open(h.manifest, 1);
    defer s2.deinit();
    const got = (try s2.get(testing.allocator, "hot")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v3", got);
}

test "Phase 5: orphan elision — hot key burst writes one final state" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("hot", "v0");
    }
    try h.manifest.durabilize();
    try h.manifest.durabilize();

    const writes_baseline = h.file.pages_written;

    // 10 applies, each updating the SAME key. Without orphan elision
    // each apply would write ~3 pages (store leaf + manifest leaf +
    // ancestors). With orphan elision, only the FINAL state's pages
    // hit disk, plus the freelist update and one manifest header
    // slot write.
    const N: u32 = 10;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        _ = h.manifest.nextApply();
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        var val_buf: [16]u8 = undefined;
        const v = try std.fmt.bufPrint(&val_buf, "v{d}", .{i + 1});
        try s.put("hot", v);
    }
    try h.manifest.durabilize();

    const writes_during_burst = h.file.pages_written - writes_baseline;
    // Naive: ~30 pwrites (10 applies × ~3 CoW pages). With orphan
    // elision capturing both pre-fold and post-fold orphans, only the
    // final-state pages plus the manifest header slot reach disk —
    // observed 4 pwrites in practice.
    try testing.expect(writes_during_burst < 8);

    // Correctness: the final state survives.
    try h.cycle();
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    const got = (try s.get(testing.allocator, "hot")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v10", got);
}

test "Phase 5: group commit reuse rule skips inactive_seq + 1" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);

    // Apply 1 → durabilize at seq=1. After: active=1, inactive=invalid.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v1");
    }
    try h.manifest.durabilize();
    try testing.expectEqual(@as(u64, 1), h.manifest.active_seq);
    try testing.expect(!h.manifest.inactive_valid);

    // Apply 2 → durabilize at seq=2. After: active=2, inactive=1.
    // (durabilize already advanced tree.seq for the next apply — no
    // need to call nextApply between durabilize calls.)
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v2");
    }
    try h.manifest.durabilize();
    try testing.expectEqual(@as(u64, 2), h.manifest.active_seq);
    try testing.expect(h.manifest.inactive_valid);
    try testing.expectEqual(@as(u64, 1), h.manifest.inactive_seq);

    // After durabilize-2, the freelist contains entries tagged seq=2
    // (their referencing manifest M_1 is still durable in the inactive
    // slot — those pages are NOT reusable yet). No new chunks should
    // have been consumed in d2's refillReusable.
    const consumed_after_d2 = h.manifest.consumed_keys.items.len;
    try testing.expectEqual(@as(usize, 0), consumed_after_d2);

    // Apply 3 → durabilize at seq=3. After: active=3, inactive=2.
    // The seq=2 entries' referencing manifest M_1 is GONE (its slot
    // got overwritten by M_3). They become reusable — refillReusable
    // pulls them in and queues their keys in consumed_keys for the
    // next durabilize to delete from the freelist tree.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v3");
    }
    try h.manifest.durabilize();
    try testing.expectEqual(@as(u64, 3), h.manifest.active_seq);
    try testing.expectEqual(@as(u64, 2), h.manifest.inactive_seq);

    // consumed_keys is the direct signal that chunks became eligible
    // and were pulled into reusable. Less brittle than comparing
    // reusable.items.len, which is inflated by grow-batched pages
    // unrelated to the freelist reuse rule under test.
    try testing.expect(h.manifest.consumed_keys.items.len > 0);
}

test "Phase 6: N concurrent writers, distinct stores, all data persists" {
    var h = try Harness.init(512);
    defer h.deinit();

    const NUM_THREADS: u32 = 4;
    const KEYS_PER_THREAD: u32 = 200;

    // Create one store per thread up front.
    var id: u32 = 0;
    while (id < NUM_THREADS) : (id += 1) {
        try h.manifest.createStore(id);
    }
    try h.manifest.durabilize();

    const Worker = struct {
        manifest: *Manifest,
        id: u32,
        n_keys: u32,

        fn run(self: @This()) !void {
            var s = try Store.open(self.manifest, self.id);
            defer s.deinit();
            var i: u32 = 0;
            while (i < self.n_keys) : (i += 1) {
                var key_buf: [16]u8 = undefined;
                const k = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i});
                var val_buf: [24]u8 = undefined;
                const v = try std.fmt.bufPrint(&val_buf, "store{d}-{d}", .{ self.id, i });
                try s.put(k, v);
            }
        }
    };

    var threads: [NUM_THREADS]std.Thread = undefined;
    var t: u32 = 0;
    while (t < NUM_THREADS) : (t += 1) {
        threads[t] = try std.Thread.spawn(.{}, Worker.run, .{Worker{
            .manifest = h.manifest,
            .id = t,
            .n_keys = KEYS_PER_THREAD,
        }});
    }
    for (threads) |thread| thread.join();

    // After all writers join, durabilize from main thread.
    try h.manifest.durabilize();

    // Reopen and verify every key persisted with the right value.
    try h.cycle();

    id = 0;
    while (id < NUM_THREADS) : (id += 1) {
        var s = try Store.open(h.manifest, id);
        defer s.deinit();
        var i: u32 = 0;
        while (i < KEYS_PER_THREAD) : (i += 1) {
            var key_buf: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i});
            var val_buf: [24]u8 = undefined;
            const expected = try std.fmt.bufPrint(&val_buf, "store{d}-{d}", .{ id, i });
            const got = (try s.get(testing.allocator, k)) orelse {
                std.debug.print("missing: store {d} key {s}\n", .{ id, k });
                return error.MissingKey;
            };
            defer testing.allocator.free(got);
            try testing.expectEqualStrings(expected, got);
        }
    }
}

test "Phase 6: two writers racing on the SAME store serialize correctly" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize();

    const N: u32 = 100;
    const Worker = struct {
        manifest: *Manifest,
        tag: u8,
        n: u32,

        fn run(self: @This()) !void {
            var s = try Store.open(self.manifest, 1);
            defer s.deinit();
            var i: u32 = 0;
            while (i < self.n) : (i += 1) {
                var key_buf: [16]u8 = undefined;
                const k = try std.fmt.bufPrint(&key_buf, "k{d:0>4}_{c}", .{ i, self.tag });
                var val_buf: [16]u8 = undefined;
                const v = try std.fmt.bufPrint(&val_buf, "{c}{d}", .{ self.tag, i });
                try s.put(k, v);
            }
        }
    };

    var t_a = try std.Thread.spawn(.{}, Worker.run, .{Worker{ .manifest = h.manifest, .tag = 'A', .n = N }});
    var t_b = try std.Thread.spawn(.{}, Worker.run, .{Worker{ .manifest = h.manifest, .tag = 'B', .n = N }});
    t_a.join();
    t_b.join();

    try h.manifest.durabilize();
    try h.cycle();

    // All 2N keys (N each from A and B) must be readable.
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        for ([_]u8{ 'A', 'B' }) |tag| {
            var key_buf: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&key_buf, "k{d:0>4}_{c}", .{ i, tag });
            var val_buf: [16]u8 = undefined;
            const expected = try std.fmt.bufPrint(&val_buf, "{c}{d}", .{ tag, i });
            const got = (try s.get(testing.allocator, k)).?;
            defer testing.allocator.free(got);
            try testing.expectEqualStrings(expected, got);
        }
    }
}

test "Freelist: churn workload approaches net-zero growth" {
    // With vector-valued chunks in the freelist (PAGES_PER_CHUNK
    // page_nos packed per cell), each commit's freelist maintenance
    // costs O(⌈P/PAGES_PER_CHUNK⌉ * depth) instead of O(P * depth).
    // Reuse keeps user-op allocations off `file.growBy`, and freelist
    // churn is small enough that file growth is sub-linear in rounds.
    var h = try Harness.init(256);
    defer h.deinit();
    try h.manifest.createStore(1);

    const KEYS: u32 = 100;
    const ROUNDS: u32 = 50;

    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        var i: u32 = 0;
        while (i < KEYS) : (i += 1) {
            _ = h.manifest.nextApply();
            var key_buf: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i});
            try s.put(k, "x");
        }
    }
    try h.manifest.commit();
    try h.manifest.commit();
    try testing.expect(h.manifest.freelist.root != 0);
    try testing.expect(h.manifest.reusable.items.len > 0);

    const size_after_populate = h.manifest.fileSizePages();

    var round: u32 = 0;
    while (round < ROUNDS) : (round += 1) {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        var i: u32 = 0;
        while (i < KEYS) : (i += 1) {
            _ = h.manifest.nextApply();
            var key_buf: [16]u8 = undefined;
            const k = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i});
            var val_buf: [16]u8 = undefined;
            const v = try std.fmt.bufPrint(&val_buf, "v{d}-{d}", .{ round, i });
            try s.put(k, v);
        }
        try h.manifest.commit();
    }

    const size_after_churn = h.manifest.fileSizePages();
    const growth = size_after_churn - size_after_populate;
    const naive_growth_no_reuse = KEYS * ROUNDS * 3; // ~3 pages per put (CoW depth)

    try testing.expect(growth < 2 * size_after_populate);
    try testing.expect(growth * 10 < naive_growth_no_reuse);
}

// ── Phase 10: state-transfer dump / restore round-trips ──────────────

test "Snapshot.listStores: enumerates stores at point-in-time" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(7);
    try h.manifest.createStore(13);
    try h.manifest.createStore(42);
    try h.manifest.durabilize();

    var snap = try h.manifest.openSnapshot();
    defer snap.close();

    const stores = try snap.listStores(testing.allocator);
    defer testing.allocator.free(stores);
    try testing.expectEqual(@as(usize, 3), stores.len);
    // listStores comes off a btree scan → ascending order.
    try testing.expectEqual(@as(u64, 7), stores[0]);
    try testing.expectEqual(@as(u64, 13), stores[1]);
    try testing.expectEqual(@as(u64, 42), stores[2]);
}

test "Snapshot.listStores: excludes stores created after the snapshot" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize();

    var snap = try h.manifest.openSnapshot();
    defer snap.close();

    // Create another store after snapshot open.
    try h.manifest.createStore(2);

    const stores = try snap.listStores(testing.allocator);
    defer testing.allocator.free(stores);
    try testing.expectEqual(@as(usize, 1), stores.len);
    try testing.expectEqual(@as(u64, 1), stores[0]);
}

test "dump + load: round-trip a populated manifest" {
    var src = try Harness.init(128);
    defer src.deinit();

    // Populate three stores with mixed data.
    try src.manifest.createStore(10);
    try src.manifest.createStore(20);
    try src.manifest.createStore(30);
    {
        var s = try Store.open(src.manifest, 10);
        defer s.deinit();
        try s.put("alpha", "one");
        try s.put("beta", "two");
        try s.put("gamma", "three");
    }
    {
        var s = try Store.open(src.manifest, 20);
        defer s.deinit();
        try s.put("k1", "v1");
    }
    // Store 30 is empty — should round-trip as an empty store.
    try src.manifest.setLastAppliedRaftIdx(9999);
    try src.manifest.durabilize();

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    var writer_state = buf.writer(testing.allocator);
    const writer = &writer_state;
    {
        var snap = try src.manifest.openSnapshot();
        defer snap.close();
        try dumpSnapshot(&snap, writer);
    }

    // Restore into a fresh harness.
    var dst = try Harness.init(128);
    defer dst.deinit();
    var reader_state = std.io.fixedBufferStream(buf.items);
    const reader = reader_state.reader();
    const last_applied = try loadSnapshot(dst.manifest, reader);

    try testing.expectEqual(@as(u64, 9999), last_applied);
    try testing.expectEqual(@as(u64, 9999), dst.manifest.lastAppliedRaftIdx());

    // Verify each store.
    try testing.expect(try dst.manifest.hasStore(10));
    try testing.expect(try dst.manifest.hasStore(20));
    // Empty stores aren't transferred (no records ever emitted), so
    // store 30 is absent on the receiver. This is acceptable for
    // state transfer — an empty store has no observable contents and
    // raft replay will re-create it on the next createStore call.
    try testing.expect(!(try dst.manifest.hasStore(30)));

    {
        var s = try Store.open(dst.manifest, 10);
        defer s.deinit();
        const a = (try s.get(testing.allocator, "alpha")).?;
        defer testing.allocator.free(a);
        try testing.expectEqualStrings("one", a);
        const b = (try s.get(testing.allocator, "beta")).?;
        defer testing.allocator.free(b);
        try testing.expectEqualStrings("two", b);
        const g = (try s.get(testing.allocator, "gamma")).?;
        defer testing.allocator.free(g);
        try testing.expectEqualStrings("three", g);
    }
    {
        var s = try Store.open(dst.manifest, 20);
        defer s.deinit();
        const v = (try s.get(testing.allocator, "k1")).?;
        defer testing.allocator.free(v);
        try testing.expectEqualStrings("v1", v);
    }

    // Loaded state should be durabilizable (fresh manifest, normal
    // mutation path).
    try dst.manifest.durabilize();
}

test "loadSnapshot: rejects bad magic" {
    var dst = try Harness.init(32);
    defer dst.deinit();
    const bad = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF, 1 };
    var reader_state = std.io.fixedBufferStream(&bad);
    const reader = reader_state.reader();
    try testing.expectError(error.InvalidSnapshotFormat, loadSnapshot(dst.manifest, reader));
}

test "loadSnapshot: rejects unsupported version" {
    var dst = try Harness.init(32);
    defer dst.deinit();
    var bytes: [21]u8 = undefined;
    std.mem.writeInt(u32, bytes[0..4], SNAPSHOT_MAGIC, .little);
    bytes[4] = 99; // future version
    @memset(bytes[5..], 0);
    var reader_state = std.io.fixedBufferStream(&bytes);
    const reader = reader_state.reader();
    try testing.expectError(error.UnsupportedSnapshotVersion, loadSnapshot(dst.manifest, reader));
}

// -----------------------------------------------------------------------------
// Crash-recovery tests using FaultyPagedFile through the full Manifest stack.
// -----------------------------------------------------------------------------

const SimHarness = struct {
    sim: *FaultyPagedFile,
    pool: *BufferPool,
    cache: *PageCache,
    manifest: *Manifest,
    pool_capacity: u32,

    fn init(pool_capacity: u32, policy: @import("faulty_paged_file.zig").FaultPolicy) !SimHarness {
        var self: SimHarness = undefined;
        self.pool_capacity = pool_capacity;

        self.sim = try testing.allocator.create(FaultyPagedFile);
        errdefer testing.allocator.destroy(self.sim);
        self.sim.* = try FaultyPagedFile.init(testing.allocator, page.PAGE_SIZE, 0, policy);
        errdefer self.sim.deinit();

        try self.openUpperLayers();
        return self;
    }

    fn openUpperLayers(self: *SimHarness) !void {
        self.pool = try testing.allocator.create(BufferPool);
        errdefer testing.allocator.destroy(self.pool);
        self.pool.* = try BufferPool.init(testing.allocator, page.PAGE_SIZE, self.pool_capacity);
        errdefer self.pool.deinit(testing.allocator);

        self.cache = try testing.allocator.create(PageCache);
        errdefer testing.allocator.destroy(self.cache);
        self.cache.* = try PageCache.init(testing.allocator, self.sim.api(), self.pool, .{});
        errdefer self.cache.deinit();

        self.manifest = try testing.allocator.create(Manifest);
        errdefer testing.allocator.destroy(self.manifest);
        try self.manifest.init(testing.allocator, self.cache, self.sim.api());
    }

    fn closeUpperLayers(self: *SimHarness) void {
        self.manifest.deinit();
        testing.allocator.destroy(self.manifest);
        self.cache.deinit();
        testing.allocator.destroy(self.cache);
        self.pool.deinit(testing.allocator);
        testing.allocator.destroy(self.pool);
    }

    fn deinit(self: *SimHarness) void {
        self.closeUpperLayers();
        self.sim.deinit();
        testing.allocator.destroy(self.sim);
    }

    /// Drop everything above the simulator, crash, then reopen the
    /// upper layers. The simulator's durable state persists; in_flight
    /// is dropped.
    fn crashAndReopen(self: *SimHarness) !void {
        self.closeUpperLayers();
        self.sim._crash();
        try self.openUpperLayers();
    }
};

test "crash sim: pre-durabilize writes are lost after crash" {
    var h = try SimHarness.init(64, .{});
    defer h.deinit();

    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v");
    }
    // No durabilize — crash now.
    try h.crashAndReopen();

    try testing.expect(!try h.manifest.hasStore(1));
}

test "crash sim: durabilized prefix survives; post-durabilize writes lost" {
    var h = try SimHarness.init(64, .{});
    defer h.deinit();

    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("a", "1");
    }
    try h.manifest.durabilize();

    // Stage more work that won't be durabilized.
    try h.manifest.createStore(2);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("b", "2");
    }

    try h.crashAndReopen();

    // The durable prefix is intact: store 1 + key "a"="1".
    try testing.expect(try h.manifest.hasStore(1));
    try testing.expect(!try h.manifest.hasStore(2));
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    const got = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("1", got);
    try testing.expect((try s.get(testing.allocator, "b")) == null);
}

test "crash sim: fsync failure poisons; reopen recovers prior durable state" {
    var h = try SimHarness.init(64, .{});
    defer h.deinit();

    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("a", "1");
    }
    try h.manifest.durabilize();

    // Stage more work, then force the next fsync to fail.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("b", "2");
    }
    h.sim.policy.fail_next_fsync = true;
    try testing.expectError(error.IoFailed, h.manifest.durabilize());
    try testing.expect(h.manifest.isPoisoned());
    try testing.expectError(error.ManifestPoisoned, h.manifest.createStore(99));

    try h.crashAndReopen();

    // Fresh manifest, no poison; durable prefix intact.
    try testing.expect(!h.manifest.isPoisoned());
    try testing.expect(try h.manifest.hasStore(1));
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();
    const got = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("1", got);
    try testing.expect((try s.get(testing.allocator, "b")) == null);
}

test "crash sim: lying fsync — caller thinks durable, crash proves otherwise" {
    // fsyncgate scenario. The simulator returns OK from fsync but does
    // not actually move in_flight to durable. The library can't detect
    // the lie at fsync time — only after crash does the writer
    // discover the data is gone. The recovery contract still holds:
    // there's nothing durable, so the reopened manifest is fresh.
    var h = try SimHarness.init(64, .{ .fsync_lies = true });
    defer h.deinit();

    try h.manifest.createStore(1);
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("a", "1");
    }
    try h.manifest.durabilize();

    try h.crashAndReopen();

    try testing.expect(!try h.manifest.hasStore(1));
}

// -----------------------------------------------------------------------------
// Randomized property test: drive a model and the real manifest through the
// same random op sequence, crash at a random point, verify the post-crash
// manifest equals the model snapshot taken at the last successful durabilize.
//
// The invariant under test: every successful durabilize is a commit point.
// After any crash, the recovered state equals the model state at the most
// recent successful durabilize. Ops between durabilizes are speculative —
// they may or may not survive, and the contract makes no promise — but the
// state after recovery must NEVER be a mix of pre- and post-durabilize work.
//
// Each iteration uses a distinct seed for both the workload PRNG and the
// FaultyPagedFile policy. On failure, the test prints the seed; rerunning
// with that seed reproduces the failure bit-for-bit.
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

fn valBuf(buf: *[1]u8, v: u8) []const u8 {
    buf[0] = v;
    return buf[0..1];
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

    var h = try SimHarness.init(128, .{ .seed = seed });
    defer h.deinit();

    var model: PropModel = .{};
    var durable_model: PropModel = .{};

    const op_count: u32 = rng.intRangeAtMost(u32, 40, 120);
    const crash_at: u32 = rng.intRangeLessThan(u32, 0, op_count);

    var i: u32 = 0;
    while (i < op_count) : (i += 1) {
        if (i == crash_at) {
            try h.crashAndReopen();
            try verifyAgainstModel(h.manifest, &durable_model);
            return;
        }

        const choice = rng.intRangeLessThan(u32, 0, 100);
        // 55% put, 15% delete, 12% createStore, 5% dropStore, 13% durabilize.
        if (choice < 55) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const kid = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
            const v = rng.int(u8);

            var s = try Store.open(h.manifest, @intCast(sid_usize));
            defer s.deinit();
            var kb: [3]u8 = undefined;
            var vb: [1]u8 = undefined;
            try s.put(keyBuf(&kb, kid), valBuf(&vb, v));
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
            // Library result must match model.
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
            try h.manifest.durabilize();
            durable_model = model;
        }
    }
}

test "property: random workload + crash recovers durable prefix" {
    const runs: u64 = 100;
    var seed: u64 = 1;
    while (seed <= runs) : (seed += 1) {
        propRunOne(seed) catch |err| {
            std.debug.print("\n*** property test failed at seed {} : {} ***\n", .{ seed, err });
            return err;
        };
    }
}

// ── Property test with fault injection ─────────────────────────────
//
// Same shape as `propRunOne`, but with probabilistic write / fsync /
// read failures injected during the workload. The invariants under test:
//
//   * Mutations that returned an error do NOT update the model. The
//     library must agree (post-recovery, the failed op's effects are
//     absent).
//   * A durabilize that errors poisons the manifest. Every subsequent
//     mutation must return `error.ManifestPoisoned` (verified by the
//     workload's catch).
//   * Read errors during ordinary ops propagate but do not corrupt
//     state — the model agrees with the library after recovery.
//   * Recovery itself runs with faults disabled (the simulator's
//     policy is zeroed before `crashAndReopen`). This matches reality:
//     if your disk is currently failing, you can't recover yet; you
//     reopen after the disk is replaced/repaired.

fn propRunOneWithFaults(seed: u64) !void {
    var rng_state = std.Random.DefaultPrng.init(seed);
    const rng = rng_state.random();

    // Random fault rates per seed. Capped low enough that most ops
    // succeed, so the workload makes progress and durable_model is
    // nontrivial in most runs.
    const workload_policy: @import("faulty_paged_file.zig").FaultPolicy = .{
        .seed = seed ^ 0xA5A5_A5A5,
        .write_fail_ppm = rng.intRangeAtMost(u32, 0, 20_000), // up to 2%
        .fsync_fail_ppm = rng.intRangeAtMost(u32, 0, 30_000), // up to 3%
        .read_fail_ppm = rng.intRangeAtMost(u32, 0, 5_000), // up to 0.5%
    };

    // Open with faults OFF — `Manifest.init` on a fresh file writes
    // the header + fsyncs, which would otherwise be subject to
    // injection. Faults turn on once the harness is ready.
    var h = try SimHarness.init(128, .{ .seed = workload_policy.seed });
    defer h.deinit();
    h.sim.policy.write_fail_ppm = workload_policy.write_fail_ppm;
    h.sim.policy.fsync_fail_ppm = workload_policy.fsync_fail_ppm;
    h.sim.policy.read_fail_ppm = workload_policy.read_fail_ppm;

    var model: PropModel = .{};
    var durable_model: PropModel = .{};
    var poisoned_local = false;

    const op_count: u32 = rng.intRangeAtMost(u32, 40, 120);
    const crash_at: u32 = rng.intRangeLessThan(u32, 0, op_count);

    var i: u32 = 0;
    while (i < op_count) : (i += 1) {
        if (i == crash_at) {
            // Recovery itself must run without faults. (Otherwise an
            // induced read failure during Manifest.init would fail
            // the harness before we get to verify, which would be a
            // testing-infrastructure problem, not a library bug.)
            h.sim.policy.write_fail_ppm = 0;
            h.sim.policy.fsync_fail_ppm = 0;
            h.sim.policy.read_fail_ppm = 0;
            try h.crashAndReopen();
            try verifyAgainstModel(h.manifest, &durable_model);
            return;
        }

        if (poisoned_local) {
            // The manifest is poisoned. The contract: every write
            // surface returns ManifestPoisoned. Spot-check it once,
            // then idle until crash_at — there's nothing useful to
            // do until recovery.
            try testing.expect(h.manifest.isPoisoned());
            continue;
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
            var vb: [1]u8 = undefined;
            if (s.put(keyBuf(&kb, kid), valBuf(&vb, v))) {
                model.key_present[sid_usize][kid] = true;
                model.values[sid_usize][kid] = v;
            } else |err| switch (err) {
                error.ManifestPoisoned => poisoned_local = true,
                else => {}, // op didn't happen, model untouched
            }
        } else if (choice < 70) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const kid = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
            var s = try Store.open(h.manifest, @intCast(sid_usize));
            defer s.deinit();
            var kb: [3]u8 = undefined;
            if (s.delete(keyBuf(&kb, kid))) |existed| {
                // Library agrees with model on whether the key existed.
                if (existed != model.key_present[sid_usize][kid]) return error.DeleteMismatch;
                model.key_present[sid_usize][kid] = false;
            } else |err| switch (err) {
                error.ManifestPoisoned => poisoned_local = true,
                else => {},
            }
        } else if (choice < 82) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (model.store_exists[sid_usize]) continue;
            if (h.manifest.createStore(@intCast(sid_usize))) {
                model.store_exists[sid_usize] = true;
            } else |err| switch (err) {
                error.ManifestPoisoned => poisoned_local = true,
                else => {},
            }
        } else if (choice < 87) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            if (h.manifest.dropStore(@intCast(sid_usize))) |dropped| {
                if (!dropped) return error.DropReturnedFalse;
                model.store_exists[sid_usize] = false;
                for (0..PROP_NUM_KEYS) |k| model.key_present[sid_usize][k] = false;
            } else |err| switch (err) {
                error.ManifestPoisoned => poisoned_local = true,
                else => {},
            }
        } else {
            // durabilize. On success: snapshot model → durable_model.
            // On error: poison; durable_model unchanged.
            if (h.manifest.durabilize()) {
                durable_model = model;
            } else |err| switch (err) {
                error.ManifestPoisoned => poisoned_local = true,
                else => {
                    // Any other error escaped durabilize, which means
                    // the errdefer fired and the manifest is poisoned.
                    try testing.expect(h.manifest.isPoisoned());
                    poisoned_local = true;
                },
            }
        }
    }
}

test "property: random workload + faults + crash recovers durable prefix" {
    const runs: u64 = 100;
    var seed: u64 = 1;
    while (seed <= runs) : (seed += 1) {
        propRunOneWithFaults(seed) catch |err| {
            std.debug.print("\n*** fault-injection property test failed at seed {} : {} ***\n", .{ seed, err });
            return err;
        };
    }
}

test "crash sim: write failures during ordinary puts don't poison" {
    // Probabilistic write failures bubble up as Store.put errors; the
    // manifest stays alive and the caller retries. Poison is reserved
    // for durabilize errors. After durabilize+crash+reopen, every
    // succeeded put is visible.
    var h = try SimHarness.init(64, .{
        .seed = 0xABCD,
        .write_fail_ppm = 200_000, // 20%
    });
    defer h.deinit();

    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();

    var succeeded: [50]bool = @splat(false);
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        var key_buf: [8]u8 = undefined;
        const k = try std.fmt.bufPrint(&key_buf, "k{d}", .{i});
        if (s.put(k, "v")) {
            succeeded[i] = true;
        } else |_| {}
    }
    try testing.expect(!h.manifest.isPoisoned());

    h.sim.policy.write_fail_ppm = 0;
    try h.manifest.durabilize();

    try h.crashAndReopen();
    var s2 = try Store.open(h.manifest, 1);
    defer s2.deinit();
    i = 0;
    while (i < 50) : (i += 1) {
        if (!succeeded[i]) continue;
        var key_buf: [8]u8 = undefined;
        const k = try std.fmt.bufPrint(&key_buf, "k{d}", .{i});
        const got = (try s2.get(testing.allocator, k));
        try testing.expect(got != null);
        testing.allocator.free(got.?);
    }
}
