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
const PagedFile = @import("paged_file.zig").PagedFile;
const PageCache = @import("page_cache.zig").PageCache;

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
    _reserved: [page.PAGE_SIZE - 36]u8 align(1) = @splat(0),
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

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    cache: *PageCache,
    file: *PagedFile,

    tree: Tree, // manifest B-tree
    freelist: Tree, // free-page B-tree

    /// Which slot (0 or 1) is currently "active" — holds the
    /// highest-seq valid manifest.
    active_slot: u32,
    /// Sequence of the active slot. Highest durable seq.
    active_seq: u64,
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
    ///   store_lock(id)     — held by one writer to that store at a
    ///                        time; uncontended across distinct stores
    ///   tree_lock          — manifest tree mutations + reads
    ///   freelist_tree_lock — freelist tree mutations + reads (only
    ///                        held during durabilize)
    ///   alloc_lock         — reusable/pending_free/consumed_keys/
    ///                        file.growBy
    /// `cache.lock` (interior, owned by PageCache) is acquired
    /// briefly *inside* allocImpl/freeImpl-free regions.
    tree_lock: std.Thread.Mutex = .{},
    freelist_tree_lock: std.Thread.Mutex = .{},
    alloc_lock: std.Thread.Mutex = .{},

    /// Per-store write locks, keyed by store_id. Allocated lazily on
    /// first Store.put for that id; not freed (small overhead per
    /// active store).
    store_locks: std.AutoHashMapUnmanaged(u64, *std.Thread.Mutex),
    store_locks_lock: std.Thread.Mutex = .{},

    pub fn pageAllocator(self: *Manifest) PageAllocator {
        return .{ .ctx = self, .vtable = &alloc_vtable };
    }

    const alloc_vtable: PageAllocator.VTable = .{
        .alloc = allocImpl,
        .free = freeImpl,
    };

    fn allocImpl(ctx: *anyopaque) anyerror!u64 {
        const self: *Manifest = @ptrCast(@alignCast(ctx));
        self.alloc_lock.lock();
        defer self.alloc_lock.unlock();
        if (self.reusable.pop()) |p| return p;
        return try self.file.growBy(1);
    }

    fn freeImpl(ctx: *anyopaque, page_no: u64, freed_at_seq: u64) anyerror!void {
        const self: *Manifest = @ptrCast(@alignCast(ctx));
        self.alloc_lock.lock();
        defer self.alloc_lock.unlock();
        try self.pending_free.append(self.allocator, .{
            .page_no = page_no,
            .freed_at_seq = freed_at_seq,
        });
    }

    /// In-place init so `self` has a stable address (the page allocator
    /// captures `&self` as ctx).
    pub fn init(self: *Manifest, allocator: std.mem.Allocator, cache: *PageCache, file: *PagedFile) !void {
        self.allocator = allocator;
        self.cache = cache;
        self.file = file;
        self.reusable = .empty;
        self.consumed_keys = .empty;
        self.pending_free = .empty;
        self.store_locks = .empty;
        self.tree_lock = .{};
        self.freelist_tree_lock = .{};
        self.alloc_lock = .{};
        self.store_locks_lock = .{};

        while (file.pageCount() < FIRST_DATA_PAGE) {
            _ = try file.growBy(1);
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
        if (va and vb) {
            if (slot_a.sequence >= slot_b.sequence) {
                active_slot = 0;
                active_seq = slot_a.sequence;
                inactive_seq = slot_b.sequence;
                manifest_root = slot_a.manifest_root;
                freelist_root = slot_a.freelist_root;
            } else {
                active_slot = 1;
                active_seq = slot_b.sequence;
                inactive_seq = slot_a.sequence;
                manifest_root = slot_b.manifest_root;
                freelist_root = slot_b.freelist_root;
            }
            inactive_valid = true;
        } else if (va) {
            active_slot = 0;
            active_seq = slot_a.sequence;
            manifest_root = slot_a.manifest_root;
            freelist_root = slot_a.freelist_root;
        } else if (vb) {
            active_slot = 1;
            active_seq = slot_b.sequence;
            manifest_root = slot_b.manifest_root;
            freelist_root = slot_b.freelist_root;
        }
        self.active_slot = active_slot;
        self.active_seq = active_seq;
        self.inactive_seq = inactive_seq;
        self.inactive_valid = inactive_valid;
        self.sequence = active_seq;

        const page_alloc = self.pageAllocator();
        self.tree = try Tree.init(allocator, cache, file, page_alloc);
        self.tree.root = manifest_root;
        self.tree.seq = active_seq + 1;
        self.freelist = try Tree.init(allocator, cache, file, page_alloc);
        self.freelist.root = freelist_root;
        self.freelist.seq = active_seq + 1;

        try self.refillReusable();
    }

    /// Start a new apply unit. Subsequent mutations tag dirty pages
    /// with the returned sequence. Multiple applies between durabilize
    /// calls accumulate in-memory state tagged with distinct seqs;
    /// orphan elision skips intermediate page versions at durabilize.
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

    pub fn deinit(self: *Manifest) void {
        self.reusable.deinit(self.allocator);
        self.consumed_keys.deinit(self.allocator);
        self.pending_free.deinit(self.allocator);
        var it = self.store_locks.valueIterator();
        while (it.next()) |m| self.allocator.destroy(m.*);
        self.store_locks.deinit(self.allocator);
        self.* = undefined;
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
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        return try self.hasStoreLocked(id);
    }

    fn hasStoreLocked(self: *Manifest, id: u64) !bool {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        const v = try self.tree.get(self.allocator, k);
        if (v) |bytes| {
            self.allocator.free(bytes);
            return true;
        }
        return false;
    }

    pub fn storeRoot(self: *Manifest, id: u64) !?u64 {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        return try self.storeRootLocked(id);
    }

    fn storeRootLocked(self: *Manifest, id: u64) !?u64 {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        const v = try self.tree.get(self.allocator, k);
        if (v) |bytes| {
            defer self.allocator.free(bytes);
            return decodeRoot(bytes);
        }
        return null;
    }

    pub fn createStore(self: *Manifest, id: u64) !void {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        if (try self.hasStoreLocked(id)) return error.StoreAlreadyExists;
        try self.setStoreRootLocked(id, 0);
    }

    pub fn dropStore(self: *Manifest, id: u64) !bool {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        return try self.tree.delete(k);
    }

    pub fn setStoreRoot(self: *Manifest, id: u64, root: u64) !void {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        try self.setStoreRootLocked(id, root);
    }

    fn setStoreRootLocked(self: *Manifest, id: u64, root: u64) !void {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        var val_buf: [STORE_VAL_LEN]u8 = undefined;
        const v = encodeRoot(root, &val_buf);
        try self.tree.put(k, v);
    }

    pub fn listStores(self: *Manifest, allocator: std.mem.Allocator) ![]u64 {
        self.tree_lock.lock();
        defer self.tree_lock.unlock();
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        var cursor = try self.tree.scanPrefix("");
        defer cursor.deinit();
        while (try cursor.next()) {
            try list.append(allocator, decodeStoreId(cursor.key()));
        }
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
    /// likewise free of `tree_lock`. `alloc_lock` is taken in short
    /// bursts only.
    pub fn durabilize(self: *Manifest) !void {
        // Capture K under tree_lock so a concurrent nextApply doesn't
        // race the read.
        self.tree_lock.lock();
        const K = self.tree.seq;
        self.tree_lock.unlock();
        if (K <= self.active_seq) return;

        // 1. Snapshot pending_free + consumed_keys (brief).
        self.alloc_lock.lock();
        const pf = try self.pending_free.toOwnedSlice(self.allocator);
        const ck = try self.consumed_keys.toOwnedSlice(self.allocator);
        self.alloc_lock.unlock();
        defer self.allocator.free(pf);
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
        //    internally take alloc_lock (via allocImpl/freeImpl).
        self.freelist_tree_lock.lock();
        try self.foldPendingFree(pf);
        for (ck) |key| _ = try self.freelist.delete(&key);
        self.freelist_tree_lock.unlock();

        // 4. Extend orphan set with pages freed during fold/delete.
        self.alloc_lock.lock();
        for (self.pending_free.items) |fp| {
            if (fp.freed_at_seq <= K) {
                try orphans.put(self.allocator, fp.page_no, {});
            }
        }
        self.alloc_lock.unlock();

        // 5. Flush dirty pages tagged seq <= K, skipping orphans.
        try self.cache.flushUpToSkipping(K, &orphans);
        try self.file.fsync();

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
        };
        slot.computeChecksum();
        try self.file.writePage(slot_page, &slot_buf);
        try self.file.fsync();

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

        // freelist_tree_lock serializes against any other freelist tree
        // mutation (only durabilize mutates it, but defensive). The
        // alloc_lock guards the reusable + consumed_keys lists.
        self.freelist_tree_lock.lock();
        defer self.freelist_tree_lock.unlock();
        self.alloc_lock.lock();
        defer self.alloc_lock.unlock();

        if (self.reusable.items.len >= REUSABLE_BATCH) return;
        var cursor = try self.freelist.scanPrefix("");
        defer cursor.deinit();
        while (try cursor.next()) {
            const decoded = decodeFreelistKey(cursor.key());
            if (self.inactive_valid and decoded.freed_at_seq == self.inactive_seq + 1) continue;
            if (decoded.freed_at_seq == self.active_seq + 1) continue;
            // Safe to consume.
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

        // Phase 2: store-tree CoW without manifest locks. allocImpl /
        // freeImpl take alloc_lock briefly; cache takes cache_lock
        // briefly. Writers on different stores can run this part
        // concurrently.
        self.tree.root = root;
        self.tree.seq = seq;
        try self.tree.put(key, value);

        // Phase 3: publish the new store root via the manifest tree.
        try self.manifest.setStoreRoot(self.id, self.tree.root);
    }

    pub fn delete(self: *Store, key: []const u8) !bool {
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
        self.cache.* = try PageCache.init(testing.allocator, self.file, self.pool);

        self.manifest = try testing.allocator.create(Manifest);
        errdefer testing.allocator.destroy(self.manifest);
        try self.manifest.init(testing.allocator, self.cache, self.file);
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

    try s.put("a", "1");
    try s.put("b", "2");
    try s.put("c", "3");
    // Each put produces CoW orphans.
    try testing.expect(h.manifest.pending_free.items.len > 0);

    try h.manifest.commit();
    // After commit, pending_free is drained. Some entries may have been
    // re-populated by the freelist's own CoW operations, but the count
    // should be much smaller than before.
    try testing.expect(h.manifest.pending_free.items.len < 10);
}

test "Freelist: reuse after two-commit lag" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(h.manifest, 1);
    defer s.deinit();

    // Drive churn so freelist accumulates entries.
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
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
    // slot — those pages are NOT reusable yet). Reusable should be
    // empty or contain only pages drawn from earlier chunks.
    const reusable_after_d2 = h.manifest.reusable.items.len;

    // Apply 3 → durabilize at seq=3. After: active=3, inactive=2.
    // The seq=2 entries' referencing manifest M_1 is GONE (its slot
    // got overwritten by M_3). They become reusable.
    {
        var s = try Store.open(h.manifest, 1);
        defer s.deinit();
        try s.put("k", "v3");
    }
    try h.manifest.durabilize();
    try testing.expectEqual(@as(u64, 3), h.manifest.active_seq);
    try testing.expectEqual(@as(u64, 2), h.manifest.inactive_seq);

    // Now reusable should be populated (seq=2 chunks are safe; the
    // new dangerous seq is inactive_seq + 1 = 3, but most freelist
    // entries are tagged 2).
    try testing.expect(h.manifest.reusable.items.len > reusable_after_d2);
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
