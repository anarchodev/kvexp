//! In-process page cache: page_no → buffer index, with clock-based
//! eviction, pin/unpin, and sequence-tagged dirty tracking.
//!
//! This is the layer where "we own writeback" pays off. Dirty pages
//! sit in cache until an explicit `flushUpTo(seq)` is issued by the
//! caller (later: the apply/durabilize machinery at phase 5). The OS
//! never writes back on its own — `O_DIRECT` plus our explicit pwrite
//! is the only path to disk.
//!
//! ## Sharding
//!
//! The cache is split into N shards, each with its own mutex, hash
//! index, slot array, free-buffer list, and clock hand. A page_no
//! always lives in the same shard (`page_no % shard_count`). Workers
//! on disjoint page_nos contend on different mutexes — the cache no
//! longer has a global lock.
//!
//! Buffer ownership is fixed at init: each shard claims a contiguous
//! range of buffer indices from the global pool. Buffers never migrate
//! between shards.
//!
//! ## Lock-free eviction writeback
//!
//! When a worker needs a free buffer and the cache is full, the clock
//! walk finds a victim. If the victim is dirty, the slot transitions
//! to `.evicting`, the index entry is dropped, `dirty_seq` is cleared,
//! and the shard lock is **released**. `writePage` then runs without
//! holding the lock — other threads on the same shard can proceed.
//! The buffer's content stays valid during the write (no one can
//! reuse a slot that's `.evicting`). When the write completes, the
//! shard lock is re-acquired briefly to transition `.evicting → .empty`
//! and return the buffer to the caller.

const std = @import("std");
const paged_file = @import("paged_file.zig");
const PagedFile = paged_file.PagedFile;
const PagedFileApi = paged_file.PagedFileApi;
const PageWrite = paged_file.PageWrite;
const FileIoError = paged_file.IoError;
const bp = @import("buffer_pool.zig");
const BufferPool = bp.BufferPool;
const BufferIndex = bp.BufferIndex;
const page = @import("page.zig");

pub const PageRef = struct {
    cache: *PageCache,
    page_no: u64,
    buffer_idx: BufferIndex,

    pub fn buf(self: PageRef) []align(std.heap.page_size_min) u8 {
        return self.cache.pool.buf(self.buffer_idx);
    }

    pub fn markDirty(self: PageRef, seq: u64) void {
        const shard = self.cache.shardFor(self.page_no);
        shard.lock.lock();
        defer shard.lock.unlock();
        const slot = shard.slotForBuffer(self.buffer_idx);
        slot.dirty_seq = if (slot.dirty_seq) |existing| @max(existing, seq) else seq;
    }

    /// Return the slot's current dirty_seq, or null if the page has
    /// never been marked dirty (or was flushed by the last
    /// durabilize). Used by the B-tree CoW path to detect pages that
    /// have already been shadowed by the current txn — those can be
    /// mutated in place rather than re-CoW'd.
    pub fn dirtySeq(self: PageRef) ?u64 {
        const shard = self.cache.shardFor(self.page_no);
        shard.lock.lock();
        defer shard.lock.unlock();
        const slot = shard.slotForBuffer(self.buffer_idx);
        return slot.dirty_seq;
    }

    pub fn release(self: PageRef) void {
        const shard = self.cache.shardFor(self.page_no);
        shard.lock.lock();
        defer shard.lock.unlock();
        const slot = shard.slotForBuffer(self.buffer_idx);
        std.debug.assert(slot.pin_count > 0);
        slot.pin_count -= 1;
    }
};

const SlotState = enum {
    /// No page cached here; buffer is in `free_globals`.
    empty,
    /// Page is cached. Buffer content valid.
    occupied,
    /// Dirty page is being written back; buffer content valid but
    /// the slot is reserved for the evicting thread. Pin/eviction
    /// skip this state.
    evicting,
};

const Slot = struct {
    state: SlotState,
    page_no: u64,
    pin_count: u32,
    dirty_seq: ?u64,
    clock_referenced: bool,
};

const Shard = struct {
    lock: std.Thread.Mutex,
    slots: []Slot,
    capacity: u32,
    global_offset: BufferIndex,
    /// Pre-populated at init with this shard's global buffer indices.
    /// LIFO.
    free_globals: std.ArrayListUnmanaged(BufferIndex),
    index: std.AutoHashMapUnmanaged(u64, BufferIndex),
    clock_hand: u32,

    fn slotForBuffer(self: *Shard, global_idx: BufferIndex) *Slot {
        std.debug.assert(global_idx >= self.global_offset);
        std.debug.assert(global_idx < self.global_offset + self.capacity);
        return &self.slots[global_idx - self.global_offset];
    }
};

pub const PageCache = struct {
    allocator: std.mem.Allocator,
    file: PagedFileApi,
    pool: *BufferPool,
    shards: []Shard,

    pub const Options = struct {
        /// Number of shards. 0 = auto: `clamp(pool.capacity / 4, 1, 16)`.
        /// Each shard owns at least 1 buffer.
        shard_count: u32 = 0,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        file: PagedFileApi,
        pool: *BufferPool,
        options: Options,
    ) !PageCache {
        const requested = if (options.shard_count == 0)
            std.math.clamp(pool.capacity / 4, 1, 16)
        else
            options.shard_count;
        const sc: u32 = @min(requested, pool.capacity);

        const shards = try allocator.alloc(Shard, sc);
        errdefer allocator.free(shards);

        const per_shard = pool.capacity / sc;
        var built: u32 = 0;
        errdefer for (shards[0..built]) |*sh| destroyShard(allocator, sh);

        var i: u32 = 0;
        while (i < sc) : (i += 1) {
            const start: BufferIndex = @intCast(i * per_shard);
            const cap: u32 = if (i == sc - 1) pool.capacity - start else per_shard;
            const slots = try allocator.alloc(Slot, cap);
            errdefer allocator.free(slots);
            for (slots) |*s| s.* = .{
                .state = .empty,
                .page_no = 0,
                .pin_count = 0,
                .dirty_seq = null,
                .clock_referenced = false,
            };
            var free_list: std.ArrayListUnmanaged(BufferIndex) = .empty;
            errdefer free_list.deinit(allocator);
            try free_list.ensureTotalCapacity(allocator, cap);
            var b: BufferIndex = start + cap;
            while (b > start) {
                b -= 1;
                free_list.appendAssumeCapacity(b);
            }
            shards[i] = .{
                .lock = .{},
                .slots = slots,
                .capacity = cap,
                .global_offset = start,
                .free_globals = free_list,
                .index = .empty,
                .clock_hand = 0,
            };
            built += 1;
        }

        return .{
            .allocator = allocator,
            .file = file,
            .pool = pool,
            .shards = shards,
        };
    }

    fn destroyShard(allocator: std.mem.Allocator, shard: *Shard) void {
        shard.index.deinit(allocator);
        shard.free_globals.deinit(allocator);
        allocator.free(shard.slots);
    }

    pub fn deinit(self: *PageCache) void {
        for (self.shards) |*sh| destroyShard(self.allocator, sh);
        self.allocator.free(self.shards);
        self.* = undefined;
    }

    pub fn shardCount(self: *const PageCache) u32 {
        return @intCast(self.shards.len);
    }

    fn shardFor(self: *PageCache, page_no: u64) *Shard {
        return &self.shards[page_no % self.shards.len];
    }

    pub const PinError = error{ AllPagesPinned, OutOfMemory, PageChecksumMismatch, PageStructureInvalid } || FileIoError;

    /// Pin a page in cache. On miss, reads from disk first. Caller
    /// must `.release()` the returned ref when done.
    pub fn pin(self: *PageCache, page_no: u64) PinError!PageRef {
        const shard = self.shardFor(page_no);

        // Fast path: hit.
        {
            shard.lock.lock();
            if (shard.index.get(page_no)) |buffer_idx| {
                const slot = shard.slotForBuffer(buffer_idx);
                slot.pin_count += 1;
                slot.clock_referenced = true;
                shard.lock.unlock();
                return .{ .cache = self, .page_no = page_no, .buffer_idx = buffer_idx };
            }
            shard.lock.unlock();
        }

        // Miss path: assign a buffer (may evict, releasing lock during
        // dirty writeback), read content, install.
        const buffer_idx = try self.assignBuffer(shard);
        errdefer self.returnFreshBuffer(shard, buffer_idx);
        try self.file.readPage(page_no, self.pool.buf(buffer_idx));
        if (!page.verifyChecksum(self.pool.buf(buffer_idx))) return error.PageChecksumMismatch;
        // Belt-and-suspenders after CRC: a bit-flip that happened
        // *in memory* between the previous read and the writeback
        // would stamp a matching CRC for the corrupt content. The
        // structural check refuses any page whose slots/cells don't
        // fit inside PAGE_SIZE, so we never index off garbage.
        if (!page.verifyStructure(self.pool.buf(buffer_idx))) return error.PageStructureInvalid;

        shard.lock.lock();
        defer shard.lock.unlock();
        // Concurrent thread might have pinned the same page in the
        // window between our miss check and now. If so, return their
        // buffer; ours is wasted I/O but the index stays coherent.
        if (shard.index.get(page_no)) |existing_idx| {
            shard.free_globals.append(self.allocator, buffer_idx) catch
                @panic("free_globals append failed (no capacity? — bug)");
            const slot = shard.slotForBuffer(existing_idx);
            slot.pin_count += 1;
            slot.clock_referenced = true;
            return .{ .cache = self, .page_no = page_no, .buffer_idx = existing_idx };
        }
        const slot = shard.slotForBuffer(buffer_idx);
        slot.* = .{
            .state = .occupied,
            .page_no = page_no,
            .pin_count = 1,
            .dirty_seq = null,
            .clock_referenced = true,
        };
        try shard.index.put(self.allocator, page_no, buffer_idx);
        return .{ .cache = self, .page_no = page_no, .buffer_idx = buffer_idx };
    }

    /// Pin a newly-allocated page. Buffer is zero-initialized; caller
    /// is expected to write content and `markDirty()` before release.
    ///
    /// If `page_no` is already in cache (stale entry from a previous
    /// life — the page was freed and the allocator just handed it back
    /// for reuse), the stale entry is dropped *without* write-back.
    pub fn pinNew(self: *PageCache, page_no: u64) PinError!PageRef {
        return self.pinNewImpl(page_no, true);
    }

    /// Like `pinNew`, but skips the 4KB zero-fill. Caller MUST write
    /// the entire structurally-meaningful region of the page before
    /// releasing the ref. Trailing slack between the last written
    /// cell and end-of-page may contain stale buffer-pool memory and
    /// will be persisted to disk verbatim on the next durabilize.
    ///
    /// This is acceptable because:
    ///   - Tree readers never look at bytes past `nEntries`.
    ///   - Reused pages already carry old kvexp-owned data via the
    ///     freelist (no cross-process exposure surface).
    ///
    /// Used in B-tree CoW where the caller writes a header + densely
    /// packed cells immediately after pinning — the memset was 15%
    /// of total CPU on a write-heavy workload (perf-validated).
    pub fn pinNewUninit(self: *PageCache, page_no: u64) PinError!PageRef {
        return self.pinNewImpl(page_no, false);
    }

    inline fn pinNewImpl(self: *PageCache, page_no: u64, zero: bool) PinError!PageRef {
        const shard = self.shardFor(page_no);

        // First, try the no-eviction fast path under lock.
        {
            shard.lock.lock();
            if (shard.index.get(page_no)) |existing_idx| {
                const slot = shard.slotForBuffer(existing_idx);
                std.debug.assert(slot.pin_count == 0);
                std.debug.assert(slot.state == .occupied);
                if (zero) @memset(self.pool.buf(existing_idx), 0);
                slot.* = .{
                    .state = .occupied,
                    .page_no = page_no,
                    .pin_count = 1,
                    .dirty_seq = null,
                    .clock_referenced = true,
                };
                shard.lock.unlock();
                return .{ .cache = self, .page_no = page_no, .buffer_idx = existing_idx };
            }
            if (shard.free_globals.pop()) |idx| {
                if (zero) @memset(self.pool.buf(idx), 0);
                const slot = shard.slotForBuffer(idx);
                slot.* = .{
                    .state = .occupied,
                    .page_no = page_no,
                    .pin_count = 1,
                    .dirty_seq = null,
                    .clock_referenced = true,
                };
                try shard.index.put(self.allocator, page_no, idx);
                shard.lock.unlock();
                return .{ .cache = self, .page_no = page_no, .buffer_idx = idx };
            }
            shard.lock.unlock();
        }

        // Slow path: eviction may need to release the lock for a
        // sync writePage.
        const buffer_idx = try self.evictForReuse(shard);

        shard.lock.lock();
        defer shard.lock.unlock();
        if (zero) @memset(self.pool.buf(buffer_idx), 0);
        const slot = shard.slotForBuffer(buffer_idx);
        slot.* = .{
            .state = .occupied,
            .page_no = page_no,
            .pin_count = 1,
            .dirty_seq = null,
            .clock_referenced = true,
        };
        try shard.index.put(self.allocator, page_no, buffer_idx);
        return .{ .cache = self, .page_no = page_no, .buffer_idx = buffer_idx };
    }

    /// Write back all dirty pages with `dirty_seq <= seq`, then clear
    /// their dirty tag. Does NOT fsync; caller drives durability.
    pub fn flushUpTo(self: *PageCache, seq: u64) !void {
        return self.flushUpToSkipping(seq, null);
    }

    /// Like `flushUpTo`, but if a page's `page_no` is in `skip`, the
    /// dirty tag is cleared *without* writing the page to disk.
    pub fn flushUpToSkipping(
        self: *PageCache,
        seq: u64,
        skip: ?*const std.AutoHashMapUnmanaged(u64, void),
    ) !void {
        for (self.shards) |*shard| {
            try self.flushShard(shard, seq, skip);
        }
    }

    fn flushShard(
        self: *PageCache,
        shard: *Shard,
        seq: u64,
        skip: ?*const std.AutoHashMapUnmanaged(u64, void),
    ) !void {
        shard.lock.lock();
        defer shard.lock.unlock();

        var writes: std.ArrayListUnmanaged(PageWrite) = .empty;
        defer writes.deinit(self.allocator);
        var to_clean: std.ArrayListUnmanaged(BufferIndex) = .empty;
        defer to_clean.deinit(self.allocator);
        var clean_only: std.ArrayListUnmanaged(BufferIndex) = .empty;
        defer clean_only.deinit(self.allocator);

        var local: usize = 0;
        while (local < shard.slots.len) : (local += 1) {
            const s = &shard.slots[local];
            if (s.state != .occupied) continue;
            const ds = s.dirty_seq orelse continue;
            if (ds > seq) continue;
            const global_idx: BufferIndex = @intCast(shard.global_offset + local);
            if (skip) |sk| {
                if (sk.contains(s.page_no)) {
                    try clean_only.append(self.allocator, global_idx);
                    continue;
                }
            }
            // Stamp the CRC32 into the buffer just before we hand it to
            // the I/O layer. Safe under shard.lock — no other thread is
            // mutating this buffer (pin_count==0 implies no writers, and
            // the only other reader holding the buffer via PageRef would
            // not be racing with us under shard.lock).
            const buf_slice = self.pool.buf(global_idx);
            page.finalize(buf_slice);
            try writes.append(self.allocator, .{
                .page_no = s.page_no,
                .buf = buf_slice,
            });
            try to_clean.append(self.allocator, global_idx);
        }

        if (writes.items.len != 0) try self.file.writePages(writes.items);
        for (to_clean.items) |g| shard.slotForBuffer(g).dirty_seq = null;
        for (clean_only.items) |g| shard.slotForBuffer(g).dirty_seq = null;
    }

    pub const Stats = struct {
        capacity: u32,
        occupied: u32,
        dirty: u32,
        pinned: u32,
        shard_count: u32,
    };

    pub fn stats(self: *PageCache) Stats {
        var s: Stats = .{ .capacity = 0, .occupied = 0, .dirty = 0, .pinned = 0, .shard_count = @intCast(self.shards.len) };
        for (self.shards) |*shard| {
            shard.lock.lock();
            defer shard.lock.unlock();
            s.capacity += shard.capacity;
            for (shard.slots) |slot| {
                if (slot.state != .occupied) continue;
                s.occupied += 1;
                if (slot.dirty_seq != null) s.dirty += 1;
                if (slot.pin_count > 0) s.pinned += 1;
            }
        }
        return s;
    }

    fn assignBuffer(self: *PageCache, shard: *Shard) !BufferIndex {
        {
            shard.lock.lock();
            if (shard.free_globals.pop()) |idx| {
                shard.lock.unlock();
                return idx;
            }
            shard.lock.unlock();
        }
        return try self.evictForReuse(shard);
    }

    /// Return a buffer that `assignBuffer` produced but the caller
    /// couldn't install (e.g., readPage failed or CRC verify failed).
    /// The slot is left in the `.empty` state assignBuffer leaves it in;
    /// only `free_globals` needs the index back.
    fn returnFreshBuffer(self: *PageCache, shard: *Shard, buffer_idx: BufferIndex) void {
        shard.lock.lock();
        defer shard.lock.unlock();
        shard.free_globals.append(self.allocator, buffer_idx) catch
            @panic("free_globals append failed (capacity reserved at init — bug)");
    }

    /// Find a victim and produce a free buffer index. If the victim is
    /// dirty, the writeback runs **without holding the shard lock** —
    /// the slot's `.evicting` state pins the buffer for our use and
    /// blocks other threads from reassigning it during the write.
    fn evictForReuse(self: *PageCache, shard: *Shard) !BufferIndex {
        const WriteBack = struct { page_no: u64, buffer_idx: BufferIndex };
        var pending_wb: ?WriteBack = null;

        {
            shard.lock.lock();
            defer shard.lock.unlock();
            const n: u32 = @intCast(shard.slots.len);
            if (n == 0) return error.AllPagesPinned;
            // Two full sweeps: first clears reference bits, second
            // finds an unreferenced victim.
            var rounds: u32 = 0;
            while (rounds < n * 2) : (rounds += 1) {
                const i = shard.clock_hand;
                shard.clock_hand = (shard.clock_hand + 1) % n;
                const slot = &shard.slots[i];
                if (slot.state != .occupied) continue;
                if (slot.pin_count > 0) continue;
                if (slot.clock_referenced) {
                    slot.clock_referenced = false;
                    continue;
                }
                const global_idx: BufferIndex = @intCast(shard.global_offset + i);
                _ = shard.index.remove(slot.page_no);
                if (slot.dirty_seq) |_| {
                    pending_wb = .{ .page_no = slot.page_no, .buffer_idx = global_idx };
                    slot.state = .evicting;
                    slot.dirty_seq = null;
                    break; // exit clock loop; write outside the lock
                }
                slot.state = .empty;
                return global_idx;
            }
            if (pending_wb == null) return error.AllPagesPinned;
        }

        const wb = pending_wb.?;
        // Stamp CRC just before writeback. The buffer is locked by the
        // `.evicting` slot state — no concurrent reader/writer can touch
        // it until we transition back to `.empty` below.
        const wb_buf = self.pool.buf(wb.buffer_idx);
        page.finalize(wb_buf);
        try self.file.writePage(wb.page_no, wb_buf);

        shard.lock.lock();
        defer shard.lock.unlock();
        const slot = shard.slotForBuffer(wb.buffer_idx);
        std.debug.assert(slot.state == .evicting);
        slot.state = .empty;
        return wb.buffer_idx;
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

/// Cache tests now exercise the structural-verify path (failures
/// surface as `error.PageStructureInvalid` at `pin` time). Tests
/// therefore initialize a real empty Leaf at the top of every buffer
/// they round-trip, then fill the cell area with a sentinel byte for
/// equality checks. `initLeafFilled` stamps an empty-but-valid Leaf
/// header and overwrites the body; `expectBodyFilled` verifies the
/// header is intact and every byte after it is the fill (the body
/// includes the cell area but not the page header).
fn initLeafFilled(buf: []u8, fill: u8) void {
    _ = page.Leaf.init(buf);
    @memset(buf[page.HEADER_SIZE..], fill);
}

fn expectBodyFilled(buf: []const u8, fill: u8) !void {
    try testing.expectEqual(@as(u8, @intFromEnum(page.Kind.leaf)), buf[0]);
    for (buf[page.HEADER_SIZE..]) |b| try testing.expectEqual(fill, b);
}

const Harness = struct {
    tmp: std.testing.TmpDir,
    file: *PagedFile,
    pool: *BufferPool,
    cache: PageCache,

    fn init(pool_capacity: u32, file_pages: u64) !Harness {
        return initWithShards(pool_capacity, file_pages, 0);
    }

    fn initWithShards(pool_capacity: u32, file_pages: u64, shard_count: u32) !Harness {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();

        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &path_buf);
        var full_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&full_buf, "{s}/cache.test", .{dir_path});

        const file_ptr = try testing.allocator.create(PagedFile);
        errdefer testing.allocator.destroy(file_ptr);
        file_ptr.* = try PagedFile.open(path, .{ .create = true, .truncate = true });
        errdefer file_ptr.close();
        _ = try file_ptr.growBy(file_pages);

        const pool_ptr = try testing.allocator.create(BufferPool);
        errdefer testing.allocator.destroy(pool_ptr);
        pool_ptr.* = try BufferPool.init(testing.allocator, 4096, pool_capacity);
        errdefer pool_ptr.deinit(testing.allocator);

        const cache = try PageCache.init(testing.allocator, file_ptr.api(), pool_ptr, .{ .shard_count = shard_count });

        return .{
            .tmp = tmp,
            .file = file_ptr,
            .pool = pool_ptr,
            .cache = cache,
        };
    }

    fn deinit(self: *Harness) void {
        self.cache.deinit();
        self.pool.deinit(testing.allocator);
        testing.allocator.destroy(self.pool);
        self.file.close();
        testing.allocator.destroy(self.file);
        self.tmp.cleanup();
    }
};

test "PageCache: pinNew gives zeroed buffer, persists after flush + reread" {
    var h = try Harness.init(4, 8);
    defer h.deinit();

    {
        const ref = try h.cache.pinNew(3);
        defer ref.release();
        for (ref.buf()) |b| try testing.expectEqual(@as(u8, 0), b);
        initLeafFilled(ref.buf(), 0x77);
        ref.markDirty(10);
    }

    try h.cache.flushUpTo(10);
    try h.file.fsync();

    {
        const a = try h.cache.pinNew(4);
        initLeafFilled(a.buf(), 0);
        a.markDirty(11);
        defer a.release();
        const b = try h.cache.pinNew(5);
        initLeafFilled(b.buf(), 0);
        b.markDirty(11);
        defer b.release();
        const c = try h.cache.pinNew(6);
        initLeafFilled(c.buf(), 0);
        c.markDirty(11);
        defer c.release();
        const d = try h.cache.pinNew(7);
        initLeafFilled(d.buf(), 0);
        d.markDirty(11);
        defer d.release();
    }
    try h.cache.flushUpTo(11);

    const reread = try h.cache.pin(3);
    defer reread.release();
    try expectBodyFilled(reread.buf(), 0x77);
}

test "PageCache: hit returns same buffer; release allows eviction" {
    var h = try Harness.init(2, 8);
    defer h.deinit();

    const a1 = try h.cache.pinNew(0);
    @memset(a1.buf(), 0xAA);
    a1.markDirty(1);
    const first_ptr = a1.buf().ptr;
    a1.release();

    const a2 = try h.cache.pin(0);
    defer a2.release();
    try testing.expectEqual(first_ptr, a2.buf().ptr);
}

test "PageCache: pinned page cannot be evicted" {
    var h = try Harness.initWithShards(2, 8, 1);
    defer h.deinit();

    const a = try h.cache.pinNew(0);
    initLeafFilled(a.buf(), 0x11);
    a.markDirty(1);
    const b = try h.cache.pinNew(1);
    initLeafFilled(b.buf(), 0x22);
    b.markDirty(1);

    try testing.expectError(error.AllPagesPinned, h.cache.pin(2));

    a.release();
    b.release();

    const c = try h.cache.pinNew(2);
    defer c.release();
    initLeafFilled(c.buf(), 0x33);
    c.markDirty(1);

    try h.cache.flushUpTo(1);

    const a_check = try h.cache.pin(0);
    defer a_check.release();
    try expectBodyFilled(a_check.buf(), 0x11);
}

test "PageCache: flushUpTo is selective by seq" {
    var h = try Harness.init(4, 8);
    defer h.deinit();

    {
        const a = try h.cache.pinNew(0);
        defer a.release();
        @memset(a.buf(), 0xA0);
        a.markDirty(5);

        const b = try h.cache.pinNew(1);
        defer b.release();
        @memset(b.buf(), 0xB0);
        b.markDirty(10);
    }

    try h.cache.flushUpTo(5);

    var stats = h.cache.stats();
    try testing.expectEqual(@as(u32, 2), stats.occupied);
    try testing.expectEqual(@as(u32, 1), stats.dirty);

    try h.cache.flushUpTo(10);
    stats = h.cache.stats();
    try testing.expectEqual(@as(u32, 0), stats.dirty);
}

test "PageCache: markDirty bumps to higher seq" {
    var h = try Harness.init(2, 8);
    defer h.deinit();

    const ref = try h.cache.pinNew(0);
    defer ref.release();
    ref.markDirty(5);
    ref.markDirty(3);
    ref.markDirty(7);

    try h.cache.flushUpTo(5);
    try testing.expectEqual(@as(u32, 1), h.cache.stats().dirty);

    try h.cache.flushUpTo(7);
    try testing.expectEqual(@as(u32, 0), h.cache.stats().dirty);
}

test "PageCache: dirty page is written back on eviction" {
    var h = try Harness.init(1, 4);
    defer h.deinit();

    {
        const a = try h.cache.pinNew(0);
        initLeafFilled(a.buf(), 0xCC);
        a.markDirty(1);
        a.release();
    }

    {
        const b = try h.cache.pinNew(1);
        initLeafFilled(b.buf(), 0);
        b.release();
    }

    const re = try h.cache.pin(0);
    defer re.release();
    try expectBodyFilled(re.buf(), 0xCC);
}

test "PageCache: clean reread after flush verifies CRC and round-trips" {
    // Cache capacity 1: pinning a different page evicts the first, so
    // the next pin of page 0 forces a disk read + checksum verify.
    var h = try Harness.init(1, 4);
    defer h.deinit();

    {
        const a = try h.cache.pinNew(0);
        initLeafFilled(a.buf(), 0xCC);
        a.markDirty(1);
        a.release();
    }
    try h.cache.flushUpTo(1);
    try h.file.fsync();

    // Evict page 0 from the cache by pinning another page.
    {
        const b = try h.cache.pinNew(1);
        initLeafFilled(b.buf(), 0);
        b.release();
    }

    const re = try h.cache.pin(0);
    defer re.release();
    try expectBodyFilled(re.buf(), 0xCC);
}

test "PageCache: corrupt page on disk → PageChecksumMismatch on pin" {
    var h = try Harness.init(1, 4);
    defer h.deinit();

    // Write a valid page through the cache.
    {
        const a = try h.cache.pinNew(0);
        initLeafFilled(a.buf(), 0xCC);
        a.markDirty(1);
        a.release();
    }
    try h.cache.flushUpTo(1);
    try h.file.fsync();

    // Evict page 0 (capacity 1) so the next pin re-reads from disk.
    {
        const b = try h.cache.pinNew(1);
        initLeafFilled(b.buf(), 0);
        b.release();
    }

    // Tamper with a byte well outside the checksum field.
    {
        const scratch = try testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), 4096);
        defer testing.allocator.free(scratch);
        try h.file.readPage(0, scratch);
        scratch[2048] ^= 0xFF;
        try h.file.writePage(0, scratch);
        try h.file.fsync();
    }

    try testing.expectError(error.PageChecksumMismatch, h.cache.pin(0));
}

test "PageCache: corrupt byte INSIDE checksum field also rejected" {
    var h = try Harness.init(1, 4);
    defer h.deinit();

    {
        const a = try h.cache.pinNew(0);
        initLeafFilled(a.buf(), 0xAB);
        a.markDirty(1);
        a.release();
    }
    try h.cache.flushUpTo(1);
    try h.file.fsync();

    // Evict, then corrupt one byte of the CRC itself.
    {
        const b = try h.cache.pinNew(1);
        initLeafFilled(b.buf(), 0);
        b.release();
    }
    {
        const scratch = try testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), 4096);
        defer testing.allocator.free(scratch);
        try h.file.readPage(0, scratch);
        scratch[page.CHECKSUM_OFFSET] ^= 0x01;
        try h.file.writePage(0, scratch);
        try h.file.fsync();
    }

    try testing.expectError(error.PageChecksumMismatch, h.cache.pin(0));
}

test "PageCache: structurally-invalid page rejected (CRC OK)" {
    // CRC-but-not-structure attack: forge a page whose header claims
    // n_entries=999 (way past the slot array's capacity), then
    // recompute the CRC so verifyChecksum passes. verifyStructure
    // must still refuse, returning PageStructureInvalid.
    var h = try Harness.init(1, 4);
    defer h.deinit();

    {
        const a = try h.cache.pinNew(0);
        initLeafFilled(a.buf(), 0x00);
        a.markDirty(1);
        a.release();
    }
    try h.cache.flushUpTo(1);
    try h.file.fsync();

    {
        const b = try h.cache.pinNew(1);
        initLeafFilled(b.buf(), 0);
        b.release();
    }

    const scratch = try testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), 4096);
    defer testing.allocator.free(scratch);
    try h.file.readPage(0, scratch);
    // Forge: claim 999 entries (impossible — slot array alone would
    // be 1998 bytes, plus 32 header = 2030, leaving 2066 for cells
    // but each cell needs offsets that the verifier will reject).
    const h_ptr: *page.LeafHeader = @ptrCast(@alignCast(scratch.ptr));
    h_ptr.n_entries = 999;
    // Re-stamp CRC over the forged content so verifyChecksum passes.
    page.finalize(scratch);
    try h.file.writePage(0, scratch);
    try h.file.fsync();

    try testing.expectError(error.PageStructureInvalid, h.cache.pin(0));
}

test "PageCache: auto-shard count scales with pool capacity" {
    var h64 = try Harness.init(64, 4);
    defer h64.deinit();
    try testing.expectEqual(@as(u32, 16), h64.cache.shardCount());

    var h32 = try Harness.init(32, 4);
    defer h32.deinit();
    try testing.expectEqual(@as(u32, 8), h32.cache.shardCount());

    var h4 = try Harness.init(4, 4);
    defer h4.deinit();
    try testing.expectEqual(@as(u32, 1), h4.cache.shardCount());
}

test "PageCache: page_no maps consistently to same shard" {
    // Pages 0 and 16 must land in the same shard under mod-16; pages
    // 0 and 1 must land in different shards.
    var h = try Harness.initWithShards(32, 32, 16);
    defer h.deinit();

    const a = try h.cache.pinNew(0);
    defer a.release();
    const b = try h.cache.pinNew(16);
    defer b.release();
    const c = try h.cache.pinNew(1);
    defer c.release();

    // Hash maps page_no % 16, so 0 and 16 share shard 0; 1 lives in
    // shard 1. The buffer indices reflect shard ownership.
    const sh0 = &h.cache.shards[0];
    const sh1 = &h.cache.shards[1];
    try testing.expect(a.buffer_idx >= sh0.global_offset);
    try testing.expect(a.buffer_idx < sh0.global_offset + sh0.capacity);
    try testing.expect(b.buffer_idx >= sh0.global_offset);
    try testing.expect(b.buffer_idx < sh0.global_offset + sh0.capacity);
    try testing.expect(c.buffer_idx >= sh1.global_offset);
    try testing.expect(c.buffer_idx < sh1.global_offset + sh1.capacity);
}
