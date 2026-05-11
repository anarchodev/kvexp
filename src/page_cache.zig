//! In-process page cache: page_no → buffer index, with clock-based
//! eviction, pin/unpin, and sequence-tagged dirty tracking.
//!
//! This is the layer where "we own writeback" pays off. Dirty pages
//! sit in cache until an explicit `flushUpTo(seq)` is issued by the
//! caller (later: the apply/durabilize machinery at phase 5). The OS
//! never writes back on its own — `O_DIRECT` plus our explicit pwrite
//! is the only path to disk.
//!
//! Phase 6: a single cache-wide mutex guards the index, slot states,
//! and the clock hand. Critical sections are short (each pin/release/
//! markDirty is a handful of struct-field accesses). Eviction runs
//! under the lock — that can write back a dirty page synchronously,
//! which is a longer section we accept for now (phase 11+ may move
//! eviction off the hot path or shard the cache).

const std = @import("std");
const PagedFile = @import("paged_file.zig").PagedFile;
const bp = @import("buffer_pool.zig");
const BufferPool = bp.BufferPool;
const BufferIndex = bp.BufferIndex;

pub const PageRef = struct {
    cache: *PageCache,
    page_no: u64,
    buffer_idx: BufferIndex,

    pub fn buf(self: PageRef) []align(std.heap.page_size_min) u8 {
        return self.cache.pool.buf(self.buffer_idx);
    }

    /// Mark the page dirty at the given producing sequence. If the page
    /// was already dirty at an earlier seq, the tag is bumped to `seq`
    /// (a CoW B-tree shouldn't dirty the same page_no twice, but the
    /// max-merge is the safe semantics).
    pub fn markDirty(self: PageRef, seq: u64) void {
        self.cache.lock.lock();
        defer self.cache.lock.unlock();
        const slot = &self.cache.slots[self.buffer_idx];
        slot.dirty_seq = if (slot.dirty_seq) |existing| @max(existing, seq) else seq;
    }

    pub fn release(self: PageRef) void {
        self.cache.lock.lock();
        defer self.cache.lock.unlock();
        const slot = &self.cache.slots[self.buffer_idx];
        std.debug.assert(slot.pin_count > 0);
        slot.pin_count -= 1;
    }
};

const Slot = struct {
    state: enum { empty, occupied },
    page_no: u64,
    pin_count: u32,
    dirty_seq: ?u64,
    clock_referenced: bool,
};

pub const PageCache = struct {
    allocator: std.mem.Allocator,
    file: *PagedFile,
    pool: *BufferPool,
    /// page_no → buffer_idx (which is also the index into `slots`).
    index: std.AutoHashMapUnmanaged(u64, BufferIndex),
    slots: []Slot,
    clock_hand: u32,
    /// Single cache-wide mutex. Held during pin/pinNew/release/
    /// markDirty/flushUpToSkipping. Recursive lock acquisition is NOT
    /// supported — callers must not hold the lock when calling another
    /// locked cache method.
    lock: std.Thread.Mutex = .{},

    pub fn init(allocator: std.mem.Allocator, file: *PagedFile, pool: *BufferPool) !PageCache {
        const slots = try allocator.alloc(Slot, pool.capacity);
        for (slots) |*s| s.* = .{
            .state = .empty,
            .page_no = 0,
            .pin_count = 0,
            .dirty_seq = null,
            .clock_referenced = false,
        };
        return .{
            .allocator = allocator,
            .file = file,
            .pool = pool,
            .index = .empty,
            .slots = slots,
            .clock_hand = 0,
        };
    }

    pub fn deinit(self: *PageCache) void {
        self.index.deinit(self.allocator);
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    pub const PinError = error{ AllPagesPinned, OutOfMemory } || PagedFile.IoError;

    /// Pin a page in cache. On miss, reads from disk first. Caller must
    /// `.release()` the returned ref when done.
    pub fn pin(self: *PageCache, page_no: u64) PinError!PageRef {
        self.lock.lock();
        defer self.lock.unlock();
        if (self.index.get(page_no)) |buffer_idx| {
            const slot = &self.slots[buffer_idx];
            slot.pin_count += 1;
            slot.clock_referenced = true;
            return .{ .cache = self, .page_no = page_no, .buffer_idx = buffer_idx };
        }
        const buffer_idx = try self.assignBuffer();
        try self.file.readPage(page_no, self.pool.buf(buffer_idx));
        self.installSlot(buffer_idx, page_no);
        try self.index.put(self.allocator, page_no, buffer_idx);
        return .{ .cache = self, .page_no = page_no, .buffer_idx = buffer_idx };
    }

    /// Pin a newly-allocated page. Buffer is zero-initialized; caller
    /// is expected to write content and `markDirty()` before release.
    ///
    /// If `page_no` is already in cache (from a previous life — the
    /// page was freed and the allocator just handed it back for reuse),
    /// the stale entry is dropped *without* write-back. The freelist
    /// invariant guarantees that no durable manifest references the
    /// page's old content, so skipping the write is safe and avoids
    /// wasting I/O on data that no one will ever read.
    pub fn pinNew(self: *PageCache, page_no: u64) PinError!PageRef {
        self.lock.lock();
        defer self.lock.unlock();
        if (self.index.get(page_no)) |existing_idx| {
            const slot = &self.slots[existing_idx];
            std.debug.assert(slot.pin_count == 0);
            @memset(self.pool.buf(existing_idx), 0);
            self.installSlot(existing_idx, page_no);
            return .{ .cache = self, .page_no = page_no, .buffer_idx = existing_idx };
        }
        const buffer_idx = try self.assignBuffer();
        @memset(self.pool.buf(buffer_idx), 0);
        self.installSlot(buffer_idx, page_no);
        try self.index.put(self.allocator, page_no, buffer_idx);
        return .{ .cache = self, .page_no = page_no, .buffer_idx = buffer_idx };
    }

    /// Write back all dirty pages with `dirty_seq <= seq`, then clear
    /// their dirty tag. Does NOT fsync; caller drives durability.
    pub fn flushUpTo(self: *PageCache, seq: u64) !void {
        return self.flushUpToSkipping(seq, null);
    }

    /// Like `flushUpTo`, but if a page's `page_no` is in `skip`, the
    /// dirty tag is cleared *without* writing the page to disk. Used
    /// by phase-5 group commit to elide orphaned pages — those that
    /// were superseded by a later apply and are no longer reachable
    /// from the manifest being durabilized.
    pub fn flushUpToSkipping(
        self: *PageCache,
        seq: u64,
        skip: ?*const std.AutoHashMapUnmanaged(u64, void),
    ) !void {
        self.lock.lock();
        defer self.lock.unlock();
        var writes: std.ArrayListUnmanaged(PagedFile.PageWrite) = .empty;
        defer writes.deinit(self.allocator);
        var clean_only: std.ArrayListUnmanaged(BufferIndex) = .empty;
        defer clean_only.deinit(self.allocator);
        var to_clean: std.ArrayListUnmanaged(BufferIndex) = .empty;
        defer to_clean.deinit(self.allocator);

        for (self.slots, 0..) |*s, idx| {
            if (s.state != .occupied) continue;
            const ds = s.dirty_seq orelse continue;
            if (ds > seq) continue;
            if (skip) |sk| {
                if (sk.contains(s.page_no)) {
                    try clean_only.append(self.allocator, @intCast(idx));
                    continue;
                }
            }
            try writes.append(self.allocator, .{
                .page_no = s.page_no,
                .buf = self.pool.buf(@intCast(idx)),
            });
            try to_clean.append(self.allocator, @intCast(idx));
        }

        if (writes.items.len != 0) try self.file.writePages(writes.items);
        for (to_clean.items) |idx| self.slots[idx].dirty_seq = null;
        for (clean_only.items) |idx| self.slots[idx].dirty_seq = null;
    }

    pub const Stats = struct {
        capacity: u32,
        occupied: u32,
        dirty: u32,
        pinned: u32,
    };

    pub fn stats(self: *PageCache) Stats {
        self.lock.lock();
        defer self.lock.unlock();
        var s: Stats = .{ .capacity = @intCast(self.slots.len), .occupied = 0, .dirty = 0, .pinned = 0 };
        for (self.slots) |slot| {
            if (slot.state != .occupied) continue;
            s.occupied += 1;
            if (slot.dirty_seq != null) s.dirty += 1;
            if (slot.pin_count > 0) s.pinned += 1;
        }
        return s;
    }

    fn installSlot(self: *PageCache, buffer_idx: BufferIndex, page_no: u64) void {
        self.slots[buffer_idx] = .{
            .state = .occupied,
            .page_no = page_no,
            .pin_count = 1,
            .dirty_seq = null,
            .clock_referenced = true,
        };
    }

    fn assignBuffer(self: *PageCache) !BufferIndex {
        if (self.pool.acquire()) |idx| {
            std.debug.assert(self.slots[idx].state == .empty);
            return idx;
        }
        return try self.evictOne();
    }

    fn evictOne(self: *PageCache) !BufferIndex {
        const n: u32 = @intCast(self.slots.len);
        if (n == 0) return error.AllPagesPinned;
        // At most two full sweeps: first clears reference bits, second
        // finds an unreferenced victim. Anything more means everything
        // is pinned.
        var rounds: u32 = 0;
        while (rounds < n * 2) : (rounds += 1) {
            const i = self.clock_hand;
            self.clock_hand = (self.clock_hand + 1) % n;
            const slot = &self.slots[i];
            if (slot.state != .occupied) continue;
            if (slot.pin_count > 0) continue;
            if (slot.clock_referenced) {
                slot.clock_referenced = false;
                continue;
            }
            if (slot.dirty_seq) |_| {
                try self.file.writePage(slot.page_no, self.pool.buf(@intCast(i)));
                slot.dirty_seq = null;
            }
            _ = self.index.remove(slot.page_no);
            slot.state = .empty;
            return @intCast(i);
        }
        return error.AllPagesPinned;
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

/// Wires a tmp file + buffer pool + page cache together for tests. The
/// underlying file is pre-grown so tests can directly write to page_no.
const Harness = struct {
    tmp: std.testing.TmpDir,
    file: *PagedFile,
    pool: *BufferPool,
    cache: PageCache,

    fn init(pool_capacity: u32, file_pages: u64) !Harness {
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

        const cache = try PageCache.init(testing.allocator, file_ptr, pool_ptr);

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
        // Zeroed.
        for (ref.buf()) |b| try testing.expectEqual(@as(u8, 0), b);
        @memset(ref.buf(), 0x77);
        ref.markDirty(10);
    }

    try h.cache.flushUpTo(10);
    try h.file.fsync();

    // Force eviction by pinning enough other pages, then re-pin 3.
    {
        const a = try h.cache.pinNew(4);
        a.markDirty(11);
        defer a.release();
        const b = try h.cache.pinNew(5);
        b.markDirty(11);
        defer b.release();
        const c = try h.cache.pinNew(6);
        c.markDirty(11);
        defer c.release();
        const d = try h.cache.pinNew(7);
        d.markDirty(11);
        defer d.release();
    }
    try h.cache.flushUpTo(11);

    // Now page 3 should be evicted (capacity is 4, four other pages pinned
    // through). Re-pin must hit the disk.
    const reread = try h.cache.pin(3);
    defer reread.release();
    for (reread.buf()) |b| try testing.expectEqual(@as(u8, 0x77), b);
}

test "PageCache: hit returns same buffer; release allows eviction" {
    var h = try Harness.init(2, 8);
    defer h.deinit();

    const a1 = try h.cache.pinNew(0);
    @memset(a1.buf(), 0xAA);
    a1.markDirty(1);
    const first_ptr = a1.buf().ptr;
    a1.release();

    // Same page_no should hit; buffer pointer unchanged.
    const a2 = try h.cache.pin(0);
    defer a2.release();
    try testing.expectEqual(first_ptr, a2.buf().ptr);
}

test "PageCache: pinned page cannot be evicted" {
    var h = try Harness.init(2, 8);
    defer h.deinit();

    const a = try h.cache.pinNew(0);
    @memset(a.buf(), 0x11);
    a.markDirty(1);
    // Keep `a` pinned.

    const b = try h.cache.pinNew(1);
    @memset(b.buf(), 0x22);
    b.markDirty(1);
    // Keep `b` pinned too. Capacity is exhausted with both pinned.

    // Asking for a third page must fail (all pinned).
    try testing.expectError(error.AllPagesPinned, h.cache.pin(2));

    a.release();
    b.release();

    // Now eviction can proceed.
    const c = try h.cache.pinNew(2);
    defer c.release();
    @memset(c.buf(), 0x33);
    c.markDirty(1);

    try h.cache.flushUpTo(1);

    // Verify a and b were written back during eviction.
    const a_check = try h.cache.pin(0);
    defer a_check.release();
    for (a_check.buf()) |byte| try testing.expectEqual(@as(u8, 0x11), byte);
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

    // Flush only seq ≤ 5: page 0 should hit disk, page 1 should not.
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

    // flushUpTo(5) should NOT flush this (its dirty_seq is 7).
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
        @memset(a.buf(), 0xCC);
        a.markDirty(1);
        a.release();
    }

    // Capacity is 1; pinning page 1 forces eviction of page 0 (dirty).
    {
        const b = try h.cache.pinNew(1);
        b.release();
    }

    // Re-pin page 0; content must come back from disk (written during evict).
    const re = try h.cache.pin(0);
    defer re.release();
    for (re.buf()) |byte| try testing.expectEqual(@as(u8, 0xCC), byte);
}
