//! Bounded pool of page-aligned buffers backed by one contiguous arena.
//!
//! Owning our own buffer storage (rather than `O_DIRECT`-aligned slices
//! handed out by `posix_memalign` per allocation) gives us two things:
//! deterministic peak RAM (the pool *is* the budget) and stable buffer
//! identity that the page cache can key by index. The arena is one
//! `mmap` so all buffers share locality.
//!
//! Phase 0/1 is single-threaded; concurrent acquire/release is phase 6.

const std = @import("std");

pub const BufferIndex = u32;
pub const INVALID_BUFFER: BufferIndex = std.math.maxInt(BufferIndex);

pub const BufferPool = struct {
    page_size: u32,
    capacity: u32,
    storage: []align(std.heap.page_size_min) u8,
    free_list: std.ArrayListUnmanaged(BufferIndex),

    pub fn init(
        allocator: std.mem.Allocator,
        page_size: u32,
        capacity: u32,
    ) !BufferPool {
        std.debug.assert(std.math.isPowerOfTwo(page_size));
        std.debug.assert(page_size >= std.heap.page_size_min);
        std.debug.assert(capacity > 0);

        const total: usize = @as(usize, page_size) * capacity;
        const storage = try allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), total);
        errdefer allocator.free(storage);

        var free_list: std.ArrayListUnmanaged(BufferIndex) = .empty;
        try free_list.ensureTotalCapacity(allocator, capacity);
        var i: BufferIndex = capacity;
        while (i > 0) {
            i -= 1;
            free_list.appendAssumeCapacity(i);
        }

        return .{
            .page_size = page_size,
            .capacity = capacity,
            .storage = storage,
            .free_list = free_list,
        };
    }

    pub fn deinit(self: *BufferPool, allocator: std.mem.Allocator) void {
        self.free_list.deinit(allocator);
        allocator.free(self.storage);
        self.* = undefined;
    }

    pub fn acquire(self: *BufferPool) ?BufferIndex {
        return self.free_list.pop() orelse null;
    }

    pub fn release(self: *BufferPool, idx: BufferIndex) void {
        std.debug.assert(idx < self.capacity);
        // ArrayList has guaranteed capacity (we reserved `capacity` at init),
        // so append never reallocates.
        self.free_list.appendAssumeCapacity(idx);
    }

    pub fn freeCount(self: *const BufferPool) u32 {
        return @intCast(self.free_list.items.len);
    }

    /// Borrow the slice backing a buffer. Returned slice has the same
    /// page-aligned alignment as the arena; lifetime is tied to the pool.
    pub fn buf(self: *BufferPool, idx: BufferIndex) []align(std.heap.page_size_min) u8 {
        std.debug.assert(idx < self.capacity);
        const offset = @as(usize, idx) * self.page_size;
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(self.storage.ptr + offset);
        return ptr[0..self.page_size];
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

test "BufferPool: init produces fully-free pool" {
    var pool = try BufferPool.init(testing.allocator, 4096, 16);
    defer pool.deinit(testing.allocator);

    try testing.expectEqual(@as(u32, 16), pool.freeCount());
}

test "BufferPool: acquire then release recycles buffer" {
    var pool = try BufferPool.init(testing.allocator, 4096, 4);
    defer pool.deinit(testing.allocator);

    const a = pool.acquire().?;
    try testing.expectEqual(@as(u32, 3), pool.freeCount());
    const b = pool.acquire().?;
    try testing.expectEqual(@as(u32, 2), pool.freeCount());
    try testing.expect(a != b);

    pool.release(a);
    try testing.expectEqual(@as(u32, 3), pool.freeCount());
}

test "BufferPool: acquire to exhaustion returns null" {
    var pool = try BufferPool.init(testing.allocator, 4096, 3);
    defer pool.deinit(testing.allocator);

    _ = pool.acquire().?;
    _ = pool.acquire().?;
    _ = pool.acquire().?;
    try testing.expectEqual(@as(?BufferIndex, null), pool.acquire());
}

test "BufferPool: buffers are page-aligned and disjoint" {
    var pool = try BufferPool.init(testing.allocator, 4096, 4);
    defer pool.deinit(testing.allocator);

    var seen = [_]usize{0} ** 4;
    for (0..4) |i| {
        const idx = pool.acquire().?;
        const slice = pool.buf(idx);
        try testing.expect(std.mem.isAligned(@intFromPtr(slice.ptr), 4096));
        try testing.expectEqual(@as(usize, 4096), slice.len);
        seen[i] = @intFromPtr(slice.ptr);
    }
    // All four slices must be distinct.
    for (0..4) |i| for (i + 1..4) |j| try testing.expect(seen[i] != seen[j]);
}

test "BufferPool: writes through buf survive release/reacquire of same index" {
    var pool = try BufferPool.init(testing.allocator, 4096, 2);
    defer pool.deinit(testing.allocator);

    const idx = pool.acquire().?;
    @memset(pool.buf(idx), 0x5A);
    pool.release(idx);

    // Reacquire — pool is LIFO, so this is the same index. Storage is
    // not zeroed on release; tests confirming that the caller is
    // responsible for initialization.
    const idx2 = pool.acquire().?;
    try testing.expectEqual(idx, idx2);
    for (pool.buf(idx2)) |b| try testing.expectEqual(@as(u8, 0x5A), b);
}
