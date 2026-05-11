//! kvexp — multi-tenant embedded KV with raft-as-WAL.
//!
//! Phase 0/1 surface: file header, paged I/O over io_uring + O_DIRECT,
//! bounded buffer pool, in-process page cache with clock eviction and
//! sequence-tagged dirty tracking. See docs/PLAN.md for the full design.

pub const header = @import("header.zig");
pub const paged_file = @import("paged_file.zig");
pub const buffer_pool = @import("buffer_pool.zig");
pub const page_cache = @import("page_cache.zig");
pub const page = @import("page.zig");
pub const btree = @import("btree.zig");
pub const Tree = btree.Tree;
pub const manifest = @import("manifest.zig");
pub const Manifest = manifest.Manifest;
pub const Store = manifest.Store;

pub const Header = header.Header;
pub const PagedFile = paged_file.PagedFile;
pub const BufferPool = buffer_pool.BufferPool;
pub const BufferIndex = buffer_pool.BufferIndex;
pub const PageCache = page_cache.PageCache;
pub const PageRef = page_cache.PageRef;

pub const PAGE_SIZE_DEFAULT: u32 = 4096;

test {
    @import("std").testing.refAllDecls(@This());
}

// -----------------------------------------------------------------------------
// End-to-end integration test: open → write header + data pages → close
// → reopen → validate header → read back data pages.
// -----------------------------------------------------------------------------

const std = @import("std");
const testing = std.testing;

test "e2e: header + data pages survive close/reopen" {
    const allocator = testing.allocator;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try tmp.dir.realpath(".", &path_buf);
    var full_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full_buf, "{s}/e2e.kv", .{dir_path});

    const data_pages = [_]struct { page_no: u64, fill: u8 }{
        .{ .page_no = 1, .fill = 0xA1 },
        .{ .page_no = 2, .fill = 0xB2 },
        .{ .page_no = 3, .fill = 0xC3 },
    };

    // Pass 1: write.
    {
        var file = try PagedFile.open(path, .{ .create = true, .truncate = true });
        defer file.close();
        var pool = try BufferPool.init(allocator, PAGE_SIZE_DEFAULT, 8);
        defer pool.deinit(allocator);
        var cache = try PageCache.init(allocator, &file, &pool, .{});
        defer cache.deinit();

        _ = try file.growBy(4); // page 0 + 3 data pages

        // Header at page 0.
        {
            const ref = try cache.pinNew(0);
            defer ref.release();
            const h = Header.init(PAGE_SIZE_DEFAULT);
            @memcpy(ref.buf()[0..@sizeOf(Header)], h.toBytes());
            ref.markDirty(1);
        }

        for (data_pages) |dp| {
            const ref = try cache.pinNew(dp.page_no);
            defer ref.release();
            @memset(ref.buf(), dp.fill);
            ref.markDirty(1);
        }

        try cache.flushUpTo(1);
        try file.fsync();
    }

    // Pass 2: reopen + validate.
    {
        var file = try PagedFile.open(path, .{});
        defer file.close();
        var pool = try BufferPool.init(allocator, PAGE_SIZE_DEFAULT, 8);
        defer pool.deinit(allocator);
        var cache = try PageCache.init(allocator, &file, &pool, .{});
        defer cache.deinit();

        try testing.expectEqual(@as(u64, 4), file.pageCount());

        {
            const ref = try cache.pin(0);
            defer ref.release();
            const h = Header.fromBytes(ref.buf());
            try h.validate(PAGE_SIZE_DEFAULT);
        }

        for (data_pages) |dp| {
            const ref = try cache.pin(dp.page_no);
            defer ref.release();
            for (ref.buf()) |b| try testing.expectEqual(dp.fill, b);
        }
    }
}
