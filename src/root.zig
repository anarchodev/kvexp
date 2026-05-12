//! kvexp — multi-tenant embedded KV with raft-as-WAL.
//!
//! Phase 0/1 surface: file header, paged I/O over io_uring + O_DIRECT,
//! bounded buffer pool, in-process page cache with clock eviction and
//! sequence-tagged dirty tracking. See docs/PLAN.md for the full design.

pub const header = @import("header.zig");
pub const paged_file = @import("paged_file.zig");
pub const faulty_paged_file = @import("faulty_paged_file.zig");
pub const buffer_pool = @import("buffer_pool.zig");
pub const page_cache = @import("page_cache.zig");
pub const page = @import("page.zig");
pub const btree = @import("btree.zig");
pub const Tree = btree.Tree;
pub const manifest = @import("manifest.zig");
pub const Manifest = manifest.Manifest;
pub const Store = manifest.Store;
pub const Snapshot = manifest.Snapshot;
pub const dumpSnapshot = manifest.dumpSnapshot;
pub const loadSnapshot = manifest.loadSnapshot;
pub const SNAPSHOT_MAGIC = manifest.SNAPSHOT_MAGIC;
pub const SNAPSHOT_VERSION = manifest.SNAPSHOT_VERSION;

pub const Header = header.Header;
pub const PagedFile = paged_file.PagedFile;
pub const FaultyPagedFile = faulty_paged_file.FaultyPagedFile;
pub const FaultPolicy = faulty_paged_file.FaultPolicy;
pub const BufferPool = buffer_pool.BufferPool;
pub const BufferIndex = buffer_pool.BufferIndex;
pub const PageCache = page_cache.PageCache;
pub const PageRef = page_cache.PageRef;

pub const PAGE_SIZE_DEFAULT: u32 = 4096;

test {
    @import("std").testing.refAllDecls(@This());
}

// -----------------------------------------------------------------------------
// End-to-end smoke test for the io_uring + O_DIRECT path: write the file
// header at page 0 and a handful of opaque data pages directly via
// PagedFile (no cache, no page-format constraints), close, reopen, and
// validate. The page cache and B-tree layers are exercised in much
// more depth by manifest_test.zig, so this only covers the lowest
// layer.
// -----------------------------------------------------------------------------

const std = @import("std");
const testing = std.testing;

test "e2e: header + raw pages survive close/reopen via PagedFile" {
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

    const buf = try allocator.alignedAlloc(u8, .fromByteUnits(PAGE_SIZE_DEFAULT), PAGE_SIZE_DEFAULT);
    defer allocator.free(buf);

    // Pass 1: write.
    {
        var file = try PagedFile.open(path, .{ .create = true, .truncate = true });
        defer file.close();
        _ = try file.growBy(4); // page 0 + 3 data pages

        @memset(buf, 0);
        const h = Header.init(PAGE_SIZE_DEFAULT);
        @memcpy(buf[0..@sizeOf(Header)], h.toBytes());
        try file.writePage(0, buf);

        for (data_pages) |dp| {
            @memset(buf, dp.fill);
            try file.writePage(dp.page_no, buf);
        }

        try file.fsync();
    }

    // Pass 2: reopen + validate.
    {
        var file = try PagedFile.open(path, .{});
        defer file.close();

        try testing.expectEqual(@as(u64, 4), file.pageCount());

        try file.readPage(0, buf);
        const h = Header.fromBytes(buf);
        try h.validate(PAGE_SIZE_DEFAULT);

        for (data_pages) |dp| {
            try file.readPage(dp.page_no, buf);
            for (buf) |b| try testing.expectEqual(dp.fill, b);
        }
    }
}
