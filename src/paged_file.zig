//! Paged I/O surface over io_uring + O_DIRECT.
//!
//! The whole point of O_DIRECT is that the kernel never holds dirty
//! pages on our behalf — we own writeback decisions so that group
//! commit can elide orphaned pages (see docs/PLAN.md §7.3). All buffers
//! handed to read/write must be page-size aligned in both pointer and
//! length.
//!
//! Phase 0/1: single-threaded. One IoUring per PagedFile. Sync wrappers
//! over io_uring SQE/CQE. Phase 5+ will exploit the batched submit
//! shape for group commit; the API is already pluralized for that.

const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;

pub const PagedFile = struct {
    fd: posix.fd_t,
    ring: linux.IoUring,
    page_size: u32,
    page_count: u64,
    ring_entries: u16,

    pub const OpenOptions = struct {
        create: bool = false,
        truncate: bool = false,
        page_size: u32 = 4096,
        ring_entries: u16 = 64,
    };

    pub const OpenError = std.posix.OpenError || std.posix.FStatError || error{
        FileSizeNotPageAligned,
        IoUringInitFailed,
    };

    pub fn open(path: []const u8, options: OpenOptions) OpenError!PagedFile {
        const flags: posix.O = .{
            .ACCMODE = .RDWR,
            .DIRECT = true,
            .CLOEXEC = true,
            .CREAT = options.create,
            .TRUNC = options.truncate,
        };
        const fd = try posix.open(path, flags, 0o600);
        errdefer posix.close(fd);

        const stat = try posix.fstat(fd);
        const size: u64 = @intCast(stat.size);
        if (size % options.page_size != 0) return error.FileSizeNotPageAligned;
        const page_count = size / options.page_size;

        var ring = linux.IoUring.init(options.ring_entries, 0) catch return error.IoUringInitFailed;
        errdefer ring.deinit();

        return .{
            .fd = fd,
            .ring = ring,
            .page_size = options.page_size,
            .page_count = page_count,
            .ring_entries = options.ring_entries,
        };
    }

    pub fn close(self: *PagedFile) void {
        self.ring.deinit();
        posix.close(self.fd);
    }

    pub fn pageCount(self: *const PagedFile) u64 {
        return self.page_count;
    }

    /// Extend the file by `n` pages. Returns the first new page number.
    /// New pages are sparse — reads from them return zeros until
    /// written.
    pub fn growBy(self: *PagedFile, n: u64) !u64 {
        const first = self.page_count;
        const new_count = first + n;
        try posix.ftruncate(self.fd, new_count * self.page_size);
        self.page_count = new_count;
        return first;
    }

    pub const IoError = error{
        ShortRead,
        ShortWrite,
        SubmitFailed,
        IoFailed,
        OutOfBounds,
        MisalignedBuffer,
    };

    pub fn readPage(self: *PagedFile, page_no: u64, buf: []u8) IoError!void {
        try self.checkBuf(page_no, buf);
        const offset = page_no * self.page_size;
        _ = self.ring.read(0, self.fd, .{ .buffer = buf[0..self.page_size] }, offset) catch
            return error.SubmitFailed;
        _ = self.ring.submit_and_wait(1) catch return error.SubmitFailed;
        const cqe = self.ring.copy_cqe() catch return error.SubmitFailed;
        if (cqe.res < 0) return error.IoFailed;
        if (@as(u32, @intCast(cqe.res)) != self.page_size) return error.ShortRead;
    }

    pub fn writePage(self: *PagedFile, page_no: u64, buf: []const u8) IoError!void {
        try self.checkBufConst(page_no, buf);
        const offset = page_no * self.page_size;
        _ = self.ring.write(0, self.fd, buf[0..self.page_size], offset) catch
            return error.SubmitFailed;
        _ = self.ring.submit_and_wait(1) catch return error.SubmitFailed;
        const cqe = self.ring.copy_cqe() catch return error.SubmitFailed;
        if (cqe.res < 0) return error.IoFailed;
        if (@as(u32, @intCast(cqe.res)) != self.page_size) return error.ShortWrite;
    }

    pub const PageWrite = struct { page_no: u64, buf: []const u8 };

    /// Batched writes — submit in chunks sized to fit the ring, waiting
    /// for each chunk's completions before submitting the next. This is
    /// the seam group commit will exploit in phase 5.
    pub fn writePages(self: *PagedFile, writes: []const PageWrite) IoError!void {
        if (writes.len == 0) return;
        // Leave headroom so concurrent fsync/other SQEs (later phases)
        // never collide with our writes.
        const chunk: usize = @max(@as(usize, self.ring_entries) / 2, 1);
        var i: usize = 0;
        while (i < writes.len) {
            const end = @min(i + chunk, writes.len);
            try self.submitWriteChunk(writes[i..end]);
            i = end;
        }
    }

    fn submitWriteChunk(self: *PagedFile, writes: []const PageWrite) IoError!void {
        for (writes, 0..) |w, i| {
            try self.checkBufConst(w.page_no, w.buf);
            const offset = w.page_no * self.page_size;
            _ = self.ring.write(@intCast(i), self.fd, w.buf[0..self.page_size], offset) catch
                return error.SubmitFailed;
        }
        const submitted = self.ring.submit_and_wait(@intCast(writes.len)) catch
            return error.SubmitFailed;
        if (submitted < writes.len) return error.SubmitFailed;
        for (0..writes.len) |_| {
            const cqe = self.ring.copy_cqe() catch return error.SubmitFailed;
            if (cqe.res < 0) return error.IoFailed;
            if (@as(u32, @intCast(cqe.res)) != self.page_size) return error.ShortWrite;
        }
    }

    pub fn fsync(self: *PagedFile) IoError!void {
        _ = self.ring.fsync(0, self.fd, 0) catch return error.SubmitFailed;
        _ = self.ring.submit_and_wait(1) catch return error.SubmitFailed;
        const cqe = self.ring.copy_cqe() catch return error.SubmitFailed;
        if (cqe.res < 0) return error.IoFailed;
    }

    fn checkBuf(self: *const PagedFile, page_no: u64, buf: []u8) IoError!void {
        if (page_no >= self.page_count) return error.OutOfBounds;
        if (buf.len < self.page_size) return error.MisalignedBuffer;
        if (!std.mem.isAligned(@intFromPtr(buf.ptr), self.page_size)) return error.MisalignedBuffer;
    }

    fn checkBufConst(self: *const PagedFile, page_no: u64, buf: []const u8) IoError!void {
        if (page_no >= self.page_count) return error.OutOfBounds;
        if (buf.len < self.page_size) return error.MisalignedBuffer;
        if (!std.mem.isAligned(@intFromPtr(buf.ptr), self.page_size)) return error.MisalignedBuffer;
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

/// Open a PagedFile in a tmp dir on real disk. `std.testing.tmpDir`
/// places files under `.zig-cache/tmp/...` on the project's filesystem
/// (not tmpfs), so O_DIRECT is supported.
const TestFile = struct {
    pf: PagedFile,
    tmp: std.testing.TmpDir,

    fn open(opts: PagedFile.OpenOptions) !TestFile {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();

        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &path_buf);
        var full_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&full_buf, "{s}/kvexp.test", .{dir_path});

        const pf = try PagedFile.open(path, opts);
        return .{ .pf = pf, .tmp = tmp };
    }

    fn deinit(self: *TestFile) void {
        self.pf.close();
        self.tmp.cleanup();
    }
};

const ALIGN_4K: std.mem.Alignment = std.mem.Alignment.fromByteUnits(4096);

fn alignedPage(allocator: std.mem.Allocator) ![]align(4096) u8 {
    return try allocator.alignedAlloc(u8, ALIGN_4K, 4096);
}

test "PagedFile: open empty, page_count zero, grow extends" {
    var tf = try TestFile.open(.{ .create = true, .truncate = true });
    defer tf.deinit();

    try testing.expectEqual(@as(u64, 0), tf.pf.pageCount());
    const first = try tf.pf.growBy(4);
    try testing.expectEqual(@as(u64, 0), first);
    try testing.expectEqual(@as(u64, 4), tf.pf.pageCount());

    const next = try tf.pf.growBy(3);
    try testing.expectEqual(@as(u64, 4), next);
    try testing.expectEqual(@as(u64, 7), tf.pf.pageCount());
}

test "PagedFile: write page, read it back" {
    var tf = try TestFile.open(.{ .create = true, .truncate = true });
    defer tf.deinit();

    _ = try tf.pf.growBy(4);

    const buf = try alignedPage(testing.allocator);
    defer testing.allocator.free(buf);

    @memset(buf, 0xAB);
    try tf.pf.writePage(2, buf);

    @memset(buf, 0);
    try tf.pf.readPage(2, buf);
    for (buf) |b| try testing.expectEqual(@as(u8, 0xAB), b);
}

test "PagedFile: batched writePages then read each" {
    var tf = try TestFile.open(.{ .create = true, .truncate = true });
    defer tf.deinit();

    _ = try tf.pf.growBy(4);

    const bufs = try testing.allocator.alignedAlloc(u8, ALIGN_4K, 4 * 4096);
    defer testing.allocator.free(bufs);

    var writes: [4]PagedFile.PageWrite = undefined;
    for (0..4) |i| {
        const slice = bufs[i * 4096 ..][0..4096];
        @memset(slice, @intCast(0x10 + i));
        writes[i] = .{ .page_no = @intCast(i), .buf = slice };
    }
    try tf.pf.writePages(&writes);

    const read_buf = try alignedPage(testing.allocator);
    defer testing.allocator.free(read_buf);
    for (0..4) |i| {
        try tf.pf.readPage(@intCast(i), read_buf);
        for (read_buf) |b| try testing.expectEqual(@as(u8, @intCast(0x10 + i)), b);
    }
}

test "PagedFile: fsync after write succeeds" {
    var tf = try TestFile.open(.{ .create = true, .truncate = true });
    defer tf.deinit();

    _ = try tf.pf.growBy(1);
    const buf = try alignedPage(testing.allocator);
    defer testing.allocator.free(buf);
    @memset(buf, 0x42);
    try tf.pf.writePage(0, buf);
    try tf.pf.fsync();
}

test "PagedFile: read past EOF errors" {
    var tf = try TestFile.open(.{ .create = true, .truncate = true });
    defer tf.deinit();

    _ = try tf.pf.growBy(2);
    const buf = try alignedPage(testing.allocator);
    defer testing.allocator.free(buf);
    try testing.expectError(error.OutOfBounds, tf.pf.readPage(5, buf));
}

test "PagedFile: misaligned buffer rejected" {
    var tf = try TestFile.open(.{ .create = true, .truncate = true });
    defer tf.deinit();

    _ = try tf.pf.growBy(1);
    // Allocate 8KB aligned, then take an unaligned slice starting at +1.
    const big = try testing.allocator.alignedAlloc(u8, ALIGN_4K, 8192);
    defer testing.allocator.free(big);
    const misaligned = big[1..][0..4096];
    try testing.expectError(error.MisalignedBuffer, tf.pf.writePage(0, misaligned));
}
