//! Memory-backed simulator with the same surface as `PagedFile`, plus
//! explicit durability semantics and a fault-injection policy. Designed
//! for property-based crash testing of the storage stack.
//!
//! ## Durability model
//!
//! Two page maps:
//!
//!   * `durable`   — what's "on disk." `_crash()` does not touch this.
//!   * `in_flight` — written but not yet fsynced. Overlaid on top of
//!                   `durable` for reads. Dropped on `_crash()`.
//!
//! Operations:
//!
//!   * `writePage(n, buf)`  — copy buf into in_flight[n].
//!   * `readPage(n, buf)`   — if in_flight has n, return that; else
//!                            durable[n]; else error.OutOfBounds. This
//!                            matches Linux page-cache semantics: a
//!                            process can read its own writes without
//!                            fsync.
//!   * `fsync` / `fdatasync` — move in_flight → durable, atomically per
//!                             page (no torn fsyncs).
//!   * `_crash()`           — clear in_flight; durable survives.
//!
//! ## Fault injection
//!
//! All injection points consult an internal PRNG seeded from the policy.
//! Two seeds with the same policy produce identical fault sequences,
//! which makes failures reproducible and bisectable.
//!
//! Probabilistic faults are expressed as parts-per-million ("ppm") so
//! e.g. `write_fail_ppm = 1000` means 0.1% of writes return error.
//!
//! Scripted one-shots (`fail_next_*`) override probabilistic and clear
//! themselves after firing, so tests can deterministically force a
//! specific call site to fail.
//!
//! Torn writes: when the policy fires a torn write, only a random
//! prefix of the buffer lands in in_flight (truncated at a 512-byte
//! sector boundary, matching real disks). The rest stays as whatever
//! was there before. The next read sees this half-old, half-new page.
//!
//! "Lying fsync": fsync returns success without moving in_flight to
//! durable. On crash, fsynced data is lost — the "kernel ate my data"
//! bug that has bitten Postgres and others.

const std = @import("std");
const paged_file = @import("paged_file.zig");

pub const PagedFileApi = paged_file.PagedFileApi;
pub const PageWrite = paged_file.PageWrite;
pub const IoError = paged_file.IoError;

pub const PAGE_SIZE: u32 = 4096;
const SECTOR_SIZE: u32 = 512;

pub const FaultPolicy = struct {
    seed: u64 = 0xC0DE_F00D,

    /// Per-million probabilities. 1_000_000 = always fire.
    write_fail_ppm: u32 = 0,
    read_fail_ppm: u32 = 0,
    fsync_fail_ppm: u32 = 0,
    torn_write_ppm: u32 = 0,

    /// fsync returns OK but does NOT move in_flight to durable.
    /// Models the Linux "fsyncgate" pattern where a kernel-level
    /// EIO is reported once and then forgotten.
    fsync_lies: bool = false,

    /// One-shot triggers. Set by a test to force the next call to fail.
    /// Cleared after firing.
    fail_next_write: bool = false,
    fail_next_fsync: bool = false,
    fail_next_read: bool = false,
    /// Force the next writePage to be torn (cleared after firing).
    tear_next_write: bool = false,
};

pub const FaultyPagedFile = struct {
    allocator: std.mem.Allocator,
    page_size: u32,

    durable: std.AutoHashMapUnmanaged(u64, [PAGE_SIZE]u8),
    in_flight: std.AutoHashMapUnmanaged(u64, [PAGE_SIZE]u8),

    /// Tracked separately so a crash that drops in_flight pages also
    /// "shrinks" the file back to the size at the last fsync. Grow
    /// is recorded in `pending_size`; fsync moves it to `durable_size`.
    durable_size: u64,
    pending_size: u64,

    pages_written: u64 = 0,
    io_lock: std.Thread.Mutex = .{},

    policy: FaultPolicy,
    rng: std.Random.DefaultPrng,

    pub fn init(
        allocator: std.mem.Allocator,
        page_size: u32,
        initial_pages: u64,
        policy: FaultPolicy,
    ) !FaultyPagedFile {
        std.debug.assert(page_size == PAGE_SIZE);
        return .{
            .allocator = allocator,
            .page_size = page_size,
            .durable = .empty,
            .in_flight = .empty,
            .durable_size = initial_pages,
            .pending_size = initial_pages,
            .policy = policy,
            .rng = std.Random.DefaultPrng.init(policy.seed),
        };
    }

    pub fn deinit(self: *FaultyPagedFile) void {
        self.durable.deinit(self.allocator);
        self.in_flight.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn pageCount(self: *const FaultyPagedFile) u64 {
        return self.pending_size;
    }

    pub fn growBy(self: *FaultyPagedFile, n: u64) !u64 {
        self.io_lock.lock();
        defer self.io_lock.unlock();
        const first = self.pending_size;
        self.pending_size += n;
        return first;
    }

    pub fn readPage(self: *FaultyPagedFile, page_no: u64, buf: []u8) IoError!void {
        try checkBuf(self.page_size, buf);
        self.io_lock.lock();
        defer self.io_lock.unlock();
        if (page_no >= self.pending_size) return error.OutOfBounds;
        if (self.consumeFault(.read)) return error.IoFailed;

        if (self.in_flight.get(page_no)) |bytes| {
            @memcpy(buf[0..PAGE_SIZE], &bytes);
            return;
        }
        if (self.durable.get(page_no)) |bytes| {
            @memcpy(buf[0..PAGE_SIZE], &bytes);
            return;
        }
        // Unwritten page within bounds reads as zeros (matches ftruncate
        // semantics for sparse files).
        @memset(buf[0..PAGE_SIZE], 0);
    }

    pub fn writePage(self: *FaultyPagedFile, page_no: u64, buf: []const u8) IoError!void {
        try checkBufConst(self.page_size, buf);
        self.io_lock.lock();
        defer self.io_lock.unlock();
        if (page_no >= self.pending_size) return error.OutOfBounds;
        if (self.consumeFault(.write)) return error.IoFailed;

        const torn = self.consumeTorn();
        try self.installInFlight(page_no, buf, torn);
        self.pages_written += 1;
    }

    pub fn writePages(self: *FaultyPagedFile, writes: []const PageWrite) IoError!void {
        if (writes.len == 0) return;
        self.io_lock.lock();
        defer self.io_lock.unlock();
        for (writes) |w| {
            try checkBufConst(self.page_size, w.buf);
            if (w.page_no >= self.pending_size) return error.OutOfBounds;
        }
        // A batch fault drops the whole batch (matches submit_and_wait
        // semantics where an SQE rejection aborts the submission).
        if (self.consumeFault(.write)) return error.IoFailed;

        for (writes) |w| {
            const torn = self.consumeTorn();
            try self.installInFlight(w.page_no, w.buf, torn);
        }
        self.pages_written += writes.len;
    }

    pub fn fsync(self: *FaultyPagedFile) IoError!void {
        return self.syncImpl();
    }

    pub fn fdatasync(self: *FaultyPagedFile) IoError!void {
        return self.syncImpl();
    }

    fn syncImpl(self: *FaultyPagedFile) IoError!void {
        self.io_lock.lock();
        defer self.io_lock.unlock();
        if (self.consumeFault(.fsync)) return error.IoFailed;
        if (self.policy.fsync_lies) {
            // Return success but leave in_flight pages non-durable.
            // The next _crash() will eat them.
            return;
        }
        // Atomic per-page promotion: every in_flight page → durable.
        var it = self.in_flight.iterator();
        while (it.next()) |e| {
            try self.durable.put(self.allocator, e.key_ptr.*, e.value_ptr.*);
        }
        self.in_flight.clearRetainingCapacity();
        self.durable_size = self.pending_size;
    }

    /// Test hook: simulate a crash. Drops in_flight pages and shrinks
    /// pending_size back to durable_size. Does NOT clear policy or RNG
    /// — the same seed continues to drive faults after reopen, which
    /// is desirable for deterministic test replay across the crash.
    pub fn _crash(self: *FaultyPagedFile) void {
        self.io_lock.lock();
        defer self.io_lock.unlock();
        self.in_flight.clearRetainingCapacity();
        self.pending_size = self.durable_size;
    }

    /// Test introspection.
    pub fn _durablePageCount(self: *const FaultyPagedFile) u64 {
        return self.durable_size;
    }

    pub fn _inFlightCount(self: *const FaultyPagedFile) u32 {
        return self.in_flight.count();
    }

    // ── internals ──────────────────────────────────────────────────

    const FaultKind = enum { read, write, fsync };

    /// Returns true iff a fault should fire for `kind` now. Consumes a
    /// one-shot flag if applicable. Caller holds io_lock.
    fn consumeFault(self: *FaultyPagedFile, kind: FaultKind) bool {
        switch (kind) {
            .read => {
                if (self.policy.fail_next_read) {
                    self.policy.fail_next_read = false;
                    return true;
                }
                return self.fire(self.policy.read_fail_ppm);
            },
            .write => {
                if (self.policy.fail_next_write) {
                    self.policy.fail_next_write = false;
                    return true;
                }
                return self.fire(self.policy.write_fail_ppm);
            },
            .fsync => {
                if (self.policy.fail_next_fsync) {
                    self.policy.fail_next_fsync = false;
                    return true;
                }
                return self.fire(self.policy.fsync_fail_ppm);
            },
        }
    }

    fn consumeTorn(self: *FaultyPagedFile) bool {
        if (self.policy.tear_next_write) {
            self.policy.tear_next_write = false;
            return true;
        }
        return self.fire(self.policy.torn_write_ppm);
    }

    fn fire(self: *FaultyPagedFile, ppm: u32) bool {
        if (ppm == 0) return false;
        const roll = self.rng.random().intRangeLessThan(u32, 0, 1_000_000);
        return roll < ppm;
    }

    /// Copy buf into in_flight[page_no]. If `torn`, only a random
    /// sector-aligned prefix is overwritten; the rest of the page
    /// keeps whatever was there before (in_flight or durable or zeros).
    fn installInFlight(self: *FaultyPagedFile, page_no: u64, buf: []const u8, torn: bool) !void {
        const gop = try self.in_flight.getOrPut(self.allocator, page_no);
        if (!gop.found_existing) {
            // Seed with whatever was previously visible at this page.
            if (self.durable.get(page_no)) |prev| {
                gop.value_ptr.* = prev;
            } else {
                @memset(&gop.value_ptr.*, 0);
            }
        }
        if (!torn) {
            @memcpy(&gop.value_ptr.*, buf[0..PAGE_SIZE]);
            return;
        }
        // Torn write: pick a sector-aligned prefix length in [1, 7] of
        // SECTOR_SIZE bytes. The rest stays as the previous content.
        const sectors_per_page = PAGE_SIZE / SECTOR_SIZE;
        const survived_sectors: u32 = self.rng.random().intRangeAtMost(u32, 1, sectors_per_page - 1);
        const cutoff: usize = survived_sectors * SECTOR_SIZE;
        @memcpy(gop.value_ptr.*[0..cutoff], buf[0..cutoff]);
        // Bytes [cutoff..PAGE_SIZE) keep the previous in_flight/durable
        // content — exactly the half-old/half-new state a real
        // partial-sector flush produces.
    }

    fn checkBuf(page_size: u32, buf: []u8) IoError!void {
        if (buf.len < page_size) return error.MisalignedBuffer;
    }

    fn checkBufConst(page_size: u32, buf: []const u8) IoError!void {
        if (buf.len < page_size) return error.MisalignedBuffer;
    }

    /// Hand the simulator to PageCache / Manifest as a `PagedFileApi`.
    pub fn api(self: *FaultyPagedFile) PagedFileApi {
        return .{ .ptr = self, .vtable = &api_vtable };
    }

    const api_vtable: PagedFileApi.VTable = .{
        .readPageFn = apiReadPage,
        .writePageFn = apiWritePage,
        .writePagesFn = apiWritePages,
        .fsyncFn = apiFsync,
        .fdatasyncFn = apiFdatasync,
        .growByFn = apiGrowBy,
        .pageCountFn = apiPageCount,
    };

    fn apiReadPage(ptr: *anyopaque, page_no: u64, buf: []u8) IoError!void {
        const self: *FaultyPagedFile = @ptrCast(@alignCast(ptr));
        return self.readPage(page_no, buf);
    }
    fn apiWritePage(ptr: *anyopaque, page_no: u64, buf: []const u8) IoError!void {
        const self: *FaultyPagedFile = @ptrCast(@alignCast(ptr));
        return self.writePage(page_no, buf);
    }
    fn apiWritePages(ptr: *anyopaque, writes: []const PageWrite) IoError!void {
        const self: *FaultyPagedFile = @ptrCast(@alignCast(ptr));
        return self.writePages(writes);
    }
    fn apiFsync(ptr: *anyopaque) IoError!void {
        const self: *FaultyPagedFile = @ptrCast(@alignCast(ptr));
        return self.fsync();
    }
    fn apiFdatasync(ptr: *anyopaque) IoError!void {
        const self: *FaultyPagedFile = @ptrCast(@alignCast(ptr));
        return self.fdatasync();
    }
    fn apiGrowBy(ptr: *anyopaque, n: u64) anyerror!u64 {
        const self: *FaultyPagedFile = @ptrCast(@alignCast(ptr));
        return self.growBy(n);
    }
    fn apiPageCount(ptr: *anyopaque) u64 {
        const self: *FaultyPagedFile = @ptrCast(@alignCast(ptr));
        return self.pageCount();
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

test "FaultyPagedFile: basic write/read round-trip" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{});
    defer f.deinit();

    var buf: [PAGE_SIZE]u8 = undefined;
    @memset(&buf, 0xAB);
    try f.writePage(2, &buf);

    var got: [PAGE_SIZE]u8 = undefined;
    try f.readPage(2, &got);
    try testing.expectEqualSlices(u8, &buf, &got);
}

test "FaultyPagedFile: reads see own writes before fsync" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{});
    defer f.deinit();

    var buf: [PAGE_SIZE]u8 = undefined;
    @memset(&buf, 0x11);
    try f.writePage(0, &buf);
    // No fsync.
    var got: [PAGE_SIZE]u8 = undefined;
    try f.readPage(0, &got);
    try testing.expectEqualSlices(u8, &buf, &got);
}

test "FaultyPagedFile: crash drops un-fsynced writes" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{});
    defer f.deinit();

    var buf: [PAGE_SIZE]u8 = undefined;
    @memset(&buf, 0x11);
    try f.writePage(0, &buf);
    @memset(&buf, 0x22);
    try f.writePage(1, &buf);
    try f.fsync();

    @memset(&buf, 0x33);
    try f.writePage(2, &buf); // Not fsynced.

    f._crash();

    // Pages 0 and 1 survive; page 2 is gone.
    var got: [PAGE_SIZE]u8 = undefined;
    try f.readPage(0, &got);
    for (got) |b| try testing.expectEqual(@as(u8, 0x11), b);
    try f.readPage(1, &got);
    for (got) |b| try testing.expectEqual(@as(u8, 0x22), b);
    try f.readPage(2, &got);
    for (got) |b| try testing.expectEqual(@as(u8, 0), b); // zero-fill
}

test "FaultyPagedFile: growBy is rolled back by crash" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{});
    defer f.deinit();
    try testing.expectEqual(@as(u64, 4), f.pageCount());

    _ = try f.growBy(10);
    try testing.expectEqual(@as(u64, 14), f.pageCount());

    f._crash();
    try testing.expectEqual(@as(u64, 4), f.pageCount());

    // After fsync, growth is durable.
    _ = try f.growBy(10);
    try f.fsync();
    f._crash();
    try testing.expectEqual(@as(u64, 14), f.pageCount());
}

test "FaultyPagedFile: scripted fail_next_fsync" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{ .fail_next_fsync = true });
    defer f.deinit();
    var buf: [PAGE_SIZE]u8 = undefined;
    @memset(&buf, 0xCC);
    try f.writePage(0, &buf);

    try testing.expectError(error.IoFailed, f.fsync());
    // Flag self-clears after firing.
    try f.fsync();
}

test "FaultyPagedFile: scripted fail_next_write" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{ .fail_next_write = true });
    defer f.deinit();
    var buf: [PAGE_SIZE]u8 = undefined;
    @memset(&buf, 0xCC);
    try testing.expectError(error.IoFailed, f.writePage(0, &buf));
    try f.writePage(0, &buf); // Now succeeds.
}

test "FaultyPagedFile: lying fsync — crash loses 'durable' writes" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{ .fsync_lies = true });
    defer f.deinit();
    var buf: [PAGE_SIZE]u8 = undefined;
    @memset(&buf, 0xAA);
    try f.writePage(0, &buf);
    try f.fsync(); // Lies! Returns OK but doesn't actually durabilize.

    f._crash();
    // After crash: in-flight is empty (the crash dropped it), and the
    // page that "fsynced successfully" reads back as zeros — the lying
    // fsync never actually moved it to durable storage.
    try testing.expectEqual(@as(u32, 0), f._inFlightCount());
    var got: [PAGE_SIZE]u8 = undefined;
    try f.readPage(0, &got);
    for (got) |b| try testing.expectEqual(@as(u8, 0), b);
}

test "FaultyPagedFile: torn write leaves a half-old/half-new page" {
    // Set up: page 0 is fully fsynced as all-0xAA. Then write all-0xBB
    // as a torn write — some sectors land, some keep 0xAA.
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 4, .{ .seed = 1 });
    defer f.deinit();
    var buf: [PAGE_SIZE]u8 = undefined;
    @memset(&buf, 0xAA);
    try f.writePage(0, &buf);
    try f.fsync();

    f.policy.tear_next_write = true;
    @memset(&buf, 0xBB);
    try f.writePage(0, &buf);

    var got: [PAGE_SIZE]u8 = undefined;
    try f.readPage(0, &got);
    var saw_old = false;
    var saw_new = false;
    for (got) |b| {
        if (b == 0xAA) saw_old = true;
        if (b == 0xBB) saw_new = true;
    }
    try testing.expect(saw_old);
    try testing.expect(saw_new);
}

test "FaultyPagedFile: probabilistic failure honors seed" {
    // Same seed + same policy should produce identical failure pattern.
    var failures_a: u32 = 0;
    {
        var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 1024, .{
            .seed = 0xDEADBEEF,
            .write_fail_ppm = 100_000, // 10%
        });
        defer f.deinit();
        var buf: [PAGE_SIZE]u8 = undefined;
        @memset(&buf, 0xCC);
        var i: u64 = 0;
        while (i < 1000) : (i += 1) {
            f.writePage(i % 1024, &buf) catch {
                failures_a += 1;
            };
        }
    }

    var failures_b: u32 = 0;
    {
        var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 1024, .{
            .seed = 0xDEADBEEF,
            .write_fail_ppm = 100_000,
        });
        defer f.deinit();
        var buf: [PAGE_SIZE]u8 = undefined;
        @memset(&buf, 0xCC);
        var i: u64 = 0;
        while (i < 1000) : (i += 1) {
            f.writePage(i % 1024, &buf) catch {
                failures_b += 1;
            };
        }
    }

    try testing.expectEqual(failures_a, failures_b);
    // 10% of 1000 ≈ 100; allow a wide band so the test is robust.
    try testing.expect(failures_a > 50);
    try testing.expect(failures_a < 200);
}

test "FaultyPagedFile: read past EOF errors" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 2, .{});
    defer f.deinit();
    var buf: [PAGE_SIZE]u8 = undefined;
    try testing.expectError(error.OutOfBounds, f.readPage(5, &buf));
}

test "FaultyPagedFile: writePages batches semantics" {
    var f = try FaultyPagedFile.init(testing.allocator, PAGE_SIZE, 8, .{});
    defer f.deinit();
    const bufs = try testing.allocator.alloc(u8, 4 * PAGE_SIZE);
    defer testing.allocator.free(bufs);
    var writes: [4]PageWrite = undefined;
    for (0..4) |i| {
        const slice = bufs[i * PAGE_SIZE ..][0..PAGE_SIZE];
        @memset(slice, @intCast(0x10 + i));
        writes[i] = .{ .page_no = i, .buf = slice };
    }
    try f.writePages(&writes);
    try f.fsync();

    var got: [PAGE_SIZE]u8 = undefined;
    for (0..4) |i| {
        try f.readPage(i, &got);
        for (got) |b| try testing.expectEqual(@as(u8, @intCast(0x10 + i)), b);
    }
}
