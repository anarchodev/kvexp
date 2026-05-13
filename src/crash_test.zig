//! Real-crash recovery tests. The parent process spawns itself as a
//! child (via argv "child <scenario> <path>"), the child runs scripted
//! kvexp operations and writes "READY\n" to stdout once it has reached
//! the steady state to be tested. The parent reads that token, SIGKILLs
//! the child (no destructors, no fsyncs that hadn't already happened),
//! then opens the same data file and asserts the recovery contract.
//!
//! Why a separate binary instead of a unit test:
//!  - SIGKILL leaves no chance for in-process cleanup, which is the
//!    point — only LMDB's durable state can be visible after.
//!  - Subprocess control belongs out of `zig build test`. Invoke via
//!    `zig build crash-test`.

const std = @import("std");
const kvexp = @import("kvexp");

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len >= 2 and std.mem.eql(u8, args[1], "child")) {
        if (args.len < 4) {
            std.debug.print("usage: crash-test child <scenario> <path>\n", .{});
            std.process.exit(2);
        }
        runChild(allocator, args[2], args[3]) catch |e| {
            std.debug.print("child failed: {}\n", .{e});
            std.process.exit(3);
        };
        return;
    }

    runAllScenarios(allocator) catch |e| {
        std.debug.print("crash-test parent failed: {}\n", .{e});
        std.process.exit(1);
    };
}

// ─── parent ────────────────────────────────────────────────────────

fn runAllScenarios(allocator: std.mem.Allocator) !void {
    var failed: usize = 0;
    inline for (.{
        "no-durabilize",
        "after-durabilize",
        "between-durabilizes",
    }) |scenario| {
        std.debug.print("scenario: {s:<22} ... ", .{scenario});
        if (runScenario(allocator, scenario)) {
            std.debug.print("OK\n", .{});
        } else |e| {
            std.debug.print("FAIL: {}\n", .{e});
            failed += 1;
        }
    }
    if (failed > 0) return error.ScenariosFailed;
}

fn runScenario(allocator: std.mem.Allocator, scenario: []const u8) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try tmp.dir.realpath(".", &dir_buf);
    const path_tmp = try std.fmt.allocPrint(allocator, "{s}/crash.mdb", .{dir_path});
    defer allocator.free(path_tmp);
    const path = try allocator.dupeZ(u8, path_tmp);
    defer allocator.free(path);

    const self_exe = try std.fs.selfExePathAlloc(allocator);
    defer allocator.free(self_exe);

    const argv = &[_][]const u8{ self_exe, "child", scenario, path };
    var child = std.process.Child.init(argv, allocator);
    child.stdout_behavior = .Pipe;
    child.stdin_behavior = .Pipe;
    try child.spawn();

    // Read child stdout until we see "READY" (or the pipe closes).
    var collected: [256]u8 = undefined;
    var collected_len: usize = 0;
    while (true) {
        if (collected_len >= collected.len) return error.ReadyTokenNotFound;
        const n = try child.stdout.?.read(collected[collected_len..]);
        if (n == 0) {
            // Child closed stdout (probably exited) before READY.
            _ = child.wait() catch {};
            return error.ChildDiedBeforeReady;
        }
        collected_len += n;
        if (std.mem.indexOf(u8, collected[0..collected_len], "READY") != null) break;
    }

    // SIGKILL — unstoppable, no destructors, no last-chance fsyncs.
    try std.posix.kill(child.id, std.posix.SIG.KILL);
    _ = try child.wait();

    // Reopen and verify recovery.
    var mf: kvexp.Manifest = undefined;
    try mf.init(allocator, path, .{
        .max_map_size = 4 * 1024 * 1024,
        .max_stores = 16,
    });
    defer mf.deinit();

    if (std.mem.eql(u8, scenario, "no-durabilize")) {
        // Child created store 1 but never durabilized. Nothing visible.
        try expectEqual(false, try mf.hasStore(1));
        try expectEqual(@as(u64, 0), try mf.durableRaftIdx());
    } else if (std.mem.eql(u8, scenario, "after-durabilize")) {
        try expectEqual(true, try mf.hasStore(1));
        try expectEqual(@as(u64, 42), try mf.durableRaftIdx());
        var lease = try mf.acquire(1);
        defer lease.release();
        const got = (try lease.get(allocator, "k")) orelse return error.MissingKey;
        defer allocator.free(got);
        try expectEqualStrings("v", got);
    } else if (std.mem.eql(u8, scenario, "between-durabilizes")) {
        try expectEqual(true, try mf.hasStore(1));
        // raft_idx is the *first* durabilize's value (10); the second
        // never ran, so v2 isn't durable.
        try expectEqual(@as(u64, 10), try mf.durableRaftIdx());
        var lease = try mf.acquire(1);
        defer lease.release();
        const got = (try lease.get(allocator, "k")) orelse return error.MissingKey;
        defer allocator.free(got);
        try expectEqualStrings("v1", got);
    } else {
        return error.UnknownScenario;
    }
}

// ─── child ─────────────────────────────────────────────────────────

fn runChild(allocator: std.mem.Allocator, scenario: []const u8, path_arg: []const u8) !void {
    const path = try allocator.dupeZ(u8, path_arg);
    defer allocator.free(path);

    var mf: kvexp.Manifest = undefined;
    try mf.init(allocator, path, .{
        .max_map_size = 4 * 1024 * 1024,
        .max_stores = 16,
    });
    // No `defer mf.deinit()` — we're going to be SIGKILLed.

    if (std.mem.eql(u8, scenario, "no-durabilize")) {
        try mf.createStore(1);
        // No durabilize. Lifecycle change lives only in pending_creates.
    } else if (std.mem.eql(u8, scenario, "after-durabilize")) {
        try mf.createStore(1);
        {
            var lease = try mf.acquire(1);
            defer lease.release();
            var t = try lease.beginTxn();
            try t.put("k", "v");
            try t.commit();
        }
        try mf.durabilize(42);
    } else if (std.mem.eql(u8, scenario, "between-durabilizes")) {
        try mf.createStore(1);
        {
            var lease = try mf.acquire(1);
            defer lease.release();
            var t = try lease.beginTxn();
            try t.put("k", "v1");
            try t.commit();
        }
        try mf.durabilize(10);
        {
            var lease = try mf.acquire(1);
            defer lease.release();
            var t = try lease.beginTxn();
            try t.put("k", "v2");
            try t.commit();
        }
        // *Intentionally* no second durabilize.
    } else {
        std.debug.print("unknown scenario: {s}\n", .{scenario});
        std.process.exit(2);
    }

    // Tell the parent we're at the steady state.
    const stdout = std.fs.File.stdout();
    try stdout.writeAll("READY\n");

    // Block forever. Parent SIGKILLs us before this can return.
    const stdin = std.fs.File.stdin();
    var buf: [256]u8 = undefined;
    while (true) {
        _ = stdin.read(&buf) catch break;
    }
}

// ─── tiny assertion helpers ────────────────────────────────────────

fn expectEqual(expected: anytype, actual: @TypeOf(expected)) !void {
    if (expected != actual) {
        std.debug.print("expected {any}, got {any}\n", .{ expected, actual });
        return error.MismatchedValue;
    }
}

fn expectEqualStrings(expected: []const u8, actual: []const u8) !void {
    if (!std.mem.eql(u8, expected, actual)) {
        std.debug.print("expected '{s}', got '{s}'\n", .{ expected, actual });
        return error.MismatchedString;
    }
}
