//! kvexp benchmarks. Run with `zig build bench` — the build step
//! forces ReleaseFast so unrelated debug-mode safety checks don't
//! distort the numbers.
//!
//! Each benchmark:
//!  1. Initializes a fresh manifest in /tmp.
//!  2. Optionally pre-populates state.
//!  3. Records per-op wall-clock latency via std.time.Timer.
//!  4. Sorts latencies and reports throughput + p50/p99/p999.
//!
//! Latencies for every op are kept; sort + percentile is O(n log n)
//! which dominates the post-run accounting, not the measured loop.

const std = @import("std");
const kvexp = @import("kvexp");

const PATH: [:0]const u8 = "/tmp/kvexp-bench.mdb";
const LOCK_PATH: [:0]const u8 = "/tmp/kvexp-bench.mdb-lock";
const VAL_SIZE: usize = 100;
const KEY_LEN: usize = 16;

// ─── reporting ─────────────────────────────────────────────────────

const Sample = struct {
    name: []const u8,
    ops: usize,
    total_ns: u64,
    /// Sorted ascending. Owned.
    latencies_ns: []u64,

    fn deinit(self: *Sample, allocator: std.mem.Allocator) void {
        allocator.free(self.latencies_ns);
    }

    fn throughput(self: Sample) f64 {
        if (self.total_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.ops)) * 1e9 / @as(f64, @floatFromInt(self.total_ns));
    }

    fn percentile(self: Sample, p: f64) u64 {
        if (self.latencies_ns.len == 0) return 0;
        const idx_f = p * @as(f64, @floatFromInt(self.latencies_ns.len - 1));
        const idx = @as(usize, @intFromFloat(@round(idx_f)));
        return self.latencies_ns[idx];
    }
};

fn fmtNs(buf: []u8, ns: u64) []const u8 {
    if (ns < 1_000) return std.fmt.bufPrint(buf, "{d}ns", .{ns}) catch "?";
    if (ns < 1_000_000) return std.fmt.bufPrint(buf, "{d:.2}us", .{
        @as(f64, @floatFromInt(ns)) / 1_000,
    }) catch "?";
    if (ns < 1_000_000_000) return std.fmt.bufPrint(buf, "{d:.2}ms", .{
        @as(f64, @floatFromInt(ns)) / 1_000_000,
    }) catch "?";
    return std.fmt.bufPrint(buf, "{d:.2}s", .{
        @as(f64, @floatFromInt(ns)) / 1_000_000_000,
    }) catch "?";
}

fn printSample(s: Sample) void {
    var b50: [16]u8 = undefined;
    var b99: [16]u8 = undefined;
    var b999: [16]u8 = undefined;
    std.debug.print(
        "  {s:<48} {d:>10} ops  {d:>12.0} ops/sec  p50={s:>8}  p99={s:>8}  p999={s:>8}\n",
        .{
            s.name,
            s.ops,
            s.throughput(),
            fmtNs(&b50, s.percentile(0.5)),
            fmtNs(&b99, s.percentile(0.99)),
            fmtNs(&b999, s.percentile(0.999)),
        },
    );
}

// ─── helpers ───────────────────────────────────────────────────────

fn cleanFiles() void {
    std.fs.cwd().deleteFile(PATH) catch {};
    std.fs.cwd().deleteFile(LOCK_PATH) catch {};
}

fn freshManifest(allocator: std.mem.Allocator) !*kvexp.Manifest {
    cleanFiles();
    const m = try allocator.create(kvexp.Manifest);
    errdefer allocator.destroy(m);
    try m.init(allocator, PATH, .{
        .max_map_size = 1 << 30, // 1 GiB
        .max_stores = 1024,
    });
    return m;
}

fn destroyManifest(allocator: std.mem.Allocator, m: *kvexp.Manifest) void {
    m.deinit();
    allocator.destroy(m);
}

fn writeKey(buf: *[KEY_LEN]u8, i: usize) []const u8 {
    return std.fmt.bufPrint(buf, "k{d:0>15}", .{i}) catch unreachable;
}

// ─── benchmarks ────────────────────────────────────────────────────

fn benchPutCommit(allocator: std.mem.Allocator) !Sample {
    const m = try freshManifest(allocator);
    defer destroyManifest(allocator, m);
    try m.createStore(1);
    try m.flush();
    var lease = try m.acquire(1);
    defer lease.release();

    const ops: usize = 100_000;
    const latencies = try allocator.alloc(u64, ops);
    var key_buf: [KEY_LEN]u8 = undefined;
    const value = [_]u8{0} ** VAL_SIZE;

    var timer = try std.time.Timer.start();
    const start = timer.read();
    var i: usize = 0;
    while (i < ops) : (i += 1) {
        const key = writeKey(&key_buf, i);
        const op_start = timer.read();
        var t = try lease.beginTxn();
        try t.put(key, &value);
        try t.commit();
        latencies[i] = timer.read() - op_start;
    }
    const total = timer.read() - start;
    std.mem.sort(u64, latencies, {}, std.sort.asc(u64));
    return .{
        .name = "put+commit  (1 tenant, txn-per-put)",
        .ops = ops,
        .total_ns = total,
        .latencies_ns = latencies,
    };
}

fn benchPutBatch(allocator: std.mem.Allocator) !Sample {
    const m = try freshManifest(allocator);
    defer destroyManifest(allocator, m);
    try m.createStore(1);
    try m.flush();
    var lease = try m.acquire(1);
    defer lease.release();

    const batch_size: usize = 100;
    const commits: usize = 1_000;
    const total_ops: usize = batch_size * commits;
    const latencies = try allocator.alloc(u64, total_ops);
    var key_buf: [KEY_LEN]u8 = undefined;
    const value = [_]u8{0} ** VAL_SIZE;

    var timer = try std.time.Timer.start();
    const start = timer.read();
    var i: usize = 0;
    var c: usize = 0;
    while (c < commits) : (c += 1) {
        var t = try lease.beginTxn();
        var j: usize = 0;
        while (j < batch_size) : (j += 1) {
            const key = writeKey(&key_buf, i);
            const op_start = timer.read();
            try t.put(key, &value);
            latencies[i] = timer.read() - op_start;
            i += 1;
        }
        try t.commit();
    }
    const total = timer.read() - start;
    std.mem.sort(u64, latencies, {}, std.sort.asc(u64));
    return .{
        .name = "put        (1 tenant, batch 100/txn)",
        .ops = total_ops,
        .total_ns = total,
        .latencies_ns = latencies,
    };
}

fn benchGetOverlay(allocator: std.mem.Allocator) !Sample {
    const m = try freshManifest(allocator);
    defer destroyManifest(allocator, m);
    try m.createStore(1);
    try m.flush();
    const num_keys: usize = 10_000;
    {
        var lease = try m.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        var key_buf: [KEY_LEN]u8 = undefined;
        const value = [_]u8{42} ** VAL_SIZE;
        var k: usize = 0;
        while (k < num_keys) : (k += 1) {
            const key = writeKey(&key_buf, k);
            try t.put(key, &value);
        }
        try t.commit();
        // Don't flush — keep all data in main_overlay.
    }
    var lease = try m.acquire(1);
    defer lease.release();

    const ops: usize = 100_000;
    const latencies = try allocator.alloc(u64, ops);
    var rng = std.Random.DefaultPrng.init(42);
    var key_buf: [KEY_LEN]u8 = undefined;

    var timer = try std.time.Timer.start();
    const start = timer.read();
    var i: usize = 0;
    while (i < ops) : (i += 1) {
        const k = rng.random().uintLessThan(usize, num_keys);
        const key = writeKey(&key_buf, k);
        const op_start = timer.read();
        const got = try lease.get(allocator, key);
        latencies[i] = timer.read() - op_start;
        if (got) |g| allocator.free(g);
    }
    const total = timer.read() - start;
    std.mem.sort(u64, latencies, {}, std.sort.asc(u64));
    return .{
        .name = "lease.get   (overlay hit, 10k keys)",
        .ops = ops,
        .total_ns = total,
        .latencies_ns = latencies,
    };
}

fn benchGetLmdb(allocator: std.mem.Allocator) !Sample {
    const m = try freshManifest(allocator);
    defer destroyManifest(allocator, m);
    try m.createStore(1);
    const num_keys: usize = 10_000;
    {
        var lease = try m.acquire(1);
        defer lease.release();
        var t = try lease.beginTxn();
        var key_buf: [KEY_LEN]u8 = undefined;
        const value = [_]u8{42} ** VAL_SIZE;
        var k: usize = 0;
        while (k < num_keys) : (k += 1) {
            const key = writeKey(&key_buf, k);
            try t.put(key, &value);
        }
        try t.commit();
    }
    try m.flush(); // drain to LMDB so reads bypass main_overlay
    var lease = try m.acquire(1);
    defer lease.release();

    const ops: usize = 100_000;
    const latencies = try allocator.alloc(u64, ops);
    var rng = std.Random.DefaultPrng.init(42);
    var key_buf: [KEY_LEN]u8 = undefined;

    var timer = try std.time.Timer.start();
    const start = timer.read();
    var i: usize = 0;
    while (i < ops) : (i += 1) {
        const k = rng.random().uintLessThan(usize, num_keys);
        const key = writeKey(&key_buf, k);
        const op_start = timer.read();
        const got = try lease.get(allocator, key);
        latencies[i] = timer.read() - op_start;
        if (got) |g| allocator.free(g);
    }
    const total = timer.read() - start;
    std.mem.sort(u64, latencies, {}, std.sort.asc(u64));
    return .{
        .name = "lease.get   (LMDB hit, 10k keys flushed)",
        .ops = ops,
        .total_ns = total,
        .latencies_ns = latencies,
    };
}

fn benchDurabilize(allocator: std.mem.Allocator) !Sample {
    const m = try freshManifest(allocator);
    defer destroyManifest(allocator, m);
    try m.createStore(1);
    try m.flush();

    const iterations: usize = 100;
    const writes_per_dur: usize = 1_000;
    const latencies = try allocator.alloc(u64, iterations);

    var key_buf: [KEY_LEN]u8 = undefined;
    const value = [_]u8{0} ** VAL_SIZE;

    var timer = try std.time.Timer.start();
    const start = timer.read();
    var iter: usize = 0;
    var key_counter: usize = 0;
    while (iter < iterations) : (iter += 1) {
        // Setup phase (not counted): accumulate writes_per_dur entries
        // in main_overlay.
        {
            var lease = try m.acquire(1);
            defer lease.release();
            var t = try lease.beginTxn();
            var j: usize = 0;
            while (j < writes_per_dur) : (j += 1) {
                const key = writeKey(&key_buf, key_counter);
                try t.put(key, &value);
                key_counter += 1;
            }
            try t.commit();
        }
        // Measured: the durabilize itself.
        const dur_start = timer.read();
        try m.flush();
        latencies[iter] = timer.read() - dur_start;
    }
    const total = timer.read() - start;
    std.mem.sort(u64, latencies, {}, std.sort.asc(u64));
    return .{
        .name = "durabilize  (1k writes accumulated)",
        .ops = iterations,
        .total_ns = total,
        .latencies_ns = latencies,
    };
}

// ─── multi-threaded ────────────────────────────────────────────────

const WorkerCtx = struct {
    manifest: *kvexp.Manifest,
    tenant: u64,
    ops: usize,
    latencies: []u64,
    err: std.atomic.Value(usize) = .init(0),
};

fn putWorker(ctx: *WorkerCtx) void {
    putWorkerInner(ctx) catch |e| {
        _ = ctx.err.cmpxchgStrong(0, @intFromError(e), .acq_rel, .acquire);
    };
}

fn putWorkerInner(ctx: *WorkerCtx) !void {
    var lease = try ctx.manifest.acquire(ctx.tenant);
    defer lease.release();
    var key_buf: [KEY_LEN]u8 = undefined;
    const value = [_]u8{0} ** VAL_SIZE;
    var timer = try std.time.Timer.start();
    var i: usize = 0;
    while (i < ctx.ops) : (i += 1) {
        const key = writeKey(&key_buf, i);
        const op_start = timer.read();
        var t = try lease.beginTxn();
        try t.put(key, &value);
        try t.commit();
        ctx.latencies[i] = timer.read() - op_start;
    }
}

fn benchMultiTenantParallel(
    allocator: std.mem.Allocator,
    num_threads: usize,
    name: []const u8,
) !Sample {
    const m = try freshManifest(allocator);
    defer destroyManifest(allocator, m);
    var i: usize = 0;
    while (i < num_threads) : (i += 1) try m.createStore(@intCast(i));
    try m.flush();

    const ops_per_thread: usize = 25_000;
    const total_ops = ops_per_thread * num_threads;

    var per_thread_lats: [][]u64 = try allocator.alloc([]u64, num_threads);
    defer {
        for (per_thread_lats) |p| allocator.free(p);
        allocator.free(per_thread_lats);
    }
    var ctxs = try allocator.alloc(WorkerCtx, num_threads);
    defer allocator.free(ctxs);
    var threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);

    var t_idx: usize = 0;
    while (t_idx < num_threads) : (t_idx += 1) {
        per_thread_lats[t_idx] = try allocator.alloc(u64, ops_per_thread);
        ctxs[t_idx] = .{
            .manifest = m,
            .tenant = @intCast(t_idx),
            .ops = ops_per_thread,
            .latencies = per_thread_lats[t_idx],
        };
    }

    var timer = try std.time.Timer.start();
    const start = timer.read();
    t_idx = 0;
    while (t_idx < num_threads) : (t_idx += 1) {
        threads[t_idx] = try std.Thread.spawn(.{}, putWorker, .{&ctxs[t_idx]});
    }
    for (threads) |th| th.join();
    const total = timer.read() - start;

    for (ctxs) |c| {
        if (c.err.load(.acquire) != 0) return error.WorkerErrored;
    }

    const all_lats = try allocator.alloc(u64, total_ops);
    var off: usize = 0;
    for (per_thread_lats) |p| {
        @memcpy(all_lats[off..][0..p.len], p);
        off += p.len;
    }
    std.mem.sort(u64, all_lats, {}, std.sort.asc(u64));

    return .{
        .name = name,
        .ops = total_ops,
        .total_ns = total,
        .latencies_ns = all_lats,
    };
}

// ─── main ──────────────────────────────────────────────────────────

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    std.debug.print("\nkvexp benchmark — keys 16B, values 100B, ReleaseFast\n", .{});
    std.debug.print("==============================================================\n\n", .{});

    var samples: [16]Sample = undefined;
    var count: usize = 0;

    samples[count] = try benchPutCommit(allocator);
    count += 1;
    samples[count] = try benchPutBatch(allocator);
    count += 1;
    samples[count] = try benchGetOverlay(allocator);
    count += 1;
    samples[count] = try benchGetLmdb(allocator);
    count += 1;
    samples[count] = try benchDurabilize(allocator);
    count += 1;
    samples[count] = try benchMultiTenantParallel(allocator, 1, "parallel    (1 thread,  1 tenant)");
    count += 1;
    samples[count] = try benchMultiTenantParallel(allocator, 2, "parallel    (2 threads, 2 tenants)");
    count += 1;
    samples[count] = try benchMultiTenantParallel(allocator, 4, "parallel    (4 threads, 4 tenants)");
    count += 1;
    samples[count] = try benchMultiTenantParallel(allocator, 8, "parallel    (8 threads, 8 tenants)");
    count += 1;

    for (samples[0..count]) |*s| {
        printSample(s.*);
        s.deinit(allocator);
    }

    std.debug.print("\n", .{});
    cleanFiles();
}
