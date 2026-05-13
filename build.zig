const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const kvexp_mod = b.addModule("kvexp", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Link against the system's liblmdb. lmdb.h is consumed via
    // @cImport at the call site (src/lmdb.zig).
    kvexp_mod.linkSystemLibrary("lmdb", .{});
    kvexp_mod.link_libc = true;

    const kvexp_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "kvexp",
        .root_module = kvexp_mod,
    });
    b.installArtifact(kvexp_lib);

    const test_step = b.step("test", "Run kvexp unit tests");
    const tests = b.addTest(.{ .root_module = kvexp_mod });
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);

    // Benchmark executable. Always built with ReleaseFast so unrelated
    // debug-mode safety checks don't distort the numbers — we want to
    // know how kvexp performs in production, not in safe mode.
    const bench_kvexp = b.addModule("kvexp_bench", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_kvexp.linkSystemLibrary("lmdb", .{});
    bench_kvexp.link_libc = true;

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("kvexp", bench_kvexp);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run kvexp benchmarks (ReleaseFast)");
    bench_step.dependOn(&run_bench.step);

    // Real-crash recovery tests. Parent process spawns itself as a
    // child via std.process.Child, child does scripted kvexp work,
    // parent SIGKILLs it, parent reopens and verifies. Lives outside
    // `zig build test` because subprocess control + LMDB file
    // reopen are heavier than a unit test wants to be.
    const crash_test_mod = b.createModule(.{
        .root_source_file = b.path("src/crash_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    crash_test_mod.addImport("kvexp", kvexp_mod);

    const crash_test_exe = b.addExecutable(.{
        .name = "crash-test",
        .root_module = crash_test_mod,
    });
    const run_crash_test = b.addRunArtifact(crash_test_exe);
    const crash_test_step = b.step("crash-test", "Real-crash recovery tests (parent SIGKILLs child)");
    crash_test_step.dependOn(&run_crash_test.step);
}
