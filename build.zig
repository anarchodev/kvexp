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
}
