//! Page allocator interface used by the B-tree's CoW paths.
//!
//! Phase 2 used `file.growBy(1)` directly — pages leaked. Phase 4
//! threads a real allocator through every CoW operation: each
//! superseded page is reported via `.free(page_no, freed_at_seq)`. The
//! `freed_at_seq` is the producing sequence of the *new* page that
//! supersedes it; it tells the durable free-list "this page becomes
//! reclaimable once the active manifest seq has moved past freed_at_seq
//! by 2 commits" (the two-slot atomic-swap invariant: both slots must
//! have moved on before a freed page is safe to reuse).
//!
//! `GrowOnlyAllocator` is the trivial implementation used by phase-2
//! standalone tests and any caller that doesn't care about reclamation
//! — alloc grows the file, free is a no-op (pages leak).

const std = @import("std");
const PagedFileApi = @import("paged_file.zig").PagedFileApi;

pub const PageAllocator = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        alloc: *const fn (ctx: *anyopaque) anyerror!u64,
        free: *const fn (ctx: *anyopaque, page_no: u64, freed_at_seq: u64) anyerror!void,
    };

    pub fn alloc(self: PageAllocator) !u64 {
        return try self.vtable.alloc(self.ctx);
    }

    pub fn free(self: PageAllocator, page_no: u64, freed_at_seq: u64) !void {
        return try self.vtable.free(self.ctx, page_no, freed_at_seq);
    }
};

pub const GrowOnlyAllocator = struct {
    file: PagedFileApi,

    pub fn pageAllocator(self: *GrowOnlyAllocator) PageAllocator {
        return .{ .ctx = self, .vtable = &vtable };
    }

    fn allocImpl(ctx: *anyopaque) anyerror!u64 {
        const self: *GrowOnlyAllocator = @ptrCast(@alignCast(ctx));
        return try self.file.growBy(1);
    }

    fn freeImpl(ctx: *anyopaque, page_no: u64, freed_at_seq: u64) anyerror!void {
        _ = ctx;
        _ = page_no;
        _ = freed_at_seq;
    }

    const vtable: PageAllocator.VTable = .{ .alloc = allocImpl, .free = freeImpl };
};
