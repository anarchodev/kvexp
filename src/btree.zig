//! CoW B-tree over the page cache.
//!
//! A mutation never modifies a page in place. Insert/delete reads the
//! affected leaf, allocates a fresh page, writes a densely-packed copy
//! with the modification applied, and propagates that new page number
//! up to the root through CoW'd internal nodes. If a leaf or internal
//! overflows, it splits into two pages and a separator key is promoted
//! to the parent (cascading upward if the parent also overflows).
//!
//! The tree owns its root pointer in `self.root`. Once the manifest
//! layer arrives in phase 3, the root pointer will live there and this
//! field becomes a per-operation in/out value.
//!
//! Page allocation is delegated to a `PageAllocator` (see
//! `page_allocator.zig`). CoW operations call `alloc()` for each fresh
//! page and `free(old_no, self.seq)` for each superseded page. Phase 4
//! plugs in the manifest's durable free-list as the allocator; before
//! that, callers can use `GrowOnlyAllocator` (no reclamation; pages
//! leak — fine for unit tests).

const std = @import("std");
const page = @import("page.zig");
const paged_file_mod = @import("paged_file.zig");
const PagedFile = paged_file_mod.PagedFile;
const PagedFileApi = paged_file_mod.PagedFileApi;
const PageCache = @import("page_cache.zig").PageCache;
const page_allocator_mod = @import("page_allocator.zig");

pub const PageAllocator = page_allocator_mod.PageAllocator;
pub const GrowOnlyAllocator = page_allocator_mod.GrowOnlyAllocator;

const Leaf = page.Leaf;
const Internal = page.Internal;
const PAGE_SIZE = page.PAGE_SIZE;
const HEADER_SIZE = page.HEADER_SIZE;
const SLOT_SIZE = page.SLOT_SIZE;

/// Tree operations call into a PageAllocator that may do arbitrary
/// further B-tree operations (e.g. inserting into the durable
/// free-list), so the concrete error set is open. Use `anyerror` and
/// let callers `try` through; specific errors callers care about
/// (KeyTooLong, ValueTooLong, TreeTooDeep, StoreNotFound, …) are still
/// returned, just not enumerated in a named set.
pub const Error = anyerror;

pub const SpecificError = error{
    KeyTooLong,
    ValueTooLong,
    TreeTooDeep,
    Corruption,
};

pub const MAX_DEPTH: usize = 32;

/// Result of a CoW operation: either a single new page replacing the
/// input, or a split into two pages with a separator key.
const Split = struct {
    left: u64,
    right: u64,
    sep_buf: [page.MAX_KEY_LEN]u8,
    sep_len: u16,

    fn sep(self: *const Split) []const u8 {
        return self.sep_buf[0..self.sep_len];
    }

    fn setSep(self: *Split, key: []const u8) void {
        std.debug.assert(key.len <= page.MAX_KEY_LEN);
        @memcpy(self.sep_buf[0..key.len], key);
        self.sep_len = @intCast(key.len);
    }
};

const CoWResult = union(enum) {
    single: u64,
    split: Split,
};

const SearchHit = page.SearchHit;

/// Path entry recorded during descent. `child_slot == n_entries` means
/// "we descended into rightmost_child."
const PathEntry = struct {
    page_no: u64,
    child_slot: u32,
};

pub const Tree = struct {
    cache: *PageCache,
    file: PagedFileApi,
    allocator: std.mem.Allocator,
    page_allocator: PageAllocator,
    root: u64,
    seq: u64,

    /// Page 0 is reserved for the file header (phase 3+ writes it; phase
    /// 2 just keeps the slot off-limits). The tree treats `root == 0`
    /// as the empty-tree sentinel, so we must guarantee no leaf is ever
    /// allocated to page 0.
    pub fn init(
        allocator: std.mem.Allocator,
        cache: *PageCache,
        file: PagedFileApi,
        page_alloc: PageAllocator,
    ) !Tree {
        if (file.pageCount() == 0) {
            _ = try file.growBy(1);
        }
        return .{
            .cache = cache,
            .file = file,
            .allocator = allocator,
            .page_allocator = page_alloc,
            .root = 0,
            .seq = 1,
        };
    }

    pub fn setSeq(self: *Tree, seq: u64) void {
        self.seq = seq;
    }

    // -------------------------------------------------------------------------
    // Get
    // -------------------------------------------------------------------------

    /// Look up a key. Returns null if not found; otherwise returns a
    /// freshly-allocated slice owned by the caller.
    pub fn get(self: *Tree, allocator: std.mem.Allocator, key: []const u8) Error!?[]u8 {
        return treeGet(self.cache, self.root, allocator, key);
    }

    // -------------------------------------------------------------------------
    // Put
    // -------------------------------------------------------------------------

    pub fn put(self: *Tree, key: []const u8, value: []const u8) Error!void {
        if (key.len > page.MAX_KEY_LEN) return error.KeyTooLong;
        if (value.len > page.MAX_VAL_LEN) return error.ValueTooLong;

        if (self.root == 0) {
            const leaf_no = try self.page_allocator.alloc();
            const ref = try self.cache.pinNewUninit(leaf_no);
            defer ref.release();
            const leaf = Leaf.init(ref.buf());
            _ = leaf.appendCell(key, value);
            ref.markDirty(self.seq);
            self.root = leaf_no;
            return;
        }

        var path: [MAX_DEPTH]PathEntry = undefined;
        var depth: usize = 0;
        var current = self.root;
        while (true) {
            const ref = try self.cache.pin(current);
            const kind = page.pageKind(ref.buf());
            if (kind == .leaf) {
                ref.release();
                break;
            }
            const node = Internal.view(ref.buf());
            const slot = node.childSlotForKey(key);
            const child: u64 = if (slot == node.nEntries()) node.rightmostChild() else node.childAt(slot);
            if (depth >= MAX_DEPTH) {
                ref.release();
                return error.TreeTooDeep;
            }
            path[depth] = .{ .page_no = current, .child_slot = @intCast(slot) };
            depth += 1;
            ref.release();
            current = child;
        }

        // CoW the leaf with the insert/update.
        var result = try self.cowLeafInsert(current, key, value);

        // Propagate up.
        var i: usize = depth;
        while (i > 0) {
            i -= 1;
            result = try self.cowInternalReplace(path[i].page_no, path[i].child_slot, &result);
        }

        // Update root.
        switch (result) {
            .single => |p| self.root = p,
            .split => |s| {
                const new_root_no = try self.page_allocator.alloc();
                const ref = try self.cache.pinNewUninit(new_root_no);
                defer ref.release();
                const node = Internal.init(ref.buf(), s.right);
                _ = node.appendCell(s.sep(), s.left);
                ref.markDirty(self.seq);
                self.root = new_root_no;
            },
        }
    }

    fn cowLeafInsert(self: *Tree, src_page: u64, key: []const u8, value: []const u8) Error!CoWResult {
        const src_ref = try self.cache.pin(src_page);
        defer src_ref.release();
        const src = Leaf.view(src_ref.buf());
        const hit = src.search(key);

        // Compute new total size to decide single-page vs split.
        const added_cell = page.leafCellSize(key.len, value.len);
        const added_slots: usize = if (hit.found) 0 else 1;
        var removed_cell: usize = 0;
        if (hit.found) {
            removed_cell = page.leafCellSize(src.keyAt(hit.idx).len, src.valueAt(hit.idx).len);
        }
        const new_total = HEADER_SIZE +
            (@as(usize, src.nEntries()) + added_slots) * SLOT_SIZE +
            @as(usize, src.header().total_cell_bytes) + added_cell - removed_cell;

        if (new_total > PAGE_SIZE) {
            const result = try self.splitLeafInsert(src, key, value, hit, new_total);
            try self.page_allocator.free(src_page, self.seq);
            return result;
        }

        // Hybrid: if this leaf already belongs to the current txn
        // (it was previously CoW'd at this same seq), the original
        // durable version lives elsewhere and the new-location buffer
        // is our scratch — mutate in place instead of re-CoW'ing.
        // Repacking the leaf reads cells via slices into the same
        // buffer we're about to overwrite, so stage into a stack
        // buffer first, then memcpy back.
        if (src_ref.dirtySeq() == self.seq) {
            var staging: [PAGE_SIZE]u8 = undefined;
            const dst = Leaf.init(&staging);
            var i: usize = 0;
            while (i < src.nEntries()) : (i += 1) {
                if (i == hit.idx) {
                    std.debug.assert(dst.appendCell(key, value));
                    if (hit.found) continue;
                }
                std.debug.assert(dst.appendCell(src.keyAt(i), src.valueAt(i)));
            }
            if (!hit.found and hit.idx == src.nEntries()) {
                std.debug.assert(dst.appendCell(key, value));
            }
            @memcpy(src_ref.buf()[0..PAGE_SIZE], &staging);
            src_ref.markDirty(self.seq);
            return .{ .single = src_page };
        }

        const new_no = try self.page_allocator.alloc();
        const new_ref = try self.cache.pinNewUninit(new_no);
        defer new_ref.release();
        const dst = Leaf.init(new_ref.buf());

        var i: usize = 0;
        while (i < src.nEntries()) : (i += 1) {
            if (i == hit.idx) {
                std.debug.assert(dst.appendCell(key, value));
                if (hit.found) continue;
            }
            std.debug.assert(dst.appendCell(src.keyAt(i), src.valueAt(i)));
        }
        if (!hit.found and hit.idx == src.nEntries()) {
            std.debug.assert(dst.appendCell(key, value));
        }
        new_ref.markDirty(self.seq);
        try self.page_allocator.free(src_page, self.seq);
        return .{ .single = new_no };
    }

    fn splitLeafInsert(
        self: *Tree,
        src: Leaf,
        key: []const u8,
        value: []const u8,
        hit: SearchHit,
        new_total: usize,
    ) Error!CoWResult {
        _ = new_total;

        // Materialize the logical cell sequence into a temporary list,
        // then find a midpoint where both halves fit in a page.
        // Variable-size cells make the simple "threshold" heuristic
        // wrong — a small/small/big/big sequence will route both bigs
        // to the right half, overflowing it. So we compute prefix sums
        // and pick the mid that minimizes imbalance subject to the
        // hard page-fit constraint.
        const Entry = struct { key: []const u8, value: []const u8 };
        const cap = src.nEntries() + 1;
        var entries = try std.ArrayListUnmanaged(Entry).initCapacity(self.allocator, cap);
        defer entries.deinit(self.allocator);

        var i: usize = 0;
        while (i < src.nEntries()) : (i += 1) {
            if (i == hit.idx) {
                entries.appendAssumeCapacity(.{ .key = key, .value = value });
                if (hit.found) continue;
            }
            entries.appendAssumeCapacity(.{ .key = src.keyAt(i), .value = src.valueAt(i) });
        }
        if (!hit.found and hit.idx == src.nEntries()) {
            entries.appendAssumeCapacity(.{ .key = key, .value = value });
        }
        std.debug.assert(entries.items.len >= 2);

        // Cumulative cell-byte sum (each element = size including slot).
        var total_bytes: usize = 0;
        var cumulative = try std.ArrayListUnmanaged(usize).initCapacity(self.allocator, entries.items.len + 1);
        defer cumulative.deinit(self.allocator);
        cumulative.appendAssumeCapacity(0);
        for (entries.items) |e| {
            total_bytes += page.leafCellSize(e.key.len, e.value.len) + SLOT_SIZE;
            cumulative.appendAssumeCapacity(total_bytes);
        }

        // Find best valid mid: cells [0..mid-1] go left, [mid..n-1] go right.
        var best_mid: usize = 0;
        var best_imbalance: usize = std.math.maxInt(usize);
        var mid: usize = 1;
        while (mid < entries.items.len) : (mid += 1) {
            const left_size = HEADER_SIZE + cumulative.items[mid];
            const right_size = HEADER_SIZE + (total_bytes - cumulative.items[mid]);
            if (left_size > PAGE_SIZE) break; // left full; later mids also overflow
            if (right_size > PAGE_SIZE) continue; // right doesn't fit; try later mid
            const imbalance = if (left_size > right_size) left_size - right_size else right_size - left_size;
            if (imbalance < best_imbalance) {
                best_mid = mid;
                best_imbalance = imbalance;
            }
        }
        std.debug.assert(best_mid >= 1);
        std.debug.assert(best_mid < entries.items.len);

        const left_no = try self.page_allocator.alloc();
        const right_no = try self.page_allocator.alloc();
        const left_ref = try self.cache.pinNewUninit(left_no);
        defer left_ref.release();
        const right_ref = try self.cache.pinNewUninit(right_no);
        defer right_ref.release();
        const left = Leaf.init(left_ref.buf());
        const right = Leaf.init(right_ref.buf());

        var j: usize = 0;
        while (j < best_mid) : (j += 1) {
            std.debug.assert(left.appendCell(entries.items[j].key, entries.items[j].value));
        }
        while (j < entries.items.len) : (j += 1) {
            std.debug.assert(right.appendCell(entries.items[j].key, entries.items[j].value));
        }

        left_ref.markDirty(self.seq);
        right_ref.markDirty(self.seq);

        var split = Split{
            .left = left_no,
            .right = right_no,
            .sep_buf = undefined,
            .sep_len = 0,
        };
        split.setSep(entries.items[best_mid].key);
        return .{ .split = split };
    }

    fn cowInternalReplace(
        self: *Tree,
        src_page: u64,
        child_slot: u32,
        child_result: *const CoWResult,
    ) Error!CoWResult {
        const src_ref = try self.cache.pin(src_page);
        defer src_ref.release();
        const src = Internal.view(src_ref.buf());

        switch (child_result.*) {
            .single => |new_child_no| {
                const old_child_no: u64 = if (child_slot == src.nEntries())
                    src.rightmostChild()
                else
                    src.childAt(child_slot);

                // Hybrid no-op shortcut: if the child mutated in place
                // (same page_no), the parent's pointer is already
                // correct — no need to touch this page or anything
                // above it. This is the dominant case once a workload's
                // hot path is warm (all leaves and internals on the
                // path are shadow).
                if (old_child_no == new_child_no) {
                    return .{ .single = src_page };
                }

                // Hybrid in-place: if this page already belongs to the
                // current txn, patch the one child pointer rather than
                // rebuilding the whole page.
                if (src_ref.dirtySeq() == self.seq) {
                    if (child_slot == src.nEntries()) {
                        src.setRightmostChild(new_child_no);
                    } else {
                        src.setChildAt(child_slot, new_child_no);
                    }
                    src_ref.markDirty(self.seq);
                    return .{ .single = src_page };
                }

                // Full CoW: rewrite densely with one child pointer changed.
                const new_no = try self.page_allocator.alloc();
                const new_ref = try self.cache.pinNewUninit(new_no);
                defer new_ref.release();
                const rightmost: u64 = blk: {
                    if (child_slot == src.nEntries()) break :blk new_child_no;
                    break :blk src.rightmostChild();
                };
                const dst = Internal.init(new_ref.buf(), rightmost);
                var i: usize = 0;
                while (i < src.nEntries()) : (i += 1) {
                    const k = src.keyAt(i);
                    const c: u64 = if (i == child_slot) new_child_no else src.childAt(i);
                    std.debug.assert(dst.appendCell(k, c));
                }
                new_ref.markDirty(self.seq);
                try self.page_allocator.free(src_page, self.seq);
                return .{ .single = new_no };
            },
            .split => |s| {
                const new_total = HEADER_SIZE +
                    (@as(usize, src.nEntries()) + 1) * SLOT_SIZE +
                    @as(usize, src.header().total_cell_bytes) +
                    page.internalCellSize(s.sep_len);
                const result = if (new_total <= PAGE_SIZE)
                    try self.cowInternalInsertSingle(src, child_slot, &s)
                else
                    try self.splitInternalInsert(src, child_slot, &s, new_total);
                try self.page_allocator.free(src_page, self.seq);
                return result;
            },
        }
    }

    fn cowInternalInsertSingle(
        self: *Tree,
        src: Internal,
        child_slot: u32,
        s: *const Split,
    ) Error!CoWResult {
        const new_no = try self.page_allocator.alloc();
        const new_ref = try self.cache.pinNewUninit(new_no);
        defer new_ref.release();

        // Determine new rightmost.
        const new_rightmost: u64 = blk: {
            if (child_slot == src.nEntries()) break :blk s.right;
            break :blk src.rightmostChild();
        };
        const dst = Internal.init(new_ref.buf(), new_rightmost);

        var i: usize = 0;
        while (i < src.nEntries()) : (i += 1) {
            if (i == child_slot) {
                // Insert (sep, left), then existing cell with child replaced by right.
                std.debug.assert(dst.appendCell(s.sep(), s.left));
                std.debug.assert(dst.appendCell(src.keyAt(i), s.right));
            } else {
                std.debug.assert(dst.appendCell(src.keyAt(i), src.childAt(i)));
            }
        }
        if (child_slot == src.nEntries()) {
            // Split happened at rightmost.
            std.debug.assert(dst.appendCell(s.sep(), s.left));
        }
        new_ref.markDirty(self.seq);
        return .{ .single = new_no };
    }

    fn splitInternalInsert(
        self: *Tree,
        src: Internal,
        child_slot: u32,
        child_split: *const Split,
        new_total: usize,
    ) Error!CoWResult {
        // Materialize the logical cell sequence (with the new cell inserted)
        // into a temporary list, then find the midpoint and copy into two new
        // pages. Promoted separator is the cell at the midpoint (removed from
        // both halves and bubbled up).
        const Entry = struct { key: []const u8, child: u64 };
        var entries = try std.ArrayListUnmanaged(Entry).initCapacity(self.allocator, src.nEntries() + 1);
        defer entries.deinit(self.allocator);

        var i: usize = 0;
        while (i < src.nEntries()) : (i += 1) {
            if (i == child_slot) {
                entries.appendAssumeCapacity(.{ .key = child_split.sep(), .child = child_split.left });
                entries.appendAssumeCapacity(.{ .key = src.keyAt(i), .child = child_split.right });
            } else {
                entries.appendAssumeCapacity(.{ .key = src.keyAt(i), .child = src.childAt(i) });
            }
        }
        const tail_rightmost: u64 = blk: {
            if (child_slot == src.nEntries()) {
                entries.appendAssumeCapacity(.{ .key = child_split.sep(), .child = child_split.left });
                break :blk child_split.right;
            }
            break :blk src.rightmostChild();
        };

        // Find midpoint by cumulative byte count, but require ≥ 1 cell on
        // each side AND keep the promoted cell separate.
        const half = new_total / 2;
        var used: usize = HEADER_SIZE;
        var mid: usize = 0;
        while (mid < entries.items.len) : (mid += 1) {
            const e = entries.items[mid];
            const sz = page.internalCellSize(e.key.len) + SLOT_SIZE;
            if (used + sz > half and mid >= 1) break;
            used += sz;
        }
        if (mid >= entries.items.len) mid = entries.items.len - 1;
        if (mid == 0) mid = 1;
        // mid is the index of the promoted entry; cells [0..mid-1] go left,
        // cells [mid+1..N-1] go right, cells[mid].child becomes left's rightmost.

        const left_no = try self.page_allocator.alloc();
        const right_no = try self.page_allocator.alloc();
        const left_ref = try self.cache.pinNewUninit(left_no);
        defer left_ref.release();
        const right_ref = try self.cache.pinNewUninit(right_no);
        defer right_ref.release();

        const left = Internal.init(left_ref.buf(), entries.items[mid].child);
        var j: usize = 0;
        while (j < mid) : (j += 1) {
            std.debug.assert(left.appendCell(entries.items[j].key, entries.items[j].child));
        }

        const right = Internal.init(right_ref.buf(), tail_rightmost);
        j = mid + 1;
        while (j < entries.items.len) : (j += 1) {
            std.debug.assert(right.appendCell(entries.items[j].key, entries.items[j].child));
        }

        left_ref.markDirty(self.seq);
        right_ref.markDirty(self.seq);

        var result = Split{
            .left = left_no,
            .right = right_no,
            .sep_buf = undefined,
            .sep_len = 0,
        };
        result.setSep(entries.items[mid].key);
        return .{ .split = result };
    }

    // -------------------------------------------------------------------------
    // Delete
    // -------------------------------------------------------------------------

    /// Delete a key. Returns true if the key existed. Phase 2: no leaf
    /// merge / rebalance — underfull pages are tolerated. Phase 4+ adds it.
    pub fn delete(self: *Tree, key: []const u8) Error!bool {
        if (key.len > page.MAX_KEY_LEN) return error.KeyTooLong;
        if (self.root == 0) return false;

        var path: [MAX_DEPTH]PathEntry = undefined;
        var depth: usize = 0;
        var current = self.root;
        while (true) {
            const ref = try self.cache.pin(current);
            const kind = page.pageKind(ref.buf());
            if (kind == .leaf) {
                ref.release();
                break;
            }
            const node = Internal.view(ref.buf());
            const slot = node.childSlotForKey(key);
            const child: u64 = if (slot == node.nEntries()) node.rightmostChild() else node.childAt(slot);
            if (depth >= MAX_DEPTH) {
                ref.release();
                return error.TreeTooDeep;
            }
            path[depth] = .{ .page_no = current, .child_slot = @intCast(slot) };
            depth += 1;
            ref.release();
            current = child;
        }

        const cow = try self.cowLeafDelete(current, key);
        if (!cow.existed) return false;

        // Propagate up the new leaf pointer. cowLeafDelete returns single
        // (never split on delete).
        var result = CoWResult{ .single = cow.new_page };
        var i: usize = depth;
        while (i > 0) {
            i -= 1;
            result = try self.cowInternalReplace(path[i].page_no, path[i].child_slot, &result);
        }
        switch (result) {
            .single => |p| self.root = p,
            .split => unreachable,
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Prefix scan
    // -------------------------------------------------------------------------

    /// Iterate keys starting with `prefix` in ascending order. The cursor
    /// borrows `prefix` — caller must keep it alive for the cursor's
    /// lifetime. Mutations to the tree invalidate any outstanding
    /// cursor (phase 2 is single-threaded; document and don't do that).
    pub fn scanPrefix(self: *Tree, prefix: []const u8) Error!PrefixCursor {
        return PrefixCursor.open(self.cache, self.root, prefix);
    }

    fn cowLeafDelete(self: *Tree, src_page: u64, key: []const u8) Error!struct { existed: bool, new_page: u64 } {
        const src_ref = try self.cache.pin(src_page);
        defer src_ref.release();
        const src = Leaf.view(src_ref.buf());
        const hit = src.search(key);
        if (!hit.found) return .{ .existed = false, .new_page = src_page };

        // Hybrid in-place: same logic as cowLeafInsert.
        if (src_ref.dirtySeq() == self.seq) {
            var staging: [PAGE_SIZE]u8 = undefined;
            const dst = Leaf.init(&staging);
            var i: usize = 0;
            while (i < src.nEntries()) : (i += 1) {
                if (i == hit.idx) continue;
                std.debug.assert(dst.appendCell(src.keyAt(i), src.valueAt(i)));
            }
            @memcpy(src_ref.buf()[0..PAGE_SIZE], &staging);
            src_ref.markDirty(self.seq);
            return .{ .existed = true, .new_page = src_page };
        }

        const new_no = try self.page_allocator.alloc();
        const new_ref = try self.cache.pinNewUninit(new_no);
        defer new_ref.release();
        const dst = Leaf.init(new_ref.buf());
        var i: usize = 0;
        while (i < src.nEntries()) : (i += 1) {
            if (i == hit.idx) continue;
            std.debug.assert(dst.appendCell(src.keyAt(i), src.valueAt(i)));
        }
        new_ref.markDirty(self.seq);
        try self.page_allocator.free(src_page, self.seq);
        return .{ .existed = true, .new_page = new_no };
    }
};

/// Walk every page reachable from `root` (including `root` itself)
/// and append its page_no into `out`. Recursive descent; bounded by
/// MAX_DEPTH. Used by `Manifest.verify`.
pub fn collectTreePages(
    cache: *PageCache,
    root: u64,
    out: *std.AutoHashMapUnmanaged(u64, void),
    allocator: std.mem.Allocator,
) Error!void {
    if (root == 0) return;
    try out.put(allocator, root, {});
    const ref = try cache.pin(root);
    defer ref.release();
    switch (page.pageKind(ref.buf())) {
        .leaf => {},
        .internal => {
            const node = Internal.view(ref.buf());
            var i: usize = 0;
            while (i < node.nEntries()) : (i += 1) {
                try collectTreePages(cache, node.childAt(i), out, allocator);
            }
            try collectTreePages(cache, node.rightmostChild(), out, allocator);
        },
    }
}

/// Look up `key` in the B-tree rooted at `root`. Reads are pure cache
/// pins, no writer state is touched, so this can be called with a
/// snapshot's captured root without holding any tree-level lock.
pub fn treeGet(
    cache: *PageCache,
    root: u64,
    allocator: std.mem.Allocator,
    key: []const u8,
) Error!?[]u8 {
    if (key.len > page.MAX_KEY_LEN) return error.KeyTooLong;
    if (root == 0) return null;
    var current = root;
    while (true) {
        const ref = try cache.pin(current);
        defer ref.release();
        switch (page.pageKind(ref.buf())) {
            .leaf => {
                const leaf = Leaf.view(ref.buf());
                const hit = leaf.search(key);
                if (!hit.found) return null;
                const val = leaf.valueAt(hit.idx);
                const owned = try allocator.alloc(u8, val.len);
                @memcpy(owned, val);
                return owned;
            },
            .internal => {
                const node = Internal.view(ref.buf());
                current = node.childForKey(key);
            },
        }
    }
}

pub const PrefixCursor = struct {
    cache: *PageCache,
    root: u64,
    prefix: []const u8,

    leaf_pin: ?@import("page_cache.zig").PageRef,
    leaf_slot: u32,

    path: [MAX_DEPTH]PathEntry,
    depth: u32,

    exhausted: bool,
    before_first: bool,

    /// Open a cursor against the B-tree rooted at `root`. The cursor
    /// does not depend on a Tree; a snapshot can pass its captured
    /// root directly.
    pub fn open(cache: *PageCache, root: u64, prefix: []const u8) Error!PrefixCursor {
        var cursor: PrefixCursor = .{
            .cache = cache,
            .root = root,
            .prefix = prefix,
            .leaf_pin = null,
            .leaf_slot = 0,
            .path = undefined,
            .depth = 0,
            .exhausted = false,
            .before_first = true,
        };
        try cursor.seekTo(prefix);
        return cursor;
    }

    /// Like `open`, but skip directly past `after` on the initial
    /// descent — yields the first key in [max(prefix, after++0x00),
    /// prefix_upper_bound). Used for paging: pass the last key from
    /// the previous page as `after` to start the next page in
    /// O(log N) instead of O(skipped).
    pub fn openAfter(
        cache: *PageCache,
        root: u64,
        prefix: []const u8,
        after: []const u8,
    ) Error!PrefixCursor {
        var cursor: PrefixCursor = .{
            .cache = cache,
            .root = root,
            .prefix = prefix,
            .leaf_pin = null,
            .leaf_slot = 0,
            .path = undefined,
            .depth = 0,
            .exhausted = false,
            .before_first = true,
        };

        // Compute the actual descent target. `after` is exclusive —
        // we want first key > after — and the cursor is also bounded
        // below by prefix. So descend toward
        // `max(prefix, after++0x00)`.
        if (after.len == 0) {
            try cursor.seekTo(prefix);
        } else {
            if (after.len + 1 > page.MAX_KEY_LEN) return error.KeyTooLong;
            var after_succ_buf: [page.MAX_KEY_LEN + 1]u8 = undefined;
            @memcpy(after_succ_buf[0..after.len], after);
            after_succ_buf[after.len] = 0;
            const after_succ = after_succ_buf[0 .. after.len + 1];
            const target = if (std.mem.order(u8, after_succ, prefix) == .gt) after_succ else prefix;
            try cursor.seekTo(target);
        }
        return cursor;
    }

    pub fn deinit(self: *PrefixCursor) void {
        if (self.leaf_pin) |ref| ref.release();
        self.leaf_pin = null;
    }

    pub fn key(self: *const PrefixCursor) []const u8 {
        const ref = self.leaf_pin.?;
        return Leaf.view(ref.buf()).keyAt(self.leaf_slot);
    }

    pub fn value(self: *const PrefixCursor) []const u8 {
        const ref = self.leaf_pin.?;
        return Leaf.view(ref.buf()).valueAt(self.leaf_slot);
    }

    pub fn next(self: *PrefixCursor) Error!bool {
        if (self.exhausted) return false;
        if (!self.before_first) {
            self.leaf_slot += 1;
        }
        self.before_first = false;

        while (true) {
            if (self.leaf_pin == null) {
                self.exhausted = true;
                return false;
            }
            const leaf = Leaf.view(self.leaf_pin.?.buf());
            if (self.leaf_slot < leaf.nEntries()) {
                const k = leaf.keyAt(self.leaf_slot);
                if (std.mem.startsWith(u8, k, self.prefix)) return true;
                self.exhausted = true;
                return false;
            }
            try self.advanceToNextLeaf();
            if (self.exhausted) return false;
        }
    }

    /// Descend to the first key >= `target`, recording the descent
    /// path for later `advanceToNextLeaf` calls. `target` must
    /// outlive this function call (just used during descent).
    fn seekTo(self: *PrefixCursor, target: []const u8) Error!void {
        if (self.root == 0) {
            self.exhausted = true;
            return;
        }
        var current = self.root;
        while (true) {
            const ref = try self.cache.pin(current);
            switch (page.pageKind(ref.buf())) {
                .leaf => {
                    const leaf = Leaf.view(ref.buf());
                    const hit = leaf.search(target);
                    self.leaf_pin = ref;
                    self.leaf_slot = @intCast(hit.idx);
                    return;
                },
                .internal => {
                    const node = Internal.view(ref.buf());
                    const slot = node.childSlotForKey(target);
                    const child: u64 = if (slot == node.nEntries()) node.rightmostChild() else node.childAt(slot);
                    if (self.depth >= MAX_DEPTH) {
                        ref.release();
                        return error.TreeTooDeep;
                    }
                    self.path[self.depth] = .{ .page_no = current, .child_slot = @intCast(slot) };
                    self.depth += 1;
                    ref.release();
                    current = child;
                },
            }
        }
    }

    fn advanceToNextLeaf(self: *PrefixCursor) Error!void {
        if (self.leaf_pin) |ref| ref.release();
        self.leaf_pin = null;

        while (self.depth > 0) {
            const entry = &self.path[self.depth - 1];
            const ref = try self.cache.pin(entry.page_no);
            const node = Internal.view(ref.buf());
            if (entry.child_slot < node.nEntries()) {
                entry.child_slot += 1;
                const child: u64 = if (entry.child_slot == node.nEntries())
                    node.rightmostChild()
                else
                    node.childAt(entry.child_slot);
                ref.release();
                try self.descendLeftmost(child);
                return;
            }
            ref.release();
            self.depth -= 1;
        }
        self.exhausted = true;
    }

    fn descendLeftmost(self: *PrefixCursor, start_page: u64) Error!void {
        var current = start_page;
        while (true) {
            const ref = try self.cache.pin(current);
            switch (page.pageKind(ref.buf())) {
                .leaf => {
                    self.leaf_pin = ref;
                    self.leaf_slot = 0;
                    return;
                },
                .internal => {
                    const node = Internal.view(ref.buf());
                    const child: u64 = if (node.nEntries() == 0) node.rightmostChild() else node.childAt(0);
                    if (self.depth >= MAX_DEPTH) {
                        ref.release();
                        return error.TreeTooDeep;
                    }
                    self.path[self.depth] = .{ .page_no = current, .child_slot = 0 };
                    self.depth += 1;
                    ref.release();
                    current = child;
                },
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;
const BufferPool = @import("buffer_pool.zig").BufferPool;

const Harness = struct {
    tmp: std.testing.TmpDir,
    file: *PagedFile,
    pool: *BufferPool,
    cache: *PageCache,
    grow: *GrowOnlyAllocator,
    tree: Tree,

    fn init(pool_capacity: u32) !Harness {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();

        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &path_buf);
        var full_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&full_buf, "{s}/btree.test", .{dir_path});

        const file = try testing.allocator.create(PagedFile);
        errdefer testing.allocator.destroy(file);
        file.* = try PagedFile.open(path, .{ .create = true, .truncate = true });
        errdefer file.close();

        const pool = try testing.allocator.create(BufferPool);
        errdefer testing.allocator.destroy(pool);
        pool.* = try BufferPool.init(testing.allocator, page.PAGE_SIZE, pool_capacity);
        errdefer pool.deinit(testing.allocator);

        const cache = try testing.allocator.create(PageCache);
        errdefer testing.allocator.destroy(cache);
        cache.* = try PageCache.init(testing.allocator, file.api(), pool, .{});

        const grow = try testing.allocator.create(GrowOnlyAllocator);
        errdefer testing.allocator.destroy(grow);
        grow.* = .{ .file = file.api() };

        const tree = try Tree.init(testing.allocator, cache, file.api(), grow.pageAllocator());
        return .{
            .tmp = tmp,
            .file = file,
            .pool = pool,
            .cache = cache,
            .grow = grow,
            .tree = tree,
        };
    }

    fn deinit(self: *Harness) void {
        self.cache.deinit();
        testing.allocator.destroy(self.cache);
        self.pool.deinit(testing.allocator);
        testing.allocator.destroy(self.pool);
        testing.allocator.destroy(self.grow);
        self.file.close();
        testing.allocator.destroy(self.file);
        self.tmp.cleanup();
    }
};

test "Tree: empty get returns null" {
    var h = try Harness.init(32);
    defer h.deinit();
    try testing.expect((try h.tree.get(testing.allocator, "anything")) == null);
}

test "Tree: single put + get" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.tree.put("hello", "world");
    const got = (try h.tree.get(testing.allocator, "hello")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("world", got);
}

test "Tree: update existing key" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.tree.put("k", "v1");
    try h.tree.put("k", "v2");
    const got = (try h.tree.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v2", got);
}

test "Tree: many inserts trigger leaf splits, get works" {
    var h = try Harness.init(128);
    defer h.deinit();
    var oracle: std.StringHashMap([]u8) = .init(testing.allocator);
    defer {
        var it = oracle.iterator();
        while (it.next()) |e| {
            testing.allocator.free(e.key_ptr.*);
            testing.allocator.free(e.value_ptr.*);
        }
        oracle.deinit();
    }

    var rng = std.Random.DefaultPrng.init(0xC0FFEE);
    var i: u32 = 0;
    while (i < 500) : (i += 1) {
        var key_buf: [16]u8 = undefined;
        const key_slice = try std.fmt.bufPrint(&key_buf, "k{d:0>10}", .{rng.random().int(u32) % 10000});
        const val_len = (rng.random().int(u32) % 64) + 1;
        var val_buf: [128]u8 = undefined;
        for (val_buf[0..val_len]) |*b| b.* = rng.random().int(u8);

        try h.tree.put(key_slice, val_buf[0..val_len]);

        const owned_key = try testing.allocator.dupe(u8, key_slice);
        const owned_val = try testing.allocator.dupe(u8, val_buf[0..val_len]);
        const gop = try oracle.getOrPut(owned_key);
        if (gop.found_existing) {
            testing.allocator.free(owned_key);
            testing.allocator.free(gop.value_ptr.*);
        }
        gop.value_ptr.* = owned_val;
    }

    var it = oracle.iterator();
    while (it.next()) |e| {
        const got = (try h.tree.get(testing.allocator, e.key_ptr.*)) orelse {
            std.debug.print("missing key: {s}\n", .{e.key_ptr.*});
            return error.MissingKey;
        };
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(u8, e.value_ptr.*, got);
    }
}

test "Tree: delete returns true once, then false; get returns null" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.tree.put("a", "1");
    try h.tree.put("b", "2");
    try h.tree.put("c", "3");

    try testing.expect(try h.tree.delete("b"));
    try testing.expect((try h.tree.get(testing.allocator, "b")) == null);
    try testing.expect(!(try h.tree.delete("b")));

    const a = (try h.tree.get(testing.allocator, "a")).?;
    defer testing.allocator.free(a);
    const c = (try h.tree.get(testing.allocator, "c")).?;
    defer testing.allocator.free(c);
    try testing.expectEqualStrings("1", a);
    try testing.expectEqualStrings("3", c);
}

test "Tree: prefix scan on small tree" {
    var h = try Harness.init(32);
    defer h.deinit();

    try h.tree.put("aardvark", "1");
    try h.tree.put("apple", "2");
    try h.tree.put("apricot", "3");
    try h.tree.put("banana", "4");
    try h.tree.put("blueberry", "5");

    var cursor = try h.tree.scanPrefix("ap");
    defer cursor.deinit();

    try testing.expect(try cursor.next());
    try testing.expectEqualStrings("apple", cursor.key());
    try testing.expectEqualStrings("2", cursor.value());
    try testing.expect(try cursor.next());
    try testing.expectEqualStrings("apricot", cursor.key());
    try testing.expect(!try cursor.next());
}

test "Tree: prefix scan that crosses leaf boundaries" {
    var h = try Harness.init(256);
    defer h.deinit();

    // Insert enough keys with mixed prefixes that splits put matching
    // keys across multiple leaves.
    const N: u32 = 2000;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var key_buf: [20]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "p{d:0>6}", .{i});
        var val_buf: [16]u8 = undefined;
        const val = try std.fmt.bufPrint(&val_buf, "v{d}", .{i});
        try h.tree.put(key, val);
    }
    // Also insert non-matching prefix keys.
    try h.tree.put("a", "1");
    try h.tree.put("z", "1");

    var cursor = try h.tree.scanPrefix("p");
    defer cursor.deinit();

    var seen: u32 = 0;
    var last_buf: [20]u8 = undefined;
    var last_len: usize = 0;
    while (try cursor.next()) {
        const k = cursor.key();
        try testing.expect(std.mem.startsWith(u8, k, "p"));
        if (seen > 0) {
            try testing.expect(std.mem.order(u8, last_buf[0..last_len], k) == .lt);
        }
        @memcpy(last_buf[0..k.len], k);
        last_len = k.len;
        seen += 1;
    }
    try testing.expectEqual(N, seen);
}

test "PrefixCursor: openAfter seeks past cursor in O(log N)" {
    var h = try Harness.init(256);
    defer h.deinit();

    const N: u32 = 1000;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>8}", .{i});
        try h.tree.put(key, "v");
    }

    // Seek past k00000500 → first emitted key should be k00000501.
    var cursor = try @import("btree.zig").PrefixCursor.openAfter(h.cache, h.tree.root, "k", "k00000500");
    defer cursor.deinit();
    try testing.expect(try cursor.next());
    try testing.expectEqualStrings("k00000501", cursor.key());
    try testing.expect(try cursor.next());
    try testing.expectEqualStrings("k00000502", cursor.key());
}

test "PrefixCursor: openAfter past the last matching key is exhausted" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.tree.put("apple", "1");
    try h.tree.put("apricot", "2");
    try h.tree.put("banana", "3");

    var cursor = try @import("btree.zig").PrefixCursor.openAfter(h.cache, h.tree.root, "ap", "apricot");
    defer cursor.deinit();
    try testing.expect(!try cursor.next());
}

test "Tree: prefix scan with empty prefix iterates all keys" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.tree.put("a", "1");
    try h.tree.put("b", "2");
    try h.tree.put("c", "3");
    try h.tree.put("d", "4");

    var cursor = try h.tree.scanPrefix("");
    defer cursor.deinit();

    const expected = [_][]const u8{ "a", "b", "c", "d" };
    var i: usize = 0;
    while (try cursor.next()) : (i += 1) {
        try testing.expectEqualStrings(expected[i], cursor.key());
    }
    try testing.expectEqual(@as(usize, 4), i);
}

test "Tree: prefix scan empty result" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.tree.put("a", "1");
    try h.tree.put("b", "2");

    var cursor = try h.tree.scanPrefix("z");
    defer cursor.deinit();
    try testing.expect(!try cursor.next());
}

test "Tree: insert + delete mix vs oracle, prefix scan verifies survivors" {
    var h = try Harness.init(256);
    defer h.deinit();

    // ArenaAllocator owns all key copies for the oracle; freed on test exit.
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var oracle: std.StringHashMap(void) = .init(testing.allocator);
    defer oracle.deinit();

    var rng = std.Random.DefaultPrng.init(0xDEADBEEF);
    var op: u32 = 0;
    while (op < 4000) : (op += 1) {
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>10}", .{rng.random().int(u32) % 2000});
        if (rng.random().int(u8) < 200) {
            try h.tree.put(key, "v");
            if (!oracle.contains(key)) {
                const owned = try arena_alloc.dupe(u8, key);
                try oracle.put(owned, {});
            }
        } else {
            const existed_tree = try h.tree.delete(key);
            const existed_oracle = oracle.remove(key);
            try testing.expectEqual(existed_oracle, existed_tree);
        }
    }

    // Prefix scan results vs oracle survivors with prefix "k0000000001".
    var got: std.ArrayListUnmanaged([]const u8) = .empty;
    defer got.deinit(arena_alloc);
    var cursor = try h.tree.scanPrefix("k0000000001");
    defer cursor.deinit();
    while (try cursor.next()) {
        try got.append(arena_alloc, try arena_alloc.dupe(u8, cursor.key()));
    }

    var expected: std.ArrayListUnmanaged([]const u8) = .empty;
    defer expected.deinit(arena_alloc);
    var oit = oracle.keyIterator();
    while (oit.next()) |k| {
        if (std.mem.startsWith(u8, k.*, "k0000000001")) {
            try expected.append(arena_alloc, k.*);
        }
    }
    std.mem.sort([]const u8, expected.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);

    try testing.expectEqual(expected.items.len, got.items.len);
    for (expected.items, got.items) |e, g| try testing.expectEqualStrings(e, g);
}

test "Tree: deep tree from sequential inserts (forces internal splits)" {
    var h = try Harness.init(256);
    defer h.deinit();

    const N: u32 = 5000;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>10}", .{i});
        var val_buf: [64]u8 = undefined;
        const val = try std.fmt.bufPrint(&val_buf, "v{d:0>10}-{d:0>10}", .{ i, i });
        try h.tree.put(key, val);
    }

    i = 0;
    while (i < N) : (i += 1) {
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>10}", .{i});
        var val_buf: [64]u8 = undefined;
        const expected = try std.fmt.bufPrint(&val_buf, "v{d:0>10}-{d:0>10}", .{ i, i });
        const got = (try h.tree.get(testing.allocator, key)) orelse {
            std.debug.print("missing key i={d}\n", .{i});
            return error.MissingKey;
        };
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(expected, got);
    }
}
