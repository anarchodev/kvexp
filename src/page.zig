//! On-disk page formats for kvexp B-tree pages.
//!
//! Both leaf and internal pages use the same shape:
//!
//!   [ 32-byte header | slot array (u16[]) | ... free ... | cells ]
//!
//! Slots grow from the header forward; cells grow from the end of the
//! page backward. Slots are kept sorted by key. Cells are written
//! densely from the top — kvexp never modifies a page in place (CoW),
//! so we always write a fresh page with cells packed at the end. No
//! per-page compaction logic is needed.
//!
//! Leaf cell:      u16 key_len | u16 val_len | key | val
//! Internal cell:  u16 key_len | u64 child   | key
//!
//! ## Per-page CRC32 (torn-write detection)
//!
//! Bytes [CHECKSUM_OFFSET, CHECKSUM_OFFSET + 4) of every B-tree page hold
//! a CRC32 of the rest of the page. The page cache stamps the CRC just
//! before writePage (`finalize`) and verifies just after readPage
//! (`verifyChecksum`). A partial-sector write that leaves the page half
//! old / half new fails verification at next open, so the manifest's
//! slot-swap invariant ("only durabilize promotes the slot, only after
//! all referenced pages are on disk and fsynced") is reinforced against
//! sub-page tears within referenced data pages.
//!
//! The CRC field sits at the same offset (8) in both leaf and internal
//! headers, so `verifyChecksum` is kind-agnostic.

const std = @import("std");

pub const PAGE_SIZE: u32 = 4096;
pub const HEADER_SIZE: u32 = 32;
pub const SLOT_SIZE: u32 = 2;

pub const MAX_KEY_LEN: u32 = 256;
pub const MAX_VAL_LEN: u32 = 2048;

/// Offset of the CRC32 field within a B-tree page header. Identical in
/// LeafHeader and InternalHeader so the verify/finalize helpers can run
/// without inspecting `kind`.
pub const CHECKSUM_OFFSET: usize = 8;
pub const CHECKSUM_SIZE: usize = 4;

pub const Kind = enum(u8) { leaf = 1, internal = 2 };

pub fn pageKind(buf: []const u8) Kind {
    return @enumFromInt(buf[0]);
}

pub const LeafHeader = extern struct {
    kind: u8 align(1) = @intFromEnum(Kind.leaf),
    flags: u8 align(1) = 0,
    n_entries: u16 align(1) = 0,
    total_cell_bytes: u16 align(1) = 0,
    _pad: u16 align(1) = 0,
    checksum: u32 align(1) = 0,
    _reserved: [HEADER_SIZE - 12]u8 align(1) = @splat(0),

    comptime {
        std.debug.assert(@sizeOf(@This()) == HEADER_SIZE);
        std.debug.assert(@offsetOf(@This(), "checksum") == CHECKSUM_OFFSET);
    }
};

pub const InternalHeader = extern struct {
    kind: u8 align(1) = @intFromEnum(Kind.internal),
    flags: u8 align(1) = 0,
    n_entries: u16 align(1) = 0,
    total_cell_bytes: u16 align(1) = 0,
    _pad: u16 align(1) = 0,
    checksum: u32 align(1) = 0,
    rightmost_child: u64 align(1) = 0,
    _reserved: [HEADER_SIZE - 20]u8 align(1) = @splat(0),

    comptime {
        std.debug.assert(@sizeOf(@This()) == HEADER_SIZE);
        std.debug.assert(@offsetOf(@This(), "checksum") == CHECKSUM_OFFSET);
        std.debug.assert(@offsetOf(@This(), "rightmost_child") == 12);
    }
};

/// CRC32 of everything except the 4-byte checksum field. Computed
/// incrementally so it never mutates `buf`.
pub fn computeChecksum(buf: []const u8) u32 {
    std.debug.assert(buf.len == PAGE_SIZE);
    var h = std.hash.Crc32.init();
    h.update(buf[0..CHECKSUM_OFFSET]);
    h.update(buf[CHECKSUM_OFFSET + CHECKSUM_SIZE ..]);
    return h.final();
}

/// Stamp the page's CRC field. Call just before writing the page to
/// disk; safe to call repeatedly (idempotent on stable content).
pub fn finalize(buf: []u8) void {
    const crc = computeChecksum(buf);
    std.mem.writeInt(u32, buf[CHECKSUM_OFFSET..][0..4], crc, .little);
}

/// True iff the CRC field matches the page contents. Pure function —
/// does not mutate `buf`.
pub fn verifyChecksum(buf: []const u8) bool {
    const stored = std.mem.readInt(u32, buf[CHECKSUM_OFFSET..][0..4], .little);
    return stored == computeChecksum(buf);
}

/// Structural validation of a B-tree page. Defense in depth: even if a
/// page passes CRC32 (because a bit-flip happened in memory and the
/// next writeback stamped a matching CRC), the parser must not be
/// indexed off corrupt offsets. `verify` returns false on any of:
///
///   * Unknown `kind` byte.
///   * `n_entries` so large that the slot array runs past PAGE_SIZE.
///   * Any slot pointing inside the slot array (`offset < slot_area_end`)
///     or past PAGE_SIZE.
///   * Any cell extending past PAGE_SIZE.
///   * Any key_len > MAX_KEY_LEN, val_len > MAX_VAL_LEN.
///   * `total_cell_bytes` not matching the sum of decoded cell sizes.
///
/// Pages that fail this check should be treated as corrupt; the cache
/// surfaces `error.PageStructureInvalid` on a failed verify.
pub fn verifyStructure(buf: []const u8) bool {
    std.debug.assert(buf.len == PAGE_SIZE);
    if (buf[0] == @intFromEnum(Kind.leaf)) return verifyLeafStructure(buf);
    if (buf[0] == @intFromEnum(Kind.internal)) return verifyInternalStructure(buf);
    return false;
}

fn verifyLeafStructure(buf: []const u8) bool {
    const h: *const LeafHeader = @ptrCast(@alignCast(buf.ptr));
    const n_entries: usize = h.n_entries;
    const total_cell_bytes: usize = h.total_cell_bytes;
    const slot_area_end: usize = HEADER_SIZE + n_entries * SLOT_SIZE;
    if (slot_area_end > PAGE_SIZE) return false;
    if (total_cell_bytes > PAGE_SIZE - slot_area_end) return false;

    var sum: usize = 0;
    var i: usize = 0;
    while (i < n_entries) : (i += 1) {
        const off: usize = std.mem.readInt(u16, buf[HEADER_SIZE + i * SLOT_SIZE ..][0..2], .little);
        if (off < slot_area_end) return false;
        if (off + 4 > PAGE_SIZE) return false;
        const key_len: usize = std.mem.readInt(u16, buf[off..][0..2], .little);
        const val_len: usize = std.mem.readInt(u16, buf[off + 2 ..][0..2], .little);
        if (key_len > MAX_KEY_LEN) return false;
        if (val_len > MAX_VAL_LEN) return false;
        const cell_size = 4 + key_len + val_len;
        if (off + cell_size > PAGE_SIZE) return false;
        sum += cell_size;
    }
    return sum == total_cell_bytes;
}

fn verifyInternalStructure(buf: []const u8) bool {
    const h: *const InternalHeader = @ptrCast(@alignCast(buf.ptr));
    const n_entries: usize = h.n_entries;
    const total_cell_bytes: usize = h.total_cell_bytes;
    const slot_area_end: usize = HEADER_SIZE + n_entries * SLOT_SIZE;
    if (slot_area_end > PAGE_SIZE) return false;
    if (total_cell_bytes > PAGE_SIZE - slot_area_end) return false;

    var sum: usize = 0;
    var i: usize = 0;
    while (i < n_entries) : (i += 1) {
        const off: usize = std.mem.readInt(u16, buf[HEADER_SIZE + i * SLOT_SIZE ..][0..2], .little);
        if (off < slot_area_end) return false;
        if (off + 10 > PAGE_SIZE) return false;
        const key_len: usize = std.mem.readInt(u16, buf[off..][0..2], .little);
        if (key_len > MAX_KEY_LEN) return false;
        const cell_size = 10 + key_len;
        if (off + cell_size > PAGE_SIZE) return false;
        sum += cell_size;
    }
    return sum == total_cell_bytes;
}

pub fn leafCellSize(key_len: usize, val_len: usize) usize {
    return 4 + key_len + val_len;
}

pub fn internalCellSize(key_len: usize) usize {
    return 10 + key_len;
}

/// Accessor over a 4KB buffer interpreted as a leaf page.
pub const Leaf = struct {
    buf: []u8,

    pub fn init(buf: []u8) Leaf {
        @memset(buf[0..HEADER_SIZE], 0);
        const h: *LeafHeader = @ptrCast(@alignCast(buf.ptr));
        h.* = .{};
        return .{ .buf = buf };
    }

    pub fn view(buf: []u8) Leaf {
        return .{ .buf = buf };
    }

    pub fn header(self: Leaf) *LeafHeader {
        return @ptrCast(@alignCast(self.buf.ptr));
    }

    pub fn nEntries(self: Leaf) u16 {
        return self.header().n_entries;
    }

    fn slotPtr(self: Leaf, i: usize) *u16 {
        return @ptrCast(@alignCast(self.buf.ptr + HEADER_SIZE + i * SLOT_SIZE));
    }

    pub fn slotOffset(self: Leaf, i: usize) u16 {
        return self.slotPtr(i).*;
    }

    pub fn keyAt(self: Leaf, i: usize) []const u8 {
        const off = self.slotOffset(i);
        const key_len = std.mem.readInt(u16, self.buf[off..][0..2], .little);
        return self.buf[off + 4 ..][0..key_len];
    }

    pub fn valueAt(self: Leaf, i: usize) []const u8 {
        const off = self.slotOffset(i);
        const key_len = std.mem.readInt(u16, self.buf[off..][0..2], .little);
        const val_len = std.mem.readInt(u16, self.buf[off + 2 ..][0..2], .little);
        return self.buf[off + 4 + key_len ..][0..val_len];
    }

    pub fn freeSpace(self: Leaf) usize {
        const h = self.header();
        return PAGE_SIZE - HEADER_SIZE - @as(usize, h.n_entries) * SLOT_SIZE - h.total_cell_bytes;
    }

    pub fn canFit(self: Leaf, key_len: usize, val_len: usize) bool {
        return self.freeSpace() >= leafCellSize(key_len, val_len) + SLOT_SIZE;
    }

    /// Append a cell with the next slot index. Caller is responsible for
    /// appending in sorted key order. Returns false if it didn't fit.
    pub fn appendCell(self: Leaf, key: []const u8, value: []const u8) bool {
        if (!self.canFit(key.len, value.len)) return false;
        const h = self.header();
        const cell_size = leafCellSize(key.len, value.len);
        const cell_offset: u16 = @intCast(PAGE_SIZE - @as(usize, h.total_cell_bytes) - cell_size);
        std.mem.writeInt(u16, self.buf[cell_offset..][0..2], @intCast(key.len), .little);
        std.mem.writeInt(u16, self.buf[cell_offset + 2 ..][0..2], @intCast(value.len), .little);
        @memcpy(self.buf[cell_offset + 4 ..][0..key.len], key);
        @memcpy(self.buf[cell_offset + 4 + key.len ..][0..value.len], value);
        self.slotPtr(h.n_entries).* = cell_offset;
        h.n_entries += 1;
        h.total_cell_bytes += @intCast(cell_size);
        return true;
    }

    /// Binary search. `found` is true iff a slot's key equals `key`;
    /// `idx` is either that slot or the insertion point (lower_bound).
    pub fn search(self: Leaf, key: []const u8) SearchHit {
        var lo: usize = 0;
        var hi: usize = self.header().n_entries;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            switch (std.mem.order(u8, self.keyAt(mid), key)) {
                .lt => lo = mid + 1,
                .gt => hi = mid,
                .eq => return .{ .idx = mid, .found = true },
            }
        }
        return .{ .idx = lo, .found = false };
    }
};

pub const SearchHit = struct { idx: usize, found: bool };

/// Accessor over a 4KB buffer interpreted as an internal page.
///
/// Convention: each cell is `(key, child_left_of_key)`; children with
/// keys ≥ the largest cell key live in `rightmost_child`. Children
/// laid out left-to-right are
/// `[cells[0].child, ..., cells[N-1].child, rightmost_child]`.
pub const Internal = struct {
    buf: []u8,

    pub fn init(buf: []u8, rightmost: u64) Internal {
        @memset(buf[0..HEADER_SIZE], 0);
        const h: *InternalHeader = @ptrCast(@alignCast(buf.ptr));
        h.* = .{ .rightmost_child = rightmost };
        return .{ .buf = buf };
    }

    pub fn view(buf: []u8) Internal {
        return .{ .buf = buf };
    }

    pub fn header(self: Internal) *InternalHeader {
        return @ptrCast(@alignCast(self.buf.ptr));
    }

    pub fn nEntries(self: Internal) u16 {
        return self.header().n_entries;
    }

    pub fn rightmostChild(self: Internal) u64 {
        return self.header().rightmost_child;
    }

    /// In-place mutator for the rightmost child pointer. Used by the
    /// hybrid CoW path to retarget the rightmost child without
    /// reCoW'ing the whole page.
    pub fn setRightmostChild(self: Internal, child: u64) void {
        self.header().rightmost_child = child;
    }

    /// In-place mutator for an indexed child pointer.
    pub fn setChildAt(self: Internal, i: usize, child: u64) void {
        const off = self.slotOffset(i);
        std.mem.writeInt(u64, self.buf[off + 2 ..][0..8], child, .little);
    }

    fn slotPtr(self: Internal, i: usize) *u16 {
        return @ptrCast(@alignCast(self.buf.ptr + HEADER_SIZE + i * SLOT_SIZE));
    }

    pub fn slotOffset(self: Internal, i: usize) u16 {
        return self.slotPtr(i).*;
    }

    pub fn keyAt(self: Internal, i: usize) []const u8 {
        const off = self.slotOffset(i);
        const key_len = std.mem.readInt(u16, self.buf[off..][0..2], .little);
        return self.buf[off + 10 ..][0..key_len];
    }

    pub fn childAt(self: Internal, i: usize) u64 {
        const off = self.slotOffset(i);
        return std.mem.readInt(u64, self.buf[off + 2 ..][0..8], .little);
    }

    pub fn freeSpace(self: Internal) usize {
        const h = self.header();
        return PAGE_SIZE - HEADER_SIZE - @as(usize, h.n_entries) * SLOT_SIZE - h.total_cell_bytes;
    }

    pub fn canFit(self: Internal, key_len: usize) bool {
        return self.freeSpace() >= internalCellSize(key_len) + SLOT_SIZE;
    }

    pub fn appendCell(self: Internal, key: []const u8, child: u64) bool {
        if (!self.canFit(key.len)) return false;
        const h = self.header();
        const cell_size = internalCellSize(key.len);
        const cell_offset: u16 = @intCast(PAGE_SIZE - @as(usize, h.total_cell_bytes) - cell_size);
        std.mem.writeInt(u16, self.buf[cell_offset..][0..2], @intCast(key.len), .little);
        std.mem.writeInt(u64, self.buf[cell_offset + 2 ..][0..8], child, .little);
        @memcpy(self.buf[cell_offset + 10 ..][0..key.len], key);
        self.slotPtr(h.n_entries).* = cell_offset;
        h.n_entries += 1;
        h.total_cell_bytes += @intCast(cell_size);
        return true;
    }

    /// Returns the child slot index for descent. `n_entries` means
    /// "descend into `rightmost_child`."
    pub fn childSlotForKey(self: Internal, key: []const u8) usize {
        // upper_bound: first slot where cells[i].key > key.
        var lo: usize = 0;
        var hi: usize = self.header().n_entries;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (std.mem.order(u8, self.keyAt(mid), key) == .gt) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    pub fn childForKey(self: Internal, key: []const u8) u64 {
        const i = self.childSlotForKey(key);
        if (i == self.header().n_entries) return self.rightmostChild();
        return self.childAt(i);
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

test "Leaf: init empty, append, read back in order" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    try testing.expect(l.appendCell("alpha", "A"));
    try testing.expect(l.appendCell("beta", "BB"));
    try testing.expect(l.appendCell("gamma", "CCC"));

    try testing.expectEqual(@as(u16, 3), l.nEntries());
    try testing.expectEqualStrings("alpha", l.keyAt(0));
    try testing.expectEqualStrings("A", l.valueAt(0));
    try testing.expectEqualStrings("gamma", l.keyAt(2));
    try testing.expectEqualStrings("CCC", l.valueAt(2));
}

test "Leaf: search hits and misses" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    _ = l.appendCell("alpha", "A");
    _ = l.appendCell("delta", "D");
    _ = l.appendCell("kappa", "K");

    const hit = l.search("delta");
    try testing.expect(hit.found);
    try testing.expectEqual(@as(usize, 1), hit.idx);

    const miss = l.search("epsilon");
    try testing.expect(!miss.found);
    try testing.expectEqual(@as(usize, 2), miss.idx);

    const before = l.search("aaa");
    try testing.expect(!before.found);
    try testing.expectEqual(@as(usize, 0), before.idx);

    const after = l.search("zzz");
    try testing.expect(!after.found);
    try testing.expectEqual(@as(usize, 3), after.idx);
}

test "Leaf: appendCell returns false when page full" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    const big_val = [_]u8{0xAA} ** MAX_VAL_LEN;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>10}", .{i});
        if (!l.appendCell(key, &big_val)) break;
    }
    try testing.expect(i < 100);
    try testing.expect(i >= 1);
}

test "Internal: init + appendCell + childForKey" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    // Layout: child 100 holds keys < "m"; child 200 holds keys in ["m", "s"); rightmost 300 holds keys ≥ "s".
    var n = Internal.init(&buf, 300);
    try testing.expect(n.appendCell("m", 100));
    try testing.expect(n.appendCell("s", 200));

    try testing.expectEqual(@as(u64, 100), n.childForKey("apple"));
    try testing.expectEqual(@as(u64, 100), n.childForKey("l"));
    try testing.expectEqual(@as(u64, 200), n.childForKey("m"));
    try testing.expectEqual(@as(u64, 200), n.childForKey("ms"));
    try testing.expectEqual(@as(u64, 300), n.childForKey("s"));
    try testing.expectEqual(@as(u64, 300), n.childForKey("zebra"));
}

test "Internal: childSlotForKey returns sentinel n_entries for rightmost" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var n = Internal.init(&buf, 999);
    _ = n.appendCell("k1", 1);
    _ = n.appendCell("k2", 2);
    try testing.expectEqual(@as(usize, 2), n.childSlotForKey("zzz"));
    try testing.expectEqual(@as(usize, 0), n.childSlotForKey("a"));
    try testing.expectEqual(@as(usize, 1), n.childSlotForKey("k1xx"));
}

test "pageKind reads kind from header byte 0" {
    var leaf_buf: [PAGE_SIZE]u8 align(4096) = undefined;
    _ = Leaf.init(&leaf_buf);
    try testing.expectEqual(Kind.leaf, pageKind(&leaf_buf));

    var int_buf: [PAGE_SIZE]u8 align(4096) = undefined;
    _ = Internal.init(&int_buf, 7);
    try testing.expectEqual(Kind.internal, pageKind(&int_buf));
}

test "verifyStructure: empty leaf passes" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    @memset(&buf, 0xCD);
    _ = Leaf.init(&buf);
    try testing.expect(verifyStructure(&buf));
}

test "verifyStructure: populated leaf passes" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    _ = l.appendCell("alpha", "A");
    _ = l.appendCell("beta", "BB");
    _ = l.appendCell("gamma", "CCC");
    try testing.expect(verifyStructure(&buf));
}

test "verifyStructure: empty internal passes" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    @memset(&buf, 0xCD);
    _ = Internal.init(&buf, 42);
    try testing.expect(verifyStructure(&buf));
}

test "verifyStructure: populated internal passes" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var n = Internal.init(&buf, 300);
    _ = n.appendCell("m", 100);
    _ = n.appendCell("s", 200);
    try testing.expect(verifyStructure(&buf));
}

test "verifyStructure: unknown kind byte rejected" {
    var buf: [PAGE_SIZE]u8 align(4096) = [_]u8{0} ** PAGE_SIZE;
    buf[0] = 99;
    try testing.expect(!verifyStructure(&buf));
}

test "verifyStructure: oversized n_entries rejected" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    _ = Leaf.init(&buf);
    const h: *LeafHeader = @ptrCast(@alignCast(&buf));
    h.n_entries = (PAGE_SIZE - HEADER_SIZE) / SLOT_SIZE + 1; // slot array overflows
    try testing.expect(!verifyStructure(&buf));
}

test "verifyStructure: slot pointing into slot area rejected" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    _ = l.appendCell("k", "v");
    // Overwrite the slot offset to point inside the slot array.
    std.mem.writeInt(u16, buf[HEADER_SIZE..][0..2], HEADER_SIZE, .little);
    try testing.expect(!verifyStructure(&buf));
}

test "verifyStructure: cell running past PAGE_SIZE rejected" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    _ = l.appendCell("k", "v");
    // Forge a key_len that would push the cell past the end.
    const off = std.mem.readInt(u16, buf[HEADER_SIZE..][0..2], .little);
    std.mem.writeInt(u16, buf[off..][0..2], @intCast(PAGE_SIZE), .little);
    try testing.expect(!verifyStructure(&buf));
}

test "verifyStructure: total_cell_bytes mismatch rejected" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    _ = l.appendCell("k", "v");
    const h: *LeafHeader = @ptrCast(@alignCast(&buf));
    h.total_cell_bytes += 1;
    try testing.expect(!verifyStructure(&buf));
}

test "verifyStructure: key_len > MAX_KEY_LEN rejected" {
    var buf: [PAGE_SIZE]u8 align(4096) = undefined;
    var l = Leaf.init(&buf);
    _ = l.appendCell("k", "v");
    const off = std.mem.readInt(u16, buf[HEADER_SIZE..][0..2], .little);
    std.mem.writeInt(u16, buf[off..][0..2], MAX_KEY_LEN + 1, .little);
    try testing.expect(!verifyStructure(&buf));
}
