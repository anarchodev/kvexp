//! File header (page 0). Identifies the file as kvexp, pins the format
//! version, declares the page size, and (later phases) points at the
//! active manifest slot.
//!
//! Layout is fixed at exactly one page. `extern struct` so it can be
//! pwritten directly without re-encoding.

const std = @import("std");

pub const MAGIC: [8]u8 = .{ 'k', 'v', 'e', 'x', 'p', 0, 0, 1 };
/// Bumped to 2 when per-page CRC32 was added to LeafHeader /
/// InternalHeader at offset 8. v1 had no checksum field and InternalHeader's
/// rightmost_child sat at offset 8 instead of 12; opening a v1 file with v2
/// code would read rightmost_child as zero and treat every internal page
/// as having an empty rightmost child — catastrophic data loss. Refuse
/// v1 at open and force a clean wipe.
pub const FORMAT_VERSION: u32 = 2;
pub const HEADER_PAGE_NO: u64 = 0;

pub const Header = extern struct {
    magic: [8]u8 align(1),
    format_version: u32 align(1),
    page_size: u32 align(1),
    created_at_unix: i64 align(1),
    /// Index of the currently-active manifest slot (0 or 1).
    /// Phase 0/1: unused — always 0.
    manifest_active_slot: u32 align(1),
    _reserved: [4096 - 28]u8 align(1) = @splat(0),

    comptime {
        std.debug.assert(@sizeOf(Header) == 4096);
    }

    pub fn init(page_size: u32) Header {
        return .{
            .magic = MAGIC,
            .format_version = FORMAT_VERSION,
            .page_size = page_size,
            .created_at_unix = std.time.timestamp(),
            .manifest_active_slot = 0,
        };
    }

    pub const ValidateError = error{
        BadMagic,
        UnsupportedVersion,
        UnsupportedPageSize,
    };

    pub fn validate(self: *const Header, expected_page_size: u32) ValidateError!void {
        if (!std.mem.eql(u8, &self.magic, &MAGIC)) return error.BadMagic;
        if (self.format_version != FORMAT_VERSION) return error.UnsupportedVersion;
        if (self.page_size != expected_page_size) return error.UnsupportedPageSize;
    }

    /// View a page-sized byte buffer as a Header. Buffer must outlive
    /// the returned pointer.
    pub fn fromBytes(bytes: []const u8) *const Header {
        std.debug.assert(bytes.len >= @sizeOf(Header));
        return @ptrCast(@alignCast(bytes.ptr));
    }

    pub fn toBytes(self: *const Header) []const u8 {
        const ptr: [*]const u8 = @ptrCast(self);
        return ptr[0..@sizeOf(Header)];
    }
};

test "Header is exactly one 4KB page" {
    try std.testing.expectEqual(@as(usize, 4096), @sizeOf(Header));
}

test "Header init then validate round-trips" {
    const h = Header.init(4096);
    try h.validate(4096);
    try std.testing.expectEqualSlices(u8, &MAGIC, &h.magic);
    try std.testing.expectEqual(FORMAT_VERSION, h.format_version);
    try std.testing.expectEqual(@as(u32, 4096), h.page_size);
}

test "Header rejects bad magic" {
    var h = Header.init(4096);
    h.magic[0] = 'x';
    try std.testing.expectError(error.BadMagic, h.validate(4096));
}

test "Header rejects wrong version" {
    var h = Header.init(4096);
    h.format_version = 99;
    try std.testing.expectError(error.UnsupportedVersion, h.validate(4096));
}

test "Header rejects mismatched page size" {
    const h = Header.init(4096);
    try std.testing.expectError(error.UnsupportedPageSize, h.validate(8192));
}

test "Header fromBytes/toBytes round-trip" {
    const h = Header.init(4096);
    const bytes = h.toBytes();
    const h2 = Header.fromBytes(bytes);
    try h2.validate(4096);
    try std.testing.expectEqual(h.created_at_unix, h2.created_at_unix);
}
