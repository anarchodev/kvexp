//! Manifest: forest root pointer + per-store directory.
//!
//! Layout in the data file:
//!   page 0       — reserved for the higher-layer file header (phase 1)
//!   page 1       — manifest slot A
//!   page 2       — manifest slot B
//!   page 3..     — data (manifest tree pages + per-store B-tree pages)
//!
//! Each slot is a single 4KB page containing the **manifest tree root
//! pointer** plus a monotonic sequence and a CRC32 over the slot's
//! bytes. Commits alternate between slots: write the inactive slot,
//! fsync, swap "active." A torn write to one slot leaves the other
//! valid, so recovery picks the one with the highest valid sequence.
//!
//! The manifest tree itself is a regular kvexp B-tree, keyed by an
//! 8-byte big-endian store_id (so byte-order == numeric order) with
//! 8-byte little-endian root_page values.

const std = @import("std");
const page = @import("page.zig");
const btree = @import("btree.zig");
const Tree = btree.Tree;
const PagedFile = @import("paged_file.zig").PagedFile;
const PageCache = @import("page_cache.zig").PageCache;

pub const SLOT_A_PAGE: u64 = 1;
pub const SLOT_B_PAGE: u64 = 2;
pub const FIRST_DATA_PAGE: u64 = 3;

pub const STORE_ID_LEN: usize = 8;
pub const STORE_VAL_LEN: usize = 8;

/// 4KB header slot. Two of these live at pages 1 and 2 of the file.
pub const ManifestSlot = extern struct {
    magic: u32 align(1) = 0,
    slot_id: u8 align(1) = 0,
    _pad: [3]u8 align(1) = .{ 0, 0, 0 },
    sequence: u64 align(1) = 0,
    manifest_root: u64 align(1) = 0,
    _reserved: [page.PAGE_SIZE - 28]u8 align(1) = @splat(0),
    checksum: u32 align(1) = 0,

    pub const MAGIC: u32 = 0x6B764D31; // "kvM1"

    comptime {
        std.debug.assert(@sizeOf(@This()) == page.PAGE_SIZE);
    }

    pub fn computeChecksum(self: *ManifestSlot) void {
        const bytes: [*]const u8 = @ptrCast(self);
        const crc = std.hash.Crc32.hash(bytes[0 .. @sizeOf(ManifestSlot) - 4]);
        self.checksum = crc;
    }

    pub fn isValid(self: *const ManifestSlot) bool {
        if (self.magic != MAGIC) return false;
        const bytes: [*]const u8 = @ptrCast(self);
        const crc = std.hash.Crc32.hash(bytes[0 .. @sizeOf(ManifestSlot) - 4]);
        return crc == self.checksum;
    }
};

pub fn encodeStoreId(id: u64, buf: *[STORE_ID_LEN]u8) []const u8 {
    std.mem.writeInt(u64, buf, id, .big);
    return buf;
}

pub fn decodeStoreId(bytes: []const u8) u64 {
    std.debug.assert(bytes.len == STORE_ID_LEN);
    return std.mem.readInt(u64, bytes[0..STORE_ID_LEN], .big);
}

pub fn encodeRoot(root: u64, buf: *[STORE_VAL_LEN]u8) []const u8 {
    std.mem.writeInt(u64, buf, root, .little);
    return buf;
}

pub fn decodeRoot(bytes: []const u8) u64 {
    std.debug.assert(bytes.len == STORE_VAL_LEN);
    return std.mem.readInt(u64, bytes[0..STORE_VAL_LEN], .little);
}

pub const Error = btree.Error || error{
    StoreAlreadyExists,
    StoreNotFound,
    ManifestCorrupt,
};

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    cache: *PageCache,
    file: *PagedFile,
    /// The manifest B-tree itself. Its root pointer is durable in the
    /// active header slot.
    tree: Tree,
    /// Slot index (0 or 1) currently holding the highest-seq valid manifest.
    active_slot: u32,
    /// Sequence of the active slot (highest durable seq).
    sequence: u64,

    pub fn open(allocator: std.mem.Allocator, cache: *PageCache, file: *PagedFile) Error!Manifest {
        // Establish layout: page 0 (reserved) + 2 slot pages.
        while (file.pageCount() < FIRST_DATA_PAGE) {
            _ = try file.growBy(1);
        }

        var buf_a: [page.PAGE_SIZE]u8 align(4096) = undefined;
        var buf_b: [page.PAGE_SIZE]u8 align(4096) = undefined;
        try file.readPage(SLOT_A_PAGE, &buf_a);
        try file.readPage(SLOT_B_PAGE, &buf_b);

        const slot_a: *ManifestSlot = @ptrCast(@alignCast(&buf_a));
        const slot_b: *ManifestSlot = @ptrCast(@alignCast(&buf_b));
        const va = slot_a.isValid();
        const vb = slot_b.isValid();

        var active_slot: u32 = 1;
        var sequence: u64 = 0;
        var manifest_root: u64 = 0;
        if (va and vb) {
            if (slot_a.sequence >= slot_b.sequence) {
                active_slot = 0;
                sequence = slot_a.sequence;
                manifest_root = slot_a.manifest_root;
            } else {
                active_slot = 1;
                sequence = slot_b.sequence;
                manifest_root = slot_b.manifest_root;
            }
        } else if (va) {
            active_slot = 0;
            sequence = slot_a.sequence;
            manifest_root = slot_a.manifest_root;
        } else if (vb) {
            active_slot = 1;
            sequence = slot_b.sequence;
            manifest_root = slot_b.manifest_root;
        } // else: both invalid → fresh file; active_slot left at 1 so first commit writes to 0

        var tree = try Tree.init(allocator, cache, file);
        tree.root = manifest_root;
        tree.seq = sequence + 1;

        return .{
            .allocator = allocator,
            .cache = cache,
            .file = file,
            .tree = tree,
            .active_slot = active_slot,
            .sequence = sequence,
        };
    }

    pub fn deinit(self: *Manifest) void {
        _ = self;
    }

    /// Pending (in-flight) sequence — the seq tag that mutations are
    /// currently producing dirty pages with. After commit, this becomes
    /// the new durable sequence and a fresh pending seq starts.
    pub fn pendingSeq(self: *const Manifest) u64 {
        return self.tree.seq;
    }

    pub fn hasStore(self: *Manifest, id: u64) !bool {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        const v = try self.tree.get(self.allocator, k);
        if (v) |bytes| {
            self.allocator.free(bytes);
            return true;
        }
        return false;
    }

    pub fn storeRoot(self: *Manifest, id: u64) !?u64 {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        const v = try self.tree.get(self.allocator, k);
        if (v) |bytes| {
            defer self.allocator.free(bytes);
            return decodeRoot(bytes);
        }
        return null;
    }

    pub fn createStore(self: *Manifest, id: u64) Error!void {
        if (try self.hasStore(id)) return error.StoreAlreadyExists;
        try self.setStoreRoot(id, 0);
    }

    pub fn dropStore(self: *Manifest, id: u64) !bool {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        return try self.tree.delete(k);
    }

    pub fn setStoreRoot(self: *Manifest, id: u64, root: u64) Error!void {
        var id_buf: [STORE_ID_LEN]u8 = undefined;
        const k = encodeStoreId(id, &id_buf);
        var val_buf: [STORE_VAL_LEN]u8 = undefined;
        const v = encodeRoot(root, &val_buf);
        try self.tree.put(k, v);
    }

    /// Return all store_ids in ascending order. Caller owns the slice.
    pub fn listStores(self: *Manifest, allocator: std.mem.Allocator) ![]u64 {
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        var cursor = try self.tree.scanPrefix("");
        defer cursor.deinit();
        while (try cursor.next()) {
            try list.append(allocator, decodeStoreId(cursor.key()));
        }
        return try list.toOwnedSlice(allocator);
    }

    /// Persist all pending changes:
    ///   1. fsync data pages produced at this seq
    ///   2. write the inactive slot
    ///   3. fsync the slot
    /// On success the new sequence is durable.
    pub fn commit(self: *Manifest) !void {
        try self.cache.flushUpTo(self.tree.seq);
        try self.file.fsync();

        const next_slot: u32 = 1 - self.active_slot;
        const slot_page: u64 = if (next_slot == 0) SLOT_A_PAGE else SLOT_B_PAGE;

        var slot_buf: [page.PAGE_SIZE]u8 align(4096) = undefined;
        @memset(&slot_buf, 0);
        const slot: *ManifestSlot = @ptrCast(@alignCast(&slot_buf));
        slot.* = .{
            .magic = ManifestSlot.MAGIC,
            .slot_id = @intCast(next_slot),
            .sequence = self.tree.seq,
            .manifest_root = self.tree.root,
        };
        slot.computeChecksum();
        try self.file.writePage(slot_page, &slot_buf);
        try self.file.fsync();

        self.active_slot = next_slot;
        self.sequence = self.tree.seq;
        self.tree.seq += 1;
    }
};

// -----------------------------------------------------------------------------
// Store wrapper
// -----------------------------------------------------------------------------

pub const Store = struct {
    manifest: *Manifest,
    id: u64,
    tree: Tree,

    pub fn open(manifest: *Manifest, id: u64) !Store {
        const root = (try manifest.storeRoot(id)) orelse return error.StoreNotFound;
        var tree = try Tree.init(manifest.allocator, manifest.cache, manifest.file);
        tree.root = root;
        tree.seq = manifest.tree.seq;
        return .{ .manifest = manifest, .id = id, .tree = tree };
    }

    pub fn deinit(self: *Store) void {
        _ = self;
    }

    pub fn get(self: *Store, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        // Re-read root in case another Store handle on the same id mutated it.
        const root = (try self.manifest.storeRoot(self.id)) orelse return error.StoreNotFound;
        self.tree.root = root;
        return try self.tree.get(allocator, key);
    }

    pub fn put(self: *Store, key: []const u8, value: []const u8) !void {
        const root = (try self.manifest.storeRoot(self.id)) orelse return error.StoreNotFound;
        self.tree.root = root;
        self.tree.seq = self.manifest.tree.seq;
        try self.tree.put(key, value);
        try self.manifest.setStoreRoot(self.id, self.tree.root);
    }

    pub fn delete(self: *Store, key: []const u8) !bool {
        const root = (try self.manifest.storeRoot(self.id)) orelse return error.StoreNotFound;
        self.tree.root = root;
        self.tree.seq = self.manifest.tree.seq;
        const existed = try self.tree.delete(key);
        if (existed) {
            try self.manifest.setStoreRoot(self.id, self.tree.root);
        }
        return existed;
    }

    pub fn scanPrefix(self: *Store, prefix: []const u8) !btree.PrefixCursor {
        const root = (try self.manifest.storeRoot(self.id)) orelse return error.StoreNotFound;
        self.tree.root = root;
        return try self.tree.scanPrefix(prefix);
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;
const BufferPool = @import("buffer_pool.zig").BufferPool;

/// Test harness that allows close-and-reopen of the same on-disk file.
const Harness = struct {
    tmp: std.testing.TmpDir,
    path_buf: [std.fs.max_path_bytes]u8,
    path_len: usize,
    pool_capacity: u32,

    file: *PagedFile,
    pool: *BufferPool,
    cache: *PageCache,
    manifest: Manifest,

    fn init(pool_capacity: u32) !Harness {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();

        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &dir_buf);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const written = try std.fmt.bufPrint(&path_buf, "{s}/manifest.test", .{dir_path});
        const path_len = written.len;

        var h: Harness = .{
            .tmp = tmp,
            .path_buf = path_buf,
            .path_len = path_len,
            .pool_capacity = pool_capacity,
            .file = undefined,
            .pool = undefined,
            .cache = undefined,
            .manifest = undefined,
        };
        try h.openLayers(.{ .create = true, .truncate = true });
        return h;
    }

    fn deinit(self: *Harness) void {
        self.closeLayers();
        self.tmp.cleanup();
    }

    fn path(self: *const Harness) []const u8 {
        return self.path_buf[0..self.path_len];
    }

    fn openLayers(self: *Harness, open_opts: PagedFile.OpenOptions) !void {
        self.file = try testing.allocator.create(PagedFile);
        errdefer testing.allocator.destroy(self.file);
        self.file.* = try PagedFile.open(self.path(), open_opts);
        errdefer self.file.close();

        self.pool = try testing.allocator.create(BufferPool);
        errdefer testing.allocator.destroy(self.pool);
        self.pool.* = try BufferPool.init(testing.allocator, page.PAGE_SIZE, self.pool_capacity);
        errdefer self.pool.deinit(testing.allocator);

        self.cache = try testing.allocator.create(PageCache);
        errdefer testing.allocator.destroy(self.cache);
        self.cache.* = try PageCache.init(testing.allocator, self.file, self.pool);

        self.manifest = try Manifest.open(testing.allocator, self.cache, self.file);
    }

    fn closeLayers(self: *Harness) void {
        self.manifest.deinit();
        self.cache.deinit();
        testing.allocator.destroy(self.cache);
        self.pool.deinit(testing.allocator);
        testing.allocator.destroy(self.pool);
        self.file.close();
        testing.allocator.destroy(self.file);
    }

    /// Close everything, then reopen. Used to verify durable state.
    fn cycle(self: *Harness) !void {
        self.closeLayers();
        try self.openLayers(.{});
    }
};

test "Manifest: fresh file has no stores; commit + reopen still has none" {
    var h = try Harness.init(32);
    defer h.deinit();
    try testing.expectEqual(@as(u64, 0), h.manifest.sequence);
    try testing.expect(!try h.manifest.hasStore(42));
    try h.manifest.commit();
    try testing.expectEqual(@as(u64, 1), h.manifest.sequence);

    try h.cycle();
    try testing.expect(!try h.manifest.hasStore(42));
    try testing.expectEqual(@as(u64, 1), h.manifest.sequence);
}

test "Manifest: createStore + commit + reopen → store still exists" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.manifest.createStore(42);
    try h.manifest.createStore(7);
    try h.manifest.commit();

    try h.cycle();
    try testing.expect(try h.manifest.hasStore(42));
    try testing.expect(try h.manifest.hasStore(7));
    try testing.expect(!try h.manifest.hasStore(99));
}

test "Manifest: createStore rejects duplicate" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.manifest.createStore(1);
    try testing.expectError(error.StoreAlreadyExists, h.manifest.createStore(1));
}

test "Manifest: dropStore removes the entry" {
    var h = try Harness.init(32);
    defer h.deinit();
    try h.manifest.createStore(10);
    try h.manifest.commit();
    try testing.expect(try h.manifest.hasStore(10));

    try testing.expect(try h.manifest.dropStore(10));
    try testing.expect(!try h.manifest.dropStore(10));
    try h.manifest.commit();

    try h.cycle();
    try testing.expect(!try h.manifest.hasStore(10));
}

test "Store: put/get/delete round-trip in one store" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    var s = try Store.open(&h.manifest, 1);
    defer s.deinit();

    try s.put("hello", "world");
    const got = (try s.get(testing.allocator, "hello")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("world", got);

    try testing.expect(try s.delete("hello"));
    try testing.expect((try s.get(testing.allocator, "hello")) == null);
}

test "Manifest: multi-store writes commit and survive reopen" {
    var h = try Harness.init(128);
    defer h.deinit();
    const ids = [_]u64{ 1, 2, 3, 100, 999 };
    for (ids) |id| try h.manifest.createStore(id);

    for (ids) |id| {
        var s = try Store.open(&h.manifest, id);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d}", .{id});
        var val_buf: [16]u8 = undefined;
        const val = try std.fmt.bufPrint(&val_buf, "v{d}", .{id});
        try s.put(key, val);
    }
    try h.manifest.commit();

    try h.cycle();
    for (ids) |id| {
        var s = try Store.open(&h.manifest, id);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d}", .{id});
        var val_buf: [16]u8 = undefined;
        const expected = try std.fmt.bufPrint(&val_buf, "v{d}", .{id});
        const got = (try s.get(testing.allocator, key)).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(expected, got);
    }
}

test "Manifest: listStores returns ids in ascending order" {
    var h = try Harness.init(64);
    defer h.deinit();
    const ids = [_]u64{ 5, 1, 9, 3, 100, 42 };
    for (ids) |id| try h.manifest.createStore(id);

    const got = try h.manifest.listStores(testing.allocator);
    defer testing.allocator.free(got);

    var expected = ids;
    std.mem.sort(u64, &expected, {}, struct {
        fn lt(_: void, a: u64, b: u64) bool {
            return a < b;
        }
    }.lt);
    try testing.expectEqualSlices(u64, &expected, got);
}

test "Manifest: torn write to active slot — recovery picks the other" {
    var h = try Harness.init(64);
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.createStore(2);
    try h.manifest.commit(); // commit #1 writes slot 0

    try h.manifest.createStore(3);
    try h.manifest.commit(); // commit #2 writes slot 1; active_slot == 1

    // Sanity: slot 1 is active.
    try testing.expectEqual(@as(u32, 1), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 2), h.manifest.sequence);

    h.closeLayers();

    // Corrupt slot 1 by writing zeros. Open the file with default options
    // (no O_DIRECT here would be a problem) — we do it through a fresh
    // PagedFile so we can pwrite a zeroed page through the same path.
    {
        var pf = try PagedFile.open(h.path(), .{});
        defer pf.close();
        const zero_buf = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), page.PAGE_SIZE);
        defer testing.allocator.free(zero_buf);
        @memset(zero_buf, 0);
        try pf.writePage(SLOT_B_PAGE, zero_buf);
        try pf.fsync();
    }

    try h.openLayers(.{});

    // Recovery must have fallen back to slot 0 (the older but still-valid
    // commit). Stores 1 and 2 exist; store 3 does not.
    try testing.expectEqual(@as(u32, 0), h.manifest.active_slot);
    try testing.expectEqual(@as(u64, 1), h.manifest.sequence);
    try testing.expect(try h.manifest.hasStore(1));
    try testing.expect(try h.manifest.hasStore(2));
    try testing.expect(!try h.manifest.hasStore(3));
}

test "Manifest: 1000-store stress, commit, reopen, sample-verify" {
    var h = try Harness.init(512);
    defer h.deinit();

    const N: u64 = 1000;
    var id: u64 = 0;
    while (id < N) : (id += 1) try h.manifest.createStore(id);
    id = 0;
    while (id < N) : (id += 1) {
        var s = try Store.open(&h.manifest, id);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{id});
        var val_buf: [16]u8 = undefined;
        const val = try std.fmt.bufPrint(&val_buf, "v{d}", .{id});
        try s.put(key, val);
    }
    try h.manifest.commit();

    try h.cycle();

    // Spot-check a sparse sample.
    const samples = [_]u64{ 0, 1, 7, 99, 500, 999 };
    for (samples) |sid| {
        var s = try Store.open(&h.manifest, sid);
        defer s.deinit();
        var key_buf: [16]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{sid});
        var val_buf: [16]u8 = undefined;
        const expected = try std.fmt.bufPrint(&val_buf, "v{d}", .{sid});
        const got = (try s.get(testing.allocator, key)).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(expected, got);
    }

    const all = try h.manifest.listStores(testing.allocator);
    defer testing.allocator.free(all);
    try testing.expectEqual(N, all.len);
}
