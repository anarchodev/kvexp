//! Per-store in-memory write buffer ("memtable" / overlay).
//!
//! ## Why this exists
//!
//! The original kvexp design paid full CoW cost per `Store.put` — every
//! mutation walked the tree, allocated new pages, copied content,
//! propagated CoW upward, and dirtied the page cache. Profiled per-put
//! overhead was ~30% of CPU, vs SQLite's WAL append at near-zero. The
//! overlay collapses every mutation into a single in-memory map insert;
//! the CoW machinery runs only at `durabilize` time, on the union of
//! mutations rather than each one individually. Hot keys updated 100×
//! between durabilizes produce one CoW chain, not 100.
//!
//! ## Semantics
//!
//! Each `Store` has at most one `Overlay`, allocated lazily on the
//! first put or delete. The overlay maps keys to `OverlayEntry`:
//!
//!   * `.value` — the key was put with this value. Overrides whatever
//!                the durable tree says for this key.
//!   * `.tombstone` — the key was deleted. Reads must NOT fall through
//!                    to the durable tree.
//!
//! Reads check the overlay first; on miss (key absent), fall through
//! to the tree. On hit with `.value`, return that value. On hit with
//! `.tombstone`, return null without checking the tree.
//!
//! `durabilize` drains the overlay into the tree via `tree.put` /
//! `tree.delete`, then clears the overlay. All the per-op CoW cost is
//! paid here, in bulk, single-threaded.
//!
//! ## Memory ownership
//!
//! The overlay owns all key and value bytes. Inserts clone the caller's
//! slices into overlay-allocated buffers. Overwrites free the old
//! buffer and clone the new. Tombstones free the value buffer but
//! keep the key. `deinit` / `clear` free everything.
//!
//! ## Concurrency
//!
//! Each overlay has its own mutex. Workers on different stores never
//! contend (each Store routes to its own Overlay). Reads and writes
//! within a single store serialize on this mutex — same isolation
//! guarantee as the per-store write lock today, but cheaper.

const std = @import("std");

pub const OverlayEntry = union(enum) {
    value: []u8,
    tombstone,
};

pub const Overlay = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMapUnmanaged(OverlayEntry) = .empty,
    lock: std.Thread.Mutex = .{},
    /// Sum of `key.len + value.len` for every `.value` entry, plus
    /// `key.len` for every `.tombstone`. Tombstones don't count value
    /// bytes (they don't store any) but they do count their key
    /// allocation. Maintained inline by put/tombstone/clear/moveInto;
    /// callers reach for this via `Manifest.max_overlay_bytes_per_store`
    /// to back-pressure tenants that would otherwise grow the overlay
    /// unboundedly between durabilizes.
    bytes: usize = 0,

    pub fn init(allocator: std.mem.Allocator) Overlay {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Overlay) void {
        self.clear();
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Drop every entry, freeing the underlying key/value buffers.
    /// Called by durabilize after applying the overlay to the tree.
    pub fn clear(self: *Overlay) void {
        var it = self.entries.iterator();
        while (it.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            switch (e.value_ptr.*) {
                .value => |v| self.allocator.free(v),
                .tombstone => {},
            }
        }
        self.entries.clearRetainingCapacity();
        self.bytes = 0;
    }

    /// Insert or replace `key → value`. Both slices are cloned into
    /// overlay-owned memory; the caller retains ownership of its own
    /// buffers. If the key was previously tombstoned, the tombstone
    /// is replaced. Caller must hold `self.lock`.
    ///
    /// Panics on OOM — kvexp's recovery model is raft replay from the
    /// durable watermark, so failed internal allocations crash the
    /// host hard and loud rather than silently propagating.
    pub fn putLocked(self: *Overlay, key: []const u8, value: []const u8) void {
        const gop = self.entries.getOrPut(self.allocator, key) catch
            @panic("OOM in Overlay.putLocked.getOrPut");
        if (gop.found_existing) {
            switch (gop.value_ptr.*) {
                .value => |old| {
                    self.bytes -= old.len;
                    self.allocator.free(old);
                },
                .tombstone => {},
            }
        } else {
            // New key: clone the bytes so the hashmap owns them.
            gop.key_ptr.* = self.allocator.dupe(u8, key) catch
                @panic("OOM in Overlay.putLocked duping key");
            self.bytes += key.len;
        }
        gop.value_ptr.* = .{ .value = self.allocator.dupe(u8, value) catch
            @panic("OOM in Overlay.putLocked duping value") };
        self.bytes += value.len;
    }

    /// Mark `key` deleted. Frees any prior value bytes; keeps the key
    /// allocation. Caller must hold `self.lock`. Panics on OOM.
    pub fn tombstoneLocked(self: *Overlay, key: []const u8) void {
        const gop = self.entries.getOrPut(self.allocator, key) catch
            @panic("OOM in Overlay.tombstoneLocked.getOrPut");
        if (gop.found_existing) {
            switch (gop.value_ptr.*) {
                .value => |old| {
                    self.bytes -= old.len;
                    self.allocator.free(old);
                },
                .tombstone => {},
            }
        } else {
            gop.key_ptr.* = self.allocator.dupe(u8, key) catch
                @panic("OOM in Overlay.tombstoneLocked duping key");
            self.bytes += key.len;
        }
        gop.value_ptr.* = .tombstone;
    }

    /// Look up `key`. Returns:
    ///   * `null` → key not in overlay; caller should consult the tree.
    ///   * `OverlayEntry.value` → return this value (bytes alias the
    ///       overlay's memory; caller must dupe if it needs to outlive
    ///       the lock).
    ///   * `OverlayEntry.tombstone` → the key is deleted; do NOT
    ///       consult the tree.
    pub fn getLocked(self: *const Overlay, key: []const u8) ?OverlayEntry {
        return self.entries.get(key);
    }

    /// Move every entry from `src` into `dest`, transferring ownership
    /// of the key/value bytes rather than re-cloning. After the call,
    /// `src` is empty; `dest` has the union of both sets with `src`'s
    /// entries winning collisions (key existed in both → dest gets
    /// src's value, dest's old value/tombstone is freed). Caller must
    /// hold both overlays' locks (or otherwise serialize access).
    pub fn moveInto(src: *Overlay, dest: *Overlay) void {
        std.debug.assert(src.allocator.ptr == dest.allocator.ptr);
        var it = src.entries.iterator();
        while (it.next()) |e| {
            const k = e.key_ptr.*;
            const v = e.value_ptr.*;
            const src_v_bytes: usize = switch (v) {
                .value => |val| val.len,
                .tombstone => 0,
            };
            const gop = dest.entries.getOrPut(dest.allocator, k) catch
                @panic("OOM in Overlay.moveInto.getOrPut");
            if (gop.found_existing) {
                // dest already has this key. Free OUR key bytes (dest
                // keeps its own copy), free dest's prior value, install
                // our value.
                src.allocator.free(k);
                switch (gop.value_ptr.*) {
                    .value => |old| {
                        dest.bytes -= old.len;
                        dest.allocator.free(old);
                    },
                    .tombstone => {},
                }
                dest.bytes += src_v_bytes;
                gop.value_ptr.* = v;
            } else {
                // Transfer src's key and value into dest verbatim.
                gop.key_ptr.* = k;
                gop.value_ptr.* = v;
                dest.bytes += k.len + src_v_bytes;
            }
        }
        src.entries.clearRetainingCapacity();
        src.bytes = 0;
    }

    /// True iff the overlay has no entries (use to skip durabilize-time
    /// work for untouched stores). Caller must hold `self.lock`.
    pub fn isEmptyLocked(self: *const Overlay) bool {
        return self.entries.count() == 0;
    }

    /// Count of entries (values + tombstones combined). Test
    /// introspection. Caller must hold `self.lock`.
    pub fn countLocked(self: *const Overlay) usize {
        return self.entries.count();
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

test "Overlay: put then get returns value" {
    var ov = Overlay.init(testing.allocator);
    defer ov.deinit();
    ov.lock.lock();
    defer ov.lock.unlock();
    ov.putLocked("k", "v");
    const got = ov.getLocked("k").?;
    try testing.expectEqualStrings("v", got.value);
}

test "Overlay: put then put overwrites; old value freed" {
    var ov = Overlay.init(testing.allocator);
    defer ov.deinit();
    ov.lock.lock();
    defer ov.lock.unlock();
    ov.putLocked("k", "v1");
    ov.putLocked("k", "v2");
    const got = ov.getLocked("k").?;
    try testing.expectEqualStrings("v2", got.value);
}

test "Overlay: tombstone after put marks deleted" {
    var ov = Overlay.init(testing.allocator);
    defer ov.deinit();
    ov.lock.lock();
    defer ov.lock.unlock();
    ov.putLocked("k", "v");
    ov.tombstoneLocked("k");
    const got = ov.getLocked("k").?;
    try testing.expect(got == .tombstone);
}

test "Overlay: put after tombstone resurrects key" {
    var ov = Overlay.init(testing.allocator);
    defer ov.deinit();
    ov.lock.lock();
    defer ov.lock.unlock();
    ov.tombstoneLocked("k");
    ov.putLocked("k", "v");
    const got = ov.getLocked("k").?;
    try testing.expectEqualStrings("v", got.value);
}

test "Overlay: missing key returns null" {
    var ov = Overlay.init(testing.allocator);
    defer ov.deinit();
    ov.lock.lock();
    defer ov.lock.unlock();
    try testing.expect(ov.getLocked("absent") == null);
}

test "Overlay: clear frees everything" {
    var ov = Overlay.init(testing.allocator);
    defer ov.deinit();
    ov.lock.lock();
    defer ov.lock.unlock();
    ov.putLocked("a", "1");
    ov.putLocked("b", "2");
    ov.tombstoneLocked("c");
    try testing.expectEqual(@as(usize, 3), ov.countLocked());
    ov.clear();
    try testing.expectEqual(@as(usize, 0), ov.countLocked());
    try testing.expect(ov.isEmptyLocked());
}

test "Overlay: many puts then clear (no leaks under GPA)" {
    var ov = Overlay.init(testing.allocator);
    defer ov.deinit();
    ov.lock.lock();
    defer ov.lock.unlock();
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        var kb: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&kb, "key{d}", .{i});
        var vb: [16]u8 = undefined;
        const v = try std.fmt.bufPrint(&vb, "value{d}", .{i});
        ov.putLocked(k, v);
    }
    try testing.expectEqual(@as(usize, 100), ov.countLocked());
    ov.clear();
    try testing.expectEqual(@as(usize, 0), ov.countLocked());
}
