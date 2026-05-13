//! Multi-tenant manifest over LMDB with per-tenant chains of nested
//! transactions.
//!
//! Architecture:
//!
//!   Manifest
//!   ├─ LMDB env (durable state)
//!   │    ├─ "_meta"   (raft_apply_idx watermark)
//!   │    ├─ "_stores" (directory of tenant ids)
//!   │    └─ "s_<hex>" sub-DBI per tenant
//!   │
//!   └─ per-tenant TenantState (in memory, lazy)
//!        ├─ main_overlay  ← confirmed writes pending durabilize
//!        └─ chain of open top-level Txns (in raft propose order)
//!              each Txn has:
//!              ├─ overlay   (this txn's writes)
//!              └─ open_child  ← LIFO stack: at most one savepoint open
//!                                (the savepoint can itself have one open, etc.)
//!
//! Writes go through Txns, which are opened from a `StoreLease`.
//! `Manifest.acquire(tenant_id)` takes the per-tenant dispatch lock
//! and returns a `StoreLease`; `lease.beginTxn()` opens a new top-
//! level Txn at the tail of that tenant's chain. `Txn.put` /
//! `Txn.delete` land in the txn's own overlay. `Txn.savepoint()`
//! pushes a child Txn whose parent is this one — savepoints stack
//! LIFO (single `open_child` slot). The lease may be released as soon
//! as dispatch is done; the Txn lives in the chain until its own
//! commit / rollback.
//!
//! Reads walk inside-out: savepoint stack → enclosing top-level Txn's
//! chain backward → tenant's main_overlay → LMDB read txn. Cross-
//! tenant in-flight state is never visible (each tenant has its own
//! chain and overlay; lease-based reads and `Snapshot` see only
//! main_overlay plus LMDB).
//!
//! Commit:
//!   - Top-level Txn: must be the chain head (raft-oldest); merges
//!     into tenant.main_overlay; unlinks from chain.
//!   - Savepoint: merges into parent.overlay; pops the parent's
//!     open_child slot.
//!
//! Rollback:
//!   - Top-level Txn: drops self AND every successor in the chain.
//!     Matches raft's tail-rejection semantics; on leadership loss
//!     the integration calls rollback on the oldest pending txn and
//!     everything past it falls.
//!   - Savepoint: drops self and any nested sub-savepoint.
//!
//! Durabilize:
//!   - Snapshots pending createStore/dropStore, swaps each tenant's
//!     main_overlay into a local buffer (open txn overlays are NOT
//!     touched — they live in the chain), opens one LMDB write txn,
//!     applies all of it + the raft watermark in a single commit.
//!   - Open txns are unaffected; their writes will land in main_overlay
//!     when they commit, and be drained at the next durabilize.

const std = @import("std");
const lmdb = @import("lmdb.zig");
const overlay_mod = @import("overlay.zig");
const Overlay = overlay_mod.Overlay;
const OverlayEntry = overlay_mod.OverlayEntry;

pub const InitOptions = struct {
    /// Maximum number of stores (LMDB sub-DBIs). Plus 2 reserved for
    /// `_meta` and `_stores`. Fixed at env open.
    max_stores: u32 = 65534,
    /// LMDB mmap size in bytes. Sparse — only touched pages cost RAM.
    max_map_size: usize = 16 * 1024 * 1024 * 1024,
    no_meta_sync: bool = false,
    no_sync: bool = false,
};

pub const SpecificError = error{
    StoreAlreadyExists,
    StoreNotFound,
    ManifestPoisoned,
    InvalidSnapshotFormat,
    UnsupportedSnapshotVersion,
    /// Tried to commit a top-level Txn that wasn't the chain head.
    NotChainHead,
    /// Tried to commit/rollback a Txn that has an open savepoint
    /// child. Close the inner one first (LIFO).
    SavepointStillOpen,
};

const META_DBI_NAME: [:0]const u8 = "_meta";
const STORES_DBI_NAME: [:0]const u8 = "_stores";
const META_RAFT_APPLY_KEY: []const u8 = "raft_apply_idx";

fn storeDbiName(id: u64, buf: *[19]u8) [:0]const u8 {
    return std.fmt.bufPrintZ(buf, "s_{x:0>16}", .{id}) catch unreachable;
}

fn encodeStoreIdKey(id: u64, buf: *[8]u8) []const u8 {
    std.mem.writeInt(u64, buf, id, .big);
    return buf;
}

fn decodeStoreIdKey(bytes: []const u8) u64 {
    std.debug.assert(bytes.len == 8);
    return std.mem.readInt(u64, bytes[0..8], .big);
}

// ─── Manifest ───────────────────────────────────────────────────────

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    env: lmdb.Env,
    meta_dbi: lmdb.Dbi,
    stores_dbi: lmdb.Dbi,

    /// store_id → DBI handle for durably-registered tenants.
    dbis: std.AutoHashMapUnmanaged(u64, lmdb.Dbi) = .empty,
    pending_creates: std.AutoHashMapUnmanaged(u64, void) = .empty,
    pending_drops: std.AutoHashMapUnmanaged(u64, void) = .empty,
    dbis_lock: std.Thread.Mutex = .{},

    /// store_id → per-tenant in-memory state (overlay + chain of open
    /// Txns). Lazily created on first acquire / lease.beginTxn / write.
    tenants: std.AutoHashMapUnmanaged(u64, *TenantState) = .empty,
    tenants_lock: std.Thread.Mutex = .{},

    poisoned: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Single-caller serializer for durabilize / openSnapshot.
    durabilize_lock: std.Thread.Mutex = .{},

    pub fn init(
        self: *Manifest,
        allocator: std.mem.Allocator,
        path: [:0]const u8,
        options: InitOptions,
    ) !void {
        self.* = .{
            .allocator = allocator,
            .env = undefined,
            .meta_dbi = undefined,
            .stores_dbi = undefined,
        };
        self.env = try lmdb.Env.open(path, .{
            .max_dbs = options.max_stores + 2,
            .max_map_size = options.max_map_size,
            .no_meta_sync = options.no_meta_sync,
            .no_sync = options.no_sync,
            .no_lock = true,
        });
        errdefer self.env.close();

        // Bootstrap _meta and _stores.
        {
            var txn = try lmdb.Txn.beginWrite(&self.env);
            errdefer txn.abort();
            self.meta_dbi = try txn.openDbi(META_DBI_NAME, true);
            self.stores_dbi = try txn.openDbi(STORES_DBI_NAME, true);
            try txn.commit();
        }
        errdefer self.deinitMaps();

        // Populate `dbis` from `_stores`.
        {
            var txn = try lmdb.Txn.beginWrite(&self.env);
            errdefer txn.abort();
            self.stores_dbi = try txn.openDbi(STORES_DBI_NAME, false);
            self.meta_dbi = try txn.openDbi(META_DBI_NAME, false);

            var cur = try txn.openCursor(self.stores_dbi);
            defer cur.close();
            var pair = try cur.first();
            while (pair) |p| : (pair = try cur.next()) {
                const id = decodeStoreIdKey(p.key);
                var name_buf: [19]u8 = undefined;
                const name = storeDbiName(id, &name_buf);
                const dbi = try txn.openDbi(name, false);
                try self.dbis.put(self.allocator, id, dbi);
            }
            try txn.commit();
        }
    }

    fn deinitMaps(self: *Manifest) void {
        self.dbis.deinit(self.allocator);
        self.pending_creates.deinit(self.allocator);
        self.pending_drops.deinit(self.allocator);
        // Free every tenant's state including any open chain / overlays.
        var it = self.tenants.iterator();
        while (it.next()) |entry| {
            const ts = entry.value_ptr.*;
            // Free all txns in the chain (and their savepoint subtrees).
            var cur = ts.chain_head;
            while (cur) |c| {
                const next = c.chain_next;
                c.freeSubtreeLocked();
                cur = next;
            }
            ts.main_overlay.deinit();
            self.allocator.destroy(ts);
        }
        self.tenants.deinit(self.allocator);
    }

    pub fn deinit(self: *Manifest) void {
        self.deinitMaps();
        self.env.close();
        self.* = undefined;
    }

    // ── poison ──────────────────────────────────────────────────────

    pub fn isPoisoned(self: *const Manifest) bool {
        return self.poisoned.load(.monotonic);
    }

    fn poison(self: *Manifest) void {
        self.poisoned.store(true, .monotonic);
    }

    pub fn _testPoison(self: *Manifest) void {
        self.poison();
    }

    inline fn checkAlive(self: *const Manifest) !void {
        if (self.isPoisoned()) return error.ManifestPoisoned;
    }

    // ── stores (durable lifecycle, buffered) ────────────────────────

    pub fn createStore(self: *Manifest, id: u64) !void {
        try self.checkAlive();
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (self.hasStoreLocked(id)) return error.StoreAlreadyExists;
        try self.pending_creates.put(self.allocator, id, {});
    }

    pub fn dropStore(self: *Manifest, id: u64) !bool {
        try self.checkAlive();

        // Block on the per-tenant dispatch lock if a lease holder
        // is active. Drop happens with that lock held so that we can
        // tear the chain down safely; subsequent acquire(id) will
        // see hasStore == false and fail.
        //
        // Callers must not hold a lease on `id` from the same thread
        // (would self-deadlock). Per-thread caveat documented in the
        // README.
        const ts_opt = self.lookupTenantState(id);
        if (ts_opt) |ts| ts.dispatch_lock.lock();
        defer if (ts_opt) |ts| ts.dispatch_lock.unlock();

        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (!self.hasStoreLocked(id)) return false;

        // Cancel a pending_create that hasn't been durabilized yet.
        if (self.pending_creates.remove(id)) {
            try self.clearTenantStateLocked(id);
            // If the store is also durable (a drop-recreate-drop cycle
            // on the same id), the underlying durable DBI also needs
            // emptying at next durabilize.
            if (self.dbis.contains(id)) {
                try self.pending_drops.put(self.allocator, id, {});
            }
            return true;
        }

        // Durable store: queue the drop.
        try self.pending_drops.put(self.allocator, id, {});
        try self.clearTenantStateLocked(id);
        return true;
    }

    /// Called from dropStore. Walks the tenant's open chain (rolling
    /// back each Txn) and clears its main_overlay. Caller holds
    /// dbis_lock; we take tenants_lock + tenant_state lock briefly.
    fn clearTenantStateLocked(self: *Manifest, id: u64) !void {
        self.tenants_lock.lock();
        defer self.tenants_lock.unlock();
        const ts = self.tenants.get(id) orelse return;
        ts.lock.lock();
        defer ts.lock.unlock();

        // Drop every open Txn (and their savepoint subtrees). Callers
        // holding pointers to these Txns now hold dangling pointers
        // — they should have been the ones to roll back. This is the
        // explicit-drop-from-under-you case for dropStore.
        var cur = ts.chain_head;
        while (cur) |c| {
            const next = c.chain_next;
            c.freeSubtreeLocked();
            cur = next;
        }
        ts.chain_head = null;
        ts.chain_tail = null;
        ts.main_overlay.clear();
    }

    pub fn hasStore(self: *Manifest, id: u64) !bool {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        return self.hasStoreLocked(id);
    }

    fn hasStoreLocked(self: *const Manifest, id: u64) bool {
        if (self.pending_creates.contains(id)) return true;
        if (self.pending_drops.contains(id)) return false;
        return self.dbis.contains(id);
    }

    pub fn listStores(self: *Manifest, allocator: std.mem.Allocator) ![]u64 {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        var it = self.dbis.keyIterator();
        while (it.next()) |id_ptr| {
            if (self.pending_drops.contains(id_ptr.*)) continue;
            try list.append(allocator, id_ptr.*);
        }
        var pc_it = self.pending_creates.keyIterator();
        while (pc_it.next()) |id_ptr| try list.append(allocator, id_ptr.*);
        std.mem.sort(u64, list.items, {}, std.sort.asc(u64));
        return try list.toOwnedSlice(allocator);
    }

    fn lookupDbi(self: *Manifest, id: u64) ?lmdb.Dbi {
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        if (self.pending_drops.contains(id)) return null;
        return self.dbis.get(id);
    }

    // ── tenant state ────────────────────────────────────────────────

    fn getOrCreateTenantState(self: *Manifest, id: u64) !*TenantState {
        self.tenants_lock.lock();
        defer self.tenants_lock.unlock();
        const gop = try self.tenants.getOrPut(self.allocator, id);
        if (!gop.found_existing) {
            const ts = try self.allocator.create(TenantState);
            errdefer self.allocator.destroy(ts);
            ts.* = .{
                .main_overlay = Overlay.init(self.allocator),
            };
            gop.value_ptr.* = ts;
        }
        return gop.value_ptr.*;
    }

    fn lookupTenantState(self: *Manifest, id: u64) ?*TenantState {
        self.tenants_lock.lock();
        defer self.tenants_lock.unlock();
        return self.tenants.get(id);
    }

    // ── acquire / tryAcquire ────────────────────────────────────────

    /// Take exclusive dispatch ownership of `tenant_id`. Blocks until
    /// any current lease holder releases. The returned `StoreLease` is
    /// the only handle through which `beginTxn`, `get`, or `scanPrefix`
    /// can be issued for this tenant.
    ///
    /// Returns `error.StoreNotFound` if the tenant doesn't exist.
    ///
    /// The lease holder must call `release` to unlock. Other operations
    /// that touch the tenant (durabilize, openSnapshot, lockless cross-
    /// tenant reads) run concurrently — the lease only serializes
    /// dispatch.
    pub fn acquire(self: *Manifest, tenant_id: u64) !StoreLease {
        try self.checkAlive();
        if (!try self.hasStore(tenant_id)) return error.StoreNotFound;
        const ts = try self.getOrCreateTenantState(tenant_id);
        ts.dispatch_lock.lock();
        return .{
            .manifest = self,
            .tenant_id = tenant_id,
            .tenant_state = ts,
        };
    }

    /// Non-blocking variant of `acquire`. Returns null if another
    /// holder already owns the lease.
    pub fn tryAcquire(self: *Manifest, tenant_id: u64) !?StoreLease {
        try self.checkAlive();
        if (!try self.hasStore(tenant_id)) return error.StoreNotFound;
        const ts = try self.getOrCreateTenantState(tenant_id);
        if (!ts.dispatch_lock.tryLock()) return null;
        return StoreLease{
            .manifest = self,
            .tenant_id = tenant_id,
            .tenant_state = ts,
        };
    }

    // ── durabilize ──────────────────────────────────────────────────

    pub fn durabilize(self: *Manifest, raft_idx: u64) !void {
        try self.checkAlive();
        self.durabilize_lock.lock();
        defer self.durabilize_lock.unlock();
        errdefer self.poison();

        // 1. Snapshot pending creates/drops.
        var creates_to_apply: std.ArrayListUnmanaged(u64) = .empty;
        defer creates_to_apply.deinit(self.allocator);
        var drops_to_apply: std.ArrayListUnmanaged(struct { id: u64, dbi: lmdb.Dbi }) = .empty;
        defer drops_to_apply.deinit(self.allocator);
        {
            self.dbis_lock.lock();
            defer self.dbis_lock.unlock();
            var pc_it = self.pending_creates.keyIterator();
            while (pc_it.next()) |id_ptr| try creates_to_apply.append(self.allocator, id_ptr.*);
            var pd_it = self.pending_drops.keyIterator();
            while (pd_it.next()) |id_ptr| {
                const dbi = self.dbis.get(id_ptr.*) orelse continue;
                try drops_to_apply.append(self.allocator, .{ .id = id_ptr.*, .dbi = dbi });
            }
        }

        // 2. Enumerate every tenant.
        const TenantPair = struct { id: u64, ts: *TenantState };
        var tenant_pairs: std.ArrayListUnmanaged(TenantPair) = .empty;
        defer tenant_pairs.deinit(self.allocator);
        {
            self.tenants_lock.lock();
            defer self.tenants_lock.unlock();
            var it = self.tenants.iterator();
            while (it.next()) |entry| {
                try tenant_pairs.append(self.allocator, .{
                    .id = entry.key_ptr.*,
                    .ts = entry.value_ptr.*,
                });
            }
        }

        // 3. Swap each tenant's main_overlay into a local buffer.
        //    Workers committing top-level Txns during durabilize land
        //    in a fresh main_overlay that we leave behind; their data
        //    will be in the next durabilize.
        var swapped_list: std.ArrayListUnmanaged(SwappedEntry) = .empty;
        defer {
            for (swapped_list.items) |*s| {
                var it = s.entries.iterator();
                while (it.next()) |e| {
                    self.allocator.free(e.key_ptr.*);
                    switch (e.value_ptr.*) {
                        .value => |v| self.allocator.free(v),
                        .tombstone => {},
                    }
                }
                s.entries.deinit(self.allocator);
            }
            swapped_list.deinit(self.allocator);
        }
        for (tenant_pairs.items) |t| {
            t.ts.lock.lock();
            const empty = t.ts.main_overlay.entries.count() == 0;
            if (empty) {
                t.ts.lock.unlock();
                continue;
            }
            const taken = t.ts.main_overlay.entries;
            t.ts.main_overlay.entries = .empty;
            t.ts.lock.unlock();
            try swapped_list.append(self.allocator, .{
                .id = t.id,
                .entries = taken,
            });
        }

        // 4. Apply everything in one LMDB write txn.
        var txn = try lmdb.Txn.beginWrite(&self.env);
        errdefer txn.abort();

        for (drops_to_apply.items) |d| {
            try txn.dropDbi(d.dbi, false);
            var id_key_buf: [8]u8 = undefined;
            _ = try txn.del(self.stores_dbi, encodeStoreIdKey(d.id, &id_key_buf));
        }

        var new_dbis: std.ArrayListUnmanaged(NewDbi) = .empty;
        defer new_dbis.deinit(self.allocator);
        for (creates_to_apply.items) |id| {
            var name_buf: [19]u8 = undefined;
            const name = storeDbiName(id, &name_buf);
            const dbi = try txn.openDbi(name, true);
            var id_key_buf: [8]u8 = undefined;
            try txn.put(self.stores_dbi, encodeStoreIdKey(id, &id_key_buf), &.{});
            try new_dbis.append(self.allocator, .{ .id = id, .dbi = dbi });
        }

        for (swapped_list.items) |s| {
            const dbi = self.lookupDbiForApply(s.id, new_dbis.items) orelse continue;
            var it = s.entries.iterator();
            while (it.next()) |e| {
                switch (e.value_ptr.*) {
                    .value => |v| try txn.put(dbi, e.key_ptr.*, v),
                    .tombstone => _ = try txn.del(dbi, e.key_ptr.*),
                }
            }
        }

        if (raft_idx != 0) {
            var idx_buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &idx_buf, raft_idx, .little);
            try txn.put(self.meta_dbi, META_RAFT_APPLY_KEY, &idx_buf);
        }

        try txn.commit();

        // 5. Bookkeeping.
        {
            self.dbis_lock.lock();
            defer self.dbis_lock.unlock();
            for (drops_to_apply.items) |d| {
                _ = self.dbis.remove(d.id);
                _ = self.pending_drops.remove(d.id);
            }
            for (new_dbis.items) |n| {
                try self.dbis.put(self.allocator, n.id, n.dbi);
                _ = self.pending_creates.remove(n.id);
            }
        }
    }

    /// Convenience: durabilize without disturbing the raft watermark.
    /// Named `flush` (rather than `commit`) to avoid collision with
    /// `Txn.commit`, which means something completely different.
    pub fn flush(self: *Manifest) !void {
        return self.durabilize(0);
    }

    /// Read the durable raft watermark from LMDB. Returns 0 if no
    /// durabilize has ever written one.
    pub fn durableRaftIdx(self: *Manifest) !u64 {
        var txn = try lmdb.Txn.beginRead(&self.env);
        defer txn.abort();
        const bytes = (try txn.get(self.meta_dbi, META_RAFT_APPLY_KEY)) orelse return 0;
        if (bytes.len != 8) return 0;
        return std.mem.readInt(u64, bytes[0..8], .little);
    }

    const NewDbi = struct { id: u64, dbi: lmdb.Dbi };
    const SwappedEntry = struct {
        id: u64,
        entries: std.StringHashMapUnmanaged(OverlayEntry),
    };

    fn lookupDbiForApply(self: *Manifest, id: u64, new_dbis: []const NewDbi) ?lmdb.Dbi {
        for (new_dbis) |n| if (n.id == id) return n.dbi;
        self.dbis_lock.lock();
        defer self.dbis_lock.unlock();
        return self.dbis.get(id);
    }

    // ── snapshot ────────────────────────────────────────────────────

    pub fn openSnapshot(self: *Manifest) !Snapshot {
        // Serialize against durabilize so we get a coherent moment.
        self.durabilize_lock.lock();
        defer self.durabilize_lock.unlock();

        var read_txn = try lmdb.Txn.beginRead(&self.env);
        errdefer read_txn.abort();

        // Capture every tenant's main_overlay (open txn state is NOT
        // captured — snapshots see committed-but-not-durabilized
        // state, never speculation).
        var ov_capture: std.AutoHashMapUnmanaged(u64, []SnapshotEntry) = .empty;
        errdefer freeSnapshotOverlay(self.allocator, &ov_capture);

        const TenantPair = struct { id: u64, ts: *TenantState };
        var pairs: std.ArrayListUnmanaged(TenantPair) = .empty;
        defer pairs.deinit(self.allocator);
        {
            self.tenants_lock.lock();
            defer self.tenants_lock.unlock();
            var it = self.tenants.iterator();
            while (it.next()) |entry| {
                try pairs.append(self.allocator, .{
                    .id = entry.key_ptr.*,
                    .ts = entry.value_ptr.*,
                });
            }
        }
        for (pairs.items) |p| {
            var entries: std.ArrayListUnmanaged(SnapshotEntry) = .empty;
            errdefer {
                for (entries.items) |e| {
                    self.allocator.free(e.key);
                    if (e.value) |v| self.allocator.free(v);
                }
                entries.deinit(self.allocator);
            }
            {
                p.ts.lock.lock();
                defer p.ts.lock.unlock();
                if (p.ts.main_overlay.entries.count() == 0) continue;
                var oe_it = p.ts.main_overlay.entries.iterator();
                while (oe_it.next()) |oe| {
                    const k = try self.allocator.dupe(u8, oe.key_ptr.*);
                    errdefer self.allocator.free(k);
                    const v: ?[]u8 = switch (oe.value_ptr.*) {
                        .value => |val| try self.allocator.dupe(u8, val),
                        .tombstone => null,
                    };
                    try entries.append(self.allocator, .{ .key = k, .value = v });
                }
            }
            std.mem.sort(SnapshotEntry, entries.items, {}, SnapshotEntry.lessThan);
            const slice = try entries.toOwnedSlice(self.allocator);
            try ov_capture.put(self.allocator, p.id, slice);
        }

        return .{
            .manifest = self,
            .read_txn = read_txn,
            .overlays = ov_capture,
        };
    }

};

// ─── TenantState ────────────────────────────────────────────────────

const TenantState = struct {
    /// Internal serialization for main_overlay + chain mutations
    /// (taken briefly inside individual API calls).
    lock: std.Thread.Mutex = .{},
    /// Application-level dispatch lock: held by a StoreLease holder
    /// across many API calls. Top of the lock stack; never taken from
    /// inside kvexp's internal lock paths.
    dispatch_lock: std.Thread.Mutex = .{},
    main_overlay: Overlay,
    chain_head: ?*Txn = null,
    chain_tail: ?*Txn = null,
};

// ─── Txn ────────────────────────────────────────────────────────────

pub const Txn = struct {
    manifest: *Manifest,
    tenant_id: u64,
    tenant_state: *TenantState,
    /// null for top-level (chain entry); non-null for savepoint.
    parent: ?*Txn,
    /// Linked-list pointers for the per-tenant chain. Used only on
    /// top-level Txns (parent == null).
    chain_prev: ?*Txn,
    chain_next: ?*Txn,
    /// This Txn's own writes.
    overlay: Overlay,
    /// The currently-open savepoint child, if any. LIFO: at most one.
    open_child: ?*Txn,

    // ── writes ──────────────────────────────────────────────────────

    pub fn put(self: *Txn, key: []const u8, value: []const u8) !void {
        try self.manifest.checkAlive();
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        try self.overlay.putLocked(key, value);
    }

    pub fn delete(self: *Txn, key: []const u8) !bool {
        try self.manifest.checkAlive();
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        const existed = try self.keyExistsLocked(key);
        try self.overlay.tombstoneLocked(key);
        return existed;
    }

    // ── reads ───────────────────────────────────────────────────────

    pub fn get(self: *Txn, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        return self.getLocked(allocator, key);
    }

    fn getLocked(self: *Txn, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        // Walk savepoint stack: self → parent → parent.parent → ...
        // After top-level, walk chain backward.
        var cur: ?*Txn = self;
        while (cur) |t| {
            if (t.overlay.entries.get(key)) |entry| {
                switch (entry) {
                    .value => |v| return try allocator.dupe(u8, v),
                    .tombstone => return null,
                }
            }
            cur = if (t.parent) |p| p else t.chain_prev;
        }
        // main_overlay.
        if (self.tenant_state.main_overlay.entries.get(key)) |entry| {
            switch (entry) {
                .value => |v| return try allocator.dupe(u8, v),
                .tombstone => return null,
            }
        }
        // LMDB.
        const dbi = self.manifest.lookupDbi(self.tenant_id) orelse return null;
        var read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
        defer read_txn.abort();
        if (try read_txn.get(dbi, key)) |bytes| {
            return try allocator.dupe(u8, bytes);
        }
        return null;
    }

    /// Like getLocked but only returns whether the key resolves to a
    /// value (true) or null/tombstone (false). Used by delete() to
    /// compute the return bool without allocating.
    fn keyExistsLocked(self: *Txn, key: []const u8) !bool {
        var cur: ?*Txn = self;
        while (cur) |t| {
            if (t.overlay.entries.get(key)) |entry| {
                return switch (entry) {
                    .value => true,
                    .tombstone => false,
                };
            }
            cur = if (t.parent) |p| p else t.chain_prev;
        }
        if (self.tenant_state.main_overlay.entries.get(key)) |entry| {
            return switch (entry) {
                .value => true,
                .tombstone => false,
            };
        }
        const dbi = self.manifest.lookupDbi(self.tenant_id) orelse return false;
        var read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
        defer read_txn.abort();
        return (try read_txn.get(dbi, key)) != null;
    }

    pub fn scanPrefix(self: *Txn, prefix: []const u8) !TxnPrefixCursor {
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();

        // Flatten the layered view into a sorted slice, with closer-
        // to-self layers winning collisions. Tombstones are preserved
        // as null values — needed during the LMDB merge to shadow
        // entries that have been deleted in any overlay layer.
        var dedup: std.StringHashMapUnmanaged(?[]u8) = .empty;
        defer dedup.deinit(self.manifest.allocator);
        // Track key ownership separately so we can transfer to the
        // cursor's final slice without re-cloning.
        var ordered_keys: std.ArrayListUnmanaged([]u8) = .empty;
        defer ordered_keys.deinit(self.manifest.allocator);

        var cur: ?*Txn = self;
        while (cur) |t| {
            try absorbOverlayLocked(self.manifest.allocator, &t.overlay, prefix, &dedup, &ordered_keys);
            cur = if (t.parent) |p| p else t.chain_prev;
        }
        try absorbOverlayLocked(self.manifest.allocator, &self.tenant_state.main_overlay, prefix, &dedup, &ordered_keys);

        // Materialize sorted slice.
        var entries: std.ArrayListUnmanaged(TxnPrefixCursor.Entry) = .empty;
        errdefer {
            for (entries.items) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            entries.deinit(self.manifest.allocator);
        }
        try entries.ensureTotalCapacity(self.manifest.allocator, ordered_keys.items.len);
        for (ordered_keys.items) |k| {
            const v = dedup.get(k).?;
            entries.appendAssumeCapacity(.{ .key = k, .value = v });
        }
        std.mem.sort(TxnPrefixCursor.Entry, entries.items, {}, TxnPrefixCursor.Entry.lessThan);
        const overlay_slice = try entries.toOwnedSlice(self.manifest.allocator);
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        // LMDB cursor (if the tenant has a durable DBI).
        var read_txn: lmdb.Txn = .{ .ptr = null };
        var lmdb_cur: lmdb.Cursor = .{ .ptr = null };
        var has_lmdb = false;
        if (self.manifest.lookupDbi(self.tenant_id)) |dbi| {
            read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
            errdefer read_txn.abort();
            lmdb_cur = try read_txn.openCursor(dbi);
            has_lmdb = true;
        }
        errdefer if (has_lmdb) {
            lmdb_cur.close();
            read_txn.abort();
        };

        var cursor: TxnPrefixCursor = .{
            .allocator = self.manifest.allocator,
            .overlay = overlay_slice,
            .ov_idx = 0,
            .read_txn = read_txn,
            .lmdb_cur = lmdb_cur,
            .has_lmdb = has_lmdb,
            .prefix = try self.manifest.allocator.dupe(u8, prefix),
            .have_tree = false,
            .tree_key = &.{},
            .tree_val = &.{},
            .out_key = &.{},
            .out_val = &.{},
        };
        try cursor.primeTree();
        return cursor;
    }

    // ── savepoint / commit / rollback ──────────────────────────────

    /// Push a new savepoint onto this Txn (or onto an existing chain
    /// of savepoints if there's already one open and you want to nest
    /// — but you'd call savepoint on the deepest open one, not on
    /// this Txn). Returns the child handle.
    pub fn savepoint(self: *Txn) !*Txn {
        try self.manifest.checkAlive();
        self.tenant_state.lock.lock();
        defer self.tenant_state.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        const child = try self.manifest.allocator.create(Txn);
        errdefer self.manifest.allocator.destroy(child);
        child.* = .{
            .manifest = self.manifest,
            .tenant_id = self.tenant_id,
            .tenant_state = self.tenant_state,
            .parent = self,
            .chain_prev = null,
            .chain_next = null,
            .overlay = Overlay.init(self.manifest.allocator),
            .open_child = null,
        };
        self.open_child = child;
        return child;
    }

    pub fn commit(self: *Txn) !void {
        try self.manifest.checkAlive();
        // Capture tenant_state up front so the deferred unlock doesn't
        // dereference `self` after destroy(self) has freed it — under
        // multi-threaded contention the allocator can reuse that slot
        // before the defer runs.
        const ts = self.tenant_state;
        ts.lock.lock();
        defer ts.lock.unlock();
        if (self.open_child != null) return error.SavepointStillOpen;
        if (self.parent) |parent| {
            // Savepoint commit.
            try Overlay.moveInto(&self.overlay, &parent.overlay);
            parent.open_child = null;
        } else {
            // Top-level commit: must be chain head.
            if (ts.chain_head != self) return error.NotChainHead;
            try Overlay.moveInto(&self.overlay, &ts.main_overlay);
            if (self.chain_next) |next| {
                next.chain_prev = null;
                ts.chain_head = next;
            } else {
                ts.chain_head = null;
                ts.chain_tail = null;
            }
        }
        self.overlay.deinit();
        self.manifest.allocator.destroy(self);
    }

    pub fn rollback(self: *Txn) void {
        // Same hazard as commit: capture tenant_state before freeing self.
        const ts = self.tenant_state;
        ts.lock.lock();
        defer ts.lock.unlock();
        if (self.parent == null) {
            // Top-level: detach the tail of the chain at self, then
            // free self + all successors.
            if (self.chain_prev) |prev| {
                prev.chain_next = null;
                ts.chain_tail = prev;
            } else {
                ts.chain_head = null;
                ts.chain_tail = null;
            }
            var cur: ?*Txn = self;
            while (cur) |c| {
                const next = c.chain_next;
                c.freeSubtreeLocked();
                cur = next;
            }
        } else {
            // Savepoint: detach from parent + free subtree.
            self.parent.?.open_child = null;
            self.freeSubtreeLocked();
        }
    }

    /// Recursively free self + any open_child subtree. Caller holds
    /// tenant_state.lock. Does not touch the chain (caller has
    /// already detached this Txn from the chain if needed).
    fn freeSubtreeLocked(self: *Txn) void {
        if (self.open_child) |child| child.freeSubtreeLocked();
        self.overlay.deinit();
        self.manifest.allocator.destroy(self);
    }
};

/// Add prefix-matching entries from `ov` into `dedup`, transferring
/// ownership of newly-cloned key bytes into `ordered_keys`. Caller
/// owns the dedup map and the ordered_keys list; both are freed by
/// the caller. Closer-to-cursor layers should be called first; entries
/// already in `dedup` are skipped.
fn absorbOverlayLocked(
    allocator: std.mem.Allocator,
    ov: *Overlay,
    prefix: []const u8,
    dedup: *std.StringHashMapUnmanaged(?[]u8),
    ordered_keys: *std.ArrayListUnmanaged([]u8),
) !void {
    var it = ov.entries.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.startsWith(u8, k, prefix)) continue;
        const gop = try dedup.getOrPut(allocator, k);
        if (gop.found_existing) continue;
        const key_copy = try allocator.dupe(u8, k);
        errdefer allocator.free(key_copy);
        const val_copy: ?[]u8 = switch (e.value_ptr.*) {
            .value => |v| try allocator.dupe(u8, v),
            .tombstone => null,
        };
        gop.key_ptr.* = key_copy;
        gop.value_ptr.* = val_copy;
        try ordered_keys.append(allocator, key_copy);
    }
}

// ─── TxnPrefixCursor ────────────────────────────────────────────────

pub const TxnPrefixCursor = struct {
    allocator: std.mem.Allocator,
    read_txn: lmdb.Txn,
    lmdb_cur: lmdb.Cursor,
    has_lmdb: bool,
    prefix: []u8,
    overlay: []Entry,
    ov_idx: usize,
    have_tree: bool,
    tree_key: []const u8,
    tree_val: []const u8,
    out_key: []const u8,
    out_val: []const u8,

    pub const Entry = struct {
        key: []u8,
        value: ?[]u8,
        pub fn lessThan(_: void, a: Entry, b: Entry) bool {
            return std.mem.lessThan(u8, a.key, b.key);
        }
    };

    fn primeTree(self: *TxnPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = if (self.prefix.len == 0)
            try self.lmdb_cur.first()
        else
            try self.lmdb_cur.seekGe(self.prefix);
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    fn advanceTree(self: *TxnPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = try self.lmdb_cur.next();
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    pub fn next(self: *TxnPrefixCursor) !bool {
        while (true) {
            const have_ov = self.ov_idx < self.overlay.len;
            if (!self.have_tree and !have_ov) return false;
            if (self.have_tree and have_ov) {
                const ord = std.mem.order(u8, self.tree_key, self.overlay[self.ov_idx].key);
                switch (ord) {
                    .lt => {
                        self.out_key = self.tree_key;
                        self.out_val = self.tree_val;
                        try self.advanceTree();
                        return true;
                    },
                    .gt => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                    .eq => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        try self.advanceTree();
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                }
            } else if (self.have_tree) {
                self.out_key = self.tree_key;
                self.out_val = self.tree_val;
                try self.advanceTree();
                return true;
            } else {
                const e = self.overlay[self.ov_idx];
                self.ov_idx += 1;
                if (e.value) |v| {
                    self.out_key = e.key;
                    self.out_val = v;
                    return true;
                }
                continue;
            }
        }
    }

    pub fn key(self: *const TxnPrefixCursor) []const u8 {
        return self.out_key;
    }
    pub fn value(self: *const TxnPrefixCursor) []const u8 {
        return self.out_val;
    }

    pub fn deinit(self: *TxnPrefixCursor) void {
        if (self.has_lmdb) {
            self.lmdb_cur.close();
            self.read_txn.abort();
        }
        for (self.overlay) |e| {
            self.allocator.free(e.key);
            if (e.value) |v| self.allocator.free(v);
        }
        self.allocator.free(self.overlay);
        self.allocator.free(self.prefix);
        self.* = undefined;
    }
};

// ─── StoreLease ─────────────────────────────────────────────────────

/// Exclusive dispatch handle for a tenant. While held, no other thread
/// can begin Txns for this tenant. Reads through the lease see only
/// `main_overlay` + LMDB — never an in-flight chain (those are this
/// lease holder's own; read through the Txn handle for that).
///
/// Release is mandatory; the lease drops the dispatch_lock on
/// `release`. A Txn obtained via `beginTxn` outlives the lease — the
/// speculative-apply recipe releases the lease as soon as dispatch is
/// done, while the Txn stays in the chain until raft commits or
/// rejects.
pub const StoreLease = struct {
    manifest: *Manifest,
    tenant_id: u64,
    tenant_state: *TenantState,

    /// Open a new top-level Txn at the tail of this tenant's chain.
    /// Reads through the Txn see its own writes + older chain entries
    /// + the tenant's main_overlay + LMDB.
    pub fn beginTxn(self: *StoreLease) !*Txn {
        try self.manifest.checkAlive();
        const ts = self.tenant_state;
        ts.lock.lock();
        defer ts.lock.unlock();

        const txn = try self.manifest.allocator.create(Txn);
        errdefer self.manifest.allocator.destroy(txn);
        txn.* = .{
            .manifest = self.manifest,
            .tenant_id = self.tenant_id,
            .tenant_state = ts,
            .parent = null,
            .chain_prev = ts.chain_tail,
            .chain_next = null,
            .overlay = Overlay.init(self.manifest.allocator),
            .open_child = null,
        };
        if (ts.chain_tail) |tail| {
            tail.chain_next = txn;
        } else {
            ts.chain_head = txn;
        }
        ts.chain_tail = txn;
        return txn;
    }

    /// Point-read against committed state (main_overlay + LMDB). Does
    /// NOT see this lease's in-flight Txns.
    pub fn get(self: *StoreLease, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        const ts = self.tenant_state;
        ts.lock.lock();
        defer ts.lock.unlock();
        if (ts.main_overlay.entries.get(key)) |entry| {
            switch (entry) {
                .value => |v| return try allocator.dupe(u8, v),
                .tombstone => return null,
            }
        }
        const dbi = self.manifest.lookupDbi(self.tenant_id) orelse return null;
        var txn = try lmdb.Txn.beginRead(&self.manifest.env);
        defer txn.abort();
        if (try txn.get(dbi, key)) |bytes| return try allocator.dupe(u8, bytes);
        return null;
    }

    /// Prefix scan over committed state (main_overlay + LMDB).
    pub fn scanPrefix(self: *StoreLease, prefix: []const u8) !TxnPrefixCursor {
        var entries: std.ArrayListUnmanaged(TxnPrefixCursor.Entry) = .empty;
        errdefer {
            for (entries.items) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            entries.deinit(self.manifest.allocator);
        }
        {
            const ts = self.tenant_state;
            ts.lock.lock();
            defer ts.lock.unlock();
            var it = ts.main_overlay.entries.iterator();
            while (it.next()) |e| {
                const k = e.key_ptr.*;
                if (!std.mem.startsWith(u8, k, prefix)) continue;
                const k_copy = try self.manifest.allocator.dupe(u8, k);
                errdefer self.manifest.allocator.free(k_copy);
                const v_copy: ?[]u8 = switch (e.value_ptr.*) {
                    .value => |v| try self.manifest.allocator.dupe(u8, v),
                    .tombstone => null,
                };
                try entries.append(self.manifest.allocator, .{ .key = k_copy, .value = v_copy });
            }
        }
        std.mem.sort(TxnPrefixCursor.Entry, entries.items, {}, TxnPrefixCursor.Entry.lessThan);
        const overlay_slice = try entries.toOwnedSlice(self.manifest.allocator);
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        var read_txn: lmdb.Txn = .{ .ptr = null };
        var lmdb_cur: lmdb.Cursor = .{ .ptr = null };
        var has_lmdb = false;
        if (self.manifest.lookupDbi(self.tenant_id)) |dbi| {
            read_txn = try lmdb.Txn.beginRead(&self.manifest.env);
            errdefer read_txn.abort();
            lmdb_cur = try read_txn.openCursor(dbi);
            has_lmdb = true;
        }
        errdefer if (has_lmdb) {
            lmdb_cur.close();
            read_txn.abort();
        };

        var cur: TxnPrefixCursor = .{
            .allocator = self.manifest.allocator,
            .read_txn = read_txn,
            .lmdb_cur = lmdb_cur,
            .has_lmdb = has_lmdb,
            .prefix = try self.manifest.allocator.dupe(u8, prefix),
            .overlay = overlay_slice,
            .ov_idx = 0,
            .have_tree = false,
            .tree_key = &.{},
            .tree_val = &.{},
            .out_key = &.{},
            .out_val = &.{},
        };
        try cur.primeTree();
        return cur;
    }

    /// Drop the dispatch lock. Required. Any Txns this lease opened
    /// remain in the chain until their own commit / rollback.
    pub fn release(self: *StoreLease) void {
        self.tenant_state.dispatch_lock.unlock();
        self.* = undefined;
    }
};

// ─── Snapshot ───────────────────────────────────────────────────────

const SnapshotEntry = struct {
    key: []u8,
    value: ?[]u8,
    fn lessThan(_: void, a: SnapshotEntry, b: SnapshotEntry) bool {
        return std.mem.lessThan(u8, a.key, b.key);
    }
};

fn freeSnapshotOverlay(
    allocator: std.mem.Allocator,
    map: *std.AutoHashMapUnmanaged(u64, []SnapshotEntry),
) void {
    var it = map.iterator();
    while (it.next()) |entry| {
        for (entry.value_ptr.*) |kv| {
            allocator.free(kv.key);
            if (kv.value) |v| allocator.free(v);
        }
        allocator.free(entry.value_ptr.*);
    }
    map.deinit(allocator);
}

pub const Snapshot = struct {
    manifest: *Manifest,
    read_txn: lmdb.Txn,
    overlays: std.AutoHashMapUnmanaged(u64, []SnapshotEntry),

    pub fn close(self: *Snapshot) void {
        freeSnapshotOverlay(self.manifest.allocator, &self.overlays);
        self.read_txn.abort();
        self.* = undefined;
    }

    pub fn get(
        self: *Snapshot,
        allocator: std.mem.Allocator,
        store_id: u64,
        key: []const u8,
    ) !?[]u8 {
        if (self.overlays.get(store_id)) |entries| {
            if (binarySearchEntry(entries, key)) |idx| {
                const e = entries[idx];
                if (e.value) |v| return try allocator.dupe(u8, v);
                return null;
            }
        }
        const dbi = self.manifest.lookupDbi(store_id) orelse return null;
        if (try self.read_txn.get(dbi, key)) |bytes| return try allocator.dupe(u8, bytes);
        return null;
    }

    pub fn scanPrefix(self: *Snapshot, store_id: u64, prefix: []const u8) !SnapshotPrefixCursor {
        var entries: std.ArrayListUnmanaged(TxnPrefixCursor.Entry) = .empty;
        errdefer {
            for (entries.items) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            entries.deinit(self.manifest.allocator);
        }
        if (self.overlays.get(store_id)) |captured| {
            for (captured) |e| {
                if (!std.mem.startsWith(u8, e.key, prefix)) continue;
                const k_copy = try self.manifest.allocator.dupe(u8, e.key);
                errdefer self.manifest.allocator.free(k_copy);
                const v_copy: ?[]u8 = if (e.value) |v| try self.manifest.allocator.dupe(u8, v) else null;
                try entries.append(self.manifest.allocator, .{ .key = k_copy, .value = v_copy });
            }
        }
        const overlay_slice = try entries.toOwnedSlice(self.manifest.allocator);
        errdefer {
            for (overlay_slice) |e| {
                self.manifest.allocator.free(e.key);
                if (e.value) |v| self.manifest.allocator.free(v);
            }
            self.manifest.allocator.free(overlay_slice);
        }

        var lmdb_cur: lmdb.Cursor = .{ .ptr = null };
        var has_lmdb = false;
        if (self.manifest.lookupDbi(store_id)) |dbi| {
            lmdb_cur = try self.read_txn.openCursor(dbi);
            has_lmdb = true;
        }
        errdefer if (has_lmdb) lmdb_cur.close();

        var cur: SnapshotPrefixCursor = .{
            .allocator = self.manifest.allocator,
            .lmdb_cur = lmdb_cur,
            .has_lmdb = has_lmdb,
            .prefix = try self.manifest.allocator.dupe(u8, prefix),
            .overlay = overlay_slice,
            .ov_idx = 0,
            .have_tree = false,
            .tree_key = &.{},
            .tree_val = &.{},
            .out_key = &.{},
            .out_val = &.{},
        };
        try cur.primeTree();
        return cur;
    }

    pub fn listStores(self: *Snapshot, allocator: std.mem.Allocator) ![]u64 {
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(allocator);
        var cur = try self.read_txn.openCursor(self.manifest.stores_dbi);
        defer cur.close();
        var pair = try cur.first();
        while (pair) |p| : (pair = try cur.next()) {
            try list.append(allocator, decodeStoreIdKey(p.key));
        }
        return try list.toOwnedSlice(allocator);
    }
};

/// SnapshotPrefixCursor uses TxnPrefixCursor's merge logic; the only
/// difference is it doesn't own the read_txn (the Snapshot does).
pub const SnapshotPrefixCursor = struct {
    allocator: std.mem.Allocator,
    lmdb_cur: lmdb.Cursor,
    has_lmdb: bool,
    prefix: []u8,
    overlay: []TxnPrefixCursor.Entry,
    ov_idx: usize,
    have_tree: bool,
    tree_key: []const u8,
    tree_val: []const u8,
    out_key: []const u8,
    out_val: []const u8,

    fn primeTree(self: *SnapshotPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = if (self.prefix.len == 0)
            try self.lmdb_cur.first()
        else
            try self.lmdb_cur.seekGe(self.prefix);
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    fn advanceTree(self: *SnapshotPrefixCursor) !void {
        if (!self.has_lmdb) {
            self.have_tree = false;
            return;
        }
        const pair = try self.lmdb_cur.next();
        if (pair) |p| {
            if (std.mem.startsWith(u8, p.key, self.prefix)) {
                self.tree_key = p.key;
                self.tree_val = p.value;
                self.have_tree = true;
                return;
            }
        }
        self.have_tree = false;
    }

    pub fn next(self: *SnapshotPrefixCursor) !bool {
        while (true) {
            const have_ov = self.ov_idx < self.overlay.len;
            if (!self.have_tree and !have_ov) return false;
            if (self.have_tree and have_ov) {
                const ord = std.mem.order(u8, self.tree_key, self.overlay[self.ov_idx].key);
                switch (ord) {
                    .lt => {
                        self.out_key = self.tree_key;
                        self.out_val = self.tree_val;
                        try self.advanceTree();
                        return true;
                    },
                    .gt => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                    .eq => {
                        const e = self.overlay[self.ov_idx];
                        self.ov_idx += 1;
                        try self.advanceTree();
                        if (e.value) |v| {
                            self.out_key = e.key;
                            self.out_val = v;
                            return true;
                        }
                        continue;
                    },
                }
            } else if (self.have_tree) {
                self.out_key = self.tree_key;
                self.out_val = self.tree_val;
                try self.advanceTree();
                return true;
            } else {
                const e = self.overlay[self.ov_idx];
                self.ov_idx += 1;
                if (e.value) |v| {
                    self.out_key = e.key;
                    self.out_val = v;
                    return true;
                }
                continue;
            }
        }
    }

    pub fn key(self: *const SnapshotPrefixCursor) []const u8 {
        return self.out_key;
    }
    pub fn value(self: *const SnapshotPrefixCursor) []const u8 {
        return self.out_val;
    }

    pub fn deinit(self: *SnapshotPrefixCursor) void {
        if (self.has_lmdb) self.lmdb_cur.close();
        for (self.overlay) |e| {
            self.allocator.free(e.key);
            if (e.value) |v| self.allocator.free(v);
        }
        self.allocator.free(self.overlay);
        self.allocator.free(self.prefix);
        self.* = undefined;
    }
};

fn binarySearchEntry(entries: []const SnapshotEntry, key: []const u8) ?usize {
    var lo: usize = 0;
    var hi: usize = entries.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        switch (std.mem.order(u8, entries[mid].key, key)) {
            .lt => lo = mid + 1,
            .gt => hi = mid,
            .eq => return mid,
        }
    }
    return null;
}

// ─── State transfer dump/load ──────────────────────────────────────

pub const SNAPSHOT_MAGIC: u32 = 0x4B565853;
pub const SNAPSHOT_VERSION: u8 = 1;
pub const SNAP_TAG_KV: u8 = 1;
pub const SNAP_TAG_END: u8 = 2;

pub fn dumpSnapshot(snap: *Snapshot, writer: anytype) !void {
    try writer.writeInt(u32, SNAPSHOT_MAGIC, .little);
    try writer.writeByte(SNAPSHOT_VERSION);
    try writer.writeInt(u64, 0, .little); // reserved
    try writer.writeInt(u64, try snap.manifest.durableRaftIdx(), .little);

    const stores = try snap.listStores(snap.manifest.allocator);
    defer snap.manifest.allocator.free(stores);

    for (stores) |id| {
        var cur = try snap.scanPrefix(id, "");
        defer cur.deinit();
        while (try cur.next()) {
            const k = cur.key();
            const v = cur.value();
            try writer.writeByte(SNAP_TAG_KV);
            try writer.writeInt(u64, id, .little);
            try writer.writeInt(u16, @intCast(k.len), .little);
            try writer.writeInt(u16, @intCast(v.len), .little);
            try writer.writeAll(k);
            try writer.writeAll(v);
        }
    }
    try writer.writeByte(SNAP_TAG_END);
}

pub fn loadSnapshot(manifest: *Manifest, reader: anytype) !u64 {
    const magic = try reader.readInt(u32, .little);
    if (magic != SNAPSHOT_MAGIC) return error.InvalidSnapshotFormat;
    const version = try reader.readByte();
    if (version != SNAPSHOT_VERSION) return error.UnsupportedSnapshotVersion;
    _ = try reader.readInt(u64, .little);
    const last_applied = try reader.readInt(u64, .little);

    const KMAX = 256;
    const VMAX = 1 << 20;
    var key_buf: [KMAX]u8 = undefined;
    const val_buf = try manifest.allocator.alloc(u8, VMAX);
    defer manifest.allocator.free(val_buf);

    // One lease + one Txn per tenant. Leases are released at the end;
    // Txns commit at the end. Both maps share the same key set.
    var leases: std.AutoHashMapUnmanaged(u64, StoreLease) = .empty;
    defer {
        var lit = leases.iterator();
        while (lit.next()) |entry| entry.value_ptr.release();
        leases.deinit(manifest.allocator);
    }
    var txns: std.AutoHashMapUnmanaged(u64, *Txn) = .empty;
    defer txns.deinit(manifest.allocator);
    errdefer {
        var it = txns.iterator();
        while (it.next()) |entry| entry.value_ptr.*.rollback();
    }

    while (true) {
        const tag = try reader.readByte();
        switch (tag) {
            SNAP_TAG_END => break,
            SNAP_TAG_KV => {
                const id = try reader.readInt(u64, .little);
                const klen = try reader.readInt(u16, .little);
                const vlen = try reader.readInt(u16, .little);
                if (klen > key_buf.len or vlen > val_buf.len) return error.InvalidSnapshotFormat;
                try reader.readNoEof(key_buf[0..klen]);
                try reader.readNoEof(val_buf[0..vlen]);
                if (!(try manifest.hasStore(id))) try manifest.createStore(id);
                const lease_gop = try leases.getOrPut(manifest.allocator, id);
                if (!lease_gop.found_existing) {
                    lease_gop.value_ptr.* = try manifest.acquire(id);
                }
                const txn_gop = try txns.getOrPut(manifest.allocator, id);
                if (!txn_gop.found_existing) {
                    txn_gop.value_ptr.* = try lease_gop.value_ptr.beginTxn();
                }
                try txn_gop.value_ptr.*.put(key_buf[0..klen], val_buf[0..vlen]);
            },
            else => return error.InvalidSnapshotFormat,
        }
    }
    // Commit all txns. Each tenant's chain has exactly one entry (we
    // created it at first put for that tenant), so each commit is the
    // chain head. After this, the txns map holds dangling pointers
    // — clear it so the errdefer above doesn't double-free.
    var it = txns.iterator();
    while (it.next()) |entry| try entry.value_ptr.*.commit();
    txns.clearRetainingCapacity();
    return last_applied;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

const Harness = struct {
    tmp: std.testing.TmpDir,
    path_buf: [std.fs.max_path_bytes:0]u8,
    path: [:0]const u8,
    manifest: *Manifest,

    fn init() !Harness {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var self: Harness = .{
            .tmp = tmp,
            .path_buf = undefined,
            .path = undefined,
            .manifest = undefined,
        };
        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &dir_buf);
        const p = try std.fmt.bufPrintZ(&self.path_buf, "{s}/kvexp-test.mdb", .{dir_path});
        self.path = p;
        self.manifest = try testing.allocator.create(Manifest);
        errdefer testing.allocator.destroy(self.manifest);
        try self.manifest.init(testing.allocator, self.path, .{
            .max_map_size = 32 * 1024 * 1024,
            .max_stores = 256,
        });
        return self;
    }

    fn deinit(self: *Harness) void {
        self.manifest.deinit();
        testing.allocator.destroy(self.manifest);
        self.tmp.cleanup();
    }

    fn cycle(self: *Harness) !void {
        self.manifest.deinit();
        try self.manifest.init(testing.allocator, self.path, .{
            .max_map_size = 32 * 1024 * 1024,
            .max_stores = 256,
        });
    }

    /// Test helper: acquire the lease, begin a Txn, release the lease.
    /// Mirrors the speculative-apply pattern where the lease is held
    /// only for dispatch and the Txn outlives it. Caller is responsible
    /// for committing or rolling back the returned Txn.
    fn quickTxn(self: *Harness, tenant_id: u64) !*Txn {
        var lease = try self.manifest.acquire(tenant_id);
        defer lease.release();
        return try lease.beginTxn();
    }
};

test "Manifest: create + reopen survives" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(7);
    try h.manifest.flush();
    try h.cycle();
    try testing.expect(try h.manifest.hasStore(7));
}

test "Txn: simple put then commit lands in main_overlay, reads see it" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var txn = try h.quickTxn(1);
    try txn.put("k", "v");
    try txn.commit();

    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Txn: rollback drops writes; main_overlay untouched" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var txn = try h.quickTxn(1);
    try txn.put("k", "v");
    txn.rollback();

    var s = try h.manifest.acquire(1);
    defer s.release();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Txn: reads see its own writes before commit" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var txn = try h.quickTxn(1);
    defer txn.rollback();
    try txn.put("k", "v");
    const got = (try txn.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Txn: chain reads see earlier txns' writes" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    try t1.put("a", "1");
    var t2 = try h.quickTxn(1);
    // t2 should see t1's writes (chain reads backward).
    const got = (try t2.get(testing.allocator, "a")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("1", got);

    t2.rollback();
    t1.rollback();
}

test "Txn: top-level rollback cascades to successors" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    try t1.put("a", "1");
    var t2 = try h.quickTxn(1);
    try t2.put("b", "2");
    var t3 = try h.quickTxn(1);
    try t3.put("c", "3");

    // Rollback t2 → t2 and t3 are dropped.
    t2.rollback();

    // Chain head is t1; only t1 remains open.
    try testing.expect(h.manifest.tenants.get(1).?.chain_head == t1);
    try testing.expect(h.manifest.tenants.get(1).?.chain_tail == t1);

    try t1.commit();

    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("1", got);
    try testing.expect((try s.get(testing.allocator, "b")) == null);
    try testing.expect((try s.get(testing.allocator, "c")) == null);
}

test "Txn: commit must be chain head" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    var t2 = try h.quickTxn(1);
    try testing.expectError(error.NotChainHead, t2.commit());
    t2.rollback();
    try t1.commit();
}

test "Savepoint: commit merges into parent overlay" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    try t.put("a", "1");
    var sp = try t.savepoint();
    try sp.put("b", "2");
    try sp.commit();
    // After sp.commit(), b is in t.overlay.
    const got = (try t.get(testing.allocator, "b")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("2", got);
}

test "Savepoint: rollback drops only the savepoint's writes" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    try t.put("a", "1");
    var sp = try t.savepoint();
    try sp.put("b", "2");
    sp.rollback();
    // a still present, b gone.
    const ga = (try t.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
    try testing.expect((try t.get(testing.allocator, "b")) == null);
}

test "Savepoint: nested savepoints LIFO" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    var sp1 = try t.savepoint();
    try sp1.put("a", "1");
    var sp2 = try sp1.savepoint();
    try sp2.put("b", "2");
    try sp2.commit();      // b merges into sp1
    try sp1.commit();      // a, b merge into t

    const ga = (try t.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
    const gb = (try t.get(testing.allocator, "b")).?;
    defer testing.allocator.free(gb);
    try testing.expectEqualStrings("2", gb);
}

test "Savepoint: commit/rollback while inner savepoint open errors" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    var sp = try t.savepoint();
    try testing.expectError(error.SavepointStillOpen, t.commit());
    try testing.expectError(error.SavepointStillOpen, t.put("x", "y"));
    sp.rollback();
    try t.put("x", "y"); // works after closing the savepoint
}

test "durabilize: drains main_overlay; in-flight txn stays" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    var t1 = try h.quickTxn(1);
    try t1.put("durable", "yes");
    try t1.commit(); // → main_overlay

    var t2 = try h.quickTxn(1);
    try t2.put("inflight", "maybe");

    try h.manifest.flush();    // drains main_overlay; t2 untouched

    // After commit, t2 still sees its own write.
    const got = (try t2.get(testing.allocator, "inflight")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("maybe", got);

    // A fresh Store sees the durable value, not t2's in-flight one.
    var s = try h.manifest.acquire(1);
    defer s.release();
    const dg = (try s.get(testing.allocator, "durable")).?;
    defer testing.allocator.free(dg);
    try testing.expectEqualStrings("yes", dg);
    try testing.expect((try s.get(testing.allocator, "inflight")) == null);

    t2.rollback();
}

test "Txn.scanPrefix: merges chain + main + LMDB in order" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("a", "lmdb-a");
        try t.put("c", "lmdb-c");
        try t.commit();
    }
    try h.manifest.flush(); // a, c are in LMDB now

    var t1 = try h.quickTxn(1);
    defer t1.rollback();
    try t1.put("b", "t1-b");
    var t2 = try h.quickTxn(1);
    defer t2.rollback();
    try t2.put("d", "t2-d");
    try t2.put("a", "t2-a"); // override LMDB's a

    var cur = try t2.scanPrefix("");
    defer cur.deinit();
    var collected: [4]struct { k: []const u8, v: []const u8 } = undefined;
    var n: usize = 0;
    while (try cur.next()) : (n += 1) {
        collected[n] = .{ .k = cur.key(), .v = cur.value() };
    }
    try testing.expectEqual(@as(usize, 4), n);
    try testing.expectEqualStrings("a", collected[0].k);
    try testing.expectEqualStrings("t2-a", collected[0].v);
    try testing.expectEqualStrings("b", collected[1].k);
    try testing.expectEqualStrings("t1-b", collected[1].v);
    try testing.expectEqualStrings("c", collected[2].k);
    try testing.expectEqualStrings("lmdb-c", collected[2].v);
    try testing.expectEqualStrings("d", collected[3].k);
    try testing.expectEqualStrings("t2-d", collected[3].v);
}

test "Txn: delete tombstone shadows everything below" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "lmdb");
        try t.commit();
    }
    try h.manifest.flush();

    var t = try h.quickTxn(1);
    defer t.rollback();
    try testing.expect(try t.delete("k"));
    try testing.expect((try t.get(testing.allocator, "k")) == null);
}

test "durableRaftIdx: durabilize(idx) stamps; survives reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try testing.expectEqual(@as(u64, 0), try h.manifest.durableRaftIdx());
    try h.manifest.durabilize(42);
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());
    try h.cycle();
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());
    // commit() preserves the watermark.
    try h.manifest.flush();
    try testing.expectEqual(@as(u64, 42), try h.manifest.durableRaftIdx());
}

test "poison blocks writes; reopen clears" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    h.manifest._testPoison();
    try testing.expect(h.manifest.isPoisoned());
    try testing.expectError(error.ManifestPoisoned, h.manifest.durabilize(0));
    try testing.expectError(error.ManifestPoisoned, h.manifest.createStore(2));
    try testing.expectError(error.ManifestPoisoned, h.quickTxn(1));

    try h.cycle();
    try testing.expect(!h.manifest.isPoisoned());
    try testing.expect(try h.manifest.hasStore(1));
}

test "Snapshot: sees main_overlay + LMDB; not in-flight Txn" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("durable", "yes");
        try t.commit();
    }
    try h.manifest.flush();

    // Now a committed-but-not-durabilized write:
    {
        var t = try h.quickTxn(1);
        try t.put("commitnotdurable", "yes");
        try t.commit();
    }
    // And an open Txn:
    var inflight = try h.quickTxn(1);
    defer inflight.rollback();
    try inflight.put("inflight", "no");

    var snap = try h.manifest.openSnapshot();
    defer snap.close();

    const a = (try snap.get(testing.allocator, 1, "durable")).?;
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("yes", a);
    const b = (try snap.get(testing.allocator, 1, "commitnotdurable")).?;
    defer testing.allocator.free(b);
    try testing.expectEqualStrings("yes", b);
    try testing.expect((try snap.get(testing.allocator, 1, "inflight")) == null);
}

test "dumpSnapshot / loadSnapshot round-trip" {
    var src = try Harness.init();
    defer src.deinit();
    try src.manifest.createStore(10);
    {
        var t = try src.quickTxn(10);
        try t.put("a", "1");
        try t.put("b", "2");
        try t.commit();
    }
    try src.manifest.durabilize(99);

    var snap = try src.manifest.openSnapshot();
    defer snap.close();

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    var writer_state = buf.writer(testing.allocator);
    try dumpSnapshot(&snap, &writer_state);

    var dst = try Harness.init();
    defer dst.deinit();
    var reader_state = std.io.fixedBufferStream(buf.items);
    const last = try loadSnapshot(dst.manifest, reader_state.reader());
    try testing.expectEqual(@as(u64, 99), last);
    try dst.manifest.durabilize(last);
    try testing.expectEqual(@as(u64, 99), try dst.manifest.durableRaftIdx());

    var s = try dst.manifest.acquire(10);
    defer s.release();
    const ga = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
}

test "dropStore: while txns open, drops everything cleanly" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush();

    // Open a txn, then drop the store. The txn's memory is freed.
    _ = try h.quickTxn(1);
    // Don't keep a handle past dropStore — that's the "explicit
    // drop-from-under-you" caveat.
    _ = try h.manifest.dropStore(1);
    try h.manifest.flush();
    try testing.expect(!try h.manifest.hasStore(1));
}

test "Property: random workload + simulated crash recovers durable prefix" {
    const runs: u64 = 30;
    var seed: u64 = 1;
    while (seed <= runs) : (seed += 1) {
        propRunOne(seed) catch |err| {
            std.debug.print("\n*** property test failed at seed {} : {} ***\n", .{ seed, err });
            return err;
        };
    }
}

const PROP_NUM_STORES: usize = 4;
const PROP_NUM_KEYS: usize = 10;

const PropModel = struct {
    store_exists: [PROP_NUM_STORES]bool = @splat(false),
    key_present: [PROP_NUM_STORES][PROP_NUM_KEYS]bool = @splat(@splat(false)),
    values: [PROP_NUM_STORES][PROP_NUM_KEYS]u8 = @splat(@splat(0)),
};

fn keyBuf(buf: *[3]u8, i: usize) []const u8 {
    buf[0] = 'k';
    buf[1] = '0' + @as(u8, @intCast(i / 10));
    buf[2] = '0' + @as(u8, @intCast(i % 10));
    return buf[0..3];
}

fn verifyAgainstModel(manifest: *Manifest, model: *const PropModel) !void {
    for (0..PROP_NUM_STORES) |sid_usize| {
        const sid: u64 = @intCast(sid_usize);
        const exists = try manifest.hasStore(sid);
        if (exists != model.store_exists[sid_usize]) return error.StoreExistenceMismatch;
        if (!exists) continue;
        var s = try manifest.acquire(sid);
        defer s.release();
        for (0..PROP_NUM_KEYS) |kid| {
            var kb: [3]u8 = undefined;
            const k = keyBuf(&kb, kid);
            const got_opt = try s.get(testing.allocator, k);
            defer if (got_opt) |g| testing.allocator.free(g);
            const expected_present = model.key_present[sid_usize][kid];
            if (expected_present and got_opt == null) return error.KeyMissing;
            if (!expected_present and got_opt != null) return error.KeyUnexpected;
            if (expected_present) {
                if (got_opt.?.len != 1) return error.ValueLenMismatch;
                if (got_opt.?[0] != model.values[sid_usize][kid]) return error.ValueMismatch;
            }
        }
    }
}

fn propRunOne(seed: u64) !void {
    var rng_state = std.Random.DefaultPrng.init(seed);
    const rng = rng_state.random();
    var h = try Harness.init();
    defer h.deinit();

    var model: PropModel = .{};
    var durable_model: PropModel = .{};
    const op_count: u32 = rng.intRangeAtMost(u32, 30, 80);
    const crash_at: u32 = rng.intRangeLessThan(u32, 0, op_count);

    var i: u32 = 0;
    while (i < op_count) : (i += 1) {
        if (i == crash_at) {
            try h.cycle();
            try verifyAgainstModel(h.manifest, &durable_model);
            return;
        }
        const choice = rng.intRangeLessThan(u32, 0, 100);
        if (choice < 55) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const kid = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
            const v = rng.int(u8);
            var t = try h.quickTxn(@intCast(sid_usize));
            var kb: [3]u8 = undefined;
            var vb: [1]u8 = .{v};
            try t.put(keyBuf(&kb, kid), vb[0..1]);
            try t.commit();
            model.key_present[sid_usize][kid] = true;
            model.values[sid_usize][kid] = v;
        } else if (choice < 70) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const kid = rng.intRangeLessThan(usize, 0, PROP_NUM_KEYS);
            var t = try h.quickTxn(@intCast(sid_usize));
            var kb: [3]u8 = undefined;
            const existed = try t.delete(keyBuf(&kb, kid));
            try t.commit();
            if (existed != model.key_present[sid_usize][kid]) return error.DeleteMismatch;
            model.key_present[sid_usize][kid] = false;
        } else if (choice < 82) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (model.store_exists[sid_usize]) continue;
            try h.manifest.createStore(@intCast(sid_usize));
            model.store_exists[sid_usize] = true;
        } else if (choice < 87) {
            const sid_usize = rng.intRangeLessThan(usize, 0, PROP_NUM_STORES);
            if (!model.store_exists[sid_usize]) continue;
            const dropped = try h.manifest.dropStore(@intCast(sid_usize));
            if (!dropped) return error.DropReturnedFalse;
            model.store_exists[sid_usize] = false;
            for (0..PROP_NUM_KEYS) |k| model.key_present[sid_usize][k] = false;
        } else {
            try h.manifest.flush();
            durable_model = model;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Recovery contract: what survives a crash is exactly what the last
// successful durabilize wrote. h.cycle() simulates a crash by tearing down
// the in-memory manifest and reopening the same LMDB file.
// ───────────────────────────────────────────────────────────────────────────

test "Recovery: committed-but-not-flushed write is lost on crash" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.flush(); // store is durable; no keys yet

    {
        var t = try h.quickTxn(1);
        try t.put("k", "v"); // committed → main_overlay; NOT yet durable
        try t.commit();
    }

    try h.cycle();

    try testing.expect(try h.manifest.hasStore(1));
    var s = try h.manifest.acquire(1);
    defer s.release();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Recovery: in-flight Txn writes are lost on crash; earlier durable writes survive" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("durable", "yes");
        try t.commit();
    }
    try h.manifest.flush(); // "durable" is now in LMDB

    // Open a Txn that never commits or rolls back before the crash.
    var inflight = try h.quickTxn(1);
    try inflight.put("inflight", "no");

    try h.cycle(); // deinit tears down the open Txn; nothing reached LMDB

    var s = try h.manifest.acquire(1);
    defer s.release();
    const a = (try s.get(testing.allocator, "durable")).?;
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("yes", a);
    try testing.expect((try s.get(testing.allocator, "inflight")) == null);
}

test "Recovery: createStore without flush does not survive crash" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(42);
    try testing.expect(try h.manifest.hasStore(42)); // pending; visible in-memory

    try h.cycle();

    try testing.expect(!try h.manifest.hasStore(42));
}

test "Recovery: dropStore without flush does not survive crash; store comes back with data" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush(); // store + key durable

    _ = try h.manifest.dropStore(1); // pending drop, not flushed
    try testing.expect(!try h.manifest.hasStore(1));

    try h.cycle();

    try testing.expect(try h.manifest.hasStore(1));
    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Recovery: flushed dropStore removes the tenant on reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(7);
    {
        var t = try h.quickTxn(7);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush();
    _ = try h.manifest.dropStore(7);
    try h.manifest.flush(); // drop is now durable

    try h.cycle();

    try testing.expect(!try h.manifest.hasStore(7));
}

test "Recovery: flushed delete tombstone survives crash" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush(); // k=v durable

    {
        var t = try h.quickTxn(1);
        try testing.expect(try t.delete("k"));
        try t.commit();
    }
    try h.manifest.flush(); // tombstone durable

    try h.cycle();

    var s = try h.manifest.acquire(1);
    defer s.release();
    try testing.expect((try s.get(testing.allocator, "k")) == null);
}

test "Recovery: durabilize across multiple tenants is atomic on reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.createStore(2);
    try h.manifest.createStore(3);
    inline for (.{ .{ 1, "1-a" }, .{ 2, "2-a" }, .{ 3, "3-a" } }) |c| {
        var t = try h.quickTxn(c[0]);
        try t.put("a", c[1]);
        try t.commit();
    }
    try h.manifest.durabilize(100); // one atomic LMDB commit

    try h.cycle();

    try testing.expectEqual(@as(u64, 100), try h.manifest.durableRaftIdx());
    inline for (.{ .{ 1, "1-a" }, .{ 2, "2-a" }, .{ 3, "3-a" } }) |c| {
        try testing.expect(try h.manifest.hasStore(c[0]));
        var s = try h.manifest.acquire(c[0]);
        defer s.release();
        const got = (try s.get(testing.allocator, "a")).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(c[1], got);
    }
}

test "Recovery: successive durabilizes — last state wins after reopen" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    {
        var t = try h.quickTxn(1);
        try t.put("k", "v1");
        try t.commit();
    }
    try h.manifest.durabilize(10);

    {
        var t = try h.quickTxn(1);
        try t.put("k", "v2");
        try t.commit();
    }
    try h.manifest.durabilize(20);

    try h.cycle();

    try testing.expectEqual(@as(u64, 20), try h.manifest.durableRaftIdx());
    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v2", got);
}

test "Recovery: flush() persists data without touching the watermark" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);
    try h.manifest.durabilize(50); // watermark=50

    {
        var t = try h.quickTxn(1);
        try t.put("k", "v");
        try t.commit();
    }
    try h.manifest.flush(); // data durable; watermark unchanged

    try h.cycle();

    try testing.expectEqual(@as(u64, 50), try h.manifest.durableRaftIdx());
    var s = try h.manifest.acquire(1);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}

test "Recovery: only committed Txns durabilize; rolled-back siblings don't" {
    var h = try Harness.init();
    defer h.deinit();
    try h.manifest.createStore(1);

    var t1 = try h.quickTxn(1);
    try t1.put("a", "1");
    var t2 = try h.quickTxn(1);
    try t2.put("b", "2");
    t2.rollback(); // drops t2
    try t1.commit(); // t1 → main_overlay
    try h.manifest.flush();

    try h.cycle();

    var s = try h.manifest.acquire(1);
    defer s.release();
    const ga = (try s.get(testing.allocator, "a")).?;
    defer testing.allocator.free(ga);
    try testing.expectEqualStrings("1", ga);
    try testing.expect((try s.get(testing.allocator, "b")) == null);
}

test "Recovery: snapshot install becomes durable through durabilize + reopen" {
    // Build the source state and serialize a snapshot, then tear src down
    // entirely so dst doesn't share an open LMDB env with anything else.
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    {
        var src = try Harness.init();
        defer src.deinit();
        try src.manifest.createStore(5);
        {
            var t = try src.quickTxn(5);
            try t.put("k", "v");
            try t.commit();
        }
        try src.manifest.durabilize(77);

        var snap = try src.manifest.openSnapshot();
        defer snap.close();
        var w = buf.writer(testing.allocator);
        try dumpSnapshot(&snap, &w);
    }

    var dst = try Harness.init();
    defer dst.deinit();
    var stream = std.io.fixedBufferStream(buf.items);
    const last = try loadSnapshot(dst.manifest, stream.reader());
    try dst.manifest.durabilize(last);

    try dst.cycle();

    try testing.expectEqual(@as(u64, 77), try dst.manifest.durableRaftIdx());
    try testing.expect(try dst.manifest.hasStore(5));
    var s = try dst.manifest.acquire(5);
    defer s.release();
    const got = (try s.get(testing.allocator, "k")).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("v", got);
}
