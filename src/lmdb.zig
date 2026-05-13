//! Thin Zig wrapper over LMDB's C API.
//!
//! Scope: just the surface kvexp needs — Env / Txn / Dbi, put / get /
//! del, cursor. LMDB-isms (the int error codes, MDB_val struct shape,
//! MDB_dbi handle type) are wrapped in Zig-friendly types, but the
//! semantics are LMDB's: single writer per env, CoW B-tree with
//! atomic txn commit, mmap-backed reads.
//!
//! Reference: <http://www.lmdb.tech/doc/group__mdb.html>

const std = @import("std");

const c = @cImport({
    @cInclude("lmdb.h");
});

pub const Error = error{
    KeyExist,
    NotFound,
    PageNotFound,
    Corrupted,
    Panic,
    VersionMismatch,
    Invalid,
    MapFull,
    DbsFull,
    ReadersFull,
    TlsFull,
    TxnFull,
    CursorFull,
    PageFull,
    MapResized,
    Incompatible,
    BadRslot,
    BadTxn,
    BadValSize,
    BadDbi,
    LmdbError,
    OutOfMemory,
    NoSpaceOnDevice,
    PermissionDenied,
    Io,
};

/// Translate an LMDB C return code into a Zig error. 0 is success.
pub fn check(rc: c_int) Error!void {
    if (rc == 0) return;
    return switch (rc) {
        c.MDB_KEYEXIST => error.KeyExist,
        c.MDB_NOTFOUND => error.NotFound,
        c.MDB_PAGE_NOTFOUND => error.PageNotFound,
        c.MDB_CORRUPTED => error.Corrupted,
        c.MDB_PANIC => error.Panic,
        c.MDB_VERSION_MISMATCH => error.VersionMismatch,
        c.MDB_INVALID => error.Invalid,
        c.MDB_MAP_FULL => error.MapFull,
        c.MDB_DBS_FULL => error.DbsFull,
        c.MDB_READERS_FULL => error.ReadersFull,
        c.MDB_TLS_FULL => error.TlsFull,
        c.MDB_TXN_FULL => error.TxnFull,
        c.MDB_CURSOR_FULL => error.CursorFull,
        c.MDB_PAGE_FULL => error.PageFull,
        c.MDB_MAP_RESIZED => error.MapResized,
        c.MDB_INCOMPATIBLE => error.Incompatible,
        c.MDB_BAD_RSLOT => error.BadRslot,
        c.MDB_BAD_TXN => error.BadTxn,
        c.MDB_BAD_VALSIZE => error.BadValSize,
        c.MDB_BAD_DBI => error.BadDbi,
        @as(c_int, @intFromEnum(std.posix.E.NOMEM)) => error.OutOfMemory,
        @as(c_int, @intFromEnum(std.posix.E.NOSPC)) => error.NoSpaceOnDevice,
        @as(c_int, @intFromEnum(std.posix.E.ACCES)) => error.PermissionDenied,
        @as(c_int, @intFromEnum(std.posix.E.IO)) => error.Io,
        else => error.LmdbError,
    };
}

pub const Env = struct {
    ptr: ?*c.MDB_env,

    pub const OpenOptions = struct {
        max_dbs: c_uint = 64,
        /// Maximum mmap size, in bytes. LMDB requires this to fit the
        /// whole database; sparse so unused address space costs nothing.
        max_map_size: usize = 1024 * 1024 * 1024 * 16, // 16 GiB
        /// Maximum number of concurrent read txns.
        max_readers: c_uint = 126,
        /// `MDB_NOSUBDIR` lets `path` be a single file rather than a
        /// directory containing data.mdb + lock.mdb. Cleaner for our
        /// single-file model.
        no_subdir: bool = true,
        /// `MDB_NOMETASYNC`: skip the metadata page fsync on commit
        /// (kept on the data fsync). Trades a small durability window
        /// (last commit may roll back on crash) for ~30% commit speed.
        /// Default OFF — raft requires the watermark to be durable.
        no_meta_sync: bool = false,
        /// `MDB_NOSYNC`: skip ALL fsync. Don't enable unless you
        /// understand the risk; mostly useful in tests.
        no_sync: bool = false,
        /// `MDB_NOLOCK`: skip LMDB's lock file + reader table.
        ///
        /// Default OFF. Even though kvexp is single-process, we run
        /// many threads through the env: writers commit, readers (Txn
        /// chain misses, StoreLease.get, openSnapshot) open read txns
        /// concurrently. LMDB's reader table is what tells writers
        /// "this page is still being read, don't recycle it." Without
        /// it, a durabilize that lands while a snapshot is alive can
        /// recycle pages the snapshot's read txn is pointing at —
        /// undefined behavior per LMDB docs.
        no_lock: bool = false,
        /// `MDB_NOTLS`: tie reader-table slots to the MDB_txn object
        /// rather than to TLS. Required for us because the snapshot
        /// path holds a long-lived read txn while other code paths
        /// (dumpSnapshot calls durableRaftIdx; Txn.get does a chain
        /// miss; etc.) open additional read txns on the same thread.
        /// Without NOTLS, LMDB allows at most one read txn per thread.
        no_tls: bool = true,
        /// Mode for create.
        mode: c.mdb_mode_t = 0o600,
    };

    pub fn open(path: [:0]const u8, options: OpenOptions) Error!Env {
        var env: ?*c.MDB_env = null;
        try check(c.mdb_env_create(&env));
        errdefer c.mdb_env_close(env);
        try check(c.mdb_env_set_maxdbs(env, options.max_dbs));
        try check(c.mdb_env_set_mapsize(env, options.max_map_size));
        try check(c.mdb_env_set_maxreaders(env, options.max_readers));

        var flags: c_uint = 0;
        if (options.no_subdir) flags |= c.MDB_NOSUBDIR;
        if (options.no_meta_sync) flags |= c.MDB_NOMETASYNC;
        if (options.no_sync) flags |= c.MDB_NOSYNC;
        if (options.no_lock) flags |= c.MDB_NOLOCK;
        if (options.no_tls) flags |= c.MDB_NOTLS;

        try check(c.mdb_env_open(env, path.ptr, flags, options.mode));
        return .{ .ptr = env };
    }

    pub fn close(self: *Env) void {
        c.mdb_env_close(self.ptr);
        self.ptr = null;
    }

    /// Force a sync of the env's data + metadata to disk. Useful for
    /// callers that opened with MDB_NOMETASYNC and want to force a
    /// durability point.
    pub fn sync(self: *Env, force: bool) Error!void {
        try check(c.mdb_env_sync(self.ptr, if (force) 1 else 0));
    }
};

pub const Txn = struct {
    ptr: ?*c.MDB_txn,

    pub fn beginWrite(env: *Env) Error!Txn {
        var txn: ?*c.MDB_txn = null;
        try check(c.mdb_txn_begin(env.ptr, null, 0, &txn));
        return .{ .ptr = txn };
    }

    pub fn beginRead(env: *Env) Error!Txn {
        var txn: ?*c.MDB_txn = null;
        try check(c.mdb_txn_begin(env.ptr, null, c.MDB_RDONLY, &txn));
        return .{ .ptr = txn };
    }

    pub fn commit(self: *Txn) Error!void {
        try check(c.mdb_txn_commit(self.ptr));
        self.ptr = null;
    }

    pub fn abort(self: *Txn) void {
        if (self.ptr) |p| c.mdb_txn_abort(p);
        self.ptr = null;
    }

    /// Open or create a sub-DBI within this txn. The DBI handle is
    /// stable across txns once it's been opened in any txn that
    /// committed. `name` may be empty for the main DBI.
    pub fn openDbi(self: *Txn, name: ?[:0]const u8, create: bool) Error!Dbi {
        var dbi: c.MDB_dbi = undefined;
        const flags: c_uint = if (create) c.MDB_CREATE else 0;
        const name_ptr = if (name) |n| n.ptr else null;
        try check(c.mdb_dbi_open(self.ptr, name_ptr, flags, &dbi));
        return .{ .handle = dbi };
    }

    /// Drop a sub-DBI's contents. `delete_dbi=true` removes the DBI
    /// entirely; `delete_dbi=false` empties it but keeps the handle.
    pub fn dropDbi(self: *Txn, dbi: Dbi, delete_dbi: bool) Error!void {
        try check(c.mdb_drop(self.ptr, dbi.handle, if (delete_dbi) 1 else 0));
    }

    pub fn put(self: *Txn, dbi: Dbi, key: []const u8, value: []const u8) Error!void {
        var mk: c.MDB_val = .{ .mv_size = key.len, .mv_data = @constCast(@ptrCast(key.ptr)) };
        var mv: c.MDB_val = .{ .mv_size = value.len, .mv_data = @constCast(@ptrCast(value.ptr)) };
        try check(c.mdb_put(self.ptr, dbi.handle, &mk, &mv, 0));
    }

    /// Look up `key`. On hit, returns a slice that aliases the txn's
    /// mmap — valid until the next put/del on this dbi or until the
    /// txn ends. On miss, returns null (NotFound mapped to null).
    pub fn get(self: *Txn, dbi: Dbi, key: []const u8) Error!?[]const u8 {
        var mk: c.MDB_val = .{ .mv_size = key.len, .mv_data = @constCast(@ptrCast(key.ptr)) };
        var mv: c.MDB_val = undefined;
        const rc = c.mdb_get(self.ptr, dbi.handle, &mk, &mv);
        if (rc == c.MDB_NOTFOUND) return null;
        try check(rc);
        const bytes: [*]const u8 = @ptrCast(mv.mv_data);
        return bytes[0..mv.mv_size];
    }

    /// Delete `key`. Returns true if the key existed.
    pub fn del(self: *Txn, dbi: Dbi, key: []const u8) Error!bool {
        var mk: c.MDB_val = .{ .mv_size = key.len, .mv_data = @constCast(@ptrCast(key.ptr)) };
        const rc = c.mdb_del(self.ptr, dbi.handle, &mk, null);
        if (rc == c.MDB_NOTFOUND) return false;
        try check(rc);
        return true;
    }

    pub fn openCursor(self: *Txn, dbi: Dbi) Error!Cursor {
        var cur: ?*c.MDB_cursor = null;
        try check(c.mdb_cursor_open(self.ptr, dbi.handle, &cur));
        return .{ .ptr = cur };
    }
};

pub const Dbi = struct {
    handle: c.MDB_dbi,
};

pub const Cursor = struct {
    ptr: ?*c.MDB_cursor,

    pub fn close(self: *Cursor) void {
        if (self.ptr) |p| c.mdb_cursor_close(p);
        self.ptr = null;
    }

    pub const Pair = struct { key: []const u8, value: []const u8 };

    /// Position at the first key ≥ `key`. Returns the pair there, or
    /// null if no such key exists.
    pub fn seekGe(self: *Cursor, key: []const u8) Error!?Pair {
        var mk: c.MDB_val = .{ .mv_size = key.len, .mv_data = @constCast(@ptrCast(key.ptr)) };
        var mv: c.MDB_val = undefined;
        const rc = c.mdb_cursor_get(self.ptr, &mk, &mv, c.MDB_SET_RANGE);
        if (rc == c.MDB_NOTFOUND) return null;
        try check(rc);
        return Pair{
            .key = sliceFromVal(mk),
            .value = sliceFromVal(mv),
        };
    }

    /// Position at the first key in the DBI.
    pub fn first(self: *Cursor) Error!?Pair {
        var mk: c.MDB_val = undefined;
        var mv: c.MDB_val = undefined;
        const rc = c.mdb_cursor_get(self.ptr, &mk, &mv, c.MDB_FIRST);
        if (rc == c.MDB_NOTFOUND) return null;
        try check(rc);
        return Pair{ .key = sliceFromVal(mk), .value = sliceFromVal(mv) };
    }

    /// Advance to the next key. Null at end-of-data.
    pub fn next(self: *Cursor) Error!?Pair {
        var mk: c.MDB_val = undefined;
        var mv: c.MDB_val = undefined;
        const rc = c.mdb_cursor_get(self.ptr, &mk, &mv, c.MDB_NEXT);
        if (rc == c.MDB_NOTFOUND) return null;
        try check(rc);
        return Pair{ .key = sliceFromVal(mk), .value = sliceFromVal(mv) };
    }
};

inline fn sliceFromVal(v: c.MDB_val) []const u8 {
    const ptr: [*]const u8 = @ptrCast(v.mv_data);
    return ptr[0..v.mv_size];
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

const TestEnv = struct {
    tmp: std.testing.TmpDir,
    env: Env,
    path: [:0]const u8,
    path_buf: [std.fs.max_path_bytes:0]u8,

    fn init() !TestEnv {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var self: TestEnv = .{
            .tmp = tmp,
            .env = undefined,
            .path = undefined,
            .path_buf = undefined,
        };
        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try tmp.dir.realpath(".", &dir_buf);
        const printed = try std.fmt.bufPrintZ(&self.path_buf, "{s}/lmdb-test.mdb", .{dir_path});
        self.path = printed;
        self.env = try Env.open(self.path, .{ .max_map_size = 16 * 1024 * 1024 });
        return self;
    }

    fn deinit(self: *TestEnv) void {
        self.env.close();
        self.tmp.cleanup();
    }
};

test "lmdb: hello-world put/get round-trip" {
    var t = try TestEnv.init();
    defer t.deinit();

    {
        var txn = try Txn.beginWrite(&t.env);
        errdefer txn.abort();
        const dbi = try txn.openDbi(null, true);
        try txn.put(dbi, "k", "v");
        try txn.commit();
    }
    {
        var txn = try Txn.beginRead(&t.env);
        defer txn.abort();
        const dbi = try txn.openDbi(null, false);
        const got = (try txn.get(dbi, "k")) orelse return error.MissingKey;
        try testing.expectEqualStrings("v", got);
    }
}

test "lmdb: subdbi per tenant" {
    var t = try TestEnv.init();
    defer t.deinit();

    // First txn: create two sub-DBIs.
    {
        var txn = try Txn.beginWrite(&t.env);
        errdefer txn.abort();
        const tenant_a = try txn.openDbi("tenant_a", true);
        const tenant_b = try txn.openDbi("tenant_b", true);
        try txn.put(tenant_a, "k", "a-value");
        try txn.put(tenant_b, "k", "b-value");
        try txn.commit();
    }
    // Second txn: read back from a fresh-but-existing DBI.
    {
        var txn = try Txn.beginRead(&t.env);
        defer txn.abort();
        const tenant_a = try txn.openDbi("tenant_a", false);
        const tenant_b = try txn.openDbi("tenant_b", false);
        const va = (try txn.get(tenant_a, "k")) orelse return error.MissingKey;
        const vb = (try txn.get(tenant_b, "k")) orelse return error.MissingKey;
        try testing.expectEqualStrings("a-value", va);
        try testing.expectEqualStrings("b-value", vb);
    }
}

test "lmdb: cursor walks keys in sorted order" {
    var t = try TestEnv.init();
    defer t.deinit();

    {
        var txn = try Txn.beginWrite(&t.env);
        errdefer txn.abort();
        const dbi = try txn.openDbi(null, true);
        try txn.put(dbi, "b", "2");
        try txn.put(dbi, "a", "1");
        try txn.put(dbi, "c", "3");
        try txn.commit();
    }
    {
        var txn = try Txn.beginRead(&t.env);
        defer txn.abort();
        const dbi = try txn.openDbi(null, false);
        var cur = try txn.openCursor(dbi);
        defer cur.close();

        var collected: [3][1]u8 = undefined;
        var i: usize = 0;
        var pair = try cur.first();
        while (pair) |p| : (pair = try cur.next()) {
            collected[i] = .{p.key[0]};
            i += 1;
        }
        try testing.expectEqualSlices(u8, "a", &collected[0]);
        try testing.expectEqualSlices(u8, "b", &collected[1]);
        try testing.expectEqualSlices(u8, "c", &collected[2]);
    }
}

test "lmdb: delete returns false on missing key, true on present" {
    var t = try TestEnv.init();
    defer t.deinit();

    var txn = try Txn.beginWrite(&t.env);
    errdefer txn.abort();
    const dbi = try txn.openDbi(null, true);
    try txn.put(dbi, "k", "v");
    try testing.expect(try txn.del(dbi, "k"));
    try testing.expect(!try txn.del(dbi, "k"));
    try testing.expect(!try txn.del(dbi, "absent"));
    try txn.commit();
}

test "lmdb: read txn sees pre-commit state, post-commit needs new txn" {
    var t = try TestEnv.init();
    defer t.deinit();

    // Initial state.
    {
        var txn = try Txn.beginWrite(&t.env);
        errdefer txn.abort();
        const dbi = try txn.openDbi(null, true);
        try txn.put(dbi, "k", "v0");
        try txn.commit();
    }
    // Long-running read txn captures v0.
    var read_txn = try Txn.beginRead(&t.env);
    defer read_txn.abort();
    const read_dbi = try read_txn.openDbi(null, false);

    // Writer commits v1.
    {
        var txn = try Txn.beginWrite(&t.env);
        errdefer txn.abort();
        const dbi = try txn.openDbi(null, false);
        try txn.put(dbi, "k", "v1");
        try txn.commit();
    }
    // Read txn still sees v0.
    const got = (try read_txn.get(read_dbi, "k")) orelse return error.MissingKey;
    try testing.expectEqualStrings("v0", got);
}
