# kvexp

Multi-tenant embedded key-value store in Zig, designed to sit under a
raft log. Many independent stores share one LMDB file; each store has
its own key space and atomic durability boundary. The raft log is the
write-ahead log; kvexp is a periodically-checkpointed materialization
of the applied prefix.

Status: working over LMDB + in-memory memtable (29 tests pass). Earlier
revisions of kvexp implemented a from-scratch CoW B-tree; that's been
replaced by LMDB after profiling showed the per-write CoW overhead
was unrecoverable. The public API is unchanged save for `Manifest.init`.

## Why this exists

The original `rove` project ran SQLite-per-tenant and hit three walls
past a few thousand tenants:

- **Per-tenant fds + shm + memory.** Even with an LRU of open handles
  these scale poorly at very-many-tenants.
- **shm-lock contention** for concurrent writers on the same file.
- **No fsync amortization** across tenants — each commits its own WAL.

kvexp gives every tenant its own ordered key space backed by an LMDB
sub-DBI inside one shared env. Writes batch into a per-store in-memory
memtable; a periodic `durabilize()` drains all memtables and the
raft watermark into a single LMDB write transaction. Reads consult
the memtable first, then LMDB. Snapshot reads use an LMDB read txn
plus a captured copy of the memtable.

## Architecture in one paragraph

```
                Manifest
                ┌─────────────────────────────────────┐
                │  LMDB env                           │
                │  ├─ "_meta"   (last_applied_raft_idx)
                │  ├─ "_stores" (store-id directory)  │
                │  ├─ "s_<hex>" sub-DBI per tenant    │
                │  └─ ...                             │
                │                                     │
                │  per-tenant in-memory overlays      │
                │  ├─ tenant 1 → {k→v, k→tombstone}   │
                │  ├─ tenant 2 → {k→v, ...}           │
                │  └─ ...                             │
                └─────────────────────────────────────┘

  Store.put / Store.delete  →  overlay (in-memory; no I/O)
  Store.get / scanPrefix    →  overlay first, LMDB on miss
  durabilize()              →  one LMDB write txn:
                                 - apply pending createStores / dropStores
                                 - apply every overlay (puts, tombstones)
                                 - write last_applied_raft_idx
                               commit (atomic)
  openSnapshot()            →  LMDB read txn + captured overlay
  discardOverlays()         →  drop all in-flight memtables; LMDB untouched
```

The crucial property: **every successful `durabilize()` is a single
atomic LMDB commit covering memtable data + the raft watermark.**
Recovery on reopen reads the watermark from LMDB; the raft layer
replays log entries past it.

## Quick start

```zig
const kvexp = @import("kvexp");

var manifest: kvexp.Manifest = undefined;
try manifest.init(allocator, "data.mdb", .{
    .max_stores = 65534,
    .max_map_size = 16 * 1024 * 1024 * 1024,  // 16 GiB sparse mmap
});
defer manifest.deinit();

// Tenant lifecycle. Buffered until next durabilize.
try manifest.createStore(42);
const exists = try manifest.hasStore(42);

// Per-tenant ops.
var store = try kvexp.Store.open(&manifest, 42);
defer store.deinit();
try store.put("key", "value");
const v = try store.get(allocator, "key");
defer if (v) |b| allocator.free(b);
_ = try store.delete("key");

var cursor = try store.scanPrefix("user/");
defer cursor.deinit();
while (try cursor.next()) {
    // cursor.key() / cursor.value() — slices alias internal memory;
    // copy if you need to retain past the next next() call.
}

// Make everything durable.
try manifest.setLastAppliedRaftIdx(current_raft_idx);
try manifest.durabilize();
```

## API surface

```zig
// Manifest
pub fn init(self, allocator, path: [:0]const u8, options: InitOptions) !void
pub fn deinit(self) void
pub fn createStore(self, id) !void
pub fn dropStore(self, id) !bool
pub fn hasStore(self, id) !bool
pub fn listStores(self, allocator) ![]u64
pub fn lastAppliedRaftIdx(self) u64
pub fn setLastAppliedRaftIdx(self, idx) !void
pub fn durabilize(self) !void                 // alias: commit
pub fn discardOverlays(self) void
pub fn openSnapshot(self) !Snapshot
pub fn isPoisoned(self) bool                  // becomes true on durabilize failure
pub fn verify(self, allocator) !VerifyReport  // admin

// Store
pub fn open(manifest, id) !Store
pub fn get(self, allocator, key) !?[]u8
pub fn put(self, key, value) !void
pub fn delete(self, key) !bool
pub fn scanPrefix(self, prefix) !StorePrefixCursor

// Snapshot
pub fn close(self) void
pub fn get(self, allocator, store_id, key) !?[]u8
pub fn scanPrefix(self, store_id, prefix) !SnapshotPrefixCursor
pub fn listStores(self, allocator) ![]u64

// Free functions
pub fn dumpSnapshot(snap, writer) !void
pub fn loadSnapshot(manifest, reader) !u64    // returns last_applied_raft_idx

// Errors callers handle
error.StoreAlreadyExists
error.StoreNotFound
error.ManifestPoisoned
error.InvalidSnapshotFormat
error.UnsupportedSnapshotVersion
```

## Recipes

### 1. Speculative apply (optimistic per-tenant writes)

The classic raft-driven pattern: workers apply writes to the memtable
*before* raft accepts the proposal, then gate response release on the
raft commit. Cheap per-put cost (one hashmap insert) lets workers
keep pushing requests into the raft propose queue.

```zig
const PendingResponse = struct {
    request: *Request,
    raft_idx: u64,
};

var pending: std.fifo.LinearFifo(PendingResponse, .Dynamic) =
    std.fifo.LinearFifo(PendingResponse, .Dynamic).init(allocator);

/// Per-tenant worker loop. One worker per tenant at a time; multiple
/// tenants' workers run in parallel.
fn workerLoop(manifest: *kvexp.Manifest, tenant_id: u64) !void {
    while (true) {
        const request = try requestQueue(tenant_id).pop();

        // Optimistic apply: the JS / business handler calls store.put,
        // store.delete, store.get during its run. All puts/deletes
        // land in the per-tenant overlay — no I/O.
        var store = try kvexp.Store.open(manifest, tenant_id);
        defer store.deinit();
        try runHandler(&store, request);

        // Propose to raft. We DON'T wait for consensus here — the
        // worker continues to the next request immediately.
        const raft_idx = try raft.propose(request.payload);

        // Queue the response release for after raft commits this idx.
        try pending.writeItem(.{ .request = request, .raft_idx = raft_idx });
    }
}

/// On every raft commit (driven by the raft thread).
fn onRaftCommit(manifest: *kvexp.Manifest, committed_idx: u64) !void {
    // The data is already in the overlay (workers applied it
    // optimistically). Just advance the watermark.
    try manifest.setLastAppliedRaftIdx(committed_idx);

    // Release every pending response with raft_idx <= committed_idx.
    while (pending.peekItem()) |head| {
        if (head.raft_idx > committed_idx) break;
        _ = pending.readItem();
        try respond(head.request);
    }
}

/// Periodic checkpoint — every N commits, every T seconds, whichever
/// comes first.
fn checkpoint(manifest: *kvexp.Manifest) !void {
    try manifest.durabilize();
    // raft.compactLog(through: manifest.lastAppliedRaftIdx())
    //   is now safe — every entry up to that idx is durable in LMDB.
}
```

Key invariants:

- **A response is never released for an idx > `lastAppliedRaftIdx`.**
  The watermark advances only on `onRaftCommit`. So clients only see
  "your write succeeded" after raft consensus.
- **A request may be applied before consensus** (it's in the overlay
  from `workerLoop`), but `durabilize` only happens after `setLastAppliedRaftIdx`
  has advanced. If something kills the process between propose and
  commit, the overlay is lost and the raft layer re-proposes.
- **`durabilize` is exclusive with `openSnapshot`** (durabilize_lock
  serializes them) but otherwise concurrent with `workerLoop` and
  `onRaftCommit`. Workers writing during durabilize land in the fresh-
  empty overlay that the swap leaves behind; their writes go in the
  next durabilize round.

### 2. Leader switch (rollback after losing leadership)

When raft loses leadership, every uncommitted proposal from this
leader is invalidated. The overlay contains a mix of (a) confirmed
writes from raft-committed proposals and (b) speculative writes from
proposals that won't ever commit. Drop the whole overlay and replay
the confirmed prefix from the raft log:

```zig
fn onLeadershipLoss(manifest: *kvexp.Manifest) !void {
    // Drop every per-tenant overlay. LMDB is untouched — the state
    // up to lastAppliedRaftIdx is still durable on disk.
    manifest.discardOverlays();

    // Drop any pending responses; we'll never release them. The
    // requestors will retry against the new leader.
    pending.discard(pending.count);

    // Replay confirmed log entries past the durable watermark.
    // Each replay populates the overlay via store.put / store.delete.
    const from = manifest.lastAppliedRaftIdx() + 1;
    const to = raft.committedIndex();
    var idx = from;
    while (idx <= to) : (idx += 1) {
        const writeset = try raft.entryAt(idx);
        try applyWriteset(manifest, writeset);
        try manifest.setLastAppliedRaftIdx(idx);
    }

    // The overlay now reflects exactly the raft-committed-but-not-
    // yet-durabilized prefix. Next periodic checkpoint commits it.
}

/// Apply a raft-committed writeset to kvexp. Same code path as
/// optimistic apply, just driven by the raft thread instead of a
/// worker thread.
fn applyWriteset(manifest: *kvexp.Manifest, writeset: WriteSet) !void {
    var store = try kvexp.Store.open(manifest, writeset.store_id);
    defer store.deinit();
    for (writeset.ops) |op| switch (op.kind) {
        .put    => try store.put(op.key, op.value),
        .delete => _ = try store.delete(op.key),
    };
}
```

Key invariants:

- **Speculative-but-not-committed writes vanish atomically with
  `discardOverlays`.** They were never in the raft log; no consumer
  ever saw a "succeeded" response for them.
- **Confirmed-but-not-durable writes are recovered via raft replay.**
  Their raft entries are still in the log (haven't been compacted past
  `lastAppliedRaftIdx`).
- **No I/O happens during `discardOverlays`.** It's a pure in-memory
  reset, fast enough to run on the raft thread on every leadership
  transition.

### 3. State transfer for catching-up followers

When a follower has fallen so far behind that the leader's raft log
has compacted past its `lastAppliedRaftIdx`, log replay alone can't
catch it up. The leader ships a full snapshot file; the follower
wipes its local state and reloads from the file.

```zig
/// Leader: produce a snapshot file at dest_path.
fn produceSnapshot(manifest: *kvexp.Manifest, dest_path: []const u8) !void {
    // Make the overlay durable first so the snapshot reflects fully-
    // committed state. (Snapshots taken on a non-empty overlay still
    // work — openSnapshot captures the overlay too — but draining
    // first simplifies the transfer.)
    try manifest.durabilize();

    var snap = try manifest.openSnapshot();
    defer snap.close();

    var file = try std.fs.cwd().createFile(dest_path, .{ .truncate = true });
    defer file.close();
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);
    var w = buf.writer(allocator);
    try kvexp.dumpSnapshot(&snap, &w);

    try file.writeAll(buf.items);
    try file.sync();
}

/// Follower: install a snapshot file from src_path.
fn installSnapshot(manifest: *kvexp.Manifest, src_path: []const u8) !void {
    // Wipe every existing store so the receiver converges to the
    // sender's exact set. Buffered drops; the next durabilize emits
    // them in one txn.
    const existing = try manifest.listStores(allocator);
    defer allocator.free(existing);
    for (existing) |id| _ = try manifest.dropStore(id);

    var file = try std.fs.cwd().openFile(src_path, .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 1 << 30);
    defer allocator.free(bytes);
    var stream = std.io.fixedBufferStream(bytes);

    // loadSnapshot streams records, calling createStore + store.put.
    // Everything lands in overlays first.
    const last_applied_idx = try kvexp.loadSnapshot(manifest, stream.reader());

    // One durabilize commits the drops + all the loaded data + the
    // watermark in a single LMDB txn.
    try manifest.durabilize();

    // Tell raft: "my state covers everything up to this index;
    // resume applying from idx+1."
    try raft.loadSnapshot(last_applied_idx);
}
```

Key invariants:

- **Wire format is record-oriented**, not LMDB-page-level: a magic
  header followed by `{store_id, key_len, val_len, key, val}` records
  ending with a terminator. The sender doesn't need to know the
  receiver's LMDB version or page size.
- **Transport is the caller's concern.** kvexp produces a file; the
  application's HTTP / gRPC / whatever layer ships it. The raft
  message offering the snapshot is metadata only.
- **Snapshots are infrastructure long-reads.** They are NOT for the
  request path. Application requests serialize reads and writes
  inside the per-tenant worker; they have no use for MVCC.

## Concurrency model

| Lock | Purpose | When acquired |
|---|---|---|
| `Manifest.dbis_lock` | DBI handle map + pending create/drop sets | createStore, dropStore, hasStore, etc. — brief |
| `Manifest.overlays_lock` | overlays hashmap (top-level) | overlayFor / maybeOverlayFor — brief |
| `Overlay.lock` | per-store overlay contents | put / delete / get / scanPrefix snapshot — brief |
| `Manifest.store_locks_lock` | per-store mutex map | once per first put per store |
| `manifest.storeLock(id)` (per-store) | serialize writers on one tenant | Store.put / Store.delete |
| `Manifest.meta_lock` | last_applied_raft_idx | very brief |
| `Manifest.durabilize_lock` | single-caller for durabilize, openSnapshot | held for the entire operation |

Writers on **different tenants** never share a lock (per-store
locks + per-overlay locks). Writers on the same tenant serialize via
the per-store mutex.

`durabilize` holds `durabilize_lock` outermost; takes each overlay's
lock briefly to swap. Writers writing during a durabilize land in the
fresh post-swap overlay; no torn state visible.

## Failure modes

- **Process crash mid-operation.** LMDB's commit is atomic; if it
  hadn't completed, the LMDB env reverts to the previous commit. The
  in-memory overlay is lost. Raft replays from `lastAppliedRaftIdx`.
- **fsync failure / I/O error during durabilize.** `errdefer self.poison()`
  fires; the manifest's `isPoisoned()` returns true. Every subsequent
  mutation (`put`, `delete`, `createStore`, `dropStore`, `durabilize`)
  returns `error.ManifestPoisoned`. The caller must close + reopen the
  manifest to recover.
- **Disk full.** LMDB commits fail; `durabilize` poisons.
- **mmap exhausted.** LMDB's mmap size is fixed at open. If you blow
  through `max_map_size`, commits fail. Set it large (sparse mmap;
  costs nothing unless touched).

## Limitations

- **Linux + LMDB-only.** The Zig wrapper uses `@cImport("lmdb.h")`
  against the system liblmdb. Portability is technically possible
  but not validated.
- **No cross-store atomicity.** Each store is its own LMDB sub-DBI.
  Cross-store consistency is the application's responsibility (TCC
  in the rove model).
- **Snapshot files load into memory at install time.** Streaming
  `loadSnapshot` is possible but not implemented; expected snapshot
  sizes (MB range) make this fine in practice.
- **Empty stores don't round-trip in snapshots.** A store with zero
  entries emits no records; the receiver never learns it existed.
  Acceptable for raft state transfer where raft replay re-creates
  empty stores on the next createStore.

## Code layout

```
src/
  lmdb.zig          Thin Zig wrapper over the LMDB C API.
  overlay.zig       Per-store in-memory write buffer (memtable).
  manifest.zig      Manifest, Store, Snapshot, dumpSnapshot /
                    loadSnapshot, prefix cursors.
  root.zig          Public re-exports.
```

## Building

Requires the system `liblmdb` and its development headers (`lmdb.h`):

```
# Fedora / RHEL
sudo dnf install lmdb-devel

# Debian / Ubuntu
sudo apt install liblmdb-dev
```

Then:

```
zig build           # builds libkvexp.a + module
zig build test      # runs the test suite
```
