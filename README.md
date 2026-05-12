# kvexp

Multi-tenant embedded key-value store in Zig, designed to sit under a
raft log. Many independent stores share one LMDB file; each tenant
has its own key space and atomic durability boundary. The raft log is
the write-ahead log; kvexp is a periodically-checkpointed materialization
of the applied prefix.

Writes go through **per-tenant chains of nested transactions**.
`Manifest.beginTxn(tenant_id)` opens a top-level Txn at the tail of
that tenant's chain. `Txn.savepoint()` pushes a LIFO save point. The
model cleanly supports raft-driven speculative writes: open a Txn per
proposal, continue to the next proposal without waiting, commit when
raft accepts, rollback on rejection — the chain handles ordering and
cascade-rollback naturally.

## Architecture

```
              Manifest
              ┌──────────────────────────────────────────────────┐
              │  LMDB env (durable state)                        │
              │    ├─ "_meta"   raft_apply_idx watermark         │
              │    ├─ "_stores" directory of tenant ids          │
              │    └─ "s_<hex>" sub-DBI per tenant               │
              │                                                  │
              │  per-tenant TenantState (in memory, lazy)        │
              │    ├─ main_overlay   ← committed but not durable │
              │    └─ chain of open top-level Txns               │
              │           (raft propose order)                   │
              │           each Txn has:                          │
              │           ├─ overlay (this txn's writes)         │
              │           └─ open_child  ← single LIFO savepoint │
              └──────────────────────────────────────────────────┘

  beginTxn(tenant)        →  new Txn at chain tail
  Txn.put / .delete       →  into Txn.overlay (no I/O)
  Txn.get / .scanPrefix   →  walks savepoint stack → chain backward
                              → main_overlay → LMDB
  Txn.commit (top-level)  →  must be chain head; merges into main_overlay
  Txn.commit (savepoint)  →  merges into parent.overlay
  Txn.rollback (top)      →  discards self + every successor in chain
  Txn.rollback (sp)       →  discards self + nested savepoints

  durabilize(raft_idx)    →  one atomic LMDB write txn:
                              - apply pending createStores / dropStores
                              - drain every tenant's main_overlay
                              - write raft_idx into _meta
                             (open Txns are NOT touched — their writes
                              live in their own overlays, which only
                              hit main_overlay when the Txn commits)
  openSnapshot()          →  LMDB read txn + captured main_overlay
                             (in-flight Txns NOT visible)
  openStore(tenant_id)    →  read-only handle: main_overlay + LMDB
```

**Crucial invariants:**

- Every successful `durabilize(raft_idx)` is a single atomic LMDB
  commit covering tenant data + the watermark.
- `main_overlay` only ever receives **committed** writes (via `Txn.commit`).
  Speculative state lives in open Txns and never reaches LMDB except
  through commit → main_overlay → durabilize.
- A `Txn` reads its own writes plus older chain entries' writes plus
  the tenant's `main_overlay` plus LMDB — never another tenant's
  speculation and never its own siblings' speculation (because there
  are no siblings — chains are linear).

## Why this exists

The original `rove` project ran SQLite-per-tenant and hit three walls
past a few thousand tenants:

- **Per-tenant fds + shm + memory.** Even with an LRU of open handles
  these scale poorly at very-many-tenants.
- **shm-lock contention** for concurrent writers on the same file.
- **No fsync amortization** across tenants — each commits its own WAL.

kvexp gives every tenant its own ordered key space backed by an LMDB
sub-DBI inside one shared env. Workers absorb writes into per-Txn
overlays (free of any I/O). A periodic `durabilize` drains every
tenant's `main_overlay` plus the raft watermark in one LMDB commit.

## Quick start

```zig
const kvexp = @import("kvexp");

var manifest: kvexp.Manifest = undefined;
try manifest.init(allocator, "data.mdb", .{
    .max_stores = 65534,
    .max_map_size = 16 * 1024 * 1024 * 1024,  // 16 GiB sparse mmap
});
defer manifest.deinit();

// Tenant lifecycle (buffered until next durabilize).
try manifest.createStore(42);
const exists = try manifest.hasStore(42);

// Writes go through a transaction.
var txn = try manifest.beginTxn(42);
try txn.put("key", "value");
const v = try txn.get(allocator, "key");      // sees its own write
defer if (v) |b| allocator.free(b);
try txn.commit();                              // merge into main_overlay

// Read-only access (sees main_overlay + LMDB; no in-flight txns).
var store = try manifest.openStore(42);
defer store.deinit();
const v2 = try store.get(allocator, "key");

// Make everything durable, stamping the raft watermark atomically.
try manifest.durabilize(current_raft_idx);
// `commit()` is the alias for `durabilize(0)` — flushes without
// disturbing the watermark.
```

## API surface

```zig
// Manifest
pub fn init(self, allocator, path: [:0]const u8, options: InitOptions) !void
pub fn deinit(self) void

// Stores (buffered lifecycle).
pub fn createStore(self, id) !void
pub fn dropStore(self, id) !bool
pub fn hasStore(self, id) !bool
pub fn listStores(self, allocator) ![]u64

// Writes: go through a Txn. Reads-only: openStore.
pub fn beginTxn(self, tenant_id) !*Txn
pub fn openStore(self, tenant_id) !Store

// Commit + recovery.
pub fn durabilize(self, raft_idx: u64) !void   // raft_idx=0 ↦ don't touch watermark
pub fn commit(self) !void                       // alias: durabilize(0)
pub fn durableRaftIdx(self) !u64
pub fn openSnapshot(self) !Snapshot

// Health / admin.
pub fn isPoisoned(self) bool
pub fn verify(self, allocator) !VerifyReport

// Txn (returned by beginTxn or by .savepoint()).
pub fn put(self, key, value) !void
pub fn delete(self, key) !bool
pub fn get(self, allocator, key) !?[]u8
pub fn scanPrefix(self, prefix) !TxnPrefixCursor
pub fn savepoint(self) !*Txn       // pushes onto open_child slot
pub fn commit(self) !void          // top-level must be chain head;
                                   // savepoint merges into parent
pub fn rollback(self) void         // top-level cascades to successors;
                                   // savepoint drops self + nested

// Store (read-only).
pub fn get(self, allocator, key) !?[]u8
pub fn scanPrefix(self, prefix) !StorePrefixCursor

// Snapshot (point-in-time read txn + captured main_overlay).
pub fn close(self) void
pub fn get(self, allocator, store_id, key) !?[]u8
pub fn scanPrefix(self, store_id, prefix) !SnapshotPrefixCursor
pub fn listStores(self, allocator) ![]u64

// Free functions.
pub fn dumpSnapshot(snap, writer) !void
pub fn loadSnapshot(manifest, reader) !u64   // returns last_applied_raft_idx

// Errors callers commonly handle.
error.StoreAlreadyExists
error.StoreNotFound
error.ManifestPoisoned
error.NotChainHead            // commit a non-head top-level Txn
error.SavepointStillOpen      // commit/write a Txn with an open child
error.InvalidSnapshotFormat
error.UnsupportedSnapshotVersion
```

## Recipes

### 1. Speculative apply (optimistic per-tenant writes)

Open a Txn per request. Don't wait for raft — continue to the next
request, opening a new Txn at the tail of the chain. Responses gate
on raft commit; commits gate on raft commit; rollback drops on raft
reject. Workers on different tenants run completely independently.

```zig
const Pending = struct {
    txn: *kvexp.Txn,
    raft_idx: u64,
    request: *Request,
};

var pending: std.fifo.LinearFifo(Pending, .Dynamic) =
    std.fifo.LinearFifo(Pending, .Dynamic).init(allocator);
var latest_committed: u64 = 0;

/// Per-tenant worker loop. One worker per tenant at a time; different
/// tenants in parallel.
fn workerLoop(manifest: *kvexp.Manifest, tenant_id: u64) !void {
    while (true) {
        const request = try requestQueue(tenant_id).pop();
        var txn = try manifest.beginTxn(tenant_id);
        errdefer txn.rollback();
        try runHandler(txn, request);    // handler reads its own writes
        const raft_idx = try raft.propose(request.payload);
        try pending.writeItem(.{
            .txn = txn,
            .raft_idx = raft_idx,
            .request = request,
        });
    }
}

/// On every raft commit (raft thread).
fn onRaftCommit(committed_idx: u64) !void {
    latest_committed = committed_idx;
    while (pending.peekItem()) |head| {
        if (head.raft_idx > committed_idx) break;
        try head.txn.commit();           // merges into main_overlay
        try respond(head.request);
        _ = pending.readItem();
    }
}

/// Periodic checkpoint — every N commits, every T seconds, whichever
/// comes first.
fn checkpoint(manifest: *kvexp.Manifest) !void {
    try manifest.durabilize(latest_committed);
    // raft.compactLog(through: latest_committed) is now safe.
}
```

Inside the handler, **try/except blocks become savepoints**:

```zig
fn runHandler(txn: *kvexp.Txn, request: Request) !void {
    try txn.put("audit", "started");

    var sp = try txn.savepoint();
    riskyOp(sp) catch {
        sp.rollback();                   // undo whatever riskyOp wrote
        try txn.put("audit", "failed");
        return;
    };
    try sp.commit();                     // merge sp's writes into txn
    try txn.put("audit", "ok");
}
```

Key invariants:

- A response only releases after `onRaftCommit` for its `raft_idx`.
- Each Txn's writes are **invisible to siblings** — when worker pulls
  request K+1, that handler opens its own Txn, but it does see K's
  writes (chain reads backward to K). What it does NOT see is its own
  later writes or any other tenant's Txns.
- `Txn.commit` is gated to chain head; if a worker forgets to call
  `onRaftCommit` in order, commits return `error.NotChainHead`.

### 2. Leader switch (rollback after losing leadership)

Tail rejection in raft means every proposal from the failed leader
past some point is invalid. Roll back the oldest pending Txn — chain
rollback cascades to every successor.

```zig
fn onLeadershipLoss() void {
    // The pending queue is in raft-propose order. Rollback the first
    // (oldest) pending Txn; the chain rollback cascades to drop every
    // later open Txn for that tenant. Across tenants, do this per-
    // tenant since each tenant has its own independent chain.
    while (pending.readItem()) |item| {
        item.txn.rollback();             // cascades for top-level Txns
    }
}
```

That's it — no `discardOverlays` call, no raft replay. The next
proposals from the new leader will open fresh Txns; if they include
state that was previously speculative but raft-committed, the new
leader's raft log will include those entries and `onRaftCommit` will
process them in order.

If your durable watermark falls behind by more than the raft log
retains, do state transfer instead (recipe 3).

### 3. State transfer for catching-up followers

When a follower has fallen so far behind that the leader's raft log
has compacted past `durableRaftIdx`, log replay alone can't catch it
up. The leader ships a full snapshot file; the follower wipes and
reloads.

```zig
/// Leader: produce a snapshot file at dest_path.
fn produceSnapshot(manifest: *kvexp.Manifest, dest_path: []const u8) !void {
    // Drain main_overlay first so the snapshot reflects fully durable
    // state. (Open Txns are NOT in main_overlay yet, so they're not
    // shipped — correct semantics for state transfer.)
    try manifest.durabilize(latest_committed);

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
    // Drop every existing store so the receiver converges to the
    // sender's exact set.
    const existing = try manifest.listStores(allocator);
    defer allocator.free(existing);
    for (existing) |id| _ = try manifest.dropStore(id);

    var file = try std.fs.cwd().openFile(src_path, .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 1 << 30);
    defer allocator.free(bytes);
    var stream = std.io.fixedBufferStream(bytes);

    // loadSnapshot streams records, opening a Txn per tenant and
    // committing them at the end.
    const last_applied = try kvexp.loadSnapshot(manifest, stream.reader());

    // One durabilize commits everything — drops + loaded data +
    // watermark — in a single LMDB txn.
    try manifest.durabilize(last_applied);

    try raft.loadSnapshot(last_applied);
}
```

## Concurrency model

| Lock | Purpose |
|---|---|
| `Manifest.dbis_lock` | durable lifecycle: dbis + pending_creates + pending_drops |
| `Manifest.tenants_lock` | tenants map (top-level lookup only; per-tenant work uses the tenant's own lock) |
| `TenantState.lock` (per-tenant) | this tenant's main_overlay + open Txn chain + savepoint substructure |
| `Manifest.durabilize_lock` | single-caller for durabilize + openSnapshot |

Writers on different tenants share no lock. Per-tenant Txn operations
serialize through that tenant's lock. `durabilize` iterates tenants
and grabs each TenantState lock briefly to swap its main_overlay.

## Failure modes

- **Process crash mid-operation.** LMDB's commit is atomic; if it
  hadn't completed, LMDB reverts to the previous commit. Every Txn
  in memory is lost. Raft replays from `durableRaftIdx`.
- **fsync failure / I/O error during durabilize.** `errdefer self.poison()`
  fires; `isPoisoned()` returns true. Every subsequent mutating call
  (including `Txn.put`, `Txn.commit`, `createStore`, `durabilize`)
  returns `error.ManifestPoisoned`. Close + reopen to recover.
- **Disk full / mmap exhausted.** LMDB commits fail; `durabilize`
  poisons. Set `max_map_size` large up front (sparse mmap; only
  touched pages cost RAM).
- **Use-after-drop**: calling `Txn.put` (or anything) on a Txn whose
  tenant was just dropped is undefined. `dropStore` proactively frees
  open Txn memory for that tenant; the caller is expected to drop its
  handles to those Txns. Don't drop a tenant while another thread
  holds a Txn into it.

## Limitations

- **Linux + LMDB-only.** The Zig wrapper uses `@cImport("lmdb.h")`
  against the system liblmdb.
- **No cross-tenant atomicity.** Cross-tenant consistency is the
  application's responsibility (TCC in the rove model).
- **Snapshot files load into memory at install time.** Streaming is
  possible but not implemented; expected snapshot sizes (MB range)
  make this fine.
- **Empty stores don't round-trip in snapshots.** A store with zero
  entries emits no records; the receiver never learns it existed.
  Acceptable for raft state transfer where raft replay re-creates
  empty stores on the next createStore.

## Code layout

```
src/
  lmdb.zig          Thin Zig wrapper over the LMDB C API.
  overlay.zig       Per-Txn / per-tenant write buffer (memtable).
  manifest.zig      Manifest, Txn, Store, Snapshot, prefix cursors,
                    dumpSnapshot / loadSnapshot.
  root.zig          Public re-exports.
```

## Building

Requires the system `liblmdb` and its development headers:

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
