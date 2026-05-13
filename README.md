# kvexp

Multi-tenant embedded key-value store in Zig, designed to sit under a
raft log. Many independent stores share one LMDB file; each tenant
has its own key space and atomic durability boundary. The raft log is
the write-ahead log; kvexp is a periodically-checkpointed materialization
of the applied prefix.

Writes go through **per-tenant chains of nested transactions**, gated
by a **per-tenant dispatch lease**. `Manifest.acquire(tenant_id)`
takes the lease; `lease.beginTxn()` opens a top-level Txn at the tail
of that tenant's chain; `lease.release()` drops the lease (the Txn
lives on in the chain until raft commits or rejects). `Txn.savepoint()`
pushes a LIFO save point. The model cleanly supports raft-driven
speculative writes: acquire, open a Txn, release; on raft commit, the
Txn commits and merges into `main_overlay`; on raft reject, rollback
cascades to every later open Txn for that tenant.

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
              │    ├─ dispatch_lock  ← held by current StoreLease│
              │    ├─ main_overlay   ← committed but not durable │
              │    └─ chain of open top-level Txns               │
              │           (raft propose order)                   │
              │           each Txn has:                          │
              │           ├─ overlay (this txn's writes)         │
              │           └─ open_child  ← single LIFO savepoint │
              └──────────────────────────────────────────────────┘

  acquire(tenant)         →  blocks on dispatch_lock; returns Lease
  tryAcquire(tenant)      →  null if held; else Lease
  Lease.beginTxn          →  new Txn at chain tail
  Lease.get / .scanPrefix →  main_overlay + LMDB (no chain reads)
  Lease.release           →  drops dispatch_lock; Txns survive

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
  dropStore(id)           →  blocks on dispatch_lock; safely tears
                             down chain before queueing pending_drop
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
sub-DBI inside one shared env. Workers acquire a per-tenant lease for
dispatch (one in-flight dispatcher per tenant), absorb writes into
per-Txn overlays (free of any I/O), and release. A periodic
`durabilize` drains every tenant's `main_overlay` plus the raft
watermark in one LMDB commit.

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

// Acquire the per-tenant lease; open a Txn under it.
var lease = try manifest.acquire(42);
defer lease.release();
var txn = try lease.beginTxn();
try txn.put("key", "value");
const v = try txn.get(allocator, "key");      // sees its own write
defer if (v) |b| allocator.free(b);
try txn.commit();                              // merge into main_overlay

// Reads through the lease see main_overlay + LMDB (no in-flight Txns).
const v2 = try lease.get(allocator, "key");

// Make everything durable, stamping the raft watermark atomically.
try manifest.durabilize(current_raft_idx);
// `flush()` is the alias for `durabilize(0)` — flushes without
// disturbing the watermark.
```

## API surface

```zig
// Manifest
pub fn init(self, allocator, path: [:0]const u8, options: InitOptions) !void
pub fn deinit(self) void

// Stores (buffered lifecycle). dropStore blocks on the dispatch_lock
// while any lease is held for that tenant.
pub fn createStore(self, id) !void
pub fn dropStore(self, id) !bool
pub fn hasStore(self, id) !bool
pub fn listStores(self, allocator) ![]u64

// Per-tenant dispatch lease — exclusive holder for writes + reads.
pub fn acquire(self, tenant_id) !StoreLease       // blocks
pub fn tryAcquire(self, tenant_id) !?StoreLease   // null if held

// Commit + recovery.
pub fn durabilize(self, raft_idx: u64) !void   // raft_idx=0 ↦ don't touch watermark
pub fn flush(self) !void                        // alias: durabilize(0)
pub fn durableRaftIdx(self) !u64
pub fn openSnapshot(self) !Snapshot

// Health + observability.
pub fn isPoisoned(self) bool
pub fn metricsSnapshot(self) MetricsSnapshot   // counters + gauges + histograms

// StoreLease (returned by acquire / tryAcquire).
pub fn beginTxn(self) !*Txn        // opens a Txn at chain tail
pub fn get(self, allocator, key) !?[]u8           // main_overlay + LMDB
pub fn scanPrefix(self, prefix) !TxnPrefixCursor  // main_overlay + LMDB
pub fn release(self) void          // drops dispatch_lock; Txns survive

// Txn (returned by lease.beginTxn or by .savepoint()).
pub fn put(self, key, value) !void
pub fn delete(self, key) !bool
pub fn get(self, allocator, key) !?[]u8
pub fn scanPrefix(self, prefix) !TxnPrefixCursor
pub fn savepoint(self) !*Txn       // pushes onto open_child slot
pub fn commit(self) !void          // top-level must be chain head;
                                   // savepoint merges into parent
pub fn rollback(self) void         // top-level cascades to successors;
                                   // savepoint drops self + nested

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

The recipes below walk a raft integration end-to-end:

1. **Leader speculative apply** — accept and start work optimistically.
2. **Leader switch** — discard speculation when leadership is lost.
3. **State transfer** — ship/receive a snapshot when log replay won't catch a follower up.
4. **Cold start** — open the manifest and tell raft where to resume.
5. **Follower apply** — apply already-committed entries without speculation.
6. **Graceful shutdown** — drain, checkpoint, close.

### 1. Leader speculative apply (optimistic per-tenant writes)

Acquire the per-tenant lease, open a Txn under it, dispatch, release.
The lease serializes dispatchers per tenant; different tenants run
fully in parallel. Don't wait for raft — release the lease and pick up
the next request. The Txn stays in the chain until raft commits or
rejects it.

```zig
const Pending = struct {
    txn: *kvexp.Txn,
    raft_idx: u64,
    request: *Request,
};

var pending: std.fifo.LinearFifo(Pending, .Dynamic) =
    std.fifo.LinearFifo(Pending, .Dynamic).init(allocator);
var latest_committed: u64 = 0;

/// Workers pull off a shared request queue; the lease enforces
/// per-tenant serialization at acquire time.
fn workerLoop(manifest: *kvexp.Manifest) !void {
    while (true) {
        const request = try requestQueue.pop();    // any tenant

        var lease = try manifest.acquire(request.tenant_id);
        defer lease.release();
        var txn = try lease.beginTxn();
        errdefer txn.rollback();
        try runHandler(txn, request);    // handler reads its own writes
        const raft_idx = try raft.propose(request.payload);
        try pending.writeItem(.{
            .txn = txn,
            .raft_idx = raft_idx,
            .request = request,
        });
        // lease released here; Txn lives on until onRaftCommit
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
- Each Txn's writes are **invisible to siblings** — when a worker
  picks up request K+1 and opens its own Txn, it sees K's writes
  (chain reads backward to K). It does NOT see its own later writes
  or any other tenant's Txns.
- `Txn.commit` is gated to chain head; with the lease in place this
  is impossible to hit through normal dispatch (the lease serializes
  acquires). `error.NotChainHead` is a defensive check for out-of-
  order callers.

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

### 4. Cold start / recovery

On process boot, open the manifest, ask LMDB for the durable raft
watermark, and tell raft to resume from there. Raft delivers every
committed entry past the watermark through your apply path (recipe 5).

```zig
fn nodeStartup(path: [:0]const u8) !*kvexp.Manifest {
    const manifest = try allocator.create(kvexp.Manifest);
    errdefer allocator.destroy(manifest);
    try manifest.init(allocator, path, .{
        .max_stores = 65534,
        .max_map_size = 16 * 1024 * 1024 * 1024,
    });
    errdefer manifest.deinit();

    // Raft resumes from the watermark. Everything up to and including
    // `resume_from` is materialized in LMDB; everything past replays
    // through `applyEntry` (recipe 5).
    const resume_from = try manifest.durableRaftIdx();
    try raft.openAt(resume_from);

    return manifest;
}
```

If `durableRaftIdx` is older than the leader's compaction floor, raft
will refuse to replay and instead ask for a snapshot — apply recipe 3.

### 5. Follower apply (non-leader path)

A follower never proposes; it just applies entries the leader committed.
No pending queue, no speculation, no rollback path — open a Txn per
entry, write the ops, commit.

```zig
/// Called by raft once an entry passes the commit threshold and is the
/// next to apply. Same hook runs on the new leader when it replays its
/// own log after recovery.
fn applyEntry(manifest: *kvexp.Manifest, entry: RaftEntry) !void {
    const ws = entry.decodeWriteset();

    // Tenant lifecycle ops flow through the manifest, not a Txn.
    switch (ws.kind) {
        .create_store => { try manifest.createStore(ws.tenant_id); },
        .drop_store   => { _ = try manifest.dropStore(ws.tenant_id); },
        .writeset     => {
            var lease = try manifest.acquire(ws.tenant_id);
            defer lease.release();
            var txn = try lease.beginTxn();
            errdefer txn.rollback();
            for (ws.ops) |op| switch (op.kind) {
                .put    => try txn.put(op.key, op.value),
                .delete => _ = try txn.delete(op.key),
            };
            try txn.commit();        // chain is length 1; head == tail
        },
    }

    latest_committed = entry.idx;
}

/// Same checkpoint cadence as the leader.
fn checkpoint(manifest: *kvexp.Manifest) !void {
    try manifest.durabilize(latest_committed);
}
```

When a follower wins an election, no kvexp-side handoff is needed: its
chain is empty (every prior entry was committed by `applyEntry`), so
new client requests can start using the leader path (recipe 1)
immediately.

### 6. Graceful shutdown

Drain workers, flush, close. The final `durabilize` lets raft compact
past `latest_committed`; skipping it just means more replay next boot.

```zig
fn shutdown(manifest: *kvexp.Manifest) !void {
    // Stop accepting new client work and let in-flight proposals
    // resolve (commit or reject) so the pending queue empties.
    try workers.drain();

    // Any Txn still in the chain at this point was never accepted by
    // raft and is not durable anywhere — let it die with the process.
    try manifest.durabilize(latest_committed);

    manifest.deinit();
    allocator.destroy(manifest);
}
```

## Concurrency model

| Lock | Purpose |
|---|---|
| `TenantState.dispatch_lock` (per-tenant) | application-level lease; held by the current `StoreLease` holder across many calls |
| `Manifest.dbis_lock` | durable lifecycle: dbis + pending_creates + pending_drops |
| `Manifest.tenants_lock` | tenants map (top-level lookup only; per-tenant work uses the tenant's own lock) |
| `TenantState.lock` (per-tenant) | this tenant's main_overlay + open Txn chain + savepoint substructure |
| `Manifest.durabilize_lock` | single-caller for durabilize + openSnapshot |

Lock order: **dispatch_lock → durabilize_lock → dbis_lock →
tenants_lock → TenantState.lock**. The dispatch lock sits at the top
because the application holds it across many kvexp calls; internal
locks are taken below it and released before returning.

Writers on different tenants share no lock. Per-tenant Txn operations
serialize through the lease (acquire) and then take TenantState.lock
briefly for the chain mutation. `durabilize` and `openSnapshot` grab
each TenantState lock briefly to swap / capture `main_overlay` —
they do not contend with lease holders.

Under the hood, LMDB is opened with its lock file + reader table
enabled (no `MDB_NOLOCK`). That's what lets `openSnapshot` keep a
long-lived read txn alive while `durabilize` runs concurrently —
the reader table tells LMDB which pages can't be recycled yet.
Cost: ~ns of overhead per read-txn open (overlay misses); writes
unaffected.

## Observability

`Manifest.metricsSnapshot()` returns a point-in-time `MetricsSnapshot`
with atomic counters and gauges plus two duration histograms (one
each for `durabilize` and `openSnapshot`). Counters cover the
per-API-op rate (put / delete / get, bytes_put), Txn lifecycle
(commit / rollback / savepoint variants), dispatch (acquire,
try-acquire contention), durability (durabilize success vs failed),
snapshot opens, and poison events. Gauges: active_leases,
active_snapshots.

The hottest counters (the per-op ones) are sharded across cache
lines so they don't false-share or true-share under contention —
threads map to shards via TLS-cached `gettid() & 63`. The snapshot
aggregates across shards transparently.

The histograms use Prometheus's default bucket boundaries (5 ms .. 10 s,
plus +Inf) and store cumulative counts in the snapshot, so callers
can map them straight to Prometheus's `_bucket{le="..."} / _sum /
_count` wire format. `Histogram.bucket_bounds_nanos` is exposed as a
`pub const` for the renderer.

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
- **Allocator OOM in internal bookkeeping.** `@panic`. Hard and loud.
  kvexp sits under a raft log: process recovery is the host's
  responsibility, and a partially-rolled-back manifest mid-OOM is
  more dangerous than a crash. The exceptions are allocations that
  use a *caller-supplied* allocator (`Lease.get`, `Snapshot.get`,
  `listStores`) — those still propagate `error.OutOfMemory`.
- **Drop while leased.** `dropStore` blocks on the per-tenant
  dispatch lock; if a worker holds a `StoreLease`, drop waits. Once
  drop returns, the chain has been torn down and subsequent `acquire`
  on that id returns `error.StoreNotFound`. Caveat: a single thread
  must not call `dropStore` on a tenant whose lease it currently holds
  (would self-deadlock — release the lease first).
- **Use-after-drop on stale Txn pointers.** If you obtained a Txn,
  released the lease, and another thread dropped the tenant, that
  Txn's memory is freed; using it is undefined. Either keep the lease
  while the Txn is open, or ensure drops happen only when no Txns are
  outstanding. The TenantState itself is refcounted — it stays alive
  as long as any lease holds it — so a *currently-held* lease against
  a freshly-dropped tenant won't UAF; writes via that lease will just
  vanish (the TenantState is no longer in the tenants map, so
  durabilize doesn't see it).
- **Multi-process access.** The env file is openable by other
  processes — LMDB's lock file (`*.mdb-lock` next to the data file)
  provides multi-process reader/writer coordination at the LMDB
  layer. But *kvexp's* invariants (per-tenant lease, in-memory chain,
  main_overlay) are in-process; a second writer process will corrupt
  state at the kvexp layer. External processes opening read-only is
  fine and intentional (useful for inspection).

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
  manifest.zig      Manifest, StoreLease, Txn, Snapshot, prefix
                    cursors, dumpSnapshot / loadSnapshot.
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
zig build bench     # runs the benchmark suite (ReleaseFast)
```
