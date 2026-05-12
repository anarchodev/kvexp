# kvexp

Embedded multi-tenant key-value store in Zig. Many independent KV stores
share one disk file, one fsync, and one page cache — but each store has
its own B-tree, its own write lock, and its own atomic durability
boundary. Designed to sit under a raft log: the log is the WAL, kvexp is
a periodically-checkpointed materialization of the applied prefix.

Status: storage engine works end-to-end. 81 tests pass. Integration into
a raft-driven application (rove-loop46) is in progress on a separate
branch in that consumer's repo.

## Why this exists

Multi-tenant servers that store per-tenant state in SQLite-per-file run
into structural limits past a few thousand tenants:

- **Per-tenant fds + shm + memory.** SQLite needs `~2-3` fds per open
  database and a per-process `shm` mapping. At 10k tenants × 200KB cache
  = 2GB RSS just for the cache layer. The original "rove" project hit
  this wall.
- **shm-lock contention with concurrent writers on the same file.**
  Two threads writing the same database serialize through WAL writer
  acquisition. Past 8 concurrent writers per file the lock dominates.
- **Per-file fsync amortization is zero.** Each tenant write hits its
  own WAL with its own fsync. A burst across 100 tenants is 100 fsyncs.

kvexp keeps the SQLite-per-tenant API model (each tenant has its own
ordered key space, its own write lock, its own logical "database") but
collapses them into one file with one fsync cadence:

- 10k mostly-idle tenants × **1 file**, **6 fds**, **17 MB RSS**, **0
  background work** when nothing's writing.
- Writers on different tenants don't contend.
- One `durabilize()` flushes every dirty page across every tenant in
  one fsync. The amortization factor equals the number of tenants
  touched between checkpoints.

## How it works

### B-tree forest under a manifest tree

The file contains a forest of CoW B-trees. The **manifest tree** maps
`store_id: u64 → root_page_no: u64`. Each `root_page_no` is the root of
an independent per-tenant B-tree.

```
                    manifest_root
                  /       |       \
            (store_1)  (store_2) (store_3) ...
            tree root  tree root  tree root
              |          |          |
             ...        ...        ...
```

Per-tenant isolation is structural: a key under `store_1` can never
collide with `store_2` because they're literally different trees.

### Slot pages + atomic snapshot

Page 1 and Page 2 are reserved as **manifest slot pages**: each holds
a `(sequence, manifest_root, freelist_root, last_applied_raft_idx)`
record plus a CRC. At any time exactly one slot is *active* (highest
valid sequence); the other holds the previous snapshot.

`durabilize()` is a two-phase commit:
1. Flush all dirty data pages to disk + fsync.
2. Write the new metadata into the *inactive* slot + fsync, then it
   becomes active.

A crash anywhere except step 2's slot-write atomicity leaves the old
snapshot intact and complete on disk; recovery reads the active slot
and proceeds. No WAL replay, no torn-write fixup — the slot swap is
the entire durability protocol.

### CoW with same-txn in-place mutation (hybrid)

A pure CoW B-tree allocates a new page on every write and propagates
new pointers all the way up to the root. Page count per put = depth
of the tree (~3-4 for typical workloads), plus the manifest tree update
(another ~2-3).

kvexp short-circuits this when the same apply unit hits the same path
twice. Each cached page carries a `dirty_seq`; if it equals
`self.tree.seq` (the current apply seq), the page was already CoW'd in
this txn and the in-memory buffer *is* the txn's scratch space — we
mutate it directly instead of allocating a new page. The parent walk
also short-circuits: if the child's page_no didn't change, the parent
doesn't need to be CoW'd either.

The net effect: the first write to a leaf in an apply unit pays full
CoW cost; subsequent writes to the same leaf are O(N_in_leaf) byte
copies with zero page allocations.

### In-memory store_root cache

`setStoreRoot(id, root)` happens after every per-store write, since
each leaf CoW changes the store's root pointer. If this went straight
to the manifest tree on every put, the manifest tree would be the
single global contention point.

Instead, `setStoreRoot` updates an in-memory `HashMap<u64, RootSlot>`
under a dedicated mutex. The manifest tree is only touched in step 0
of `durabilize()`, which flushes every dirty cache entry into the tree
before the regular page flush. Hot-path setters become an O(1)
hashmap write with no tree_lock contention.

### In-process page cache

kvexp owns its own page cache — sharded by `page_no` with clock
eviction. Eviction writeback releases the shard lock before issuing
the I/O so writers on other shards aren't blocked. `pinNewUninit`
skips the 4KB zero-fill on new allocations (the caller overwrites the
page header + cells immediately).

mmap is deliberately avoided: kvexp needs to control writeback
ordering to make orphan elision correct (intermediate page versions
within a commit window can be skipped at flush time because the next
durable manifest never references them).

## Consistency model

### Per-store ACID, no cross-store atomicity

Each `Store` is a separate B-tree behind a per-store write lock. A
single put-or-delete is atomic. Multi-op transactions within a store
are atomic across `durabilize()` boundaries — see "deferred durabilize"
below for the contract.

Cross-store atomicity (write to store A and store B in one logical
transaction) is **not** provided by kvexp. The intended pattern is
TCC over Percolator-style timestamps at the application layer; kvexp
just provides the per-store atomicity and the snapshot boundary.

### Apply vs durabilize

The API distinguishes two seq counters:

- **`applySeq`** — the seq stamped on dirty pages by in-progress
  writes. Advances via `nextApply()` at logical apply-unit boundaries
  (e.g. once per raft writeset).
- **`durableSeq`** — the active slot's sequence. Advances only when
  `durabilize()` completes the slot swap.

Between `durabilize()` calls, all mutations are in-memory only.
Crashes between calls lose the in-memory state but the previous slot's
state survives. After `durabilize()`, all writes with `apply_seq <=
durable_seq` are guaranteed on disk.

### Snapshots are point-in-time

`openSnapshot()` returns a `Snapshot` handle that captures the active
`manifest_root` and registers a refcount entry. The freelist's
"reusable" promotion logic respects the minimum live snap_seq — pages
freed at or after a live snapshot's seq stay reserved until the
snapshot closes. So a snapshot is guaranteed to read a consistent
view at its open time, even as concurrent writes churn the tree
underneath it.

Snapshots are designed for infrastructure long-reads (state transfer,
backup, debug dump). Application-level MVCC is not the intent — apply
units batch many app reads behind a single durabilize, so the
canonical "latest applied state" is usually what callers want.

### Cold-start replay

On `Manifest.init`, the active slot's `last_applied_raft_idx` is loaded
into memory. The intended pattern: the raft library reads this on
startup, then replays log entries past it. Already-durabilized writes
get filtered (idempotent re-application via raft's apply filter).
Writes the raft log has but kvexp's last snapshot didn't include get
replayed and become durable on the next checkpoint.

## Raft integration contract

kvexp is designed to be used with the raft log as the WAL. The
canonical integration looks like this:

### The deferred-durabilize contract

The apply path stages writes into the in-memory state via `Store.put`
/ `Store.delete`. **Do not call `durabilize()` per writeset.**
Durabilize is the *checkpoint* boundary — typically aligned with raft
log compaction.

```zig
// On each raft commit (every node, including the leader after consensus):
fn applyWriteSet(manifest: *kvexp.Manifest, writeset: WriteSet) !void {
    _ = manifest.nextApply();
    var store = try kvexp.Store.open(manifest, writeset.store_id);
    defer store.deinit();
    for (writeset.ops) |op| {
        switch (op.kind) {
            .put    => try store.put(op.key, op.value),
            .delete => _ = try store.delete(op.key),
        }
    }
    // No durabilize here — see checkpoint trigger below.
}

// On raft snapshot (rare — every N commits or T seconds):
fn checkpoint(manifest: *kvexp.Manifest, apply_idx: u64) !void {
    manifest.setLastAppliedRaftIdx(apply_idx);
    try manifest.durabilize();
    // raft.compactLog() can now trim entries up to apply_idx.
}
```

### Leader-side optimistic writes

Leaders often want to stage a write *before* raft accepts it — so
local handlers can read their own pending mutations. kvexp supports
this via a captured store-root revert:

```zig
// Leader path:
const pre_root = (try manifest.storeRoot(store_id)) orelse return error.NoStore;
const txn_seq = manifest.nextApply();
// ... handler runs, calls store.put / store.delete ...
// Now: propose the writeset via raft.

if (raft_accepted) {
    // No further work — the apply on this node already ran.
    // The next checkpoint makes it durable.
} else {
    // Raft rejected (NotLeader, timeout, etc.). Revert by
    // pointing the store_root back at pre_root.
    _ = manifest.nextApply();
    try manifest.setStoreRoot(store_id, pre_root);
}
```

**Important:** if `durabilize()` runs between TXN commit and raft
accept/reject, the rollback path via root-pointer revert no longer
works (the new pages are durable; an old root would still work but
freelist reclamation might already have moved pages around). Hence
"deferred-durabilize": the integration must hold off durabilize until
all in-flight raft proposals have resolved.

### State transfer (wipe-and-restore)

When a follower falls so far behind that the leader's raft log has
compacted past it, the follower needs the whole state, not a delta.
kvexp's `dumpSnapshot` and `loadSnapshot` cover this:

```zig
// Leader: produce a snapshot file at `dest_path`.
{
    var snap = try manifest.openSnapshot();
    defer snap.close();
    var file = try std.fs.cwd().createFile(dest_path, .{});
    defer file.close();
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(alloc);
    var w = buf.writer(alloc);
    try kvexp.dumpSnapshot(&snap, &w);
    try file.writeAll(buf.items);
    try file.sync();
}
try manifest.durabilize();
// Tell raft this snapshot file covers up to lastAppliedRaftIdx.

// Follower: install.
{
    // Drop every existing store so the receiver converges to the
    // sender's exact set.
    const existing = try manifest.listStores(alloc);
    defer alloc.free(existing);
    for (existing) |id| _ = try manifest.dropStore(id);

    var file = try std.fs.cwd().openFile(src_path, .{});
    defer file.close();
    const bytes = try file.readAll(alloc);  // small in practice
    defer alloc.free(bytes);
    var r = std.io.fixedBufferStream(bytes);
    const last_applied_idx = try kvexp.loadSnapshot(manifest, r.reader());
    try manifest.durabilize();
    // raft.loadSnapshot(last_applied_idx, snap_last_term) tells the
    // raft library where to resume applying.
}
```

### Transport is the caller's concern

kvexp deliberately doesn't pick the snapshot transport. The leader
writes the file; the application's HTTP layer (or whatever) serves
it. The raft message carrying the offer is metadata only (URL +
metadata), so the raft thread never blocks on the actual transfer.

## API surface

```zig
const kvexp = @import("kvexp");

// Open a manifest on a file (creates if missing). The PagedFile +
// BufferPool + PageCache are caller-owned; many ways to wire them.
var file = try kvexp.PagedFile.open(path, .{ .create = true });
defer file.close();
var pool = try kvexp.BufferPool.init(alloc, kvexp.PAGE_SIZE_DEFAULT, 16384);
defer pool.deinit(alloc);
var cache = try kvexp.PageCache.init(alloc, &file, &pool, .{});
defer cache.deinit();
var manifest: kvexp.Manifest = undefined;
try manifest.init(alloc, &cache, &file);
defer manifest.deinit();

// Tenant lifecycle.
try manifest.createStore(tenant_id);     // u64 — caller hashes from string
const exists = try manifest.hasStore(id);
const dropped = try manifest.dropStore(id);
const tenant_ids = try manifest.listStores(alloc);  // []u64
defer alloc.free(tenant_ids);

// Per-tenant ops.
var store = try kvexp.Store.open(&manifest, tenant_id);
defer store.deinit();
try store.put("key", "value");
const v = try store.get(alloc, "key");        // ?[]u8 — caller frees
defer if (v) |b| alloc.free(b);
_ = try store.delete("key");                  // returns true if existed
var cursor = try store.scanPrefix("user/");
defer cursor.deinit();
while (try cursor.next()) {
    // cursor.key() / cursor.value() — slices alias the page cache;
    // copy if you need to retain past the next .next() call.
}

// Apply / durabilize.
_ = manifest.nextApply();                      // begin a logical apply unit
try store.put(k, v);                            // ...mutate...
try manifest.durabilize();                     // checkpoint to disk

// Raft watermark.
manifest.setLastAppliedRaftIdx(idx);
const idx = manifest.lastAppliedRaftIdx();

// Snapshots.
var snap = try manifest.openSnapshot();
defer snap.close();
const root = try snap.storeRoot(tenant_id);
const v2 = try snap.get(alloc, tenant_id, "key");
var c2 = try snap.scanPrefix(tenant_id, "user/");
defer c2.deinit();
const ids = try snap.listStores(alloc);
defer alloc.free(ids);
try kvexp.dumpSnapshot(&snap, writer);
const last_applied = try kvexp.loadSnapshot(&manifest, reader);

// Verify (debugging / fault injection tests).
const report = try manifest.verify(alloc);
// → { store_count, manifest_tree_pages, store_tree_pages,
//     freelist_tree_pages, orphan_pages, durable_seq, ... }
```

### Concurrency model

- **Reads** (`Store.get`, `scanPrefix`) take only the page cache shard
  lock, briefly per page. Multiple readers across stores never contend.
- **Writes** (`Store.put`, `delete`) take a per-store lock via
  `manifest.storeLock(id)`. Writers on different stores don't
  contend; writers on the same store serialize.
- **`setStoreRoot`** (called internally by every write) takes the
  in-memory store-root cache lock briefly — O(1) hashmap ops.
- **`durabilize`** takes the tree_lock + alloc_lock during the flush
  phase. Writers can keep running on the per-store locks; their seq
  stamps may or may not be in this round depending on timing.
- **Snapshots** are read-only; opening one increments a refcount that
  blocks the freelist from reclaiming pages until the snapshot closes.

The lock ordering, top to bottom:

```
store_lock(id)
  → tree_lock | freelist_tree_lock
    → store_root_cache_lock
      → alloc_lock
        → cache.shard.lock (interior to PageCache)
```

No code path holds an earlier lock while acquiring a later one.

## Performance

(Numbers from `bench/kvstore_bench.zig` in the rove-kvexp consumer
repo. Run on a fast NVMe; B6/B7 in particular are I/O bound.)

### B4 — idle-tenant footprint (10k tenants × 1 write each)

|           | sqlite     | kvexp    |
|-----------|------------|----------|
| RSS       | 1841 MB    | 43 MB    |
| fds       | 30,004     | 6        |
| disk      | 862 MB     | 40 MB    |
| open time | 8.0 s      | 0.16 s   |

### B6 — concurrent saturated-load (WAL-via-raft cadence, no fsync during workload)

|   W | sqlite ops/s | kvexp ops/s | kvexp:sqlite |
|----:|-------------:|------------:|:-------------|
|   1 |       55,730 |     180,771 | 3.2×         |
|   4 |       54,011 |     373,460 | 6.9×         |
|   8 |       54,816 |     378,753 | 6.9×         |
|  16 |       55,400 |     305,342 | 5.5×         |
|  32 |       53,560 |     306,746 | 5.7×         |

### B6 — same-file contention (32 writers, 1 tenant)

|         | ops/s   | notes                                             |
|---------|--------:|---------------------------------------------------|
| sqlite  |       0 | all workers crash with SQLITE_BUSY after retry    |
| kvexp   |  25,377 | per-store lock serializes writers cleanly         |

### B7 — snapshot transfer cycle

|  tenants × keys |  file size  |  total time  |
|----------------:|------------:|-------------:|
|     100 × 1     |   0.02 MB   |       4 ms   |
|   1,000 × 1     |   0.15 MB   |      29 ms   |
|  10,000 × 1     |   1.53 MB   |     347 ms   |
|   1,000 × 100   |  15.26 MB   |     490 ms   |

A cold-start follower joining a 10k-tenant cluster fetches and
installs in ~350 ms.

## Limitations

- **Linux + io_uring + O_DIRECT.** Portability is a non-goal.
- **No cross-store atomicity.** Application layer responsibility.
- **No background compaction.** The freelist reclaims orphan pages
  immediately at `durabilize`-promotion time; there's no compaction
  for fragmented stores.
- **Snapshot files load into memory at apply time.** Streaming
  `loadSnapshot` is possible (kvexp's primitives stream; only the
  legacy Reader interface we used on Zig 0.15.2 buffers) but not
  required at expected sizes (MB range per snapshot).
- **Empty stores don't round-trip in snapshots.** A store with zero
  entries emits zero records; the receiver never learns it existed.
  Acceptable for raft state transfer where raft replay re-creates it
  on the next createStore.

## Code layout

```
src/
  paged_file.zig    direct I/O + io_uring wrapper
  buffer_pool.zig   page-aligned slab; provides BufferIndex handles
  page_cache.zig    sharded cache with clock eviction + pin/release
  page.zig          on-disk page layout (Leaf + Internal headers + cells)
  btree.zig         CoW B-tree with hybrid in-place mutation
  page_allocator.zig  PageAllocator interface (manifest implements it)
  header.zig        4KB file-header layout (offset 0, page 0)
  manifest.zig      Manifest (tree forest + slot swap + freelist) +
                    Store + Snapshot + dumpSnapshot/loadSnapshot
  root.zig          public re-exports
docs/
  PLAN.md           full design discussion + non-goals + rejected ideas
```

## Where to read next

- `docs/PLAN.md` — the locked architecture document, rejected ideas,
  phased delivery breakdown.
- `src/manifest.zig` top doc — the slot-swap protocol, freelist
  reuse rules, snapshot pages-freed accounting.
- `src/btree.zig` top doc — the hybrid CoW invariant and why the
  shadow check + parent shortcut compose.
