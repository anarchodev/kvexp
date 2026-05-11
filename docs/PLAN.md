# kvexp — multi-tenant embedded KV with raft-as-WAL

## 1. What this is

`kvexp` is an embedded key-value store written in Zig. It hosts many independent KV stores per node ("stores", analogous to RocksDB column families or bbolt buckets), with durability provided by an **external raft log** rather than a per-store write-ahead log. It is intended as the next-generation storage layer for `rove-kv` (`~/src/rove/src/kv/`), which currently uses one SQLite database per tenant and hits a ceiling on SQLite shm lock contention as worker counts increase past ~8.

This document is the locked architecture and phased delivery plan. Read it before contradicting direction. §11 lists explicitly rejected ideas — do not re-propose without new information.

## 2. Goals

1. **10k mostly-idle stores per node.** Idle cost: a single row in the manifest (page pointer + small metadata). No memtable, no per-store WAL, no open files, no buffers.
2. **Per-store transactional unit.** Writes within a store are ACID; cross-store atomicity is the application's job via a TCC try/confirm/cancel pattern at the JS layer above. The storage engine never sees cross-store transactions.
3. **Ordered KV with prefix scans.** Keys are byte strings ordered lexicographically. Prefix-bounded iteration is a first-class operation.
4. **Per-store write concurrency.** Two workers writing to *different* stores must not serialize on file-level locks (bbolt/LMDB single-writer-per-file is the anti-pattern).
5. **Raft log is the WAL.** Local on-disk state is a checkpoint of the applied raft prefix. Recovery replays the raft tail past the durable manifest sequence.
6. **Group commit at storage layer.** fsync is amortized across multiple raft proposes when raft commits arrive faster than fsync turnaround. Page versions superseded inside a commit window are elided as orphans — never written.

## 3. Non-goals

- Cross-store transactions at the storage layer.
- Auto-sharding within a logical store. (Users opt into multiple stores explicitly.)
- Portability beyond Linux. The implementation assumes io_uring and `O_DIRECT`.
- MVCC at the API surface. Reads see the latest manifest at read-tx start; writes produce a new manifest. Long historical history is not retained.
- Per-store background work for idle stores. No compaction, no expirer threads, no per-store flushers.
- Server / network surface. `kvexp` is in-process embedded only; multi-node concerns live above it.

## 4. Locked architectural decisions

| # | Decision | Why |
|---|---|---|
| A1 | One store per tenant; no transparent sharding | Users decide sharding boundaries themselves |
| A2 | No cross-store ACID; TCC in JS handles cross-store | Drops the "I" by design; matches product surface |
| A3 | One raft log per node, shared across all stores | Enables free movement of stores between nodes; collapses N WALs into one |
| A4 | Write locks at store boundaries only | Independent stores ≠ shared lock surface |
| A5 | Apply / durabilize split; storage does not own raft | Allows pipelined apply while raft is in-flight |
| A6 | Target: 10k mostly-idle stores per node | Drives idle-cost design |
| A7 | KV with prefix scans; TTL deferred to v2 | Minimal surface, room to grow |

## 5. Locked engineering picks

| # | Decision | Why |
|---|---|---|
| E1 | Page size 4KB (configurable in file header) | Matches block device default, sane CoW amplification |
| E2 | io_uring + `O_DIRECT`, Linux-only | We own the page cache; no kernel writeback interfering with orphan elision |
| E3 | Single file per node (data + manifest + free-list) | Simpler atomicity; split is a v2 consideration |
| E4 | Inline values up to ~¼ page; overflow chain for larger | Dense leaves for the common small-value case |
| E5 | Little-endian on-disk, magic bytes, version field | Standard format-evolution hygiene |
| E6 | In-process page cache (not mmap) | Orphan elision under group commit requires control of writeback |

## 6. On-disk format (sketch)

```
File layout (offsets in pages):
┌────────────────────────────────────────────┐
│ 0: Header (magic, version, page size,      │
│            active manifest slot pointer)   │
├────────────────────────────────────────────┤
│ 1: Manifest header slot A                  │
│ 2: Manifest header slot B  (atomic swap)   │
├────────────────────────────────────────────┤
│ 3+: data pages, manifest pages, free-list  │
│     pages, overflow chains — interleaved   │
│     by allocation order, NOT grouped by    │
│     store                                  │
└────────────────────────────────────────────┘
```

**Page kinds:** B-tree leaf, B-tree internal, manifest leaf, manifest internal, free-list leaf, overflow chain link. Each page has a 32-byte header (kind tag, checksum, sequence-produced-at, free-after-sequence) and 4064 bytes of body.

**Manifest** is a B-tree keyed by `store_id` (u64), valued with `{root_page: u64, store_meta: …}`. Lookups go through it. Per-store idle state is one entry.

**Free list** is a B-tree keyed by `(freed_at_sequence, page_no)` (LMDB-style). A page joins the free list only after its freed_at_sequence is older than the minimum live read snapshot.

## 7. Mechanics

### 7.1 CoW shadow-paging
A write to a leaf produces a new leaf page (allocated from the free list or end-of-file); the parent is then CoW'd to point at the new leaf; the cascade reaches the store's root. The old chain is still reachable from the previous manifest version. Commit = atomically swap the manifest root pointer in the header slot.

### 7.2 Apply vs durabilize
- **Apply** is in-memory: produce shadow pages tagged with the producing raft sequence, build a new in-memory manifest version. No I/O. Cheap.
- **Durabilize(K)** is the I/O path: gather the dirty page set produced by sequences `(last_durable, K]`, **subtract orphans** (pages superseded within that window), pwrite the survivors via io_uring, fsync data, write the manifest into the inactive header slot, fsync header. Atomic at the header-slot swap.

### 7.3 Group commit + orphan elision
When raft commits `N, N+1, N+2` faster than one fsync cycle, `Durabilize(N+2)` skips intermediate manifests entirely. Pages allocated for `N` and `N+1` that are superseded by `N+2`'s CoW are freelisted without ever being pwritten. A write-hot key under fast raft cadence costs ≪ one disk write per propose. Requires owning the page cache (E6).

### 7.4 Concurrent writers across stores
- **Per-store write lock** — two workers mutating the *same* store serialize. (Already the case in the rove worker dispatch model.)
- **Free-page allocator** — workers allocating for *different* stores enter a short shared critical section. Initial implementation: one mutex over the in-RAM free-page cache. v2: per-CPU caches with periodic rebalance.
- **Manifest assembly** — at the end of apply, one thread updates the in-memory manifest version with new roots. This is small (one manifest-tree leaf per touched store) and happens in raft-apply order (which is total).
- **Durabilize** — single-writer (driven by the raft thread on commit). Group commit makes this cheap.

### 7.5 Watermarks and response release
Two watermarks are exposed:
- `applied_index`: highest raft index whose writes are in the in-memory manifest chain.
- `durable_index`: highest raft index whose manifest is on disk and fsynced.

A worker releases a response for a request that was part of raft propose `p` when `min(raft_committed_index, durable_index) >= p`.

### 7.6 Crash recovery
1. Open file, read header.
2. Validate manifest slots A and B (checksums, sequence field). Pick the higher valid one as active.
3. Read durable free-list from that manifest. (Optionally walk reachable pages to verify free-list invariant on `--repair`.)
4. Surface `durable_index` to the caller. The raft layer replays any committed-but-undurable raft entries forward from there.

Pages allocated but not referenced by the active manifest are orphans from a torn pre-commit state; they get reclaimed into the free list on the next allocation cycle (or eagerly on repair).

## 8. Phase plan

Each phase ends in a testable state. Inline Zig tests are co-located with code (rove convention).

### Phase 0/1 — Page I/O + in-process page cache (one PR)
File header struct; io_uring + `O_DIRECT` page reader/writer with aligned buffer management; in-process buffer pool with bounded RAM budget; page cache (hash table page_no → BufferRef) with pin/unpin, clock eviction, dirty tracking tagged by producing-sequence. Demo: hammer with random read/write, see dirty pages persist via explicit flush, see eviction respect pinned pages.

### Phase 2 — Single CoW B-tree
Leaf and internal page formats, length-prefixed key/value encoding, get/put/delete/prefix-scan, CoW shadow paths, split/merge/rebalance, stub in-memory free-page list (no durability yet). Demo: insert 10M keys, prefix-scan, delete half, verify shape against a reference `std.BoundedArray`/`std.AutoArrayHashMap`-based oracle.

### Phase 3 — Manifest + forest
Manifest as B-tree keyed by `store_id`. Two-slot manifest header with atomic swap. `open_store`/`create_store`/`drop_store`. Multi-store apply produces disjoint shadow subtrees + new manifest root. Demo: 10k stores created, key into each, prefix-scan one, drop one. Test torn-write recovery between data-fsync and header-fsync.

### Phase 4 — Durable free-page allocator
Free list as B-tree keyed by `(freed_at_sequence, page_no)`. Per-transaction "pending free" lists. Reclamation tied to minimum-live-reader sequence. Demo: net-zero workload, verify file size and free-list balance. Stress with churn, fault-inject across allocator state to verify no leaks.

**Freelist format:** vector-valued chunks. Key is `(freed_at_seq, first_page_no_in_chunk)` and the value packs up to `PAGES_PER_CHUNK ≈ 255` page_nos. A commit that frees `P` pages writes `⌈P/255⌉` freelist puts, not `P`, so the freelist's own CoW cost stays a constant factor of the user workload. `refillReusable` reads whole chunks at a time and queues each chunk's key for deletion at the next commit — the on-disk freelist size stays proportional to *in-flight reusable pages*, not to historical freeds.

### Phase 5 — Apply / durabilize split + group commit + orphan elision ✅
Apply produces in-memory manifest version + tagged dirty pages, zero I/O. `durabilize()` collapses intermediate manifests, subtracts orphans, pwrites + fsyncs. Watermark API. Demo: hot-key 10-apply burst writes **4 pwrites** total vs ~30 naive.

`Manifest.nextApply()` bumps `tree.seq` so subsequent mutations tag with a fresh seq. `Manifest.durabilize()` snapshots `pending_free`, builds an orphan set of pages tagged `freed_at_seq ≤ K`, folds into the freelist, then **extends the orphan set with the post-fold pending_free entries** (the intermediate freelist-tree pages that the fold itself just CoW'd away — their content is junk no manifest will ever reference). Finally calls `cache.flushUpToSkipping(K, &orphans)` so dirty pages in the orphan set are marked clean without hitting disk.

The two-slot reuse rule generalizes to "skip freelist entries with `freed_at_seq == inactive_seq + 1`" (only the older durable manifest's referenced pages are unsafe; large group commits expose many safe entries at once).

### Phase 6 — Per-store concurrent writers ✅
Per-store write lock, concurrent free-page allocator, serialized manifest tree access. Demo: 4 threads × 200 keys on 4 distinct stores all persist correctly across reopen; 2 threads racing the SAME store serialize cleanly via the per-store mutex.

Lock layout:
- `Manifest.store_locks` — `AutoHashMap(u64, *Mutex)`, allocated lazily per store_id on first `Store.put`. Caller-managed in spirit but exposed for testing.
- `Manifest.tree_lock` — short critical sections around manifest tree reads/writes (`storeRoot`, `setStoreRoot`, `nextApply`, capturing snapshot in `durabilize`).
- `Manifest.freelist_tree_lock` — only held during durabilize's fold + delete and during `refillReusable`.
- `Manifest.alloc_lock` — short critical sections around `reusable` / `pending_free` / `consumed_keys` / `file.growBy`. Acquired inside `allocImpl`/`freeImpl` so worker threads' store-tree CoW operations contend only briefly.
- `PageCache` — **sharded**, one mutex per shard. `page_no % shard_count` picks the shard; auto-shard count = `clamp(pool.capacity / 4, 1, 16)`. Workers on disjoint page_nos contend on different mutexes.
- `PageCache.evictForReuse` releases its shard lock during the writeback `pwrite`. A new `.evicting` slot state pins the buffer for the calling thread; other threads on the same shard can proceed. The long sync I/O happens **off the lock**.
- `PagedFile.io_lock` — the io_uring instance isn't safe for concurrent SQ/CQ access from multiple threads, so we serialize all read/write/fsync through this mutex. A future optimization would be per-thread rings.

Per-store CoW happens with **no manifest lock held** — workers contend only briefly on alloc_lock and per-shard cache locks.

### Phase 7 — Infrastructure long-reads (snapshots) ✅
**Not for the request path.** Application requests dispatch through per-store batches: a worker grabs the store's lock, drains its queue under a single in-memory transaction with savepoints between requests, and releases. Reads and writes inside a request are serialized under that lock; the 10ms budget plus the bounded prefix-scan size means a request can never hold a read open across someone else's writes. MVCC isn't load-bearing for that pattern.

What snapshots ARE for: infrastructure callers that need a coherent view *while writers continue*. The motivating case is **phase 10 raft snapshot transfer** — the leader has to walk the durable manifest plus every reachable page to ship state to a catching-up follower, which can run for seconds or minutes; without snapshot pinning, pages freed during the transfer become eligible for reuse two commits later and the export reads garbage. Same machinery serves online backups and debug/admin dumps (`kvexp.fsck`, `kvexp.dump`).

`Manifest.openSnapshot()` captures `(snap_seq, manifest_root)` atomically under `tree_lock`. The returned `Snapshot` provides `get(store_id, key)` and `scanPrefix(store_id, prefix)` that walk the captured manifest tree root and the captured per-store tree root — invariant against concurrent writers' CoW operations.

`refillReusable` consults `minLiveSnapSeq()`: a page tagged `freed_at_seq=N` is potentially in a snapshot's view iff some live snapshot has `snap_seq >= N`, so the page stays out of the reusable queue. When the last referencing snapshot closes, the next `refillReusable` (which `durabilize` calls at the end) picks up the newly-eligible chunks.

The `PrefixCursor` was refactored to hold `(cache, root)` directly rather than `*Tree`, so snapshots can produce cursors against their captured root without aliasing a live Tree. (This refactor is independently useful regardless of the snapshot machinery.)

### Phase 8 — Recovery
Header validation, manifest slot picking, free-list rebuild (or trust + verify), `durable_index` exposure. Demo: `kill -9` across the durabilize sequence, verify recovery to last durable manifest.

### Phase 9 — Raft adapter (rove glue)
Decode rove writeset envelopes → kvexp apply call → produce dirty-page set tagged with raft index → durabilize on raft commit → expose `min(raft_committed, durable_index)` for response release. Slot `kvexp` under `rove-kv`'s existing tests for one tenant; all rove tests pass.

### Phase 10 — Snapshot/checkpoint integration
Hook raft's snapshot-and-truncate flow. The durable manifest already *is* the snapshot — this phase wires the protocol so log truncation is gated on `durable_index >= snapshot_index`.

### Phase 11 — Hardening
Sim-test framework integration, perf tuning (allocator hot path, page-cache lookup, leaf scan), stats surface (per-store size, dirty-page count, allocator contention, fsync latency histograms).

## 9. Deferred

- **TTL.** Lazy "check on read" + opportunistic sweep; no per-store background expirer. Phase 11+.
- **Page compression.** Possibly never; the working set is small-per-store and CPU cost would dominate.
- **Page-level encryption.** Format header should leave room for a per-page MAC; implementation deferred until rove's encryption story settles.
- **Multi-file storage.** Only if profiling at 10k stores shows the single allocator/file is the bottleneck.

## 10. Rejected ideas

Do not re-propose without new information.

| Idea | Why rejected |
|---|---|
| LSM (RocksDB-style) | Per-CF memtables, bloom filters, compaction, multi-file pressure are all per-CF costs that don't fit 10k mostly-idle stores. LSM's write-throughput win is moot under our raft cadence; dispatch-layer batching already coalesces hot keys. |
| bbolt or LMDB as-is | Single-writer-per-file. We need per-store write concurrency. |
| SQLite-per-tenant | Already tested in rove: shm lock contention dies past 8 workers. |
| RocksDB with column families | Per-CF buffers don't scale to 10k mostly-idle CFs. |
| mmap-based page cache | Kernel writeback breaks orphan elision; we need to own write decisions. |
| Percolator-style cross-store transactions | Replaced by TCC in JS at application layer. |
| One SQLite DB with table-per-tenant | Single SQLite writer kills per-store concurrency; large schema fragility. |
| pread/pwrite portability fallback | Linux-only is acceptable; io_uring + O_DIRECT is the only I/O path. |

## 11. References

- `~/src/rove/src/kv/` — current `rove-kv` module: per-tenant SQLite, willemt/raft, raft_net over liburing, proposal batcher.
- `~/src/rove/CLAUDE.md`, `~/src/rove/docs/PLAN.md` — rove product/engine context. kvexp is expected to replace or sit under `rove-kv`.
- **LMDB** (`mdb.c`) — canonical CoW B-tree, named sub-databases are a forest in one file.
- **bbolt** (`github.com/etcd-io/bbolt`) — readable Go CoW B-tree, "buckets" are nested forests.
- **WiredTiger** — concurrent CoW B-tree with finer-grained locking.
