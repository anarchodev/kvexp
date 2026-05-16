# Changelog

All notable changes to `kvexp` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/). Pre-1.0: the public API may change
between minor versions.

## [0.1.0] - 2026-05-16

First tagged release. `kvexp` is an embedded multi-tenant KV store designed to
sit under a raft log: the raft log is the WAL, kvexp is a periodically
checkpointed materialization of the applied prefix. One raft log is shared
across many mostly-idle tenants on a node, with per-tenant isolation in the
in-memory layer instead of separate raft groups.

### Storage & architecture
- LMDB-backed durable store: one environment file with per-tenant sub-DBIs,
  a `_meta` raft-watermark slot, and a `_stores` tenant directory.
- Per-tenant in-memory overlay ("memtable") in front of LMDB; `main_overlay`
  holds committed-but-not-yet-durabilized writes.
- `durabilize(raft_idx)` is the only path that writes to LMDB: one atomic
  write txn applies pending createStore/dropStore, drains every tenant's
  `main_overlay`, and stamps the raft watermark. Three-step drain via a
  `draining_overlay` handoff slot keeps keys continuously visible to
  concurrent readers across the move→commit gap.
- `flush()` (= `durabilize(0)`) checkpoints without disturbing the raft
  watermark.

### Transactions & concurrency
- `StoreLease`: exclusive per-tenant dispatch handle; the only write path.
  Lease may be released before the Txn commits (speculative-apply pattern) —
  the Txn lives in a per-tenant chain in raft-propose order until raft decides.
- Per-Txn overlays with LIFO savepoints (`open_child`); inside-out read walk:
  savepoint stack → enclosing Txn chain → `main_overlay` → `draining_overlay`
  → LMDB. Cross-tenant in-flight state is never visible.
- Top-level `Txn.rollback` cascades to all chain successors (matches raft
  tail-rejection on leadership loss).
- Read-only + speculation-free Txn fast path: splices out of the chain in
  place; `Txn.canSkipRaftPropose()` lets the application skip the raft propose.
- Batch read view (`Txn.beginReadView` / `refreshReadView` / `endReadView`):
  parks one `MDB_RDONLY` txn so point reads skip per-`get` reader-slot churn;
  read-latest, force-released on commit / rollback / cascade / `dropStore`.
- Documented lock order; refcounted `TenantState` freed on `dropStore` after
  the last lease releases.

### State transfer & recovery
- Snapshot dump / load (`dumpSnapshot` / `loadSnapshot`) for raft state
  transfer; recovery from the raft watermark; in-memory poison guard for the
  durabilize partial-failure window (does not persist; cleared on reopen).

### Observability
- `Metrics` struct + `metricsSnapshot()` with Prometheus-shaped duration
  histograms; hot counters sharded across cache-line-sized slots.

### Testing & tooling
- 70 tests: recovery-contract, multi-threaded property tests (workers +
  checkpointer + recovery), speculative-apply, real-LMDB-error fault
  injection, `loadSnapshot` fuzzing, and real-crash (SIGKILL) recovery.
- `zig build` / `zig build test` / `zig build bench`.

[0.1.0]: https://github.com/anarchodev/kvexp/releases/tag/v0.1.0
