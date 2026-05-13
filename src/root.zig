//! kvexp — multi-tenant embedded KV. LMDB-backed durable B-tree
//! fronted by per-tenant chains of nested transactions (memtable +
//! savepoints). The raft log is the WAL; kvexp is a periodically-
//! checkpointed materialization of the applied prefix.

pub const overlay = @import("overlay.zig");
pub const lmdb = @import("lmdb.zig");
pub const manifest = @import("manifest.zig");

pub const Manifest = manifest.Manifest;
pub const StoreLease = manifest.StoreLease;
pub const Txn = manifest.Txn;
pub const Snapshot = manifest.Snapshot;
pub const TxnPrefixCursor = manifest.TxnPrefixCursor;
pub const SnapshotPrefixCursor = manifest.SnapshotPrefixCursor;
pub const InitOptions = manifest.InitOptions;
pub const Metrics = manifest.Metrics;
pub const MetricsSnapshot = manifest.MetricsSnapshot;
pub const Histogram = manifest.Histogram;
pub const HistogramSnapshot = manifest.HistogramSnapshot;
pub const dumpSnapshot = manifest.dumpSnapshot;
pub const loadSnapshot = manifest.loadSnapshot;
pub const SNAPSHOT_MAGIC = manifest.SNAPSHOT_MAGIC;
pub const SNAPSHOT_VERSION = manifest.SNAPSHOT_VERSION;

test {
    @import("std").testing.refAllDecls(@This());
}
