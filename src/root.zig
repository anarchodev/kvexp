//! kvexp — multi-tenant embedded KV. LMDB-backed durable B-tree
//! fronted by an in-memory per-store memtable (overlay). The raft log
//! is the WAL; kvexp is a periodically-checkpointed materialization
//! of the applied prefix.

pub const overlay = @import("overlay.zig");
pub const lmdb = @import("lmdb.zig");
pub const manifest = @import("manifest.zig");

pub const Manifest = manifest.Manifest;
pub const Store = manifest.Store;
pub const Snapshot = manifest.Snapshot;
pub const StorePrefixCursor = manifest.StorePrefixCursor;
pub const SnapshotPrefixCursor = manifest.SnapshotPrefixCursor;
pub const dumpSnapshot = manifest.dumpSnapshot;
pub const loadSnapshot = manifest.loadSnapshot;
pub const SNAPSHOT_MAGIC = manifest.SNAPSHOT_MAGIC;
pub const SNAPSHOT_VERSION = manifest.SNAPSHOT_VERSION;
pub const InitOptions = manifest.InitOptions;

test {
    @import("std").testing.refAllDecls(@This());
}
