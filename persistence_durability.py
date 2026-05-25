"""
Persistence and Durability Layer
- Point-in-time recovery
- Background snapshots
- Compaction and cleanup
- Crash recovery
"""

import os
import json
import time
import threading
import logging
import pickle
import gzip
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

log = logging.getLogger("PERSISTENCE")


@dataclass
class SnapshotMetadata:
    """Metadata for snapshot."""
    snapshot_id: str
    timestamp: float
    node_id: str
    state_size_bytes: int
    entries_count: int
    compressed_size_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None
    checksum: str = ""


@dataclass
class BackupManifest:
    """Manifest for backup set."""
    backup_id: str
    created_at: float
    node_id: str
    snapshots: List[SnapshotMetadata] = field(default_factory=list)
    wal_segments: List[str] = field(default_factory=list)
    total_size_bytes: int = 0


class SnapshotManager:
    """
    Manages snapshots for point-in-time recovery.
    
    Strategy:
    - Periodic snapshots
    - Incremental snapshots
    - Compression
    """

    def __init__(self, storage_dir: str = "./persistence/snapshots",
                 retention_days: int = 30):
        """
        Initialize snapshot manager.
        
        Args:
            storage_dir: Directory for snapshots
            retention_days: Keep snapshots for N days
        """
        self.storage_dir = storage_dir
        self.retention_days = retention_days
        self.snapshots: Dict[str, SnapshotMetadata] = {}
        self.lock = threading.RLock()

        os.makedirs(storage_dir, exist_ok=True)
        self._load_snapshots_from_disk()
        log.info(f"Initialized SnapshotManager with {len(self.snapshots)} existing snapshots")

    def create_snapshot(self, node_id: str, state: Dict[str, Any]) -> SnapshotMetadata:
        """
        Create snapshot of state.
        
        Args:
            node_id: Node creating snapshot
            state: State to snapshot
            
        Returns:
            SnapshotMetadata
        """
        with self.lock:
            snapshot_id = f"snap_{node_id}_{int(time.time()*1000)}"
            timestamp = time.time()

            # Serialize state
            state_bytes = pickle.dumps(state)
            state_size = len(state_bytes)

            # Compress
            filepath = os.path.join(self.storage_dir, f"{snapshot_id}.pkl.gz")

            with gzip.open(filepath, 'wb') as f:
                f.write(state_bytes)

            compressed_size = os.path.getsize(filepath)
            compression_ratio = state_size / compressed_size if compressed_size > 0 else 1.0

            # Compute checksum
            import hashlib
            checksum = hashlib.sha256(state_bytes).hexdigest()

            # Create metadata
            metadata = SnapshotMetadata(
                snapshot_id=snapshot_id,
                timestamp=timestamp,
                node_id=node_id,
                state_size_bytes=state_size,
                entries_count=len(state),
                compressed_size_bytes=compressed_size,
                compression_ratio=compression_ratio,
                checksum=checksum
            )

            self.snapshots[snapshot_id] = metadata
            log.info(f"Created snapshot {snapshot_id}: {state_size} bytes -> {compressed_size} bytes "
                    f"({compression_ratio:.2f}x compression)")

            return metadata

    def restore_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore state from snapshot.
        
        Args:
            snapshot_id: Snapshot to restore
            
        Returns:
            State dict or None if not found
        """
        with self.lock:
            if snapshot_id not in self.snapshots:
                log.warning(f"Snapshot not found: {snapshot_id}")
                return None

            filepath = os.path.join(self.storage_dir, f"{snapshot_id}.pkl.gz")

            if not os.path.exists(filepath):
                log.error(f"Snapshot file not found: {filepath}")
                return None

            try:
                with gzip.open(filepath, 'rb') as f:
                    state = pickle.loads(f.read())

                log.info(f"Restored snapshot {snapshot_id}")
                return state
            except Exception as e:
                log.error(f"Failed to restore snapshot: {e}")
                return None

    def get_latest_snapshot(self, node_id: str) -> Optional[SnapshotMetadata]:
        """Get latest snapshot for node."""
        with self.lock:
            node_snapshots = [
                s for s in self.snapshots.values()
                if s.node_id == node_id
            ]

            if not node_snapshots:
                return None

            return sorted(node_snapshots, key=lambda s: s.timestamp)[-1]

    def cleanup_old_snapshots(self) -> int:
        """
        Delete snapshots older than retention period.
        
        Returns:
            Number of snapshots deleted
        """
        with self.lock:
            cutoff_time = time.time() - (self.retention_days * 86400)
            to_delete = []

            for snap_id, metadata in self.snapshots.items():
                if metadata.timestamp < cutoff_time:
                    to_delete.append(snap_id)

            deleted = 0
            for snap_id in to_delete:
                filepath = os.path.join(self.storage_dir, f"{snap_id}.pkl.gz")

                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    del self.snapshots[snap_id]
                    deleted += 1
                    log.info(f"Deleted old snapshot: {snap_id}")
                except Exception as e:
                    log.error(f"Failed to delete snapshot {snap_id}: {e}")

            return deleted

    def _load_snapshots_from_disk(self):
        """Load snapshot metadata from disk."""
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.pkl.gz'):
                    snap_id = filename[:-7]  # Remove .pkl.gz
                    filepath = os.path.join(self.storage_dir, filename)

                    try:
                        # Compute checksum from file
                        import hashlib
                        with gzip.open(filepath, 'rb') as f:
                            checksum = hashlib.sha256(f.read()).hexdigest()

                        size = os.path.getsize(filepath)

                        # Create minimal metadata for loaded snapshots
                        self.snapshots[snap_id] = SnapshotMetadata(
                            snapshot_id=snap_id,
                            timestamp=os.path.getmtime(filepath),
                            node_id=snap_id.split('_')[1] if '_' in snap_id else "unknown",
                            state_size_bytes=0,
                            entries_count=0,
                            compressed_size_bytes=size,
                            checksum=checksum
                        )
                    except Exception as e:
                        log.warning(f"Failed to load snapshot metadata for {filename}: {e}")
        except Exception as e:
            log.error(f"Failed to load snapshots from disk: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get snapshot manager status."""
        with self.lock:
            total_size = sum(
                s.compressed_size_bytes or 0
                for s in self.snapshots.values()
            )

            return {
                'total_snapshots': len(self.snapshots),
                'total_size_bytes': total_size,
                'retention_days': self.retention_days,
                'storage_dir': self.storage_dir
            }


class CrashRecovery:
    """
    Crash recovery mechanism.
    
    Process:
    1. Load latest snapshot
    2. Replay WAL entries after snapshot
    3. Rebuild state
    """

    def __init__(self, snapshot_manager: SnapshotManager):
        """
        Initialize crash recovery.
        
        Args:
            snapshot_manager: SnapshotManager instance
        """
        self.snapshot_manager = snapshot_manager
        self.recovery_in_progress = False
        self.lock = threading.RLock()

    def recover_node_state(self, node_id: str, wal_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Recover node state after crash.
        
        Args:
            node_id: Node to recover
            wal_entries: WAL entries for replay
            
        Returns:
            Recovered state dict
        """
        with self.lock:
            if self.recovery_in_progress:
                log.warning("Recovery already in progress")
                return {}

            self.recovery_in_progress = True

            try:
                # Step 1: Load latest snapshot
                latest_snap = self.snapshot_manager.get_latest_snapshot(node_id)
                state = {}

                if latest_snap:
                    state = self.snapshot_manager.restore_snapshot(latest_snap.snapshot_id)
                    if state is None:
                        state = {}
                    log.info(f"Loaded snapshot {latest_snap.snapshot_id} with {len(state)} entries")

                # Step 2: Replay WAL entries after snapshot
                if latest_snap:
                    entries_to_replay = [
                        e for e in wal_entries
                        if e.get('timestamp', 0) > latest_snap.timestamp
                    ]
                else:
                    entries_to_replay = wal_entries

                for entry in entries_to_replay:
                    try:
                        operation = entry.get('operation', 'SET')
                        key = entry.get('key')
                        value = entry.get('value')

                        if operation == 'SET' and key:
                            state[key] = value
                        elif operation == 'DELETE' and key:
                            state.pop(key, None)
                    except Exception as e:
                        log.error(f"Failed to replay WAL entry: {e}")

                log.info(f"Recovered {node_id} state with {len(state)} entries")
                return state
            finally:
                self.recovery_in_progress = False

    def get_status(self) -> Dict[str, Any]:
        """Get recovery status."""
        with self.lock:
            return {
                'recovery_in_progress': self.recovery_in_progress,
                'snapshot_manager_status': self.snapshot_manager.get_status()
            }


class DurabilityManager:
    """
    Overall durability management.
    
    Coordinates:
    - Snapshots
    - WAL
    - Backup
    - Recovery
    """

    def __init__(self, storage_dir: str = "./persistence"):
        """
        Initialize durability manager.
        
        Args:
            storage_dir: Root directory for all persistence
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.snapshot_manager = SnapshotManager(
            os.path.join(storage_dir, "snapshots")
        )
        self.crash_recovery = CrashRecovery(self.snapshot_manager)
        self.backup_manifests: Dict[str, BackupManifest] = {}
        self.lock = threading.RLock()

    def create_backup(self, node_id: str, state: Dict[str, Any]) -> BackupManifest:
        """
        Create full backup.
        
        Args:
            node_id: Node being backed up
            state: State to backup
            
        Returns:
            BackupManifest
        """
        with self.lock:
            backup_id = f"backup_{node_id}_{int(time.time())}"
            timestamp = time.time()

            # Create snapshot
            snapshot = self.snapshot_manager.create_snapshot(node_id, state)

            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                created_at=timestamp,
                node_id=node_id,
                snapshots=[snapshot],
                total_size_bytes=snapshot.compressed_size_bytes or 0
            )

            self.backup_manifests[backup_id] = manifest
            log.info(f"Created backup {backup_id}")

            return manifest

    def list_backups(self, node_id: str) -> List[BackupManifest]:
        """List backups for node."""
        with self.lock:
            backups = [
                m for m in self.backup_manifests.values()
                if m.node_id == node_id
            ]

            return sorted(backups, key=lambda b: b.created_at, reverse=True)

    def get_status(self) -> Dict[str, Any]:
        """Get overall durability status."""
        with self.lock:
            return {
                'storage_dir': self.storage_dir,
                'snapshot_manager': self.snapshot_manager.get_status(),
                'backup_count': len(self.backup_manifests),
                'crash_recovery': self.crash_recovery.get_status()
            }
