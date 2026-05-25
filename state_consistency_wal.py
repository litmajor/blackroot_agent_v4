"""
Write-Ahead Log (WAL) for State Durability
- Guarantees state recovery after node failures
- Implements version vectors for causal consistency
- Persists commands before application to state machine
- Provides consistent snapshots
"""

import json
import time
import threading
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pickle

log = logging.getLogger("WRITE-AHEAD-LOG")


@dataclass
class VersionVector:
    """Vector clock for tracking causal relationships."""
    vector: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str):
        """Increment clock for this node."""
        self.vector[node_id] = self.vector.get(node_id, 0) + 1

    def merge(self, other: 'VersionVector'):
        """Merge with another vector (take maximum)."""
        for node_id, value in other.vector.items():
            self.vector[node_id] = max(self.vector.get(node_id, 0), value)

    def happens_before(self, other: 'VersionVector') -> bool:
        """Check if this vector happened before other."""
        less_than_some = False
        for node_id in set(self.vector.keys()) | set(other.vector.keys()):
            self_val = self.vector.get(node_id, 0)
            other_val = other.vector.get(node_id, 0)

            if self_val > other_val:
                return False  # This is greater, so no
            if self_val < other_val:
                less_than_some = True

        return less_than_some

    def to_dict(self):
        return {'vector': self.vector}

    @staticmethod
    def from_dict(data):
        return VersionVector(vector=data.get('vector', {}))


@dataclass
class WALEntry:
    """Single entry in Write-Ahead Log."""
    entry_id: str
    timestamp: float
    node_id: str
    operation: str  # "SET", "DELETE", "COMMAND", etc.
    key: str
    value: Any
    version_vector: VersionVector
    checksum: str = ""

    def to_dict(self):
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp,
            'node_id': self.node_id,
            'operation': self.operation,
            'key': self.key,
            'value': self.value,
            'version_vector': self.version_vector.to_dict(),
            'checksum': self.checksum
        }


class WriteAheadLog:
    """
    Write-Ahead Log for durability and recovery.
    
    Pattern:
    1. Entry is written to WAL on disk
    2. Entry is acknowledged
    3. Entry is applied to state machine
    """

    def __init__(self, log_dir: str = "./wal_logs", segment_size_mb: int = 100):
        """
        Initialize WAL.
        
        Args:
            log_dir: Directory for WAL files
            segment_size_mb: Maximum size of each log segment
        """
        self.log_dir = log_dir
        self.segment_size_mb = segment_size_mb
        self.segment_size_bytes = segment_size_mb * 1024 * 1024

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        self.entries: List[WALEntry] = []
        self.applied_entries: Dict[str, WALEntry] = {}  # key -> entry
        self.current_segment = 0
        self.lock = threading.RLock()

        self._load_from_disk()
        log.info(f"Initialized WAL with {len(self.entries)} entries")

    def append(self, node_id: str, operation: str, key: str, value: Any,
               version_vector: Optional[VersionVector] = None) -> str:
        """
        Append entry to WAL.
        
        Returns:
            entry_id
        """
        with self.lock:
            entry_id = f"{node_id}:{time.time()}:{len(self.entries)}"
            version_vector = version_vector or VersionVector()

            entry = WALEntry(
                entry_id=entry_id,
                timestamp=time.time(),
                node_id=node_id,
                operation=operation,
                key=key,
                value=value,
                version_vector=version_vector,
                checksum=self._compute_checksum(key, value)
            )

            self.entries.append(entry)
            self._write_entry_to_disk(entry)

            log.debug(f"Appended WAL entry: {entry_id}")
            return entry_id

    def apply_entry(self, entry_id: str, apply_fn):
        """
        Apply WAL entry to state machine.
        
        Args:
            entry_id: ID of entry to apply
            apply_fn: Callable to apply the entry
        """
        with self.lock:
            # Find entry
            entry = None
            for e in self.entries:
                if e.entry_id == entry_id:
                    entry = e
                    break

            if not entry:
                log.warning(f"Entry not found: {entry_id}")
                return False

            try:
                # Apply entry
                apply_fn(entry)
                self.applied_entries[entry.key] = entry
                log.debug(f"Applied WAL entry: {entry_id}")
                return True
            except Exception as e:
                log.error(f"Failed to apply entry {entry_id}: {e}")
                return False

    def create_snapshot(self, node_id: str) -> Dict[str, Any]:
        """
        Create consistent snapshot of current state.
        
        Returns:
            Snapshot dict with state and metadata
        """
        with self.lock:
            snapshot = {
                'node_id': node_id,
                'timestamp': time.time(),
                'entries_count': len(self.entries),
                'applied_count': len(self.applied_entries),
                'applied_state': {
                    key: entry.value
                    for key, entry in self.applied_entries.items()
                },
                'version_vectors': {
                    key: entry.version_vector.to_dict()
                    for key, entry in self.applied_entries.items()
                }
            }

            self._write_snapshot_to_disk(snapshot)
            log.info(f"Created snapshot: {snapshot['timestamp']}")
            return snapshot

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """
        Restore state from snapshot.
        
        Returns:
            True if restoration succeeded
        """
        with self.lock:
            try:
                self.applied_entries = {}
                for key, value in snapshot.get('applied_state', {}).items():
                    entry = WALEntry(
                        entry_id=f"restore:{key}",
                        timestamp=snapshot['timestamp'],
                        node_id=snapshot['node_id'],
                        operation="SET",
                        key=key,
                        value=value,
                        version_vector=VersionVector.from_dict(
                            snapshot.get('version_vectors', {}).get(key, {})
                        )
                    )
                    self.applied_entries[key] = entry

                log.info(f"Restored from snapshot: {len(self.applied_entries)} entries")
                return True
            except Exception as e:
                log.error(f"Snapshot restoration failed: {e}")
                return False

    def get_unapplied_entries(self) -> List[WALEntry]:
        """Get entries that haven't been applied yet."""
        with self.lock:
            applied_ids = {e.entry_id for e in self.applied_entries.values()}
            return [e for e in self.entries if e.entry_id not in applied_ids]

    def get_entry_by_key(self, key: str) -> Optional[WALEntry]:
        """Get latest version of entry for key."""
        with self.lock:
            return self.applied_entries.get(key)

    def _write_entry_to_disk(self, entry: WALEntry):
        """Persist entry to disk."""
        try:
            filename = os.path.join(self.log_dir, f"segment_{self.current_segment}.jsonl")

            with open(filename, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')

            # Check if segment needs rotation
            if os.path.getsize(filename) > self.segment_size_bytes:
                self.current_segment += 1
                log.info(f"Rotated WAL segment to {self.current_segment}")
        except Exception as e:
            log.error(f"Failed to write WAL entry to disk: {e}")

    def _write_snapshot_to_disk(self, snapshot: Dict[str, Any]):
        """Persist snapshot to disk."""
        try:
            filename = os.path.join(self.log_dir, f"snapshot_{int(snapshot['timestamp'])}.json")

            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=2)

            log.info(f"Wrote snapshot: {filename}")
        except Exception as e:
            log.error(f"Failed to write snapshot: {e}")

    def _load_from_disk(self):
        """Load WAL entries from disk on startup."""
        try:
            for filename in sorted(os.listdir(self.log_dir)):
                if filename.startswith('segment_') and filename.endswith('.jsonl'):
                    filepath = os.path.join(self.log_dir, filename)

                    with open(filepath, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                entry = WALEntry(
                                    entry_id=data['entry_id'],
                                    timestamp=data['timestamp'],
                                    node_id=data['node_id'],
                                    operation=data['operation'],
                                    key=data['key'],
                                    value=data['value'],
                                    version_vector=VersionVector.from_dict(data['version_vector']),
                                    checksum=data.get('checksum', '')
                                )
                                self.entries.append(entry)
                            except json.JSONDecodeError as e:
                                log.warning(f"Failed to parse WAL entry: {e}")

            log.info(f"Loaded {len(self.entries)} WAL entries from disk")
        except Exception as e:
            log.error(f"Failed to load WAL from disk: {e}")

    def _compute_checksum(self, key: str, value: Any) -> str:
        """Compute checksum for data integrity."""
        import hashlib
        data = f"{key}:{json.dumps(value, sort_keys=True)}".encode()
        return hashlib.sha256(data).hexdigest()

    def get_status(self) -> Dict[str, Any]:
        """Get WAL status."""
        with self.lock:
            return {
                'total_entries': len(self.entries),
                'applied_entries': len(self.applied_entries),
                'unapplied_entries': len(self.entries) - len(self.applied_entries),
                'current_segment': self.current_segment,
                'log_dir': self.log_dir
            }


class CausalConsistencyManager:
    """
    Ensures causal consistency across distributed state.
    
    Guarantees: If write A happens-before write B, all replicas
    will see A before B (even if reads happen at different nodes).
    """

    def __init__(self, node_id: str):
        """Initialize consistency manager."""
        self.node_id = node_id
        self.version_vector = VersionVector()
        self.version_vectors_by_key: Dict[str, VersionVector] = {}
        self.lock = threading.RLock()

    def record_write(self, key: str, value: Any):
        """Record a local write."""
        with self.lock:
            self.version_vector.increment(self.node_id)
            self.version_vectors_by_key[key] = VersionVector(self.version_vector.vector.copy())

    def record_read(self, version_vector: VersionVector):
        """Record reading from another node."""
        with self.lock:
            # Update vector to at least the read version
            self.version_vector.merge(version_vector)

    def get_version_vector_for_key(self, key: str) -> VersionVector:
        """Get version vector for a key."""
        with self.lock:
            return self.version_vectors_by_key.get(key, VersionVector())

    def is_causally_consistent(self, key: str, remote_vector: VersionVector) -> bool:
        """
        Check if local state is causally consistent with remote version.
        """
        with self.lock:
            local_vector = self.version_vectors_by_key.get(key, VersionVector())

            # Local is consistent if it happened after or equal to remote
            return not remote_vector.happens_before(local_vector)

    def get_read_context(self) -> VersionVector:
        """Get current read context to propagate to other nodes."""
        with self.lock:
            return VersionVector(self.version_vector.vector.copy())

    def get_status(self) -> Dict[str, Any]:
        """Get consistency manager status."""
        with self.lock:
            return {
                'node_id': self.node_id,
                'version_vector': self.version_vector.to_dict(),
                'keys_tracked': len(self.version_vectors_by_key)
            }
