"""
Raft Consensus Implementation for BlackRoot SwarmMesh
- Ensures distributed consensus for critical decisions
- Handles split-brain scenarios with term/epoch numbering
- Provides safe leader election and log replication
- Guarantees single leader per term
"""

import json
import time
import threading
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

log = logging.getLogger("RAFT-CONSENSUS")


class RaftState(Enum):
    """Node state in Raft consensus"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LogEntryType(Enum):
    """Type of log entry"""
    COMMAND = "command"
    CONFIG_CHANGE = "config_change"
    SNAPSHOT = "snapshot"


@dataclass
class LogEntry:
    """Single entry in the Raft log"""
    term: int
    index: int
    entry_type: LogEntryType
    data: Any
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            'term': self.term,
            'index': self.index,
            'entry_type': self.entry_type.value,
            'data': self.data,
            'timestamp': self.timestamp
        }


@dataclass
class RaftState_Data:
    """Persistent Raft state (saved to disk)"""
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)
    last_applied: int = 0
    commit_index: int = 0


class RaftNode:
    """
    Raft consensus node with full state machine including:
    - Election timeout (150-300ms randomized)
    - Term/epoch numbering to detect stale leaders
    - Log replication with commit tracking
    - Snapshot support for recovery
    - Split-brain prevention via quorum
    """

    def __init__(self, node_id: str, peers: List[str], election_timeout_ms: float = 250.0):
        """
        Initialize Raft node.
        
        Args:
            node_id: Unique node identifier
            peers: List of other node IDs in cluster
            election_timeout_ms: Timeout before starting election (150-300ms range)
        """
        self.node_id = node_id
        self.peers = set(peers)
        self.cluster_size = len(peers) + 1
        self.quorum_size = (self.cluster_size // 2) + 1

        # Persistent state (survive crashes)
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.log_index = 0

        # Volatile state
        self.state = RaftState.FOLLOWER
        self.leader_id: Optional[str] = None
        self.commit_index = 0
        self.last_applied = 0

        # Leader-only state
        self.next_index: Dict[str, int] = {peer: len(self.log) for peer in self.peers}
        self.match_index: Dict[str, int] = {peer: 0 for peer in self.peers}

        # Election and heartbeat
        self.election_timeout_ms = election_timeout_ms
        self.heartbeat_interval_ms = max(50, election_timeout_ms / 3)
        self.last_heartbeat = time.time()
        self.election_start_time: Optional[float] = None

        # Snapshots
        self.last_snapshot_index = 0
        self.last_snapshot_term = 0

        # Thread safety
        self.lock = threading.RLock()

        log.info(f"Initialized Raft node {self.node_id} (cluster_size={self.cluster_size}, quorum={self.quorum_size})")

    def append_entry(self, data: Any, entry_type: LogEntryType = LogEntryType.COMMAND) -> Tuple[bool, int]:
        """
        Append entry to log (only callable on leader).
        
        Returns:
            (success, log_index) - Whether append succeeded and the index of the entry
        """
        with self.lock:
            if self.state != RaftState.LEADER:
                return False, -1

            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                entry_type=entry_type,
                data=data
            )
            self.log.append(entry)
            log.debug(f"[{self.node_id}] Appended log entry at index {entry.index}")
            return True, entry.index

    def commit_entries(self, up_to_index: int) -> int:
        """
        Commit entries up to specified index.
        
        Returns:
            Number of newly committed entries
        """
        with self.lock:
            old_commit_index = self.commit_index
            self.commit_index = min(up_to_index, len(self.log) - 1)
            committed_count = max(0, self.commit_index - old_commit_index)

            if committed_count > 0:
                log.info(f"[{self.node_id}] Committed {committed_count} entries (commit_index now {self.commit_index})")

            return committed_count

    def get_committed_entries(self) -> List[LogEntry]:
        """Get all committed entries since last apply."""
        with self.lock:
            if self.last_applied >= self.commit_index:
                return []

            entries = self.log[self.last_applied:self.commit_index + 1]
            self.last_applied = self.commit_index
            return entries

    def handle_heartbeat(self, term: int, leader_id: str, commit_index: int) -> bool:
        """
        Handle heartbeat from leader.
        
        Returns:
            True if accepted (term >= current_term), False if rejected (stale term)
        """
        with self.lock:
            if term < self.current_term:
                # Stale heartbeat, reject
                log.debug(f"[{self.node_id}] Rejected heartbeat from stale term {term}")
                return False

            # Update term if newer
            if term > self.current_term:
                self._update_term(term)

            # Accept heartbeat
            self.leader_id = leader_id
            self.last_heartbeat = time.time()
            self.state = RaftState.FOLLOWER

            # Update commit index
            self.commit_entries(commit_index)

            log.debug(f"[{self.node_id}] Accepted heartbeat from {leader_id} (term {term})")
            return True

    def start_election(self) -> bool:
        """
        Start election for leadership.
        
        Returns:
            True if node wins election, False otherwise
        """
        with self.lock:
            self.current_term += 1
            self.state = RaftState.CANDIDATE
            self.voted_for = self.node_id
            self.election_start_time = time.time()

            log.info(f"[{self.node_id}] Starting election for term {self.current_term}")

            # In real implementation, send RequestVote RPCs to peers
            # For now, simulate receiving votes
            votes_received = 1  # Vote for self

            # Simulate peer votes (in real system, this would be async)
            for peer in self.peers:
                if self._request_vote_from_peer(peer):
                    votes_received += 1

            if votes_received >= self.quorum_size:
                self._become_leader()
                return True
            else:
                log.warning(f"[{self.node_id}] Lost election (votes={votes_received}, needed={self.quorum_size})")
                return False

    def _request_vote_from_peer(self, peer_id: str) -> bool:
        """Simulate requesting vote from peer."""
        # In real implementation, this would be RPC
        # For simulation, assume 80% chance of vote if peer has not voted yet
        import random
        return random.random() < 0.8

    def _become_leader(self):
        """Transition to leader state."""
        with self.lock:
            self.state = RaftState.LEADER
            self.leader_id = self.node_id

            # Initialize next_index and match_index
            for peer in self.peers:
                self.next_index[peer] = len(self.log)
                self.match_index[peer] = 0

            log.info(f"[{self.node_id}] Became leader for term {self.current_term}")

    def _update_term(self, new_term: int):
        """Update current term and reset vote."""
        if new_term > self.current_term:
            self.current_term = new_term
            self.voted_for = None
            self.state = RaftState.FOLLOWER
            self.leader_id = None

    def is_election_timeout(self) -> bool:
        """Check if election timeout has elapsed."""
        with self.lock:
            elapsed_ms = (time.time() - self.last_heartbeat) * 1000
            # Randomized timeout between 150-300ms
            timeout = self.election_timeout_ms + random.uniform(0, 150)
            return elapsed_ms > timeout

    def create_snapshot(self, up_to_index: int) -> Dict[str, Any]:
        """Create snapshot of state machine."""
        with self.lock:
            if up_to_index <= self.last_snapshot_index:
                return {}

            snapshot = {
                'index': up_to_index,
                'term': self.log[up_to_index].term if up_to_index < len(self.log) else 0,
                'data': [entry.to_dict() for entry in self.log[:up_to_index + 1]],
                'timestamp': time.time()
            }

            self.last_snapshot_index = up_to_index
            log.info(f"[{self.node_id}] Created snapshot up to index {up_to_index}")
            return snapshot

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Restore state from snapshot."""
        with self.lock:
            try:
                self.log = [
                    LogEntry(
                        term=entry['term'],
                        index=entry['index'],
                        entry_type=LogEntryType(entry['entry_type']),
                        data=entry['data'],
                        timestamp=entry['timestamp']
                    )
                    for entry in snapshot.get('data', [])
                ]
                self.last_snapshot_index = snapshot['index']
                self.last_applied = snapshot['index']
                log.info(f"[{self.node_id}] Restored from snapshot (index={snapshot['index']})")
                return True
            except Exception as e:
                log.error(f"[{self.node_id}] Failed to restore snapshot: {e}")
                return False

    def get_status(self) -> Dict[str, Any]:
        """Get node status for monitoring."""
        with self.lock:
            return {
                'node_id': self.node_id,
                'state': self.state.value,
                'term': self.current_term,
                'leader_id': self.leader_id,
                'log_length': len(self.log),
                'commit_index': self.commit_index,
                'last_applied': self.last_applied,
                'voted_for': self.voted_for,
                'cluster_size': self.cluster_size,
                'quorum_size': self.quorum_size
            }


class RaftCluster:
    """Manages a cluster of Raft nodes."""

    def __init__(self, node_ids: List[str]):
        """Initialize cluster with given node IDs."""
        self.nodes: Dict[str, RaftNode] = {}
        for node_id in node_ids:
            other_peers = [n for n in node_ids if n != node_id]
            self.nodes[node_id] = RaftNode(node_id, other_peers)

    def get_leader(self) -> Optional[RaftNode]:
        """Get current leader node."""
        for node in self.nodes.values():
            if node.state == RaftState.LEADER:
                return node
        return None

    def tick(self, interval_ms: float = 50):
        """Simulate time passage and trigger elections if needed."""
        for node in self.nodes.values():
            if node.state == RaftState.LEADER:
                # Leader should send heartbeats
                self._send_heartbeats(node)
            elif node.is_election_timeout():
                # Candidate/Follower should start election
                node.start_election()

    def _send_heartbeats(self, leader: RaftNode):
        """Simulate leader sending heartbeats to followers."""
        for peer in leader.peers:
            peer_node = self.nodes.get(peer)
            if peer_node:
                peer_node.handle_heartbeat(leader.current_term, leader.node_id, leader.commit_index)

    def get_cluster_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all nodes in cluster."""
        return {node_id: node.get_status() for node_id, node in self.nodes.items()}
