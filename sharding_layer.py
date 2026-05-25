"""
Redis Sharding Strategy for Scalability
- Consistent hashing for key distribution
- Shard leadership for failover
- Cross-shard gossip protocol
- Load balancing between shards
"""

import hashlib
import logging
import threading
import json
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict

log = logging.getLogger("SHARDING-LAYER")


@dataclass
class ShardNode:
    """Individual shard node in cluster."""
    shard_id: str
    node_id: str
    redis_host: str
    redis_port: int
    is_leader: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    keys_count: int = 0
    memory_usage: int = 0


@dataclass
class GossipMessage:
    """Gossip protocol message for cross-shard communication."""
    message_id: str
    sender_shard: str
    timestamp: float
    message_type: str  # "KEYSTATE", "LOAD", "HEALTH"
    data: Dict[str, Any]
    ttl: int = 10  # hops


class ConsistentHashRing:
    """
    Consistent hashing for distributed key-value storage.
    
    Properties:
    - Minimal key redistribution on node addition/removal
    - Load balancing across nodes
    - Replica placement for fault tolerance
    """

    def __init__(self, replicas: int = 3, virtual_nodes: int = 160):
        """
        Initialize hash ring.
        
        Args:
            replicas: Number of replica placements per key
            virtual_nodes: Virtual nodes per physical node (for load balancing)
        """
        self.replicas = replicas
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}  # hash -> node_id
        self.sorted_keys = []
        self.nodes: Set[str] = set()
        self.lock = threading.RLock()

    def add_node(self, node_id: str):
        """Add node to ring."""
        with self.lock:
            if node_id in self.nodes:
                return

            self.nodes.add(node_id)

            # Add virtual nodes
            for i in range(self.virtual_nodes):
                virtual_key = f"{node_id}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = node_id

            self.sorted_keys = sorted(self.ring.keys())
            log.info(f"Added node {node_id} with {self.virtual_nodes} virtual nodes")

    def remove_node(self, node_id: str):
        """Remove node from ring."""
        with self.lock:
            if node_id not in self.nodes:
                return

            self.nodes.remove(node_id)

            # Remove virtual nodes
            keys_to_remove = [k for k, v in self.ring.items() if v == node_id]
            for key in keys_to_remove:
                del self.ring[key]

            self.sorted_keys = sorted(self.ring.keys())
            log.info(f"Removed node {node_id}")

    def get_node_for_key(self, key: str) -> Optional[str]:
        """Get primary node for key."""
        if not self.ring:
            return None

        hash_value = self._hash(key)
        return self._find_node(hash_value)

    def get_replica_nodes(self, key: str) -> List[str]:
        """Get replica nodes for key (for fault tolerance)."""
        if not self.ring or not self.nodes:
            return []

        hash_value = self._hash(key)
        replicas = []
        seen = set()

        # Find primary and replicas
        idx = self._find_node_index(hash_value)
        for _ in range(self.replicas * 2):  # Search enough to find replicas
            if idx >= len(self.sorted_keys):
                idx = 0

            node = self.ring[self.sorted_keys[idx]]
            if node not in seen:
                replicas.append(node)
                seen.add(node)
                if len(replicas) >= self.replicas:
                    break

            idx += 1

        return replicas

    def _hash(self, key: str) -> int:
        """Hash key to position on ring."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _find_node(self, hash_value: int) -> Optional[str]:
        """Find node clockwise from hash value."""
        if not self.sorted_keys:
            return None

        # Binary search for closest node clockwise
        for key in self.sorted_keys:
            if hash_value <= key:
                return self.ring[key]

        # Wrap around
        return self.ring[self.sorted_keys[0]]

    def _find_node_index(self, hash_value: int) -> int:
        """Find index of node clockwise from hash value."""
        if not self.sorted_keys:
            return 0

        for i, key in enumerate(self.sorted_keys):
            if hash_value <= key:
                return i

        return 0

    def get_status(self) -> Dict[str, Any]:
        """Get ring status."""
        with self.lock:
            return {
                'nodes': len(self.nodes),
                'ring_entries': len(self.ring),
                'replicas': self.replicas,
                'virtual_nodes_per_node': self.virtual_nodes
            }


class ShardingManager:
    """
    Manages sharding layer for horizontal scalability.
    
    Strategy:
    - Consistent hashing distributes keys across shards
    - Each shard can have leader/replicas
    - Cross-shard gossip disseminates load info
    """

    def __init__(self, shard_count: int = 8, replicas_per_key: int = 3):
        """
        Initialize sharding manager.
        
        Args:
            shard_count: Number of shards
            replicas_per_key: Replica count per key
        """
        self.shard_count = shard_count
        self.replicas_per_key = replicas_per_key
        self.hash_ring = ConsistentHashRing(replicas=replicas_per_key)
        self.shards: Dict[str, ShardNode] = {}
        self.gossip_state: Dict[str, GossipMessage] = {}  # message_id -> message
        self.shard_loads: Dict[str, float] = defaultdict(float)  # shard_id -> load %
        self.lock = threading.RLock()

        self._initialize_shards()
        log.info(f"Initialized sharding manager with {shard_count} shards")

    def _initialize_shards(self):
        """Initialize shard nodes."""
        for i in range(self.shard_count):
            shard_id = f"shard_{i}"
            node_id = f"shard_node_{i}"
            shard = ShardNode(
                shard_id=shard_id,
                node_id=node_id,
                redis_host=f"127.0.0.1",
                redis_port=6379 + i,
                is_leader=True
            )
            self.shards[node_id] = shard
            self.hash_ring.add_node(node_id)

    def get_shard_for_key(self, key: str) -> Optional[ShardNode]:
        """Get primary shard for key."""
        with self.lock:
            node_id = self.hash_ring.get_node_for_key(key)
            if node_id:
                return self.shards.get(node_id)
            return None

    def get_replica_shards(self, key: str) -> List[ShardNode]:
        """Get replica shards for key."""
        with self.lock:
            replica_nodes = self.hash_ring.get_replica_nodes(key)
            return [self.shards[n] for n in replica_nodes if n in self.shards]

    def record_shard_load(self, shard_id: str, load_percent: float):
        """Record shard load percentage."""
        with self.lock:
            self.shard_loads[shard_id] = load_percent

    def get_least_loaded_shard(self) -> Optional[ShardNode]:
        """Get least loaded shard for balancing."""
        with self.lock:
            min_load = float('inf')
            least_loaded = None

            for node_id, shard in self.shards.items():
                if shard.is_leader:
                    load = self.shard_loads.get(shard.shard_id, 0)
                    if load < min_load:
                        min_load = load
                        least_loaded = shard

            return least_loaded

    def add_shard(self, node_id: str, redis_host: str, redis_port: int) -> ShardNode:
        """Add new shard for horizontal scaling."""
        with self.lock:
            # Generate shard ID
            shard_id = f"shard_{len(self.shards)}"

            shard = ShardNode(
                shard_id=shard_id,
                node_id=node_id,
                redis_host=redis_host,
                redis_port=redis_port,
                is_leader=True
            )

            self.shards[node_id] = shard
            self.hash_ring.add_node(node_id)

            log.info(f"Added shard: {shard_id} ({node_id})")
            return shard

    def remove_shard(self, node_id: str) -> bool:
        """Remove shard (data will rehash to others)."""
        with self.lock:
            if node_id not in self.shards:
                return False

            shard = self.shards.pop(node_id)
            self.hash_ring.remove_node(node_id)

            log.info(f"Removed shard: {shard.shard_id}")
            return True

    def broadcast_gossip(self, message_type: str, data: Dict[str, Any],
                        sender_shard: str, ttl: int = 10) -> str:
        """
        Broadcast gossip message across shards.
        
        This enables cross-shard awareness of:
        - Key state changes
        - Load information
        - Health status
        """
        with self.lock:
            message_id = f"gossip_{sender_shard}_{int(time.time()*1000)}"
            message = GossipMessage(
                message_id=message_id,
                sender_shard=sender_shard,
                timestamp=time.time(),
                message_type=message_type,
                data=data,
                ttl=ttl
            )

            self.gossip_state[message_id] = message
            log.debug(f"Broadcast gossip: {message_type} from {sender_shard}")
            return message_id

    def get_gossip_message(self, message_id: str) -> Optional[GossipMessage]:
        """Get gossip message by ID."""
        with self.lock:
            return self.gossip_state.get(message_id)

    def propagate_gossip(self) -> List[str]:
        """
        Propagate gossip messages with TTL decrement.
        
        Returns:
            Expired message IDs
        """
        with self.lock:
            expired = []

            for msg_id, message in list(self.gossip_state.items()):
                message.ttl -= 1

                if message.ttl <= 0:
                    expired.append(msg_id)
                    del self.gossip_state[msg_id]
                    log.debug(f"Expired gossip: {msg_id}")

            return expired

    def get_key_distribution(self) -> Dict[str, Any]:
        """Get key distribution across shards."""
        with self.lock:
            distribution = {}

            for node_id, shard in self.shards.items():
                distribution[shard.shard_id] = {
                    'keys_count': shard.keys_count,
                    'memory_usage': shard.memory_usage,
                    'load_percent': self.shard_loads.get(shard.shard_id, 0),
                    'is_leader': shard.is_leader
                }

            return distribution

    def rebalance_check(self, max_load_diff: float = 30.0) -> bool:
        """
        Check if rebalancing is needed.
        
        Args:
            max_load_diff: Maximum allowed load difference percentage
            
        Returns:
            True if rebalancing recommended
        """
        with self.lock:
            loads = [l for l in self.shard_loads.values()]

            if not loads:
                return False

            max_load = max(loads)
            min_load = min(loads)

            return (max_load - min_load) > max_load_diff

    def get_status(self) -> Dict[str, Any]:
        """Get sharding status."""
        with self.lock:
            return {
                'shard_count': len(self.shards),
                'replicas_per_key': self.replicas_per_key,
                'active_shards': sum(1 for s in self.shards.values() if s.is_leader),
                'gossip_messages': len(self.gossip_state),
                'hash_ring': self.hash_ring.get_status(),
                'rebalancing_needed': self.rebalance_check()
            }
