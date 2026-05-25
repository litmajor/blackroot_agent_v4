"""
Phi Accrual Failure Detection for BlackRoot SwarmMesh
- Detects node failures based on heartbeat intervals and historical variance
- Adapts suspicion level based on network conditions
- Prevents false positives in high-latency scenarios
- Provides confidence scores for failure detection
"""

import math
import time
import statistics
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging

log = logging.getLogger("PHI-ACCRUAL-FD")


@dataclass
class HeartbeatHistory:
    """Track heartbeat history for a node."""
    node_id: str
    heartbeats: deque = field(default_factory=lambda: deque(maxlen=100))  # Last 100 heartbeats
    last_heartbeat_time: float = 0.0
    suspected_down_time: Optional[float] = None
    confidence: float = 0.0  # 0-1, where 1 = definitely down

    def add_heartbeat(self):
        """Record a heartbeat."""
        now = time.time()
        if self.last_heartbeat_time > 0:
            interval = now - self.last_heartbeat_time
            self.heartbeats.append(interval)
        self.last_heartbeat_time = now
        self.suspected_down_time = None
        self.confidence = 0.0

    def get_interval_stats(self) -> Tuple[float, float]:
        """Get mean and stddev of heartbeat intervals."""
        if len(self.heartbeats) < 2:
            return 1.0, 0.5  # Default: mean=1s, stddev=0.5s

        intervals = list(self.heartbeats)
        mean = statistics.mean(intervals)
        stddev = statistics.stdev(intervals) if len(intervals) > 1 else 0.1
        return mean, stddev

    def is_suspected_down(self, phi_threshold: float = 1.0) -> bool:
        """Check if node is suspected to be down."""
        return self.get_phi() > phi_threshold

    def get_phi(self) -> float:
        """
        Calculate phi accrual value.
        phi = -log10(probability that node is alive)
        
        Returns:
            phi value (higher = more likely dead)
            - phi < 1.0: probably alive
            - phi = 1.0: 90% confidence node is dead
            - phi > 2.0: 99% confidence node is dead
        """
        if self.last_heartbeat_time == 0:
            return 0.0

        now = time.time()
        time_since_heartbeat = now - self.last_heartbeat_time
        mean, stddev = self.get_interval_stats()

        # More variance = need longer timeout before suspicion
        # Accounts for network jitter and GC pauses
        threshold = mean + (3 * stddev)

        if time_since_heartbeat < threshold:
            # Very recent heartbeat, definitely alive
            return 0.0

        # Use exponential model for failure detection
        # P(alive) = exp(-time_since_heartbeat / (mean * scale_factor))
        scale_factor = max(1.0, stddev / mean) if mean > 0 else 1.0
        p_alive = math.exp(-time_since_heartbeat / (mean * scale_factor * 2))

        # Convert to phi: phi = -log10(p_alive)
        if p_alive > 0.0001:
            phi = -math.log10(p_alive)
        else:
            phi = 5.0  # Effectively certain to be dead

        return max(0.0, phi)


class PhiAccrualFailureDetector:
    """
    Implements Phi Accrual Failure Detection algorithm.
    
    Key features:
    - Adaptive timeout based on historical variance
    - Confidence scores (0-1) for failure likelihood
    - Handles network jitter and temporary slowdowns
    - Statistical approach rather than hard timeouts
    """

    def __init__(self, phi_threshold: float = 1.0, adaptive: bool = True):
        """
        Initialize failure detector.
        
        Args:
            phi_threshold: Phi value above which node is suspected down
                - 1.0 = ~90% confidence
                - 2.0 = ~99% confidence
            adaptive: Whether to adjust threshold based on network conditions
        """
        self.phi_threshold = phi_threshold
        self.adaptive = adaptive
        self.heartbeats: Dict[str, HeartbeatHistory] = {}
        self.network_jitter_factor = 1.0  # Multiplier based on observed network conditions

    def report_heartbeat(self, node_id: str):
        """Record heartbeat from a node."""
        if node_id not in self.heartbeats:
            self.heartbeats[node_id] = HeartbeatHistory(node_id=node_id)

        history = self.heartbeats[node_id]
        history.add_heartbeat()
        log.debug(f"Heartbeat from {node_id}, intervals: {list(history.heartbeats)}")

    def is_suspected_down(self, node_id: str) -> bool:
        """Check if node is suspected to be down."""
        if node_id not in self.heartbeats:
            return False

        history = self.heartbeats[node_id]
        phi = history.get_phi()
        threshold = self.phi_threshold * self.network_jitter_factor

        is_suspected = phi > threshold
        history.confidence = min(1.0, phi / 3.0)  # Normalize to 0-1

        if is_suspected and not history.suspected_down_time:
            history.suspected_down_time = time.time()
            log.warning(f"Node {node_id} suspected down (phi={phi:.2f}, threshold={threshold:.2f})")

        return is_suspected

    def get_phi(self, node_id: str) -> float:
        """Get raw phi value for a node (for monitoring)."""
        if node_id not in self.heartbeats:
            return 0.0
        return self.heartbeats[node_id].get_phi()

    def get_confidence(self, node_id: str) -> float:
        """Get confidence score (0-1) that node is down."""
        if node_id not in self.heartbeats:
            return 0.0
        return self.heartbeats[node_id].confidence

    def get_status(self, node_id: str) -> Dict[str, Any]:
        """Get detailed status for a node."""
        if node_id not in self.heartbeats:
            return {'node_id': node_id, 'status': 'unknown'}

        history = self.heartbeats[node_id]
        mean, stddev = history.get_interval_stats()

        return {
            'node_id': node_id,
            'phi': history.get_phi(),
            'confidence': history.confidence,
            'suspected_down': self.is_suspected_down(node_id),
            'heartbeat_mean_ms': mean * 1000,
            'heartbeat_stddev_ms': stddev * 1000,
            'time_since_heartbeat_ms': (time.time() - history.last_heartbeat_time) * 1000,
            'heartbeat_count': len(history.heartbeats),
            'suspected_since': history.suspected_down_time
        }

    def update_network_conditions(self, avg_latency_ms: float):
        """
        Update network jitter factor based on observed latency.
        Higher latency = higher jitter factor to avoid false positives.
        """
        if avg_latency_ms < 10:
            self.network_jitter_factor = 1.0
        elif avg_latency_ms < 50:
            self.network_jitter_factor = 1.2
        elif avg_latency_ms < 100:
            self.network_jitter_factor = 1.5
        else:
            self.network_jitter_factor = 2.0

        log.info(f"Updated network jitter factor: {self.network_jitter_factor} (latency={avg_latency_ms}ms)")

    def get_cluster_health(self) -> Dict[str, Any]:
        """Get health status of entire cluster."""
        suspected_down = []
        suspected_up = []

        for node_id, history in self.heartbeats.items():
            if self.is_suspected_down(node_id):
                suspected_down.append(node_id)
            else:
                suspected_up.append(node_id)

        return {
            'total_nodes': len(self.heartbeats),
            'suspected_up': len(suspected_up),
            'suspected_down': len(suspected_down),
            'down_nodes': suspected_down,
            'network_jitter_factor': self.network_jitter_factor,
            'down_percentage': (len(suspected_down) / len(self.heartbeats) * 100) if self.heartbeats else 0
        }

    def reset_node(self, node_id: str):
        """Reset suspected state for a node (e.g., after recovery)."""
        if node_id in self.heartbeats:
            self.heartbeats[node_id].suspected_down_time = None
            self.heartbeats[node_id].confidence = 0.0
            log.info(f"Reset suspected state for {node_id}")

    def forget_node(self, node_id: str):
        """Forget about a node (e.g., after permanent failure)."""
        if node_id in self.heartbeats:
            del self.heartbeats[node_id]
            log.info(f"Forgot node {node_id}")
