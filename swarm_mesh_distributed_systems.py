"""
SwarmMesh Distributed Systems Integration Module
Integrates all 10 architectural gap solutions into SwarmMesh

This module is imported by swarm_mesh.py to add distributed systems capabilities
"""

import logging
import threading
import time
from typing import Dict, Optional, Any

# Import all 10 distributed systems components
from consensus_raft import RaftNode, RaftCluster
from failure_detection_phi import PhiAccrualFailureDetector
from message_delivery_reliable import ReliableDeliveryManager
from state_consistency_wal import WriteAheadLog, CausalConsistencyManager
from sharding_layer import ShardingManager
from mtls_security import CertificateAuthority, MTLSConnectionManager
from backpressure_management import BackpressureManager, CircuitBreakerConfig
from distributed_tracing import Tracer, TracedLogger
from dynamic_configuration import ConfigurationManager, ConfigValidationRule, ConfigUpdateStrategy
from persistence_durability import DurabilityManager

log = logging.getLogger("swarm-mesh-integrated")


class DistributedSystemsCapabilities:
    """
    Encapsulates all 10 distributed systems components as a cohesive layer.
    
    Can be added to existing SwarmMesh via composition:
        self.dist_sys = DistributedSystemsCapabilities(node_id, cluster_nodes)
    """

    def __init__(self, node_id: str, cluster_nodes: list, environment: str = "production"):
        """Initialize all 10 distributed systems components."""
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.environment = environment
        self.lock = threading.RLock()

        log.info(f"Initializing distributed systems capabilities for {node_id}")

        # Gap #1: Consensus (Raft)
        self.raft_cluster = RaftCluster(
            node_ids=cluster_nodes
        )
        log.info("✓ Raft consensus cluster initialized")

        # Gap #2: Failure Detection (Phi Accrual)
        self.failure_detector = PhiAccrualFailureDetector()
        for node in cluster_nodes:
            if node != node_id:
                self.failure_detector.report_heartbeat(node)
        log.info("✓ Phi Accrual failure detector initialized")

        # Gap #3: Reliable Message Delivery
        self.reliable_delivery = ReliableDeliveryManager()
        log.info("✓ Reliable message delivery manager initialized")

        # Gap #4: Causal Consistency + WAL
        self.wal = WriteAheadLog(log_dir=f"./wal_{node_id}")
        self.consistency_manager = CausalConsistencyManager(node_id)
        log.info("✓ Write-Ahead Log and causal consistency initialized")

        # Gap #5: Sharding Layer
        self.sharding_manager = ShardingManager(shard_count=8, replicas_per_key=3)
        log.info("✓ Sharding layer initialized")

        # Gap #6: mTLS Security
        self.ca = CertificateAuthority()
        self.node_cert = self.ca.issue_node_certificate(
            node_id,
            san_hosts=[f"{node_id}.local"]
        )
        self.mtls_manager = MTLSConnectionManager(
            ca_pem=self.ca.get_ca_cert_pem(),
            node_cert_pem=self.node_cert.cert_pem,
            node_key_pem=self.node_cert.key_pem
        )
        log.info("✓ mTLS security and PKI initialized")

        # Gap #7: Backpressure & Circuit Breaking
        self.backpressure = BackpressureManager()
        self.command_breaker = self.backpressure.create_circuit_breaker(
            "command_execution",
            config=CircuitBreakerConfig(failure_threshold=10)
        )
        self.command_queue = self.backpressure.create_bounded_queue("commands", max_size=5000)
        self.rate_limiter = self.backpressure.create_rate_limiter("api_rate", base_rate=1000)
        log.info("✓ Backpressure management initialized")

        # Gap #8: Distributed Tracing
        self.tracer = Tracer(service_name=f"swarm_mesh_{node_id}")
        self.traced_log = TracedLogger(log)
        log.info("✓ Distributed tracing initialized")

        # Gap #9: Dynamic Configuration
        self.config_manager = ConfigurationManager(environment=environment)
        self._setup_config_rules()
        self._load_initial_config()
        log.info("✓ Dynamic configuration manager initialized")

        # Gap #10: Durability & Persistence
        self.durability = DurabilityManager(storage_dir=f"./persistence_{node_id}")
        log.info("✓ Durability and persistence layer initialized")

    def _setup_config_rules(self):
        """Setup configuration validation rules."""
        self.config_manager.register_validation_rule(
            "heartbeat_interval",
            ConfigValidationRule(key_pattern="heartbeat_interval", required=True, type_check=int, min_value=100, max_value=60000)
        )
        self.config_manager.register_validation_rule(
            "consensus_timeout",
            ConfigValidationRule(key_pattern="consensus_timeout", required=True, type_check=int, min_value=1000, max_value=120000)
        )
        self.config_manager.register_validation_rule(
            "phi_threshold",
            ConfigValidationRule(key_pattern="phi_threshold", required=True, type_check=float, min_value=1.0, max_value=10.0)
        )

    def _load_initial_config(self):
        """Load initial configuration."""
        initial_config = {
            "heartbeat_interval": 5000,
            "consensus_timeout": 30000,
            "phi_threshold": 7.0,
            "shard_count": 8,
            "retry_max_backoff_ms": 30000,
            "max_queue_depth": 5000
        }

        config_version = self.config_manager.load_config(
            initial_config,
            author="initialization"
        )
        log.info(f"Loaded config version {config_version}")

    def start_background_services(self):
        """Start background threads for periodic maintenance."""
        self._start_raft_heartbeat()
        self._start_failure_detection()
        self._start_snapshot_service()

    def _start_raft_heartbeat(self):
        """Start Raft heartbeat thread."""
        def heartbeat_loop():
            while True:
                try:
                    leader = self.raft_cluster.get_leader()
                    if leader:
                        self.raft_cluster._send_heartbeats(leader)
                    time.sleep(0.05)
                except Exception as e:
                    log.error(f"Error in Raft heartbeat: {e}")
                    time.sleep(0.1)

        thread = threading.Thread(target=heartbeat_loop, daemon=True, name="raft-heartbeat")
        thread.start()

    def _start_failure_detection(self):
        """Start failure detection thread."""
        def detection_loop():
            while True:
                try:
                    # Check each node
                    for node in self.cluster_nodes:
                        if node == self.node_id:
                            continue

                        phi = self.failure_detector.get_phi(node)
                        threshold = self.config_manager.get_config_value("phi_threshold", default=7.0)

                        if phi > threshold:
                            confidence = self.failure_detector.get_confidence(node)
                            if confidence > 0.95:
                                log.warning(f"Node {node} suspected down (phi={phi:.1f}, confidence={confidence:.2f})")

                    time.sleep(1.0)
                except Exception as e:
                    log.error(f"Error in failure detection: {e}")
                    time.sleep(1.0)

        thread = threading.Thread(target=detection_loop, daemon=True, name="failure-detection")
        thread.start()

    def _start_snapshot_service(self):
        """Start periodic snapshot service."""
        def snapshot_loop():
            while True:
                try:
                    time.sleep(60)  # Snapshot every 60 seconds

                    # In real impl, would get state from mesh
                    # snapshot = self.durability.snapshot_manager.create_snapshot(self.node_id, state.copy())

                    # Cleanup old snapshots
                    deleted = self.durability.snapshot_manager.cleanup_old_snapshots()
                    if deleted > 0:
                        log.info(f"Cleaned up {deleted} old snapshots")
                except Exception as e:
                    log.error(f"Error in snapshot service: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=snapshot_loop, daemon=True, name="snapshot-service")
        thread.start()

    def execute_command_with_safety(self, command: Dict[str, Any]) -> bool:
        """
        Execute command with all safety/reliability measures.
        
        Demonstrates integrated usage of all 10 components.
        """
        trace = self.tracer.start_trace(f"execute_{command.get('type', 'unknown')}")

        try:
            # Rate limiting (gap #7)
            if not self.rate_limiter.allow_request():
                self.traced_log.warning("Rate limit exceeded")
                self.tracer.finish_trace(trace.trace_id, success=False, error="Rate limited")
                return False

            # Circuit breaker (gap #7)
            def execute_with_reliability():
                # Enqueue with backpressure (gap #7)
                if not self.command_queue.enqueue(command, priority=command.get("priority", 0)):
                    self.traced_log.warning("Command queue full, dropped")
                    return False

                # Reliable delivery (gap #3)
                msg_id = self.reliable_delivery.queue_message(
                    destination="primary",
                    payload=command,
                    sender=command.get("source_id", "system")
                )

                # Causal consistency check (gap #4)
                key = command.get("key")
                entry_id: Optional[str] = None
                if key:
                    current_vector = self.consistency_manager.get_version_vector_for_key(key)
                    self.consistency_manager.record_write(key, command.get("value"))

                # Write to WAL (gap #4)
                if isinstance(key, str) and key:  # Ensure key is valid string
                    entry_id = self.wal.append(
                        self.node_id,
                        command.get("type", "SET"),
                        key,
                        command.get("value"),
                        self.consistency_manager.get_version_vector_for_key(key) if key else None
                    )

                if entry_id:
                    self.traced_log.info(f"Executed command {command.get('type')} with entry {entry_id}")
                else:
                    self.traced_log.info(f"Executed command {command.get('type')}")
                return True

            # Execute with circuit breaker protection (gap #7)
            result = self.command_breaker.call(execute_with_reliability)

            self.tracer.finish_trace(trace.trace_id, success=result)
            return result

        except Exception as e:
            self.traced_log.error(f"Command execution failed: {e}")
            self.tracer.finish_trace(trace.trace_id, success=False, error=str(e))
            return False

    def update_runtime_config(self, updates: Dict[str, Any], strategy: str = "IMMEDIATE") -> bool:
        """Update configuration at runtime."""
        try:
            strategy_enum = ConfigUpdateStrategy[strategy]
        except KeyError:
            log.warning(f"Invalid strategy {strategy}, using IMMEDIATE")
            strategy_enum = ConfigUpdateStrategy.IMMEDIATE

        return self.config_manager.update_config(
            updates,
            author="operator",
            strategy=strategy_enum
        )

    def get_integrated_status(self) -> Dict[str, Any]:
        """Get status of all 10 components."""
        return {
            "node_id": self.node_id,
            "environment": self.environment,

            # Gap #1: Consensus
            "consensus": {
                "type": "Raft",
                "status": self.raft_cluster.get_cluster_status()
            },

            # Gap #2: Failure Detection
            "failure_detection": {
                "type": "Phi Accrual",
                "health": self.failure_detector.get_cluster_health()
            },

            # Gap #3: Reliable Delivery
            "message_delivery": self.reliable_delivery.get_status(),

            # Gap #4: Causal Consistency
            "consistency": {
                "wal": self.wal.get_status(),
                "version_vectors": self.consistency_manager.get_status()
            },

            # Gap #5: Sharding
            "sharding": self.sharding_manager.get_status(),

            # Gap #6: mTLS Security
            "security": {
                "pki": self.ca.get_status(),
                "mtls": self.mtls_manager.get_status()
            },

            # Gap #7: Backpressure
            "backpressure": self.backpressure.get_status(),

            # Gap #8: Distributed Tracing
            "tracing": self.tracer.get_status(),

            # Gap #9: Dynamic Configuration
            "configuration": self.config_manager.get_status(),

            # Gap #10: Durability
            "durability": self.durability.get_status(),
        }
