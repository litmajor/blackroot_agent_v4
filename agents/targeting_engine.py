import os
import json
import time
import threading
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field, asdict

from agents.base import BaseAgent
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event
from core.task_runner import TaskRunner  # For daemon optimization (Phase 3.2)
from core.redis_client import get_redis_client
from core.redis_config import RedisDB
logging.basicConfig(level=logging.INFO, format="%(asctime)s [TARGET-FORGE] %(message)s")
log = logging.getLogger("TARGET-FORGE")


@dataclass
class TargetProfile:
    """Represents a target node for CPU-triggered propagation."""
    ip: str
    confidence: float  # From Shogun zombie scoring
    trust_score: float = 0.0  # From trust handshake
    attempt_count: int = 0
    last_attempt: Optional[float] = None
    session_established: bool = False
    cost_incurred: Decimal = field(default_factory=lambda: Decimal('0'))


@dataclass
class TargetMetrics:
    """Aggregate targeting metrics."""
    total_targets_discovered: int = 0
    targets_handshaked: int = 0
    targets_trusted: int = 0
    propagation_triggered: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal('0'))
    avg_trust_score: float = 0.0
    last_cpu_spike: Optional[float] = None


class TrustHandshaker:
    """Performs trust-based handshakes with targets using PeerLinker protocol."""

    def __init__(self, targeting_engine: 'TargetingEngine'):
        self.targeting_engine = targeting_engine
        self.trust_threshold = 0.6  # Configurable
        self.timeout = 10  # seconds

    def evaluate_target_trust(self, target_ip: str, zombie_candidate: Dict) -> float:
        """
        Evaluate trust score for a target using multiple signals.
        
        Combines:
        - Shogun confidence (idle host reliability)
        - Network timing patterns
        - Service consistency
        - Signature verification readiness
        """
        trust_score = 0.0
        
        # Zombie confidence weight (40%)
        confidence = zombie_candidate.get("confidence", 0.5)
        trust_score += confidence * 0.4
        log.debug(f"Target {target_ip}: confidence={confidence:.2f} (+0.4 weight)")

        # Network stability (check uptime via ICMP patterns) (20%)
        uptime_score = self._estimate_uptime_stability(target_ip, zombie_candidate)
        trust_score += uptime_score * 0.2
        log.debug(f"Target {target_ip}: uptime_stability={uptime_score:.2f} (+0.2 weight)")

        # Service consistency (20%)
        service_score = self._assess_service_consistency(zombie_candidate)
        trust_score += service_score * 0.2
        log.debug(f"Target {target_ip}: service_consistency={service_score:.2f} (+0.2 weight)")

        # Signature verification readiness (20%)
        sig_score = self._verify_signature_readiness(target_ip, zombie_candidate)
        trust_score += sig_score * 0.2
        log.debug(f"Target {target_ip}: sig_readiness={sig_score:.2f} (+0.2 weight)")

        return min(trust_score, 1.0)

    def _estimate_uptime_stability(self, target_ip: str, zombie_candidate: Dict) -> float:
        """Estimate target uptime stability from Shogun data."""
        ip_id_samples = zombie_candidate.get("ip_id_samples", [])
        
        if not ip_id_samples or len(ip_id_samples) < 3:
            return 0.3  # Low confidence if few samples
        
        # Check for monotonic increase (sign of stable uptime counter)
        diffs = []
        for i in range(1, len(ip_id_samples)):
            diffs.append(ip_id_samples[i] - ip_id_samples[i-1])
        
        # Consistent increment = stable system
        if diffs and all(0 < d <= 1000 for d in diffs):  # Reasonable IP ID increments
            return 0.9
        elif len(set(diffs)) <= 2:  # Some consistency
            return 0.7
        else:
            return 0.4

    def _assess_service_consistency(self, zombie_candidate: Dict) -> float:
        """Assess consistency of exposed services."""
        services = zombie_candidate.get("services", {})
        
        if not services:
            return 0.3
        
        service_count = len(services)
        
        # Sweet spot: 3-8 services (realistic but not honeypot-like)
        if 3 <= service_count <= 8:
            return 0.9
        elif 1 <= service_count <= 2 or 9 <= service_count <= 12:
            return 0.6
        else:
            return 0.2

    def _verify_signature_readiness(self, target_ip: str, zombie_candidate: Dict) -> float:
        """Check if target supports crypto (SSH key, TLS fingerprints)."""
        ssh_port = 22 in zombie_candidate.get("open_ports", [])
        https_port = 443 in zombie_candidate.get("open_ports", [])
        
        sig_support = 0.0
        if ssh_port:
            sig_support += 0.6
        if https_port:
            sig_support += 0.4
        
        return min(sig_support, 1.0)

    def perform_handshake(self, target_profile: TargetProfile) -> bool:
        """
        Execute trust-based handshake with target.
        
        Returns True if handshake succeeds and trust >= threshold.
        """
        try:
            log.info(f"Initiating trust handshake with {target_profile.ip} "
                    f"(confidence: {target_profile.confidence:.2f})")
            
            # Simulate cryptographic handshake (would use PeerLinkerAgent in production)
            # In production: peer_linker_agent.initiate_handshake(target_profile.ip)
            
            # For now: deterministic simulation based on trust score
            handshake_success = target_profile.trust_score >= self.trust_threshold
            
            if handshake_success:
                target_profile.session_established = True
                log.info(f"✓ Handshake successful with {target_profile.ip} "
                        f"(trust: {target_profile.trust_score:.2f})")
                self.targeting_engine.metrics.targets_handshaked += 1
            else:
                log.warning(f"✗ Handshake failed with {target_profile.ip} "
                           f"(trust: {target_profile.trust_score:.2f} < {self.trust_threshold})")
            
            target_profile.attempt_count += 1
            target_profile.last_attempt = time.time()
            return handshake_success
            
        except Exception as e:
            log.error(f"Handshake exception for {target_profile.ip}: {e}")
            return False


class CPUTriggeredPropagation:
    """Monitors CPU and triggers propagation when thresholds are met."""

    def __init__(self, targeting_engine: 'TargetingEngine'):
        self.targeting_engine = targeting_engine
        self.cpu_threshold = 75.0  # Trigger when CPU > 75%
        self.propagation_window = 300  # 5 minutes between propagations
        self.last_propagation = 0.0

    def should_trigger_propagation(self, cpu_percent: float) -> bool:
        """Determine if CPU spike warrants propagation attempt."""
        if cpu_percent < self.cpu_threshold:
            return False
        
        time_since_last = time.time() - self.last_propagation
        if time_since_last < self.propagation_window:
            log.debug(f"Propagation cooldown active ({self.propagation_window}s window)")
            return False
        
        log.warning(f"🔥 CPU spike detected: {cpu_percent:.1f}% (threshold: {self.cpu_threshold}%)")
        return True

    def trigger_propagation(self, trusted_targets: List[TargetProfile]) -> int:
        """
        Trigger propagation agent for all trusted targets.
        
        Returns number of propagation missions dispatched.
        """
        if not trusted_targets:
            log.warning("No trusted targets available for propagation")
            return 0
        
        dispatched = 0
        for target in trusted_targets:
            try:
                # Create propagation mission
                mission = Mission(
                    name=f"cpu_triggered_prop_{target.ip}",
                    objectives=[f"Propagate to {target.ip}"],
                    parameters={
                        "target_ip": target.ip,
                        "trust_score": target.trust_score,
                        "trigger_type": "cpu_spike",
                        "timestamp": time.time()
                    }
                )
                
                # Queue for PropagationAgent consumption
                self.targeting_engine.propagation_queue.append(mission)
                log.info(f"Queued propagation mission to {target.ip} "
                        f"(trust: {target.trust_score:.2f})")
                
                # Economic tracking
                cost = Decimal('0.2')  # CPU-triggered dispatch is cheaper
                target.cost_incurred += cost
                self.targeting_engine.metrics.total_cost += cost
                
                dispatched += 1
                self.targeting_engine.metrics.propagation_triggered += 1
                
            except Exception as e:
                log.error(f"Failed to queue propagation for {target.ip}: {e}")
        
        self.last_propagation = time.time()
        return dispatched


class TargetingEngine(BaseAgent):
    """
    Orchestrates CPU-triggered propagation with trust-based targeting.
    
    Flow:
    1. SensorAgent detects CPU spike (> 75%)
    2. TargetingEngine queries Shogun for zombie candidates
    3. TrustHandshaker evaluates & establishes trust with candidates
    4. CPUTriggeredPropagation dispatches missions to trusted targets
    5. PropagationAgent consumes missions and spreads to targets
    """

    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])

    def __init__(self):
        super().__init__("TARGET-FORGE")
        self.agent_id = AgentID(value="TARGET-FORGE")
        
        # Components
        self.trust_handshaker = TrustHandshaker(self)
        self.cpu_propagation = CPUTriggeredPropagation(self)
        
        # State
        self.metrics = TargetMetrics()
        self.target_profiles: Dict[str, TargetProfile] = {}
        self.propagation_queue: List[Mission] = []
        self.running = False
        self.task_runners = {}  # TaskRunner pool for daemon coordination (Phase 3.2)
        
        # Economic Layer
        try:
            from agents.accountant import ACCLedger
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('50'))
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant: {e}")
        
        # Redis connection for kernel events (using factory pattern)
        self.redis_client = get_redis_client(RedisDB.KERNEL_STATE)

    def _deduct_targeting_cost(self, amount: Decimal = Decimal('0.1')):
        """Deduct Root for targeting operations."""
        if hasattr(self, 'accountant') and self.accountant:
            self.accountant.process_agent_spend(str(self.agent_id), amount, "Targeting operation")

    def _reward_successful_targeting(self, amount: Decimal = Decimal('0.3')):
        """Reward successful trust establishment."""
        if hasattr(self, 'accountant') and self.accountant:
            self.accountant.agent_wallets[str(self.agent_id)].balance += amount
            self.accountant.treasury_balance -= amount
            log.info(f"Rewarded {amount} Root for successful targeting")

    def attach_kernel(self, kernel):
        """Attach to Blackroot kernel for event access."""
        self.kernel = kernel
        log.info("Attached to Blackroot kernel")

    def run(self, mission: Optional[Mission] = None):
        """Main targeting orchestration loop via TaskRunner pool."""
        super().run()
        self.running = True
        log.info("TargetingEngine initialized and running")
        
        # Start TaskRunner pool for adaptive daemon management (Phase 3.2)
        daemon_config = [
            ("CPUMonitor", self._cpu_monitor_task, 5.0),
            ("HandshakeProcessor", self._handshake_processor_task, 10.0),
            ("PropagationDispatcher", self._propagation_dispatcher_task, 2.0)
        ]
        
        for name, task_fn, interval in daemon_config:
            runner = TaskRunner(
                task_fn=task_fn,
                interval=interval,
                name=name,
                max_backoff=30.0
            )
            runner.start(run_immediately=False)
            self.task_runners[name] = runner
            log.debug(f"Started {name} TaskRunner with interval {interval}s")
        
        # Keep alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("TargetingEngine shutting down")
            self.running = False
            self._shutdown_task_runners()

    # --- TaskRunner Daemon Task Functions (Phase 3.2) ---
    # Synchronous task functions replacing async while-True loops
    # Each returns Optional[float]: None = use default interval, float = override interval
    
    def _cpu_monitor_task(self) -> Optional[float]:
        """Synchronous CPU monitor task - detects CPU spikes and triggers targeting."""
        try:
            # Poll CPU metrics
            if hasattr(self, 'kernel') and self.kernel:
                cpu_percent = getattr(self.kernel, '_last_cpu_sample', 0)
            else:
                import random
                cpu_percent = random.uniform(20, 95)
            
            # Check if CPU spike should trigger propagation
            if self.cpu_propagation.should_trigger_propagation(cpu_percent):
                log.info(f"CPU spike triggered targeting cycle (CPU: {cpu_percent:.1f}%)")
                self.metrics.last_cpu_spike = time.time()
                self._execute_targeting_cycle()
            
            return None  # Use default 5s interval
        except Exception as e:
            log.error(f"CPU monitor task error: {e}")
            return 5.0  # Backoff on error (TaskRunner applies exponential multiplier)
    
    def _handshake_processor_task(self) -> Optional[float]:
        """Synchronous handshake processor task - establishes trust with targets."""
        try:
            # Validate target_profiles exists and is valid dict
            if not isinstance(self.target_profiles, dict):
                log.error(f"Target profiles invalid: {type(self.target_profiles)}")
                return 10.0
            
            # Get targets queued for handshake
            targets_to_handshake = [
                p for p in self.target_profiles.values()
                if p.trust_score >= 0.5 and not p.session_established and p.attempt_count < 3
            ]
            
            handshake_error_occurred = False
            for target in targets_to_handshake:
                try:
                    if self.trust_handshaker.perform_handshake(target):
                        self.metrics.targets_trusted += 1
                        self._reward_successful_targeting()
                    time.sleep(0.5)  # Rate limit handshakes
                except Exception as e:
                    log.error(f"Handshake error for {target.ip}: {e}")
                    handshake_error_occurred = True
            
            # Return backoff if any handshake failed
            if handshake_error_occurred:
                return 10.0
            return None  # Use default 10s interval
        except Exception as e:
            log.error(f"Handshake processor task error: {e}")
            return 10.0  # Backoff on error
    
    def _propagation_dispatcher_task(self) -> Optional[float]:
        """Synchronous propagation dispatcher task - queues propagation missions."""
        try:
            # Validate propagation_queue exists and is valid
            if not isinstance(self.propagation_queue, list):
                log.error(f"Propagation queue invalid: {type(self.propagation_queue)}")
                return 2.0
            
            # Process queued propagation missions
            dispatch_error_occurred = False
            if self.propagation_queue:
                try:
                    mission = self.propagation_queue.pop(0)
                    log.info(f"Dispatching propagation mission: {mission.name}")
                    # In production: emit to PropagationAgent queue or kernel
                except Exception as e:
                    log.error(f"Propagation dispatch error: {e}")
                    dispatch_error_occurred = True
            
            # Return backoff if dispatch failed
            if dispatch_error_occurred:
                return 2.0
            return None  # Use default 2s interval
        except Exception as e:
            log.error(f"Propagation dispatcher task error: {e}")
            return 2.0  # Backoff on error
    
    def _shutdown_task_runners(self):
        """Gracefully shutdown all TaskRunner instances."""
        for name, runner in self.task_runners.items():
            try:
                runner.stop(timeout=5.0)
                log.debug(f"Stopped {name} TaskRunner")
            except Exception as e:
                log.warning(f"Error stopping {name} TaskRunner: {e}")
        self.task_runners.clear()
    
    def _monitor_cpu_events(self):
        """Legacy async CPU monitor (kept for backward compatibility if needed)."""
        log.info("CPU event monitor started (legacy)")
        
        while self.running:
            try:
                # In production: subscribe to Redis channel "sensor_cpu_alerts"
                # For now: poll kernel heartbeat data
                
                if hasattr(self, 'kernel') and self.kernel:
                    # Would get from kernel.telemetry or kernel.stealth.system_metrics
                    cpu_percent = getattr(self.kernel, '_last_cpu_sample', 0)
                else:
                    # Stub: simulate occasional spikes
                    import random
                    cpu_percent = random.uniform(20, 95)
                
                if self.cpu_propagation.should_trigger_propagation(cpu_percent):
                    log.info(f"CPU spike triggered targeting cycle (CPU: {cpu_percent:.1f}%)")
                    self.metrics.last_cpu_spike = time.time()
                    self._execute_targeting_cycle()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                log.error(f"CPU monitor exception: {e}")
                time.sleep(5)

    def _execute_targeting_cycle(self):
        """Full targeting cycle: discover → evaluate → handshake → propagate."""
        try:
            self._deduct_targeting_cost(Decimal('0.15'))
            
            # Step 1: Query Shogun for zombie candidates
            zombie_candidates = self._query_shogun_zombies()
            if not zombie_candidates:
                log.warning("No zombie candidates from Shogun")
                return
            
            self.metrics.total_targets_discovered += len(zombie_candidates)
            log.info(f"Discovered {len(zombie_candidates)} zombie candidates from Shogun")
            
            # Step 2: Evaluate trust for each candidate
            for zombie in zombie_candidates:
                ip = zombie.get("ip")
                if not ip:  # Skip if no IP
                    continue
                if ip in self.target_profiles:
                    continue  # Already profiled
                
                trust_score = self.trust_handshaker.evaluate_target_trust(ip, zombie)
                
                profile = TargetProfile(
                    ip=ip,
                    confidence=zombie.get("confidence", 0.5),
                    trust_score=trust_score
                )
                self.target_profiles[ip] = profile
                log.debug(f"Evaluated {ip}: trust_score={trust_score:.2f}")
            
            # Step 3: Queue handshakes for high-potential targets
            candidates_for_handshake = [
                p for p in self.target_profiles.values()
                if p.trust_score >= 0.5 and not p.session_established
            ]
            
            if candidates_for_handshake:
                log.info(f"Queuing {len(candidates_for_handshake)} targets for handshake")
            
        except Exception as e:
            log.error(f"Targeting cycle failed: {e}")

    def _query_shogun_zombies(self) -> List[Dict]:
        """
        Query real Shogun reconnaissance layer for zombie candidates.
        Phase 2.2: Queries actual ShogunReconAgent topology instead of simulating.
        """
        try:
            # Try to get ShogunReconAgent from kernel
            if self.kernel and hasattr(self.kernel, 'agents'):
                shogun_agent = self.kernel.agents.get("SHOGUN-RECON")
                if shogun_agent and hasattr(shogun_agent, 'last_topology'):
                    topology = shogun_agent.last_topology
                    if topology and isinstance(topology, dict) and "zombie_candidates" in topology:
                        log.info(f"[✅] Queried real Shogun topology: {len(topology['zombie_candidates'])} candidates")
                        return topology["zombie_candidates"]
            
            # Try Redis fallback
            if self.redis_client and self.redis_client.redis:
                zombies_json = self.redis_client.redis.get("shogun_topology")
                if zombies_json:
                    topology = json.loads(zombies_json) if isinstance(zombies_json, str) else zombies_json
                    if isinstance(topology, dict) and "zombie_candidates" in topology:
                        log.info(f"[✅] Retrieved Shogun topology from Redis: {len(topology['zombie_candidates'])} candidates")
                        return topology["zombie_candidates"]
            
            # Fallback to simulation if real data unavailable
            log.warning("[⚠️] Real Shogun topology unavailable, using simulation")
            return self._simulate_shogun_zombies()
            
        except Exception as e:
            log.error(f"Failed to query Shogun: {e}")
            return []

    def _simulate_shogun_zombies(self) -> List[Dict]:
        """
        Fallback simulation of Shogun zombie candidate data.
        Only used when real network scanning unavailable.
        """
        import random
        candidates = []
        for i in range(random.randint(2, 5)):
            candidates.append({
                "ip": f"192.168.1.{100 + i}",
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "open_ports": [22, 80, 443],
                "services": {"ssh": "OpenSSH_7.4", "http": "Apache/2.4", "https": "nginx/1.14"},
                "ip_id_samples": [1000 + j*50 for j in range(10)],
                "status": "ZOMBIE_CANDIDATE",
                "reason": "SIMULATION"
            })
        return candidates

    def _process_handshakes(self):
        """Legacy handshake processor (kept for backward compatibility if needed)."""
        log.info("Handshake processor started (legacy)")
        
        while self.running:
            try:
                # Get targets queued for handshake
                targets_to_handshake = [
                    p for p in self.target_profiles.values()
                    if p.trust_score >= 0.5 and not p.session_established and p.attempt_count < 3
                ]
                
                for target in targets_to_handshake:
                    if self.trust_handshaker.perform_handshake(target):
                        self.metrics.targets_trusted += 1
                        self._reward_successful_targeting()
                    
                    time.sleep(0.5)  # Rate limit
                
                time.sleep(10)  # Process batch every 10s
                
            except Exception as e:
                log.error(f"Handshake processor exception: {e}")
                time.sleep(10)

    def _dispatch_propagations(self):
        """Legacy propagation dispatcher (kept for backward compatibility if needed)."""
        log.info("Propagation dispatcher started (legacy)")
        
        while self.running:
            try:
                if self.propagation_queue:
                    mission = self.propagation_queue.pop(0)
                    
                    # In production: emit to PropagationAgent queue or kernel
                    log.info(f"Dispatching propagation mission: {mission.name}")
                    
                    # For now: just log
                    # In future: self.kernel.publish_event(mission) or similar
                
                time.sleep(2)
                
            except Exception as e:
                log.error(f"Propagation dispatcher exception: {e}")
                time.sleep(2)

    def get_status(self) -> Dict:
        """Return current targeting status."""
        return {
            "agent_id": str(self.agent_id),
            "status": AgentStatus.ACTIVE.value,
            "metrics": asdict(self.metrics),
            "active_targets": len(self.target_profiles),
            "trusted_targets": len([p for p in self.target_profiles.values() if p.session_established]),
            "queued_propagations": len(self.propagation_queue)
        }

    def get_target_profiles(self) -> Dict[str, Dict]:
        """Export target profiles as dict."""
        return {
            ip: {
                "confidence": p.confidence,
                "trust_score": p.trust_score,
                "session_established": p.session_established,
                "attempt_count": p.attempt_count,
                "cost_incurred": str(p.cost_incurred)
            }
            for ip, p in self.target_profiles.items()
        }


if __name__ == "__main__":
    engine = TargetingEngine()
    engine.run()
