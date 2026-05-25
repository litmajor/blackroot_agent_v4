"""
Agent health registry and monitoring for Blackroot.

Tracks:
- Agent heartbeats and last activity
- Agent status (healthy, degraded, dead)
- Auto-restart failures
- Resource usage per agent
- Overall system health
"""

import logging
import time
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock, Thread
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class AgentHealthMetrics:
    """Health metrics for a single agent."""
    agent_id: str
    agent_name: str
    status: str
    last_heartbeat: float
    last_activity: float
    error_count: int
    restart_count: int
    cpu_percent: float
    memory_mb: float
    uptime_seconds: float
    current_mission: Optional[str] = None
    last_error: Optional[str] = None
    
    def is_stale(self, timeout_seconds: float = 30.0) -> bool:
        """Check if heartbeat is stale."""
        return (time.time() - self.last_heartbeat) > timeout_seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class AgentHealthRegistry:
    """
    Central registry for agent health monitoring.
    
    Enables:
    - Per-agent heartbeat tracking
    - Automatic stale agent detection
    - Error counting and reporting
    - Restart tracking
    - System-wide health assessment
    """
    
    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        stale_check_interval: float = 5.0,
        max_restart_attempts: int = 3
    ):
        self.heartbeat_timeout = heartbeat_timeout
        self.stale_check_interval = stale_check_interval
        self.max_restart_attempts = max_restart_attempts
        
        # Health tracking
        self.agents: Dict[str, AgentHealthMetrics] = {}
        self._lock = Lock()
        
        # Monitoring thread
        self.monitoring_thread: Optional[Thread] = None
        self._monitoring_active = False
    
    def register_agent(self, agent_id: str, agent_name: str) -> None:
        """Register a new agent for monitoring."""
        with self._lock:
            now = time.time()
            self.agents[agent_id] = AgentHealthMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
                status=AgentStatus.HEALTHY.value,
                last_heartbeat=now,
                last_activity=now,
                error_count=0,
                restart_count=0,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                current_mission=None,
                last_error=None
            )
            logger.info(f"[💚] Agent registered: {agent_name} ({agent_id})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from monitoring."""
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"[💔] Agent unregistered: {agent_id}")
    
    def heartbeat(
        self,
        agent_id: str,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        current_mission: Optional[str] = None
    ) -> None:
        """Record agent heartbeat."""
        with self._lock:
            if agent_id not in self.agents:
                logger.warning(f"[⚠️] Heartbeat from unregistered agent: {agent_id}")
                return
            
            metrics = self.agents[agent_id]
            metrics.last_heartbeat = time.time()
            metrics.last_activity = time.time()
            metrics.cpu_percent = cpu_percent
            metrics.memory_mb = memory_mb
            
            if current_mission:
                metrics.current_mission = current_mission
            
            # Reset status to healthy if it was degraded
            if metrics.status == AgentStatus.DEGRADED.value:
                metrics.status = AgentStatus.HEALTHY.value
                logger.info(f"[💚] Agent recovered: {metrics.agent_name}")
    
    def record_activity(
        self,
        agent_id: str,
        activity: str = "activity",
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0
    ) -> None:
        """Record agent activity (alternative to heartbeat)."""
        self.heartbeat(agent_id, cpu_percent, memory_mb)
    
    def record_error(
        self,
        agent_id: str,
        error: Exception,
        severity: str = "warning"
    ) -> None:
        """Record error for an agent."""
        with self._lock:
            if agent_id not in self.agents:
                logger.warning(f"[⚠️] Error recorded for unregistered agent: {agent_id}")
                return
            
            metrics = self.agents[agent_id]
            metrics.error_count += 1
            metrics.last_error = str(error)
            
            # Transition to degraded after 3 errors
            if metrics.error_count >= 3 and metrics.status == AgentStatus.HEALTHY.value:
                metrics.status = AgentStatus.DEGRADED.value
                logger.warning(
                    f"[🟡] Agent degraded: {metrics.agent_name} "
                    f"({metrics.error_count} errors)"
                )
            
            logger.debug(
                f"[❌] Error in {metrics.agent_name}: {error} "
                f"(total: {metrics.error_count})"
            )
    
    def record_restart(self, agent_id: str) -> bool:
        """
        Record agent restart attempt.
        
        Returns:
            True if restart allowed, False if max attempts exceeded
        """
        with self._lock:
            if agent_id not in self.agents:
                return False
            
            metrics = self.agents[agent_id]
            metrics.restart_count += 1
            
            if metrics.restart_count > self.max_restart_attempts:
                metrics.status = AgentStatus.DEAD.value
                logger.error(
                    f"[💀] Agent DEAD: {metrics.agent_name} "
                    f"(restart attempts: {metrics.restart_count})"
                )
                return False
            
            metrics.status = AgentStatus.HEALTHY.value
            metrics.error_count = 0
            logger.info(
                f"[🔄] Agent restarted: {metrics.agent_name} "
                f"(attempt {metrics.restart_count}/{self.max_restart_attempts})"
            )
            return True
    
    def get_agent_health(self, agent_id: str) -> Optional[AgentHealthMetrics]:
        """Get health metrics for a specific agent."""
        with self._lock:
            return self.agents.get(agent_id)
    
    def get_all_agent_health(self) -> Dict[str, AgentHealthMetrics]:
        """Get health metrics for all agents."""
        with self._lock:
            return dict(self.agents)
    
    def get_agents_by_status(self, status: AgentStatus) -> List[str]:
        """Get all agents with a specific status."""
        with self._lock:
            return [
                agent_id for agent_id, metrics in self.agents.items()
                if metrics.status == status.value
            ]
    
    def _check_stale_agents(self) -> None:
        """Check for and mark stale agents as degraded."""
        with self._lock:
            for agent_id, metrics in self.agents.items():
                if metrics.is_stale(self.heartbeat_timeout):
                    if metrics.status == AgentStatus.HEALTHY.value:
                        metrics.status = AgentStatus.DEGRADED.value
                        logger.warning(
                            f"[🟡] Agent stale (no heartbeat > {self.heartbeat_timeout}s): "
                            f"{metrics.agent_name}"
                        )
    
    def start_monitoring(self) -> None:
        """Start background health monitoring thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("[🔍] Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("[🔍] Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self._check_stale_agents()
                time.sleep(self.stale_check_interval)
            except Exception as e:
                logger.error(f"[❌] Error in health monitoring loop: {e}\n{traceback.format_exc()}")
                time.sleep(1.0)
    
    def get_system_health(self) -> Dict:
        """
        Get overall system health summary.
        
        Returns:
            Dict with health statistics
        """
        with self._lock:
            if not self.agents:
                return {
                    "status": "unknown",
                    "total_agents": 0,
                    "healthy": 0,
                    "degraded": 0,
                    "dead": 0,
                    "agents": {}
                }
            
            healthy_count = sum(
                1 for m in self.agents.values()
                if m.status == AgentStatus.HEALTHY.value
            )
            degraded_count = sum(
                1 for m in self.agents.values()
                if m.status == AgentStatus.DEGRADED.value
            )
            dead_count = sum(
                1 for m in self.agents.values()
                if m.status == AgentStatus.DEAD.value
            )
            
            # Overall status: degraded if any dead, warning if any degraded
            if dead_count > 0:
                overall_status = "critical"
            elif degraded_count > 0:
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            return {
                "status": overall_status,
                "total_agents": len(self.agents),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "dead": dead_count,
                "agents": {
                    agent_id: metrics.to_dict()
                    for agent_id, metrics in self.agents.items()
                }
            }
    
    def reset_errors(self, agent_id: str) -> None:
        """Reset error count for an agent (e.g., after maintenance)."""
        with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].error_count = 0
                self.agents[agent_id].status = AgentStatus.HEALTHY.value
                logger.info(f"[🔄] Errors reset for: {self.agents[agent_id].agent_name}")


# Global singleton registry
_health_registry: Optional[AgentHealthRegistry] = None
_registry_lock = Lock()


def initialize_global_registry(
    heartbeat_timeout: float = 30.0,
    stale_check_interval: float = 5.0,
    max_restart_attempts: int = 3,
    start_monitoring: bool = True
) -> AgentHealthRegistry:
    """Initialize or get the global health registry."""
    global _health_registry
    
    with _registry_lock:
        if _health_registry is None:
            _health_registry = AgentHealthRegistry(
                heartbeat_timeout=heartbeat_timeout,
                stale_check_interval=stale_check_interval,
                max_restart_attempts=max_restart_attempts
            )
            if start_monitoring:
                _health_registry.start_monitoring()
        return _health_registry


def get_global_registry() -> Optional[AgentHealthRegistry]:
    """Get the global health registry."""
    return _health_registry
