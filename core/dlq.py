"""
Mission Dead-Letter Queue (DLQ) for Blackroot.

Handles:
- Failed mission storage and retry
- Exponential backoff for retries
- DLQ inspection and recovery
- Mission deduplication
- Audit trail for failures
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FailedMission:
    """Representation of a failed mission in the DLQ."""
    mission_id: str
    agent_id: str
    target: str
    mission_type: str
    payload: Dict[str, Any]
    error: str
    failure_time: float
    retry_count: int
    next_retry_time: float
    original_cost: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def is_ready_for_retry(self) -> bool:
        """Check if mission is ready for retry."""
        return time.time() >= self.next_retry_time


class MissionDeadLetterQueue:
    """
    Dead-letter queue for failed missions.
    
    Enables:
    - Storing failed missions with context
    - Scheduled retries with exponential backoff
    - Mission deduplication (prevent duplicate failures)
    - Audit trail and recovery
    - DLQ metrics and inspection
    """
    
    def __init__(
        self,
        initial_backoff: float = 1.0,
        max_backoff: float = 3600.0,  # 1 hour
        max_retries: int = 5,
        cleanup_age_hours: int = 24
    ):
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.max_retries = max_retries
        self.cleanup_age_hours = cleanup_age_hours
        
        # Mission storage
        self.queue: Dict[str, FailedMission] = {}
        self._lock = Lock()
        
        # Deduplication tracking (mission_signature -> last_failure_time)
        self.dedup_cache: Dict[str, float] = {}
        
        # Metrics
        self.total_dlq_entries = 0
        self.total_retried = 0
        self.total_recovered = 0
    
    def _calculate_mission_signature(
        self,
        agent_id: str,
        target: str,
        mission_type: str
    ) -> str:
        """Calculate unique signature for mission deduplication."""
        content = f"{agent_id}:{target}:{mission_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _should_deduplicate(
        self,
        agent_id: str,
        target: str,
        mission_type: str,
        dedup_window_seconds: float = 300.0
    ) -> bool:
        """Check if mission looks like a duplicate failure."""
        signature = self._calculate_mission_signature(agent_id, target, mission_type)
        now = time.time()
        
        if signature in self.dedup_cache:
            last_failure = self.dedup_cache[signature]
            if (now - last_failure) < dedup_window_seconds:
                logger.info(
                    f"[⚠️] Duplicate mission detected (signature: {signature}), "
                    f"skipping DLQ entry"
                )
                return True
        
        self.dedup_cache[signature] = now
        return False
    
    def add_failed_mission(
        self,
        mission_id: str,
        agent_id: str,
        target: str,
        mission_type: str,
        payload: Dict[str, Any],
        error: str,
        original_cost: float = 0.0,
        initial_retry_delay: Optional[float] = None
    ) -> bool:
        """
        Add a failed mission to the DLQ.
        
        Args:
            mission_id: Unique mission identifier
            agent_id: ID of agent that failed
            target: Target of the mission
            mission_type: Type of mission (propagate, replicate, etc.)
            payload: Mission payload
            error: Error that caused failure
            original_cost: Economic cost of mission
            initial_retry_delay: Custom retry delay (uses exponential backoff if None)
            
        Returns:
            True if added, False if deduplicated
        """
        if self._should_deduplicate(agent_id, target, mission_type):
            return False
        
        with self._lock:
            now = time.time()
            
            # If mission already in DLQ, update retry count
            if mission_id in self.queue:
                failed_mission = self.queue[mission_id]
                failed_mission.retry_count += 1
                failed_mission.error = error
                failed_mission.failure_time = now
                
                if failed_mission.retry_count > self.max_retries:
                    logger.error(
                        f"[💀] Mission exceeded max retries: {mission_id} "
                        f"(target: {target}, retries: {failed_mission.retry_count})"
                    )
                    return False
            else:
                # Calculate retry delay
                if initial_retry_delay is not None:
                    retry_delay = initial_retry_delay
                else:
                    retry_delay = self.initial_backoff
                
                next_retry = now + retry_delay
                
                failed_mission = FailedMission(
                    mission_id=mission_id,
                    agent_id=agent_id,
                    target=target,
                    mission_type=mission_type,
                    payload=payload,
                    error=error,
                    failure_time=now,
                    retry_count=0,
                    next_retry_time=next_retry,
                    original_cost=original_cost
                )
                
                self.queue[mission_id] = failed_mission
                self.total_dlq_entries += 1
            
            logger.warning(
                f"[📋] Mission added to DLQ: {mission_id} "
                f"(target: {target}, retry: {failed_mission.retry_count + 1}/{self.max_retries + 1})"
            )
            return True
    
    def get_ready_for_retry(self, max_count: int = 100) -> List[FailedMission]:
        """
        Get missions ready for retry (respecting exponential backoff).
        
        Args:
            max_count: Maximum number of missions to return
            
        Returns:
            List of FailedMission objects ready for retry
        """
        with self._lock:
            ready = []
            for mission_id, mission in list(self.queue.items()):
                if mission.is_ready_for_retry() and mission.retry_count <= self.max_retries:
                    ready.append(mission)
                    if len(ready) >= max_count:
                        break
            return ready
    
    def record_retry_success(self, mission_id: str) -> bool:
        """
        Record that a mission was successfully retried.
        
        Args:
            mission_id: Mission that succeeded
            
        Returns:
            True if removed from DLQ, False if not found
        """
        with self._lock:
            if mission_id in self.queue:
                mission = self.queue[mission_id]
                del self.queue[mission_id]
                self.total_retried += 1
                self.total_recovered += 1
                logger.info(
                    f"[✅] Mission recovered from DLQ: {mission_id} "
                    f"(after {mission.retry_count} retries)"
                )
                return True
            return False
    
    def record_retry_failure(
        self,
        mission_id: str,
        error: str
    ) -> bool:
        """
        Record that a retry attempt failed.
        
        Updates retry count and calculates next backoff.
        
        Args:
            mission_id: Mission that failed again
            error: New error message
            
        Returns:
            True if mission still has retries, False if exceeded max
        """
        with self._lock:
            if mission_id not in self.queue:
                return False
            
            mission = self.queue[mission_id]
            mission.retry_count += 1
            mission.error = error
            mission.failure_time = time.time()
            
            if mission.retry_count > self.max_retries:
                del self.queue[mission_id]
                logger.error(
                    f"[💀] Mission failed permanently: {mission_id} "
                    f"(exhausted {self.max_retries} retries)"
                )
                return False
            
            # Calculate exponential backoff: initial_backoff * 2^retry_count
            backoff = min(
                self.initial_backoff * (2 ** mission.retry_count),
                self.max_backoff
            )
            mission.next_retry_time = time.time() + backoff
            
            logger.warning(
                f"[⚠️] Retry failed, scheduling next attempt: {mission_id} "
                f"(next retry in {backoff:.0f}s)"
            )
            return True
    
    def remove_mission(self, mission_id: str) -> bool:
        """
        Manually remove a mission from DLQ.
        
        Args:
            mission_id: Mission to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if mission_id in self.queue:
                del self.queue[mission_id]
                logger.info(f"[🗑️] Mission removed from DLQ: {mission_id}")
                return True
            return False
    
    def cleanup_old_missions(self) -> int:
        """
        Remove missions that have been in DLQ longer than cleanup_age_hours.
        
        Returns:
            Number of missions cleaned up
        """
        with self._lock:
            now = time.time()
            cutoff_time = now - (self.cleanup_age_hours * 3600)
            
            to_remove = [
                mid for mid, m in self.queue.items()
                if m.failure_time < cutoff_time
            ]
            
            for mission_id in to_remove:
                del self.queue[mission_id]
            
            if to_remove:
                logger.info(f"[🧹] DLQ cleanup: removed {len(to_remove)} old missions")
            
            return len(to_remove)
    
    def get_dlq_stats(self) -> Dict[str, Any]:
        """
        Get DLQ statistics.
        
        Returns:
            Dictionary with DLQ metrics
        """
        with self._lock:
            if not self.queue:
                pending_by_type = {}
                pending_by_agent = {}
            else:
                pending_by_type = {}
                pending_by_agent = {}
                
                for mission in self.queue.values():
                    pending_by_type[mission.mission_type] = \
                        pending_by_type.get(mission.mission_type, 0) + 1
                    pending_by_agent[mission.agent_id] = \
                        pending_by_agent.get(mission.agent_id, 0) + 1
            
            return {
                "queue_size": len(self.queue),
                "total_dlq_entries_lifetime": self.total_dlq_entries,
                "total_retried": self.total_retried,
                "total_recovered": self.total_recovered,
                "recovery_rate": (
                    f"{(self.total_recovered / max(self.total_dlq_entries, 1) * 100):.1f}%"
                ),
                "pending_by_type": pending_by_type,
                "pending_by_agent": pending_by_agent
            }
    
    def get_all_missions(self) -> List[FailedMission]:
        """Get all missions currently in DLQ."""
        with self._lock:
            return list(self.queue.values())
    
    def get_missions_for_agent(self, agent_id: str) -> List[FailedMission]:
        """Get all failed missions for a specific agent."""
        with self._lock:
            return [
                m for m in self.queue.values()
                if m.agent_id == agent_id
            ]


# Global singleton DLQ
_mission_dlq: Optional[MissionDeadLetterQueue] = None
_dlq_lock = Lock()


def initialize_global_dlq(
    initial_backoff: float = 1.0,
    max_backoff: float = 3600.0,
    max_retries: int = 5,
    cleanup_age_hours: int = 24
) -> MissionDeadLetterQueue:
    """Initialize or get the global mission DLQ."""
    global _mission_dlq
    
    with _dlq_lock:
        if _mission_dlq is None:
            _mission_dlq = MissionDeadLetterQueue(
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                max_retries=max_retries,
                cleanup_age_hours=cleanup_age_hours
            )
        return _mission_dlq


def get_global_dlq() -> Optional[MissionDeadLetterQueue]:
    """Get the global mission DLQ."""
    return _mission_dlq
