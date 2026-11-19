#!/usr/bin/env python3
"""
Checkpoint/Resume Logic - Crash Recovery for Mission Execution

Purpose:
- Persist mission state at each stage checkpoint
- Resume from last successful stage on failure/restart
- Manage exponential backoff and retry limits
- Audit trail of all checkpoint events
"""

import json
import logging
import os
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Single checkpoint in mission execution"""
    mission_id: str
    stage_name: str
    stage_index: int
    timestamp: str
    state: str                                 # "created", "started", "completed", "failed"
    attempt: int
    result_data: Optional[Dict] = None
    error: Optional[str] = None
    backoff_seconds: float = 0.0
    recovery_hint: Optional[str] = None       # How to resume if interrupted

    def to_dict(self) -> Dict:
        return asdict(self)


class CheckpointManager:
    """
    Manage checkpoints for mission recovery.
    
    Usage:
        manager = CheckpointManager()
        manager.create_checkpoint(mission_id, stage_name, index)
        manager.mark_stage_complete(mission_id, stage_name, result)
        recovered = manager.get_last_checkpoint(mission_id)
    """
    
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    CHECKPOINT_TTL = 604800  # 7 days
    
    def __init__(self, use_redis: bool = True):
        self.use_redis = use_redis
        self.redis_client = None
        self.memory_checkpoints: Dict[str, list] = {}
        
        if use_redis and redis:
            try:
                self.redis_client = redis.Redis(
                    host=self.REDIS_HOST,
                    port=self.REDIS_PORT,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info(f"[✓] CheckpointManager Redis initialized")
            except Exception as e:
                logger.warning(f"[!] Redis unavailable for CheckpointManager: {e}")
                self.redis_client = None
        
        logger.info("[✓] CheckpointManager initialized")
    
    def create_checkpoint(self, mission_id: str, stage_name: str, 
                         stage_index: int, attempt: int = 1) -> Checkpoint:
        """
        Create a new checkpoint at stage entry.
        
        Args:
            mission_id: ID of mission
            stage_name: e.g., "SCAN", "ESCALATE"
            stage_index: 0-based index in pipeline
            attempt: Retry attempt number
            
        Returns:
            Checkpoint object
        """
        checkpoint = Checkpoint(
            mission_id=mission_id,
            stage_name=stage_name,
            stage_index=stage_index,
            timestamp=datetime.utcnow().isoformat(),
            state="created",
            attempt=attempt,
            recovery_hint=f"Resume from {stage_name} (attempt {attempt})"
        )
        
        self._persist_checkpoint(checkpoint)
        logger.info(f"[📍] Checkpoint created: {mission_id[:8]}.../{stage_name} (attempt {attempt})")
        
        return checkpoint
    
    def mark_stage_started(self, mission_id: str, stage_name: str) -> Checkpoint:
        """Mark checkpoint as started"""
        checkpoint = self.get_last_checkpoint(mission_id)
        if checkpoint and checkpoint.state == "created":
            checkpoint.state = "started"
            checkpoint.timestamp = datetime.utcnow().isoformat()
            self._persist_checkpoint(checkpoint)
            logger.info(f"[▶️] Stage started: {mission_id[:8]}.../{stage_name}")
        return checkpoint
    
    def mark_stage_completed(self, mission_id: str, stage_name: str, 
                            result_data: Dict) -> Checkpoint:
        """Mark checkpoint as completed successfully"""
        checkpoint = self.get_last_checkpoint(mission_id)
        if checkpoint:
            checkpoint.state = "completed"
            checkpoint.result_data = result_data
            checkpoint.timestamp = datetime.utcnow().isoformat()
            self._persist_checkpoint(checkpoint)
            logger.info(f"[✅] Stage completed: {mission_id[:8]}.../{stage_name}")
        return checkpoint
    
    def mark_stage_failed(self, mission_id: str, stage_name: str, 
                         error: str, attempt: int, max_retries: int,
                         backoff_base: float = 2.0) -> Checkpoint:
        """
        Mark checkpoint as failed and calculate backoff.
        
        Args:
            mission_id: Mission ID
            stage_name: Stage name
            error: Error message
            attempt: Current attempt number
            max_retries: Maximum retry attempts
            backoff_base: Base for exponential backoff (2^attempt seconds)
            
        Returns:
            Updated checkpoint with backoff info
        """
        checkpoint = self.get_last_checkpoint(mission_id)
        if checkpoint:
            checkpoint.state = "failed"
            checkpoint.error = error
            checkpoint.timestamp = datetime.utcnow().isoformat()
            
            # Calculate backoff: 2^(attempt-1) seconds, capped at 5 minutes
            if attempt < max_retries:
                backoff_seconds = min(300.0, backoff_base ** (attempt - 1))
                checkpoint.backoff_seconds = backoff_seconds
                checkpoint.recovery_hint = (
                    f"Retry {stage_name} in {backoff_seconds:.0f}s (attempt {attempt}/{max_retries})"
                )
                logger.warning(f"[⏳] Stage failed, will retry in {backoff_seconds:.0f}s: {error[:50]}...")
            else:
                checkpoint.recovery_hint = f"Max retries ({max_retries}) exceeded, giving up"
                logger.error(f"[❌] Stage failed (max retries exceeded): {error}")
            
            self._persist_checkpoint(checkpoint)
        
        return checkpoint
    
    def get_last_checkpoint(self, mission_id: str) -> Optional[Checkpoint]:
        """Get most recent checkpoint for mission"""
        # Try Redis first
        if self.redis_client:
            try:
                key = f"checkpoint:{mission_id}:latest"
                data = self.redis_client.get(key)
                if data:
                    return Checkpoint(**json.loads(data))
            except Exception as e:
                logger.warning(f"[!] Failed to get checkpoint from Redis: {e}")
        
        # Try memory cache
        if mission_id in self.memory_checkpoints and self.memory_checkpoints[mission_id]:
            return self.memory_checkpoints[mission_id][-1]
        
        return None
    
    def get_checkpoint_history(self, mission_id: str) -> list[Checkpoint]:
        """Get all checkpoints for mission"""
        checkpoints = []
        
        # Try Redis first
        if self.redis_client:
            try:
                pattern = f"checkpoint:{mission_id}:*"
                keys = self.redis_client.keys(pattern)
                for key in sorted(keys):
                    data = self.redis_client.get(key)
                    if data:
                        checkpoints.append(Checkpoint(**json.loads(data)))
                if checkpoints:
                    return sorted(checkpoints, key=lambda c: c.timestamp)
            except Exception as e:
                logger.warning(f"[!] Failed to get checkpoint history from Redis: {e}")
        
        # Fallback to memory cache
        if mission_id in self.memory_checkpoints:
            return self.memory_checkpoints[mission_id]
        
        return []
    
    def should_resume_from_checkpoint(self, mission_id: str, 
                                      last_known_stage: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Determine if mission should resume from checkpoint.
        
        Returns:
            (should_resume: bool, stage_name: Optional[str])
        """
        checkpoint = self.get_last_checkpoint(mission_id)
        if not checkpoint:
            return False, None
        
        # If checkpoint is completed, don't resume (already done)
        if checkpoint.state == "completed":
            return False, None
        
        # If checkpoint failed and has a hint, resume that stage
        if checkpoint.state == "failed" and checkpoint.recovery_hint:
            return True, checkpoint.stage_name
        
        # If checkpoint was started but not completed, resume it
        if checkpoint.state == "started":
            return True, checkpoint.stage_name
        
        return False, None
    
    def calculate_resume_delay(self, mission_id: str) -> float:
        """Calculate delay before resuming (based on backoff)"""
        checkpoint = self.get_last_checkpoint(mission_id)
        if checkpoint and checkpoint.backoff_seconds > 0:
            return checkpoint.backoff_seconds
        return 0.0
    
    def cleanup_old_checkpoints(self, older_than_days: int = 7):
        """Clean up old checkpoints from Redis"""
        if not self.redis_client:
            return
        
        try:
            cutoff_time = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
            
            # Get all checkpoint keys
            keys = self.redis_client.keys("checkpoint:*")
            deleted = 0
            
            for key in keys:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        checkpoint = Checkpoint(**json.loads(data))
                        if checkpoint.timestamp < cutoff_time:
                            self.redis_client.delete(key)
                            deleted += 1
                except Exception as e:
                    logger.warning(f"[!] Failed to clean checkpoint {key}: {e}")
            
            logger.info(f"[🧹] Cleaned {deleted} old checkpoints (older than {older_than_days} days)")
        
        except Exception as e:
            logger.error(f"[!] Checkpoint cleanup failed: {e}")
    
    def _persist_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to Redis and memory"""
        # Memory cache
        if checkpoint.mission_id not in self.memory_checkpoints:
            self.memory_checkpoints[checkpoint.mission_id] = []
        self.memory_checkpoints[checkpoint.mission_id].append(checkpoint)
        
        # Redis persistence
        if self.redis_client:
            try:
                key = f"checkpoint:{checkpoint.mission_id}:latest"
                serialized = json.dumps(checkpoint.to_dict(), default=str)
                self.redis_client.setex(key, self.CHECKPOINT_TTL, serialized)
                
                # Also store timestamped version
                ts_key = f"checkpoint:{checkpoint.mission_id}:{checkpoint.timestamp}"
                self.redis_client.setex(ts_key, self.CHECKPOINT_TTL, serialized)
            except Exception as e:
                logger.warning(f"[!] Failed to persist checkpoint to Redis: {e}")


class ResumableExecution:
    """
    Wrapper around mission execution to add resume capability.
    
    Usage:
        executor = ResumableExecution(mission, orchestrator, checkpoint_manager)
        results = executor.execute_with_recovery()
    """
    
    def __init__(self, mission_id: str, orchestrator, checkpoint_manager: CheckpointManager):
        self.mission_id = mission_id
        self.orchestrator = orchestrator
        self.checkpoint_manager = checkpoint_manager
        self.mission = orchestrator.load_mission(mission_id)
    
    def execute_with_recovery(self) -> Dict[str, Any]:
        """
        Execute mission with automatic resume capability.
        
        On restart, will:
        1. Check for last checkpoint
        2. If recoverable, resume from that stage
        3. Apply exponential backoff delays
        """
        if not self.mission:
            logger.error(f"[❌] Mission {self.mission_id} not found")
            return {'error': 'Mission not found'}
        
        # Check if should resume
        should_resume, resume_from_stage = self.checkpoint_manager.should_resume_from_checkpoint(
            self.mission_id
        )
        
        if should_resume:
            logger.info(f"[↺] Resuming mission from stage: {resume_from_stage}")
            delay = self.checkpoint_manager.calculate_resume_delay(self.mission_id)
            if delay > 0:
                logger.info(f"[⏳] Waiting {delay:.0f}s before resume (exponential backoff)")
                import time
                time.sleep(delay)
            
            # Find stage to resume from
            resume_index = None
            for i, stage in enumerate(self.mission.stages):
                if stage.name == resume_from_stage:
                    resume_index = i
                    break
            
            if resume_index is not None:
                # Skip already-completed stages
                logger.info(f"[→] Skipping {resume_index} completed stages")
                self.mission.stages = self.mission.stages[resume_index:]
        
        # Execute mission (with checkpoints at each stage)
        endpoint = self.mission.stages[0] if self.mission.stages else None
        try:
            results = self.orchestrator.execute_mission(self.mission)
            return {'success': True, 'results': results}
        except Exception as e:
            logger.error(f"[❌] Execution failed: {e}")
            return {'success': False, 'error': str(e)}
