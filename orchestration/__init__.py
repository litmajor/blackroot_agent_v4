"""
Orchestration Package - Agent Coordination & Mission Management

Modules:
- mission_orchestrator: Main orchestrator for multi-stage missions
- handoff_protocol: Typed, versioned agent-to-agent communication
- checkpoint_resume: Crash recovery and state persistence
"""

from orchestration.mission_orchestrator import (
    MissionOrchestrator,
    OrchestrationMission,
    MissionStage,
    MissionStageState,
    HandoffResult,
    MissionPriority
)

from orchestration.handoff_protocol import (
    HandoffValidator,
    HandoffChain,
    ScanHandoff,
    EscalateHandoff,
    ExtractHandoff,
    ExfilHandoff,
    HandoffProtocolVersion
)

from orchestration.checkpoint_resume import (
    CheckpointManager,
    Checkpoint,
    ResumableExecution
)

__all__ = [
    'MissionOrchestrator',
    'OrchestrationMission',
    'MissionStage',
    'MissionStageState',
    'HandoffResult',
    'MissionPriority',
    'HandoffValidator',
    'HandoffChain',
    'ScanHandoff',
    'EscalateHandoff',
    'ExtractHandoff',
    'ExfilHandoff',
    'HandoffProtocolVersion',
    'CheckpointManager',
    'Checkpoint',
    'ResumableExecution',
]
