"""
PHASE 2.5.3: Penalty Calculator
Dynamic penalty calculation for constitution violations

Provides:
- Penalty scoring based on violation severity
- Dynamic multipliers for repeat offenses
- Economic impact assessment
- Audit trail for all penalties
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from laws.rule_engine import RuleViolation, ViolationSeverity

log = logging.getLogger("PENALTY-CALC")


@dataclass
class PenaltyRecord:
    """Record of a penalty applied to an agent."""
    penalty_id: str
    agent_id: str
    violation_rule_id: str
    violation_severity: int
    base_penalty: float
    multipliers: Dict[str, float]  # factor_name -> multiplier_value
    final_penalty: float
    penalty_type: str  # financial, restriction, suspension, termination
    duration_seconds: Optional[int] = None
    imposed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    executed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "penalty_id": self.penalty_id,
            "agent_id": self.agent_id,
            "violation_rule_id": self.violation_rule_id,
            "violation_severity": self.violation_severity,
            "base_penalty": self.base_penalty,
            "multipliers": self.multipliers,
            "final_penalty": self.final_penalty,
            "penalty_type": self.penalty_type,
            "duration_seconds": self.duration_seconds,
            "imposed_at": self.imposed_at,
            "executed": self.executed,
        }


@dataclass
class AgentPenaltyStatus:
    """Current penalty status for an agent."""
    agent_id: str
    total_penalties_imposed: float
    total_penalties_executed: float
    penalties_by_type: Dict[str, float]
    violation_history: List[str]
    is_suspended: bool
    suspension_end_time: Optional[str]
    is_quarantined: bool
    latest_violation: Optional[str]
    violation_count: int
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "total_penalties_imposed": self.total_penalties_imposed,
            "total_penalties_executed": self.total_penalties_executed,
            "penalties_by_type": self.penalties_by_type,
            "violation_history": self.violation_history,
            "is_suspended": self.is_suspended,
            "suspension_end_time": self.suspension_end_time,
            "is_quarantined": self.is_quarantined,
            "latest_violation": self.latest_violation,
            "violation_count": self.violation_count,
        }


class PenaltyCalculator:
    """Dynamic penalty calculation for constitutional violations."""
    
    def __init__(self):
        """Initialize penalty calculator."""
        self.penalties: Dict[str, PenaltyRecord] = {}
        self.agent_penalties: Dict[str, List[PenaltyRecord]] = {}
        self.violation_history: Dict[str, List[str]] = {}
        self.active_suspensions: Dict[str, str] = {}  # agent_id -> suspension_end_time
        self.active_quarantines: Dict[str, bool] = {}  # agent_id -> is_quarantined
        
        # Penalty base amounts (in BEL units)
        self.penalty_base_amounts = {
            "GOV": 500.0,   # Governance violations: high penalty
            "SEC": 400.0,   # Security violations: medium-high
            "ECO": 300.0,   # Economic violations: medium
            "MND": 200.0,   # Mandate violations: lower (agent can still operate)
        }
        
        # Dynamic multipliers
        self.multiplier_repeat_offense = 1.5  # 50% increase per repeat
        self.multiplier_public_harm = 2.0     # 100% if harms other agents
        self.multiplier_creator_directive = 2.5  # 150% if violates creator directive
        self.multiplier_system_impact = 1.8   # 80% if impacts system integrity
    
    def calculate_penalty(
        self,
        violation: RuleViolation,
        agent_id: str,
        violation_count: int = 0,
        prior_violations: Optional[List[str]] = None,
    ) -> PenaltyRecord:
        """
        Calculate penalty for a violation.
        
        Args:
            violation: RuleViolation object
            agent_id: ID of offending agent
            violation_count: Number of prior violations by this agent
            prior_violations: List of prior violation rule IDs
        
        Returns:
            PenaltyRecord with calculated penalty
        """
        import uuid
        
        prior_violations = prior_violations or []
        
        log.info(f"Calculating penalty for {agent_id}: {violation.rule.rule_id} (violation #{violation_count})")
        
        # Step 1: Get base penalty amount
        rule_category_prefix = violation.rule.rule_id.split("-")[0]
        base_penalty = self.penalty_base_amounts.get(rule_category_prefix, 200.0)
        
        # If rule defines specific penalty, use that
        if violation.rule.penalty_amount > 0:
            base_penalty = violation.rule.penalty_amount
        
        # Step 2: Calculate multipliers
        multipliers = {}
        severity_multiplier = violation.severity / 5.0  # Normalize 1-10 to 0.2-2.0
        multipliers["severity"] = severity_multiplier
        
        # Repeat offense multiplier - escalates with each violation
        if violation_count > 0:
            repeat_multiplier = 1 + (violation_count * 0.5)
            multipliers["repeat_offense"] = min(repeat_multiplier, 3.0)  # Cap at 3x
        
        # Public harm multiplier
        if violation.context.get("affects_multiple_agents"):
            multipliers["public_harm"] = self.multiplier_public_harm
        
        # Creator directive multiplier
        if violation.context.get("involves_creator_directive"):
            multipliers["creator_directive"] = self.multiplier_creator_directive
        
        # System impact multiplier
        if violation.context.get("system_impact"):
            multipliers["system_impact"] = self.multiplier_system_impact
        
        # Step 3: Calculate final penalty
        total_multiplier = 1.0
        for multiplier in multipliers.values():
            total_multiplier *= multiplier
        
        final_penalty = base_penalty * total_multiplier
        
        # Step 4: Determine penalty type and duration
        penalty_type, duration = self._determine_penalty_type(violation.severity, violation_count)
        
        # Create penalty record
        penalty = PenaltyRecord(
            penalty_id=f"PEN-{str(uuid.uuid4())[:8]}",
            agent_id=agent_id,
            violation_rule_id=violation.rule.rule_id,
            violation_severity=violation.severity,
            base_penalty=base_penalty,
            multipliers=multipliers,
            final_penalty=final_penalty,
            penalty_type=penalty_type,
            duration_seconds=duration,
        )
        
        # Track penalty
        self.penalties[penalty.penalty_id] = penalty
        if agent_id not in self.agent_penalties:
            self.agent_penalties[agent_id] = []
        self.agent_penalties[agent_id].append(penalty)
        
        log.info(f"Penalty calculated: {penalty_type} of {final_penalty} BEL for {agent_id}")
        
        return penalty
    
    def _determine_penalty_type(self, severity: int, violation_count: int) -> Tuple[str, Optional[int]]:
        """
        Determine penalty type based on severity and violation history.
        
        Returns:
            (penalty_type, duration_seconds or None)
        """
        # Severity-based escalation
        if severity <= 2:
            if violation_count == 0:
                return ("warning", None)
            else:
                return ("financial_fine", None)
        
        elif severity <= 4:
            return ("financial_fine", None)
        
        elif severity <= 6:
            if violation_count > 2:
                return ("restriction", 3600)  # 1 hour restriction
            else:
                return ("financial_fine", None)
        
        elif severity <= 8:
            if violation_count > 1:
                return ("suspension", 7200)  # 2 hour suspension
            else:
                return ("restriction", 7200)
        
        else:  # severity > 8
            if violation_count > 0:
                return ("quarantine", None)  # Indefinite quarantine
            else:
                return ("suspension", 86400)  # 24 hour suspension
    
    def execute_penalty(self, penalty: PenaltyRecord) -> bool:
        """
        Execute penalty on the agent (economic deduction).
        
        Returns:
            True if penalty executed successfully
        """
        try:
            # Try to deduct from agent's balance
            try:
                from economic.wallet import Wallet as EconomicWallet
            except Exception:
                EconomicWallet = None

            agent_wallet = None
            try:
                if callable(EconomicWallet):
                    agent_wallet = EconomicWallet(agent_id=penalty.agent_id)
            except Exception:
                agent_wallet = None

            if penalty.penalty_type in ["warning"]:
                log.warning(f"Warning issued to {penalty.agent_id}")
                penalty.executed = True
                return True

            # Deduct penalty from agent if wallet available
            if agent_wallet is not None:
                try:
                    agent_wallet.balance -= penalty.final_penalty
                    penalty.executed = True
                except Exception:
                    penalty.executed = False

            log.info(f"Penalty executed: {penalty.final_penalty} BEL deducted from {penalty.agent_id}")

            # Apply restrictions/suspensions if applicable
            if penalty.penalty_type == "restriction":
                self._apply_restriction(penalty.agent_id, penalty.duration_seconds)
            elif penalty.penalty_type == "suspension":
                self._apply_suspension(penalty.agent_id, penalty.duration_seconds)
            elif penalty.penalty_type == "quarantine":
                self._apply_quarantine(penalty.agent_id)
            
            return True
        
        except Exception as e:
            log.error(f"Failed to execute penalty {penalty.penalty_id}: {e}")
            return False
    
    def _apply_restriction(self, agent_id: str, duration_seconds: Optional[int]) -> None:
        """Apply operational restriction to agent."""
        log.info(f"Restricting agent {agent_id} for {duration_seconds}s")
        # Would integrate with agent control system
    
    def _apply_suspension(self, agent_id: str, duration_seconds: Optional[int]) -> None:
        """Suspend agent operations."""
        from datetime import datetime, timedelta
        
        if duration_seconds:
            end_time = (datetime.utcnow() + timedelta(seconds=duration_seconds)).isoformat()
            self.active_suspensions[agent_id] = end_time
            log.warning(f"Agent {agent_id} suspended until {end_time}")
        else:
            self.active_suspensions[agent_id] = "indefinite"
            log.error(f"Agent {agent_id} suspended indefinitely")
    
    def _apply_quarantine(self, agent_id: str) -> None:
        """Quarantine agent (isolate from mesh)."""
        self.active_quarantines[agent_id] = True
        log.critical(f"Agent {agent_id} QUARANTINED")
    
    def get_agent_penalty_status(self, agent_id: str) -> AgentPenaltyStatus:
        """Get current penalty status for an agent."""
        agent_penalties = self.agent_penalties.get(agent_id, [])
        
        total_imposed = sum(p.final_penalty for p in agent_penalties)
        total_executed = sum(p.final_penalty for p in agent_penalties if p.executed)
        
        penalties_by_type = {}
        for penalty in agent_penalties:
            penalties_by_type[penalty.penalty_type] = \
                penalties_by_type.get(penalty.penalty_type, 0) + penalty.final_penalty
        
        violation_history = self.violation_history.get(agent_id, [])
        
        is_suspended = agent_id in self.active_suspensions
        suspension_end = self.active_suspensions.get(agent_id)
        
        is_quarantined = self.active_quarantines.get(agent_id, False)
        
        latest_violation = violation_history[-1] if violation_history else None
        
        return AgentPenaltyStatus(
            agent_id=agent_id,
            total_penalties_imposed=total_imposed,
            total_penalties_executed=total_executed,
            penalties_by_type=penalties_by_type,
            violation_history=violation_history,
            is_suspended=is_suspended,
            suspension_end_time=suspension_end,
            is_quarantined=is_quarantined,
            latest_violation=latest_violation,
            violation_count=len(violation_history),
        )
    
    def can_execute_mission(self, agent_id: str) -> Tuple[bool, str]:
        """
        Check if agent can execute missions based on penalty status.
        
        Returns:
            (can_execute, reason)
        """
        status = self.get_agent_penalty_status(agent_id)
        
        if status.is_quarantined:
            return (False, "Agent is quarantined")
        
        if status.is_suspended:
            return (False, f"Agent is suspended until {status.suspension_end_time}")
        
        return (True, "Agent can execute missions")
    
    def record_violation(self, agent_id: str, violation_rule_id: str) -> None:
        """Record a violation for an agent."""
        if agent_id not in self.violation_history:
            self.violation_history[agent_id] = []
        
        self.violation_history[agent_id].append(violation_rule_id)
    
    def get_penalty_history(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get penalty history."""
        if agent_id:
            penalties = self.agent_penalties.get(agent_id, [])
        else:
            penalties = list(self.penalties.values())
        
        return [p.to_dict() for p in penalties]


def get_penalty_calculator() -> PenaltyCalculator:
    """Factory function for PenaltyCalculator singleton."""
    return PenaltyCalculator()
