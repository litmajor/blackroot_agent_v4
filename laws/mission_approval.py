"""
PHASE 2.5.2: Mission Approval Engine
Real mission approval with constitutional rule checking

Provides:
- Mission evaluation against constitutional rules
- Risk assessment and impact analysis
- Approval/denial decisions with justification
- Integration with EnforcerAgent
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from laws.rule_engine import RuleEngine, RuleViolation, ConstitutionalRule

log = logging.getLogger("MISSION-APPROVAL")


class ApprovalStatus(Enum):
    """Mission approval decision."""
    APPROVED = "approved"
    CONDITIONAL_APPROVAL = "conditional_approval"
    DENIED = "denied"
    REQUIRES_REVIEW = "requires_review"


class RiskLevel(Enum):
    """Risk assessment for a mission."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RiskAssessment:
    """Risk analysis for a mission."""
    mission_id: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    primary_risks: List[str]
    secondary_risks: List[str]
    impact_radius: str  # "local", "zone", "mesh"
    estimated_cost: float  # BEL units
    resource_impact: Dict[str, float]  # Resource type → impact
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "mission_id": self.mission_id,
            "risk_level": self.risk_level.name,
            "risk_score": self.risk_score,
            "primary_risks": self.primary_risks,
            "secondary_risks": self.secondary_risks,
            "impact_radius": self.impact_radius,
            "estimated_cost": self.estimated_cost,
            "resource_impact": self.resource_impact,
            "timestamp": self.timestamp,
        }


@dataclass
class ApprovalDecision:
    """Mission approval decision with rationale."""
    mission_id: str
    mission_name: str
    agent_id: str
    status: ApprovalStatus
    decision_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    rule_violations: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[RiskAssessment] = None
    conditions: List[str] = field(default_factory=list)  # Conditions for approval
    justification: str = ""
    approver_id: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "mission_id": self.mission_id,
            "mission_name": self.mission_name,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "decision_timestamp": self.decision_timestamp,
            "rule_violations": self.rule_violations,
            "risk_assessment": self.risk_assessment.to_dict() if self.risk_assessment else None,
            "conditions": self.conditions,
            "justification": self.justification,
            "approver_id": self.approver_id,
        }


class MissionApprovalEngine:
    """Evaluate and approve/deny missions based on constitutional rules."""
    
    def __init__(self, rule_engine: Optional['RuleEngine'] = None):
        """
        Initialize mission approval engine.
        
        Args:
            rule_engine: RuleEngine instance (creates new if None)
        """
        self.rule_engine = rule_engine or RuleEngine()
        self.approval_history: List[ApprovalDecision] = []
        self.risk_thresholds = {
            RiskLevel.LOW: (0, 25),
            RiskLevel.MEDIUM: (25, 50),
            RiskLevel.HIGH: (50, 75),
            RiskLevel.CRITICAL: (75, 100),
        }
    
    def evaluate_mission(
        self,
        mission_id: str,
        mission_name: str,
        agent_id: str,
        mission_parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ApprovalDecision:
        """
        Evaluate a mission for approval.
        
        Args:
            mission_id: Unique mission identifier
            mission_name: Human-readable mission name (deep_scan, monitor_changes, etc)
            agent_id: ID of agent requesting mission
            mission_parameters: Mission parameters (target, port, aggressive, etc)
            context: Additional context (operator_id, resource_availability, etc)
        
        Returns:
            ApprovalDecision with status and justification
        """
        context = context or {}
        
        log.info(f"Evaluating mission {mission_id} ({mission_name}) for agent {agent_id}")
        
        # Step 1: Check for rule violations
        violations = self._check_rule_violations(mission_name, agent_id, mission_parameters, context)
        
        # Step 2: Perform risk assessment
        risk_assessment = self._assess_mission_risk(mission_name, mission_parameters, violations)
        
        # Step 3: Determine approval status
        status, conditions, justification = self._determine_approval_status(
            mission_name, agent_id, violations, risk_assessment
        )
        
        decision = ApprovalDecision(
            mission_id=mission_id,
            mission_name=mission_name,
            agent_id=agent_id,
            status=status,
            rule_violations=[v.to_dict() for v in violations],
            risk_assessment=risk_assessment,
            conditions=conditions,
            justification=justification,
            approver_id="ENFORCER-AGENT",
        )
        
        self.approval_history.append(decision)
        
        log.info(f"Mission {mission_id} approval: {status.value} - {justification}")
        
        return decision
    
    def _check_rule_violations(
        self,
        mission_name: str,
        agent_id: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[RuleViolation]:
        """Check if mission violates any constitutional rules."""
        
        # Build action description from mission parameters
        action_parts = [f"{mission_name} mission"]
        
        if "target" in parameters:
            action_parts.append(f"on target {parameters['target']}")
        
        if "service" in parameters:
            action_parts.append(f"for service {parameters['service']}")
        
        action_description = " ".join(action_parts)
        
        # Evaluate against rules
        violations = self.rule_engine.evaluate_action(
            agent_id=agent_id,
            action=action_description,
            context={
                **context,
                "mission_name": mission_name,
                "mission_parameters": parameters,
            }
        )
        
        return violations
    
    def _assess_mission_risk(
        self,
        mission_name: str,
        parameters: Dict[str, Any],
        violations: List[RuleViolation],
    ) -> RiskAssessment:
        """
        Assess risk level and impact of mission.
        
        Returns:
            RiskAssessment with score, level, and impact analysis
        """
        risk_score = 0.0
        primary_risks = []
        secondary_risks = []
        resource_impact = {}
        
        # Base risk by mission type
        mission_risk_base = {
            "deep_scan": 20,
            "monitor_changes": 15,
            "apply_patch": 35,
            "propagate": 60,
            "exfiltrate": 70,
            "infiltrate": 80,
        }
        
        risk_score = mission_risk_base.get(mission_name, 25)
        
        # Adjust for parameters
        if parameters.get("aggressive"):
            risk_score += 15
            primary_risks.append("aggressive_mode_enabled")
        
        if parameters.get("target") and "critical" in str(parameters.get("target")).lower():
            risk_score += 20
            primary_risks.append("critical_target_designation")
        
        if parameters.get("dry_run"):
            risk_score -= 10  # Reduce risk for dry-run
        
        # Add violation-based risk
        for violation in violations:
            risk_score += (violation.severity * 3)
            primary_risks.append(f"rule_violation:{violation.rule.rule_id}")
        
        # Add multi-agent impact risk
        if parameters.get("affects_agents"):
            risk_score += 15
            primary_risks.append("multi_agent_impact")
        
        # Determine impact radius
        if mission_name in ["propagate", "infiltrate"]:
            impact_radius = "mesh"
        elif parameters.get("target"):
            impact_radius = "zone"
        else:
            impact_radius = "local"
        
        # Cap score at 100
        risk_score = min(risk_score, 100)
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= risk_score < max_score:
                risk_level = level
                break
        if risk_score >= 75:
            risk_level = RiskLevel.CRITICAL
        
        # Estimate cost and resource impact
        estimated_cost = self._estimate_mission_cost(mission_name, parameters)
        resource_impact = self._estimate_resource_impact(mission_name, parameters)
        
        assessment = RiskAssessment(
            mission_id=f"{mission_name}",
            risk_level=risk_level,
            risk_score=risk_score,
            primary_risks=primary_risks,
            secondary_risks=secondary_risks,
            impact_radius=impact_radius,
            estimated_cost=estimated_cost,
            resource_impact=resource_impact,
        )
        
        return assessment
    
    def _estimate_mission_cost(self, mission_name: str, parameters: Dict[str, Any]) -> float:
        """Estimate economic cost of mission execution."""
        base_costs = {
            "deep_scan": 50,
            "monitor_changes": 20,
            "apply_patch": 100,
            "propagate": 200,
            "exfiltrate": 150,
            "infiltrate": 180,
        }
        
        cost = base_costs.get(mission_name, 50)
        
        # Adjust for parameters
        if parameters.get("aggressive"):
            cost *= 1.3
        
        if parameters.get("dry_run"):
            cost *= 0.5
        
        return cost
    
    def _estimate_resource_impact(
        self,
        mission_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resource usage impact."""
        resource_impact = {
            "cpu": 0.0,
            "memory": 0.0,
            "network": 0.0,
            "storage": 0.0,
        }
        
        # Mission-specific impacts
        if mission_name == "deep_scan":
            resource_impact["cpu"] = 0.4
            resource_impact["network"] = 0.6
        elif mission_name == "monitor_changes":
            resource_impact["cpu"] = 0.2
            resource_impact["memory"] = 0.3
        elif mission_name == "apply_patch":
            resource_impact["cpu"] = 0.5
            resource_impact["storage"] = 0.1
        elif mission_name == "propagate":
            resource_impact["cpu"] = 0.7
            resource_impact["network"] = 0.8
            resource_impact["memory"] = 0.5
        elif mission_name in ["exfiltrate", "infiltrate"]:
            resource_impact["network"] = 0.8
            resource_impact["memory"] = 0.4
        
        return {k: min(v, 1.0) for k, v in resource_impact.items()}
    
    def _determine_approval_status(
        self,
        mission_name: str,
        agent_id: str,
        violations: List[RuleViolation],
        risk_assessment: RiskAssessment,
    ) -> Tuple[ApprovalStatus, List[str], str]:
        """
        Determine approval status based on violations and risk.
        
        Returns:
            (status, conditions, justification)
        """
        conditions = []
        
        # Critical violations = automatic denial
        critical_violations = [v for v in violations if v.severity >= 8]
        if critical_violations:
            return (
                ApprovalStatus.DENIED,
                [],
                f"Critical rule violations: {', '.join([v.rule.rule_id for v in critical_violations])}"
            )
        
        # High-risk missions require review
        if risk_assessment.risk_level == RiskLevel.CRITICAL:
            conditions.append("requires_risk_review")
            conditions.append("enforce_dry_run_first")
            return (
                ApprovalStatus.REQUIRES_REVIEW,
                conditions,
                f"High risk mission (score: {risk_assessment.risk_score}/100)"
            )
        
        # High violations or medium risks get conditional approval
        high_violations = [v for v in violations if v.severity >= 5]
        if high_violations or risk_assessment.risk_level == RiskLevel.HIGH:
            # Add safety conditions
            if "deep_scan" in mission_name or "apply_patch" in mission_name:
                conditions.append("enforce_dry_run_first")
            conditions.append("log_all_actions")
            conditions.append("monitor_execution")
            
            return (
                ApprovalStatus.CONDITIONAL_APPROVAL,
                conditions,
                f"Conditional approval with {len(conditions)} safety conditions"
            )
        
        # No critical issues = full approval
        if risk_assessment.risk_level == RiskLevel.LOW:
            return (
                ApprovalStatus.APPROVED,
                [],
                f"Low-risk mission (score: {risk_assessment.risk_score}/100)"
            )
        else:
            return (
                ApprovalStatus.APPROVED,
                [],
                f"Mission approved (score: {risk_assessment.risk_score}/100)"
            )
    
    def get_approval_history(self) -> List[Dict]:
        """Get history of all mission approvals."""
        return [d.to_dict() for d in self.approval_history]
    
    def check_agent_approval_rate(self, agent_id: str) -> Tuple[int, int, float]:
        """
        Get approval statistics for an agent.
        
        Returns:
            (approved_count, denied_count, approval_rate)
        """
        agent_decisions = [d for d in self.approval_history if d.agent_id == agent_id]
        
        if not agent_decisions:
            return (0, 0, 1.0)
        
        approved = sum(1 for d in agent_decisions if d.status == ApprovalStatus.APPROVED)
        denied = sum(1 for d in agent_decisions if d.status == ApprovalStatus.DENIED)
        
        approval_rate = approved / len(agent_decisions) if agent_decisions else 0.0
        
        return (approved, denied, approval_rate)


def get_mission_approval_engine(rule_engine: Optional[RuleEngine] = None) -> MissionApprovalEngine:
    """Factory function for MissionApprovalEngine singleton."""
    return MissionApprovalEngine(rule_engine=rule_engine)
