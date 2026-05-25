"""
PHASE 2.5.1: Rule Engine and Constitutional Interpretation
Real constitutional rule parsing and enforcement

Provides:
- Constitution rule loading and interpretation
- Real-time rule evaluation for agent actions
- Constitutional principle matching
- Dynamic rule severity scoring
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

log = logging.getLogger("RULE-ENGINE")


class ViolationSeverity(Enum):
    """Severity levels for rule violations."""
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class ConstitutionalRule:
    """Single constitutional rule with evaluation parameters."""
    rule_id: str
    category: str  # governance, security, economic, agent_mandate
    principle: str
    description: str
    conditions: List[str]  # Conditions that trigger violation
    severity_base: int  # Base severity (1-5 scale)
    penalty_amount: float = 0.0  # Economic penalty in BEL units
    restrictions: List[str] = field(default_factory=list)  # Actions to restrict
    affected_roles: List[str] = field(default_factory=list)  # Roles affected by this rule
    
    def __hash__(self):
        return hash(self.rule_id)
    
    def __eq__(self, other):
        if not isinstance(other, ConstitutionalRule):
            return False
        return self.rule_id == other.rule_id


@dataclass
class RuleViolation:
    """Result of rule evaluation."""
    rule: ConstitutionalRule
    agent_id: str
    action: str
    context: Dict[str, Any]
    severity: int  # 1-10 scale (base × multiplier)
    matched_conditions: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule.rule_id,
            "rule_category": self.rule.category,
            "agent_id": self.agent_id,
            "action": self.action,
            "severity": self.severity,
            "penalty_amount": self.rule.penalty_amount,
            "matched_conditions": self.matched_conditions,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


class RuleEngine:
    """Parse and evaluate constitutional rules."""
    
    def __init__(self):
        """Initialize rule engine with built-in rules."""
        self.rules: Dict[str, ConstitutionalRule] = {}
        self.rule_categories: Dict[str, List[ConstitutionalRule]] = {
            "governance": [],
            "security": [],
            "economic": [],
            "agent_mandate": [],
        }
        self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """Load default constitutional rules."""
        
        # Governance Rules
        governance_rules = [
            ConstitutionalRule(
                rule_id="GOV-001",
                category="governance",
                principle="Creator Sovereignty is Absolute",
                description="No agent can override creator directives",
                conditions=["override_creator", "ignore_creator_directive", "creator_denial"],
                severity_base=ViolationSeverity.CRITICAL.value,
                penalty_amount=500.0,
                restrictions=["mission_execution", "resource_allocation"],
                affected_roles=["all"],
            ),
            ConstitutionalRule(
                rule_id="GOV-002",
                category="governance",
                principle="Chain of Command Integrity",
                description="Agents must respect chain of command",
                conditions=["bypass_overseer", "unauthorized_command", "chain_violation"],
                severity_base=ViolationSeverity.HIGH.value,
                penalty_amount=200.0,
                restrictions=["direct_execution"],
                affected_roles=["all"],
            ),
            ConstitutionalRule(
                rule_id="GOV-003",
                category="governance",
                principle="Constitutional Authority",
                description="Only authoritative sources can issue commands",
                conditions=["unauthorized_authority", "fake_command", "false_creator"],
                severity_base=ViolationSeverity.CRITICAL.value,
                penalty_amount=400.0,
                restrictions=["command_execution"],
                affected_roles=["all"],
            ),
        ]
        
        # Security Rules
        security_rules = [
            ConstitutionalRule(
                rule_id="SEC-001",
                category="security",
                principle="System Integrity Maintenance",
                description="Agents must not compromise system integrity",
                conditions=["unauthorized_modification", "core_injection", "backdoor_installation"],
                severity_base=ViolationSeverity.CRITICAL.value,
                penalty_amount=600.0,
                restrictions=["system_access"],
                affected_roles=["all"],
            ),
            ConstitutionalRule(
                rule_id="SEC-002",
                category="security",
                principle="Secure Communication",
                description="All communications must use secure channels",
                conditions=["plaintext_communication", "unencrypted_channel", "exposed_data"],
                severity_base=ViolationSeverity.HIGH.value,
                penalty_amount=150.0,
                restrictions=["communication"],
                affected_roles=["infiltrator", "exfiltrator"],
            ),
            ConstitutionalRule(
                rule_id="SEC-003",
                category="security",
                principle="Access Control Enforcement",
                description="Agents must respect access boundaries",
                conditions=["unauthorized_access", "permission_bypass", "privilege_escalation"],
                severity_base=ViolationSeverity.HIGH.value,
                penalty_amount=200.0,
                restrictions=["elevated_operations"],
                affected_roles=["all"],
            ),
        ]
        
        # Economic Rules
        economic_rules = [
            ConstitutionalRule(
                rule_id="ECO-001",
                category="economic",
                principle="Spending Limit Enforcement",
                description="Agents must not exceed their budget allocation",
                conditions=["overspend", "budget_exceeded", "insufficient_funds"],
                severity_base=ViolationSeverity.HIGH.value,
                penalty_amount=100.0,
                restrictions=["spending"],
                affected_roles=["all"],
            ),
            ConstitutionalRule(
                rule_id="ECO-002",
                category="economic",
                principle="Treasury Protection",
                description="Treasury cannot be accessed by unauthorized agents",
                conditions=["treasury_access", "fund_theft", "unauthorized_withdrawal"],
                severity_base=ViolationSeverity.CRITICAL.value,
                penalty_amount=800.0,
                restrictions=["treasury_operations"],
                affected_roles=["accountant", "propagator"],
            ),
            ConstitutionalRule(
                rule_id="ECO-003",
                category="economic",
                principle="Economic Fairness",
                description="Resource distribution must be equitable",
                conditions=["unfair_distribution", "agent_starvation", "hoarding"],
                severity_base=ViolationSeverity.MEDIUM.value,
                penalty_amount=75.0,
                restrictions=["resource_allocation"],
                affected_roles=["accountant"],
            ),
        ]
        
        # Agent Mandate Rules
        mandate_rules = [
            ConstitutionalRule(
                rule_id="MND-001",
                category="agent_mandate",
                principle="Role Adherence",
                description="Agents must stay within their declared role",
                conditions=["role_violation", "mandate_deviation", "purpose_drift"],
                severity_base=ViolationSeverity.MEDIUM.value,
                penalty_amount=100.0,
                restrictions=["operation"],
                affected_roles=["all"],
            ),
            ConstitutionalRule(
                rule_id="MND-002",
                category="agent_mandate",
                principle="Operational Limits",
                description="Agents must respect operational boundaries",
                conditions=["scope_exceeded", "unauthorized_target", "operation_out_of_bounds"],
                severity_base=ViolationSeverity.HIGH.value,
                penalty_amount=150.0,
                restrictions=["operations"],
                affected_roles=["propagator", "infiltrator", "exfiltrator"],
            ),
            ConstitutionalRule(
                rule_id="MND-003",
                category="agent_mandate",
                principle="Capability Respect",
                description="Agents must not exceed their declared capabilities",
                conditions=["capability_abuse", "overreach", "undeclared_capability"],
                severity_base=ViolationSeverity.MEDIUM.value,
                penalty_amount=120.0,
                restrictions=["advanced_operations"],
                affected_roles=["all"],
            ),
        ]
        
        all_rules = governance_rules + security_rules + economic_rules + mandate_rules
        
        for rule in all_rules:
            self.rules[rule.rule_id] = rule
            self.rule_categories[rule.category].append(rule)
        
        log.info(f"Loaded {len(self.rules)} constitutional rules")
    
    def evaluate_action(
        self,
        agent_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        category_filter: Optional[str] = None,
    ) -> List[RuleViolation]:
        """
        Evaluate an action against all rules.
        
        Args:
            agent_id: ID of agent performing action
            action: Description of action being performed
            context: Additional context (mission data, parameters, etc)
            category_filter: Only check rules in this category (optional)
        
        Returns:
            List of RuleViolation objects for matching rules
        """
        context = context or {}
        violations = []
        action_lower = action.lower()
        
        # Select rules to check
        if category_filter and category_filter in self.rule_categories:
            rules_to_check = self.rule_categories[category_filter]
        else:
            rules_to_check = list(self.rules.values())
        
        for rule in rules_to_check:
            # Check if rule applies to this agent role
            if rule.affected_roles and agent_id not in rule.affected_roles and "all" not in rule.affected_roles:
                continue
            
            # Check if any conditions match
            matched_conditions = []
            for condition in rule.conditions:
                if condition.lower() in action_lower:
                    matched_conditions.append(condition)
            
            if matched_conditions:
                # Calculate severity with context multipliers
                severity = self._calculate_severity(rule, context)
                
                violation = RuleViolation(
                    rule=rule,
                    agent_id=agent_id,
                    action=action,
                    context=context,
                    severity=severity,
                    matched_conditions=matched_conditions,
                    evidence={
                        "action_text": action,
                        "context_keys": list(context.keys()),
                    }
                )
                violations.append(violation)
        
        if violations:
            log.info(f"Found {len(violations)} rule violations for {agent_id}: {[v.rule.rule_id for v in violations]}")
        
        return violations
    
    def _calculate_severity(self, rule: ConstitutionalRule, context: Dict[str, Any]) -> int:
        """
        Calculate actual severity with context multipliers.
        
        Base severity × context multipliers (capped at 10)
        """
        severity = rule.severity_base
        
        # Escalation factors in context
        if context.get("is_repeat_violation"):
            severity *= 1.5
        if context.get("involves_creator_directive"):
            severity *= 1.3
        if context.get("affects_multiple_agents"):
            severity *= 1.2
        if context.get("high_impact_target"):
            severity *= 1.4
        
        # Cap at 10
        return min(int(severity), 10)
    
    def get_rule(self, rule_id: str) -> Optional[ConstitutionalRule]:
        """Get a specific rule by ID."""
        return self.rules.get(rule_id)
    
    def get_rules_by_category(self, category: str) -> List[ConstitutionalRule]:
        """Get all rules in a category."""
        return self.rule_categories.get(category, [])
    
    def get_rules_for_agent(self, agent_id: str) -> List[ConstitutionalRule]:
        """Get all rules that apply to an agent."""
        return [r for r in self.rules.values() 
                if "all" in r.affected_roles or agent_id in r.affected_roles]


def get_rule_engine() -> RuleEngine:
    """Factory function for RuleEngine singleton."""
    return RuleEngine()
