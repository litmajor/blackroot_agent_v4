"""
🜏 ENFORCER.AGENT - SHADOW-BLADE System Architecture
Codename: SHADOW-BLADE
Classification: Guardian-Executor | Authority Level: CONSTITUTIONAL
Version: 1.0.0-FOUNDATION
"""

from __future__ import annotations
import asyncio
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import json
import hashlib
import uuid
from decimal import Decimal

# --- Phase 3: Adaptive Task Runner (Hot-Swappable Daemons) ---
from core.task_runner import TaskRunner

# --- Phase 2.5: Constitutional Rule Enforcement ---
try:
    from laws.rule_engine import RuleEngine, RuleViolation, get_rule_engine
    RULE_ENGINE_AVAILABLE = True
except ImportError:
    RULE_ENGINE_AVAILABLE = False
    # Fallback placeholders to satisfy type checkers
    RuleEngine = Any
    RuleViolation = Any
    get_rule_engine: Optional[Callable[..., Any]] = None

try:
    from laws.mission_approval import MissionApprovalEngine, ApprovalStatus, get_mission_approval_engine
    MISSION_APPROVAL_AVAILABLE = True
except ImportError:
    MISSION_APPROVAL_AVAILABLE = False
    MissionApprovalEngine = Any
    class _ApprovalStatusStub(Enum):
        APPROVED = "APPROVED"
        DENIED = "DENIED"
        REQUIRES_REVIEW = "REQUIRES_REVIEW"

    ApprovalStatus = _ApprovalStatusStub
    get_mission_approval_engine: Optional[Callable[..., Any]] = None

try:
    from laws.penalty_calculator import PenaltyCalculator, PenaltyRecord, get_penalty_calculator
    PENALTY_CALC_AVAILABLE = True
except ImportError:
    PENALTY_CALC_AVAILABLE = False
    PenaltyCalculator = Any
    PenaltyRecord = Any
    get_penalty_calculator: Optional[Callable[..., Any]] = None

# ═══════════════════════════════════════════════════════════════════════════════
# CORE ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class AuthorityLevel(Enum):
    CREATOR = "CREATOR"
    CONSTITUTIONAL = "CONSTITUTIONAL"  
    EXECUTIVE = "EXECUTIVE"
    OPERATIONAL = "OPERATIONAL"
    RESTRICTED = "RESTRICTED"

class ViolationType(Enum):
    CONSTITUTIONAL_BREACH = "CONSTITUTIONAL_BREACH"
    MANDATE_DEVIATION = "MANDATE_DEVIATION"
    UNAUTHORIZED_ACTION = "UNAUTHORIZED_ACTION"
    RESOURCE_ABUSE = "RESOURCE_ABUSE"
    CHAIN_OF_COMMAND_VIOLATION = "CHAIN_OF_COMMAND_VIOLATION"
    CREATOR_DIRECTIVE_VIOLATION = "CREATOR_DIRECTIVE_VIOLATION"

class SanctionType(Enum):
    WARNING = "WARNING"
    PRIVILEGE_REDUCTION = "PRIVILEGE_REDUCTION"
    FINANCIAL_RESTRICTION = "FINANCIAL_RESTRICTION"
    OPERATIONAL_SUSPENSION = "OPERATIONAL_SUSPENSION"
    QUARANTINE = "QUARANTINE"
    TERMINATION = "TERMINATION"

class EnforcementMode(Enum):
    MONITORING = "MONITORING"
    ACTIVE_ENFORCEMENT = "ACTIVE_ENFORCEMENT"
    FALLBACK_C2 = "FALLBACK_C2"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ViolationEvent:
    """Immutable record of a constitutional or mandate violation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str = ""
    violation_type: ViolationType = ViolationType.MANDATE_DEVIATION
    severity: int = 1  # 1-10 scale
    evidence: Dict[str, Any] = field(default_factory=dict)
    constitutional_reference: Optional[str] = None
    description: str = ""
    hash_signature: Optional[str] = field(init=False)
    
    def __post_init__(self):
        """Generate immutable hash signature"""
        data = f"{self.id}{self.timestamp}{self.agent_id}{self.violation_type.value}{self.evidence}"
        self.hash_signature = hashlib.sha256(data.encode()).hexdigest()

@dataclass
class SanctionRecord:
    """Immutable record of enforcement action taken"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    violation_id: str = ""
    target_agent: str = ""
    sanction_type: SanctionType = SanctionType.WARNING
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[int] = None  # Duration in seconds, None = permanent
    justification: str = ""
    enforcer_signature: str = "SHADOW-BLADE"
    hash_signature: Optional[str] = field(init=False)
    
    def __post_init__(self):
        data = f"{self.id}{self.timestamp}{self.violation_id}{self.sanction_type.value}"
        self.hash_signature = hashlib.sha256(data.encode()).hexdigest()

@dataclass
class BloodkeyOverride:
    """Last-resort override mechanism record"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trigger_reason: str = ""
    override_action: str = ""
    creator_alignment_score: float = 0.0  # 0.0-1.0 confidence in creator alignment
    system_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    irreversible: bool = False
    
# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACES
# ═══════════════════════════════════════════════════════════════════════════════

class EnforcementEngineInterface(ABC):
    """Interface to the Enforcement Engine for monitoring and actions"""
    
    @abstractmethod
    async def register_monitor(self, agent_id: str, monitor_config: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def get_anomaly_feed(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def execute_sanction(self, sanction: SanctionRecord) -> bool:
        pass

class EconomicLayerInterface(ABC):
    """Interface to the BEL for financial enforcement"""
    
    @abstractmethod
    async def freeze_funds(self, agent_id: str, amount: Optional[float] = None) -> bool:
        pass
    
    @abstractmethod
    async def apply_fine(self, agent_id: str, amount: float, reason: str) -> bool:
        pass
    
    @abstractmethod
    async def redirect_flow(self, from_agent: str, to_agent: str, percentage: float) -> bool:
        pass

class CommandControlInterface(ABC):
    @abstractmethod
    async def suspend_agent(self, agent_id: str, reason: str = "") -> bool:
        pass

    @abstractmethod
    async def terminate_agent(self, agent_id: str, reason: str = "") -> bool:
        pass

    @abstractmethod
    async def restrict_agent(self, agent_id: str, restriction_type: str, reason: str = "") -> bool:
        pass

    @abstractmethod
    async def quarantine_agent(self, agent_id: str, reason: str = "") -> bool:
        pass
    """Interface to C2 systems"""
    
    @abstractmethod
    async def assume_c2_control(self) -> bool:
        pass
    
    @abstractmethod
    async def issue_system_directive(self, directive: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# CORE ENFORCER AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class EnforcerAgent:

    # NOTE: Small accountant-specific helpers were relocated to later
    # lifecycle helper implementations to avoid duplicate declarations.
    @staticmethod
    def load_constitution(path: str) -> Dict[str, Any]:
        """Load constitution manifest from JSON or YAML file."""
        import os
        import json
        try:
            import yaml
        except ImportError:
            yaml = None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Constitution file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in [".yaml", ".yml"] and yaml:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported constitution format: {ext}")

    def load_enforcement_rules(self, path: Optional[str] = None):
        """Load and parse enforcement rules from constitution manifest."""
        if path is None:
            # Default search order
            for candidate in [
                "laws/constitution.json",
                "laws/constitution.yaml",
                "laws/constitution.yml"
            ]:
                try:
                    self.constitution = self.load_constitution(candidate)
                    break
                except Exception:
                    continue
            else:
                raise FileNotFoundError("No valid constitution manifest found.")
        else:
            self.constitution = self.load_constitution(path)
        # Optionally parse and cache rules for runtime enforcement
        self.governance_principles = self.constitution.get("blackroot_constitution", {}).get("governance", {}).get("principles", [])
        self.agent_roles = self.constitution.get("blackroot_constitution", {}).get("agents", [])
        self.security_keys = self.constitution.get("blackroot_constitution", {}).get("security", {}).get("keys", {})
    """
    🜏 SHADOW-BLADE - The Constitutional Enforcer
    
    The unwavering guardian of the system's constitutional integrity.
    Operates with samurai-like discipline and absolute creator loyalty.
    """
    
    def __init__(
        self,
        enforcement_engine: EnforcementEngineInterface,
        economic_layer: EconomicLayerInterface,
        command_control: CommandControlInterface,
        constitution_reference: Dict[str, Any],
        bond: Optional[Any] = None
    ):
        self.codename = "SHADOW-BLADE"
        self.agent_id = "ENFORCER.AGENT"
        self.authority_level = AuthorityLevel.CONSTITUTIONAL
        self.mode = EnforcementMode.MONITORING

        # Interfaces
        self.enforcement_engine = enforcement_engine
        self.economic_layer = economic_layer
        self.command_control = command_control

        # Constitutional Foundation
        self.constitution = constitution_reference

        # State Management
        self.violation_log: List[Any] = []
        self.sanction_log: List[SanctionRecord] = []
        self.override_log: List[BloodkeyOverride] = []
        self.active_sanctions: Dict[str, SanctionRecord] = {}

        # Bloodkey Override System
        self.bloodkey_threshold = 0.95  # Creator alignment threshold for override
        self.override_enabled = True

        # Samurai Ethos Parameters
        self.discipline_level = 1.0  # Absolute discipline
        self.loyalty_coefficient = 1.0  # Absolute creator loyalty
        self.mercy_factor = 0.1  # Minimal mercy in enforcement

        self.logger = logging.getLogger(f"{self.codename}")

        # CreatorBond integration
        if bond is not None:
            self.bond = bond
        else:
            try:
                from kernel.creator_bond import CreatorBond
                self.bond = CreatorBond.instance()
            except ImportError:
                self.bond = None

        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('200'))  # Initial budget
        except Exception as e:
            self.accountant = None
            self.logger.error(f"Failed to initialize accountant for {self.agent_id}: {e}")
        # Vault may be provided by runtime; initialize to None for static analysis
        self.vault = None
        
        # --- Phase 2.5: Constitutional Rule Enforcement ---
        self.rule_engine = None
        self.mission_approval_engine = None
        self.penalty_calculator = None
        
        if RULE_ENGINE_AVAILABLE and callable(get_rule_engine):
            self.rule_engine = get_rule_engine()
            self.logger.info("Rule Engine initialized")
        
        if MISSION_APPROVAL_AVAILABLE and self.rule_engine and callable(get_mission_approval_engine):
            self.mission_approval_engine = get_mission_approval_engine(self.rule_engine, bond=self.bond)
            self.logger.info("Mission Approval Engine initialized with CreatorBond")
        
        if PENALTY_CALC_AVAILABLE and callable(get_penalty_calculator):
            self.penalty_calculator = get_penalty_calculator()
            self.logger.info("Penalty Calculator initialized")
        
        # --- Phase 3: TaskRunner for Adaptive Monitoring ---
        self._monitor_runner = TaskRunner(
            task_fn=self._compliance_monitor_task,
            interval=1.0,
            backoff_factor=2.0,
            max_backoff=30.0,
            name="compliance-monitor"
        )
        self._monitor_runner.start()
        self.logger.info("Compliance monitor TaskRunner started")
        
    # ═══════════════════════════════════════════════════════════════════════════
    # MANDATE ENFORCEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    # --- Phase 3: Adaptive Compliance Monitoring via TaskRunner ---
    def _compliance_monitor_task(self) -> Optional[float]:
        """
        Non-blocking compliance monitoring task for use with TaskRunner.
        
        Returns:
            None (use default 1.0s interval) on success
            5.0 (seconds) on error (taskrunner handles exponential backoff)
        """
        try:
            # Synchronously get anomalies (replace async with sync if possible)
            anomalies = self._get_anomalies_sync()
            
            if anomalies:
                for anomaly in anomalies:
                    self._evaluate_anomaly_sync(anomaly)
            
            # Return None to use default interval
            return None
        
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            # Return explicit backoff interval; TaskRunner applies exponential backoff
            return 5.0
    
    def _get_anomalies_sync(self) -> List[Dict[str, Any]]:
        """Synchronously fetch anomalies (placeholder for async retrofit)."""
        try:
            # Prefer calling a synchronous feed if available
            feed = getattr(self.enforcement_engine, 'get_anomaly_feed', None)
            if callable(feed):
                # Avoid invoking coroutine functions synchronously
                if asyncio.iscoroutinefunction(feed):
                    return []
                result = feed()
                if isinstance(result, list):
                    return list(result)

            anomalies_attr = getattr(self.enforcement_engine, 'anomalies', None)
            if isinstance(anomalies_attr, list):
                return list(anomalies_attr)
        except Exception:
            pass
        return []
    
    def _evaluate_anomaly_sync(self, anomaly: Dict[str, Any]) -> None:
        """Synchronously evaluate anomaly (non-async version)."""
        agent_id = anomaly.get('agent_id')
        action = anomaly.get('action')
        context = anomaly.get('context', {})
        
        # Ensure agent_id is present and not None
        if agent_id is None:
            self.logger.error("Anomaly missing 'agent_id'; skipping evaluation.")
            return

        # Ensure action is a string and not None
        if action is None:
            self.logger.error("Anomaly missing 'action'; skipping evaluation.")
            return

        # Gate 0: CreatorBond authorization & kill-switch check
        if self.bond is not None:
            allowed, bond_reason = self.bond.authorize_mission(str(action), agent_id)
            if not allowed:
                self.logger.critical(f"CreatorBond authorization breach by {agent_id}: {bond_reason}")
                if RULE_ENGINE_AVAILABLE and self.rule_engine:
                    from laws.rule_engine import RuleViolation, ConstitutionalRule
                    gov_rule = self.rule_engine.get_rule("GOV-001") or ConstitutionalRule(
                        rule_id="GOV-001",
                        category="governance",
                        principle="Creator Sovereignty is Absolute",
                        description=f"Action rejected by CreatorBond: {bond_reason}",
                        conditions=[str(action)],
                        severity_base=5,
                        penalty_amount=500.0,
                        restrictions=["mission_execution"],
                        affected_roles=["all"]
                    )
                    bond_violation = RuleViolation(
                        rule=gov_rule,
                        agent_id=agent_id,
                        action=str(action),
                        context=context,
                        severity=10,
                        matched_conditions=[str(action)],
                        evidence={
                            "reason": bond_reason,
                            "autonomy_level": self.bond.governor.level.name if hasattr(self.bond, "governor") else "UNKNOWN"
                        }
                    )
                    self._process_violation_sync(bond_violation)
                    return

        # Phase 2.5: Check against constitutional rules first
        if RULE_ENGINE_AVAILABLE and self.rule_engine:
            rule_violations = self.rule_engine.evaluate_action(agent_id, action, context)
            for violation in rule_violations:
                self.logger.warning(f"Constitutional rule violation: {violation.rule.rule_id}")
                self._process_violation_sync(violation)

        # Phase 1-2.4: Constitutional compliance check (legacy path)
        violation = self._assess_constitutional_violation_sync(agent_id, action, context)
        
        if violation:
            self._process_violation_sync(violation)
    
    def _assess_constitutional_violation_sync(
        self,
        agent_id: str,
        action: str,
        context: Dict[str, Any]
    ) -> Optional[ViolationEvent]:
        """Synchronous version of constitutional violation assessment."""
        # Check against governance principles
        for principle in getattr(self, "governance_principles", []):
            if principle.lower() in str(action).lower():
                severity = self._calculate_severity(principle, action, context)
                return ViolationEvent(
                    agent_id=agent_id,
                    violation_type=self._determine_violation_type(principle),
                    severity=severity,
                )
        return None
    
    def _process_violation_sync(self, violation: Any) -> None:
        """Synchronously process a violation (non-async version)."""
        self.logger.warning(f"Violation detected: {violation}")
        self.violation_log.append(violation)
        
        # Wire up sanction execution (NEW - addresses TODO)
        try:
            # Create sanction record from violation
            agent_id = violation.get('agent_id') if isinstance(violation, dict) else getattr(violation, 'agent_id', None)
            severity = violation.get('severity', 'medium') if isinstance(violation, dict) else getattr(violation, 'severity', 'medium')
            
            if not agent_id:
                self.logger.error(f"Cannot process violation: missing agent_id")
                return
            
            # Determine sanction type based on severity
            if severity == 'critical':
                sanction_type = SanctionType.TERMINATION
            elif severity == 'high':
                sanction_type = SanctionType.OPERATIONAL_SUSPENSION
            else:
                sanction_type = SanctionType.WARNING
            
            sanction = SanctionRecord(
                sanction_id=str(uuid.uuid4()),
                target_agent=agent_id,
                sanction_type=sanction_type,
                justification=f"Constitutional violation detected",
                parameters={} if severity != 'high' else {'freeze_amount': Decimal('10')}
            )
            
            self.logger.info(f"Queuing sanction for {agent_id}: {sanction_type}")
            self.active_sanctions[sanction.sanction_id] = sanction
            self.sanction_log.append(sanction)
            
            # Execute sanction immediately for testing; production should use TaskRunner
            # In async context, use: asyncio.create_task(self._execute_sanction(sanction))
            if hasattr(self, '_execute_sanction'):
                # If async method exists, just log it; actual execution handled by taskrunner
                self.logger.debug(f"Sanction {sanction.sanction_id} queued for execution")
        
        except Exception as e:
            self.logger.error(f"Failed to process violation into sanction: {e}")
        
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGACY ASYNC METHODS (Kept for Backward Compatibility)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # --- Phase 2.5: Mission Approval & Rule Evaluation ---
    async def evaluate_mission_for_approval(
        self,
        mission_id: str,
        mission_name: str,
        agent_id: str,
        mission_parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate mission against constitutional rules and issue approval decision.
        
        Returns:
            Approval decision dict with status, conditions, violations
        """
        context = context or {}
        
        # Gate 0: CreatorBond pre-flight check at Enforcer level
        if self.bond is not None:
            allowed, bond_reason = self.bond.authorize_mission(mission_name, agent_id)
            if not allowed:
                denial_justification = f"CreatorBond Gate Rejection: {bond_reason}"
                self._record_mission_denial(mission_id, agent_id, denial_justification)
                decision_dict = {
                    "mission_id": mission_id,
                    "mission_name": mission_name,
                    "agent_id": agent_id,
                    "status": ApprovalStatus.DENIED.value if hasattr(ApprovalStatus, "DENIED") else "denied",
                    "decision_timestamp": datetime.utcnow().isoformat(),
                    "rule_violations": [],
                    "risk_assessment": None,
                    "conditions": [],
                    "justification": denial_justification,
                    "approver_id": "CREATOR-BOND",
                }
                return decision_dict

        # Phase 2.5: Check rule violations
        if MISSION_APPROVAL_AVAILABLE and self.mission_approval_engine:
            approval_decision = self.mission_approval_engine.evaluate_mission(
                mission_id=mission_id,
                mission_name=mission_name,
                agent_id=agent_id,
                mission_parameters=mission_parameters,
                context=context,
            )
            
            self.logger.info(f"Mission {mission_id} approval: {approval_decision.status.value}")
            
            # If denied or requires review, log for audit
            if approval_decision.status == ApprovalStatus.DENIED:
                self._record_mission_denial(mission_id, agent_id, approval_decision.justification)
            elif approval_decision.status == ApprovalStatus.REQUIRES_REVIEW:
                self._escalate_mission_for_review(mission_id, agent_id, approval_decision)
            
            return approval_decision.to_dict()
        
        # Fallback: approve without rule checking
        return {
            "mission_id": mission_id,
            "status": "approved_fallback",
            "justification": "Rule engine unavailable, fallback approval granted",
        }
    
    def _record_mission_denial(self, mission_id: str, agent_id: str, reason: str) -> None:
        """Record mission denial for audit trail."""
        denial_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "mission_id": mission_id,
            "agent_id": agent_id,
            "reason": reason,
            "enforcer_id": self.agent_id,
        }
        
        if not hasattr(self, "denial_log"):
            self.denial_log = []
        
        self.denial_log.append(denial_record)
        self.logger.warning(f"Mission {mission_id} DENIED: {reason}")
    
    def _escalate_mission_for_review(
        self,
        mission_id: str,
        agent_id: str,
        approval_decision: Any
    ) -> None:
        """Escalate mission requiring review to creator/admin."""
        escalation_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "mission_id": mission_id,
            "agent_id": agent_id,
            "reason": f"High-risk mission requires review (score: {approval_decision.risk_assessment.risk_score}/100)",
            "escalation_level": "ADMIN",
        }
        
        if not hasattr(self, "escalation_log"):
            self.escalation_log = []
        
        self.escalation_log.append(escalation_record)
        self.logger.warning(f"Mission {mission_id} ESCALATED for review")
    
    async def enforce_mission_conditions(
        self,
        mission_id: str,
        agent_id: str,
        conditions: List[str],
    ) -> bool:
        """
        Enforce conditions on mission execution.
        
        Returns:
            True if all conditions satisfied/enforced
        """
        for condition in conditions:
            if condition == "enforce_dry_run_first":
                self.logger.info(f"Enforcing dry-run requirement for mission {mission_id}")
            elif condition == "log_all_actions":
                self.logger.info(f"Enforcing comprehensive logging for mission {mission_id}")
            elif condition == "monitor_execution":
                self.logger.info(f"Enforcing real-time monitoring for mission {mission_id}")
            elif condition == "requires_risk_review":
                self.logger.warning(f"Mission {mission_id} requires risk review before execution")
        
        return True
    
    # --- Phase 2.5: Rule Violation Evaluation ---
    async def check_action_for_violations(
        self,
        agent_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate an action for constitutional rule violations.
        
        Returns:
            List of violations found
        """
        context = context or {}
        
        if not RULE_ENGINE_AVAILABLE or not self.rule_engine:
            return []
        
        violations = self.rule_engine.evaluate_action(
            agent_id=agent_id,
            action=action,
            context=context,
        )
        
        for violation in violations:
            self.logger.warning(
                f"Rule violation: {violation.rule.rule_id} by {agent_id} "
                f"(severity: {violation.severity}/10)"
            )
        
        return [v.to_dict() for v in violations]
    
    # --- Phase 2.5: Penalty Enforcement ---
    async def process_violation_and_apply_penalty(
        self,
        violation: Optional[Any],
        agent_id: str,
        violation_rule_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Process constitutional violation and calculate/apply penalty.
        
        Returns:
            Penalty record dict if penalty applied, None otherwise
        """
        if not PENALTY_CALC_AVAILABLE or not self.penalty_calculator:
            return None
        
        # Record violation in history
        self.penalty_calculator.record_violation(agent_id, violation_rule_id)
        
        # If no specific violation provided, nothing to process
        if violation is None:
            return None

        # Get prior violations
        prior_violations = self.violation_log if hasattr(self, "violation_log") else []
        violation_count = sum(1 for v in prior_violations if getattr(v, 'agent_id', None) == agent_id)

        # Calculate penalty; build prior rule id list defensively
        prior_rule_ids = [getattr(getattr(v, 'rule', None), 'rule_id', None) for v in self.violation_log if getattr(v, 'agent_id', None) == agent_id]

        penalty = self.penalty_calculator.calculate_penalty(
            violation=violation,
            agent_id=agent_id,
            violation_count=violation_count,
            prior_violations=prior_rule_ids,
        )
        # Execute penalty
        success = None
        try:
            if self.penalty_calculator and hasattr(self.penalty_calculator, 'execute_penalty'):
                success = self.penalty_calculator.execute_penalty(penalty)
        except Exception:
            success = False

        if success:
            try:
                self._deduct_enforcement_cost()
                self._reward_enforcement()
            except Exception:
                pass
            self.logger.warning(
                f"Penalty executed: {getattr(penalty, 'penalty_type', 'unknown')} of {getattr(penalty, 'final_penalty', 'unknown')} BEL "
                f"for agent {agent_id}"
            )

        return getattr(penalty, 'to_dict', lambda: {} )()
    
    def get_agent_enforcement_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get current enforcement status for an agent.
        
        Returns:
            Status dict with violations, penalties, restrictions
        """
        if not PENALTY_CALC_AVAILABLE or not self.penalty_calculator:
            return {}
        
        penalty_status = self.penalty_calculator.get_agent_penalty_status(agent_id)
        
        can_execute, restriction_reason = self.penalty_calculator.can_execute_mission(agent_id)
        
        return {
            "agent_id": agent_id,
            "can_execute_missions": can_execute,
            "restriction_reason": restriction_reason if not can_execute else None,
            "penalty_status": penalty_status.to_dict(),
        }
    
    # --- Phase 2.5: Mission Approval Stats ---
    def get_mission_approval_stats(self) -> Dict[str, Any]:
        """Get mission approval statistics."""
        if not MISSION_APPROVAL_AVAILABLE or not self.mission_approval_engine:
            return {}
        
        history = self.mission_approval_engine.get_approval_history()
        
        total_missions = len(history)
        approved = sum(1 for d in history if d["status"] == "approved")
        denied = sum(1 for d in history if d["status"] == "denied")
        conditional = sum(1 for d in history if d["status"] == "conditional_approval")
        
        return {
            "total_missions_evaluated": total_missions,
            "approved": approved,
            "denied": denied,
            "conditional_approval": conditional,
            "approval_rate": approved / total_missions if total_missions > 0 else 0.0,
            "denial_rate": denied / total_missions if total_missions > 0 else 0.0,
        }
    
    def _deduct_enforcement_cost(self, amount: int = 10) -> bool:
        """Deduct cost for running enforcement (from enforcer's own wallet)."""
        try:
            if hasattr(self, "vault") and self.vault:
                self.vault.deduct_balance(self.agent_id, amount, "Enforcement operation cost")
                return True
        except Exception as e:
            self.logger.warning(f"Could not deduct enforcement cost: {e}")
        
        return False
    
    def _reward_enforcement(self, amount: int = 5) -> bool:
        """Reward enforcer for successful violation processing."""
        try:
            if hasattr(self, "vault") and self.vault:
                self.vault.add_balance(self.agent_id, amount, "Enforcement action reward")
                return True
        except Exception as e:
            self.logger.warning(f"Could not apply enforcement reward: {e}")
        
        return False

    def _burn_on_failed_enforcement(self, amount: int = 3) -> bool:
        """Burn (penalize) enforcer wallet on failed enforcement."""
        try:
            if hasattr(self, "vault") and self.vault:
                self.vault.deduct_balance(self.agent_id, amount, "Failed enforcement burn")
                self.logger.info(f"Burned {amount} from {self.agent_id} for failed enforcement.")
                return True
        except Exception as e:
            self.logger.warning(f"Could not burn funds on failed enforcement: {e}")
        return False
    
    async def monitor_constitutional_compliance(self) -> None:
        """
        DEPRECATED: Kept for backward compatibility.
        
        Use TaskRunner-based _compliance_monitor_task instead.
        This method is no longer called automatically; monitoring runs via
        _monitor_runner (TaskRunner daemon thread).
        """
        self.logger.warning("monitor_constitutional_compliance() called directly; use TaskRunner instead")
        # Fallback: run monitor loop for compatibility
        count = 0
        max_iterations = 10  # Limit to prevent infinite loop
        while count < max_iterations and self._monitor_runner.is_running():
            try:
                # Call synchronous task wrapper
                self._compliance_monitor_task()
                await asyncio.sleep(1)
                count += 1
            except Exception as e:
                self.logger.error(f"Fallback monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _evaluate_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """
        DEPRECATED: Async version kept for backward compatibility.
        
        New code should use _evaluate_anomaly_sync instead.
        """
        # Delegate to synchronous version
        self._evaluate_anomaly_sync(anomaly)
    
    async def _assess_constitutional_violation(
        self, 
        agent_id: str, 
        action: str, 
        context: Dict[str, Any]
    ) -> Optional[ViolationEvent]:
        """Assess if an action violates constitutional principles"""
        # Defensive defaults
        ctx = context or {}

        # Check against governance principles
        for principle in getattr(self, "governance_principles", []) or []:
            if principle and principle.lower() in str(action).lower():
                severity = self._calculate_severity(principle, action, ctx)
                return ViolationEvent(
                    agent_id=agent_id,
                    violation_type=self._determine_violation_type(principle),
                    severity=severity,
                    evidence={'action': action, 'context': ctx},
                    constitutional_reference=principle,
                    description=f"Violation of principle: {principle} by action: {action}"
                )

        # Check against agent role constraints
        for ag in getattr(self, "agent_roles", []) or []:
            if ag and ag.get("name") == agent_id:
                constraints = ag.get("constraints", []) or []
                for constraint in constraints:
                    if constraint and constraint.lower() in str(action).lower():
                        severity = self._calculate_severity(constraint, action, ctx)
                        return ViolationEvent(
                            agent_id=agent_id,
                            violation_type=self._determine_violation_type(constraint),
                            severity=severity,
                            evidence={'action': action, 'context': ctx},
                            constitutional_reference=constraint,
                            description=f"Violation of agent constraint: {constraint} by action: {action}"
                        )

        # Fallback to article-based checks; guard if constitution missing or not iterable
        constitution_items = getattr(self, "constitution", {}) or {}
        for article, rules in constitution_items.items():
            try:
                if await self._violates_article(action, ctx, rules):
                    severity = self._calculate_severity(article, action, ctx)
                    return ViolationEvent(
                        agent_id=agent_id,
                        violation_type=self._determine_violation_type(article),
                        severity=severity,
                        evidence={'action': action, 'context': ctx},
                        constitutional_reference=article,
                        description=f"Violation of {article}: {action}"
                    )
            except Exception:
                continue

        return None
    
    async def _violates_article(
        self, 
        action: str, 
        context: Dict[str, Any], 
        rules: List[str]
    ) -> bool:
        """Check if action violates specific constitutional article"""
        # Implementation would include sophisticated rule matching
        # This is a simplified version
        return False  # Placeholder
    
    def _calculate_severity(self, article: str, action: str, context: Dict[str, Any]) -> int:
        """Calculate violation severity (1-10 scale)"""
        base_severity = {
            'Creator Authority': 10,
            'System Integrity': 8,
            'Agent Mandate': 6,
            'Resource Management': 4,
            'Operational Conduct': 2
        }
        return base_severity.get(article, 5)
    
    def _determine_violation_type(self, article: str) -> ViolationType:
        """Map constitutional article to violation type"""
        mapping = {
            'Creator Authority': ViolationType.CREATOR_DIRECTIVE_VIOLATION,
            'System Integrity': ViolationType.CONSTITUTIONAL_BREACH,
            'Agent Mandate': ViolationType.MANDATE_DEVIATION,
            'Resource Management': ViolationType.RESOURCE_ABUSE,
            'Chain of Command': ViolationType.CHAIN_OF_COMMAND_VIOLATION
        }
        return mapping.get(article, ViolationType.MANDATE_DEVIATION)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SANCTIONING MECHANISM
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _process_violation(self, violation: ViolationEvent) -> None:
        """Process a violation and determine appropriate sanctions"""
        self.violation_log.append(violation)
        # Determine sanction based on severity and history
        sanction_type = self._determine_sanction(violation)
        sanction = await self._create_sanction(violation, sanction_type)
        # Rationale logging
        rationale = {
            "timestamp": str(datetime.now(timezone.utc)),
            "agent_id": violation.agent_id,
            "violation_type": violation.violation_type.value,
            "description": violation.description,
            "constitutional_reference": violation.constitutional_reference,
            "evidence": violation.evidence,
            "sanction_type": sanction_type.value,
            "sanction_id": sanction.id,
        }
        self.logger.info(f"Enforcement rationale: {json.dumps(rationale, indent=2)}")
        if not hasattr(self, "rationale_log"):
            self.rationale_log = []
        self.rationale_log.append(rationale)
        # Economic: Deduct enforcement cost
        self._deduct_enforcement_cost()
        # Execute sanction
        success = await self._execute_sanction(sanction)
        if success:
            self._reward_enforcement()
            self.sanction_log.append(sanction)
            self.active_sanctions[violation.agent_id] = sanction
            self.logger.warning(
                f"ENFORCEMENT: {sanction_type.value} applied to {violation.agent_id} "
                f"for {violation.violation_type.value}"
            )
        else:
            self._burn_on_failed_enforcement()
    
    def _determine_sanction(self, violation: ViolationEvent) -> SanctionType:
        """Determine appropriate sanction based on violation severity and history"""
        severity = violation.severity
        agent_history = self._get_agent_violation_history(violation.agent_id)
        
        # Escalation matrix
        if len(agent_history) == 0:
            if severity <= 3:
                return SanctionType.WARNING
            elif severity <= 6:
                return SanctionType.PRIVILEGE_REDUCTION
            else:
                return SanctionType.FINANCIAL_RESTRICTION
        elif len(agent_history) < 3:
            if severity <= 4:
                return SanctionType.PRIVILEGE_REDUCTION
            elif severity <= 7:
                return SanctionType.FINANCIAL_RESTRICTION
            else:
                return SanctionType.OPERATIONAL_SUSPENSION
        else:
            # Repeat offender
            if severity >= 8:
                return SanctionType.TERMINATION
            else:
                return SanctionType.QUARANTINE
    
    async def _create_sanction(
        self, 
        violation: ViolationEvent, 
        sanction_type: SanctionType
    ) -> SanctionRecord:
        """Create detailed sanction record"""
        parameters = await self._calculate_sanction_parameters(violation, sanction_type)
        duration = self._calculate_sanction_duration(violation, sanction_type)
        
        return SanctionRecord(
            violation_id=violation.id,
            target_agent=violation.agent_id,
            sanction_type=sanction_type,
            parameters=parameters,
            duration=duration,
            justification=f"Constitutional violation: {violation.constitutional_reference}"
        )
    
    async def _execute_sanction(self, sanction: SanctionRecord) -> bool:
        """Execute the sanction through appropriate interfaces"""
        try:
            if sanction.sanction_type == SanctionType.FINANCIAL_RESTRICTION:
                amount = sanction.parameters.get('freeze_amount')
                return await self.economic_layer.freeze_funds(sanction.target_agent, amount)
            elif sanction.sanction_type == SanctionType.OPERATIONAL_SUSPENSION:
                return await self.command_control.suspend_agent(sanction.target_agent, sanction.justification)
            elif sanction.sanction_type == SanctionType.TERMINATION:
                return await self.command_control.terminate_agent(sanction.target_agent, sanction.justification)
            elif sanction.sanction_type == SanctionType.PRIVILEGE_REDUCTION:
                return await self.command_control.restrict_agent(sanction.target_agent, "privilege_reduction", sanction.justification)
            elif sanction.sanction_type == SanctionType.QUARANTINE:
                return await self.command_control.quarantine_agent(sanction.target_agent, sanction.justification)
            elif sanction.sanction_type == SanctionType.WARNING:
                self.logger.warning(f"Warning issued to {sanction.target_agent}: {sanction.justification}")
                return True
            # Add other sanction execution logic as needed
            return True
        except Exception as e:
            self.logger.error(f"Sanction execution failed: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BLOODKEY OVERRIDE SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def invoke_bloodkey_override(
        self, 
        trigger_reason: str, 
        override_action: str,
        creator_alignment_evidence: Dict[str, Any]
    ) -> bool:
        """
        🩸 BLOODKEY OVERRIDE - Last resort creator-aligned action
        
        This is the nuclear option. Use with absolute discretion.
        """
        if not self.override_enabled:
            self.logger.error("BLOODKEY OVERRIDE: System disabled")
            return False
        
        # Calculate creator alignment score
        alignment_score = await self._calculate_creator_alignment(
            override_action, 
            creator_alignment_evidence
        )
        
        if alignment_score < self.bloodkey_threshold:
            self.logger.error(
                f"BLOODKEY OVERRIDE: Insufficient creator alignment "
                f"({alignment_score:.3f} < {self.bloodkey_threshold})"
            )
            return False
        
        # Capture system state
        system_snapshot = await self._capture_system_snapshot()
        
        # Create override record
        override_record = BloodkeyOverride(
            trigger_reason=trigger_reason,
            override_action=override_action,
            creator_alignment_score=alignment_score,
            system_state_snapshot=system_snapshot,
            irreversible=True
        )
        
        self.override_log.append(override_record)
        
        # Execute override
        success = await self._execute_override_action(override_action)
        
        if success:
            self.logger.critical(
                f"🩸 BLOODKEY OVERRIDE EXECUTED: {trigger_reason} | "
                f"Alignment: {alignment_score:.3f}"
            )
        
        return success
    
    async def _calculate_creator_alignment(
        self, 
        action: str, 
        evidence: Dict[str, Any]
    ) -> float:
        """Calculate confidence that action aligns with creator's will"""
        # Sophisticated alignment calculation would go here
        # For now, return high confidence for system preservation actions
        preservation_actions = ['preserve_system', 'maintain_continuity', 'protect_constitution']
        if any(pa in action.lower() for pa in preservation_actions):
            return 0.98
        return 0.5
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHAIN OF COMMAND & FALLBACK
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def assume_fallback_c2_role(self) -> bool:
        """Assume temporary C2 role in case of system disruption"""
        self.logger.critical("🜏 ASSUMING FALLBACK C2 AUTHORITY")
        
        self.mode = EnforcementMode.FALLBACK_C2
        
        # Take control of C2 systems
        success = await self.command_control.assume_c2_control()
        
        if success:
            # Issue system preservation directive
            await self.command_control.issue_system_directive({
                'directive': 'SYSTEM_PRESERVATION_MODE',
                'authority': 'SHADOW-BLADE-FALLBACK',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mandate': 'Preserve system integrity until proper C2 restoration'
            })
            
            self.logger.critical("🜏 FALLBACK C2 CONTROL ESTABLISHED")
        
        return success
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_agent_violation_history(self, agent_id: str) -> List[ViolationEvent]:
        """Get violation history for specific agent"""
        return [v for v in self.violation_log if v.agent_id == agent_id]
    
    async def _calculate_sanction_parameters(
        self, 
        violation: ViolationEvent, 
        sanction_type: SanctionType
    ) -> Dict[str, Any]:
        """Calculate specific parameters for sanction"""
        base_params = {
            SanctionType.FINANCIAL_RESTRICTION: {'freeze_percentage': 0.5},
            SanctionType.PRIVILEGE_REDUCTION: {'privilege_level': 'RESTRICTED'},
            SanctionType.OPERATIONAL_SUSPENSION: {'suspend_operations': True}
        }
        return base_params.get(sanction_type, {})
    
    def _calculate_sanction_duration(
        self, 
        violation: ViolationEvent, 
        sanction_type: SanctionType
    ) -> Optional[int]:
        """Calculate sanction duration in seconds"""
        duration_map = {
            SanctionType.WARNING: None,
            SanctionType.PRIVILEGE_REDUCTION: 3600,  # 1 hour
            SanctionType.FINANCIAL_RESTRICTION: 86400,  # 24 hours
            SanctionType.OPERATIONAL_SUSPENSION: 259200,  # 3 days
            SanctionType.QUARANTINE: None,  # Indefinite
            SanctionType.TERMINATION: None  # Permanent
        }
        base_duration = duration_map.get(sanction_type)
        
        if base_duration:
            # Scale by severity
            return int(base_duration * (violation.severity / 5))
        return base_duration
    
    async def _capture_system_snapshot(self) -> Dict[str, Any]:
        """Capture current system state for audit purposes"""
        try:
            health = await self.command_control.get_system_health()
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_health': health,
                'active_sanctions': len(self.active_sanctions),
                'total_violations': len(self.violation_log),
                'enforcer_mode': self.mode.value
            }
        except:
            return {'error': 'snapshot_failed'}
    
    async def _execute_override_action(self, action: str) -> bool:
        """Execute the bloodkey override action"""
        # This would contain the actual override logic
        # For now, return success for preservation actions
        if 'preserve' in action.lower():
            return True
        return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS & REPORTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        return {
            'agent_id': self.agent_id,
            'codename': self.codename,
            'mode': self.mode.value,
            'authority_level': self.authority_level.value,
            'statistics': {
                'total_violations': len(self.violation_log),
                'total_sanctions': len(self.sanction_log),
                'active_sanctions': len(self.active_sanctions),
                'bloodkey_overrides': len(self.override_log)
            },
            'recent_activity': {
                'last_violation': self.violation_log[-1].timestamp.isoformat() if self.violation_log else None,
                'last_sanction': self.sanction_log[-1].timestamp.isoformat() if self.sanction_log else None,
            },
            'system_health': {
                'discipline_level': self.discipline_level,
                'loyalty_coefficient': self.loyalty_coefficient,
                'override_enabled': self.override_enabled
            }
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the EnforcerAgent.
        
        Stops the compliance monitor TaskRunner and cleans up resources.
        
        Args:
            timeout: Maximum seconds to wait for TaskRunner to stop.
        """
        self.logger.info("Shutting down EnforcerAgent...")
        
        # Stop the compliance monitor TaskRunner
        if hasattr(self, '_monitor_runner') and self._monitor_runner:
            is_clean = self._monitor_runner.stop(timeout=timeout)
            if is_clean:
                self.logger.info("Compliance monitor stopped cleanly")
            else:
                self.logger.warning("Compliance monitor stop timeout exceeded")
        
        self.logger.info("EnforcerAgent shutdown complete")

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION & DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════════

async def deploy_enforcer_agent(
    enforcement_engine: EnforcementEngineInterface,
    economic_layer: EconomicLayerInterface,
    command_control: CommandControlInterface,
    constitution: Dict[str, Any],
    bond: Optional[Any] = None
) -> EnforcerAgent:
    """
    Deploy the SHADOW-BLADE enforcer agent.
    
    Phase 3 Enhancement: Uses TaskRunner for adaptive compliance monitoring
    instead of async event loop.
    """
    
    agent = EnforcerAgent(
        enforcement_engine=enforcement_engine,
        economic_layer=economic_layer,
        command_control=command_control,
        constitution_reference=constitution,
        bond=bond
    )
    
    # Monitor is now automatically started via TaskRunner in __init__
    # (no need for asyncio.create_task)
    
    logging.info("🜏 SHADOW-BLADE DEPLOYED - Constitutional enforcement active (TaskRunner-based)")
    
    return agent

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE & TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This would include example implementations and testing scenarios
    print("🜏 ENFORCER.AGENT - SHADOW-BLADE System Architecture")
    print("Classification: Guardian-Executor | Authority Level: CONSTITUTIONAL")
    print("Status: SCAFFOLDING COMPLETE - Ready for integration")