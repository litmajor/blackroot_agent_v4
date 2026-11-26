"""
World State: Single Source of Truth for Operational Context
============================================================

Instead of passing variables around, maintain a continuously evolving
world state that represents:
- Risk exposure accumulated during operation
- Resource consumption (time, bandwidth, privilege)
- Uncertainty about environment
- Operational momentum
- Effectiveness of previous actions

This creates stateful intelligence where each decision:
- Reads world state
- Changes world state
- Influences future decisions

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger("WorldState")


class OperationalPhase(Enum):
    """Stages in the operational lifecycle"""
    INITIAL_RECONNAISSANCE = "initial_reconnaissance"
    INFILTRATION = "infiltration"
    INITIAL_EXECUTION = "initial_execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    CLEANUP = "cleanup"


@dataclass
class RiskProfile:
    """Accumulated risk exposure"""
    detection_probability: float = 0.0  # 0-1, how likely to be detected now
    edr_alert_risk: float = 0.0  # 0-1, EDR specifically
    forensic_risk: float = 0.0  # 0-1, forensic traces left
    escalation_risk: float = 0.0  # 0-1, risk of incident escalation
    
    def total_exposure(self) -> float:
        """Overall risk exposure"""
        # Not linear - risk compounds
        return min(1.0, (
            self.detection_probability * 0.4 +
            self.edr_alert_risk * 0.35 +
            self.forensic_risk * 0.15 +
            self.escalation_risk * 0.1
        ))
    
    def is_critical(self) -> bool:
        """Whether we've crossed critical detection threshold"""
        return self.total_exposure() > 0.75


@dataclass
class ResourceState:
    """Available resources"""
    time_remaining_hours: float = 72.0  # Hours until window closes
    bandwidth_available_mbps: float = 10.0  # Available exfil bandwidth
    privilege_level: str = "user"  # "user", "admin", "system", "domain_admin"
    machine_count: int = 1  # Machines we control
    
    def is_time_critical(self) -> bool:
        """Whether we're in final window"""
        return self.time_remaining_hours < 6
    
    def is_bandwidth_constrained(self) -> bool:
        """Whether bandwidth is limiting"""
        return self.bandwidth_available_mbps < 1.0


@dataclass
class EnvironmentModel:
    """What we know (and don't know) about target"""
    os_versions: Dict[str, int] = field(default_factory=dict)  # OS -> count
    edr_detected: bool = False
    edr_confidence: float = 0.0  # 0-1, how sure are we
    network_segmentation: float = 0.5  # 0-1, how isolated
    av_present: bool = False
    av_confidence: float = 0.0
    domain_functional_level: Optional[str] = None  # "2008", "2012", "2016", "2019", "2022"
    user_sophistication: str = "low"  # "low", "medium", "high"
    uncertainty: float = 0.5  # 0-1, overall env uncertainty
    
    def is_hardened(self) -> bool:
        """Whether environment is security-hardened"""
        hardened_score = (
            (1.0 if self.edr_detected else 0.0) * 0.35 +
            (1.0 if self.av_present else 0.0) * 0.20 +
            self.network_segmentation * 0.25 +
            (0.0 if self.user_sophistication == "low" else 0.5 if self.user_sophistication == "medium" else 1.0) * 0.20
        )
        return hardened_score > 0.6


@dataclass
class DecisionHistory:
    """Track decisions and outcomes"""
    decisions: List[Dict] = field(default_factory=list)
    outcomes: Dict[str, any] = field(default_factory=dict)
    failed_techniques: List[str] = field(default_factory=list)
    successful_techniques: List[str] = field(default_factory=list)
    
    def record_decision(self, phase: str, decision: str, confidence: float, reasoning: str):
        """Log a decision"""
        self.decisions.append({
            "phase": phase,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_failure(self, technique: str, reason: str):
        """Log failed technique"""
        self.failed_techniques.append(technique)
        logger.warning(f"[WorldState] Technique failed: {technique} ({reason})")
    
    def record_success(self, technique: str, quiet: bool = False):
        """Log successful technique"""
        self.successful_techniques.append(technique)
        if not quiet:
            logger.info(f"[WorldState] Technique succeeded: {technique}")
    
    def is_technique_viable(self, technique: str) -> bool:
        """Whether technique has been shown to work"""
        if technique in self.failed_techniques:
            return False
        return technique not in self.failed_techniques


class WorldState:
    """
    Unified operational context - single source of truth
    
    Every decision reads from this and potentially modifies it.
    This creates continuity and feedback loops.
    
    Usage:
        world = WorldState()
        
        # Simulate a decision's effects
        world.record_decision(
            phase=OperationalPhase.INFILTRATION,
            decision="phishing",
            risk_delta={"detection_probability": +0.15}
        )
        world.update_risk_exposure()
        
        # Future decisions adapt
        if world.risk.is_critical():
            strategy = "abort"
    """
    
    def __init__(self):
        """Initialize fresh world state"""
        self.risk = RiskProfile()
        self.resources = ResourceState()
        self.environment = EnvironmentModel()
        self.history = DecisionHistory()
        
        self.current_phase = OperationalPhase.INITIAL_RECONNAISSANCE
        self.strategy = None  # Will be set by strategy engine
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        logger.info("[WorldState] Initialized fresh world state")
    
    def advance_phase(self, new_phase: OperationalPhase):
        """Move to next operational phase"""
        prev = self.current_phase
        self.current_phase = new_phase
        logger.info(f"[WorldState] Phase transition: {prev.value} -> {new_phase.value}")
    
    def consume_time(self, hours: float):
        """Time passed during operation"""
        self.resources.time_remaining_hours = max(0, self.resources.time_remaining_hours - hours)
        
        # Time pressure increases risk as we approach deadline
        time_factor = max(0, (72 - self.resources.time_remaining_hours) / 72)
        self.risk.detection_probability += time_factor * 0.05
        
        logger.debug(f"[WorldState] Consumed {hours}h, {self.resources.time_remaining_hours}h remaining")
    
    def update_risk_from_decision(self, decision_name: str, risk_deltas: Dict[str, float]):
        """
        A decision impacted risk exposure
        
        Args:
            decision_name: What decision was made
            risk_deltas: {"detection_probability": +0.15, "edr_alert_risk": +0.10}
        """
        for risk_attr, delta in risk_deltas.items():
            current = getattr(self.risk, risk_attr, 0.0)
            new_val = max(0.0, min(1.0, current + delta))
            setattr(self.risk, risk_attr, new_val)
        
        self.last_updated = datetime.now()
        
        if self.risk.is_critical():
            logger.warning(f"[WorldState] CRITICAL RISK after {decision_name}: {self.risk.total_exposure():.1%}")
    
    def update_environment_knowledge(self, findings: Dict[str, any]):
        """
        Reconnaissance reveals environment details
        
        Args:
            findings: {"edr_detected": True, "os": {"windows10": 5, "windows11": 3}}
        """
        if "edr_detected" in findings:
            self.environment.edr_detected = findings["edr_detected"]
            self.environment.edr_confidence = findings.get("edr_confidence", 0.8)
        
        if "av_present" in findings:
            self.environment.av_present = findings["av_present"]
            self.environment.av_confidence = findings.get("av_confidence", 0.7)
        
        if "os" in findings:
            self.environment.os_versions = findings["os"]
        
        if "network_segmentation" in findings:
            self.environment.network_segmentation = findings["network_segmentation"]
        
        # Reduce uncertainty as we learn
        self.environment.uncertainty = max(0.1, self.environment.uncertainty - 0.2)
        
        logger.info(f"[WorldState] Environment updated: EDR={self.environment.edr_detected}, Uncertainty={self.environment.uncertainty:.1%}")
    
    def elevate_privilege(self, new_level: str):
        """Privilege escalation succeeded"""
        levels = ["user", "admin", "system", "domain_admin"]
        if new_level in levels and levels.index(new_level) > levels.index(self.resources.privilege_level):
            self.resources.privilege_level = new_level
            logger.info(f"[WorldState] Privilege elevated to: {new_level}")
            return True
        return False
    
    def compromise_machine(self):
        """Lateral movement to new machine succeeded"""
        self.resources.machine_count += 1
        logger.info(f"[WorldState] Machine count: {self.resources.machine_count}")
    
    def allocate_bandwidth(self, mbps: float):
        """Use bandwidth for exfiltration"""
        if mbps <= self.resources.bandwidth_available_mbps:
            self.resources.bandwidth_available_mbps -= mbps
            logger.debug(f"[WorldState] Allocated {mbps}Mbps, {self.resources.bandwidth_available_mbps}Mbps remaining")
            return True
        return False
    
    def decision_confidence(self, decision_name: str) -> float:
        """
        How confident are we in a decision type?
        
        Based on:
        - Environment uncertainty
        - Whether similar decisions succeeded before
        - Resource constraints
        """
        base_confidence = 1.0 - self.environment.uncertainty
        
        # Did we succeed with this before?
        if decision_name in self.history.successful_techniques:
            base_confidence += 0.2
        elif decision_name in self.history.failed_techniques:
            base_confidence -= 0.3
        
        # Resource constraints reduce confidence
        if self.resources.is_time_critical():
            base_confidence -= 0.1
        if self.resources.is_bandwidth_constrained():
            base_confidence -= 0.05
        
        return max(0.1, min(1.0, base_confidence))
    
    def should_abort(self) -> Tuple[bool, str]:
        """
        Is it time to abort operation?
        
        Returns:
            (should_abort, reason)
        """
        if self.risk.is_critical():
            return True, f"Critical risk: {self.risk.total_exposure():.1%}"
        
        if self.resources.time_remaining_hours < 1:
            return True, "Time window closing"
        
        if self.resources.machine_count == 0:
            return True, "Lost all machines"
        
        # Progressive escalation risk after certain phase
        if self.current_phase == OperationalPhase.DATA_EXFILTRATION:
            if self.risk.escalation_risk > 0.8:
                return True, "Escalation imminent"
        
        return False, ""
    
    def snapshot(self) -> Dict[str, any]:
        """Current state snapshot for decision making"""
        abort, reason = self.should_abort()
        
        return {
            "phase": self.current_phase.value,
            "risk_exposure": self.risk.total_exposure(),
            "risk_detail": {
                "detection": self.risk.detection_probability,
                "edr_alert": self.risk.edr_alert_risk,
                "forensic": self.risk.forensic_risk,
                "escalation": self.risk.escalation_risk,
            },
            "resources": {
                "time_hours": self.resources.time_remaining_hours,
                "bandwidth_mbps": self.resources.bandwidth_available_mbps,
                "privilege": self.resources.privilege_level,
                "machines": self.resources.machine_count,
            },
            "environment": {
                "edr_detected": self.environment.edr_detected,
                "edr_confidence": self.environment.edr_confidence,
                "av_present": self.environment.av_present,
                "uncertainty": self.environment.uncertainty,
                "hardened": self.environment.is_hardened(),
            },
            "strategy": self.strategy,
            "decisions_made": len(self.history.decisions),
            "should_abort": abort,
            "abort_reason": reason,
        }
    
    def __repr__(self) -> str:
        return (
            f"WorldState(\n"
            f"  phase={self.current_phase.value}\n"
            f"  risk={self.risk.total_exposure():.1%}\n"
            f"  time={self.resources.time_remaining_hours:.1f}h\n"
            f"  privilege={self.resources.privilege_level}\n"
            f"  machines={self.resources.machine_count}\n"
            f"  edr={self.environment.edr_detected}\n"
            f")"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo
    world = WorldState()
    print(world)
    
    # Simulate decisions
    print("\n--- Simulating reconnaissance ---")
    world.update_environment_knowledge({
        "edr_detected": True,
        "edr_confidence": 0.85,
        "os": {"windows10": 3, "windows11": 2},
    })
    
    print("\n--- Simulating infiltration risk ---")
    world.update_risk_from_decision(
        "phishing_email",
        {"detection_probability": +0.20, "edr_alert_risk": +0.15}
    )
    
    print("\n--- Time passing ---")
    world.consume_time(24)  # 24 hours passed
    
    print("\n--- Escalation success ---")
    world.elevate_privilege("admin")
    
    print("\n--- Current state ---")
    print(world)
    
    print("\n--- Decision snapshot ---")
    import json
    print(json.dumps(world.snapshot(), indent=2))
