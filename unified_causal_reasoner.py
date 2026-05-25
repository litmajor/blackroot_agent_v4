"""
Unified Causal Reasoner: The True Integration
==============================================

This is the "one brain" that actually unifies all reasoning.

Before:
- Multiple simulators, coordinator picks from them
- Stateless decisions
- No cross-domain effects

Now:
- Single world model that evolves
- Global strategy that constrains all decisions
- Every decision updates world state
- Causality is consistent

The loop:
  Read world state
    ↓
  Run strategy engine (pick global strategy)
    ↓
  Query causal graph (what are the options?)
    ↓
  Simulate each option on world state
    ↓
  Reweight by strategy
    ↓
  Pick option (with confidence interval)
    ↓
  Execute decision (update world state)
    ↓
  Loop to next phase

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from world_state import WorldState, OperationalPhase
from strategy_engine import StrategyEngine, Strategy
from causal_graph import CausalGraph
from causal_simulator import CausalSimulator, ThreatEnvironmentType
from operational_causal_graphs import build_all_operational_graphs

logger = logging.getLogger("UnifiedCausalReasoner")


@dataclass
class Decision:
    """A decision with uncertainty bounds"""
    choice: str
    confidence: float  # 0-1, how sure are we
    variance: float  # 0-1, how uncertain
    reasoning: str
    risk_estimate: Dict[str, float]  # Projected risk impacts
    strategy_alignment: float  # 0-1, how well does this align with strategy
    alternatives: List[Tuple[str, float]]  # Other options and scores
    
    def lower_bound(self) -> float:
        """Confidence interval lower bound"""
        return self.confidence - (1.96 * self.variance)
    
    def upper_bound(self) -> float:
        """Confidence interval upper bound"""
        return self.confidence + (1.96 * self.variance)
    
    def interval(self) -> Tuple[float, float]:
        """Return (lower, upper) confidence interval"""
        return (self.lower_bound(), self.upper_bound())
    
    def is_confident(self) -> bool:
        """Whether we have sufficient confidence (>70% + tight interval)"""
        return self.confidence > 0.7 and self.variance < 0.15
    
    def is_uncertain(self) -> bool:
        """Whether we should be cautious (wide interval or low confidence)"""
        return self.confidence < 0.5 or self.variance > 0.25


class UnifiedCausalReasoner:
    """
    Single brain for operational reasoning.
    
    Maintains world state, enforces strategy consistency, models uncertainty.
    
    Usage:
        reasoner = UnifiedCausalReasoner()
        
        # Main loop per phase
        decision = reasoner.decide(
            phase=OperationalPhase.INFILTRATION,
            options=["phishing", "exploit", "supply_chain"],
        )
        
        # Execute and get feedback
        outcome = execute_decision(decision.choice)
        reasoner.record_outcome(decision.choice, outcome, risk_impact)
        
        # Advance to next phase
        reasoner.world.advance_phase(OperationalPhase.INITIAL_EXECUTION)
    """
    
    def __init__(self):
        """Initialize unified reasoner"""
        self.world = WorldState()
        self.strategy_engine = StrategyEngine()
        
        # Load all causal graphs
        graphs_data = build_all_operational_graphs()
        self.graphs = {}
        for domain, edges in graphs_data.items():
            graph = CausalGraph(name=domain)
            for cause, effects in edges.items():
                for effect, (weight, confidence) in effects.items():
                    graph.add_edge(cause, effect, weight=weight, confidence=confidence)
            self.graphs[domain] = graph
        
        logger.info("[UnifiedCausalReasoner] Initialized with unified model")
    
    def decide(
        self,
        phase: OperationalPhase,
        options: List[str],
        additional_context: Optional[Dict] = None
    ) -> Decision:
        """
        Make a decision that respects:
        - World state
        - Global strategy
        - Causal relationships
        - Uncertainty
        
        Args:
            phase: Current operational phase
            options: Available choices
            additional_context: Extra factors (time pressure, resources, etc.)
            
        Returns:
            Decision with confidence intervals
        """
        
        # Update world state with context
        if additional_context:
            if "time_hours" in additional_context:
                self.world.consume_time(additional_context["time_hours"])
            if "environment" in additional_context:
                self.world.update_environment_knowledge(additional_context["environment"])
        
        # Step 1: Advance phase if needed
        if phase != self.world.current_phase:
            self.world.advance_phase(phase)
        
        # Step 2: Pick global strategy
        strategy = self.strategy_engine.choose(self.world)
        self.world.strategy = strategy
        
        # Step 3: Query causal graph for this phase
        graph = self._get_graph_for_phase(phase)
        
        # Step 4: Simulate each option
        simulator = CausalSimulator(graph)
        threat_env = self._threat_to_env(self.world.risk.total_exposure())
        
        simulations = {}
        for option in options:
            result = simulator.simulate_intervention(
                option,
                threat_environment=threat_env,
                current_threat_score=self.world.risk.total_exposure() * 100
            )
            simulations[option] = result
        
        # Step 5: Reweight by strategy
        base_scores = {opt: sim.overall_score for opt, sim in simulations.items()}
        reweighted_choice = self.strategy_engine.adapt_decision_by_strategy(
            options,
            base_scores,
            strategy
        )
        
        # Step 6: Build confidence interval
        confidence = self.world.decision_confidence(reweighted_choice)
        variance = self.world.environment.uncertainty * 0.5  # Uncertainty propagates
        
        # Cross-domain effects
        risk_impacts = self._project_risk_impacts(reweighted_choice, strategy)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            phase, reweighted_choice, strategy, confidence, risk_impacts
        )
        
        # Build decision
        decision = Decision(
            choice=reweighted_choice,
            confidence=confidence,
            variance=variance,
            reasoning=reasoning,
            risk_estimate=risk_impacts,
            strategy_alignment=self._alignment_score(reweighted_choice, strategy),
            alternatives=sorted(
                [(opt, simulations[opt].overall_score) for opt in options if opt != reweighted_choice],
                key=lambda x: x[1],
                reverse=True
            )
        )
        
        logger.info(
            f"[UnifiedCausalReasoner] Decision for {phase.value}: "
            f"{decision.choice} (conf={confidence:.1%}, strategy={strategy.value})"
        )
        
        return decision
    
    def execute_decision(
        self,
        decision: Decision,
        outcome: Dict[str, any]
    ) -> bool:
        """
        Record outcome of decision and update world state.
        
        Args:
            decision: The decision object
            outcome: {"success": True, "risk_delta": {...}, "resources": {...}}
            
        Returns:
            Whether we should continue or abort
        """
        
        # Record in history
        self.world.history.record_decision(
            phase=self.world.current_phase.value,
            decision=decision.choice,
            confidence=decision.confidence,
            reasoning=decision.reasoning
        )
        
        # Update world state based on outcome
        if outcome.get("success"):
            self.world.history.record_success(decision.choice)
        else:
            self.world.history.record_failure(decision.choice, outcome.get("reason", "unknown"))
        
        # Apply risk impacts
        if "risk_delta" in outcome:
            self.world.update_risk_from_decision(decision.choice, outcome["risk_delta"])
        
        # Apply resource impacts
        if "privilege_gained" in outcome:
            self.world.elevate_privilege(outcome["privilege_gained"])
        
        if "machines_compromised" in outcome:
            for _ in range(outcome["machines_compromised"]):
                self.world.compromise_machine()
        
        if "bandwidth_consumed_mbps" in outcome:
            self.world.allocate_bandwidth(outcome["bandwidth_consumed_mbps"])
        
        # Check abort condition
        should_abort, reason = self.world.should_abort()
        if should_abort:
            logger.warning(f"[UnifiedCausalReasoner] ABORT: {reason}")
        
        return not should_abort
    
    def replan(self) -> Decision:
        """
        Adaptive replanning when conditions change.
        
        Called when:
        - Decision fails unexpectedly
        - World state changes significantly
        - Risk rises unpredictably
        """
        # Get current best options for current phase
        phase_options = self._get_phase_options(self.world.current_phase)
        
        # Re-evaluate
        logger.info("[UnifiedCausalReasoner] Replanning due to changed conditions")
        return self.decide(self.world.current_phase, phase_options)
    
    # ========== Private Helpers ==========
    
    def _get_graph_for_phase(self, phase: OperationalPhase) -> CausalGraph:
        """Get appropriate causal graph for phase"""
        phase_to_graph = {
            OperationalPhase.INITIAL_RECONNAISSANCE: "initial_access_recon",
            OperationalPhase.INFILTRATION: "initial_access_infiltration",
            OperationalPhase.PRIVILEGE_ESCALATION: "privilege_escalation",
            OperationalPhase.LATERAL_MOVEMENT: "lateral_movement_techniques",
            OperationalPhase.DATA_EXFILTRATION: "exfiltration",
        }
        graph_key = phase_to_graph.get(phase, "initial_access_recon")
        return self.graphs.get(graph_key, self.graphs["initial_access_recon"])
    
    def _threat_to_env(self, threat_score: float) -> ThreatEnvironmentType:
        """Convert threat score (0-1) to environment type"""
        if threat_score > 0.75:
            return ThreatEnvironmentType.CRITICAL
        elif threat_score > 0.5:
            return ThreatEnvironmentType.HIGH
        elif threat_score > 0.25:
            return ThreatEnvironmentType.MEDIUM
        else:
            return ThreatEnvironmentType.LOW
    
    def _project_risk_impacts(self, choice: str, strategy: Strategy) -> Dict[str, float]:
        """Project risk impacts from a decision"""
        impacts = {
            "detection_probability": 0.0,
            "edr_alert_risk": 0.0,
            "forensic_risk": 0.0,
            "escalation_risk": 0.0,
        }
        
        # Heuristics based on choice characteristics
        if "active" in choice.lower() or "exploit" in choice.lower():
            impacts["detection_probability"] += 0.25
            impacts["edr_alert_risk"] += 0.20
        
        if "bulk" in choice.lower() or "https" in choice.lower():
            impacts["detection_probability"] += 0.30
            impacts["forensic_risk"] += 0.15
        
        # Strategy modulates impact
        profile = self.strategy_engine.get_profile(strategy)
        multiplier = 1.0 - (profile.evasion_strictness * 0.5)
        
        for key in impacts:
            impacts[key] *= multiplier
        
        return impacts
    
    def _alignment_score(self, choice: str, strategy: Strategy) -> float:
        """How well does choice align with strategy?"""
        profile = self.strategy_engine.get_profile(strategy)
        
        score = 0.5  # Baseline
        
        # Check alignment with strategy parameters
        if "fast" in choice.lower() and profile.time_aggression > 0.7:
            score += 0.3
        elif "stealthy" in choice.lower() and profile.evasion_strictness > 0.7:
            score += 0.3
        elif "lateral" in choice.lower() and profile.lateral_depth > 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    def _generate_reasoning(
        self,
        phase: OperationalPhase,
        choice: str,
        strategy: Strategy,
        confidence: float,
        risk_impacts: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning"""
        return (
            f"Phase {phase.value}: chose {choice} "
            f"under strategy {strategy.value} "
            f"(confidence {confidence:.0%}). "
            f"Projected risk impact: detection +{risk_impacts['detection_probability']:.0%}, "
            f"edr_alert +{risk_impacts['edr_alert_risk']:.0%}"
        )
    
    def _get_phase_options(self, phase: OperationalPhase) -> List[str]:
        """Default options for a phase"""
        options = {
            OperationalPhase.INITIAL_RECONNAISSANCE: ["osint_recon", "passive_recon", "active_recon"],
            OperationalPhase.INFILTRATION: ["phishing", "exploit_unpatched", "supply_chain"],
            OperationalPhase.PRIVILEGE_ESCALATION: [
                "uac_bypass", "kernel_exploit", "scheduled_task", "seimpersonate", "service_abuse"
            ],
            OperationalPhase.LATERAL_MOVEMENT: ["psexec", "wmiexec", "ssh_lateral", "credential_dumping"],
            OperationalPhase.DATA_EXFILTRATION: ["bulk_https", "dns_exfil", "scheduled_exfil", "cloud_upload"],
        }
        return options.get(phase, [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    reasoner = UnifiedCausalReasoner()
    
    print("\n" + "="*70)
    print("UNIFIED CAUSAL REASONER DEMONSTRATION")
    print("="*70)
    
    # Decision 1: Reconnaissance
    print("\n--- Phase 1: Reconnaissance ---")
    dec1 = reasoner.decide(
        phase=OperationalPhase.INITIAL_RECONNAISSANCE,
        options=["osint_recon", "passive_recon", "active_recon"]
    )
    print(f"Decision: {dec1.choice}")
    print(f"Confidence: {dec1.confidence:.0%} [{dec1.lower_bound():.0%}, {dec1.upper_bound():.0%}]")
    print(f"Reasoning: {dec1.reasoning}")
    
    # Execute with outcome
    print("\n--- Executing decision ---")
    reasoner.execute_decision(
        dec1,
        {
            "success": True,
            "environment": {"edr_detected": False, "os": {"windows10": 5}},
            "risk_delta": {"detection_probability": +0.05}
        }
    )
    
    # Decision 2: Infiltration
    print("\n--- Phase 2: Infiltration ---")
    print(f"Current world state: {reasoner.world}")
    
    dec2 = reasoner.decide(
        phase=OperationalPhase.INFILTRATION,
        options=["phishing", "exploit_unpatched", "supply_chain"]
    )
    print(f"Decision: {dec2.choice}")
    print(f"Confidence: {dec2.confidence:.0%}")
    print(f"Strategy: {reasoner.world.strategy.value}")
    
    # Execute
    reasoner.execute_decision(
        dec2,
        {
            "success": True,
            "risk_delta": {"detection_probability": +0.15},
            "machines_compromised": 1
        }
    )
    
    # Decision 3: Privilege escalation
    print("\n--- Phase 3: Privilege Escalation ---")
    print(f"Risk exposure: {reasoner.world.risk.total_exposure():.0%}")
    
    dec3 = reasoner.decide(
        phase=OperationalPhase.PRIVILEGE_ESCALATION,
        options=["uac_bypass", "kernel_exploit", "scheduled_task"]
    )
    print(f"Decision: {dec3.choice}")
    print(f"Confidence: {dec3.confidence:.0%}")
    
    reasoner.execute_decision(
        dec3,
        {
            "success": True,
            "risk_delta": {"edr_alert_risk": +0.10},
            "privilege_gained": "admin"
        }
    )
    
    # Decision 4: Show world state evolution
    print("\n--- World State After 3 Decisions ---")
    snapshot = reasoner.world.snapshot()
    print(f"Phase: {snapshot['phase']}")
    print(f"Risk exposure: {snapshot['risk_exposure']:.0%}")
    print(f"Privilege: {snapshot['resources']['privilege']}")
    print(f"Machines: {snapshot['resources']['machines']}")
    print(f"Strategy: {snapshot['strategy'].value if snapshot['strategy'] else 'None'}")
    print(f"Should abort: {snapshot['should_abort']}")
