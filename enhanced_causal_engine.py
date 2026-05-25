"""
Enhanced Causal Engine with Non-Linear Dynamics and Learning
===========================================================

Advanced causal reasoning that captures:
1. Non-linear relationships (sigmoid thresholds)
2. Temporal dynamics (immediate vs delayed effects)
3. Causal memory (pattern-based learning)
4. Hidden variables (latent risk factors)
5. Proper learning loop (weight adjustment from errors)

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger("EnhancedCausalEngine")


def sigmoid(x: float, steepness: float = 1.0, midpoint: float = 0.0) -> float:
    """
    Non-linear sigmoid function for threshold-based effects
    
    Used for modeling detection risk, survival probability, etc.
    Risk doesn't scale linearly - it spikes at certain thresholds.
    
    Formula: 1 / (1 + e^(-steepness * (x - midpoint)))
    
    Args:
        x: Input value
        steepness: How sharp the threshold is (higher = sharper)
        midpoint: Where the 50% point is
    
    Returns:
        Value between 0.0 and 1.0
    
    Examples:
        # Gentle slope: risk increases gradually
        sigmoid(threat_level, steepness=0.5)
        
        # Sharp threshold: risk spikes suddenly
        sigmoid(threat_level, steepness=2.0)
    """
    try:
        exponent = -steepness * (x - midpoint)
        # Clamp exponent to prevent overflow
        if exponent > 500:
            return 0.0
        if exponent < -500:
            return 1.0
        return 1.0 / (1.0 + math.exp(exponent))
    except Exception:
        return 0.5


def tanh_nonlinearity(x: float, scale: float = 1.0) -> float:
    """
    Hyperbolic tangent for bounded non-linear effects
    
    Used for interactions between multiple factors
    """
    return math.tanh(scale * x) if isinstance(x, (int, float)) else 0.5


@dataclass
class CausalWeight:
    """Learned causal weight with confidence tracking"""
    base_weight: float  # Initial scientific estimate
    learned_weight: float  # Adjusted from observations
    confidence: float  # How sure we are (0-1)
    observations: int = 1  # How many times observed
    last_error: float = 0.0  # Last prediction error
    error_history: List[float] = field(default_factory=list)  # Track learning trajectory
    
    def update_from_error(self, error: float, learning_rate: float = 0.1):
        """
        Update weight when prediction was wrong
        
        error = actual - predicted
        weight += learning_rate * error
        """
        self.last_error = error
        self.error_history.append(error)
        
        # Keep last 100 errors
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # Adjust learned weight based on error
        weight_adjustment = learning_rate * error
        self.learned_weight += weight_adjustment
        self.learned_weight = min(max(self.learned_weight, -1.0), 1.0)  # Clamp
        
        # Increase confidence based on observation count
        self.observations += 1
        self.confidence = min(self.confidence + 0.01, 1.0)
        
        logger.debug(
            f"[CausalWeight] Updated: error={error:.3f}, "
            f"weight={self.learned_weight:.3f}, confidence={self.confidence:.2%}"
        )
    
    def get_effective_weight(self) -> float:
        """Return confidence-weighted combination of base + learned"""
        return (
            self.base_weight * (1 - self.confidence) +
            self.learned_weight * self.confidence
        )


@dataclass
class NonLinearEdge:
    """Causal edge with non-linear, temporal, and memory properties"""
    cause: str
    effect: str
    
    # Weight properties
    weight: CausalWeight
    
    # Non-linearity: f(x) = sigmoid(weight * steepness * (x - threshold))
    is_nonlinear: bool = False
    nonlinear_type: str = "sigmoid"  # sigmoid, threshold, saturation
    steepness: float = 1.0  # How sharp the threshold is
    threshold: float = 0.0  # Where the effect kicks in
    
    # Temporal properties: Does effect manifest immediately or delayed?
    immediate_fraction: float = 1.0  # How much of effect is immediate (0-1)
    delay_steps: int = 0  # How many observation cycles until full effect
    
    # Interaction: Does this edge interact with other edges?
    interaction_with: List[str] = field(default_factory=list)  # Other edge names
    interaction_strength: float = 0.0  # How strongly it interacts
    
    # Memory: Pattern-based causality
    memory_enabled: bool = False
    context_patterns: Dict[str, List[float]] = field(default_factory=dict)  # pattern → outcomes
    
    # Hidden variables: Unknown confounders
    latent_risk_factor: float = 0.0  # Unknown hidden variable (0-1)
    
    def apply_nonlinearity(self, base_effect: float) -> float:
        """Apply non-linear transformation to effect"""
        if not self.is_nonlinear:
            return base_effect
        
        if self.nonlinear_type == "sigmoid":
            # Effect increases sigmoidally with input
            return sigmoid(base_effect, self.steepness, self.threshold)
        
        elif self.nonlinear_type == "threshold":
            # Effect only manifests above threshold
            return base_effect if base_effect > self.threshold else 0.0
        
        elif self.nonlinear_type == "saturation":
            # Effect saturates at high values
            return min(base_effect, self.threshold)
        
        return base_effect
    
    def apply_temporal_dynamics(self, base_effect: float, time_step: int = 0) -> float:
        """Apply temporal dynamics (immediate vs delayed effects)"""
        if self.delay_steps == 0 or time_step == 0:
            # Full immediate effect
            return base_effect * self.immediate_fraction
        
        # Progressive reveal over time
        progress = min(time_step / self.delay_steps, 1.0)
        delayed_fraction = (1.0 - self.immediate_fraction) * progress
        
        return base_effect * (self.immediate_fraction + delayed_fraction)


class EnhancedCausalGraph:
    """Causal graph with non-linearity, memory, and learning"""
    
    def __init__(self, name: str = "enhanced_graph"):
        self.name = name
        self.edges: Dict[str, List[NonLinearEdge]] = {}  # cause → edges
        self.learned_weights: Dict[str, CausalWeight] = {}  # "cause→effect" → weight
        self.memory: Dict[str, List[float]] = {}  # context → outcomes
        logger.info(f"[EnhancedCausalGraph] Initialized: {name}")
    
    def add_nonlinear_edge(
        self,
        cause: str,
        effect: str,
        base_weight: float = 0.5,
        is_nonlinear: bool = False,
        nonlinear_type: str = "sigmoid",
        steepness: float = 1.0,
        threshold: float = 0.0,
        immediate_fraction: float = 1.0,
        delay_steps: int = 0
    ) -> NonLinearEdge:
        """Add edge with non-linear dynamics"""
        
        edge_key = f"{cause}→{effect}"
        
        weight_obj = CausalWeight(
            base_weight=base_weight,
            learned_weight=base_weight,
            confidence=0.6
        )
        
        edge = NonLinearEdge(
            cause=cause,
            effect=effect,
            weight=weight_obj,
            is_nonlinear=is_nonlinear,
            nonlinear_type=nonlinear_type,
            steepness=steepness,
            threshold=threshold,
            immediate_fraction=immediate_fraction,
            delay_steps=delay_steps
        )
        
        if cause not in self.edges:
            self.edges[cause] = []
        self.edges[cause].append(edge)
        
        self.learned_weights[edge_key] = weight_obj
        
        logger.debug(
            f"[EnhancedCausalGraph] Added edge: {cause} → {effect} "
            f"(weight={base_weight:.2f}, nonlinear={is_nonlinear})"
        )
        
        return edge
    
    def query_effects_nonlinear(
        self,
        intervention: str,
        context: Optional[Dict[str, float]] = None,
        time_step: int = 0
    ) -> Dict[str, float]:
        """
        Query effects with non-linear dynamics
        
        Args:
            intervention: Action taken
            context: Environmental context (threat level, etc.)
            time_step: Which observation cycle we're at (for temporal dynamics)
        
        Returns:
            Dict of effects with non-linear + temporal adjustments
        """
        if intervention not in self.edges:
            return {}
        
        context = context or {}
        impacts = {}
        
        for edge in self.edges[intervention]:
            # Get base weight (confidence-weighted)
            base_effect = edge.weight.get_effective_weight()
            
            # Apply non-linearity
            nonlinear_effect = edge.apply_nonlinearity(base_effect)
            
            # Apply temporal dynamics
            temporal_effect = edge.apply_temporal_dynamics(nonlinear_effect, time_step)
            
            # Apply hidden variable influence
            with_latent = temporal_effect * (1.0 + edge.latent_risk_factor)
            
            # Combine with context
            if "threat_level" in context and edge.cause.startswith("extract"):
                # Higher threat amplifies detection risk
                threat_amplification = sigmoid(
                    context["threat_level"] / 100.0,
                    steepness=2.0,  # Sharp threshold at midpoint
                    midpoint=0.5
                )
                with_latent *= threat_amplification
            
            if edge.effect in impacts:
                impacts[edge.effect] += with_latent
            else:
                impacts[edge.effect] = with_latent
        
        return impacts
    
    def record_outcome(
        self,
        intervention: str,
        actual_outcome: float,
        predicted_outcome: float,
        context: Optional[Dict[str, float]] = None
    ):
        """
        Learn from actual outcome (critical learning mechanism)
        
        Args:
            intervention: What we did
            actual_outcome: What really happened
            predicted_outcome: What we predicted
            context: Environmental context
        """
        error = actual_outcome - predicted_outcome
        
        logger.info(
            f"[EnhancedCausalGraph] Learning update: {intervention} "
            f"| Predicted: {predicted_outcome:.1%}, Actual: {actual_outcome:.1%}, "
            f"Error: {error:+.1%}"
        )
        
        if intervention not in self.edges:
            logger.warning(f"[EnhancedCausalGraph] No edges for {intervention}")
            return
        
        # Update all edges for this intervention with the error signal
        for edge in self.edges[intervention]:
            edge_key = f"{intervention}→{edge.effect}"
            
            # Weight adjustment proportional to error
            learning_rate = 0.1  # Can be made adaptive
            edge.weight.update_from_error(error * edge.weight.learned_weight, learning_rate)
            
            # Store in memory for pattern-based learning
            if edge.effect not in self.memory:
                self.memory[edge.effect] = []
            self.memory[edge.effect].append(actual_outcome)
        
        logger.debug(
            f"[EnhancedCausalGraph] Updated weights for {intervention} "
            f"based on error signal"
        )


@dataclass
class RegressionAnalysis:
    """Detailed counterfactual regression analysis"""
    actual_intervention: str
    actual_outcome: float
    predicted_outcome: float
    prediction_error: float  # actual - predicted
    prediction_error_magnitude: float  # |error|
    
    best_alternative: str
    best_alternative_score: float
    opportunity_cost: float  # best_alt - actual result
    
    regret_magnitude: float  # Total regret (prediction error + opportunity cost)
    
    counterfactuals: List[Tuple[str, float, float]] = field(default_factory=list)  # (intervention, predicted, opportunity)


class EnhancedCausalSimulator:
    """Simulator with proper regret calculation and learning"""
    
    def __init__(self, graph: EnhancedCausalGraph):
        self.graph = graph
        logger.info(f"[EnhancedCausalSimulator] Initialized with graph: {graph.name}")
    
    def regress_counterfactual(
        self,
        actual_intervention: str,
        actual_outcome: float,
        predicted_outcome: float,
        alternatives: List[Tuple[str, float]],  # (intervention, predicted_score)
    ) -> RegressionAnalysis:
        """
        Proper counterfactual/regret analysis
        
        Regret = Prediction Error + Opportunity Cost
        
        Where:
        - Prediction Error = |actual - predicted|
        - Opportunity Cost = max(alternatives) - actual_result
        """
        
        # Calculate prediction error
        prediction_error = actual_outcome - predicted_outcome
        prediction_error_magnitude = abs(prediction_error)
        
        # Find best alternative that we didn't choose
        if alternatives:
            best_alt_name, best_alt_score = max(alternatives, key=lambda x: x[1])
        else:
            best_alt_name, best_alt_score = "none", 0.0
        
        # Opportunity cost: what we left on the table
        opportunity_cost = best_alt_score - predicted_outcome  # What we COULD have predicted
        
        # TOTAL REGRET = both components
        total_regret = prediction_error_magnitude + max(0.0, opportunity_cost)
        
        # Counterfactuals with details
        counterfactuals = [
            (name, score, score - predicted_outcome)
            for name, score in alternatives
        ]
        
        analysis = RegressionAnalysis(
            actual_intervention=actual_intervention,
            actual_outcome=actual_outcome,
            predicted_outcome=predicted_outcome,
            prediction_error=prediction_error,
            prediction_error_magnitude=prediction_error_magnitude,
            best_alternative=best_alt_name,
            best_alternative_score=best_alt_score,
            opportunity_cost=opportunity_cost,
            regret_magnitude=total_regret,
            counterfactuals=counterfactuals
        )
        
        logger.info(
            f"[EnhancedCausalSimulator] Regression analysis:\n"
            f"  Actual: {actual_outcome:.1%}, Predicted: {predicted_outcome:.1%}\n"
            f"  Prediction Error: {prediction_error:+.1%} ({prediction_error_magnitude:.1%} magnitude)\n"
            f"  Best Alternative: {best_alt_name} ({best_alt_score:.1%})\n"
            f"  Opportunity Cost: {opportunity_cost:+.1%}\n"
            f"  TOTAL REGRET: {total_regret:.1%}"
        )
        
        return analysis


def create_nonlinear_extraction_graph() -> EnhancedCausalGraph:
    """Build non-linear causal graph for credential extraction"""
    graph = EnhancedCausalGraph(name="nonlinear_extraction")
    
    # Extract LSA with non-linear detection risk
    # Risk doesn't grow smoothly - it spikes when threat level gets high
    graph.add_nonlinear_edge(
        cause="extract_lsa",
        effect="detection_risk",
        base_weight=0.85,
        is_nonlinear=True,
        nonlinear_type="sigmoid",
        steepness=2.0,  # Sharp threshold
        threshold=0.5,  # At medium threat
        immediate_fraction=0.8,  # 80% immediate, 20% delayed
        delay_steps=5
    )
    
    # Extract LSA provides high privilege gain (non-linear benefit)
    graph.add_nonlinear_edge(
        cause="extract_lsa",
        effect="privilege_gain",
        base_weight=0.90,
        is_nonlinear=True,
        nonlinear_type="saturation",
        threshold=0.95  # Benefit saturates quickly
    )
    
    # Extract SSH with temporal dynamics
    # Lateral reach builds over time (delayed effect)
    graph.add_nonlinear_edge(
        cause="extract_ssh",
        effect="lateral_reach",
        base_weight=0.85,
        is_nonlinear=False,
        immediate_fraction=0.3,  # Only 30% immediate
        delay_steps=10  # Full effect after 10 steps
    )
    
    # Extract Chrome is safe initially but gets riskier
    # (detection improves over time)
    graph.add_nonlinear_edge(
        cause="extract_chrome",
        effect="detection_risk",
        base_weight=0.25,
        is_nonlinear=True,
        nonlinear_type="sigmoid",
        steepness=1.5,
        immediate_fraction=0.1,  # Small immediate risk
        delay_steps=8  # Risk increases over time as EDR learns
    )
    
    return graph


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("ENHANCED CAUSAL ENGINE TEST")
    print("="*70 + "\n")
    
    # Test 1: Non-linear effects
    print("TEST 1: Non-linear Detection Risk")
    print("-" * 70)
    graph = create_nonlinear_extraction_graph()
    
    contexts = [
        {"threat_level": 0},
        {"threat_level": 50},
        {"threat_level": 100},
    ]
    
    for ctx in contexts:
        effects = graph.query_effects_nonlinear("extract_lsa", context=ctx)
        print(f"Threat {ctx['threat_level']:3d}: detection_risk = {effects.get('detection_risk', 0.0):.1%}")
    
    # Test 2: Proper regret calculation
    print("\nTEST 2: Regret Calculation")
    print("-" * 70)
    
    sim = EnhancedCausalSimulator(graph)
    
    analysis = sim.regress_counterfactual(
        actual_intervention="extract_chrome",
        actual_outcome=0.10,  # EDR caught it
        predicted_outcome=0.459,
        alternatives=[
            ("extract_lsa", 0.371),
            ("extract_ssh", 0.306),
            ("extract_git", 0.380),
        ]
    )
    
    print(f"\nActual outcome: {analysis.actual_outcome:.1%}")
    print(f"Predicted outcome: {analysis.predicted_outcome:.1%}")
    print(f"Prediction error: {analysis.prediction_error:+.1%}")
    print(f"Best alternative: {analysis.best_alternative} ({analysis.best_alternative_score:.1%})")
    print(f"Opportunity cost: {analysis.opportunity_cost:+.1%}")
    print(f"\n✓ TOTAL REGRET: {analysis.regret_magnitude:.1%}")
    
    # Test 3: Learning loop
    print("\nTEST 3: Learning from Error")
    print("-" * 70)
    
    edge_key = "extract_chrome→detection_risk"
    old_weight = graph.learned_weights[edge_key].learned_weight
    
    # Record the outcome
    graph.record_outcome(
        "extract_chrome",
        actual_outcome=0.10,
        predicted_outcome=0.459,
        context={"threat_level": 75}
    )
    
    new_weight = graph.learned_weights[edge_key].learned_weight
    print(f"Weight before: {old_weight:.3f}")
    print(f"Weight after:  {new_weight:.3f}")
    print(f"Adjustment:    {new_weight - old_weight:+.3f}")
    print(f"Confidence:    {graph.learned_weights[edge_key].confidence:.1%}")
