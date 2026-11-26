"""
Causal Graph Module for Strategic Decision Making
================================================

Core engine for building and querying causal relationships between:
- Interventions (actions the agent takes)
- Effects (consequences and outcomes)

Used by decision engines to:
1. Predict consequences before acting
2. Compare alternative strategies
3. Learn from actual vs. predicted outcomes

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("CausalGraph")


class EffectType(Enum):
    """Types of effects that interventions can have"""
    DETECTION_RISK = "detection_risk"
    SURVIVAL_PROBABILITY = "survival_probability"
    DATA_VALUE = "data_value"
    EXFIL_SPEED = "exfil_speed"
    PERSISTENCE_QUALITY = "persistence_quality"
    PRIVILEGE_GAIN = "privilege_gain"
    FOOTPRINT_SIZE = "footprint_size"
    LATERAL_REACH = "lateral_reach"
    EDR_VISIBILITY = "edr_visibility"
    OPERATOR_VISIBILITY = "operator_visibility"
    C2_RELIABILITY = "c2_reliability"
    DGA_RESILIENCE = "dga_resilience"
    STEALTH_COST = "stealth_cost"
    RELIABILITY = "reliability"
    REMOVAL_DIFFICULTY = "removal_difficulty"
    SUCCESS_RATE = "success_rate"
    EVASION_LEVEL = "evasion_level"


@dataclass
class CausalEdge:
    """
    Represents a causal relationship: cause → effect
    
    Example:
        extract_chrome ──[weight=0.25, confidence=0.80]──> detection_risk
        
    Interpretation:
        Extracting Chrome credentials causes detection_risk to increase by 0.25
        (with 80% confidence based on empirical observations)
    """
    cause: str  # Intervention name (e.g., "extract_chrome")
    effect: str  # Effect name (e.g., "detection_risk")
    weight: float  # Strength: 0.0-1.0 (0=no relationship, 1.0=maximum)
    direction: int  # +1 (increases effect), -1 (decreases effect)
    confidence: float  # 0.0-1.0 (how certain we are about this relationship)
    
    # Learning metadata
    observations: int = 1  # How many times we've observed this
    last_updated: str = ""  # ISO timestamp of last observation
    tags: List[str] = None  # ["high_threat", "edr_detected", "windows_only"]
    total_residual_error: float = 0.0  # Sum of absolute prediction residuals
    mean_absolute_error: float = 0.0  # Running MAE

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def learn(self, actual_impact: float, predicted_impact: Optional[float] = None, alpha: float = 0.25):
        """
        Online residual update via Exponential Moving Average (EMA).
        Adjusts weight and confidence based on error between prediction and observed reality.
        """
        from datetime import datetime
        self.observations += 1
        self.last_updated = datetime.utcnow().isoformat()

        # Actual impact absolute value represents observed strength [0.0, 1.0]
        observed_strength = min(max(abs(actual_impact), 0.0), 1.0)
        # Update direction if actual sign differs
        if actual_impact != 0:
            self.direction = 1 if actual_impact > 0 else -1

        # Prediction residual
        pred = (self.direction * self.weight * self.confidence) if predicted_impact is None else predicted_impact
        residual = abs(actual_impact - pred)
        self.total_residual_error += residual
        self.mean_absolute_error = self.total_residual_error / self.observations

        # EMA update on weight
        self.weight = (1.0 - alpha) * self.weight + alpha * observed_strength

        # Confidence update: small residual increases confidence, large residual decreases it
        accuracy = max(1.0 - residual, 0.0)
        self.confidence = min(max((1.0 - alpha) * self.confidence + alpha * accuracy, 0.1), 0.99)
    
    def __repr__(self):
        sign = "+" if self.direction > 0 else "-"
        return f"{self.cause} {sign}{self.weight:.2f}→ {self.effect} (conf: {self.confidence:.1%})"
    
    def to_dict(self):
        """Serialize to dict for storage"""
        data = asdict(self)
        data['direction'] = self.direction  # Preserve sign
        return data


class CausalGraph:
    """
    Directed acyclic graph of causal relationships
    
    Usage:
        graph = CausalGraph()
        graph.add_edge("extract_chrome", "detection_risk", weight=0.25, confidence=0.80)
        graph.add_edge("extract_lsa", "detection_risk", weight=0.85, confidence=0.95)
        
        # Query: What happens if I extract LSA secrets?
        effects = graph.query_effects("extract_lsa")
        # Returns: {"detection_risk": +0.85, "privilege_gain": +0.90, ...}
        
        # Query: What impacts have higher confidence?
        effects = graph.query_effects("extract_lsa", min_confidence=0.80)
        # Returns: high-confidence effects only
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.edges: Dict[str, List[CausalEdge]] = {}  # cause → [edges]
        self.effect_index: Dict[str, List[CausalEdge]] = {}  # effect → [edges]
        self.all_causes: Set[str] = set()
        self.all_effects: Set[str] = set()
        logger.info(f"[CausalGraph] Initialized causal graph: {name}")
    
    def add_edge(
        self,
        cause: str,
        effect: str,
        weight: float = 0.5,
        direction: int = 1,
        confidence: float = 0.5,
        tags: List[str] = None,
    ) -> CausalEdge:
        """
        Add a causal relationship to the graph
        
        Args:
            cause: Action/intervention name
            effect: Consequence name
            weight: Strength (0.0-1.0)
            direction: +1 (increases) or -1 (decreases)
            confidence: How sure we are (0.0-1.0)
            tags: Context tags (e.g., ["high_threat", "windows_only"])
        
        Returns:
            The created CausalEdge
        """
        edge = CausalEdge(
            cause=cause,
            effect=effect,
            weight=min(max(weight, 0.0), 1.0),  # Clamp to [0, 1]
            direction=1 if direction > 0 else -1,
            confidence=min(max(confidence, 0.0), 1.0),
            tags=tags or []
        )
        
        # Index by cause
        if cause not in self.edges:
            self.edges[cause] = []
        self.edges[cause].append(edge)
        
        # Index by effect
        if effect not in self.effect_index:
            self.effect_index[effect] = []
        self.effect_index[effect].append(edge)
        
        self.all_causes.add(cause)
        self.all_effects.add(effect)
        
        logger.debug(f"[CausalGraph] Added edge: {edge}")
        return edge
    
    def query_effects(
        self,
        intervention: str,
        min_confidence: float = 0.0,
        effects_only: bool = False
    ) -> Dict[str, float]:
        """
        Query: "What happens if I do X?"
        
        Args:
            intervention: The action to simulate
            min_confidence: Only include effects with confidence >= this
            effects_only: If True, return effect names only; if False, return weighted impacts
        
        Returns:
            Dict mapping effect names to weighted impact (can be negative if direction=-1)
        
        Example:
            graph.query_effects("extract_chrome")
            # Returns: {
            #   "detection_risk": +0.25,
            #   "data_value": +0.70,
            #   "exfil_speed": +0.30,
            #   ...
            # }
        """
        if intervention not in self.edges:
            return {}
        
        impacts = {}
        for edge in self.edges[intervention]:
            if edge.confidence < min_confidence:
                continue
            
            # Impact = direction * weight * confidence
            impact = edge.direction * edge.weight * edge.confidence
            
            if edge.effect in impacts:
                impacts[edge.effect] += impact  # Combine multiple edges
            else:
                impacts[edge.effect] = impact
        
        if effects_only:
            return {k: v for k, v in impacts.items()}
        
        return impacts
    
    def multi_hop_effects(
        self,
        intervention: str,
        depth: int = 2,
        min_confidence: float = 0.0
    ) -> Dict[str, float]:
        """
        Query: "What are the INDIRECT consequences of X?"
        
        Example:
            do(extract_lsa_secrets) → privilege_gain
                                  → privilege_gain → privilege_exploits
                                                  → privilege_exploits → persistence_quality
        
        This follows chains of causality up to 'depth' hops.
        
        Args:
            intervention: Starting action
            depth: How many hops to follow
            min_confidence: Minimum confidence threshold
        
        Returns:
            Dict of direct + indirect effects with reduced confidence (decays by hop distance)
        """
        all_impacts = {}
        visited = set()
        
        def _traverse(current_action: str, current_depth: int, decay_factor: float):
            if current_depth <= 0 or current_action in visited:
                return
            visited.add(current_action)
            
            # Get direct effects of current action
            direct_effects = self.query_effects(current_action, min_confidence)
            
            for effect, impact in direct_effects.items():
                # Decay confidence by hop distance
                decayed_impact = impact * decay_factor
                
                if effect in all_impacts:
                    all_impacts[effect] += decayed_impact
                else:
                    all_impacts[effect] = decayed_impact
                
                # Check if this effect is also a cause (chains effects)
                if effect in self.edges and current_depth > 1:
                    _traverse(effect, current_depth - 1, decay_factor * 0.8)
        
        _traverse(intervention, depth, 1.0)
        return all_impacts
    
    def compare_interventions(
        self,
        options: List[str],
        objective: str = "survival_probability",
        weights: Dict[str, float] = None
    ) -> List[Tuple[str, float]]:
        """
        Compare multiple intervention options and rank by objective
        
        Args:
            options: List of intervention names to compare
            objective: Primary objective to optimize for (e.g., "survival_probability")
            weights: Custom scoring weights for effects
                     e.g., {"survival_probability": 1.0, "detection_risk": -0.5}
        
        Returns:
            List of (intervention, score) tuples, sorted best-to-worst
        
        Example:
            graph.compare_interventions(
                ["extract_chrome", "extract_lsa", "extract_ssh"],
                objective="survival_probability",
                weights={
                    "survival_probability": 1.0,
                    "detection_risk": -0.5,
                    "data_value": 0.3
                }
            )
            # Returns: [
            #   ("extract_lsa", 0.62),
            #   ("extract_ssh", 0.58),
            #   ("extract_chrome", 0.41)
            # ]
        """
        if weights is None:
            weights = {objective: 1.0}
        
        scores = []
        for intervention in options:
            effects = self.query_effects(intervention)
            
            # Calculate weighted score
            score = 0.0
            for effect, weight in weights.items():
                if effect in effects:
                    score += effects[effect] * weight
            
            scores.append((intervention, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def learn_from_outcome(
        self,
        intervention: str,
        actual_effects: Dict[str, float],
        predicted_effects: Optional[Dict[str, float]] = None,
        alpha: float = 0.25,
        create_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Online residual learning across all observed effects of an intervention.
        Updates matching causal edges with EMA adjustments and residual tracking.
        """
        results: Dict[str, Any] = {"updated": [], "created": [], "residuals": {}}
        pred_dict = predicted_effects or self.query_effects(intervention)

        for effect_name, actual_val in actual_effects.items():
            pred_val = pred_dict.get(effect_name, 0.0)
            residual = abs(actual_val - pred_val)
            results["residuals"][effect_name] = residual

            # Find matching edge
            matched = False
            for edge in self.edges.get(intervention, []):
                if edge.effect == effect_name:
                    edge.learn(actual_impact=actual_val, predicted_impact=pred_val, alpha=alpha)
                    results["updated"].append(effect_name)
                    matched = True
                    break

            if not matched and create_missing:
                # Dynamically instantiate discovered causal relationship
                direction = 1 if actual_val >= 0 else -1
                new_edge = self.add_edge(
                    cause=intervention,
                    effect=effect_name,
                    weight=min(max(abs(actual_val), 0.0), 1.0),
                    direction=direction,
                    confidence=0.5,
                )
                new_edge.learn(actual_impact=actual_val, predicted_impact=pred_val, alpha=alpha)
                results["created"].append(effect_name)

        logger.info(
            f"[CausalGraph:{self.name}] Learned from {intervention}: "
            f"{len(results['updated'])} updated, {len(results['created'])} created"
        )
        return results

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Observability metrics for convergence, total observations, and MAE."""
        total_obs = 0
        total_error = 0.0
        edge_count = 0
        per_edge: Dict[str, Dict[str, Any]] = {}

        for cause, edges in self.edges.items():
            for edge in edges:
                edge_count += 1
                total_obs += edge.observations
                total_error += edge.total_residual_error
                per_edge[f"{cause}->{edge.effect}"] = {
                    "weight": round(edge.weight, 4),
                    "confidence": round(edge.confidence, 4),
                    "observations": edge.observations,
                    "mae": round(edge.mean_absolute_error, 4),
                }

        global_mae = total_error / max(total_obs, 1)
        return {
            "graph_name": self.name,
            "total_edges": edge_count,
            "total_observations": total_obs,
            "global_mae": round(global_mae, 4),
            "edges": per_edge,
        }

    def update_edge(
        self,
        cause: str,
        effect: str,
        new_weight: Optional[float] = None,
        new_confidence: Optional[float] = None,
        observation: bool = True
    ):
        """
        Update an existing edge (learning mechanism)
        
        Used when we observe actual outcomes that differ from predictions
        
        Args:
            cause: Cause name
            effect: Effect name
            new_weight: New weight (or leave None to keep current)
            new_confidence: New confidence (or leave None to keep current)
            observation: If True, increment observation counter
        """
        if cause not in self.edges:
            logger.warning(f"Edge {cause} → {effect} not found in graph")
            return
        
        for edge in self.edges[cause]:
            if edge.effect == effect:
                if new_weight is not None:
                    edge.weight = min(max(new_weight, 0.0), 1.0)
                if new_confidence is not None:
                    edge.confidence = min(max(new_confidence, 0.0), 1.0)
                if observation:
                    edge.observations += 1
                
                logger.info(f"[CausalGraph] Updated edge: {edge}")
                return
    
    def save(self, filepath: str):
        """Persist causal graph to JSON"""
        data = {
            "name": self.name,
            "edges": [edge.to_dict() for edges in self.edges.values() for edge in edges]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"[CausalGraph] Saved graph to {filepath}")
    
    def load(self, filepath: str):
        """Load causal graph from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.edges.clear()
        self.effect_index.clear()
        self.all_causes.clear()
        self.all_effects.clear()
        
        for edge_data in data.get("edges", []):
            self.add_edge(
                cause=edge_data["cause"],
                effect=edge_data["effect"],
                weight=edge_data["weight"],
                direction=edge_data["direction"],
                confidence=edge_data["confidence"],
                tags=edge_data.get("tags", [])
            )
        
        logger.info(f"[CausalGraph] Loaded graph from {filepath}")
    
    def __repr__(self):
        return (
            f"CausalGraph(name={self.name}, "
            f"causes={len(self.all_causes)}, "
            f"effects={len(self.all_effects)}, "
            f"edges={sum(len(e) for e in self.edges.values())})"
        )


def create_credential_extraction_graph() -> CausalGraph:
    """Factory: Pre-built causal graph for credential extraction decisions"""
    graph = CausalGraph(name="credential_extraction")
    
    # Extraction sources with their causal impacts
    extractions = {
        "extract_chrome": {
            "detection_risk": (0.25, 0.80),
            "data_value": (0.70, 0.85),
            "exfil_speed": (0.30, 0.75),
        },
        "extract_lsa": {
            "detection_risk": (0.85, 0.95),
            "privilege_gain": (0.90, 0.95),
            "persistence_quality": (0.70, 0.90),
        },
        "extract_ssh": {
            "detection_risk": (0.45, 0.85),
            "lateral_reach": (0.85, 0.90),
            "exfil_speed": (0.90, 0.90),
            "data_value": (0.80, 0.85),
        },
        "extract_git": {
            "detection_risk": (0.30, 0.80),
            "data_value": (0.60, 0.85),
            "lateral_reach": (0.50, 0.75),
        },
        "extract_email": {
            "detection_risk": (0.40, 0.75),
            "data_value": (0.55, 0.80),
            "exfil_speed": (0.70, 0.80),
        },
        "extract_cloud": {
            "detection_risk": (0.60, 0.85),
            "data_value": (0.90, 0.80),
            "privilege_gain": (0.40, 0.75),
        },
    }
    
    for extraction, effects in extractions.items():
        for effect, (weight, confidence) in effects.items():
            graph.add_edge(extraction, effect, weight=weight, confidence=confidence)
    
    return graph


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)
    
    graph = create_credential_extraction_graph()
    print(f"\n{graph}\n")
    
    # Test query
    print("Effects of extracting LSA secrets:")
    effects = graph.query_effects("extract_lsa")
    for effect, impact in effects.items():
        print(f"  {effect}: {impact:+.2f}")
    
    # Test comparison
    print("\nComparing extraction options:")
    comparison = graph.compare_interventions(
        ["extract_chrome", "extract_lsa", "extract_ssh"],
        weights={
            "survival_probability": 1.0,
            "detection_risk": -0.5,
            "data_value": 0.3,
            "privilege_gain": 0.5,
            "lateral_reach": 0.4,
        }
    )
    for intervention, score in comparison:
        print(f"  {intervention}: {score:.3f}")
