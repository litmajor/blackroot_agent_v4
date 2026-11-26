"""
tests/test_causal_learning.py
=============================
Tests for online residual learning on CausalGraph:
  - CausalEdge.learn() convergence and MAE tracking
  - CausalGraph.learn_from_outcome() edge updating & dynamic edge creation
  - CausalGraph.get_learning_metrics()
  - LearningAgent.attach_causal_graph & record_causal_outcome
  - Backward compatibility with query_effects
"""

import pytest
from causal_graph import CausalGraph, CausalEdge
from agents.learning import LearningAgent


def test_causal_edge_learn():
    edge = CausalEdge(cause="test_action", effect="detection_risk", weight=0.5, direction=1, confidence=0.5)
    assert edge.observations == 1
    assert edge.mean_absolute_error == 0.0

    # Observe actual impact = 0.8 repeatedly
    for _ in range(5):
        edge.learn(actual_impact=0.8, alpha=0.3)

    assert edge.observations == 6
    # Weight should have moved toward 0.8
    assert edge.weight > 0.5
    assert edge.confidence > 0.5
    assert edge.mean_absolute_error > 0.0


def test_causal_graph_learn_from_outcome():
    graph = CausalGraph("test_domain")
    graph.add_edge("extract_lsa", "detection_risk", weight=0.4, direction=1, confidence=0.6)
    graph.add_edge("extract_lsa", "data_value", weight=0.7, direction=1, confidence=0.8)

    # Prior query
    prior_effects = graph.query_effects("extract_lsa")
    assert "detection_risk" in prior_effects

    # Learn from actual operation: detection_risk was actually higher (0.85), lateral_reach discovered (0.9)
    res = graph.learn_from_outcome(
        intervention="extract_lsa",
        actual_effects={
            "detection_risk": 0.85,
            "lateral_reach": 0.90,  # Unseen effect
        },
        alpha=0.4,
        create_missing=True,
    )

    assert "detection_risk" in res["updated"]
    assert "lateral_reach" in res["created"]

    # Verify updated graph reflects new reality
    new_effects = graph.query_effects("extract_lsa")
    assert new_effects["detection_risk"] > prior_effects["detection_risk"]
    assert "lateral_reach" in new_effects

    metrics = graph.get_learning_metrics()
    assert metrics["graph_name"] == "test_domain"
    assert metrics["total_edges"] == 3
    assert metrics["total_observations"] >= 4
    assert metrics["global_mae"] >= 0.0


def test_learning_agent_causal_hooks():
    graph = CausalGraph("agent_test_graph")
    graph.add_edge("harvest_tokens", "data_value", weight=0.3, direction=1, confidence=0.5)

    agent = LearningAgent(cfg={"interval": 9999})
    agent.attach_causal_graph(graph)

    # Feed simulated observation
    res = agent.record_causal_outcome(
        intervention="harvest_tokens",
        actual_effects={"data_value": 0.95},
        alpha=0.5
    )

    assert len(res) >= 1
    # Weight itself converged towards 0.95
    edge = [e for e in graph.edges["harvest_tokens"] if e.effect == "data_value"][0]
    assert edge.weight > 0.3
    assert edge.observations == 2
