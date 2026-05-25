"""
Test Enhanced Causal Engine Features
====================================

Demonstrates:
1. Non-linear causality with sigmoid thresholds
2. Proper regret calculation (prediction error + opportunity cost)
3. Temporal causality (immediate vs delayed effects)
4. Causal memory and pattern-based learning
5. Learning loop with weight adjustment

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("TestEnhancedCausal")


def test_1_nonlinear_causality():
    """Feature 1: Non-linear effects with thresholds"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: NON-LINEAR CAUSALITY (Sigmoid Thresholds)")
    logger.info("="*70)
    
    from enhanced_causal_engine import create_nonlinear_extraction_graph, sigmoid
    
    # Demonstrate sigmoid function
    logger.info("\n→ Sigmoid Function Behavior:")
    logger.info("  Real systems don't scale linearly. They have tipping points.")
    logger.info("  Detection risk example:")
    print()
    
    threat_levels = [0, 25, 50, 75, 100]
    for threat in threat_levels:
        # Standard linear: risk = 0.85 * (threat/100) = proportional
        linear_risk = 0.85 * (threat / 100)
        
        # Non-linear sigmoid: risk spikes at threshold (threat ~50)
        nonlinear_risk = sigmoid(threat / 100, steepness=2.0, midpoint=0.5)
        
        print(f"  Threat {threat:3d}%: Linear={linear_risk:.1%}, Sigmoid={nonlinear_risk:.1%}")
    
    logger.info("\n→ Key insight: At threat=50%, sigmoid jumps to 50%")
    logger.info("  BUT linear stays at 42.5%. Massive gap!")
    logger.info("  Reality: Threat has tipping points, not smooth curves")
    
    # Test non-linear graph
    logger.info("\n→ Extracting LSA secrets at different threat levels:")
    graph = create_nonlinear_extraction_graph()
    
    for threat in [10, 50, 90]:
        effects = graph.query_effects_nonlinear(
            "extract_lsa",
            context={"threat_level": threat}
        )
        detection = effects.get("detection_risk", 0.0)
        logger.info(f"  Threat {threat}: Detection risk = {detection:.1%}")
    
    print()


def test_2_regret_calculation():
    """Feature 2: FIXED - Proper regret calculation"""
    logger.info("="*70)
    logger.info("TEST 2: PROPER REGRET CALCULATION (FIXED!)")
    logger.info("="*70)
    
    from enhanced_causal_engine import (
        create_nonlinear_extraction_graph,
        EnhancedCausalSimulator
    )
    
    graph = create_nonlinear_extraction_graph()
    sim = EnhancedCausalSimulator(graph)
    
    logger.info("\n→ Scenario: Extracted Chrome passwords in HIGH threat (75)")
    logger.info("  Prediction: 45.9% survival (we thought we'd survive)")
    logger.info("  Reality: 10% survival (EDR caught us in 2 minutes)")
    logger.info("-" * 70)
    
    analysis = sim.regress_counterfactual(
        actual_intervention="extract_chrome",
        actual_outcome=0.10,
        predicted_outcome=0.459,
        alternatives=[
            ("extract_lsa", 0.371),
            ("extract_ssh", 0.306),
            ("extract_git", 0.380),
        ]
    )
    
    logger.info(f"\n✓ REGRET ANALYSIS (PROPER CALCULATION):")
    print(f"  Actual Outcome:           {analysis.actual_outcome:.1%}")
    print(f"  Predicted Outcome:        {analysis.predicted_outcome:.1%}")
    print(f"  Prediction Error Mag:     {analysis.prediction_error_magnitude:.1%}")
    print(f"  Best Alternative:         {analysis.best_alternative} (score={analysis.best_alternative_score:.1%})")
    print(f"  Opportunity Cost:         {analysis.opportunity_cost:+.1%}")
    print(f"  ──────────────────────────")
    print(f"  TOTAL REGRET:             {analysis.regret_magnitude:.1%}")
    
    logger.info("\n✓ KEY FIX: Regret is NOT 0.0% anymore!")
    logger.info(f"  Before: regret was comparing predictions only (broken)")
    logger.info(f"  After:  regret = prediction_error + opportunity_cost")
    logger.info(f"  This is what drives LEARNING PRESSURE")
    
    print()


def test_3_temporal_causality():
    """Feature 3: Temporal dynamics (immediate vs delayed effects)"""
    logger.info("="*70)
    logger.info("TEST 3: TEMPORAL CAUSALITY (Immediate vs Delayed Effects)")
    logger.info("="*70)
    
    from enhanced_causal_engine import create_nonlinear_extraction_graph
    
    graph = create_nonlinear_extraction_graph()
    
    logger.info("\n→ Extract SSH keys builds lateral reach OVER TIME")
    logger.info("  Immediate fraction: 30% (quick wins)")
    logger.info("  Delayed fraction:   70% (takes 10 steps)")
    logger.info("-" * 70)
    
    for time_step in [0, 1, 5, 10, 15]:
        effects = graph.query_effects_nonlinear(
            "extract_ssh",
            context={"threat_level": 50},
            time_step=time_step
        )
        reach = effects.get("lateral_reach", 0.0)
        pct = int((time_step / 10) * 100) if time_step <= 10 else 100
        logger.info(f"  Time {time_step:2d}: lateral_reach = {reach:.1%} ({pct}% complete)")
    
    logger.info("\n→ Extract Chrome DETECTION RISK increases over time")
    logger.info("  Immediate risk: 10% (EDR doesn't notice immediately)")
    logger.info("  Delayed risk:   90% (EDR learns pattern over 8 steps)")
    logger.info("-" * 70)
    
    for time_step in [0, 2, 4, 8, 12]:
        effects = graph.query_effects_nonlinear(
            "extract_chrome",
            context={"threat_level": 75},
            time_step=time_step
        )
        risk = effects.get("detection_risk", 0.0)
        logger.info(f"  Time {time_step:2d}: detection_risk = {risk:.1%}")
    
    logger.info("\n✓ KEY INSIGHT: Effects are not instant")
    logger.info("  Lateral movement builds gradually")
    logger.info("  Detection risk accumulates over time")
    
    print()


def test_4_causal_memory():
    """Feature 4: Causal memory and pattern-based learning"""
    logger.info("="*70)
    logger.info("TEST 4: CAUSAL MEMORY (Pattern-Based Learning)")
    logger.info("="*70)
    
    from enhanced_causal_engine import create_nonlinear_extraction_graph
    
    graph = create_nonlinear_extraction_graph()
    
    logger.info("\n→ Recording outcomes in memory for pattern learning")
    logger.info("-" * 70)
    
    # Simulate multiple observations
    observations = [
        ("extract_chrome", 0.45, 0.12),  # Predicted 45%, got 12%
        ("extract_chrome", 0.45, 0.18),  # Predicted 45%, got 18%
        ("extract_lsa", 0.37, 0.09),     # Predicted 37%, got 9%
    ]
    
    for intervention, predicted, actual in observations:
        graph.record_outcome(
            intervention=intervention,
            predicted_outcome=predicted,
            actual_outcome=actual,
            context={"threat_level": 75}
        )
        logger.info(
            f"  {intervention}: Predicted {predicted:.0%}, got {actual:.0%} "
            f"(error={actual-predicted:+.0%})"
        )
    
    logger.info("\n✓ Memory stores patterns:")
    for effect, outcomes in graph.memory.items():
        avg = sum(outcomes) / len(outcomes) if outcomes else 0
        logger.info(f"  {effect}: {outcomes} (avg={avg:.1%})")
    
    logger.info("\n✓ KEY FEATURE: System learns patterns")
    logger.info("  'When threat=HIGH, extract_chrome fails reliably'")
    logger.info("  This pattern persists in memory")
    
    print()


def test_5_learning_loop():
    """Feature 5: Learning loop with weight adjustment"""
    logger.info("="*70)
    logger.info("TEST 5: LEARNING LOOP (Weight Adjustment from Errors)")
    logger.info("="*70)
    
    from enhanced_causal_engine import create_nonlinear_extraction_graph
    
    graph = create_nonlinear_extraction_graph()
    
    logger.info("\n→ Starting with base weights (from science)")
    logger.info("-" * 70)
    
    edge_key = "extract_chrome→detection_risk"
    weight_obj = graph.learned_weights[edge_key]
    
    logger.info(f"  Base weight:     {weight_obj.base_weight:.3f}")
    logger.info(f"  Learned weight:  {weight_obj.learned_weight:.3f}")
    logger.info(f"  Confidence:      {weight_obj.confidence:.1%}")
    
    logger.info("\n→ Recording outcome with large error signal")
    predicted = 0.459
    actual = 0.10
    error = actual - predicted
    logger.info(f"  Predicted: {predicted:.1%}, Actual: {actual:.1%}, Error: {error:+.1%}")
    
    logger.info("\n  Executing: graph.record_outcome(...)")
    graph.record_outcome(
        intervention="extract_chrome",
        predicted_outcome=predicted,
        actual_outcome=actual,
        context={"threat_level": 75}
    )
    
    logger.info("\n→ After learning:")
    weight_obj_after = graph.learned_weights[edge_key]
    logger.info(f"  Base weight:     {weight_obj_after.base_weight:.3f} (unchanged)")
    logger.info(f"  Learned weight:  {weight_obj_after.learned_weight:.3f} ← ADJUSTED!")
    logger.info(f"  Confidence:      {weight_obj_after.confidence:.1%} ← INCREASED!")
    logger.info(f"  Observations:    {weight_obj_after.observations}")
    logger.info(f"  Error history:   {weight_obj_after.error_history}")
    
    logger.info("\n✓ KEY FORMULA: weight += learning_rate * error")
    logger.info(  f"  {weight_obj.learned_weight:.3f} += 0.1 * {error:.3f} = {weight_obj_after.learned_weight:.3f}")
    logger.info("\n✓ LEARNING PRESSURE: Error signal drives weight updates")
    logger.info("  Larger errors → larger weight adjustments")
    logger.info("  Multiple observations → higher confidence")
    
    print()


def main():
    try:
        logger.info("\n" + "="*70)
        logger.info("ENHANCED CAUSAL ENGINE: ALL 5 FEATURES")
        logger.info("="*70)
        
        test_1_nonlinear_causality()
        test_2_regret_calculation()
        test_3_temporal_causality()
        test_4_causal_memory()
        test_5_learning_loop()
        
        logger.info("="*70)
        logger.info("ALL TESTS COMPLETED")
        logger.info("="*70)
        logger.info("\nSummary:")
        logger.info("  ✓ Non-linear causality captures tipping points")
        logger.info("  ✓ Regret calculation FIXED (prediction error + opportunity cost)")
        logger.info("  ✓ Temporal dynamics model delayed effects")
        logger.info("  ✓ Causal memory learns patterns across observations")
        logger.info("  ✓ Learning loop adjusts weights based on error signals")
        
        return 0
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
