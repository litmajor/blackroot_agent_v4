"""
Test: Unified Causal Reasoner Integration
==========================================

Demonstrates the "one brain" that actually unifies reasoning:
- Global world state
- Strategy-constrained decisions
- Cross-domain effects
- Uncertainty modeling
"""

import logging
from world_state import WorldState, OperationalPhase
from strategy_engine import StrategyEngine
from unified_causal_reasoner import UnifiedCausalReasoner

logging.basicConfig(level=logging.WARNING)


def test_unified_reasoning():
    """Test unified reasoning with state evolution"""
    
    print("\n" + "="*70)
    print("UNIFIED CAUSAL REASONING TEST")
    print("="*70)
    
    reasoner = UnifiedCausalReasoner()
    
    # Phase 1: Reconnaissance
    print("\n[PHASE 1] RECONNAISSANCE")
    print(f"Current state: {reasoner.world}")
    
    dec1 = reasoner.decide(
        phase=OperationalPhase.INITIAL_RECONNAISSANCE,
        options=["osint_recon", "passive_recon", "active_recon"]
    )
    print(f"\nDecision: {dec1.choice}")
    print(f"Confidence: {dec1.confidence:.0%}")
    print(f"Strategy: {reasoner.world.strategy.value}")
    print(f"Reasoning: {dec1.reasoning}")
    
    # This decision affects world state
    reasoner.execute_decision(
        dec1,
        {
            "success": True,
            "environment": {"edr_detected": False, "os": {"windows10": 5}},
            "risk_delta": {"detection_probability": +0.10}
        }
    )
    print(f"After execution - Risk: {reasoner.world.risk.total_exposure():.0%}")
    
    # Phase 2: Infiltration
    print("\n[PHASE 2] INFILTRATION")
    print(f"Current state: {reasoner.world}")
    
    dec2 = reasoner.decide(
        phase=OperationalPhase.INFILTRATION,
        options=["phishing", "exploit_unpatched", "supply_chain"]
    )
    print(f"\nDecision: {dec2.choice}")
    print(f"Confidence: {dec2.confidence:.0%}")
    print(f"Strategy change? {reasoner.world.strategy.value}")
    
    reasoner.execute_decision(
        dec2,
        {
            "success": True,
            "risk_delta": {"detection_probability": +0.15, "edr_alert_risk": +0.05},
            "machines_compromised": 1
        }
    )
    print(f"After execution - Risk: {reasoner.world.risk.total_exposure():.0%}, Machines: {reasoner.world.resources.machine_count}")
    
    # Phase 3: Privilege Escalation
    print("\n[PHASE 3] PRIVILEGE ESCALATION")
    print(f"Current state: {reasoner.world}")
    
    dec3 = reasoner.decide(
        phase=OperationalPhase.PRIVILEGE_ESCALATION,
        options=["uac_bypass", "kernel_exploit", "scheduled_task"]
    )
    print(f"\nDecision: {dec3.choice}")
    print(f"Confidence: {dec3.confidence:.0%}")
    print(f"Strategy: {reasoner.world.strategy.value}")
    
    reasoner.execute_decision(
        dec3,
        {
            "success": True,
            "risk_delta": {"edr_alert_risk": +0.10},
            "privilege_gained": "admin"
        }
    )
    print(f"After execution - Privilege: {reasoner.world.resources.privilege_level}")
    
    # Final snapshot
    print("\n[FINAL STATE]")
    print(f"Risk exposure: {reasoner.world.risk.total_exposure():.0%}")
    print(f"  - Detection probability: {reasoner.world.risk.detection_probability:.0%}")
    print(f"  - EDR alert risk: {reasoner.world.risk.edr_alert_risk:.0%}")
    print(f"Privilege: {reasoner.world.resources.privilege_level}")
    print(f"Machines: {reasoner.world.resources.machine_count}")
    print(f"Time remaining: {reasoner.world.resources.time_remaining_hours:.1f}h")
    print(f"Decisions made: {len(reasoner.world.history.decisions)}")
    print(f"Strategy: {reasoner.world.strategy.value}")
    
    abort, reason = reasoner.world.should_abort()
    print(f"Should abort: {abort} ({reason if abort else 'continuing'})")
    
    print("\n[PASS] Unified reasoning test complete")
    return True


def test_strategy_evolution():
    """Test how strategy changes as conditions change"""
    
    print("\n" + "="*70)
    print("STRATEGY EVOLUTION TEST")
    print("="*70)
    
    reasoner = UnifiedCausalReasoner()
    strategy_engine = StrategyEngine()
    
    # Scenario: Risk rises over time
    scenarios = [
        ("Low risk, early phase", 0.15, OperationalPhase.INITIAL_RECONNAISSANCE),
        ("Moderate risk, mid phase", 0.45, OperationalPhase.LATERAL_MOVEMENT),
        ("High risk, late phase", 0.70, OperationalPhase.DATA_EXFILTRATION),
        ("Critical risk", 0.85, OperationalPhase.DATA_EXFILTRATION),
    ]
    
    for name, risk, phase in scenarios:
        reasoner.world.risk.detection_probability = risk
        reasoner.world.current_phase = phase
        
        strategy = strategy_engine.choose(reasoner.world)
        print(f"\n{name}:")
        print(f"  Risk: {risk:.0%}, Phase: {phase.value}")
        print(f"  Strategy: {strategy.value}")
        
        profile = strategy_engine.get_profile(strategy)
        print(f"  Profile: risk_tolerance={profile.risk_tolerance:.0%}, evasion={profile.evasion_strictness:.0%}")
    
    print("\n[PASS] Strategy evolution test complete")
    return True


def test_uncertainty_modeling():
    """Test confidence intervals and uncertainty"""
    
    print("\n" + "="*70)
    print("UNCERTAINTY MODELING TEST")
    print("="*70)
    
    reasoner = UnifiedCausalReasoner()
    
    # Scenario: Make decision with high environment uncertainty
    reasoner.world.environment.uncertainty = 0.8
    
    print(f"\nHigh environment uncertainty: {reasoner.world.environment.uncertainty:.0%}")
    
    dec = reasoner.decide(
        phase=OperationalPhase.INITIAL_RECONNAISSANCE,
        options=["osint_recon", "passive_recon", "active_recon"]
    )
    
    print(f"\nDecision: {dec.choice}")
    print(f"Confidence: {dec.confidence:.0%}")
    print(f"Variance: {dec.variance:.0%}")
    print(f"95% CI: [{dec.lower_bound():.0%}, {dec.upper_bound():.0%}]")
    print(f"Is confident? {dec.is_confident()}")
    print(f"Is uncertain? {dec.is_uncertain()}")
    
    # After gathering intel, uncertainty drops
    reasoner.world.update_environment_knowledge({"edr_detected": False})
    
    print(f"\nAfter reconnaissance - Uncertainty: {reasoner.world.environment.uncertainty:.0%}")
    
    dec2 = reasoner.decide(
        phase=OperationalPhase.INFILTRATION,
        options=["phishing", "exploit_unpatched", "supply_chain"]
    )
    
    print(f"\nDecision 2: {dec2.choice}")
    print(f"Confidence: {dec2.confidence:.0%}")
    print(f"Variance: {dec2.variance:.0%}")
    print(f"95% CI: [{dec2.lower_bound():.0%}, {dec2.upper_bound():.0%}]")
    print(f"Is confident? {dec2.is_confident()}")
    
    print("\n[PASS] Uncertainty modeling test complete")
    return True


if __name__ == "__main__":
    tests = [
        test_unified_reasoning,
        test_strategy_evolution,
        test_uncertainty_modeling,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*70)
    
    if passed == len(tests):
        print("\n[OK] UNIFIED CAUSAL REASONING VALIDATED")
        exit(0)
    else:
        exit(1)
