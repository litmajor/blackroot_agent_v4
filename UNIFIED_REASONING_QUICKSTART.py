"""
UNIFIED CAUSAL REASONING: QUICK START GUIDE
============================================

How to use the new unified reasoning system in your code.

Author: Blackroot Agent v4
Date: April 6, 2026
"""

# =============================================================================
# BASIC USAGE
# =============================================================================

from unified_causal_reasoner import UnifiedCausalReasoner
from world_state import OperationalPhase

# Create the unified brain (one brain, one world)
reasoner = UnifiedCausalReasoner()

# Make a decision for a phase
decision = reasoner.decide(
    phase=OperationalPhase.INFILTRATION,
    options=["phishing", "exploit_unpatched", "supply_chain"]
)

# decision now contains:
print(f"Choice: {decision.choice}")
print(f"Confidence: {decision.confidence:.0%}")
print(f"Variance: {decision.variance:.0%}")
print(f"95% CI: [{decision.lower_bound():.0%}, {decision.upper_bound():.0%}]")
print(f"Reasoning: {decision.reasoning}")
print(f"Is confident? {decision.is_confident()}")
print(f"Is uncertain? {decision.is_uncertain()}")

# Execute the decision and update world state
success = reasoner.execute_decision(
    decision,
    {
        "success": True,
        "risk_delta": {"detection_probability": +0.15},
        "machines_compromised": 1,
        "privilege_gained": "admin",
    }
)

if not success:
    print("Operation should abort - risk too high or conditions changed")


# =============================================================================
# WORLD STATE ACCESS
# =============================================================================

# Check current world state
world = reasoner.world

print(f"Current phase: {world.current_phase.value}")
print(f"Risk exposure: {world.risk.total_exposure():.0%}")
print(f"Privilege level: {world.resources.privilege_level}")
print(f"Machines compromised: {world.resources.machine_count}")
print(f"Time remaining: {world.resources.time_remaining_hours}h")
print(f"Current strategy: {world.strategy.value if world.strategy else 'None'}")

# Check abort condition
should_abort, reason = world.should_abort()
if should_abort:
    print(f"ABORT: {reason}")


# =============================================================================
# STRATEGY ACCESS
# =============================================================================

strategy_engine = reasoner.strategy_engine

# Get current strategy
strategy = strategy_engine.choose(world)

# Get strategy parameters
profile = strategy_engine.get_profile(strategy)
print(f"Strategy: {strategy.value}")
print(f"  Risk tolerance: {profile.risk_tolerance:.0%}")
print(f"  Time aggression: {profile.time_aggression:.0%}")
print(f"  Evasion strictness: {profile.evasion_strictness:.0%}")
print(f"  Lateral depth: {profile.lateral_depth:.0%}")
print(f"  Privilege priority: {profile.privilege_priority:.0%}")


# =============================================================================
# ADAPTIVE REPLANNING
# =============================================================================

# If conditions change dramatically, replan
def replan_if_conditions_change():
    # Detect major change (e.g., EDR detected when we thought safe)
    if world.environment.edr_detected and not world.environment.edr_confidence > 0.8:
        # Replan entire strategy
        decision = reasoner.replan()
        print(f"Replanned decision: {decision.choice}")
        return decision
    return None


# =============================================================================
# INTEGRATION PATTERN
# =============================================================================

# Example: Using unified reasoning in persistence module

def select_persistence_mechanisms(threat_level):
    """Choose which persistence mechanisms to enable based on causal reasoning"""
    
    reasoner = UnifiedCausalReasoner()
    
    # Set threat context
    reasoner.world.risk.detection_probability = threat_level / 100
    reasoner.world.current_phase = OperationalPhase.PERSISTENCE
    
    # What persistence techniques are available?
    mechanisms = ["registry", "scheduled_task", "wmi_event", "service", "startup_folder"]
    
    # Get decision
    decision = reasoner.decide(
        phase=OperationalPhase.PERSISTENCE,
        options=mechanisms
    )
    
    # Check confidence before acting
    if not decision.is_confident():
        print(f"Warning: Uncertain decision (CI={decision.interval()})")
    
    # Execute with feedback
    try:
        enable_mechanism(decision.choice)
        reasoner.execute_decision(
            decision,
            {
                "success": True,
                "risk_delta": {"forensic_risk": +0.10}
            }
        )
    except Exception as e:
        reasoner.execute_decision(
            decision,
            {
                "success": False,
                "reason": str(e)
            }
        )
        # Next decision will avoid this technique
    
    return decision.choice, reasoner.world.strategy.value


# =============================================================================
# WORLD STATE UPDATES
# =============================================================================

# Update based on reconnaissance findings
reasoner.world.update_environment_knowledge({
    "edr_detected": True,
    "edr_confidence": 0.85,
    "av_present": False,
    "os": {"windows10": 3, "windows11": 2},
    "network_segmentation": 0.7,
})

# Record time passing
reasoner.world.consume_time(hours=24)

# Record successful technique
reasoner.world.history.record_success("phishing_email")

# Record failed technique
reasoner.world.history.record_failure("kernel_exploit", "not vulnerable to CVE-2021-1732")

# Check if technique has viability
can_use = reasoner.world.history.is_technique_viable("scheduled_task_hijack")


# =============================================================================
# DECISION CONFIDENCE THRESHOLDS
# =============================================================================

# Example: Different actions based on confidence level

def execute_with_confidence_check(decision, action_func):
    """Execute action only if decision is sufficiently confident"""
    
    if decision.is_confident():
        # Commit fully
        action_func(decision.choice, aggressive=True)
        return True
    
    elif decision.is_uncertain():
        # Very cautious
        print(f"High uncertainty (CI={decision.interval()}), aborting")
        return False
    
    else:
        # Medium confidence, proceed cautiously
        action_func(decision.choice, aggressive=False)
        return True


# =============================================================================
# MONITORING STRATEGY SHIFTS
# =============================================================================

# Strategy changes as conditions evolve

def monitor_strategy_evolution():
    """Track how strategy changes during operation"""
    
    reasoner = UnifiedCausalReasoner()
    strategy_history = []
    
    phases = [
        OperationalPhase.INITIAL_RECONNAISSANCE,
        OperationalPhase.INFILTRATION,
        OperationalPhase.PRIVILEGE_ESCALATION,
        OperationalPhase.LATERAL_MOVEMENT,
        OperationalPhase.DATA_EXFILTRATION,
    ]
    
    for phase in phases:
        # Simulate decision that increases risk
        decision = reasoner.decide(
            phase=phase,
            options=["option1", "option2", "option3"]
        )
        
        reasoner.execute_decision(
            decision,
            {
                "success": True,
                "risk_delta": {"detection_probability": +0.10}
            }
        )
        
        strategy_history.append({
            "phase": phase.value,
            "risk": reasoner.world.risk.total_exposure(),
            "strategy": reasoner.world.strategy.value,
            "decision": decision.choice,
        })
    
    # Show evolution
    print("Strategy evolution during operation:")
    for entry in strategy_history:
        print(f"  {entry['phase']}: risk={entry['risk']:.0%}, strategy={entry['strategy']}")
    
    return strategy_history


# =============================================================================
# UNCERTAINTY-DRIVEN CAUTION
# =============================================================================

def make_conservative_decision_if_uncertain(phase, options):
    """Make decision but favor conservative options if uncertain"""
    
    reasoner = UnifiedCausalReasoner()
    
    decision = reasoner.decide(phase=phase, options=options)
    
    if decision.is_uncertain():
        # Uncertainty is high - pick most stealthy/safe option instead
        conservative_options = [opt for opt in options if "stealth" in opt or "passive" in opt]
        if conservative_options:
            decision = reasoner.decide(phase=phase, options=conservative_options)
    
    return decision


# =============================================================================
# CROSS-DOMAIN EFFECT EXAMPLE
# =============================================================================

"""
High-risk escalation in phase N affects phase N+1 options

Phase 1 (Infiltration):
  → Aggressive phishing campaign
  → Risk: 0% → 20%
  → EDR alert probability: 0% → 15%

Phase 2 (Privilege Escalation):
  → World state shows: risk=20%, edr_alert=15%
  → Strategy shifts from AGGRESSIVE to STEALTH
  → Decision: Choose careful UAC bypass instead of kernel exploit
  → Risk impact is modulated by strategy

Phase 3 (Lateral Movement):
  → Risk now: 32%
  → Strategy: STEALTH_LONG_GAME (risk > 25%)
  → Lateral technique: Slow, careful, stealthy
  
This is emergent intelligence - no explicit rules about "if escalation risky then lateral is careful"
It emerges from:
  - Risk accumulation
  - Strategy adaptation
  - Strategy-constrained decisions
"""


# =============================================================================
# TESTING YOUR INTEGRATION
# =============================================================================

def test_integration():
    """Quick test of integration"""
    
    reasoner = UnifiedCausalReasoner()
    
    # Test 1: Basic decision
    dec1 = reasoner.decide(
        phase=OperationalPhase.INFILTRATION,
        options=["phishing", "exploit_unpatched"]
    )
    assert dec1.choice in ["phishing", "exploit_unpatched"]
    assert 0 <= dec1.confidence <= 1
    assert 0 <= dec1.variance <= 1
    print("✓ Basic decision works")
    
    # Test 2: State update
    reasoner.execute_decision(
        dec1,
        {"success": True, "risk_delta": {"detection_probability": +0.15}}
    )
    assert reasoner.world.risk.detection_probability > 0
    print("✓ State update works")
    
    # Test 3: Strategy tracking
    assert reasoner.world.strategy is not None
    print(f"✓ Strategy selected: {reasoner.world.strategy.value}")
    
    # Test 4: Abort checking
    reasoner.world.risk.detection_probability = 0.8
    should_abort, reason = reasoner.world.should_abort()
    assert should_abort
    print(f"✓ Abort detection works: {reason}")
    
    print("\n✅ All integration tests passed")


if __name__ == "__main__":
    # Quick demo
    reasoner = UnifiedCausalReasoner()
    
    print("="*70)
    print("UNIFIED CAUSAL REASONING - QUICK START DEMO")
    print("="*70)
    
    # Demo 1: Basic usage
    print("\n[Demo 1] Basic Decision Making")
    decision = reasoner.decide(
        phase=OperationalPhase.INFILTRATION,
        options=["phishing", "exploit_unpatched", "supply_chain"]
    )
    print(f"  Choice: {decision.choice}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Strategy: {reasoner.world.strategy.value}")
    
    # Demo 2: Execute and state update
    print("\n[Demo 2] Decision Execution")
    reasoner.execute_decision(
        decision,
        {
            "success": True,
            "risk_delta": {"detection_probability": +0.20},
            "machines_compromised": 1
        }
    )
    print(f"  Risk now: {reasoner.world.risk.total_exposure():.0%}")
    print(f"  Machines: {reasoner.world.resources.machine_count}")
    
    # Demo 3: Strategy adaptation
    print("\n[Demo 3] Strategy Adaptation")
    reasoner.world.risk.detection_probability = 0.5
    reasoner.world.current_phase = OperationalPhase.LATERAL_MOVEMENT
    new_strategy = reasoner.strategy_engine.choose(reasoner.world)
    print(f"  Risk increased → Strategy: {new_strategy.value}")
    
    print("\n✅ Done!")
