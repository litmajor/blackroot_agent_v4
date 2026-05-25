"""
UNIFIED REASONING V2: INTEGRATION GUIDE
========================================

How to wire the intelligent unified reasoner into your operations

Author: Blackroot Agent v4
Date: April 6, 2026
"""

# =============================================================================
# QUICK START
# =============================================================================

"""
One-line summary:
  Replace all causal decision-making with UnifiedCausalReasonerV2.decide()

It handles:
  ✓ Earned confidence (not default 50%)
  ✓ Decision diversity (not greedy collapse)
  ✓ Strategy blending (not hard overrides)
  ✓ Uncertainty-driven behavior (not ignored)

All in one call.
"""


# =============================================================================
# INTEGRATION TEMPLATE: Modifying Any Operation
# =============================================================================

"""
BEFORE (using old causal_simulator):
----
def select_escalation_technique(available_techniques):
    # Query causal graph
    scores = causal_simulator.simulate(available_techniques)
    
    # Pick best
    technique = max(scores.items(), key=lambda x: x[1])[0]
    
    # Always pick same technique (no fatigue!)
    # Confidence always ~50% (no learning!)
    # Uncertainty ignored
    # Strategy is hardcoded if/then
    
    return technique


AFTER (using UnifiedCausalReasonerV2):
----
def select_escalation_technique(available_techniques):
    # Create reasoner (once per operation)
    reasoner = UnifiedCausalReasonerV2()
    
    # Make intelligent decision
    decision = reasoner.decide(
        phase=OperationalPhase.PRIVILEGE_ESCALATION,
        options=available_techniques
    )
    
    # Check confidence before acting
    if decision.is_confident():
        # Commit fully
        execute_technique(decision.choice, aggressive=True)
    elif decision.is_uncertain():
        # Would-be-risky, skip high-risk techniques
        return None  # Gather more intel instead
    else:
        # Medium confidence, proceed carefully
        execute_technique(decision.choice, aggressive=False)
    
    # Record outcome for learning
    reasoner.execute_decision(
        decision,
        {
            "success": success,
            "risk_delta": {"detection_probability": +0.15},
            "privilege_gained": "admin" if success else None,
        }
    )
    
    return decision.choice
"""


# =============================================================================
# INTEGRATION PATTERNS BY MODULE
# =============================================================================

"""
PATTERN 1: PERSISTENCE HARDENING
----------------------------------

Current (old):
  mechanisms = ["registry", "wmi_event", "startup"]
  best = pick_by_heuristics(mechanisms)
  enable(best)

With v2:
  reasoner = UnifiedCausalReasonerV2()
  decision = reasoner.decide(
      phase=OperationalPhase.PERSISTENCE,
      options=mechanisms
  )
  
  if decision.is_confident():
      enable(decision.choice)  # High confidence → commit
      reasoner.execute_decision(decision, {"success": True, ...})
  else:
      logger.warning(f"Low confidence ({decision.confidence:.0%}), " +
                     "skipping persistence")


PATTERN 2: PHONE HOME / C2 CALLBACK
-------------------------------------

Current (old):
  heartbeat_interval = 60  # Fixed
  
  if threat_level > threshold:
      heartbeat_interval = 120  # Reactive (too slow)

With v2:
  reasoner = UnifiedCausalReasonerV2()
  reasoner.world.risk.detection_probability = threat_level / 100
  
  decision = reasoner.decide(
      phase=OperationalPhase.MAINTAIN_PRESENCE,
      options=["c2_60s", "c2_300s", "c2_dns_fallback", "silence"]
  )
  
  if decision.choice == "silence":
      stop_c2()  # Uncertainty is too high
  elif decision.choice == "c2_dns_fallback":
      use_dns_channel()  # Stealthy but slower
  else:
      heartbeat_interval = parse_interval(decision.choice)
  
  # Important: Record outcome
  reasoner.execute_decision(decision, {"success": c2_successful, ...})


PATTERN 3: PRIVILEGE ESCALATION
---------------------------------

Current (old):
  techniques = ["uac_bypass", "kernel_exploit", "token_impersonation"]
  
  if threat_level > 0.75:
      # Greedy: always choose same technique that worked before
      return "kernel_exploit"
  else:
      return pick_fastest(techniques)

With v2:
  reasoner = UnifiedCausalReasonerV2()
  decision = reasoner.decide(
      phase=OperationalPhase.PRIVILEGE_ESCALATION,
      options=techniques
  )
  
  # v2 automatically:
  # ✓ Prevents greedy collapse (tries different techniques)
  # ✓ High risk → avoids kernel_exploit (uncertainty-driven)
  # ✓ Applies earned confidence (proven techniques get bonus)
  # ✓ Blends strategy (takes current operation strategy into account)
  
  if decision.is_confident():
      return decision.choice
  else:
      # Low confidence, try gentler approach first
      return "token_impersonation"  # Safer fallback


PATTERN 4: LATERAL MOVEMENT
-----------------------------

Current (old):
  targets = query_network_for_targets()
  
  # Always pick same targeting strategy
  priority = "high_value"  # Fixed
  
  for target in sort_by(targets, priority):
      compromise(target)

With v2:
  reasoner = UnifiedCausalReasonerV2()
  
  # Strategy selection and target scoring happen TOGETHER
  decision = reasoner.decide(
      phase=OperationalPhase.LATERAL_MOVEMENT,
      options=[
          "target_all",
          "target_high_value",
          "target_stealthy",
          "target_minimum",
      ]
  )
  
  strategy = reasoner.strategy_engine.choose(reasoner.world)
  
  if strategy == Strategy.AGGRESSIVE_SPEED:
      prioritize = decision.choice  # Could be "target_all"
  elif strategy == Strategy.STEALTH_LONG_GAME:
      prioritize = "target_stealthy"  # Overrides to careful approach
  elif strategy == Strategy.LATERAL_THEN_EXFIL:
      prioritize = "target_high_value"  # Focused targets
  
  for target in sort_by(targets, prioritize):
      success = compromise(target)
      reasoner.execute_decision(decision, {
          "success": success,
          "machines_compromised": 1 if success else 0,
          "risk_delta": {"detection_probability": +0.10}
      })
  
  # Check: should we stop?
  if reasoner.world.should_abort():
      break


PATTERN 5: EXFILTRATION
------------------------

Current (old):
  channels = ["https", "dns", "icmp", "smtp"]
  
  # Always use fastest (greedy!)
  channel = "https"
  exfil_data(channel)

With v2:
  reasoner = UnifiedCausalReasonerV2()
  
  decision = reasoner.decide(
      phase=OperationalPhase.DATA_EXFILTRATION,
      options=channels
  )
  
  # v2 considers:
  # - Risk (high risk → avoids https, uses dns)
  # - Uncertainty (don't know endpoint detection → avoids risky channels)
  # - Strategy (SURVIVAL → minimum bandwidth, AGGRESSIVE → full speed)
  # - Technique history (proven channels get bonus)
  # - Fatigue (doesn't always use same channel)
  
  # If uncertainty is too high
  if decision.uncertainty_level > 0.7:
      use_channel = "dns"  # Safest option
      data_rate = minimal
  else:
      use_channel = decision.choice
      data_rate = max_speed
  
  exfil_data(use_channel, rate=data_rate)
  
  reasoner.execute_decision(decision, {
      "success": exfil_ok,
      "risk_delta": {"forensic_risk": +0.20} if exfil_ok else {}
  })
"""


# =============================================================================
# CONFIDENCE CHECKLIST: When to Use Decisions
# =============================================================================

"""
BEFORE COMMITTING TO A DECISION:

Check the confidence breakdown:

if decision.confidence > 0.7 and decision.variance < 0.15:
    # HIGH CONFIDENCE
    print("✅ Commit fully to decision")
    execute_aggressive(decision.choice)

elif decision.confidence > 0.5 and decision.variance < 0.25:
    # MEDIUM CONFIDENCE
    print("⚠️  Proceed carefully to decision")
    execute_cautious(decision.choice)

elif decision.uncertainty_level > 0.7:
    # HIGH UNCERTAINTY (should be exploring, not acting)
    print("❌ Too uncertain, gather intel instead")
    return None  # Abort this operation

elif decision.fatigue_adjusted and len(recent_decisions) > 1:
    # FATIGUE-ADJUSTED (exploration forcing)
    print("↻ Fatigue is encouraging exploration")
    execute(decision.choice)  # It's intentional exploration

else:
    print("⚠️  Uncertain action, use as fallback")
    return None


CONFIDENCE INTERPRETATION:

Confidence   Variance   Meaning
----------   --------   -------
> 0.7        < 0.15     HIGH - Commit fully
0.5-0.7      0.15-0.25  MEDIUM - Proceed cautiously
< 0.5        > 0.25     LOW - Abort or gather intel
"""


# =============================================================================
# WORLD STATE INTERACTION
# =============================================================================

"""
The world state is shared across all decisions:

reasoner = UnifiedCausalReasonerV2()

# Phase 1: Infiltration
decision1 = reasoner.decide(
    phase=OperationalPhase.INFILTRATION,
    options=["phishing", "exploit", "supply_chain"]
)
reasoner.execute_decision(decision1, {"success": True, "risk_delta": {...}})

# World state is now updated:
# - Risk accumulated
# - Techniques used recorded
# - Resources consumed
# - Strategy may have shifted

# Phase 2: Escalation (uses updated world state!)
decision2 = reasoner.decide(
    phase=OperationalPhase.PRIVILEGE_ESCALATION,
    options=["uac_bypass", "kernel_exploit", "token_impersonation"]
)

# Because world state was updated, decision2 considers:
# - Higher risk (from decision1)
# - Proven techniques (from decision1 history)
# - Current strategy (may have shifted based on risk)
# - Current uncertainty (changed from recon intel)

This is emergent intelligence - no explicit rules connecting phases.
"""


# =============================================================================
# TESTING YOUR INTEGRATION
# =============================================================================

"""
TEMPLATE TEST:

def test_escalation_with_v2():
    reasoner = UnifiedCausalReasonerV2()
    
    # Set initial conditions
    reasoner.world.resources.privilege_level = "user"
    reasoner.world.risk.detection_probability = 0.2
    
    # Make decision
    decision = reasoner.decide(
        phase=OperationalPhase.PRIVILEGE_ESCALATION,
        options=["uac_bypass", "kernel_exploit", "token_impersonation"]
    )
    
    # Validate
    assert decision.choice in ["uac_bypass", "kernel_exploit", "token_impersonation"]
    assert 0 <= decision.confidence <= 1
    assert decision.is_confident() or decision.is_uncertain() or True
    
    # Execute
    success = execute_escalation(decision.choice)
    
    # Record
    reasoner.execute_decision(decision, {
        "success": success,
        "privilege_gained": "admin" if success else None,
        "risk_delta": {"detection_probability": +0.10}
    })
    
    # Validate state updated
    assert reasoner.world.risk.detection_probability > 0.2
    if success:
        assert reasoner.world.resources.privilege_level == "admin"
    
    print("✅ Integration test passed")
"""


# =============================================================================
# MIGRATION CHECKLIST
# =============================================================================

"""
To migrate from old causal system to v2:

For each operation module (persistence, escalation, lateral, etc.):

□ 1. Import UnifiedCausalReasonerV2
    from unified_causal_reasoner_v2 import UnifiedCausalReasonerV2

□ 2. Replace decision logic with reasoner.decide()
    OLD: technique = max(scores.items(), key=lambda x: x[1])[0]
    NEW: decision = reasoner.decide(phase=..., options=techniques)
         technique = decision.choice

□ 3. Add confidence checking
    if decision.is_uncertain():
        return None  # Don't execute low-confidence decisions

□ 4. Record outcomes
    reasoner.execute_decision(decision, {
        "success": success,
        "risk_delta": {...},
        "machines_compromised": count if success else 0,
    })

□ 5. Test integration
    - Does confidence vary? (not stuck at 50%)
    - Do decisions vary? (not greedy)
    - Does high uncertainty change behavior? (no aggressive acts)
    - Does strategy affect decisions? (not override)

□ 6. Monitor execution
    - Log decision.uncertainty_mode (should shift: CONFIDENT → CAUTIOUS → EXPLORING)
    - Log decision.confidence (should improve with history)
    - Log decision.fatigue_adjusted (should be True sometimes)
    - Log decision.strategy_alignment (should vary 0.3-0.9)
"""


# =============================================================================
# ANTIPATTERNS: What NOT to Do
# =============================================================================

"""
❌ WRONG: Ignoring low confidence

  decision = reasoner.decide(phase=..., options=...)
  execute_anyway(decision.choice)  # ← DON'T DO THIS
  
  RIGHT:
  if decision.is_uncertain():
      gather_more_intel()  # Or abort
  else:
      execute(decision.choice)


❌ WRONG: Resetting reasoner per decision

  for option in options:
      reasoner = UnifiedCausalReasonerV2()  # ← DON'T DO THIS
      decision = reasoner.decide(...)
  
  RIGHT:
  reasoner = UnifiedCausalReasonerV2()  # Once per operation
  for option in options:
      decision = reasoner.decide(...)


❌ WRONG: Ignoring world state changes

  decision = reasoner.decide(...)
  execute(decision.choice)
  # World state updated inside execute_decision()
  # But you forgot to call reasoner.execute_decision()!
  
  RIGHT:
  decision = reasoner.decide(...)
  success = execute(decision.choice)
  reasoner.execute_decision(decision, {"success": success, ...})


❌ WRONG: Using old confidence formula

  confidence = uncertainty * 0.5
  
  RIGHT:
  decision.confidence  # Already earned via confidence_engine


❌ WRONG: Making same decision every time

  decision = reasoner.decide(...)
  for i in range(10):
      execute_same(decision.choice)  # ← Greedy, no learning
  
  RIGHT:
  for i in range(10):
      decision = reasoner.decide(...)  # Fresh decision each time
      execute(decision.choice)
      reasoner.execute_decision(decision, {...})
"""


# =============================================================================
# EXAMPLE: COMPLETE INFILTRATION -> ESCALATION FLOW
# =============================================================================

"""
def run_operation():
    # Create reasoner once
    reasoner = UnifiedCausalReasonerV2()
    
    # PHASE 1: INFILTRATION
    infiltration_options = ["phishing", "exploit", "supply_chain"]
    infiltration_decision = reasoner.decide(
        phase=OperationalPhase.INFILTRATION,
        options=infiltration_options
    )
    
    print(f"Decision: {infiltration_decision.choice}")
    print(f"Confidence: {infiltration_decision.confidence:.0%}")
    print(f"Mode: {infiltration_decision.uncertainty_mode}")
    
    if infiltration_decision.is_uncertain():
        print("Avoiding infiltration, uncertainty too high")
        return False
    
    success = execute_infiltration(infiltration_decision.choice)
    reasoner.execute_decision(infiltration_decision, {
        "success": success,
        "risk_delta": {"detection_probability": +0.15}
    })
    
    # PHASE 2: PRIVILEGE ESCALATION
    escalation_options = ["uac_bypass", "kernel_exploit", "token_impersonation"]
    escalation_decision = reasoner.decide(
        phase=OperationalPhase.PRIVILEGE_ESCALATION,
        options=escalation_options
    )
    
    # Note: escalation_decision uses UPDATED world state!
    # Higher risk from Phase 1, updated technique history, etc.
    
    print(f"\\nDecision: {escalation_decision.choice}")
    print(f"Confidence: {escalation_decision.confidence:.0%}")
    print(f"Strategy: {reasoner.strategy_engine.choose(reasoner.world)}")
    
    if escalation_decision.is_uncertain():
        print("Avoiding escalation, uncertainty too high")
        return False
    
    success = execute_escalation(escalation_decision.choice)
    reasoner.execute_decision(escalation_decision, {
        "success": success,
        "privilege_gained": "admin" if success else None,
        "risk_delta": {"detection_probability": +0.10}
    })
    
    # PHASE 3: LATERAL MOVEMENT
    lateral_options = ["psexec", "wmi_lateral", "pass_hash"]
    lateral_decision = reasoner.decide(
        phase=OperationalPhase.LATERAL_MOVEMENT,
        options=lateral_options
    )
    
    # Check abort condition
    should_abort, reason = reasoner.world.should_abort()
    if should_abort:
        print(f"\\nABORTING: {reason}")
        return False
    
    success = execute_lateral(lateral_decision.choice)
    reasoner.execute_decision(lateral_decision, {
        "success": success,
        "machines_compromised": 5 if success else 0,
        "risk_delta": {"detection_probability": +0.20}
    })
    
    return True
"""

if __name__ == "__main__":
    print(__doc__)
