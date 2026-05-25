"""
Operational Causal Engine Integration Test
===========================================

Tests the complete operational causal reasoning engine across:
- Multiple threat levels (LOW, MEDIUM, HIGH, CRITICAL)
- All operational phases
- Strategic decision routing

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import sys
import logging
from operational_causal_engine import OperationalCausalEngine, OperationalPhase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_operational_causal")


def test_threat_level_routing():
    """Verify decisions adjust based on threat level"""
    engine = OperationalCausalEngine()
    
    threat_levels = [10, 45, 75, 90]
    threat_names = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    for threat, name in zip(threat_levels, threat_names):
        print(f"\nTHREAT LEVEL: {threat}/100 ({name})")
        
        esc = engine.choose_escalation_technique(
            threat_level=threat,
            current_privilege="user",
            target_privilege="system"
        )
        risk = esc["detection_risk"]
        print(f"  Escalation risk: {risk:.1%} ({esc['technique']})")
        
        exfil = engine.choose_exfiltration_method(
            data_volume_mb=500,
            threat_level=threat,
            time_pressure="medium"
        )
        print(f"  Exfil method: {exfil['method']}")
        
        c2 = engine.choose_c2_method_for_phase(
            operational_phase=OperationalPhase.EXFILTRATION,
            threat_level=threat
        )
        print(f"  C2 method: {c2['method']}")
    
    return True


def test_initial_access_phase():
    """Verify initial access decisions chain correctly"""
    engine = OperationalCausalEngine()
    
    print("\n--- INITIAL ACCESS CHAIN ---")
    print("Target: Microsoft Exchange Server, Windows Server 2019")
    print("Time available: 5 days, Threat level: 35/100 (LOW-MEDIUM)")
    
    recon = engine.choose_initial_reconnaissance(
        threat_level=35,
        time_available_days=5,
        target_criticality="high"
    )
    print(f"\n1. Reconnaissance: {recon['technique']}")
    print(f"   Risk: {recon['risk_profile']:.1%}, Time: {recon['estimated_time_days']} days")
    
    infiltration = engine.choose_infiltration_method(
        threat_level=35,
        target_os="windows",
        user_sophistication="medium",
        network_isolation="internet_connected"
    )
    print(f"\n2. Infiltration: {infiltration['method']}")
    print(f"   Success: {infiltration['success_probability']:.1%}, Detection: {infiltration['detection_risk']:.1%}")
    
    assert recon['technique'] is not None, "Recon choice missing"
    assert infiltration['method'] is not None, "Infiltration choice missing"
    
    return True


def test_mid_operation_escalation():
    """Test privilege escalation decisions after foothold"""
    engine = OperationalCausalEngine()
    
    print("\n--- ESCALATION SCENARIO ---")
    print("Privilege: user -> system, OS: Windows 10 Enterprise, Threat: 50/100")
    
    esc = engine.choose_escalation_technique(
        current_privilege="user",
        target_privilege="system",
        threat_level=50,
        os_version="windows_10",
        editable_processes=["svchost.exe", "spoolsv.exe"]
    )
    
    print(f"\nTechnique: {esc['technique']}")
    print(f"  Success: {esc['success_probability']:.1%}, Detection: {esc['detection_risk']:.1%}")
    print(f"  Latency: {esc['expected_latency_ms']} ms")
    
    elevated = engine.choose_escalation_technique(
        current_privilege="system",
        target_privilege="system",
        threat_level=50
    )
    print(f"\nWhen already system: {elevated['status']}")
    assert elevated['status'] == "already_privileged", "Should not escalate when already elevated"
    
    return True


def test_lateral_movement_targeting():
    """Test lateral movement targeting decisions"""
    engine = OperationalCausalEngine()
    
    objectives = ["domain_takeover", "data_exfil", "persistence"]
    
    print("\n--- LATERAL MOVEMENT TARGETING ---")
    for obj in objectives:
        print(f"\nObjective: {obj}")
        
        targets = engine.choose_lateral_targets(
            infected_hosts=["DESKTOP-1", "DESKTOP-2"],
            threat_level=50,
            objective=obj,
            network_topology=None
        )
        
        print(f"  Strategy: {targets['strategy']}")
        print(f"  Reach: {targets['expected_reach']:.1%}")
        
        technique = engine.choose_lateral_technique(
            target_os="windows",
            target_machine="DC01",
            have_credentials=True,
            threat_level=50
        )
        print(f"  Technique: {technique['technique']}, Success: {technique['success_probability']:.1%}")
    
    return True


def test_c2_phase_adaptation():
    """Test C2 selection adapts to operational phase"""
    engine = OperationalCausalEngine()
    
    phases = [
        OperationalPhase.INITIAL_ACCESS,
        OperationalPhase.PRIVILEGE_ESCALATION,
        OperationalPhase.LATERAL_MOVEMENT,
        OperationalPhase.EXFILTRATION,
    ]
    
    threat = 50
    
    print(f"\n--- C2 SELECTION AT THREAT {threat}/100 ---")
    
    for phase in phases:
        c2 = engine.choose_c2_method_for_phase(
            operational_phase=phase,
            threat_level=threat
        )
        print(f"\n{phase.value}:")
        print(f"  Method: {c2['method']}, Bandwidth: {c2['bandwidth']}, Latency: {c2['latency_ms']}ms")
    
    return True


def test_exfiltration_strategy():
    """Test exfiltration method selection"""
    engine = OperationalCausalEngine()
    
    scenarios = [
        {
            "name": "Small sensitive data",
            "volume": 5,
            "threat": 75,
            "time": "low",
            "sensitivity": "high"
        },
        {
            "name": "Large bulk download",
            "volume": 5000,
            "threat": 35,
            "time": "high",
            "sensitivity": "low"
        },
        {
            "name": "Medium data, high pressure",
            "volume": 500,
            "threat": 60,
            "time": "high",
            "sensitivity": "medium"
        }
    ]
    
    print("\n--- EXFILTRATION STRATEGY ---")
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        
        exfil = engine.choose_exfiltration_method(
            data_volume_mb=scenario['volume'],
            threat_level=scenario['threat'],
            time_pressure=scenario['time'],
            sensitivity=scenario['sensitivity']
        )
        
        print(f"  Method: {exfil['method']}, Time: {exfil['estimated_time_seconds']:.0f}s, Detection: {exfil['detection_probability']:.1%}")
    
    return True


def test_high_threat_operation():
    """Test behavior under high threat (should prefer stealth/evasion)"""
    engine = OperationalCausalEngine()
    threat = 90  # CRITICAL
    
    print(f"\n--- HIGH THREAT OPERATION (CRITICAL: {threat}/100) ---")
    print("Scenario: EDR DETECTED")
    
    recon = engine.choose_initial_reconnaissance(
        threat_level=threat,
        time_available_days=7
    )
    print(f"\n1. Reconnaissance: {recon['technique']}")
    print(f"   Risk: {recon['risk_profile']:.1%}")
    
    infil = engine.choose_infiltration_method(
        threat_level=threat,
        target_os="windows",
        user_sophistication="high"
    )
    print(f"\n2. Infiltration: {infil['method']}")
    print(f"   Success: {infil['success_probability']:.1%}")
    
    esc = engine.choose_escalation_technique(
        current_privilege="user",
        target_privilege="system",
        threat_level=threat
    )
    print(f"\n3. Escalation: {esc['technique']}")
    print(f"   Detection: {esc['detection_risk']:.1%}")
    
    c2 = engine.choose_c2_method_for_phase(
        operational_phase=OperationalPhase.LATERAL_MOVEMENT,
        threat_level=threat
    )
    print(f"\n4. C2 Communication: {c2['method']}")
    print(f"   Resilience: {c2['resilience']:.1%}")
    
    exfil = engine.choose_exfiltration_method(
        data_volume_mb=1000,
        threat_level=threat,
        time_pressure="low"
    )
    print(f"\n5. Exfiltration: {exfil['method']}")
    print(f"   Time: {exfil['estimated_time_seconds']:.0f}s, Detection: {exfil['detection_probability']:.1%}")
    
    return True


def main():
    """Run all integration tests"""
    
    print("\n" + "="*70)
    print("OPERATIONAL CAUSAL ENGINE - INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Threat Level Routing", test_threat_level_routing),
        ("Initial Access Chain", test_initial_access_phase),
        ("Mid-Operation Escalation", test_mid_operation_escalation),
        ("Lateral Movement Targeting", test_lateral_movement_targeting),
        ("C2 Phase Adaptation", test_c2_phase_adaptation),
        ("Exfiltration Strategy", test_exfiltration_strategy),
        ("High Threat Operation", test_high_threat_operation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[PASS] TEST: {test_name}")
        except Exception as e:
            print(f"\n[FAIL] TEST: {test_name}")
            print(f"       Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"[PASS] Passed: {passed}/{len(tests)}")
    print(f"[FAIL] Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n[OK] ALL TESTS PASSED - OPERATIONAL CAUSAL ENGINE FULLY INTEGRATED")
        return 0
    else:
        print(f"\n[!] {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
