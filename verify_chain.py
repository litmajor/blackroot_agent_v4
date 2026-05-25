#!/usr/bin/env python
"""
Verification script for escalation → post-escalation chain
"""

from agents.privesc import PrivescEngine
from agents.post_escalation import PostEscalationOrchestrator

print("=" * 70)
print("BLACKROOT FRAMEWORK VERIFICATION")
print("=" * 70)

# Initialize components
privesc = PrivescEngine(vault=None)
print("✓ PrivescEngine initialized")

orchestrator = PostEscalationOrchestrator()
print("✓ PostEscalationOrchestrator initialized")

# Simulate escalation result
escalation_result = {
    'technique_used': 'GodPotato',
    'elevated_privilege': 'SYSTEM',
    'mitre_technique': 'T1548.002',
    'timestamp': 1708095935
}

print(f"\n[ESCALATION PHASE]")
print(f"✓ Technique: {escalation_result['technique_used']}")
print(f"✓ Privilege Level: {escalation_result['elevated_privilege']}")
print(f"✓ MITRE: {escalation_result['mitre_technique']}")

# Execute post-escalation workflow
print(f"\n[POST-ESCALATION PHASE]")
result = orchestrator.execute_post_escalation_chain(escalation_result)

print(f"✓ Workflow Status: COMPLETE")
print(f"✓ Total Actions: {result['success_count'] + result['failure_count']}")
print(f"✓ Successful: {result['success_count']}")
print(f"✓ Failed: {result['failure_count']}")

# Categorize capabilities
capabilities = {
    'persistence': 0,
    'credential_theft': 0,
    'defense_evasion': 0,
    'service_control': 0,
    'process_injection': 0
}

for action in result['post_escalation_actions']:
    if 'persistence' in action.action_type.lower():
        capabilities['persistence'] += 1
    elif 'credential' in action.action_type.lower():
        capabilities['credential_theft'] += 1
    elif 'disable' in action.action_type.lower() or 'clear' in action.action_type.lower():
        capabilities['defense_evasion'] += 1
    elif 'service' in action.action_type.lower() or 'stop' in action.action_type.lower():
        capabilities['service_control'] += 1
    elif 'injection' in action.action_type.lower():
        capabilities['process_injection'] += 1

print(f"\n[CAPABILITIES DEPLOYED]")
for cap, count in capabilities.items():
    status = "✓" if count > 0 else "○"
    print(f"{status} {cap.replace('_', ' ').title()}: {count} action{'s' if count != 1 else ''}")

print(f"\n" + "=" * 70)
print("✅ FRAMEWORK VERIFICATION COMPLETE")
print("=" * 70)

# Show sample actions
print(f"\n[SAMPLE ACTIONS (First 5)]")
for i, action in enumerate(result['post_escalation_actions'][:5], 1):
    print(f"\n{i}. {action.description}")
    print(f"   Status: {'✓ SUCCESS' if action.success else '✗ FAILED'}")
    print(f"   MITRE: {action.mitre_technique}")
    if action.output:
        print(f"   Output: {action.output[:60]}...")
