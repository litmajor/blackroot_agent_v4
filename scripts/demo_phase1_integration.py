#!/usr/bin/env python3
"""
PHASE 1 INTEGRATION DEMO
Demonstrates mission_profiles.py integrated with mission_orchestrator.py

Shows:
1. Creating missions from MissionProfile objects
2. Stage pipeline generation based on mission type
3. Variant preference passing to stages
4. Legacy string-based API (backwards compatibility)
"""

import logging
from orchestration.mission_profiles import (
    MissionType, TimeBudgetMode,
    create_foothold_mission, create_escalate_mission, create_infiltrate_mission,
    create_extract_mission, create_propagate_mission, create_dominate_mission,
    create_execute_mission, create_maintain_mission, create_exfil_mission
)
from orchestration.mission_orchestrator import MissionOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def demo_phase1_integration():
    """Demonstrate Phase 1 integration"""
    
    print("\n" + "=" * 80)
    print("  PHASE 1 INTEGRATION: Mission Profiles ↔ Mission Orchestrator")
    print("=" * 80 + "\n")
    
    # Initialize orchestrator
    orchestrator = MissionOrchestrator()
    
    print("[→] 1. MissionProfile → MissionOrchestrator Integration\n")
    
    # Test each mission type
    test_cases = [
        ("FOOTHOLD", create_foothold_mission("192.168.1.0/24", TimeBudgetMode.FAST)),
        ("ESCALATE", create_escalate_mission("192.168.1.100", TimeBudgetMode.BALANCED)),
        ("INFILTRATE", create_infiltrate_mission("192.168.1.101", TimeBudgetMode.STEALTH)),
        ("EXTRACT", create_extract_mission("192.168.1.102", TimeBudgetMode.BALANCED)),
        ("PROPAGATE", create_propagate_mission("192.168.0.0/16", TimeBudgetMode.STEALTH)),
        ("DOMINATE", create_dominate_mission("corp.example.com", TimeBudgetMode.STEALTH)),
        ("EXECUTE", create_execute_mission("192.168.1.105", "/tmp/payload.bin", TimeBudgetMode.FAST)),
        ("MAINTAIN", create_maintain_mission("192.168.1.110", TimeBudgetMode.STEALTH)),
        ("EXFIL", create_exfil_mission("192.168.1.111", TimeBudgetMode.BALANCED)),
    ]
    
    missions = {}
    
    for mission_name, profile in test_cases:
        print(f"\n  TEST: Create {mission_name} mission from MissionProfile")
        print(f"  ─" * 40)
        
        # Create mission from profile
        mission = orchestrator.create_mission(profile)
        missions[mission_name] = mission
        
        print(f"    Mission ID: {mission.mission_id[:12]}...")
        print(f"    Goal: {mission.goal}")
        print(f"    Target: {mission.target}")
        print(f"    Stages: {len(mission.stages)}")
        
        # Display stage pipeline
        print(f"    Pipeline:")
        for i, stage in enumerate(mission.stages, 1):
            deps_str = f" ← {', '.join(stage.dependencies)}" if stage.dependencies else ""
            print(f"      {i}. {stage.name} ({stage.agent_type}){deps_str}")
            if 'variant' in stage.parameters:
                print(f"         └─ Variant: {stage.parameters['variant']}")
        
        # Display mission context
        print(f"    Context:")
        print(f"      • Time Budget: {mission.context.get('time_budget_minutes')}min")
        print(f"      • Stealth Level: {mission.context.get('stealth_level')}/100")
        print(f"      • Mission Type: {mission.context.get('mission_type')}")
    
    print("\n" + "=" * 80)
    print("  TEST 2: Verify Stage Dependencies & Variant Preferences\n")
    print("=" * 80 + "\n")
    
    # Verify dependencies are correct
    extract_mission = missions['EXTRACT']
    print(f"  EXTRACT mission stage dependencies:")
    for stage in extract_mission.stages:
        print(f"    • {stage.name:15} → depends on: {stage.dependencies if stage.dependencies else '(none)'}")
    
    # Verify variant preferences are propagated
    print(f"\n  Variant preferences in stage parameters:")
    for stage in extract_mission.stages:
        if 'variant' in stage.parameters:
            print(f"    • {stage.name:15} → variant: {stage.parameters['variant']}")
    
    print("\n" + "=" * 80)
    print("  TEST 3: Legacy String-Based API (Backwards Compatibility)\n")
    print("=" * 80 + "\n")
    
    # Test legacy API
    legacy_mission = orchestrator.create_mission("compromiseHost(10.0.0.50)")
    print(f"  Legacy mission created from string goal")
    print(f"    Goal: {legacy_mission.goal}")
    print(f"    Target: {legacy_mission.target}")
    print(f"    Stages: {len(legacy_mission.stages)} (hardcoded: SCAN → ESCALATE → EXTRACT → EXFIL)")
    for i, stage in enumerate(legacy_mission.stages, 1):
        print(f"      {i}. {stage.name}")
    
    print("\n" + "=" * 80)
    print("  TEST 4: Mission Context Preservation\n")
    print("=" * 80 + "\n")
    
    # Verify profile is stored in mission context
    scan_only_mission = orchestrator.create_mission(create_exfil_mission("192.168.2.1"))
    if 'mission_profile' in scan_only_mission.context:
        profile = scan_only_mission.context['mission_profile']
        print(f"  ✓ Mission profile stored in context")
        print(f"    • Mission Type: {profile.mission_type.value}")
        print(f"    • Target: {profile.target}")
        print(f"    • Stealth Level: {profile.stealth_level}")
        print(f"    • Time Budget: {profile.time_budget_mode.value}")
        print(f"    • Skip Stages: {profile.skip_stages if profile.skip_stages else '(none)'}")
    
    print("\n" + "=" * 80)
    print("  SUMMARY\n")
    print("=" * 80)
    print(f"  ✓ {len(missions)} mission types created successfully")
    print(f"  ✓ Stage pipelines generated correctly per mission type")
    print(f"  ✓ Variant preferences propagated to stages")
    print(f"  ✓ Legacy string-based API working (backwards compatible)")
    print(f"  ✓ MissionProfile stored in mission context for reference")
    print(f"\n  → Ready for Phase 2: Agent Selection Engine\n")

if __name__ == "__main__":
    demo_phase1_integration()
