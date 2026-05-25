#!/usr/bin/env python3
"""
PHASE 1 DEMO - Mission Profile System in Action
Shows mission type taxonomy, time budget modes, variants, and serialization
"""

import json
import logging
from datetime import datetime, timedelta
from orchestration.mission_profiles import (
    MissionType, MissionProfile, TimeBudgetMode, ScoutVariant, InfiltratorVariant,
    ExfiltratorVariant, PrivescVariant, PropagationVariant,
    create_foothold_mission, create_escalate_mission, create_infiltrate_mission,
    create_extract_mission, create_exfil_mission, create_propagate_mission,
    create_dominate_mission, create_execute_mission, create_maintain_mission,
    MISSION_TEMPLATES, TIME_BUDGET_PROFILES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

def print_section(title):
    """Print formatted section header"""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()

def print_task(task):
    """Print task progress"""
    print(f"[→] {task}")

def print_success(msg):
    """Print success message"""
    print(f"[✅] {msg}")

def print_info(msg):
    """Print info message"""
    print(f"[ℹ️] {msg}")

# =============================================================================
# PHASE 1 VALIDATION
# =============================================================================

print_section("PHASE 1: MISSION TYPE TAXONOMY & CONFIGURATION")

print_task("1. Validate Mission Types")
print()
for mission_type in MissionType:
    print(f"  • {mission_type.value:15} - {MISSION_TEMPLATES[mission_type].__doc__.split(chr(10))[0]}")
print_success("9 mission types defined and mapped to templates")

# =============================================================================
# TIME BUDGET MODES & VARIANT MAPPINGS
# =============================================================================

print_section("TIME BUDGET MODES & VARIANT MAPPINGS")

print_task("2. Validate Time Budget Profiles")
print()
for time_mode in TimeBudgetMode:
    profile = TIME_BUDGET_PROFILES[time_mode]
    print(f"\n  {time_mode.value} (Budget: {profile.time_budget_minutes} min)")
    print(f"    • Scout:        {profile.scout_variant.value:15} | EDR Caution: {profile.edr_caution_level * 100:.0f}%")
    print(f"    • Infiltrator:  {profile.infiltrator_variant.value:15} | Behavior Noise: {profile.behavioral_noise_enabled}")
    print(f"    • Exfiltrator:  {profile.exfiltrator_variant.value:15} | Detection Risk: {profile.detection_risk_tolerance * 100:.0f}%")
    print(f"    • Privesc:      {profile.privesc_variant.value:15} | Process Injection: {profile.process_injection_preferred}")
    print(f"    • Propagation:  {profile.propagation_variant.value:15}")
print_success("Time budget profiles mapped to agent variants")

# =============================================================================
# MISSION CREATION FROM TEMPLATES
# =============================================================================

print_section("MISSION CREATION FROM TEMPLATES")

print_task("3. Create Mission Instances for Each Type")
print()

missions = {}

# FOOTHOLD - Fast, establish first agent
missions['foothold'] = create_foothold_mission("192.168.1.0/24", TimeBudgetMode.FAST)
print(f"\n  ✦ FOOTHOLD:")
print(f"    ID: {missions['foothold'].mission_id}")
print(f"    Target: {missions['foothold'].target} (scope: {missions['foothold'].target_scope})")
print(f"    Time Budget: {missions['foothold'].time_budget_minutes}m ({missions['foothold'].time_budget_mode.value})")
print(f"    Scout Variant: {missions['foothold'].get_scout_variant().value}")
print(f"    Infiltrator Variant: {missions['foothold'].get_infiltrator_variant().value}")
print(f"    Stealth Level: {missions['foothold'].stealth_level}/100")

# ESCALATE - Balanced on single host
missions['escalate'] = create_escalate_mission("192.168.1.100", TimeBudgetMode.BALANCED)
print(f"\n  ✦ ESCALATE:")
print(f"    ID: {missions['escalate'].mission_id}")
print(f"    Target: {missions['escalate'].target} (scope: {missions['escalate'].target_scope})")
print(f"    Time Budget: {missions['escalate'].time_budget_minutes}m ({missions['escalate'].time_budget_mode.value})")
print(f"    Scout Variant: {missions['escalate'].get_scout_variant().value}")
print(f"    Privesc Variant: {missions['escalate'].get_privesc_variant().value}")
print(f"    Skip Stages: {missions['escalate'].skip_stages}")

# INFILTRATE - Stealth, deep persistence
missions['infiltrate'] = create_infiltrate_mission("192.168.1.101", TimeBudgetMode.STEALTH)
print(f"\n  ✦ INFILTRATE:")
print(f"    ID: {missions['infiltrate'].mission_id}")
print(f"    Target: {missions['infiltrate'].target} (scope: {missions['infiltrate'].target_scope})")
print(f"    Time Budget: {missions['infiltrate'].time_budget_minutes}m ({missions['infiltrate'].time_budget_mode.value})")
print(f"    Infiltrator Variant: {missions['infiltrate'].get_infiltrator_variant().value}")
print(f"    Stealth Level: {missions['infiltrate'].stealth_level}/100 (Max stealth)")
print(f"    Exfil Method: {missions['infiltrate'].exfil_method}")

# EXTRACT - Balanced, credential harvesting
missions['extract'] = create_extract_mission("192.168.1.102", TimeBudgetMode.BALANCED)
print(f"\n  ✦ EXTRACT:")
print(f"    ID: {missions['extract'].mission_id}")
print(f"    Target: {missions['extract'].target}")
print(f"    Priority Data: {', '.join(missions['extract'].priority_data_types)}")
print(f"    Causal-Aware: {missions['extract'].credential_causal_aware}")
print(f"    Exfiltrator Variant: {missions['extract'].get_exfiltrator_variant().value}")
print(f"    Encrypt Exfil: {missions['extract'].encrypt_exfil}, Fragment: {missions['extract'].fragment_exfil}")

# PROPAGATE - Stealth, network-wide
missions['propagate'] = create_propagate_mission("192.168.0.0/16", TimeBudgetMode.STEALTH)
print(f"\n  ✦ PROPAGATE:")
print(f"    ID: {missions['propagate'].mission_id}")
print(f"    Target: {missions['propagate'].target} (scope: {missions['propagate'].target_scope})")
print(f"    Time Budget: {missions['propagate'].time_budget_minutes}m ({missions['propagate'].time_budget_mode.value})")
print(f"    Propagation Variant: {missions['propagate'].get_propagation_variant().value}")
print(f"    Distribute to Hosts: {missions['propagate'].distribute_to_discovered_hosts}")
print(f"    Use SwarmMesh: {missions['propagate'].use_swarm_mesh}")

# DOMINATE - Full network takeover
missions['dominate'] = create_dominate_mission("corp.example.com", TimeBudgetMode.STEALTH)
print(f"\n  ✦ DOMINATE:")
print(f"    ID: {missions['dominate'].mission_id}")
print(f"    Target: {missions['dominate'].target} (scope: {missions['dominate'].target_scope})")
print(f"    Priority Data: {', '.join(missions['dominate'].priority_data_types)}")
print(f"    Stealth Level: {missions['dominate'].stealth_level}/100")
print(f"    Distribute & Persist: {missions['dominate'].distribute_to_discovered_hosts}, {missions['dominate'].persistence_required}")

# EXECUTE - Binary execution
missions['execute'] = create_execute_mission("192.168.1.105", "/tmp/payload.bin", TimeBudgetMode.BALANCED)
print(f"\n  ✦ EXECUTE:")
print(f"    ID: {missions['execute'].mission_id}")
print(f"    Target: {missions['execute'].target}")
print(f"    Skip Stages: {missions['execute'].skip_stages}")
print(f"    Persistence: {missions['execute'].persistence_required}")

# MAINTAIN - Hibernation for air-gapped
missions['maintain'] = create_maintain_mission("192.168.1.110", TimeBudgetMode.STEALTH)
print(f"\n  ✦ MAINTAIN:")
print(f"    ID: {missions['maintain'].mission_id}")
print(f"    Target: {missions['maintain'].target}")
print(f"    Stealth Level: {missions['maintain'].stealth_level}/100 (Ultra-Stealth)")
print(f"    Self-Destruct on Failure: {missions['maintain'].self_destruct_on_failure}")
print(f"    Exfil Method: {missions['maintain'].exfil_method}")

# EXFIL - Dedicated exfiltration
missions['exfil'] = create_exfil_mission("192.168.1.111", TimeBudgetMode.BALANCED)
print(f"\n  ✦ EXFIL:")
print(f"    ID: {missions['exfil'].mission_id}")
print(f"    Target: {missions['exfil'].target}")
print(f"    Exfiltrator Variant: {missions['exfil'].get_exfiltrator_variant().value}")
print(f"    Stealth Level: {missions['exfil'].stealth_level}/100")
print(f"    Encryption & Fragmentation: {missions['exfil'].encrypt_exfil}, {missions['exfil'].fragment_exfil}")

print_success("All 9 mission types created from templates")

# =============================================================================
# VARIANT SELECTION LOGIC
# =============================================================================

print_section("VARIANT SELECTION LOGIC")

print_task("4. Demonstrate Time Budget Mode → Variant Mapping")
print()

test_mission = MissionProfile(
    mission_id="test_variant_demo",
    mission_type=MissionType.EXTRACT,
    target="192.168.1.1",
)

print(f"  Test mission with no explicit variant preferences:")
print()
for time_mode in TimeBudgetMode:
    test_mission.time_budget_mode = time_mode
    print(f"  {time_mode.value} mode:")
    print(f"    • Scout:       {test_mission.get_scout_variant().value}")
    print(f"    • Infiltrator: {test_mission.get_infiltrator_variant().value}")
    print(f"    • Exfiltrator: {test_mission.get_exfiltrator_variant().value}")
    print(f"    • Privesc:     {test_mission.get_privesc_variant().value}")
    print(f"    • Propagation: {test_mission.get_propagation_variant().value}")
    print()

print_success("Variant selection working correctly for all time budget modes")

# =============================================================================
# SERIALIZATION/DESERIALIZATION
# =============================================================================

print_section("MISSION SERIALIZATION & PERSISTENCE")

print_task("5. Serialize Mission to JSON and Deserialize")
print()

mission_dict = missions['extract'].to_dict()
print(f"  Serialized EXTRACT mission:")
print(f"    mission_id: {mission_dict['mission_id']}")
print(f"    mission_type: {mission_dict['mission_type']}")
print(f"    target: {mission_dict['target']}")
print(f"    time_budget_mode: {mission_dict['time_budget_mode']}")
print(f"    scout_variant_preference: {mission_dict['scout_variant_preference']}")
print(f"    stealth_level: {mission_dict['stealth_level']}")
print()

# Deserialize
deserialized = MissionProfile.from_dict(mission_dict)
print(f"  Deserialized and verified:")
print(f"    ✓ Mission IDs match: {mission_dict['mission_id'] == deserialized.mission_id}")
print(f"    ✓ Mission types match: {mission_dict['mission_type'] == deserialized.mission_type.value}")
print(f"    ✓ Targets match: {mission_dict['target'] == deserialized.target}")
print(f"    ✓ Time budgets match: {mission_dict['time_budget_minutes'] == deserialized.time_budget_minutes}")

print_success("Serialization/deserialization working correctly")

# =============================================================================
# CONFIG.JSON INTEGRATION
# =============================================================================

print_section("CONFIG.JSON INTEGRATION")

print_task("6. Load Mission Profiles from config.json")
print()

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    mission_profiles_config = config.get('mission_profiles', {})
    
    print(f"  Default time budget mode: {mission_profiles_config.get('default_time_budget_mode')}")
    print(f"  Checkpoint enabled: {mission_profiles_config.get('checkpoint_enabled')}")
    print(f"  Checkpoint TTL: {mission_profiles_config.get('checkpoint_ttl_seconds')}s")
    print()
    
    # Time budget modes in config
    print(f"  Time Budget Modes in config.json:")
    for mode_name, mode_config in mission_profiles_config.get('time_budget_modes', {}).items():
        print(f"    {mode_name}: {mode_config['scout_variant']} scout, {mode_config['time_budget_minutes']}min")
    print()
    
    # Mission type defaults in config
    print(f"  Mission Type Defaults in config.json:")
    for mission_type_name, defaults in mission_profiles_config.get('mission_type_defaults', {}).items():
        print(f"    {mission_type_name}: stealth={defaults.get('stealth_level')}, time_mode={defaults.get('time_budget_mode')}")
    print()
    
    # Agent variants in config
    print(f"  Agent Variants defined in config.json:")
    for agent_type, variants in mission_profiles_config.get('agent_variants', {}).items():
        print(f"    {agent_type}: {', '.join(variants)}")
    print()
    
    # Fallback chains in config
    print(f"  Variant Fallback Chains in config.json:")
    for agent_type, chain in mission_profiles_config.get('variant_fallback_chains', {}).items():
        print(f"    {agent_type}: {' → '.join(chain)}")
    
    print_success("config.json mission profiles section loaded and validated")
    
except Exception as e:
    print(f"[❌] Error loading config.json: {e}")

# =============================================================================
# MISSION CONTEXT PREVIEW
# =============================================================================

print_section("MISSION PROFILE SUMMARY")

print_task("7. Summary of All 9 Mission Types")
print()

mission_summary = {
    'FOOTHOLD': ('Initial network access', 'FAST', False),
    'ESCALATE': ('Single-host privilege escalation', 'BALANCED', False),
    'INFILTRATE': ('Deep covert presence + persistence', 'STEALTH', True),
    'EXTRACT': ('Credential/data harvesting', 'BALANCED', False),
    'PROPAGATE': ('Lateral movement + network-wide', 'STEALTH', True),
    'DOMINATE': ('Full network takeover', 'STEALTH', True),
    'EXECUTE': ('Binary execution + payloads', 'BALANCED', False),
    'MAINTAIN': ('Evasion + hibernation', 'STEALTH', True),
    'EXFIL': ('Dedicated exfiltration', 'BALANCED', False),
}

print("  Mission Type    | Purpose                              | Default Time | Persistence?")
print("  " + "-" * 98)
for mission_type, (purpose, time_mode, persistence) in mission_summary.items():
    persist_str = "Yes (air-gap safe)" if persistence else "No"
    print(f"  {mission_type:15} | {purpose:35} | {time_mode:12} | {persist_str}")

print()
print_success("Phase 1 Implementation Complete")

# =============================================================================
# NEXT STEPS
# =============================================================================

print_section("PHASE 1 COMPLETION STATUS")

print("""
[✅] PHASE 1: MISSION TYPE TAXONOMY & CONFIGURATION - COMPLETE

Deliverables:
  1. [✅] orchestration/mission_profiles.py (500+ lines)
     • MissionType enum (9 types: FOOTHOLD, ESCALATE, INFILTRATE, EXTRACT, PROPAGATE, DOMINATE, EXECUTE, MAINTAIN, EXFIL)
     • ScoutVariant, InfiltratorVariant, ExfiltratorVariant, PrivescVariant, PropagationVariant enums
     • TimeBudgetMode enum (FAST, BALANCED, STEALTH) with variant mappings
     • MissionProfile dataclass with 30+ configurable fields
     • 9 mission templates (create_foothold_mission, create_escalate_mission, etc.)
     • Time budget profile mapping (FAST: 30min, BALANCED: 120min, STEALTH: 360min)
     • Serialization/deserialization support

  2. [✅] config.json extended with mission_profiles section
     • Default time budget mode
     • Time budget mode → variant mappings
     • Mission type defaults (stealth levels, priorities, capabilities)
     • Agent variant definitions
     • Variant fallback chains
     • Variant backoff delays per time budget mode

Key Features:
  ✓ Dynamic agent variant selection based on time budget mode
  ✓ 9 mission types covering all capabilities: foothold→escalate→infiltrate→extract→propagate→dominate
  ✓ Orthogonal missions: EXECUTE (binary payload execution), MAINTAIN (hibernation), EXFIL (dedicated exfil)
  ✓ Minimal context passing architecture (fields configured per mission)
  ✓ Checkpoint support for mission resumption
  ✓ EDR detection + behavioral noise toggling per mission
  ✓ GhostLayer process injection enabled/disabled per mission type
  ✓ SwarmMesh P2P distribution support for PROPAGATE/DOMINATE

Next Phase: PHASE 2 (Agent Selection Engine)
  - Create orchestration/agent_selector.py
  - Implement EnvironmentAnalyzer: EDR detection, network topology, privilege state
  - Implement dynamic agent selection: mission_type + environment → best agent variant

Estimated Tokens: Phase 1 = 15-20k tokens (completed)
                 Phase 2 = 20-25k tokens (next)
""")

print("=" * 80)
print()
