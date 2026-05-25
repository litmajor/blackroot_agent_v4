#!/usr/bin/env python3
"""
MISSION PROFILES - Mission Type Taxonomy & Configuration
Defines mission types, time budgets, agent variants, and mission execution strategies
Proportionate to the Blackroot Agent capabilities
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

# =============================================================================
# MISSION TYPE TAXONOMY (9 primary mission types)
# =============================================================================

class MissionType(Enum):
    """Mission types proportionate to Blackroot capabilities"""
    
    # Horizontal: First Compromise → Full Network Takeover
    FOOTHOLD = "FOOTHOLD"           # Initial network access, establish first agent
    ESCALATE = "ESCALATE"           # Single-host privilege escalation
    INFILTRATE = "INFILTRATE"       # Deep covert presence + persistence via GhostLayer
    EXTRACT = "EXTRACT"             # Credential/data harvesting (causal-aware)
    PROPAGATE = "PROPAGATE"         # Lateral movement + network-wide compromise via mesh
    DOMINATE = "DOMINATE"           # Full network takeover + strategic targeting
    
    # Orthogonal: Capability-Specific Missions
    EXECUTE = "EXECUTE"             # Binary execution + payload delivery (complementary to FOOTHOLD)
    MAINTAIN = "MAINTAIN"           # Evasion + hibernation (air-gap safe, post-compromise)
    EXFIL = "EXFIL"                 # Dedicated exfiltration (can be standalone or post-EXTRACT)


# =============================================================================
# AGENT VARIANTS (Matched to Real Agent Implementations)
# =============================================================================

class ScoutVariant(Enum):
    """Scout agent variants - Network reconnaissance"""
    SIGMA = "SIGMA"    # Deep network analysis (thorough, slow, high confidence)
    ALPHA = "ALPHA"    # Broad coverage scanning (fast, shallow, moderate confidence)
    DELTA = "DELTA"    # Threat assessment focus (balanced, targets prioritization)
    OMEGA = "OMEGA"    # Stealth reconnaissance (evasion-optimized, slower, very high stealth)


class InfiltratorVariant(Enum):
    """Infiltrator agent variants - Covert presence & persistence"""
    BLACK = "BLACK"    # Stealth specialist (+15% stealth, slower, persistence-focused)
    RED = "RED"        # Aggressive infiltrator (-5% stealth, fast, quick entry)
    GHOST = "GHOST"    # Intelligence specialist (+10% stealth, information-gathering)
    SILK = "SILK"      # Traffic mimic (+8% stealth, blends with normal traffic)


class ExfiltratorVariant(Enum):
    """Exfiltrator agent variants - Data extraction & exfiltration"""
    STEALTH = "STEALTH"        # Ultra-stealth, minimal detection risk, very slow
    CONSERVATIVE = "CONSERVATIVE"  # High stealth, low risk, slow
    BALANCED = "BALANCED"       # Moderate stealth/speed trade-off
    AGGRESSIVE = "AGGRESSIVE"   # Low stealth, high speed, moderate risk
    BRUTAL = "BRUTAL"           # Minimal stealth, maximum speed, high risk


class PrivescVariant(Enum):
    """Privilege escalation variants"""
    STEALTHY = "STEALTHY"       # UAC bypass, kernel exploits (Windows/Linux)
    STANDARD = "STANDARD"       # Token duplication, service exploitation
    AGGRESSIVE = "AGGRESSIVE"   # Multiple techniques in parallel
    FALLBACK = "FALLBACK"       # Lateral movement instead of direct escalation


class PropagationVariant(Enum):
    """Lateral movement & propagation variants"""
    MESH = "MESH"               # P2P via SwarmMesh + PeerLinkerAgent (fast, distributed)
    CRAWL = "CRAWL"             # Sequential host-to-host (stealth, detectable)
    JUMP = "JUMP"               # Large subnet jumps (risky, fast)
    PATIENT = "PATIENT"         # Slow propagation (very stealthy, time-expensive)


# =============================================================================
# TIME BUDGET MODES & VARIANT MAPPINGS
# =============================================================================

class TimeBudgetMode(Enum):
    """Time budget modes - Map to agent variant strategies"""
    FAST = "FAST"               # 15-30 minutes, speed-optimized
    BALANCED = "BALANCED"        # 1-4 hours, balanced approach
    STEALTH = "STEALTH"         # 4-24 hours, evasion-optimized


@dataclass
class TimeBudgetProfile:
    """Maps TimeBudgetMode to agent variant preferences and behavior policy"""
    mode: TimeBudgetMode
    time_budget_minutes: int
    scout_variant: ScoutVariant
    infiltrator_variant: InfiltratorVariant
    exfiltrator_variant: ExfiltratorVariant
    privesc_variant: PrivescVariant
    propagation_variant: PropagationVariant
    
    # Behavior policy (defaults if not overridden by mission)
    behavioral_noise_enabled: bool  # Add decoy traffic
    process_injection_preferred: bool  # Use GhostLayer for stealth
    edr_caution_level: float  # 0-1, higher = more cautious about EDR
    detection_risk_tolerance: float  # 0-1, acceptable detection risk


# Define time budget profiles
TIME_BUDGET_PROFILES = {
    TimeBudgetMode.FAST: TimeBudgetProfile(
        mode=TimeBudgetMode.FAST,
        time_budget_minutes=30,
        scout_variant=ScoutVariant.ALPHA,      # Fast, broad scanning
        infiltrator_variant=InfiltratorVariant.RED,  # Fast, aggressive entry
        exfiltrator_variant=ExfiltratorVariant.AGGRESSIVE,  # Speed over stealth
        privesc_variant=PrivescVariant.STANDARD,  # Standard techniques
        propagation_variant=PropagationVariant.JUMP,  # Fast propagation
        edr_caution_level=0.3,
        behavioral_noise_enabled=False,
        process_injection_preferred=False,
        detection_risk_tolerance=0.7  # High risk acceptable
    ),
    TimeBudgetMode.BALANCED: TimeBudgetProfile(
        mode=TimeBudgetMode.BALANCED,
        time_budget_minutes=120,
        scout_variant=ScoutVariant.DELTA,      # Balanced, threat-focused
        infiltrator_variant=InfiltratorVariant.GHOST,  # Intelligence-focused
        exfiltrator_variant=ExfiltratorVariant.BALANCED,  # Balanced approach
        privesc_variant=PrivescVariant.STANDARD,  # Standard techniques
        propagation_variant=PropagationVariant.MESH,  # Distributed propagation
        edr_caution_level=0.5,
        behavioral_noise_enabled=True,
        process_injection_preferred=True,
        detection_risk_tolerance=0.5  # Moderate risk
    ),
    TimeBudgetMode.STEALTH: TimeBudgetProfile(
        mode=TimeBudgetMode.STEALTH,
        time_budget_minutes=360,  # 6 hours default
        scout_variant=ScoutVariant.SIGMA,      # Deep, thorough analysis
        infiltrator_variant=InfiltratorVariant.BLACK,  # Stealth-focused persistence
        exfiltrator_variant=ExfiltratorVariant.STEALTH,  # Ultra-stealthy extraction
        privesc_variant=PrivescVariant.STEALTHY,  # Evasion-focused escalation
        propagation_variant=PropagationVariant.PATIENT,  # Slow, careful propagation
        edr_caution_level=0.9,
        behavioral_noise_enabled=True,
        process_injection_preferred=True,
        detection_risk_tolerance=0.2  # Low risk only
    ),
}


# =============================================================================
# RESOLVED EXECUTION PLAN
# =============================================================================
# This is the SINGLE SOURCE OF TRUTH for agents.
# All competing configuration sources are merged into this at mission creation time.
# Agents NEVER read raw MissionProfile or TimeBudgetProfile.
# Agents ONLY read ResolvedExecutionPlan.
# =============================================================================

@dataclass
class ResolvedExecutionPlan:
    """
    Single compiled execution plan for a mission.
    Resolves all competing configuration sources into final behavior.
    
    This is what agents actually use to make decisions.
    Everything below this line is input; everything above should feed into this.
    """
    # Agent variants (resolved from preferences or time budget defaults)
    scout_variant: ScoutVariant
    infiltrator_variant: InfiltratorVariant
    exfiltrator_variant: ExfiltratorVariant
    privesc_variant: PrivescVariant
    propagation_variant: PropagationVariant

    # Behavior flags (FINAL RESOLVED VALUES - single source of truth)
    behavioral_noise: bool
    process_injection: bool
    edr_caution_level: float
    detection_risk_tolerance: float
    
    # Intent vs Mechanism (what to do, not how to do it)
    data_transfer_strategy: str  # "low_visibility", "balanced", "high_speed" instead of implementation detail
    
    # Additional resolved parameters
    stealth_level: int  # 0-100
    persistence_required: bool
    time_budget_minutes: int


# =============================================================================


@dataclass
class MissionProfile:
    """
    Comprehensive mission configuration
    Defines how a specific mission should be executed across all stages
    """
    
    # Mission identity (required fields first)
    mission_id: str
    mission_type: MissionType
    target: str  # Single IP, subnet (CIDR), or domain
    
    # Optional mission metadata
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    target_scope: str = "single"  # "single", "subnet", "network", "domain"
    
    # Time constraints
    time_budget_mode: TimeBudgetMode = TimeBudgetMode.BALANCED
    time_budget_minutes: int = 120
    hard_deadline: Optional[datetime] = None
    
    # Operational constraints
    stealth_level: int = 50  # 0-100, higher = more stealthy
    priority_data_types: List[str] = field(default_factory=lambda: ["credentials", "keys"])
    max_retries_per_stage: int = 3
    skip_stages: List[str] = field(default_factory=list)  # Skip specific stages
    allowed_techniques: List[str] = field(default_factory=list)  # Empty = all allowed
    forbidden_techniques: List[str] = field(default_factory=list)
    
    # Agent variant preferences (override time budget defaults)
    scout_variant_preference: Optional[ScoutVariant] = None
    infiltrator_variant_preference: Optional[InfiltratorVariant] = None
    exfiltrator_variant_preference: Optional[ExfiltratorVariant] = None
    privesc_variant_preference: Optional[PrivescVariant] = None
    propagation_variant_preference: Optional[PropagationVariant] = None
    
    # Behavioral tuning
    use_behavioral_noise: bool = True
    use_process_injection: bool = True
    enable_edr_detection: bool = True
    credential_causal_aware: bool = True  # Use causal decision engine for EXTRACT
    
    # Mesh & distribution
    use_swarm_mesh: bool = True
    distribute_to_discovered_hosts: bool = False  # Auto-propagate to new hosts in PROPAGATE
    
    # Post-mission
    persistence_required: bool = False  # Leave backdoor after mission
    self_destruct_on_failure: bool = False  # Clean up artifacts on failure
    
    # Data handling
    exfil_method: str = "swarm_mesh"  # "swarm_mesh", "steganography", "dns_tunnel", "c2_channel"
    encrypt_exfil: bool = True
    fragment_exfil: bool = True  # Split data into chunks
    
    # Tracking & checkpoint
    enable_checkpoints: bool = True
    checkpoint_ttl_seconds: int = 86400  # 24 hours
    
    def __post_init__(self):
        """Post-init validation and default assignments"""
        # Load time budget profile if not explicitly set
        if self.time_budget_mode and self.scout_variant_preference is None:
            profile = TIME_BUDGET_PROFILES.get(self.time_budget_mode)
            if profile:
                self.time_budget_minutes = profile.time_budget_minutes
    
    def get_time_budget_profile(self) -> TimeBudgetProfile:
        """Get the TimeBudgetProfile for this mission's time budget mode"""
        return TIME_BUDGET_PROFILES.get(self.time_budget_mode, 
                                       TIME_BUDGET_PROFILES[TimeBudgetMode.BALANCED])
    
    def get_scout_variant(self) -> ScoutVariant:
        """Get scout variant: explicit preference or time budget default"""
        if self.scout_variant_preference:
            return self.scout_variant_preference
        return self.get_time_budget_profile().scout_variant
    
    def get_infiltrator_variant(self) -> InfiltratorVariant:
        """Get infiltrator variant: explicit preference or time budget default"""
        if self.infiltrator_variant_preference:
            return self.infiltrator_variant_preference
        return self.get_time_budget_profile().infiltrator_variant
    
    def get_exfiltrator_variant(self) -> ExfiltratorVariant:
        """Get exfiltrator variant: explicit preference or time budget default"""
        if self.exfiltrator_variant_preference:
            return self.exfiltrator_variant_preference
        return self.get_time_budget_profile().exfiltrator_variant
    
    def get_privesc_variant(self) -> PrivescVariant:
        """Get privesc variant: explicit preference or time budget default"""
        if self.privesc_variant_preference:
            return self.privesc_variant_preference
        return self.get_time_budget_profile().privesc_variant
    
    def get_propagation_variant(self) -> PropagationVariant:
        """Get propagation variant: explicit preference or time budget default"""
        if self.propagation_variant_preference:
            return self.propagation_variant_preference
        return self.get_time_budget_profile().propagation_variant
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON/Redis storage"""
        return {
            "mission_id": self.mission_id,
            "mission_type": self.mission_type.value,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "target": self.target,
            "target_scope": self.target_scope,
            "time_budget_mode": self.time_budget_mode.value,
            "time_budget_minutes": self.time_budget_minutes,
            "hard_deadline": self.hard_deadline.isoformat() if self.hard_deadline else None,
            "stealth_level": self.stealth_level,
            "priority_data_types": self.priority_data_types,
            "max_retries_per_stage": self.max_retries_per_stage,
            "skip_stages": self.skip_stages,
            "allowed_techniques": self.allowed_techniques,
            "forbidden_techniques": self.forbidden_techniques,
            "scout_variant_preference": self.scout_variant_preference.value if self.scout_variant_preference else None,
            "infiltrator_variant_preference": self.infiltrator_variant_preference.value if self.infiltrator_variant_preference else None,
            "exfiltrator_variant_preference": self.exfiltrator_variant_preference.value if self.exfiltrator_variant_preference else None,
            "privesc_variant_preference": self.privesc_variant_preference.value if self.privesc_variant_preference else None,
            "propagation_variant_preference": self.propagation_variant_preference.value if self.propagation_variant_preference else None,
            "use_behavioral_noise": self.use_behavioral_noise,
            "use_process_injection": self.use_process_injection,
            "enable_edr_detection": self.enable_edr_detection,
            "credential_causal_aware": self.credential_causal_aware,
            "use_swarm_mesh": self.use_swarm_mesh,
            "distribute_to_discovered_hosts": self.distribute_to_discovered_hosts,
            "persistence_required": self.persistence_required,
            "self_destruct_on_failure": self.self_destruct_on_failure,
            "exfil_method": self.exfil_method,
            "encrypt_exfil": self.encrypt_exfil,
            "fragment_exfil": self.fragment_exfil,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
        }
    
    @staticmethod
    def from_dict(data: Dict) -> "MissionProfile":
        """Deserialize from dictionary"""
        # Parse variant enums
        scout_pref = ScoutVariant[data['scout_variant_preference']] if data.get('scout_variant_preference') else None
        infil_pref = InfiltratorVariant[data['infiltrator_variant_preference']] if data.get('infiltrator_variant_preference') else None
        exfil_pref = ExfiltratorVariant[data['exfiltrator_variant_preference']] if data.get('exfiltrator_variant_preference') else None
        priv_pref = PrivescVariant[data['privesc_variant_preference']] if data.get('privesc_variant_preference') else None
        prop_pref = PropagationVariant[data['propagation_variant_preference']] if data.get('propagation_variant_preference') else None
        
        return MissionProfile(
            mission_id=data['mission_id'],
            mission_type=MissionType[data['mission_type']],
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            description=data.get('description', ''),
            target=data['target'],
            target_scope=data.get('target_scope', 'single'),
            time_budget_mode=TimeBudgetMode[data.get('time_budget_mode', 'BALANCED')],
            time_budget_minutes=data.get('time_budget_minutes', 120),
            hard_deadline=datetime.fromisoformat(data['hard_deadline']) if data.get('hard_deadline') else None,
            stealth_level=data.get('stealth_level', 50),
            priority_data_types=data.get('priority_data_types', ['credentials', 'keys']),
            max_retries_per_stage=data.get('max_retries_per_stage', 3),
            skip_stages=data.get('skip_stages', []),
            allowed_techniques=data.get('allowed_techniques', []),
            forbidden_techniques=data.get('forbidden_techniques', []),
            scout_variant_preference=scout_pref,
            infiltrator_variant_preference=infil_pref,
            exfiltrator_variant_preference=exfil_pref,
            privesc_variant_preference=priv_pref,
            propagation_variant_preference=prop_pref,
            use_behavioral_noise=data.get('use_behavioral_noise', True),
            use_process_injection=data.get('use_process_injection', True),
            enable_edr_detection=data.get('enable_edr_detection', True),
            credential_causal_aware=data.get('credential_causal_aware', True),
            use_swarm_mesh=data.get('use_swarm_mesh', True),
            distribute_to_discovered_hosts=data.get('distribute_to_discovered_hosts', False),
            persistence_required=data.get('persistence_required', False),
            self_destruct_on_failure=data.get('self_destruct_on_failure', False),
            exfil_method=data.get('exfil_method', 'swarm_mesh'),
            encrypt_exfil=data.get('encrypt_exfil', True),
            fragment_exfil=data.get('fragment_exfil', True),
            enable_checkpoints=data.get('enable_checkpoints', True),
            checkpoint_ttl_seconds=data.get('checkpoint_ttl_seconds', 86400),
        )


# =============================================================================
# MISSION TEMPLATES (Pre-configured profiles for common mission types)
# =============================================================================

def create_foothold_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.BALANCED) -> MissionProfile:
    """Template: FOOTHOLD - Establish initial agent on target network"""
    return MissionProfile(
        mission_id=f"foothold_{int(__import__('time').time())}",
        mission_type=MissionType.FOOTHOLD,
        description="Establish initial network access and deploy first agent",
        target=target,
        target_scope="single" if "/" not in target else "subnet",
        time_budget_mode=time_budget,
        stealth_level=50,
        priority_data_types=["entry_point", "initial_credentials"],
        use_process_injection=False,  # Initial access may not need injection
        persistence_required=True,
        skip_stages=[],
    )


def create_escalate_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.BALANCED) -> MissionProfile:
    """Template: ESCALATE - Privilege escalation on single host"""
    return MissionProfile(
        mission_id=f"escalate_{int(__import__('time').time())}",
        mission_type=MissionType.ESCALATE,
        description="Elevate privileges on target host",
        target=target,
        target_scope="single",
        time_budget_mode=time_budget,
        stealth_level=60,
        priority_data_types=["escalation_technique", "elevated_credentials"],
        use_process_injection=True,
        persistence_required=False,
        skip_stages=["SCAN"],  # Assume already on target
    )


def create_infiltrate_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.STEALTH) -> MissionProfile:
    """Template: INFILTRATE - Deep covert presence via GhostLayer"""
    return MissionProfile(
        mission_id=f"infiltrate_{int(__import__('time').time())}",
        mission_type=MissionType.INFILTRATE,
        description="Establish deep covert presence with persistence via GhostLayer",
        target=target,
        target_scope="single",
        time_budget_mode=time_budget,
        stealth_level=85,
        priority_data_types=["persistence_method", "injected_payload"],
        use_process_injection=True,
        use_behavioral_noise=True,
        persistence_required=True,
        exfil_method="steganography",  # Very stealthy
    )


def create_extract_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.BALANCED) -> MissionProfile:
    """Template: EXTRACT - Harvest credentials & sensitive data"""
    return MissionProfile(
        mission_id=f"extract_{int(__import__('time').time())}",
        mission_type=MissionType.EXTRACT,
        description="Extract credentials, keys, and sensitive data",
        target=target,
        target_scope="single",
        time_budget_mode=time_budget,
        stealth_level=70,
        priority_data_types=["credentials", "private_keys", "database_creds", "sensitive_files"],
        credential_causal_aware=True,  # Use causal decision engine
        use_process_injection=True,
        encrypt_exfil=True,
        fragment_exfil=True,
    )


def create_exfil_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.BALANCED) -> MissionProfile:
    """Template: EXFIL - Dedicated exfiltration mission"""
    return MissionProfile(
        mission_id=f"exfil_{int(__import__('time').time())}",
        mission_type=MissionType.EXFIL,
        description="Dedicated data exfiltration via stealth channels",
        target=target,
        target_scope="single",
        time_budget_mode=time_budget,
        stealth_level=80,
        priority_data_types=["encrypted_data", "fragments"],
        use_swarm_mesh=True,
        encrypt_exfil=True,
        fragment_exfil=True,
    )


def create_propagate_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.STEALTH) -> MissionProfile:
    """Template: PROPAGATE - Lateral movement + network-wide compromise"""
    return MissionProfile(
        mission_id=f"propagate_{int(__import__('time').time())}",
        mission_type=MissionType.PROPAGATE,
        description="Lateral movement and network-wide compromise via mesh",
        target=target,
        target_scope="subnet" if "/" in target else "network",
        time_budget_mode=time_budget,
        stealth_level=75,
        priority_data_types=["compromised_hosts", "lateral_movement_vectors"],
        use_swarm_mesh=True,
        distribute_to_discovered_hosts=True,
        persistence_required=True,
    )


def create_dominate_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.STEALTH) -> MissionProfile:
    """Template: DOMINATE - Full network takeover + strategic targeting"""
    return MissionProfile(
        mission_id=f"dominate_{int(__import__('time').time())}",
        mission_type=MissionType.DOMINATE,
        description="Full network takeover with strategic targeting and enforcement",
        target=target,
        target_scope="domain" if "." in target else "network",
        time_budget_mode=time_budget,
        stealth_level=70,
        priority_data_types=["critical_assets", "domain_admin_creds", "gpo_data"],
        use_swarm_mesh=True,
        distribute_to_discovered_hosts=True,
        persistence_required=True,
        credential_causal_aware=True,
    )


def create_execute_mission(target: str, payload_path: str, time_budget: TimeBudgetMode = TimeBudgetMode.BALANCED) -> MissionProfile:
    """Template: EXECUTE - Binary execution + payload delivery"""
    return MissionProfile(
        mission_id=f"execute_{int(__import__('time').time())}",
        mission_type=MissionType.EXECUTE,
        description=f"Execute payload: {payload_path}",
        target=target,
        target_scope="single",
        time_budget_mode=time_budget,
        stealth_level=60,
        priority_data_types=["execution_result", "output"],
        use_process_injection=True,
        persistence_required=False,
        skip_stages=["EXTRACT", "EXFIL"],  # Binary execution doesn't need data harvest
    )


def create_maintain_mission(target: str, time_budget: TimeBudgetMode = TimeBudgetMode.STEALTH) -> MissionProfile:
    """Template: MAINTAIN - Evasion + hibernation for air-gapped/high-risk environments"""
    return MissionProfile(
        mission_id=f"maintain_{int(__import__('time').time())}",
        mission_type=MissionType.MAINTAIN,
        description="Maintain deep cover: evasion + hibernation for air-gapped systems",
        target=target,
        target_scope="single",
        time_budget_mode=time_budget,
        stealth_level=95,  # Ultra-stealth
        priority_data_types=["evasion_success", "hibernation_capability"],
        use_behavioral_noise=True,
        use_process_injection=True,
        persistence_required=True,
        exfil_method="steganography",  # Minimal network activity
        self_destruct_on_failure=True,  # Clean up if detected
        skip_stages=["EXTRACT", "EXFIL"],  # Maintenance doesn't data-exfil
    )


# Mission template registry
MISSION_TEMPLATES = {
    MissionType.FOOTHOLD: create_foothold_mission,
    MissionType.ESCALATE: create_escalate_mission,
    MissionType.INFILTRATE: create_infiltrate_mission,
    MissionType.EXTRACT: create_extract_mission,
    MissionType.PROPAGATE: create_propagate_mission,
    MissionType.DOMINATE: create_dominate_mission,
    MissionType.EXECUTE: create_execute_mission,
    MissionType.MAINTAIN: create_maintain_mission,
    MissionType.EXFIL: create_exfil_mission,
}


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("MissionProfiles")
