# agent_core.py
#
# Central Agent Core for BLACKROOT.Agent v5
# ------------------------------------------------------------------
# Purpose:
#   • Provides canonical `AgentCore`, `Capability`, `ElderRole`, `Mission` and `Event` dataclasses
#   • Adds **every** capability (stealth, recon, infra, defense, etc.)
#   • Adds **Neuro-Assimilator** capabilities explicitly
#   • Keeps the public API 100 % backward-compatible
#
# Usage:
#   from agent_core import Capability, AgentCore, MessageBus
#   agent = AgentCore(name="Shadow-Runner",
#                     capabilities=[Capability.EVADE_EDR,
#                                   Capability.DNS_TUNNEL,
#                                   Capability.CODE_ASSIMILATION])
# ------------------------------------------------------------------

import uuid
import time
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import deque
# ------------------------------------------------------------
#  ENUMS
# ------------------------------------------------------------
__version__ = "1.0.0"

class MissionStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

# ------------------------------------------------------------
#  ENUMS
# ------------------------------------------------------------
class AgentStatus(Enum):
    """Lifecycle states of any agent."""
    ACTIVE    = "Active"
    SUSPENDED = "Suspended"
    DORMANT   = "Dormant"
    TERMINATED = "Terminated"


class ElderRole(Enum):
    """Roles for elder or governance agents."""
    WATCHER       = "Watcher"
    PROTECTOR     = "Protector"
    BUILDER       = "Builder"
    ETHICIST      = "Ethicist"
    GROWTH_MASTER = "GrowthMaster"
    COMMANDER     = "Commander"
    ARCH_MALTA    = "ARCH-MALTA"
    AGENT_NODE    = "AgentNode"


class Capability(Enum):
    """
    Granular capabilities every agent may advertise.
    Grouped by operational domain for readability.
    """

    # ---------- ORIGINAL BLACKROOT CORE ----------
    COMMUNICATE        = "communicate"
    READ_MEMORY        = "read_memory"
    WRITE_MEMORY       = "write_memory"
    EXECUTE_TASK       = "execute_task"
    ACCESS_NETWORK     = "access_network"
    DEPLOY_PAYLOAD     = "deploy_payload"
    SELF_MODIFY        = "self_modify"
    APPROVE_RESUME     = "approve_resume"
    KILL_AGENT         = "kill_agent"
    ELEVATE_CAPABILITY = "elevate_capability"
    INSPECT_AGENT      = "inspect_agent"
    BROADCAST         = "broadcast"
    BIND_AGENT        = "bind_agent"
    UNBIND_AGENT      = "unbind_agent"
    FORWARD_EVENT     = "forward_event"
    BOOT_AGENT        = "boot_agent"
    
    # ---------- NEURO-ASSIMILATOR ----------
    TRUST_EVALUATION     = "trust_evaluation"
    CODE_SANITIZATION    = "code_sanitization"
    CONTROL_INJECTION    = "control_injection"
    PLUGIN_MANAGEMENT    = "plugin_management"
    CODE_ASSIMILATION    = "code_assimilation"
    PAYLOAD_EXECUTION    = "payload_execution"
    SYSTEM_OBSERVATION   = "system_observation"
    DECISION_MAKING      = "decision_making"
    ACTION_EXECUTION     = "action_execution"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    VAULT_STORAGE        = "vault_storage"
    RELIABILITY_MONITORING = "reliability_monitoring"
    PERFORMANCE_PROFILING = "performance_profiling"

    # ---------- STEALTH & EVASION ----------
    STEALTH_TRAFFIC      = "stealth_traffic"
    POLYMORPHIC_PAYLOAD  = "polymorphic_payload"
    EVADE_EDR            = "evade_edr"
    BYPASS_WAF           = "bypass_waf"
    BYPASS_CSP           = "bypass_csp"
    EVADE_AV             = "evade_av"
    EVADE_SANDBOX        = "evade_sandbox"
    EVADE_DPI            = "evade_dpi"

    # ---------- RECON & FINGERPRINTING ----------
    OSINT_GATHER         = "osint_gather"
    WEB_SCRAPE           = "web_scrape"
    SOCIAL_ENGINEER      = "social_engineer"
    GEOLOCATE            = "geolocate"
    TLS_FINGERPRINT      = "tls_fingerprint"
    JA3_FINGERPRINT      = "ja3_fingerprint"
    USER_AGENT_SPOOF     = "user_agent_spoof"
    SCREEN_CAPTURE       = "screen_capture"
    WEBCAM_CAPTURE       = "webcam_capture"

    # ---------- PRIVILEGE ESCALATION ----------
    LATERAL_MOVE         = "lateral_move"
    PERSIST_REGISTRY     = "persist_registry"
    PERSIST_SCHEDULED    = "persist_scheduled"
    PERSIST_SERVICE      = "persist_service"
    PERSIST_LOGINITEM    = "persist_loginitem"
    PERSIST_SYSTEMD      = "persist_systemd"
    PERSIST_PLIST        = "persist_plist"
    PERSIST_DLL_HIJACK   = "persist_dll_hijack"
    PERSIST_BOOTKIT      = "persist_bootkit"

    # ---------- EXFILTRATION & C2 ----------
    DNS_TUNNEL           = "dns_tunnel"
    ICMP_TUNNEL          = "icmp_tunnel"
    COOKIE_STEAL         = "cookie_steal"
    CLIPBOARD_GRAB       = "clipboard_grab"
    KEYLOGGER            = "keylogger"
    FILE_STEAL           = "file_steal"
    SCREENSHOT           = "screenshot"
    AUDIO_CAPTURE        = "audio_capture"
    VIDEO_CAPTURE        = "video_capture"
    CLOUD_UPLOAD         = "cloud_upload"
    TOR_TUNNEL           = "tor_tunnel"
    VPN_TUNNEL           = "vpn_tunnel"

    # ---------- INFRASTRUCTURE AUTOMATION ----------
    SPAWN_CONTAINER      = "spawn_container"
    SPAWN_VM             = "spawn_vm"
    CLOUD_DEPLOY         = "cloud_deploy"
    SERVERLESS_FUNCTION  = "serverless_function"
    SPAWN_BOTNET_NODE    = "spawn_botnet_node"
    SPAWN_TOR_HIDDEN     = "spawn_tor_hidden"
    SPAWN_VPS            = "spawn_vps"
    SPAWN_K8S_POD        = "spawn_k8s_pod"

    # ---------- DEFENSE & COUNTER-INTEL ----------
    DECEIVE_HONEYPOT     = "deceive_honeypot"
    LOG_TAMPER           = "log_tamper"
    MEMORY_SCRUB         = "memory_scrub"
    FILE_WIPE_SECURE     = "file_wipe_secure"
    REGISTRY_WIPE        = "registry_wipe"
    NETWORK_ISOLATION    = "network_isolation"
    PROCESS_KILL         = "process_kill"
    SERVICE_DISABLE      = "service_disable"

    # ---------- CLEANUP & DEFILE ------
    SECURE_DELETE        = "secure_delete"
    LOG_CLEANUP          = "log_cleanup"
    CACHE_CLEANUP        = "cache_cleanup"
    TEMP_CLEANUP         = "temp_cleanup"
    SELF_DELETE          = "self_delete"

    # ---------- REFLECTION & INTROSPECTION ------
    PEER_SCORING         = "peer_scoring"
    CONFLICT_DETECTION   = "conflict_detection"
    PRUNE_BELIEFS        = "prune_beliefs"
    MISSION_REFINEMENT   = "mission_refinement"
    SNAPSHOT_EMISSION    = "snapshot_emission"

    # ---------- SCOUT & SENSOR ------
    NETWORK_TOPOLOGY_MAP         = "network_topology_map"
    RISK_CONTOUR_ANALYSIS        = "risk_contour_analysis"
    BEHAVIORAL_ANOMALY_DETECTION = "behavioral_anomaly_detection"
    POSITION_RISK_EVALUATION     = "position_risk_evaluation"
    SERVICE_SCAN                 = "service_scan"
    THREAT_INTELLIGENCE_LOAD     = "threat_intelligence_load"
    COUNTERMEASURE_EVALUATION    = "countermeasure_evaluation"
    TRAP_LIKELIHOOD_CALCULATION  = "trap_likelihood_calculation"
    SIGNATURE_ANALYSIS           = "signature_analysis"
    ENCRYPTION_ENGINE            = "encryption_engine"
    COMPRESSION_ENGINE           = "compression_engine"
    INTELLIGENCE_TRANSMISSION    = "intelligence_transmission"
    PROCESS_MONITOR              = "process_monitor"
    ANOMALY_CLASSIFICATION       = "anomaly_classification"
    BELIEF_BUILDING              = "belief_building"

# ------------------------------------------------------------
#  MISSION / EVENT / AGENT CORE
# ------------------------------------------------------------
@dataclass
class Mission:
    name: str
    objectives: List[str]
    parameters: Dict[str, Any]
    mission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: MissionStatus = MissionStatus.PENDING

    def to_dict(self):
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "objectives": self.objectives,
            "parameters": self.parameters,
            "status": self.status.value,
        }

@dataclass
class Event:
    event_type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    sender_id: Optional[str] = None
    target_id: Optional[str] = None
    target_role: Optional[ElderRole] = None

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "target_role": self.target_role.value if self.target_role else None,
        }

@dataclass
class AgentCore:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed"
    status: AgentStatus = AgentStatus.ACTIVE
    capabilities: List[Capability] = field(default_factory=list)
    reputation: float = 0.0
    memory: deque = field(default_factory=lambda: deque(maxlen=1000))
    mission_stack: List[Mission] = field(default_factory=list)
    version: str = __version__
    last_heartbeat: float = field(default_factory=time.time)
    elder_role: Optional[ElderRole] = None
    identity_hash: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Public helpers -------------------------------------------------
    def record_event(self, ev: Event):
        """Append event and keep memory bounded."""
        self.memory.append(ev)

    def push_mission(self, mission: Mission):
        self.mission_stack.append(mission)

    def pop_mission(self) -> Optional[Mission]:
        return self.mission_stack.pop() if self.mission_stack else None

    def heartbeat(self):
        self.last_heartbeat = time.time()

# ------------------------------------------------------------
#  MESSAGE BUS (kernel-level, zero-config)
# ------------------------------------------------------------
class MessageBus:
    """Lightweight pub/sub for agents."""
    def __init__(self):
        self.agents: Dict[str, AgentCore] = {}

    def register(self, agent: AgentCore):
        self.agents[agent.id] = agent

    def send_event(self, ev: Event):
        if ev.target_id and ev.target_id in self.agents:
            self.agents[ev.target_id].record_event(ev)
        elif ev.target_role:
            for agent in self.agents.values():
                if agent.elder_role == ev.target_role:
                    agent.record_event(ev)
        else:
            for agent in self.agents.values():
                agent.record_event(ev)

    def query_agents(self, **filters) -> List[AgentCore]:
        """Return agents matching any key=value filter."""
        return [a for a in self.agents.values()
                if all(getattr(a, k, None) == v for k, v in filters.items())]

# ------------------------------------------------------------
#  SINGLETON BUS (kernel attaches this)
# ------------------------------------------------------------
bus = MessageBus()