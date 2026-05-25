"""
Operational Causal Graphs: Strategic Reasoning for All Phases
==============================================================

Causal reasoning for complete operation lifecycle:
- Initial Access (reconnaissance + infiltration + first execution)
- Persistence (which mechanisms to enable)
- Privilege Escalation (which techniques to use)
- Lateral Movement / Propagation (how to spread)
- C2 Communication (frequency + method)
- Credential Extraction (what to steal)

Each operational phase has strategic trade-offs captured in causal graphs.

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import Dict, List, Tuple
from enum import Enum

logger = logging.getLogger("OperationalCausalGraphs")


class OperationalPhase(Enum):
    """Operational lifecycle phases"""
    INITIAL_ACCESS = "initial_access"
    PERSISTENCE = "persistence"
    CREDENTIAL_EXTRACTION = "credential_extraction"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    C2_COMMUNICATION = "c2_communication"
    EXFILTRATION = "exfiltration"


class InitialAccessCausalGraph:
    """
    Initial compromise: reconnaissance + infiltration + first execution
    
    Causal relationships:
    - Recon intensity → detection risk vs intelligence quality
    - Infiltration method → success rate vs persistence quality
    - First execution location → visibility vs stability
    """
    
    @staticmethod
    def get_recon_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Reconnaissance trade-offs:
        Effect = (weight, confidence)
        """
        return {
            "active_recon": {
                "detection_risk": (0.85, 0.95),
                "information_quality": (0.90, 0.90),
                "time_to_exploit": (0.20, 0.85),  # Quick exploitation
            },
            "passive_recon": {
                "detection_risk": (0.15, 0.95),
                "information_quality": (0.50, 0.80),
                "time_to_exploit": (0.60, 0.85),  # Slower exploitation
            },
            "osint_recon": {
                "detection_risk": (0.05, 0.95),
                "information_quality": (0.40, 0.75),
                "social_engineering_readiness": (0.70, 0.85),
            },
        }
    
    @staticmethod
    def get_infiltration_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Infiltration method trade-offs
        """
        return {
            "phishing": {
                "success_rate": (0.40, 0.85),
                "user_suspicion": (0.60, 0.80),
                "payload_reliability": (0.75, 0.85),
            },
            "exploit_unpatched": {
                "success_rate": (0.85, 0.80),
                "detection_risk": (0.75, 0.85),
                "stable_persistence": (0.80, 0.85),
            },
            "supply_chain": {
                "success_rate": (0.70, 0.70),
                "detection_risk": (0.30, 0.70),
                "scope_multiplier": (0.90, 0.65),  # Multiple machines
            },
            "physical_implant": {
                "success_rate": (0.95, 0.85),
                "access_requirement": (0.95, 0.90),  # Needs physical access
                "persistence_permanence": (0.85, 0.90),
            },
        }
    
    @staticmethod
    def get_first_execution_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        First execution location strategy
        """
        return {
            "user_process": {
                "visibility_to_user": (0.60, 0.85),
                "immediate_detection": (0.40, 0.85),
                "stability": (0.50, 0.85),
            },
            "system_process": {
                "visibility_to_user": (0.10, 0.85),
                "privilege_requirement": (0.95, 0.90),
                "stability": (0.90, 0.85),
            },
            "browser_process": {
                "user_suspicion": (0.70, 0.85),
                "network_isolation_evasion": (0.60, 0.80),
                "keylogging_opportunity": (0.80, 0.85),
            },
        }


class PrivilegeEscalationCausalGraph:
    """
    Privilege escalation: which technique to use based on environment
    
    Causal relationships:
    - Technique type → detection risk vs success rate
    - Privilege gap → time to escalate vs persistence quality
    """
    
    @staticmethod
    def get_escalation_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Escalation technique trade-offs
        """
        return {
            "uac_bypass_token_duplication": {
                "success_rate": (0.85, 0.85),
                "detection_risk": (0.45, 0.85),
                "execution_time": (0.30, 0.85),  # Quick
            },
            "kernel_exploit": {
                "success_rate": (0.75, 0.75),
                "detection_risk": (0.80, 0.80),
                "system_stability_risk": (0.70, 0.80),
            },
            "scheduled_task_hijack": {
                "success_rate": (0.70, 0.85),
                "detection_risk": (0.40, 0.85),
                "persistence_duration": (0.80, 0.85),
            },
            "seimpersonate": {
                "success_rate": (0.80, 0.80),
                "detection_risk": (0.50, 0.80),
                "reliability_across_versions": (0.60, 0.75),
            },
            "service_abuse": {
                "success_rate": (0.90, 0.85),
                "detection_risk": (0.55, 0.85),
                "temporal_flexibility": (0.90, 0.85),  # Can be delayed
            },
        }


class LateralMovementCausalGraph:
    """
    Lateral movement / propagation strategy
    
    Causal relationships:
    - Technique type → success rate vs network visibility
    - Target selection → reach vs detection
    - Infection method → reliability vs traces
    """
    
    @staticmethod
    def get_technique_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Lateral movement technique trade-offs
        """
        return {
            "psexec": {
                "success_rate": (0.80, 0.85),
                "network_visibility": (0.70, 0.85),
                "admin_requirement": (0.90, 0.90),
            },
            "wmiexec": {
                "success_rate": (0.75, 0.85),
                "network_visibility": (0.50, 0.85),
                "detection_risk": (0.40, 0.85),
            },
            "ssh_key_lateral": {
                "success_rate": (0.85, 0.85),
                "network_visibility": (0.30, 0.85),
                "credential_requirement": (1.0, 0.90),
            },
            "credential_dumping_lateral": {
                "success_rate": (0.70, 0.80),
                "credential_freshness": (0.60, 0.80),
                "scale_multiplier": (0.80, 0.85),  # Can hit many targets
            },
            "unpatched_vuln_rce": {
                "success_rate": (0.80, 0.75),
                "detection_risk": (0.75, 0.80),
                "patch_status_variance": (0.85, 0.70),  # Variable across network
            },
        }
    
    @staticmethod
    def get_target_selection_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Target selection strategy
        """
        return {
            "prioritize_servers": {
                "data_value": (0.90, 0.85),
                "lateral_reach": (0.85, 0.85),
                "security_hardening": (0.85, 0.85),
                "monitoring_level": (0.80, 0.85),
            },
            "prioritize_workstations": {
                "user_data_value": (0.70, 0.85),
                "credential_harvesting": (0.75, 0.85),
                "ease_of_compromise": (0.80, 0.85),
                "visibility_to_user": (0.60, 0.85),
            },
            "prioritize_dc_admins": {
                "reach_multiplier": (0.99, 0.85),  # Domain compromise
                "detection_risk": (0.95, 0.85),
                "persistence_permanence": (0.95, 0.85),
            },
        }


class C2CausalGraph:
    """
    C2 communication strategy tailored per operational phase
    """
    
    @staticmethod
    def get_communication_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        C2 method trade-offs by operational phase
        """
        return {
            "direct_https": {
                "latency": (0.10, 0.85),  # Low latency
                "detection_risk": (0.80, 0.85),
                "firewall_evasion": (0.40, 0.85),
            },
            "dga_dns": {
                "resilience": (0.95, 0.85),
                "firewall_evasion": (0.70, 0.85),
                "callback_reliability": (0.50, 0.85),
            },
            "p2p_mesh": {
                "resilience": (0.90, 0.75),
                "operator_control": (0.50, 0.75),
                "network_discovery_time": (0.70, 0.75),
            },
            "steganography": {
                "detectability": (0.20, 0.70),
                "bandwidth": (0.10, 0.80),  # Very limited
                "sophistication": (0.95, 0.70),
            },
        }


class ExfiltrationCausalGraph:
    """
    Data exfiltration strategy based on data sensitivity and volume
    """
    
    @staticmethod
    def get_exfil_graph() -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Exfiltration method trade-offs
        """
        return {
            "bulk_https": {
                "speed": (0.90, 0.85),
                "detection_risk": (0.85, 0.85),
                "bandwidth_usage": (0.90, 0.85),
            },
            "dns_exfil": {
                "detection_risk": (0.40, 0.80),
                "bandwidth": (0.05, 0.85),  # Very slow
                "firewall_evasion": (0.85, 0.85),
            },
            "stealth_scheduled": {
                "detection_risk": (0.20, 0.85),
                "speed": (0.10, 0.85),
                "reliability": (0.80, 0.85),
            },
            "encrypted_cloud_upload": {
                "detectability": (0.35, 0.80),
                "plausible_deniability": (0.60, 0.80),
                "scale": (0.90, 0.85),
            },
        }


def build_all_operational_graphs() -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    Build all operational causal graphs
    
    Returns:
        Dictionary mapping operational phases to graphs
    """
    graphs = {
        "initial_access_recon": InitialAccessCausalGraph.get_recon_graph(),
        "initial_access_infiltration": InitialAccessCausalGraph.get_infiltration_graph(),
        "initial_access_execution": InitialAccessCausalGraph.get_first_execution_graph(),
        "privilege_escalation": PrivilegeEscalationCausalGraph.get_escalation_graph(),
        "lateral_movement_techniques": LateralMovementCausalGraph.get_technique_graph(),
        "lateral_movement_targeting": LateralMovementCausalGraph.get_target_selection_graph(),
        "c2_communication": C2CausalGraph.get_communication_graph(),
        "exfiltration": ExfiltrationCausalGraph.get_exfil_graph(),
    }
    
    logger.info(f"[OperationalCausalGraphs] Built {len(graphs)} graphs")
    return graphs


def build_causal_graph_from_definitions(
    graph_name: str,
    definitions: Dict[str, Dict[str, Tuple[float, float]]]
):
    """
    Build a CausalGraph from effect definitions
    
    Args:
        graph_name: Name of the graph (e.g., "privilege_escalation")
        definitions: Dict mapping interventions to {effect: (weight, confidence)}
    
    Returns:
        Constructed CausalGraph ready for simulation
    """
    from causal_graph import CausalGraph
    
    graph = CausalGraph(name=graph_name)
    
    # Add edges from definitions
    # (intervention) → {effect: (weight, confidence)}
    for intervention, effects in definitions.items():
        for effect, (weight, confidence) in effects.items():
            # Determine direction (most effects are positive impacts)
            direction = 1
            if "risk" in effect.lower() or "detection" in effect.lower():
                direction = 1  # Risk increases
            elif "evasion" in effect.lower() or "stealth" in effect.lower():
                direction = -1  # Reduces detection
            
            graph.add_edge(
                cause=intervention,
                effect=effect,
                weight=weight,
                direction=direction,
                confidence=confidence,
                tags=[graph_name, "operational"]
            )
    
    logger.debug(f"[OperationalCausalGraphs] Built graph: {graph_name} with {len(graph.all_causes)} interventions")
    return graph


def build_phase_simulators():
    """
    Build CausalSimulator objects for each operational phase
    
    Returns:
        Dictionary mapping phase names to CausalSimulator instances
    """
    from causal_simulator import CausalSimulator
    from world_state import OperationalPhase
    
    # Get all graph definitions
    graph_defs = build_all_operational_graphs()
    
    # Build simulators for each phase
    simulators = {}
    
    # Infiltration phase uses recon + infiltration graphs
    infiltration_graph = build_causal_graph_from_definitions(
        "infiltration",
        {
            **InitialAccessCausalGraph.get_recon_graph(),
            **InitialAccessCausalGraph.get_infiltration_graph(),
        }
    )
    simulators[OperationalPhase.INFILTRATION] = CausalSimulator(infiltration_graph)
    
    # Persistence phase
    persistence_graph = build_causal_graph_from_definitions(
        "persistence",
        InitialAccessCausalGraph.get_first_execution_graph()
    )
    simulators[OperationalPhase.PERSISTENCE] = CausalSimulator(persistence_graph)
    
    # Privilege escalation phase
    escalation_graph = build_causal_graph_from_definitions(
        "escalation",
        PrivilegeEscalationCausalGraph.get_escalation_graph()
    )
    simulators[OperationalPhase.PRIVILEGE_ESCALATION] = CausalSimulator(escalation_graph)
    
    # Lateral movement phase
    lateral_graphs = {
        **LateralMovementCausalGraph.get_technique_graph(),
        **LateralMovementCausalGraph.get_target_selection_graph(),
    }
    lateral_graph = build_causal_graph_from_definitions("lateral_movement", lateral_graphs)
    simulators[OperationalPhase.LATERAL_MOVEMENT] = CausalSimulator(lateral_graph)
    
    # Data exfiltration phase (using same credential extraction + exfil techniques)
    credential_exfil_graph = build_causal_graph_from_definitions(
        "data_exfiltration",
        {
            "extract_lsa": {
                "detection_risk": (0.85, 0.95),
                "password_quality": (0.90, 0.90),
                "lateral_utility": (0.80, 0.95),
            },
            "extract_chrome": {
                "detection_risk": (0.50, 0.85),
                "web_credential_coverage": (0.80, 0.85),
                "persistence": (0.60, 0.80),
            },
            "extract_windows_vault": {
                "detection_risk": (0.60, 0.85),
                "credential_diversity": (0.70, 0.85),
                "lateral_utility": (0.75, 0.90),
            },
            "keylog": {
                "detection_risk": (0.70, 0.85),
                "coverage": (0.90, 0.85),
                "typing_detection": (0.65, 0.80),
            },
            **ExfiltrationCausalGraph.get_exfil_graph(),
        }
    )
    simulators[OperationalPhase.DATA_EXFILTRATION] = CausalSimulator(credential_exfil_graph)
    
    # C2 communication (stored with string key for special access)
    c2_graph = build_causal_graph_from_definitions(
        "c2_communication",
        C2CausalGraph.get_communication_graph()
    )
    simulators["c2_communication"] = CausalSimulator(c2_graph)
    
    logger.info(f"[OperationalCausalGraphs] Built {len(simulators)} phase simulators")
    return simulators


if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    graphs = build_all_operational_graphs()
    
    print("\n" + "="*70)
    print("OPERATIONAL CAUSAL GRAPHS SUMMARY")
    print("="*70)
    
    for phase, graph in graphs.items():
        print(f"\n→ {phase.upper()}:")
        for intervention, effects in graph.items():
            print(f"  • {intervention}:")
            for effect, (weight, conf) in list(effects.items())[:2]:
                print(f"    - {effect}: weight={weight:.2f}, confidence={conf:.1%}")
