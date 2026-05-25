"""
Operational Causal Engine: Unified Strategic Reasoning
=======================================================

High-level coordinator that uses causal reasoning to decide:
- Which reconnaissance technique to use
- Which infiltration method to attempt
- Which privilege escalation to try
- Which machines to target for lateral movement
- How to exfiltrate data
- C2 communication strategy per phase

Integrates all domain-specific causal graphs into unified strategic engine.

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

from causal_graph import CausalGraph
from causal_simulator import CausalSimulator, ThreatEnvironmentType
from operational_causal_graphs import (
    build_all_operational_graphs,
    OperationalPhase,
    InitialAccessCausalGraph,
    PrivilegeEscalationCausalGraph,
    LateralMovementCausalGraph,
    C2CausalGraph,
    ExfiltrationCausalGraph,
)

logger = logging.getLogger("OperationalCausalEngine")


class OperationalCausalEngine:
    """
    Unified strategic decision engine for entire operation lifecycle
    
    Usage:
        engine = OperationalCausalEngine()
        
        # Initial access decisions
        recon_strategy = engine.choose_initial_reconnaissance(threat_level=30)
        infiltration = engine.choose_infiltration_method(threat_level=30, target_os="windows")
        
        # Mid-operation decisions
        escalation = engine.choose_escalation_technique(
            current_privilege="user",
            target_privilege="system",
            threat_level=50
        )
        
        lateral_targets = engine.choose_lateral_targets(
            infected_machines=["DESKTOP-1", "DESKTOP-2"],
            threat_level=50,
            objective="domain_takeover"
        )
        
        # Exfiltration decisions
        exfil_strategy = engine.choose_exfiltration_method(
            data_volume_mb=500,
            threat_level=75,
            time_pressure="high"
        )
    """
    
    def __init__(self):
        """Initialize with all operational causal graphs"""
        self.graphs_data = build_all_operational_graphs()
        
        # Build CausalGraph objects for each operational domain
        self.initial_recon_graph = self._build_causal_graph(
            "initial_access_recon",
            self.graphs_data["initial_access_recon"]
        )
        self.initial_infiltration_graph = self._build_causal_graph(
            "initial_access_infiltration",
            self.graphs_data["initial_access_infiltration"]
        )
        self.escalation_graph = self._build_causal_graph(
            "privilege_escalation",
            self.graphs_data["privilege_escalation"]
        )
        self.lateral_movement_graph = self._build_causal_graph(
            "lateral_movement_techniques",
            self.graphs_data["lateral_movement_techniques"]
        )
        self.c2_graph = self._build_causal_graph(
            "c2_communication",
            self.graphs_data["c2_communication"]
        )
        self.exfiltration_graph = self._build_causal_graph(
            "exfiltration",
            self.graphs_data["exfiltration"]
        )
        
        logger.info("[OperationalCausalEngine] Initialized with all domain graphs")
    
    def choose_initial_reconnaissance(
        self,
        threat_level: float = 50.0,
        time_available_days: float = 7.0,
        target_criticality: str = "medium"
    ) -> Dict[str, any]:
        """
        Choose reconnaissance strategy for initial phase
        
        Args:
            threat_level: 0-100
            time_available_days: Days before exploitation window closes
            target_criticality: "low", "medium", "high"
        
        Returns:
            Strategy dict with chosen technique and reasoning
        """
        
        # Map threat to environment type
        if threat_level > 75:
            threat_env = ThreatEnvironmentType.CRITICAL
        elif threat_level > 50:
            threat_env = ThreatEnvironmentType.HIGH
        elif threat_level > 25:
            threat_env = ThreatEnvironmentType.MEDIUM
        else:
            threat_env = ThreatEnvironmentType.LOW
        
        options = ["active_recon", "passive_recon", "osint_recon"]
        
        sim = CausalSimulator(self.initial_recon_graph)
        ranked = sim.rank_interventions(
            options,
            threat_environment=threat_env,
            current_threat_score=threat_level
        )
        
        chosen = ranked[0].intervention
        
        # Adjust based on time pressure
        if time_available_days < 1:
            chosen = "active_recon"  # Must be fast
        elif time_available_days > 30:
            chosen = "osint_recon"  # Can afford to be stealthy
        
        strategy = {
            "technique": chosen,
            "reasoning": f"Threat={threat_level:.0f}, Time={time_available_days}d → {chosen}",
            "risk_profile": ranked[0].risk_score,
            "value_profile": ranked[0].value_score,
            "estimated_time_days": self._estimate_recon_time(chosen),
        }
        
        logger.info(f"[OperationalCausalEngine] Reconnaissance: {chosen}")
        return strategy
    
    def choose_infiltration_method(
        self,
        threat_level: float = 50.0,
        target_os: str = "windows",
        user_sophistication: str = "low",
        network_isolation: str = "internet_connected"
    ) -> Dict[str, any]:
        """
        Choose infiltration method based on target environment
        
        Args:
            threat_level: 0-100
            target_os: "windows", "linux", "macos"
            user_sophistication: "low", "medium", "high"
            network_isolation: "internet_connected", "airgapped", "vpn_only"
        
        Returns:
            Strategy dict with chosen infiltration technique
        """
        
        threat_env = ThreatEnvironmentType.CRITICAL if threat_level > 75 else (
            ThreatEnvironmentType.HIGH if threat_level > 50 else (
                ThreatEnvironmentType.MEDIUM if threat_level > 25 else ThreatEnvironmentType.LOW
            )
        )
        
        # Some options only work on certain platforms
        all_options = ["phishing", "exploit_unpatched", "supply_chain", "physical_implant"]
        
        # Filter based on constraints
        available_options = all_options.copy()
        if network_isolation == "airgapped":
            available_options = ["supply_chain", "physical_implant"]
        elif network_isolation == "vpn_only":
            available_options = ["exploit_unpatched", "supply_chain"]
        
        if user_sophistication == "high":
            available_options.remove("phishing") if "phishing" in available_options else None
        
        sim = CausalSimulator(self.initial_infiltration_graph)
        ranked = sim.rank_interventions(
            available_options,
            threat_environment=threat_env,
            current_threat_score=threat_level
        )
        
        chosen = ranked[0].intervention
        
        strategy = {
            "method": chosen,
            "target_os": target_os,
            "reasoning": f"Network={network_isolation}, User={user_sophistication}, Threat={threat_level:.0f} → {chosen}",
            "success_probability": ranked[0].survival_score,
            "detection_risk": ranked[0].risk_score,
        }
        
        logger.info(f"[OperationalCausalEngine] Infiltration: {chosen}")
        return strategy
    
    def choose_escalation_technique(
        self,
        current_privilege: str = "user",
        target_privilege: str = "system",
        threat_level: float = 50.0,
        os_version: str = "windows_10",
        editable_processes: List[str] = None
    ) -> Dict[str, any]:
        """
        Choose privilege escalation technique
        
        Args:
            current_privilege: "user", "admin", "system"
            target_privilege: Desired privilege level
            threat_level: 0-100
            os_version: "windows_7", "windows_10", "windows_11", "linux", etc.
            editable_processes: Processes we can write to (for service abuse, etc.)
        
        Returns:
            Escalation strategy with chosen technique
        """
        
        # Only escalate if needed
        privilege_levels = {"user": 1, "admin": 2, "system": 3}
        if privilege_levels.get(current_privilege, 0) >= privilege_levels.get(target_privilege, 0):
            return {"status": "already_privileged", "current": current_privilege}
        
        threat_env = ThreatEnvironmentType.CRITICAL if threat_level > 75 else (
            ThreatEnvironmentType.HIGH if threat_level > 50 else (
                ThreatEnvironmentType.MEDIUM if threat_level > 25 else ThreatEnvironmentType.LOW
            )
        )
        
        all_techniques = [
            "uac_bypass_token_duplication",
            "kernel_exploit",
            "scheduled_task_hijack",
            "seimpersonate",
            "service_abuse"
        ]
        
        # Filter based on OS and available resources
        available = all_techniques.copy()
        
        if "linux" in os_version:
            available = ["kernel_exploit", "seimpersonate"]
        
        if not editable_processes:
            available.remove("service_abuse") if "service_abuse" in available else None
        
        sim = CausalSimulator(self.escalation_graph)
        ranked = sim.rank_interventions(
            available,
            threat_environment=threat_env,
            current_threat_score=threat_level
        )
        
        chosen = ranked[0].intervention
        
        strategy = {
            "technique": chosen,
            "from_privilege": current_privilege,
            "to_privilege": target_privilege,
            "reasoning": f"OS={os_version}, Threat={threat_level:.0f} → {chosen}",
            "success_probability": ranked[0].survival_score,
            "detection_risk": ranked[0].risk_score,
            "expected_latency_ms": self._estimate_escalation_time(chosen),
        }
        
        logger.info(f"[OperationalCausalEngine] Escalation: {chosen}")
        return strategy
    
    def choose_lateral_targets(
        self,
        infected_hosts: List[str],
        threat_level: float = 50.0,
        objective: str = "domain_takeover",
        network_topology: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Choose which machines to target for lateral movement
        
        Args:
            infected_hosts: List of already compromised machines
            threat_level: 0-100
            objective: "domain_takeover", "data_exfil", "persistence", "chaos"
            network_topology: Optional network graph
        
        Returns:
            Target selection strategy
        """
        
        threat_env = ThreatEnvironmentType.CRITICAL if threat_level > 75 else (
            ThreatEnvironmentType.HIGH if threat_level > 50 else (
                ThreatEnvironmentType.MEDIUM if threat_level > 25 else ThreatEnvironmentType.LOW
            )
        )
        
        # Choose targeting strategy based on objective
        strategies = {
            "domain_takeover": "prioritize_dc_admins",
            "data_exfil": "prioritize_servers",
            "persistence": "prioritize_servers",
            "chaos": "prioritize_dc_admins",
        }
        
        chosen_strategy = strategies.get(objective, "prioritize_servers")
        
        result = {
            "strategy": chosen_strategy,
            "objective": objective,
            "reasoning": f"Objective={objective}, Threat={threat_level:.0f} → {chosen_strategy}",
            "high_value_targets": self._identify_high_value_targets(chosen_strategy),
            "expected_reach": self._estimate_lateral_reach(chosen_strategy),
        }
        
        logger.info(f"[OperationalCausalEngine] Lateral movement targeting: {chosen_strategy}")
        return result
    
    def choose_lateral_technique(
        self,
        target_os: str = "windows",
        target_machine: str = "",
        have_credentials: bool = False,
        threat_level: float = 50.0
    ) -> Dict[str, any]:
        """
        Choose lateral movement technique
        
        Args:
            target_os: "windows", "linux"
            target_machine: DNS name or IP
            have_credentials: Whether we have valid credentials
            threat_level: 0-100
        
        Returns:
            Technique selection strategy
        """
        
        threat_env = ThreatEnvironmentType.CRITICAL if threat_level > 75 else (
            ThreatEnvironmentType.HIGH if threat_level > 50 else (
                ThreatEnvironmentType.MEDIUM if threat_level > 25 else ThreatEnvironmentType.LOW
            )
        )
        
        all_techniques = [
            "psexec",
            "wmiexec",
            "ssh_key_lateral",
            "credential_dumping_lateral",
            "unpatched_vuln_rce"
        ]
        
        # Filter based on OS
        if target_os == "linux":
            available = ["ssh_key_lateral", "credential_dumping_lateral", "unpatched_vuln_rce"]
        else:
            available = all_techniques
        
        # Can't use credential methods without credentials
        if not have_credentials and "credential" in str(available):
            # Keep general options
            pass
        
        sim = CausalSimulator(self.lateral_movement_graph)
        ranked = sim.rank_interventions(
            available,
            threat_environment=threat_env,
            current_threat_score=threat_level
        )
        
        chosen = ranked[0].intervention
        
        strategy = {
            "technique": chosen,
            "target": target_machine,
            "os": target_os,
            "reasoning": f"OS={target_os}, Creds={have_credentials}, Threat={threat_level:.0f} → {chosen}",
            "success_probability": ranked[0].survival_score,
            "detection_risk": ranked[0].risk_score,
        }
        
        logger.info(f"[OperationalCausalEngine] Lateral technique: {chosen}")
        return strategy
    
    def choose_c2_method_for_phase(
        self,
        operational_phase: OperationalPhase,
        threat_level: float = 50.0,
        primary_online: bool = True
    ) -> Dict[str, any]:
        """
        Choose C2 method appropriate for operational phase
        
        Args:
            operational_phase: Current operation phase
            threat_level: 0-100
            primary_online: Whether primary C2 is online
        
        Returns:
            C2 communication strategy
        """
        
        threat_env = ThreatEnvironmentType.CRITICAL if threat_level > 75 else (
            ThreatEnvironmentType.HIGH if threat_level > 50 else (
                ThreatEnvironmentType.MEDIUM if threat_level > 25 else ThreatEnvironmentType.LOW
            )
        )
        
        all_methods = ["direct_https", "dga_dns", "p2p_mesh", "steganography"]
        
        sim = CausalSimulator(self.c2_graph)
        ranked = sim.rank_interventions(
            all_methods,
            threat_environment=threat_env,
            current_threat_score=threat_level
        )
        
        chosen = ranked[0].intervention
        
        # Adjust based on phase
        if operational_phase == OperationalPhase.INITIAL_ACCESS:
            chosen = "direct_https"  # Speed matters in initial access
        elif operational_phase == OperationalPhase.LATERAL_MOVEMENT:
            pass  # Use default (likely stealth)
        elif operational_phase == OperationalPhase.EXFILTRATION:
            if threat_level < 50:
                chosen = "direct_https"  # Max bandwidth
        
        strategy = {
            "method": chosen,
            "phase": operational_phase.value,
            "reasoning": f"Phase={operational_phase.value}, Threat={threat_level:.0f} → {chosen}",
            "resilience": ranked[0].survival_score,
            "bandwidth": self._estimate_c2_bandwidth(chosen),
            "latency_ms": self._estimate_c2_latency(chosen),
        }
        
        logger.info(f"[OperationalCausalEngine] C2 for {operational_phase.value}: {chosen}")
        return strategy
    
    def choose_exfiltration_method(
        self,
        data_volume_mb: float,
        threat_level: float = 50.0,
        time_pressure: str = "low",
        sensitivity: str = "medium"
    ) -> Dict[str, any]:
        """
        Choose data exfiltration method
        
        Args:
            data_volume_mb: Amount of data to steal
            threat_level: 0-100
            time_pressure: "low", "medium", "high"
            sensitivity: "low", "medium", "high"
        
        Returns:
            Exfiltration strategy
        """
        
        threat_env = ThreatEnvironmentType.CRITICAL if threat_level > 75 else (
            ThreatEnvironmentType.HIGH if threat_level > 50 else (
                ThreatEnvironmentType.MEDIUM if threat_level > 25 else ThreatEnvironmentType.LOW
            )
        )
        
        all_methods = ["bulk_https", "dns_exfil", "stealth_scheduled", "encrypted_cloud_upload"]
        
        sim = CausalSimulator(self.exfiltration_graph)
        ranked = sim.rank_interventions(
            all_methods,
            threat_environment=threat_env,
            current_threat_score=threat_level
        )
        
        chosen = ranked[0].intervention
        
        # Override based on constraints
        if time_pressure == "high" and data_volume_mb > 100:
            chosen = "bulk_https"  # Must be fast
        elif data_volume_mb < 10:
            chosen = "dns_exfil"  # Small data, use stealth
        elif sensitivity == "high":
            chosen = "encrypted_cloud_upload"  # Maximum deniability
        
        strategy = {
            "method": chosen,
            "data_volume_mb": data_volume_mb,
            "threat_level": threat_level,
            "reasoning": f"Volume={data_volume_mb}MB, Time={time_pressure}, Sensitivity={sensitivity} → {chosen}",
            "estimated_time_seconds": self._estimate_exfil_time(chosen, data_volume_mb),
            "detection_probability": ranked[0].risk_score,
        }
        
        logger.info(f"[OperationalCausalEngine] Exfiltration: {chosen}")
        return strategy
    
    # ========== Private Helpers ==========
    
    def _build_causal_graph(self, name: str, edges_data: Dict) -> CausalGraph:
        """Convert edge data to CausalGraph"""
        graph = CausalGraph(name=name)
        for cause, effects in edges_data.items():
            for effect, (weight, confidence) in effects.items():
                graph.add_edge(cause, effect, weight=weight, confidence=confidence)
        return graph
    
    def _estimate_recon_time(self, technique: str) -> float:
        """Estimate days needed for reconnaissance"""
        times = {
            "active_recon": 1,
            "passive_recon": 3,
            "osint_recon": 7,
        }
        return times.get(technique, 5)
    
    def _estimate_escalation_time(self, technique: str) -> int:
        """Estimate milliseconds for escalation"""
        times = {
            "uac_bypass_token_duplication": 100,
            "kernel_exploit": 500,
            "scheduled_task_hijack": 2000,
            "seimpersonate": 200,
            "service_abuse": 3000,
        }
        return times.get(technique, 1000)
    
    def _estimate_lateral_reach(self, strategy: str) -> float:
        """Estimate percentage of network reachable"""
        reach = {
            "prioritize_servers": 0.70,
            "prioritize_workstations": 0.50,
            "prioritize_dc_admins": 0.95,
        }
        return reach.get(strategy, 0.50)
    
    def _identify_high_value_targets(self, strategy: str) -> List[str]:
        """Identify high-value targets by strategy"""
        if strategy == "prioritize_dc_admins":
            return ["DC01", "DC02", "ADMIN-MAIN"]
        elif strategy == "prioritize_servers":
            return ["FILE-SERVER", "EXCHANGE-01", "SQL-MAIN"]
        else:
            return ["WORKSTATION-*"]
    
    def _estimate_c2_bandwidth(self, method: str) -> str:
        """Estimated bandwidth for C2 method"""
        bandwidth = {
            "direct_https": "10 Mbps",
            "dga_dns": "100 Kbps",
            "p2p_mesh": "1-5 Mbps",
            "steganography": "10 Kbps",
        }
        return bandwidth.get(method, "unknown")
    
    def _estimate_c2_latency(self, method: str) -> int:
        """Estimated one-way latency in milliseconds"""
        latency = {
            "direct_https": 50,
            "dga_dns": 500,
            "p2p_mesh": 200,
            "steganography": 2000,
        }
        return latency.get(method, 1000)
    
    def _estimate_exfil_time(self, method: str, data_mb: float) -> float:
        """Estimate exfiltration time in seconds"""
        rates = {
            "bulk_https": 50,  # MB/sec
            "dns_exfil": 0.1,
            "stealth_scheduled": 0.5,
            "encrypted_cloud_upload": 30,
        }
        rate = rates.get(method, 10)
        return (data_mb / rate) if rate > 0 else 999999


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = OperationalCausalEngine()
    
    print("\n" + "="*70)
    print("OPERATIONAL CAUSAL ENGINE DEMONSTRATION")
    print("="*70)
    
    # Initial access
    print("\n→ INITIAL RECONNAISSANCE")
    recon = engine.choose_initial_reconnaissance(threat_level=30, time_available_days=7)
    print(f"  Technique: {recon['technique']}")
    print(f"  Risk: {recon['risk_profile']:.1%}")
    
    # Infiltration
    print("\n→ INFILTRATION")
    infil = engine.choose_infiltration_method(threat_level=30, target_os="windows")
    print(f"  Method: {infil['method']}")
    print(f"  Success: {infil['success_probability']:.1%}")
    
    # Escalation
    print("\n→ PRIVILEGE ESCALATION")
    esc = engine.choose_escalation_technique(
        current_privilege="user",
        target_privilege="system",
        threat_level=50
    )
    print(f"  Technique: {esc['technique']}")
    print(f"  Detection Risk: {esc['detection_risk']:.1%}")
    
    # Lateral movement
    print("\n→ LATERAL MOVEMENT")
    lateral = engine.choose_lateral_technique(
        target_os="windows",
        have_credentials=True,
        threat_level=50
    )
    print(f"  Technique: {lateral['technique']}")
    print(f"  Success: {lateral['success_probability']:.1%}")
    
    # C2 communication
    print("\n→ C2 COMMUNICATION")
    c2 = engine.choose_c2_method_for_phase(
        operational_phase=OperationalPhase.LATERAL_MOVEMENT,
        threat_level=50
    )
    print(f"  Method: {c2['method']}")
    print(f"  Bandwidth: {c2['bandwidth']}")
    
    # Exfiltration
    print("\n→ EXFILTRATION")
    exfil = engine.choose_exfiltration_method(
        data_volume_mb=500,
        threat_level=75,
        time_pressure="high"
    )
    print(f"  Method: {exfil['method']}")
    print(f"  Time: {exfil['estimated_time_seconds']:.1f}s")
