"""
INTELLIGENT LATERAL MOVEMENT MANAGER: Unified Decision-Making for Lateral Spread
==================================================================================

Uses UnifiedCausalReasonerV2 to intelligently select lateral movement techniques based on:
- Current credentials available
- Network topology and segmentation
- Target system characteristics (OS, patches, role)
- Detection risk vs. reach trade-offs
- Operational phase and strategy

Lateral Movement Techniques:
- psexec: Command execution, requires admin credentials, observable
- wmiexec: WMI-based, less observable, good for C2
- ssh_key_lateral: SSH-based spread (Unix/Linux environments)
- credential_dumping_lateral: Dump then spray credentials
- unpatched_vuln_rce: Exploit unpatched systems

Target Selection Strategies:
- prioritize_servers: High-value targets, usually hardened
- prioritize_workstations: User machines, lower security, credential harvesting
- prioritize_dc_admins: Domain controller compromise (full domain takeover)

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from world_state import OperationalPhase
from unified_integration import create_integration

logger = logging.getLogger("IntelligentLateralMovementManager")


@dataclass
class LateralStrategy:
    """Strategy for lateral movement and propagation"""
    technique: str  # Movement technique (psexec, wmiexec, etc.)
    targeting_strategy: str  # Which systems to target
    confidence: float  # How confident in this approach
    reasoning: str
    estimated_success_rate: float
    estimated_detection_risk: float
    expected_reach: float  # Expected number of new systems


class IntelligentLateralMovementManager:
    """
    Intelligent lateral movement orchestration
    
    Coordinates:
    1. Which movement technique to use (psexec, wmiexec, ssh, etc.)
    2. Which systems to target (servers, workstations, DCs)
    3. Which credentials to use (if available)
    4. How aggressive to be (detection risk)
    
    Usage:
        manager = IntelligentLateralMovementManager()
        strategy = manager.choose_lateral_strategy(
            available_credentials=["admin", "service_account"],
            network_type="domain_enterprise",
            detection_pressure=0.6
        )
        
        if strategy.confidence > 0.7:
            results = manager.conduct_lateral_movement(strategy)
    """
    
    def __init__(self):
        self.integration = create_integration("intelligent_lateral_movement")
        
        # Lateral movement technique profiles
        self.techniques = {
            "psexec": {
                "credential_types": ["admin", "domain_admin"],
                "observable": True,
                "network_visibility": 0.70,
                "success_base": 0.80,
                "detection_risk": 0.70,
                "speed": 0.9,
                "requires_impacket": True,
            },
            "wmiexec": {
                "credential_types": ["admin", "domain_admin"],
                "observable": False,
                "network_visibility": 0.50,
                "success_base": 0.75,
                "detection_risk": 0.40,
                "speed": 0.8,
                "requires_impacket": True,
            },
            "ssh_key_lateral": {
                "credential_types": ["ssh_key"],
                "observable": False,
                "network_visibility": 0.30,
                "success_base": 0.85,
                "detection_risk": 0.30,
                "speed": 0.7,
                "os_target": "linux/unix",
            },
            "credential_dumping_lateral": {
                "credential_types": ["lsa", "ntlm"],
                "observable": True,
                "network_visibility": 0.80,
                "success_base": 0.70,
                "detection_risk": 0.75,
                "speed": 0.6,
                "reach_multiplier": 0.8,  # Can hit many targets
            },
            "unpatched_vuln_rce": {
                "credential_types": [],
                "observable": True,
                "network_visibility": 0.75,
                "success_base": 0.80,
                "detection_risk": 0.75,
                "speed": 0.7,
                "requires_exploit": "CVE",
            },
        }
        
        # Target selection strategies
        self.targeting_strategies = {
            "prioritize_servers": {
                "target_type": "servers",
                "data_value": 0.90,
                "lateral_reach": 0.85,
                "security_level": "high",
                "detection_risk_multiplier": 1.2,
            },
            "prioritize_workstations": {
                "target_type": "workstations",
                "user_data_value": 0.70,
                "credential_harvesting": 0.75,
                "ease_of_compromise": 0.80,
                "detection_risk_multiplier": 0.8,
            },
            "prioritize_dc_admins": {
                "target_type": "domain_controllers",
                "reach_multiplier": 0.99,  # Near-total domain compromise
                "detection_risk_multiplier": 1.5,
                "persistence_permanence": 0.95,
            },
        }
        
        logger.info("[IntelligentLateralMovementManager] Initialized")
    
    def choose_lateral_strategy(
        self,
        available_credentials: List[str],
        network_type: str = "domain_enterprise",
        detection_pressure: float = 0.5,
        time_constraint: float = 60.0,  # Minutes
        edr_in_network: bool = False,
    ) -> LateralStrategy:
        """
        Choose optimal lateral movement strategy
        
        Args:
            available_credentials: List of credential types available
            network_type: "workgroup", "small_domain", "domain_enterprise", "segmented"
            detection_pressure: 0-1, how much detection risk the network can withstand
            time_constraint: Minutes available for lateral movement
            edr_in_network: Is EDR deployed in the network?
        
        Returns:
            LateralStrategy with technique and targeting recommendation
        """
        
        # Filter applicable techniques based on credentials
        available_techniques = []
        for technique, profile in self.techniques.items():
            # Check if we have required credential types
            required = profile.get("credential_types", [])
            if required and not any(c in available_credentials for c in required):
                continue  # Skip if we don't have required credentials
            
            available_techniques.append(technique)
        
        # Choose targeting strategy based on network type
        targeting_options = list(self.targeting_strategies.keys())
        if network_type == "segmented":
            targeting_options = ["prioritize_workstations"]  # Can't easily reach servers
        elif network_type == "domain_enterprise":
            targeting_options = ["prioritize_servers", "prioritize_dc_admins"]
        
        # Build context
        context = {
            "available_credentials": len(available_credentials),
            "network_type": network_type,
            "detection_pressure": detection_pressure,
            "edr_in_network": edr_in_network,
            "time_constraint": min(time_constraint / 60.0, 1.0),
        }
        
        # Get technique decision
        technique_decision = self.integration.decide(
            phase=OperationalPhase.LATERAL_MOVEMENT,
            options=available_techniques,
            context=context
        )
        
        technique = technique_decision.recommended_option
        
        # Get targeting decision
        targeting_decision = self.integration.decide(
            phase=OperationalPhase.LATERAL_MOVEMENT,
            options=targeting_options,
            context=context
        )
        
        targeting = targeting_decision.recommended_option
        
        # Calculate metrics
        technique_profile = self.techniques[technique]
        targeting_profile = self.targeting_strategies[targeting]
        
        success_rate = technique_profile["success_base"] * 0.9  # Assume some failures
        detection_risk = technique_profile["detection_risk"]
        
        if edr_in_network:
            detection_risk = min(1.0, detection_risk * 1.4)
            success_rate *= 0.7  # EDR reduces effectiveness
        
        # Estimate reach
        reach = 0.5
        if "reach_multiplier" in targeting_profile:
            reach = targeting_profile["reach_multiplier"]
        elif targeting == "prioritize_dc_admins":
            reach = 0.95
        
        strategy = LateralStrategy(
            technique=technique,
            targeting_strategy=targeting,
            confidence=min(
                technique_decision.confidence,
                targeting_decision.confidence
            ),
            reasoning=f"Lateral: {technique} targeting {targeting}, "
                     f"network={network_type}, edr={edr_in_network}",
            estimated_success_rate=success_rate,
            estimated_detection_risk=detection_risk,
            expected_reach=reach,
        )
        
        logger.info(
            f"[IntelligentLateralMovementManager] Chose {technique} + {targeting} "
            f"(success={success_rate:.0%}, reach={reach:.0%})"
        )
        
        return strategy
    
    def conduct_lateral_movement(
        self,
        strategy: LateralStrategy,
        target_ips: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute lateral movement according to strategy
        
        Args:
            strategy: LateralStrategy from choose_lateral_strategy()
            target_ips: Specific IPs to target (optional)
        
        Returns:
            Dict with results {success, compromised_count, detection_risk}
        """
        
        logger.info(
            f"[IntelligentLateralMovementManager] Starting {strategy.technique} "
            f"→ {strategy.targeting_strategy}"
        )
        
        try:
            # Simulate lateral movement
            results = self._simulate_lateral_movement(
                technique=strategy.technique,
                targeting=strategy.targeting_strategy,
                target_ips=target_ips
            )
            
            success = results.get("success", False)
            
            # Record outcome
            self.integration.record_outcome(
                decision=self.integration.last_decision,
                success=success,
                error=None if success else "Lateral movement failed",
                metadata={
                    "technique": strategy.technique,
                    "targeting": strategy.targeting_strategy,
                    "compromised": results.get("compromised_count", 0),
                    "detection_risk": strategy.estimated_detection_risk,
                }
            )
            
            if success:
                logger.info(
                    f"[IntelligentLateralMovementManager] ✓ {strategy.technique} "
                    f"succeeded, compromised {results.get('compromised_count', 0)} systems"
                )
            else:
                logger.error(
                    f"[IntelligentLateralMovementManager] ✗ {strategy.technique} failed"
                )
            
            return results
        
        except Exception as e:
            logger.error(
                f"[IntelligentLateralMovementManager] Exception: {e}"
            )
            self.integration.record_outcome(
                decision=self.integration.last_decision,
                success=False,
                error=str(e)
            )
            return {"success": False, "compromised_count": 0}
    
    def _simulate_lateral_movement(
        self,
        technique: str,
        targeting: str,
        target_ips: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Simulate lateral movement execution"""
        
        import random
        
        technique_profile = self.techniques[technique]
        success_rate = technique_profile["success_base"]
        
        # Simulate success
        success = random.random() < success_rate
        
        if success and target_ips:
            # Estimate compromised count
            compromised = int(len(target_ips) * random.uniform(0.3, 0.9))
        else:
            compromised = 0
        
        return {
            "success": success,
            "technique": technique,
            "targeting": targeting,
            "compromised_count": compromised,
            "detection_risk": technique_profile["detection_risk"],
        }
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get lateral movement decision history"""
        return self.integration.get_decision_history(limit=10)


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s"
    )
    
    manager = IntelligentLateralMovementManager()
    
    print("=" * 70)
    print("INTELLIGENT LATERAL MOVEMENT STRATEGY SELECTION")
    print("=" * 70)
    
    # Scenario 1: Enterprise with admin creds, no EDR
    print("\n[Scenario 1] Enterprise network, admin creds, no EDR")
    print("-" * 70)
    strategy = manager.choose_lateral_strategy(
        available_credentials=["admin", "domain_admin"],
        network_type="domain_enterprise",
        detection_pressure=0.5,
        edr_in_network=False
    )
    print(f"Technique: {strategy.technique}")
    print(f"Targeting: {strategy.targeting_strategy}")
    print(f"Expected reach: {strategy.expected_reach:.0%}")
    print(f"Success rate: {strategy.estimated_success_rate:.0%}")
    print(f"Detection risk: {strategy.estimated_detection_risk:.0%}")
    
    # Scenario 2: Segmented network with limited creds
    print("\n[Scenario 2] Segmented network, limited creds, high detection pressure")
    print("-" * 70)
    strategy = manager.choose_lateral_strategy(
        available_credentials=["service_account"],
        network_type="segmented",
        detection_pressure=0.8,
        edr_in_network=True
    )
    print(f"Technique: {strategy.technique}")
    print(f"Targeting: {strategy.targeting_strategy}")
    print(f"Expected reach: {strategy.expected_reach:.0%}")
    print(f"Success rate: {strategy.estimated_success_rate:.0%}")
    print(f"Detection risk: {strategy.estimated_detection_risk:.0%}")
