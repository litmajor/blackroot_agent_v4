#!/usr/bin/env python3
"""
Agent Handoff Protocol - Typed, Versioned Inter-Agent Communication

Purpose:
- Define standardized handoff results between agents
- Validate version compatibility
- Include confidence/uncertainty in results
- Enable dependency tracking and conditional execution
"""

import json
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HandoffProtocolVersion(Enum):
    """Handoff protocol versions for compatibility checking"""
    V1_0 = "1.0"  # Scan → Escalate → Extract → Exfil
    V1_1 = "1.1"  # Added conditional branches
    CURRENT = "1.0"


@dataclass
class ScanHandoff:
    """SCAN stage result for ESCALATE stage"""
    target: str
    discovered_hosts: List[str]
    open_ports: Dict[str, List[int]]           # {ip: [ports]}
    services: Dict[str, Dict[int, str]]        # {ip: {port: service}}
    os_guesses: Dict[str, tuple]               # {ip: (os, confidence)}
    zombie_candidates: List[str]               # IPs with high compromise probability
    confidence: float
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'ScanHandoff':
        return ScanHandoff(**{k: v for k, v in data.items()})


@dataclass
class EscalateHandoff:
    """ESCALATE stage result for EXTRACT stage"""
    target: str
    escalation_method: str                     # "GodPotato", "UAC_bypass", "service_hijack"
    success: bool
    privilege_level: str                       # "User", "Admin", "SYSTEM"
    tokens: Dict[str, Any]                     # {token_type: token_data}
    persistence_installed: bool
    confidence: float
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'EscalateHandoff':
        return EscalateHandoff(**{k: v for k, v in data.items()})


@dataclass
class ExtractHandoff:
    """EXTRACT stage result for EXFIL stage"""
    target: str
    credentials: Dict[str, List[str]]          # {type: [credentials]}
    sensitive_files: List[str]                 # Paths to files
    database_creds: Dict[str, str]             # {db_type: creds}
    private_keys: List[str]                    # SSH/TLS keys
    encryption_key: str                        # For encrypted exfil
    total_size_bytes: int
    confidence: float
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'ExtractHandoff':
        return ExtractHandoff(**{k: v for k, v in data.items()})


@dataclass
class ExfilHandoff:
    """EXFIL stage result (final)"""
    target: str
    exfil_method: str                          # "HTTP", "DNS", "ICMP", "SMB"
    exfiltrated_bytes: int
    chunks_sent: int
    status: str                                # "completed", "partial", "queued"
    c2_callback: Optional[str]                 # C2 URL for verification
    confidence: float
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'ExfilHandoff':
        return ExfilHandoff(**{k: v for k, v in data.items()})


class HandoffValidator:
    """
    Validate handoffs for correctness and compatibility.
    """
    
    # Schema definitions for each handoff type
    SCAN_SCHEMA = {
        'target': str,
        'discovered_hosts': list,
        'open_ports': dict,
        'services': dict,
        'os_guesses': dict,
        'zombie_candidates': list,
        'confidence': (float, int),
        'version': str
    }
    
    ESCALATE_SCHEMA = {
        'target': str,
        'escalation_method': str,
        'success': bool,
        'privilege_level': str,
        'tokens': dict,
        'persistence_installed': bool,
        'confidence': (float, int),
        'version': str
    }
    
    EXTRACT_SCHEMA = {
        'target': str,
        'credentials': dict,
        'sensitive_files': list,
        'database_creds': dict,
        'private_keys': list,
        'encryption_key': str,
        'total_size_bytes': int,
        'confidence': (float, int),
        'version': str
    }
    
    EXFIL_SCHEMA = {
        'target': str,
        'exfil_method': str,
        'exfiltrated_bytes': int,
        'chunks_sent': int,
        'status': str,
        'c2_callback': (str, type(None)),
        'confidence': (float, int),
        'version': str
    }
    
    @staticmethod
    def validate_scan(data: Dict) -> tuple[bool, Optional[str]]:
        """Validate SCAN handoff"""
        for field, expected_type in HandoffValidator.SCAN_SCHEMA.items():
            if field not in data:
                return False, f"Missing field: {field}"
            if not isinstance(data[field], expected_type):
                return False, f"Field {field}: expected {expected_type}, got {type(data[field])}"
        return True, None
    
    @staticmethod
    def validate_escalate(data: Dict) -> tuple[bool, Optional[str]]:
        """Validate ESCALATE handoff"""
        for field, expected_type in HandoffValidator.ESCALATE_SCHEMA.items():
            if field not in data:
                return False, f"Missing field: {field}"
            if not isinstance(data[field], expected_type):
                return False, f"Field {field}: expected {expected_type}, got {type(data[field])}"
        
        # Non-escalation (failure) requires reason
        if not data['success'] and not data.get('tokens'):
            return False, "Failed escalation must include error details in tokens"
        
        return True, None
    
    @staticmethod
    def validate_extract(data: Dict) -> tuple[bool, Optional[str]]:
        """Validate EXTRACT handoff"""
        for field, expected_type in HandoffValidator.EXTRACT_SCHEMA.items():
            if field not in data:
                return False, f"Missing field: {field}"
            if not isinstance(data[field], expected_type):
                return False, f"Field {field}: expected {expected_type}, got {type(data[field])}"
        
        # At least one extraction method must have results
        has_results = (
            bool(data.get('credentials')) or
            bool(data.get('sensitive_files')) or
            bool(data.get('database_creds')) or
            bool(data.get('private_keys'))
        )
        if not has_results:
            return False, "No credentials or files extracted"
        
        return True, None
    
    @staticmethod
    def validate_exfil(data: Dict) -> tuple[bool, Optional[str]]:
        """Validate EXFIL handoff"""
        for field, expected_type in HandoffValidator.EXFIL_SCHEMA.items():
            if field not in data:
                return False, f"Missing field: {field}"
            if not isinstance(data[field], expected_type):
                return False, f"Field {field}: expected {expected_type}, got {type(data[field])}"
        
        return True, None


class HandoffChain:
    """
    Manage chain of handoffs through pipeline.
    
    Usage:
        chain = HandoffChain()
        chain.add_scan_result(scan_data)
        chain.add_escalate_result(escalate_data)
        chain.add_extract_result(extract_data)
        chain.add_exfil_result(exfil_data)
        final_results = chain.get_all()
    """
    
    def __init__(self):
        self.scan: Optional[ScanHandoff] = None
        self.escalate: Optional[EscalateHandoff] = None
        self.extract: Optional[ExtractHandoff] = None
        self.exfil: Optional[ExfilHandoff] = None
        self.timeline: List[Dict] = []
    
    def add_scan_result(self, data: Dict) -> bool:
        """Add SCAN result with validation"""
        valid, error = HandoffValidator.validate_scan(data)
        if not valid:
            logger.error(f"[❌] Invalid SCAN handoff: {error}")
            return False
        
        self.scan = ScanHandoff(**data)
        self.timeline.append({
            'stage': 'SCAN',
            'timestamp': datetime.utcnow().isoformat(),
            'success': True
        })
        logger.info(f"[✅] SCAN handoff added: {len(self.scan.discovered_hosts)} hosts discovered")
        return True
    
    def add_escalate_result(self, data: Dict) -> bool:
        """Add ESCALATE result with validation"""
        valid, error = HandoffValidator.validate_escalate(data)
        if not valid:
            logger.error(f"[❌] Invalid ESCALATE handoff: {error}")
            return False
        
        self.escalate = EscalateHandoff(**data)
        self.timeline.append({
            'stage': 'ESCALATE',
            'timestamp': datetime.utcnow().isoformat(),
            'success': self.escalate.success,
            'privilege_level': self.escalate.privilege_level
        })
        logger.info(f"[{'✅' if self.escalate.success else '⚠️'}] ESCALATE handoff: {self.escalate.escalation_method}")
        return True
    
    def add_extract_result(self, data: Dict) -> bool:
        """Add EXTRACT result with validation"""
        valid, error = HandoffValidator.validate_extract(data)
        if not valid:
            logger.error(f"[❌] Invalid EXTRACT handoff: {error}")
            return False
        
        self.extract = ExtractHandoff(**data)
        self.timeline.append({
            'stage': 'EXTRACT',
            'timestamp': datetime.utcnow().isoformat(),
            'extracted_items': (
                len(self.extract.credentials) +
                len(self.extract.sensitive_files) +
                len(self.extract.database_creds) +
                len(self.extract.private_keys)
            )
        })
        logger.info(f"[✅] EXTRACT handoff: {self.extract.total_size_bytes} bytes extracted")
        return True
    
    def add_exfil_result(self, data: Dict) -> bool:
        """Add EXFIL result with validation"""
        valid, error = HandoffValidator.validate_exfil(data)
        if not valid:
            logger.error(f"[❌] Invalid EXFIL handoff: {error}")
            return False
        
        self.exfil = ExfilHandoff(**data)
        self.timeline.append({
            'stage': 'EXFIL',
            'timestamp': datetime.utcnow().isoformat(),
            'exfiltrated_bytes': self.exfil.exfiltrated_bytes,
            'status': self.exfil.status
        })
        logger.info(f"[✅] EXFIL handoff: {self.exfil.exfiltrated_bytes} bytes sent via {self.exfil.exfil_method}")
        return True
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete handoff chain"""
        return {
            'scan': self.scan.to_dict() if self.scan else None,
            'escalate': self.escalate.to_dict() if self.escalate else None,
            'extract': self.extract.to_dict() if self.extract else None,
            'exfil': self.exfil.to_dict() if self.exfil else None,
            'timeline': self.timeline
        }
    
    def audit_log(self) -> str:
        """Generate audit trail of all handoffs"""
        lines = ["=== MISSION HANDOFF AUDIT LOG ==="]
        for event in self.timeline:
            ts = event.get('timestamp', '?')
            stage = event.get('stage', '?')
            if stage == 'SCAN':
                hosts = event.get('hosts', 0)
                lines.append(f"[{ts}] SCAN: {hosts} hosts discovered")
            elif stage == 'ESCALATE':
                success = event.get('success', False)
                priv = event.get('privilege_level', '?')
                lines.append(f"[{ts}] ESCALATE: {priv} privilege {'✅' if success else '❌'}")
            elif stage == 'EXTRACT':
                items = event.get('extracted_items', 0)
                lines.append(f"[{ts}] EXTRACT: {items} items extracted")
            elif stage == 'EXFIL':
                bytes_sent = event.get('exfiltrated_bytes', 0)
                status = event.get('status', '?')
                lines.append(f"[{ts}] EXFIL: {bytes_sent} bytes ({status})")
        
        return "\n".join(lines)
