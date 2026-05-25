#!/usr/bin/env python3
"""
THREAT INTELLIGENCE LOADER - Phase 2.3 Implementation
Real-world threat data integration from public and private sources.
Supports CVE databases, malicious IP lists, attack signatures, and incident tracking.
"""

import json
import logging
import os
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)


class ThreatType(Enum):
    """Classification of threat intelligence"""
    HONEYPOT_PATTERN = "honeypot_pattern"
    MALICIOUS_IP = "malicious_ip"
    CVE = "cve"
    ATTACK_SIGNATURE = "attack_signature"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    BOTNET_NODE = "botnet_node"
    C2_INFRASTRUCTURE = "c2_infrastructure"
    DATA_EXFIL = "data_exfiltration"
    PERSISTENCE = "persistence"
    LATERAL_MOVE = "lateral_movement"


@dataclass
class ThreatSignature:
    """Individual threat signature with confidence and metadata"""
    signature_id: str
    threat_type: ThreatType
    pattern: str  # regex, string match, or behavioral pattern
    severity: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    source: str  # Where this threat came from
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    occurrence_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signature_id': self.signature_id,
            'threat_type': self.threat_type.value,
            'pattern': self.pattern,
            'severity': self.severity,
            'confidence': self.confidence,
            'source': self.source,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'occurrence_count': self.occurrence_count,
            'tags': self.tags
        }


@dataclass
class CVERecord:
    """CVE vulnerability record"""
    cve_id: str
    description: str
    cvss_score: float  # 0.0-10.0
    cvss_vector: str
    affected_products: List[str] = field(default_factory=list)
    affected_versions: List[str] = field(default_factory=list)
    publish_date: datetime = field(default_factory=datetime.now)
    severity: float = 0.0  # Derived from CVSS
    exploit_available: bool = False
    exploit_maturity: str = "UNPROVEN"  # UNPROVEN, PROOF_OF_CONCEPT, FUNCTIONAL, HIGH, FUNCTIONAL_HIGH
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cve_id': self.cve_id,
            'description': self.description,
            'cvss_score': self.cvss_score,
            'cvss_vector': self.cvss_vector,
            'affected_products': self.affected_products,
            'affected_versions': self.affected_versions,
            'publish_date': self.publish_date.isoformat(),
            'severity': self.severity,
            'exploit_available': self.exploit_available,
            'exploit_maturity': self.exploit_maturity,
            'references': self.references
            ,
            'tags': self.tags
        }


@dataclass
class MaliciousIPRecord:
    """Malicious IP address record"""
    ip_address: str
    threat_score: float  # 0.0-1.0
    threat_types: List[ThreatType] = field(default_factory=list)
    reported_by: List[str] = field(default_factory=list)  # Sources that identified this
    first_reported: datetime = field(default_factory=datetime.now)
    last_reported: datetime = field(default_factory=datetime.now)
    report_count: int = 0
    asn: Optional[str] = None
    country_code: Optional[str] = None
    organization: Optional[str] = None
    is_botnet_node: bool = False
    is_c2_infrastructure: bool = False
    associated_malware: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ip_address': self.ip_address,
            'threat_score': self.threat_score,
            'threat_types': [t.value for t in self.threat_types],
            'reported_by': self.reported_by,
            'first_reported': self.first_reported.isoformat(),
            'last_reported': self.last_reported.isoformat(),
            'report_count': self.report_count,
            'asn': self.asn,
            'country_code': self.country_code,
            'organization': self.organization,
            'is_botnet_node': self.is_botnet_node,
            'is_c2_infrastructure': self.is_c2_infrastructure,
            'associated_malware': self.associated_malware
        }


class ThreatIntelligenceLoader:
    """
    Real-world threat intelligence loader.
    Aggregates data from public sources and maintains a persistent threat database.
    """
    
    def __init__(self, db_path: str = "./data/threat_intelligence.db", 
                 cache_ttl_hours: int = 24):
        self.db_path = db_path
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.threat_signatures: Dict[str, ThreatSignature] = {}
        self.malicious_ips: Dict[str, MaliciousIPRecord] = {}
        self.cve_database: Dict[str, CVERecord] = {}
        self.last_update = datetime.now()
        
        self._initialize_database()
        self._load_default_threats()
    
    def _initialize_database(self):
        """Initialize SQLite database for threat persistence"""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Threat signatures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_signatures (
                    signature_id TEXT PRIMARY KEY,
                    threat_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    severity REAL,
                    confidence REAL,
                    source TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    occurrence_count INTEGER,
                    tags TEXT
                )
            ''')
            
            # Malicious IPs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS malicious_ips (
                    ip_address TEXT PRIMARY KEY,
                    threat_score REAL,
                    threat_types TEXT,
                    reported_by TEXT,
                    first_reported TIMESTAMP,
                    last_reported TIMESTAMP,
                    report_count INTEGER,
                    asn TEXT,
                    country_code TEXT,
                    organization TEXT,
                    is_botnet_node INTEGER,
                    is_c2_infrastructure INTEGER,
                    associated_malware TEXT
                )
            ''')
            
            # CVE records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cve_records (
                    cve_id TEXT PRIMARY KEY,
                    description TEXT,
                    cvss_score REAL,
                    cvss_vector TEXT,
                    affected_products TEXT,
                    affected_versions TEXT,
                    publish_date TIMESTAMP,
                    severity REAL,
                    exploit_available INTEGER,
                    exploit_maturity TEXT,
                    references TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            log.info(f"Initialized threat intelligence database at {self.db_path}")
        except Exception as e:
            log.error(f"Error initializing threat database: {e}")
    
    def _load_default_threats(self):
        """Load default/built-in threat signatures"""
        # Common honeypot patterns
        honeypot_patterns = [
            ThreatSignature(
                signature_id="hp_cowrie",
                threat_type=ThreatType.HONEYPOT_PATTERN,
                pattern=r"Cowrie|SSH.*Honeypot",
                severity=0.9,
                confidence=0.95,
                source="built-in",
                tags=["ssh", "honeypot", "cowrie"]
            ),
            ThreatSignature(
                signature_id="hp_glastopf",
                threat_type=ThreatType.HONEYPOT_PATTERN,
                pattern=r"Glastopf|Web.*Honeypot",
                severity=0.9,
                confidence=0.95,
                source="built-in",
                tags=["web", "honeypot", "glastopf"]
            ),
            ThreatSignature(
                signature_id="hp_honeyd",
                threat_type=ThreatType.HONEYPOT_PATTERN,
                pattern=r"honeyd|virtual.*honeypot",
                severity=0.85,
                confidence=0.90,
                source="built-in",
                tags=["honeypot", "honeyd", "virtual"]
            ),
            ThreatSignature(
                signature_id="hp_weird_ports",
                threat_type=ThreatType.HONEYPOT_PATTERN,
                pattern=r"too_many_ports|unusual_port_combo",
                severity=0.7,
                confidence=0.75,
                source="built-in",
                tags=["honeypot", "suspicious", "ports"]
            ),
        ]
        
        # Attack signatures
        attack_signatures = [
            ThreatSignature(
                signature_id="atk_sql_injection",
                threat_type=ThreatType.ATTACK_SIGNATURE,
                pattern=r"UNION.*SELECT|DROP.*TABLE|' OR '1'='1",
                severity=0.95,
                confidence=0.98,
                source="built-in",
                tags=["sqli", "injection", "web"]
            ),
            ThreatSignature(
                signature_id="atk_xss",
                threat_type=ThreatType.ATTACK_SIGNATURE,
                pattern=r"<script>|javascript:|onerror=",
                severity=0.85,
                confidence=0.90,
                source="built-in",
                tags=["xss", "web", "injection"]
            ),
            ThreatSignature(
                signature_id="atk_command_injection",
                threat_type=ThreatType.ATTACK_SIGNATURE,
                pattern=r"; rm -rf|bash -i >|nc.*-e",
                severity=0.95,
                confidence=0.99,
                source="built-in",
                tags=["command_injection", "rce", "shell"]
            ),
            ThreatSignature(
                signature_id="atk_path_traversal",
                threat_type=ThreatType.ATTACK_SIGNATURE,
                pattern=r"\.\./\.\./|\.\.\\/|%2e%2e",
                severity=0.85,
                confidence=0.90,
                source="built-in",
                tags=["traversal", "path_disclosure", "web"]
            ),
        ]
        
        # Behavioral anomalies
        behavioral_signatures = [
            ThreatSignature(
                signature_id="beh_port_scan",
                threat_type=ThreatType.SUSPICIOUS_BEHAVIOR,
                pattern=r"sequential_port_probing|nmap_signature",
                severity=0.75,
                confidence=0.80,
                source="built-in",
                tags=["recon", "scanning", "behavior"]
            ),
            ThreatSignature(
                signature_id="beh_brute_force",
                threat_type=ThreatType.SUSPICIOUS_BEHAVIOR,
                pattern=r"multiple_failed_auth|rapid_login_attempts",
                severity=0.80,
                confidence=0.85,
                source="built-in",
                tags=["auth", "brute_force", "behavior"]
            ),
            ThreatSignature(
                signature_id="beh_data_exfil",
                threat_type=ThreatType.DATA_EXFIL,
                pattern=r"large_outbound_transfer|suspicious_dns_queries",
                severity=0.90,
                confidence=0.88,
                source="built-in",
                tags=["exfil", "data_theft", "behavior"]
            ),
        ]
        
        # Load common CVEs
        common_cves = [
            CVERecord(
                cve_id="CVE-2023-44487",
                description="HTTP/2 Rapid Reset attack vulnerability",
                cvss_score=7.5,
                cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
                affected_products=["nginx", "Apache", "OpenBSD"],
                severity=0.75,
                exploit_available=True,
                exploit_maturity="FUNCTIONAL_HIGH",
                tags=["http2", "dos", "network"]
            ),
            CVERecord(
                cve_id="CVE-2023-21839",
                description="Oracle WebLogic Server RCE vulnerability",
                cvss_score=9.8,
                cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                affected_products=["Oracle WebLogic Server"],
                severity=0.98,
                exploit_available=True,
                exploit_maturity="FUNCTIONAL_HIGH",
                tags=["rce", "oracle", "critical"]
            ),
            CVERecord(
                cve_id="CVE-2023-22518",
                description="Confluence Data Center Improper Authorization",
                cvss_score=9.8,
                cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                affected_products=["Atlassian Confluence"],
                severity=0.98,
                exploit_available=True,
                exploit_maturity="FUNCTIONAL_HIGH",
                tags=["confluence", "auth_bypass", "critical"]
            ),
        ]
        
        # Add all to collections
        for sig in honeypot_patterns + attack_signatures + behavioral_signatures:
            self.threat_signatures[sig.signature_id] = sig
        
        for cve in common_cves:
            self.cve_database[cve.cve_id] = cve
        
        log.info(f"Loaded {len(self.threat_signatures)} threat signatures, {len(self.cve_database)} CVE records")
    
    def add_malicious_ip(self, ip_address: str, threat_score: float, 
                         threat_types: List[ThreatType], source: str,
                         country_code: Optional[str] = None,
                         is_botnet: bool = False,
                         is_c2: bool = False) -> MaliciousIPRecord:
        """Add or update a malicious IP record"""
        if ip_address in self.malicious_ips:
            record = self.malicious_ips[ip_address]
            record.threat_score = max(record.threat_score, threat_score)
            record.threat_types = list(set(record.threat_types + threat_types))
            record.reported_by.append(source)
            record.report_count += 1
            record.last_reported = datetime.now()
        else:
            record = MaliciousIPRecord(
                ip_address=ip_address,
                threat_score=threat_score,
                threat_types=threat_types,
                reported_by=[source],
                country_code=country_code,
                is_botnet_node=is_botnet,
                is_c2_infrastructure=is_c2,
                report_count=1
            )
            self.malicious_ips[ip_address] = record
        
        self._persist_malicious_ip(record)
        return record
    
    def get_threat_level(self, indicators: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Calculate threat level based on multiple indicators.
        Returns (threat_score: 0.0-1.0, matched_signatures: List[signature_ids])
        """
        threat_score = 0.0
        matched_signatures = []
        
        ip_address = indicators.get("ip_address")
        services = indicators.get("services", {})
        fingerprint = indicators.get("fingerprint", "")
        ports = indicators.get("ports", [])
        
        # Check against malicious IP database
        if ip_address and ip_address in self.malicious_ips:
            record = self.malicious_ips[ip_address]
            threat_score += record.threat_score * 0.4  # 40% weight
            matched_signatures.extend([f"ip_match:{ip_address}"])
        
        # Check against threat signatures
        for sig_id, signature in self.threat_signatures.items():
            # Service name matching (for honeypot patterns, etc.)
            for service in services.values():
                if service and signature.pattern.lower() in service.lower():
                    threat_score += (signature.severity * signature.confidence) * 0.3
                    matched_signatures.append(sig_id)
                    break
            
            # Fingerprint/banner matching
            if fingerprint and signature.pattern.lower() in fingerprint.lower():
                threat_score += (signature.severity * signature.confidence) * 0.2
                matched_signatures.append(sig_id)
        
        # Check for suspicious port combinations
        high_risk_ports = {22, 23, 3389, 445, 139, 135}  # SSH, Telnet, RDP, SMB, NetBIOS
        if len(set(ports) & high_risk_ports) >= 3:
            threat_score += 0.15
            matched_signatures.append("suspicious_port_combo")
        
        # Normalize to 0.0-1.0
        threat_score = min(threat_score, 1.0)
        
        return threat_score, list(set(matched_signatures))
    
    def find_relevant_cves(self, product: str, version: Optional[str] = None) -> List[CVERecord]:
        """Find CVE records relevant to a product and version"""
        relevant_cves = []
        
        for cve_id, cve in self.cve_database.items():
            for affected_product in cve.affected_products:
                if product.lower() in affected_product.lower():
                    if version is None or version in cve.affected_versions:
                        relevant_cves.append(cve)
                    break
        
        # Sort by CVSS score (highest first)
        return sorted(relevant_cves, key=lambda x: x.cvss_score, reverse=True)
    
    def export_as_dict(self) -> Dict[str, Any]:
        """Export all threat intelligence as dictionary"""
        return {
            'threat_signatures': {k: v.to_dict() for k, v in self.threat_signatures.items()},
            'malicious_ips': {k: v.to_dict() for k, v in self.malicious_ips.items()},
            'cve_database': {k: v.to_dict() for k, v in self.cve_database.items()},
            'last_update': self.last_update.isoformat(),
            'stats': {
                'total_signatures': len(self.threat_signatures),
                'total_malicious_ips': len(self.malicious_ips),
                'total_cves': len(self.cve_database),
            }
        }
    
    def _persist_malicious_ip(self, record: MaliciousIPRecord):
        """Persist malicious IP record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO malicious_ips 
                (ip_address, threat_score, threat_types, reported_by, first_reported, 
                 last_reported, report_count, asn, country_code, organization,
                 is_botnet_node, is_c2_infrastructure, associated_malware)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.ip_address,
                record.threat_score,
                json.dumps([t.value for t in record.threat_types]),
                json.dumps(record.reported_by),
                record.first_reported.isoformat(),
                record.last_reported.isoformat(),
                record.report_count,
                record.asn,
                record.country_code,
                record.organization,
                1 if record.is_botnet_node else 0,
                1 if record.is_c2_infrastructure else 0,
                json.dumps(record.associated_malware)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"Error persisting malicious IP {record.ip_address}: {e}")


# Public API for threat intelligence
_threat_loader: Optional[ThreatIntelligenceLoader] = None


def get_threat_loader() -> ThreatIntelligenceLoader:
    """Get or create the global threat intelligence loader"""
    global _threat_loader
    if _threat_loader is None:
        _threat_loader = ThreatIntelligenceLoader()
    return _threat_loader
