#!/usr/bin/env python3
"""
CVE Lookup Utility for Autonomous Reconnaissance
Provides fast, lightweight service/banner to CVE correlation powered by ThreatIntelligenceLoader.
"""

from typing import List, Dict, Optional, Tuple, Any
from recon.threat_intelligence import get_threat_loader, CVERecord


def lookup_cves_for_banner(banner: str, port: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Search CVE database for vulnerabilities matching a service banner string.
    
    Args:
        banner: Raw banner text (e.g. 'Apache/2.4.49 (Unix)', 'OpenSSH_7.4p1')
        port: Optional port number for contextual port-to-service hints
        
    Returns:
        List of matching CVE record dictionaries sorted by CVSS score descending.
    """
    loader = get_threat_loader()
    cves = loader.match_service_to_cves(service_name="", banner=banner, port=port)
    return [c.to_dict() for c in cves]


def lookup_cves_for_service(service_name: str, version: Optional[str] = None, port: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Search CVE database for vulnerabilities matching a service name and optional version.
    
    Args:
        service_name: Service identifier (e.g. 'smb', 'rdp', 'ssh', 'weblogic')
        version: Version string if available
        port: Port number for protocol hints
        
    Returns:
        List of matching CVE record dictionaries sorted by CVSS score descending.
    """
    loader = get_threat_loader()
    cves = loader.match_service_to_cves(service_name=service_name, version=version, port=port)
    return [c.to_dict() for c in cves]


def get_service_viability(service_name: str, port: int, banner: Optional[str] = None) -> Tuple[float, List[str]]:
    """
    Computes an exploit viability score (0.0 - 1.0) and associated threat tags for a service.
    
    Returns:
        (viability_score, list_of_threat_tags)
    """
    loader = get_threat_loader()
    assessment = loader.assess_node_threat(
        node_ip="0.0.0.0",
        open_ports=[port],
        services=[{"name": service_name, "port": port, "banner": banner or ""}]
    )
    return assessment["exploit_viability"], assessment["threat_tags"]
