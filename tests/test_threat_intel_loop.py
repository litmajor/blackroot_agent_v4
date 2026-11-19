#!/usr/bin/env python3
"""
Unit and Integration Tests for Option C: Threat Intel Loop
Validates:
1. ThreatIntelligenceLoader CVE database expansion and banner/port matching.
2. Fast CVE lookup utility functions (cve_lookup.py).
3. EnhancedNetworkMapper enrichment with CVEs and exploit viability.
4. NodeProfile prioritization bonus for weaponizable/exploitable CVE targets.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.threat_intelligence import get_threat_loader, ThreatIntelligenceLoader
from recon.cve_lookup import lookup_cves_for_banner, lookup_cves_for_service, get_service_viability
from recon.network_mapper import NodeProfile, ServiceInfo
from recon.network_mapper_integration import EnhancedNetworkMapper


def test_threat_intelligence_expanded_cves():
    loader = get_threat_loader()
    assert "CVE-2021-44228" in loader.cve_database, "Log4Shell missing"
    assert "CVE-2020-1472" in loader.cve_database, "Zerologon missing"
    assert "CVE-2017-0144" in loader.cve_database, "EternalBlue missing"
    assert "CVE-2019-0708" in loader.cve_database, "BlueKeep missing"


def test_match_service_to_cves():
    loader = get_threat_loader()
    
    # SMB port match
    smb_cves = loader.match_service_to_cves("microsoft-ds", port=445)
    cve_ids = [c.cve_id for c in smb_cves]
    assert any("2017-0144" in cid or "2020-1472" in cid for cid in cve_ids), "Failed to match SMB CVEs"

    # RDP match
    rdp_cves = loader.match_service_to_cves("ms-wbt-server", port=3389)
    assert any("2019-0708" in c.cve_id for c in rdp_cves), "Failed to match BlueKeep"

    # Apache Log4j banner match
    log4j_cves = loader.match_service_to_cves("http", banner="Apache-Coyote/1.1 log4j")
    assert any("2021-44228" in c.cve_id for c in log4j_cves), "Failed to match Log4Shell"


def test_cve_lookup_utilities():
    # Banner lookup
    cves = lookup_cves_for_banner("Apache 2.4 with http2", port=80)
    assert len(cves) > 0, "No CVEs found for Apache http2"

    # Service lookup
    smb_cves = lookup_cves_for_service("smb", port=445)
    assert len(smb_cves) > 0, "No CVEs found for smb:445"

    # Service viability
    viability, tags = get_service_viability("smb", port=445)
    assert viability > 0.7, f"Expected high viability for SMB, got {viability}"
    assert "smb" in tags or "critical" in tags, f"Missing expected tags: {tags}"


def test_enhanced_network_mapper_threat_enrichment():
    mapper = EnhancedNetworkMapper()
    
    node = NodeProfile(
        ip="192.168.1.50",
        open_ports=[445, 3389],
        services=[
            ServiceInfo(port=445, protocol="tcp", name="microsoft-ds", product="Windows SMB"),
            ServiceInfo(port=3389, protocol="tcp", name="ms-wbt-server", product="Windows RDP")
        ],
        is_up=True
    )
    
    # Calculate baseline zombie score before enrichment
    initial_score = node.zombie_candidate_score
    
    # Enrich with threat intel
    enriched = mapper.enrich_node_with_threat_intel(node)
    
    assert len(enriched.cves) > 0, "Enriched node should have matched CVEs"
    assert enriched.exploit_viability > 0.5, f"Exploit viability should be high, got {enriched.exploit_viability}"
    assert enriched.threat_score > 0.5, f"Threat score should be high, got {enriched.threat_score}"
    assert len(enriched.threat_tags) > 0, "Threat tags should be present"
    
    # Node with exploitable CVEs should get priority boost
    boosted_score = enriched.zombie_candidate_score
    assert boosted_score > initial_score, f"Expected boosted score > {initial_score}, got {boosted_score}"

    # Verify to_dict includes threat fields
    d = enriched.to_dict()
    assert "cves" in d
    assert "threat_score" in d
    assert "exploit_viability" in d
    assert "threat_tags" in d
