#!/usr/bin/env python3
"""
Network Mapper Integration with Real Persistence Layer

Enhances network_mapper.py to:
1. Load real baselines from scan history (not mocked)
2. Feed scan results back to persistence for drift detection
3. Detect anomalies against real historical baselines
4. Build NodeProfiles with real baseline data

This bridges the gap between live_scanner → persistence → network_mapper
"""

import logging
from typing import Dict, Optional, List, Any
from recon.network_mapper import RealNetworkMapper, NodeProfile
from recon.scan_persistence import ScanPersistence, DriftAnomalyDetector
from recon.baseline_builder import RealBaselineBuilder, compare_to_baseline
from recon.threat_intelligence import get_threat_loader, ThreatIntelligenceLoader

logger = logging.getLogger(__name__)


class EnhancedNetworkMapper(RealNetworkMapper):
    """
    Enhanced network mapper that integrates real persistence baselines.
    
    Extends RealNetworkMapper with:
    - Real baseline loading from scan history
    - Drift detection
    - Anomaly events from actual drift
    """
    
    def __init__(self, vault=None, config: Optional[Dict] = None):
        # Support both positional config or (vault, config) signatures
        if isinstance(vault, dict) and config is None:
            config = vault
            vault = None
        super().__init__(vault=vault, config=config)
        self.persistence = ScanPersistence()
        self.baseline_builder = RealBaselineBuilder(self.persistence)
        self.real_baselines = {}  # {ip: RealNodeBaseline}
        self.threat_loader = get_threat_loader()
    
    def load_real_baselines(self, subnet: str, max_scans: int = 10):
        """
        Load real baselines for a subnet from scan history.
        
        This replaces the mocked baseline system with actual observations.
        """
        logger.info(f"[→] Loading real baselines for {subnet}...")
        
        try:
            self.real_baselines = self.baseline_builder.build_subnet_baseline(
                subnet,
                max_scans=max_scans
            )
            logger.info(f"[✓] Loaded baselines for {len(self.real_baselines)} hosts")
        except Exception as e:
            logger.warning(f"[⚠️] Could not load real baselines: {e}")
            self.real_baselines = {}
    
    def detect_drift(self, subnet: str, current_hosts: Dict) -> Dict:
        """
        Detect drift from historical data.
        
        Returns:
            Drift analysis with events
        """
        drift = self.persistence.compute_drift(subnet, current_hosts)
        
        if drift['previous_scan_timestamp']:
            logger.info(f"[📊] Drift detected: {drift['drift_score']:.0%}")
            
            # Generate anomaly events
            events = DriftAnomalyDetector.generate_anomaly_events(subnet, drift)
            for evt in events:
                logger.info(f"    - {evt['event_type']}: {evt['host_ip']}")
        else:
            logger.info("[ℹ️] No previous scan (first scan)")
        
        return drift
    
    def scan_subnet_with_real_baselines(
        self,
        subnet: str,
        scan_mode: str = "auto",
        aggressive: bool = False,
        max_retries: int = 3,
        load_baselines: bool = True
    ) -> Dict[str, NodeProfile]:
        """
        Scan subnet and enrich with real baseline data.
        
        Workflow:
        1. Load real baselines from history
        2. Perform scan
        3. Detect drift from previous scan
        4. Compare nodes against real baselines
        5. Propagate baseline data to NodeProfile objects
        
        Args:
            subnet: Target subnet
            scan_mode: 'auto', 'arp', 'icmp'
            aggressive: Deep scanning mode
            max_retries: Retry failed nodes
            load_baselines: Load historical baselines
        
        Returns:
            Dict[ip: NodeProfile] with real baseline enrichment
        """
        logger.info(f"\n[🔄] Enhanced scan: {subnet} (with real baselines)")
        
        # Step 1: Load real baselines
        if load_baselines:
            self.load_real_baselines(subnet)
        
        # Step 2: Perform scan
        try:
            results = self.scan_subnet(subnet, scan_mode=scan_mode, aggressive=aggressive, max_retries=max_retries)
        except Exception as e:
            logger.error(f"[!] Scan failed: {e}")
            return {}
        
        # Step 3: Detect drift from previous scan
        drift = self.detect_drift(subnet, results)
        
        # Step 4: Enrich nodes with real baseline data
        for ip, node in results.items():
            if ip in self.real_baselines:
                real_baseline = self.real_baselines[ip]
                
                # Link baseline to node
                node.baseline = real_baseline
                
                # Detect anomalies against real baseline
                anomalies = compare_to_baseline(
                    {
                        'open_ports': node.open_ports,
                        'banners': {p: None for p in node.open_ports}  # Simplified
                    },
                    real_baseline
                )
                
                # Update node with anomaly data
                if anomalies['anomaly_score'] > 0.1:
                    logger.info(f"[⚠️] {ip}: Anomaly {anomalies['anomaly_score']:.0%}")
                    
                    if anomalies['port_deviation']:
                        logger.info(f"    NEW PORTS: {anomalies['port_deviation']}")
                    if anomalies['service_deviation']:
                        logger.info(f"    NEW SERVICES: {anomalies['service_deviation']}")
                    if anomalies['os_changed']:
                        logger.info(f"    OS CHANGED: {node.os}")
            else:
                # New host (not in baseline history)
                if ip in drift.get('new_hosts', []):
                    logger.info(f"[⚠️] {ip}: NEW HOST (not in baseline)")
                    node.status = "UNKNOWN"

            # Step 5: Correlate Threat Intelligence and CVEs
            self.enrich_node_with_threat_intel(node)
        
        return results

    def enrich_node_with_threat_intel(self, node: NodeProfile) -> NodeProfile:
        """
        Enriches a NodeProfile with vulnerability intelligence and exploit viability scores
        from the ThreatIntelligenceLoader.
        """
        services_data = []
        for s in getattr(node, 'services', []):
            services_data.append({
                "port": s.port,
                "name": s.name,
                "version": getattr(s, 'version', None),
                "banner": getattr(s, 'product', '') or s.name
            })

        threat_assessment = self.threat_loader.assess_node_threat(
            node_ip=node.ip,
            open_ports=node.open_ports,
            services=services_data
        )

        node.cves = threat_assessment["cves"]
        node.threat_score = threat_assessment["threat_score"]
        node.exploit_viability = threat_assessment["exploit_viability"]
        node.threat_tags = threat_assessment["threat_tags"]

        if node.cves:
            logger.info(
                f"[🎯] Threat Intel: {node.ip} matched {len(node.cves)} CVEs "
                f"(exploit viability: {node.exploit_viability:.0%}, top: {node.cves[0]['cve_id']})"
            )
        return node

    def get_cves_for_host(self, ip: str) -> List[Dict[str, Any]]:
        """Get correlated CVE records for a specific host."""
        if ip in self.discovered_nodes:
            return getattr(self.discovered_nodes[ip], 'cves', [])
        return []

    def get_baseline_for_host(self, ip: str):
        """Get real baseline for a specific host."""
        return self.real_baselines.get(ip)
    
    def get_scan_history_for_host(self, subnet: str, ip: str, limit: int = 5):
        """Get scan history observations for a specific host."""
        history = self.persistence.get_scan_history(subnet, limit)
        
        host_history = []
        for scan in history:
            if ip in scan.hosts:
                host_history.append(scan.hosts[ip])
        
        return host_history


def integrate_persistence_with_network_mapper(mapper: RealNetworkMapper) -> EnhancedNetworkMapper:
    """
    Convert existing RealNetworkMapper to EnhancedNetworkMapper.
    
    This allows migrating existing code to use real baselines.
    """
    enhanced = EnhancedNetworkMapper(mapper.config if hasattr(mapper, 'config') else {})
    # Copy existing state
    enhanced.discovered_nodes = mapper.discovered_nodes
    enhanced.scan_history = mapper.scan_history
    enhanced.anomalies = mapper.anomalies if hasattr(mapper, 'anomalies') else []
    
    return enhanced


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
    
    # Example usage
    mapper = EnhancedNetworkMapper()
    
    results = mapper.scan_subnet_with_real_baselines(
        "192.168.1.0/24",
        scan_mode="auto",
        load_baselines=True
    )
    
    print(f"\n[✓] Scanned {len(results)} hosts with real basename enrichment")
    for ip, node in list(results.items())[:3]:
        baseline = mapper.get_baseline_for_host(ip)
        if baseline:
            print(f"  {ip}: {node.status} (baseline: {len(baseline.expected_open_ports)} ports, stability: {baseline.stability_score:.0%})")
        else:
            print(f"  {ip}: {node.status} (no baseline)")
