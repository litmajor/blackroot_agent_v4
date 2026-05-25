#!/usr/bin/env python3
"""
Real Baseline Builder
- Builds NodeBaseline from actual historical scan data (not mocked)
- Tracks expected ports, services, OS across time
- Computes stability scores based on consistency
- Enables anomaly detection on real observations
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from recon.scan_persistence import ScanPersistence, ScanRecord

logger = logging.getLogger(__name__)


@dataclass
class PortStability:
    """Track port consistency across scans"""
    port: int
    appeared_in_scans: int  # How many scans showed this port open
    total_scans: int        # Total scans analyzed
    consistency: float      # appeared_in_scans / total_scans
    
    def is_stable(self, threshold: float = 0.8) -> bool:
        """Port is stable if it appears in ≥80% of scans"""
        return self.consistency >= threshold


@dataclass
class RealNodeBaseline:
    """
    Real baseline built from historical scan data.
    
    Unlike mocked baselines, this is derived from actual observations.
    """
    ip: str
    subnet: str
    
    # Observed ports (from scans)
    expected_open_ports: List[int] = field(default_factory=list)
    expected_closed_ports: List[int] = field(default_factory=list)
    
    # Port stability across scans
    port_stability: Dict[int, PortStability] = field(default_factory=dict)
    
    # Services (from banners)
    expected_services: List[str] = field(default_factory=list)
    service_history: Dict[str, int] = field(default_factory=dict)  # service → count in scans
    
    # OS
    os_guess: Optional[str] = None
    os_consistency: float = 0.0
    
    # Stability score (0-1)
    stability_score: float = 0.0
    
    # Metadata
    first_seen: Optional[str] = None  # Timestamp of first scan
    last_updated: Optional[str] = None
    scans_analyzed: int = 0
    
    def compute_stability(self, threshold: float = 0.8) -> float:
        """
        Compute stability score (0-1).
        
        Based on:
        - Port consistency (stable ports = +points)
        - Service consistency (consistent services = +points)
        - OS consistency (stable OS = +points)
        """
        if self.scans_analyzed == 0:
            return 0.0
        
        scores = []
        
        # Port stability component
        if self.port_stability:
            stable_ports = sum(1 for p in self.port_stability.values() if p.is_stable(threshold))
            port_score = stable_ports / len(self.port_stability) if self.port_stability else 0.0
            scores.append(port_score)
        
        # Service stability component
        if self.service_history:
            avg_service_consistency = sum(count / self.scans_analyzed for count in self.service_history.values()) / len(self.service_history)
            scores.append(avg_service_consistency)
        
        # OS consistency component
        scores.append(self.os_consistency)
        
        # Average all components
        self.stability_score = sum(scores) / len(scores) if scores else 0.0
        return self.stability_score


class RealBaselineBuilder:
    """
    Build realistic baselines from scan history.
    """
    
    def __init__(self, persistence: ScanPersistence):
        self.persistence = persistence
    
    def build_baseline_for_host(self, subnet: str, ip: str, max_scans: int = 10) -> RealNodeBaseline:
        """
        Build baseline for a specific host from historical scans.
        
        Args:
            subnet: e.g., "192.168.1.0/24"
            ip: e.g., "192.168.1.100"
            max_scans: analyze up to N recent scans
        
        Returns:
            RealNodeBaseline built from actual observations
        """
        baseline = RealNodeBaseline(ip=ip, subnet=subnet)
        
        # Load scan history
        history = self.persistence.get_scan_history(subnet, limit=max_scans)
        if not history:
            logger.warning(f"[!] No scan history for {ip} on {subnet}")
            return baseline
        
        baseline.scans_analyzed = len(history)
        baseline.first_seen = history[-1].timestamp  # Oldest
        baseline.last_updated = history[0].timestamp  # Newest
        
        # Track observations across scans
        port_observations = {}  # port → count
        service_observations = {}  # service → count
        os_observations = {}  # os → count
        
        for scan in history:
            host_data = scan.hosts.get(ip)
            if not host_data:
                continue
            
            # Extract ports
            for port in host_data.get('open_ports', []):
                port_observations[port] = port_observations.get(port, 0) + 1
            
            # Extract services
            for port, banner in host_data.get('banners', {}).items():
                if isinstance(banner, dict):
                    svc = banner.get('service_name', 'unknown')
                else:
                    svc = getattr(banner, 'service_name', 'unknown')
                service_observations[svc] = service_observations.get(svc, 0) + 1
            
            # Extract OS
            os_guess = host_data.get('os_guess')
            if os_guess:
                os_observations[os_guess] = os_observations.get(os_guess, 0) + 1
        
        # Build baseline from observations
        baseline.expected_open_ports = sorted(port_observations.keys())
        baseline.expected_services = sorted(service_observations.keys())
        
        # Port stability
        for port, count in port_observations.items():
            baseline.port_stability[port] = PortStability(
                port=port,
                appeared_in_scans=count,
                total_scans=baseline.scans_analyzed,
                consistency=count / baseline.scans_analyzed if baseline.scans_analyzed > 0 else 0
            )
        
        # Service history
        baseline.service_history = service_observations
        
        # OS (most common)
        if os_observations:
            most_common_os = max(os_observations.items(), key=lambda x: x[1])
            baseline.os_guess = most_common_os[0]
            baseline.os_consistency = most_common_os[1] / baseline.scans_analyzed
        
        # Compute stability
        baseline.compute_stability()
        
        return baseline
    
    def build_subnet_baseline(self, subnet: str, max_scans: int = 5) -> Dict[str, RealNodeBaseline]:
        """
        Build baselines for all hosts in a subnet.
        
        Returns:
            {ip: RealNodeBaseline}
        """
        logger.info(f"[→] Building real baseline for {subnet}...")
        
        history = self.persistence.get_scan_history(subnet, limit=max_scans)
        if not history:
            logger.warning(f"[!] No scan history for {subnet}")
            return {}
        
        # Collect all IPs seen in any scan
        all_ips = set()
        for scan in history:
            all_ips.update(scan.hosts.keys())
        
        logger.info(f"[→] Found {len(all_ips)} unique hosts across {len(history)} scans")
        
        baselines = {}
        for ip in sorted(all_ips):
            baseline = self.build_baseline_for_host(subnet, ip, max_scans)
            baselines[ip] = baseline
            
            if baseline.scans_analyzed > 0:
                logger.info(f"  [{ip}] {len(baseline.expected_open_ports)} ports (stability: {baseline.stability_score:.0%})")
        
        return baselines


def compare_to_baseline(current_host: Dict, baseline: RealNodeBaseline) -> Dict:
    """
    Compare current observation against baseline.
    
    Returns:
        {
            'port_deviation': List[port],  # Ports not in baseline
            'missing_ports': List[port],   # Baseline ports not observed
            'service_deviation': List[svc],
            'missing_services': List[svc],
            'os_changed': bool,
            'anomaly_score': float (0-1, higher = more anomalous)
        }
    """
    result = {
        'port_deviation': [],
        'missing_ports': [],
        'service_deviation': [],
        'missing_services': [],
        'os_changed': False,
        'anomaly_score': 0.0
    }
    
    # Port comparison
    current_open = set(current_host.get('open_ports', []))
    baseline_ports = set(baseline.expected_open_ports)
    
    result['port_deviation'] = sorted(list(current_open - baseline_ports))
    result['missing_ports'] = sorted(list(baseline_ports - current_open))
    
    # Service comparison
    current_services = set()
    for port, banner in current_host.get('banners', {}).items():
        if isinstance(banner, dict):
            current_services.add(banner.get('service_name', 'unknown'))
        else:
            current_services.add(getattr(banner, 'service_name', 'unknown'))
    
    baseline_services = set(baseline.expected_services)
    result['service_deviation'] = sorted(list(current_services - baseline_services))
    result['missing_services'] = sorted(list(baseline_services - current_services))
    
    # OS comparison
    current_os = current_host.get('os_guess')
    result['os_changed'] = current_os and current_os != baseline.os_guess
    
    # Anomaly score (0-1)
    anomaly_components = []
    
    # Port deviation weight (new ports opening = suspicious)
    if baseline_ports:
        port_anomaly = len(result['port_deviation']) / len(baseline_ports)
        anomaly_components.append(port_anomaly)
    
    # Service deviation weight
    if baseline_services:
        service_anomaly = len(result['service_deviation']) / len(baseline_services)
        anomaly_components.append(service_anomaly)
    
    # OS change (high significance)
    if result['os_changed']:
        anomaly_components.append(0.8)
    
    if anomaly_components:
        result['anomaly_score'] = sum(anomaly_components) / len(anomaly_components)
        result['anomaly_score'] = min(1.0, result['anomaly_score'])
    
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Example usage
    persistence = ScanPersistence()
    builder = RealBaselineBuilder(persistence)
    
    # Build baseline for a subnet
    baselines = builder.build_subnet_baseline("192.168.1.0/24", max_scans=5)
    
    if baselines:
        print(f"\n[✓] Built {len(baselines)} baselines")
        for ip, baseline in sorted(baselines.items())[:3]:
            print(f"\n  {ip}:")
            print(f"    - Ports: {baseline.expected_open_ports[:5]}...")
            print(f"    - Services: {baseline.expected_services}")
            print(f"    - OS: {baseline.os_guess}")
            print(f"    - Stability: {baseline.stability_score:.0%}")
