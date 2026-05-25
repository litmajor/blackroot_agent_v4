#!/usr/bin/env python3
"""
Demo: Step 1 - Real Scanning + Memory (Historical Intelligence)

Workflow:
1. Run network scan (real Scapy-based)
2. Persist results to disk with timestamp
3. Load previous scans (build history)
4. Compute drift (what changed)
5. Build real baseline (from actual observations)
6. Detect anomalies (compare current vs baseline)

This replaces mocke data with REAL, PERSISTENT network intelligence.
"""

import sys
import logging
import time
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recon.live_scanner import LiveNetworkScanner
from recon.scan_persistence import ScanPersistence, DriftAnomalyDetector
from recon.baseline_builder import RealBaselineBuilder, compare_to_baseline

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def demo_real_scanning_with_memory(subnet: str = "192.168.1.0/24", scan_mode: str = "auto"):
    """
    Complete Step 1 workflow: Scan → Persist → Detect Drift → Build Baseline → Anomalies
    """
    
    print("\n" + "=" * 100)
    print("STEP 1: REAL SCANNING + MEMORY (Historical Intelligence)")
    print("=" * 100)
    
    # Phase 1: Network Scan
    print(f"\n[PHASE 1] Network Scanning ({scan_mode} mode, with persistence)")
    print("-" * 100)
    
    try:
        scanner = LiveNetworkScanner(max_threads=20)
        hosts = scanner.scan_network(
            subnet,
            scan_mode=scan_mode,
            grab_banners=True,
            persist=True  # ← Enable persistence automatically
        )
        
        if not hosts:
            print("[!] No hosts discovered")
            return
        
    except PermissionError:
        print("[!] Administrator privileges required")
        print("    Run: python -m pip install scapy")
        print("    Then run this script as Administrator")
        return
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 2: Load Scan History & Drift Detection
    print(f"\n[PHASE 2] Drift Detection (compare current vs historical)")
    print("-" * 100)
    
    persistence = ScanPersistence()
    drift = persistence.compute_drift(subnet, hosts)
    
    if drift['previous_scan_timestamp']:
        print(f"Previous scan: {drift['previous_scan_timestamp']}")
        print(f"Current hosts: {drift['total_hosts_current']} vs Previous: {drift['total_hosts_previous']}")
        print(f"Drift score: {drift['drift_score']:.0%} (0%=identical, 100%=completely different)")
        
        # Drift classification
        category = DriftAnomalyDetector.classify_drift(drift['drift_score'])[0]
        print(f"Classification: {category.upper()}")
        
        # Detail changes
        if drift['new_hosts']:
            print(f"\n  ⚠️ NEW HOSTS ({len(drift['new_hosts'])}): {drift['new_hosts'][:3]}...")
        
        if drift['missing_hosts']:
            print(f"  ⚠️ MISSING HOSTS ({len(drift['missing_hosts'])}): {drift['missing_hosts'][:3]}...")
        
        if drift['port_changes']:
            print(f"  ⚠️ PORT CHANGES ({len(drift['port_changes'])} hosts):")
            for ip, changes in list(drift['port_changes'].items())[:2]:
                if changes['new']:
                    print(f"     - {ip}: Opened ports {changes['new']}")
                if changes['closed']:
                    print(f"     - {ip}: Closed ports {changes['closed']}")
        
        if drift['service_changes']:
            print(f"  ⚠️ SERVICE CHANGES ({len(drift['service_changes'])} hosts):")
            for ip, changes in list(drift['service_changes'].items())[:2]:
                if changes['new']:
                    print(f"     - {ip}: New services {changes['new']}")
        
        # Generate anomaly events
        events = DriftAnomalyDetector.generate_anomaly_events(subnet, drift)
        if events:
            print(f"\n  📋 Generated {len(events)} anomaly events")
            for evt in events[:3]:
                print(f"     - [{evt['event_type']}] {evt['host_ip']}: {evt['details']} (severity: {evt['severity']}/5)")
    else:
        print("No previous scan found - this is the first scan")
        print("Next scan will show drift detection")
    
    # Phase 3: Build Real Baseline
    print(f"\n[PHASE 3] Building Real Baseline (from actual observations)")
    print("-" * 100)
    
    builder = RealBaselineBuilder(persistence)
    baselines = builder.build_subnet_baseline(subnet, max_scans=5)
    
    if baselines:
        print(f"Built {len(baselines)} baselines from historical scans\n")
        
        # Show baseline examples (first 3 hosts)
        for ip, baseline in list(sorted(baselines.items()))[:3]:
            print(f"  {ip}:")
            print(f"    - Expected ports: {baseline.expected_open_ports[:5]}{'...' if len(baseline.expected_open_ports) > 5 else ''}")
            print(f"    - Expected services: {baseline.expected_services}")
            print(f"    - OS: {baseline.os_guess}")
            print(f"    - Stability: {baseline.stability_score:.0%} (based on {baseline.scans_analyzed} scans)")
            print(f"    - First seen: {baseline.first_seen}")
            print()
    else:
        print("No historical data yet - baselines will be built after next scan")
    
    # Phase 4: Anomaly Detection Against Baseline
    print(f"\n[PHASE 4] Anomaly Detection (current vs baseline)")
    print("-" * 100)
    
    if baselines:
        anomaly_count = 0
        for ip, host in list(sorted(hosts.items()))[:5]:
            if ip not in baselines:
                print(f"  ⚠️ {ip}: NEW HOST (not in baseline)")
                anomaly_count += 1
                continue
            
            baseline = baselines[ip]
            anomalies = compare_to_baseline(host.__dict__, baseline)
            
            if anomalies['anomaly_score'] > 0.1:  # Threshold: >10% deviation
                print(f"  ⚠️ {ip}: Anomaly score {anomalies['anomaly_score']:.0%}")
                
                if anomalies['port_deviation']:
                    print(f"     - NEW PORTS: {anomalies['port_deviation']}")
                if anomalies['missing_ports']:
                    print(f"     - MISSING PORTS: {anomalies['missing_ports']}")
                if anomalies['service_deviation']:
                    print(f"     - NEW SERVICES: {anomalies['service_deviation']}")
                if anomalies['os_changed']:
                    print(f"     - OS CHANGED: {host.os_guess} (was {baseline.os_guess})")
                
                anomaly_count += 1
        
        if anomaly_count == 0:
            print("✅ No anomalies detected - baseline consistent")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: Step 1 Complete")
    print("=" * 100)
    print(f"✅ Discovered {len(hosts)} hosts")
    print(f"✅ Persisted scan to disk (scans/ directory)")
    if drift['previous_scan_timestamp']:
        print(f"✅ Computed drift: {drift['drift_score']:.0%}")
    if baselines:
        print(f"✅ Built {len(baselines)} real baselines")
    
    print(f"\n[NEXT] Run again to see drift detection and anomalies!")
    print(f"[INFO] Scan history location: scans/{subnet.replace('/', '_').replace('.', '_')}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 1 Demo: Real Scanning + Memory")
    parser.add_argument('--subnet', default='192.168.1.0/24', help='Subnet to scan')
    parser.add_argument('--mode', choices=['auto', 'arp', 'icmp'], default='auto', help='Scan mode')
    args = parser.parse_args()
    
    demo_real_scanning_with_memory(subnet=args.subnet, scan_mode=args.mode)
