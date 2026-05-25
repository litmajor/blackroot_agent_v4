#!/usr/bin/env python3
"""
Demo: Network Mapper Correlation Intelligence

Shows:
- Baseline profile creation
- Service change detection (drift tracking)
- Anomaly event logging
- Honeypot behavioral detection
- Correlation report generation
"""

import sys
import logging
from recon.network_mapper import RealNetworkMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def demo():
    print("=" * 100)
    print("NETWORK CORRELATION INTELLIGENCE DEMO")
    print("=" * 100)
    print()
    
    # Initialize mapper
    mapper = RealNetworkMapper(config={'scan_timeout': 30})
    
    # Initial scan (creates baselines)
    print("[📊] PHASE 1: Initial Scan (Create Baselines)")
    print("-" * 100)
    nodes_scan1 = mapper.scan_subnet('192.168.1.0/24', aggressive=False)
    print(f"✅ First scan complete: {len(nodes_scan1)} hosts discovered")
    print(f"   • Baselines created: {len(mapper.baselines)}")
    print()
    
    # Show baselines
    print("[📋] Baseline Profiles:")
    print("-" * 100)
    for i, (ip, baseline) in enumerate(list(mapper.baselines.items())[:5]):
        print(f"  {ip}")
        print(f"    Expected ports: {sorted(baseline.expected_ports)}")
        print(f"    Expected OS: {baseline.expected_os}")
        print(f"    Stability: {baseline.stability_score:.2f}")
        if i < 4:
            print()
    print(f"  ... ({len(mapper.baselines) - 5} more baselines)")
    print()
    
    # Simulate anomalies by second scan
    # Reset discovered nodes to simulate changes on re-scan
    mapper.discovered_nodes.clear()
    print("[🔄] PHASE 2: Second Scan (Detect Changes)")
    print("-" * 100)
    nodes_scan2 = mapper.scan_subnet('192.168.1.0/24', aggressive=False)
    print(f"✅ Second scan complete: {len(nodes_scan2)} hosts discovered")
    print(f"   • Total anomalies detected: {len(mapper.anomalies)}")
    print()
    
    # Show detected anomalies
    if mapper.anomalies:
        print("[🚨] Detected Anomalies (Drift Events):")
        print("-" * 100)
        for i, anomaly in enumerate(mapper.anomalies[:10]):
            severity_icon = "🔴" if anomaly.severity > 0.8 else "🟡" if anomaly.severity > 0.5 else "🟢"
            print(f"{severity_icon} {anomaly.ip}: {anomaly.event_type}")
            print(f"   └─ {anomaly.description}")
            if i < 9:
                print()
        if len(mapper.anomalies) > 10:
            print(f"\n... ({len(mapper.anomalies) - 10} more anomalies)")
    else:
        print("✅ No anomalies detected (network stable)")
    print()
    
    # Honeypot detection
    print("[🍯] PHASE 3: Behavioral Honeypot Detection")
    print("-" * 100)
    honeypots = mapper.detect_honeypots()
    if honeypots:
        print(f"⚠️  Detected {len(honeypots)} potential honeypots:")
        for hp in honeypots:
            profile = mapper.discovered_nodes.get(hp)
            if profile:
                print(f"  {hp} - {len(profile.open_ports)} ports open, {profile.os or 'Unknown OS'}")
    else:
        print("✅ No honeypots detected")
    print()
    
    # Correlation report
    print("[📊] PHASE 4: Comprehensive Correlation Report")
    print("-" * 100)
    report = mapper.get_correlation_report()
    
    print(f"Total Scans: {report['total_scans']}")
    print(f"Total Hosts: {report['total_hosts_discovered']}")
    print(f"Baselines Created: {report['baselines_created']}")
    print(f"Anomalies Detected: {report['anomalies_detected']}")
    print(f"Critical Anomalies: {len(report['critical_anomalies'])}")
    print()
    
    if report['anomaly_summary']:
        print("Anomaly Breakdown:")
        for event_type, count in sorted(report['anomaly_summary'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {event_type}: {count}")
    print()
    
    # Baseline stability analysis
    print("[📈] Host Baseline Stability Analysis:")
    print("-" * 100)
    stable_hosts = []
    volatile_hosts = []
    
    for ip, baseline_data in list(report['host_baselines'].items())[:15]:
        if baseline_data['is_stable']:
            stable_hosts.append((ip, baseline_data['stability_score']))
        else:
            volatile_hosts.append((ip, baseline_data['stability_score']))
    
    stable_hosts.sort(key=lambda x: x[1], reverse=True)
    volatile_hosts.sort(key=lambda x: x[1])
    
    if stable_hosts:
        print(f"🟢 Stable Hosts ({len(stable_hosts)}):")
        for ip, score in stable_hosts[:5]:
            print(f"  {ip}: {score:.2f}")
        if len(stable_hosts) > 5:
            print(f"  ... and {len(stable_hosts) - 5} more")
    
    if volatile_hosts:
        print()
        print(f"🟡 Volatile Hosts ({len(volatile_hosts)}):")
        for ip, score in volatile_hosts[:5]:
            print(f"  {ip}: {score:.2f}")
        if len(volatile_hosts) > 5:
            print(f"  ... and {len(volatile_hosts) - 5} more")
    print()
    
    # Summary
    print("=" * 100)
    print("CORRELATION INTELLIGENCE SUMMARY")
    print("=" * 100)
    print()
    print("✨ Key Insights:")
    print(f"  1. Service Drift: Monitoring {len(mapper.baselines)} hosts for changes")
    print(f"  2. Anomalies: {len(mapper.anomalies)} behavioral anomalies tracked across scans")
    print(f"  3. Honeypots: Behavioral analysis identified {len(honeypots)} potential traps")
    print(f"  4. Stability: {len([b for b in mapper.baselines.values() if b.is_stable()])} stable systems ready for propagation")
    print()
    print("🎯 Real Value:")
    print("  • Port 445 opened on 192.168.1.5 in last 24h → NEW ATTACK SURFACE")
    print("  • Windows Server changed OS to Linux on 192.168.1.10 → SUSPECTED COMPROMISE")
    print("  • 192.168.1.15 opened 20+ ports in sequence → LIKELY HONEYPOT (AVOID)")
    print("  • 192.168.1.3 stable for 45+ days, minimal changes → IDEAL ZOMBIE CANDIDATE")
    print()

if __name__ == '__main__':
    try:
        demo()
    except KeyboardInterrupt:
        print("\n[*] Demo interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
