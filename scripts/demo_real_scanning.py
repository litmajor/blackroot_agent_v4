#!/usr/bin/env python3
"""
Demo: Real Network Scanning (with honest limitations)

Shows:
- Actual network traffic (ARP/ICMP/TCP)
- Service detection from banners (not just port guessing)
- OS detection confidence (honest: TTL is weak signal)
- Firewall/filtering effects (false negatives)
- Concurrent scanning (faster than sequential)
"""

import sys
import logging
from recon.live_scanner import LiveNetworkScanner

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def demo_real_scan():
    print("=" * 100)
    print("REAL NETWORK SCANNING DEMO (No Simulation)")
    print("=" * 100)
    print()
    
    # Demo parameters
    subnet = "192.168.1.0/24"  # Change to test other subnets
    
    print("[📝] Test Parameters:")
    print(f"  Subnet: {subnet}")
    print(f"  Mode: Auto-detect (ARP for private, ICMP for public)")
    print(f"  Concurrency: 20 threads for port scanning")
    print(f"  Service Detection: Banner grabbing enabled")
    print()
    
    print("[⚠️ ] Limitations (Honest):")
    print("  1. OS detection is WEAK (TTL easily spoofed, modified by hops)")
    print("  2. Firewalls cause FALSE NEGATIVES (we miss ports)")
    print("  3. Rate limiting affects consistency")
    print("  4. Banner grabbing may fail on secured services")
    print("  5. This is a LAN scanner → WAN scanning is slow")
    print()
    
    try:
        scanner = LiveNetworkScanner(max_threads=20)
        
        print("[🔍] Starting REAL network scan...")
        print()
        
        # Determine scan mode
        import ipaddress
        is_private = ipaddress.IPv4Network(subnet).is_private
        mode = "arp" if is_private else "icmp"
        
        print(f"[→] Using {mode.upper()} discovery for this subnet")
        if mode == "arp":
            print("    (Fast, accurate for local network, requires raw socket access)")
        else:
            print("    (Works on WAN but slower, less reliable)")
        print()
        
        hosts = scanner.scan_network(
            subnet,
            scan_mode=mode,
            grab_banners=True,
            timeout=5
        )
        
        print()
        print("=" * 100)
        print(f"RESULTS: {len(hosts)} hosts discovered and scanned")
        print("=" * 100)
        print()
        
        if not hosts:
            print("[!] No hosts found. Possible reasons:")
            print("    - No hosts on this subnet")
            - ARP requires Administrator privileges")
            print("    - Network interface not properly configured")
            print()
            return
        
        # Display results
        for ip, host in sorted(hosts.items()):
            print(f"[HOST] {ip}")
            
            # MAC address (ARP verification)
            if host.mac:
                print(f"  ✓ MAC: {host.mac} (verified via ARP - definitely on this LAN)")
            else:
                print(f"  ? No MAC (WAN host - may be unreachable)")
            
            # TTL-based OS hint (weak!)
            if host.os_guess:
                print(f"  ⚠️ OS Hint: {host.os_guess}")
                print(f"     Confidence: {host.os_confidence:.0%} (TTL is easily spoofed)")
            
            # Port analysis
            total = len(host.open_ports) + len(host.closed_ports) + len(host.filtered_ports)
            print(f"  TCP Ports Scanned: {total}")
            
            if host.open_ports:
                print(f"    • OPEN ({len(host.open_ports)}): {host.open_ports}")
                
                # Show services with banners
                for port in host.open_ports:
                    if port in host.banners:
                        banner = host.banners[port]
                        print(f"        - {port}: {banner.service_name} (v{banner.version or '?'}, banner confidence: {banner.confidence:.0%})")
                        if banner.raw_banner:
                            print(f"          Banner: {banner.raw_banner[:60]}...")
            
            if host.closed_ports:
                print(f"    • CLOSED ({len(host.closed_ports)}): {host.closed_ports[:5]}..." if len(host.closed_ports) > 5 else f"    • CLOSED ({len(host.closed_ports)}): {host.closed_ports}")
            
            if host.filtered_ports:
                print(f"    • FILTERED ({len(host.filtered_ports)}): {host.filtered_ports}")
                print(f"      ⚠️ Firewall likely blocking these ports → False negatives possible")
            
            print()
        
        # Summary
        print("=" * 100)
        print("KEY INSIGHTS")
        print("=" * 100)
        print()
        
        total_open = sum(len(h.open_ports) for h in hosts.values())
        total_filtered = sum(len(h.filtered_ports) for h in hosts.values())
        
        print(f"Found {len(hosts)} hosts with {total_open} open ports")
        
        if total_filtered > 0:
            print(f"⚠️ WARNING: {total_filtered} ports are filtered (likely false negatives)")
            print("   Real open ports could be higher due to firewall blocking")
        
        # OS detection note
        os_guesses = [h for h in hosts.values() if h.os_guess and h.os_confidence > 0.3]
        if os_guesses:
            print(f"\n⚠️ OS DETECTION IS WEAK: Only using TTL (confidence < 50%)")
            print("   Don't trust OS detection for targeting decisions")
        
        # Service banners note
        services_with_banners = sum(len(h.banners) for h in hosts.values())
        if services_with_banners > 0:
            print(f"\n✓ Banner grabbing successful on {services_with_banners} services")
            print("  This is more reliable than port-based guessing")
        else:
            print("\n? No service banners grabbed")
            print("  Services may be secured (TLS, auth, etc.)")
        
        print()
        print("=" * 100)
        print("REAL DATA CHARACTERISTICS")
        print("=" * 100)
        print()
        print("This is REAL network data (not mocked), but:")
        print("  ✓ Actual network traffic sent over the wire")
        print("  ✓ Real responses from actual devices")
        print("  ✓ Service detection from actual banners")
        print("  ✗ OS detection is crude (TTL only)")
        print("  ✗ Firewalls hide reality (false negatives)")
        print("  ✗ Rate limiting may cause inconsistency")
        print()
        
    except PermissionError:
        print("[!] ERROR: Administrator privileges required")
        print()
        print("Run this script as Administrator for ARP scanning:")
        print("  - Windows: Run Command Prompt as Administrator")
        print("  - Linux/Mac: Use 'sudo python3 demo_real_scanning.py'")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"[!] Scan failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    demo_real_scan()
