#!/usr/bin/env python3
"""
Demo script showcasing the THREE enhancements to the port scanner:
1. Jittered Timing (Timing Evasion) - Random delays to evade detection
2. Comprehensive Banner Grabbing - Protocol-specific version parsing
3. IPv6 Support - CIDR scanning for both IPv4 and IPv6
"""

import logging
from network_propagation_port_scanner import PortScanner, MultiTargetScanner, PortStatus, ServiceType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s'
)
logger = logging.getLogger("EnhancedScannerDemo")


def demo_jittered_timing():
    """
    FEATURE 1: Jittered Timing Evasion
    Demonstrates random delays between port connections to evade timing-based detection
    """
    print("\n" + "="*70)
    print("[1️⃣] FEATURE 1: JITTERED TIMING (Timing Evasion)")
    print("="*70)
    
    logger.info("Creating scanner WITH jitter enabled (default)...")
    scanner_with_jitter = PortScanner(timeout=2.0, max_workers=5, jitter_enabled=True)
    logger.info("✅ Jitter enabled: Each port will have 1-100ms random delay")
    logger.info("   This prevents detection by timing-based intrusion detection systems")
    
    logger.info("\nCreating scanner WITHOUT jitter for comparison...")
    scanner_no_jitter = PortScanner(timeout=2.0, max_workers=5, jitter_enabled=False)
    logger.info("⏱️ Jitter disabled: Direct sequential port scanning")
    
    print("\n[📝] Execution Flow with Jitter Enabled:")
    print("   Port 445: Wait [1.2ms jitter] → Connect → Check response")
    print("   Port 3389: Wait [45.8ms jitter] → Connect → Check response")
    print("   Port 5985: Wait [67.3ms jitter] → Connect → Check response")
    print("   → IDS/firewall sees irregular timing pattern (hard to distinguish from user behavior)")
    
    logger.info("\n[Multi-threaded beacon intervals] Also randomized between targets:")
    logger.info("   Target 1: Scan immediately")
    logger.info("   Target 2: Wait [0.3-2.0s random] → Perform scan")
    logger.info("   Target 3: Wait [0.3-2.0s random] → Perform scan")
    logger.info("   ✅ Result: Scanning appears non-automated, resembles manual reconnaissance")


def demo_banner_grabbing():
    """
    FEATURE 2: Comprehensive Banner Grabbing
    Demonstrates protocol-specific banner parsing for version identification
    """
    print("\n" + "="*70)
    print("[2️⃣] FEATURE 2: COMPREHENSIVE BANNER GRABBING")
    print("="*70)
    
    logger.info("Enhanced banner parsing supports all ServiceType variants:")
    protocols = {
        f"{ServiceType.HTTP.name} (80, 443)": "Parses 'Server' header for web server version (Apache, IIS, nginx)",
        f"{ServiceType.SSH.name} (22)": "Parses SSH version string (RFC 4253) - e.g., OpenSSH_7.4",
        f"{ServiceType.RDP.name} (3389)": "Detects RDP service via X.224 protocol handshake",
        f"{ServiceType.SMB.name} (139, 445)": "Detects SMB service via binary protocol signature",
        f"{ServiceType.SQL.name} (1433, 3306, 5432)": "Identifies SQL Server, MySQL, PostgreSQL versions",
        f"{ServiceType.WINRM.name} (5985, 5986)": "Identifies Windows Remote Management service",
    }
    
    for protocol, parsing_method in protocols.items():
        logger.info(f"   ✅ {protocol:<25} → {parsing_method}")
    
    print("\n[📋] Example Banner Parsing Results:")
    print("   Port 80/HTTP:")
    print("      Raw: 'HTTP/1.1 200 OK\\r\\nServer: Apache/2.4.41\\r\\n'")
    print("      Parsed: product='Apache', version='2.4.41', service='HTTP'")
    print("   ")
    print("   Port 22/SSH:")
    print("      Raw: 'SSH-2.0-OpenSSH_7.4'")
    print("      Parsed: product='OpenSSH_7.4', version='7.4', service='SSH'")
    print("   ")
    print("   Port 1433/MSSQL:")
    print("      Raw: 'Microsoft SQL Server 2019 RTM'")
    print("      Parsed: product='Microsoft SQL Server', version='2019', service='SQL'")
    
    logger.info("\n[🔍] Benefits:")
    logger.info("   ✓ Identify exploitable service versions")
    logger.info("   ✓ Detect known CVEs (e.g., Apache 2.4.0-2.4.20)")
    logger.info("   ✓ Uncover non-standard service ports (e.g., SMB on 8888)")
    logger.info("   ✓ Better target prioritization based on software maturity")


def demo_ipv6_support():
    """
    FEATURE 3: IPv6 Support
    Demonstrates CIDR scanning for both IPv4 and IPv6 networks
    """
    print("\n" + "="*70)
    print("[3️⃣] FEATURE 3: IPv6 SUPPORT (CIDR Scanning)")
    print("="*70)
    
    logger.info("Network scanning now supports IPv6 CIDR notation:")
    
    ipv4_examples = [
        "192.168.1.0/24 → Scans 254 hosts (192.168.1.1 - 192.168.1.254)",
        "10.0.0.0/16 → Scans 65,534 hosts (Class B network)",
        "172.16.0.0/22 → Scans 1,022 hosts (Detailed subnet scan)",
    ]
    
    ipv6_examples = [
        "2001:db8::/32 → Limited to first 50 hosts (2001:db8::1 onwards)",
        "fe80::/64 → Link-local network scanning",
        "2001:db8:cafe::/48 → Organization-scoped network",
    ]
    
    logger.info("\n   IPv4 Examples:")
    for example in ipv4_examples:
        logger.info(f"     ✓ {example}")
    
    logger.info("\n   IPv6 Examples:")
    for example in ipv6_examples:
        logger.info(f"     ✓ {example}")
    
    print("\n[🌐] Usage Examples:")
    print("")
    print("   # IPv4 scanning:")
    print("   scanner = MultiTargetScanner(jitter_enabled=True)")
    print("   results = scanner.scan_range('192.168.1.0/24')")
    print("")
    print("   # IPv6 scanning:")
    print("   results = scanner.scan_range('2001:db8::/48')")
    print("")
    
    logger.info("\n[⚙️] Implementation Details:")
    logger.info("   ✓ Uses Python's 'ipaddress' module for BOTH IPv4 and IPv6")
    logger.info("   ✓ Automatically detects network type (IPv4 vs IPv6)")
    logger.info("   ✓ IPv6 networks capped at 50 hosts (configurable)")
    logger.info("   ✓ Preserves all timing evasion features with IPv6")
    logger.info("   ✓ Same port scanning logic for both protocols")


def demo_integration():
    """
    Show how all three features work together
    """
    print("\n" + "="*70)
    print("[🔗] INTEGRATION: All Features Working Together")
    print("="*70)
    
    logger.info("\nComplete attack workflow with all enhancements:")
    logger.info("")
    logger.info("1. TIMING EVASION + IPv6 SUPPORT:")
    logger.info("   scanner = MultiTargetScanner(jitter_enabled=True)")
    logger.info("   results = scanner.scan_range('2001:db8:cafe::/48')")
    logger.info("   → Scans IPv6 network with randomized timing")
    logger.info("")
    logger.info("2. BANNER PARSING:")
    logger.info("   For each open port, parser identifies:")
    logger.info("   - Service type (HTTP, SSH, RDP, SMB, SQL, etc.)")
    logger.info("   - Product name (Apache, OpenSSH, IIS, etc.)")
    logger.info("   - Version string (7.4, 2.4.41, 2019 RTM, etc.)")
    logger.info("")
    logger.info("3. ATTACK VECTOR SELECTION:")
    logger.info("   - Port 445 + SMB banner detected → Route to WMI exploitation")
    logger.info("   - Port 22 + OpenSSH_7.4 detected → Target SSH key extraction")
    logger.info("   - Port 80 + IIS/10.0 detected → Look for IIS CVEs")
    logger.info("")
    logger.info("✅ Result: Faster, stealthier, more targeted reconnaissance")


def demo_commandline_usage():
    """
    Show command-line usage examples
    """
    print("\n" + "="*70)
    print("[💻] COMMAND-LINE USAGE")
    print("="*70)
    
    print("""
[Single Target - Check PortStatus and Filter by ServiceType]
$ python -c "
from network_propagation_port_scanner import PortScanner, PortStatus, ServiceType
scanner = PortScanner(jitter_enabled=True, max_workers=10)
result = scanner.scan('192.168.1.50')
print(f'Total ports scanned: {result.ports_open}')
for port in result.open_ports:
    if port.status == PortStatus.OPEN:
        print(f'  {port.port}: {port.service.name} ({port.product})')
    if port.service == ServiceType.SMB:
        print(f'    → SMB detected: Priority exploitation target')
"

[IPv4 Network Scan with Timing Evasion]
$ python -c "
from network_propagation_port_scanner import MultiTargetScanner
scanner = MultiTargetScanner(jitter_enabled=True)
results = scanner.scan_range('192.168.1.0/24')
"

[IPv6 Network Scan]
$ python -c "
from network_propagation_port_scanner import MultiTargetScanner
scanner = MultiTargetScanner(jitter_enabled=True)
results = scanner.scan_range('2001:db8::/48')
"

[No Timing Evasion (for internal/trusted networks)]
$ python -c "
scanner = PortScanner(jitter_enabled=False, max_workers=20)
result = scanner.scan('10.0.0.50')
"
    """)


def main():
    print("\n" + "█"*70)
    print("█ ENHANCED PORT SCANNER - THREE MAJOR IMPROVEMENTS")
    print("█"*70)
    
    demo_jittered_timing()
    demo_banner_grabbing()
    demo_ipv6_support()
    demo_integration()
    demo_commandline_usage()
    
    print("\n" + "="*70)
    print("[✅] ALL ENHANCEMENTS READY FOR DEPLOYMENT")
    print("="*70)
    print("""
Summary of Changes:
  1. ✅ Jittered timing evasion (1-100ms per port, 0.3-2.0s between targets)
  2. ✅ Comprehensive banner grabbing (HTTP, SSH, RDP, SMB, SQL, WinRM)
  3. ✅ IPv6 CIDR support (IPv4 AND IPv6 networks)

New Parameters:
  - PortScanner(jitter_enabled=True) - Toggle timing evasion
  - MultiTargetScanner(jitter_enabled=True) - Batch timing evasion
  - scan_range('2001:db8::/48') - IPv6 network support

Next Steps:
  1. Integration with orchestrator (network_propagation_orchestrator.py)
  2. Full propagation chain test (Phase 0 → 1 → 2)
  3. Field deployment on real targets
    """)


if __name__ == "__main__":
    main()
