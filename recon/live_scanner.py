#!/usr/bin/env python3
"""
Real Network Scanner using Scapy - Hybrid LAN/WAN scanning.

Provides:
- ARP discovery for LAN (fast, accurate, LAN-only)
- ICMP + TCP for WAN (works anywhere, less accurate)
- Banner grabbing for service detection (not just port status)
- TCP/IP fingerprinting for OS hints (TTL + response patterns)
- Concurrent port scanning (threaded, not sequential)
- Honest uncertainty reporting (firewalls affect results)

Limitations:
- OS detection is a HINT, not definitive (TTL easily spoofed)
- Firewalls cause false negatives (missing open ports)
- Rate limiting may cause inconsistencies
- WAN scanning is slow and may timeout
"""

import scapy.all as scapy
import socket
import logging
import threading
import queue
import time
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import ipaddress

# Optional persistence support
try:
    from recon.scan_persistence import ScanPersistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    ScanPersistence = None

logger = logging.getLogger(__name__)


@dataclass
class ServiceBanner:
    """Service information from banner grabbing."""
    port: int
    service_name: str = "unknown"
    version: Optional[str] = None
    product: Optional[str] = None
    confidence: float = 0.0  # 0-1: how confident is this detection
    raw_banner: Optional[str] = None


@dataclass
class DiscoveredHost:
    """Actual discovered host from real network scan."""
    ip: str
    mac: Optional[str] = None
    hostname: Optional[str] = None
    open_ports: List[int] = field(default_factory=list)
    closed_ports: List[int] = field(default_factory=list)
    filtered_ports: List[int] = field(default_factory=list)  # Firewall likely blocking
    os_guess: Optional[str] = None
    os_confidence: float = 0.0  # 0-1: confidence in OS detection
    ttl_value: Optional[int] = None
    response_time_ms: float = 0.0
    is_responsive: bool = False
    banners: Dict[int, ServiceBanner] = field(default_factory=dict)  # port -> service info
    
    @property
    def total_ports_scanned(self) -> int:
        return len(self.open_ports) + len(self.closed_ports) + len(self.filtered_ports)


class ServiceDetector:
    """Banner grabbing and service detection."""
    
    @staticmethod
    def grab_banner(ip: str, port: int, timeout: int = 2) -> Optional[str]:
        """
        Attempt to grab service banner from open port.
        
        Returns:
            Raw banner string or None if grab fails
        """
        try:
            socket.setdefaulttimeout(timeout)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            
            # Send common probes
            probes = [
                b"",  # Just connect (some services send banner immediately)
                b"\r\n",  # HTTP
                b"HELP\r\n",  # Some text protocols
            ]
            
            banner = None
            for probe in probes:
                try:
                    if probe:
                        sock.send(probe)
                    data = sock.recv(1024)
                    if data:
                        banner = data.decode('utf-8', errors='ignore').strip()
                        break
                except:
                    pass
            
            sock.close()
            return banner
            
        except (socket.timeout, socket.error, ConnectionRefusedError):
            return None
        except Exception as e:
            logger.debug(f"Banner grab error on {ip}:{port}: {e}")
            return None
    
    @staticmethod
    def identify_service(port: int, banner: Optional[str]) -> Tuple[str, Optional[str], float]:
        """
        Identify service from port number and banner.
        
        Returns:
            (service_name, version, confidence: 0-1)
        """
        service_map = {
            21: ("ftp", "FTP Server"),
            22: ("ssh", "SSH/Secure Shell"),
            25: ("smtp", "SMTP Mail"),
            53: ("dns", "DNS"),
            80: ("http", "HTTP Server"),
            110: ("pop3", "POP3 Mail"),
            143: ("imap", "IMAP Mail"),
            443: ("https", "HTTPS Server"),
            445: ("smb", "SMB/CIFS"),
            3306: ("mysql", "MySQL Database"),
            3389: ("rdp", "Remote Desktop"),
            5432: ("postgresql", "PostgreSQL Database"),
            5985: ("winrm", "Windows Remote Management"),
            8080: ("http", "HTTP Alt"),
        }
        
        service_name, product = service_map.get(port, ("unknown", None))
        version = None
        confidence = 0.5  # Port-based detection is 50% confident
        
        # Improve confidence with banner
        if banner:
            banner_lower = banner.lower()
            confidence = 0.8  # Banner-based detection is 80% confident
            
            # Extract version from common patterns
            if "openssh" in banner_lower:
                service_name = "ssh"
                import re
                match = re.search(r'openssh[_\s]+([0-9.]+)', banner_lower)
                version = match.group(1) if match else None
            elif "apache" in banner_lower:
                service_name = "http"
                import re
                match = re.search(r'apache[/\s]+([0-9.]+)', banner_lower)
                version = match.group(1) if match else None
            elif "mysql" in banner_lower:
                service_name = "mysql"
                import re
                match = re.search(r'mysql[/\s]+([0-9.]+)', banner_lower)
                version = match.group(1) if match else None
        
        return service_name, version, confidence


class LiveNetworkScanner:
    """
    Performs real network scanning (both LAN and WAN).
    
    Modes:
    - LAN: Fast ARP discovery + port scanning
    - WAN: ICMP/TCP discovery + concurrent port scanning
    """
    
    def __init__(self, max_threads: int = 20):
        self.discovered_hosts: Dict[str, DiscoveredHost] = {}
        self.scan_history: List[Dict] = []
        self.max_threads = max_threads
        self.detector = ServiceDetector()
        
    def discover_hosts_arp(self, subnet: str, timeout: int = 2) -> Dict[str, DiscoveredHost]:
        """
        Real ARP discovery - LAN only, but fast and accurate.
        """
        logger.info(f"[🔍] ARP discovery on {subnet} (LAN only)")
        results = {}
        
        try:
            arp_request = scapy.ARP(pdst=subnet)
            ether = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = ether / arp_request
            
            logger.info(f"[→] Broadcasting ARP to {subnet}")
            answered, unanswered = scapy.srp(
                packet,
                timeout=timeout,
                verbose=False,
                iface=None
            )
            
            for sent, received in answered:
                ip = received.psrc
                mac = received.hwsrc
                host = DiscoveredHost(ip=ip, mac=mac, is_responsive=True)
                results[ip] = host
                logger.info(f"[✓] ARP response: {ip} ({mac})")
            
            logger.info(f"[✅] ARP complete: {len(results)} hosts on LAN")
            return results
            
        except PermissionError:
            logger.error("[!] ARP requires Admin privileges")
            return {}
        except Exception as e:
            logger.error(f"[!] ARP failed: {e}")
            return {}
    
    def discover_hosts_icmp(self, subnet: str, timeout: int = 1) -> Dict[str, DiscoveredHost]:
        """
        ICMP ping sweep - works on WAN, less reliable.
        """
        logger.info(f"[🔍] ICMP ping sweep on {subnet} (WAN-capable)")
        results = {}
        
        try:
            network = ipaddress.IPv4Network(subnet, strict=False)
            ips = [str(ip) for ip in network if ip != network.network_address and ip != network.broadcast_address]
            
            logger.info(f"[→] ICMP pinging {len(ips)} addresses")
            
            for target_ip in ips:
                try:
                    icmp = scapy.IP(dst=target_ip) / scapy.ICMP()
                    start = datetime.now()
                    response = scapy.sr1(icmp, timeout=timeout, verbose=0)
                    elapsed_ms = (datetime.now() - start).total_seconds() * 1000
                    
                    if response:
                        ttl = response.ttl
                        host = DiscoveredHost(
                            ip=target_ip,
                            is_responsive=True,
                            ttl_value=ttl,
                            response_time_ms=elapsed_ms
                        )
                        results[target_ip] = host
                        logger.info(f"[✓] Ping reply: {target_ip} (TTL={ttl})")
                except Exception as e:
                    logger.debug(f"[·] {target_ip}: no ICMP response")
            
            logger.info(f"[✅] ICMP ping complete: {len(results)}/{len(ips)} hosts up")
            return results
            
        except Exception as e:
            logger.error(f"[!] ICMP scan failed: {e}")
            return {}
    
    def scan_ports_tcp_concurrent(
        self,
        target_ip: str,
        ports: Optional[List[int]] = None,
        timeout: int = 2,
        grab_banners: bool = True,
        retries: int = 2
    ) -> Tuple[List[int], List[int], List[int], Dict[int, ServiceBanner]]:
        """
        Concurrent TCP port scanning with threading and retry logic.
        
        Args:
            target_ip: Target IP to scan
            ports: List of ports to scan
            timeout: Response timeout per port
            grab_banners: Attempt service banner grabbing
            retries: Retry attempts per port (majority vote on result)
            
        Returns:
            (open_ports, closed_ports, filtered_ports, banners_dict)
        """
        if ports is None:
            ports = [21, 22, 25, 53, 80, 110, 143, 443, 445, 3306, 3389, 5432, 5985, 8080]
        
        open_ports = []
        closed_ports = []
        filtered_ports = []
        banners_dict = {}
        port_results = {}  # Track multiple scans per port for majority vote
        
        # Thread-safe queues
        port_queue = queue.Queue()
        results_queue = queue.Queue()
        
        # Add ports with retries
        for port in ports:
            for attempt in range(retries):
                port_queue.put((port, attempt))
        
        def worker():
            """Worker thread for port scanning with rate limiting."""
            while True:
                try:
                    port, attempt = port_queue.get_nowait()
                except queue.Empty:
                    break
                
                # Rate limiting: small delay between attempts per thread
                if attempt > 0:
                    time.sleep(0.1)
                
                try:
                    # SYN packet
                    syn_packet = scapy.IP(dst=target_ip) / scapy.TCP(dport=port, flags="S", sport=scapy.RandShort())
                    response = scapy.sr1(syn_packet, timeout=timeout, verbose=0)
                    
                    if response:
                        if response.haslayer(scapy.TCP):
                            tcp_layer = response[scapy.TCP]
                            
                            if tcp_layer.flags & 0x12:  # SYN-ACK = OPEN
                                results_queue.put(("open", port, attempt))
                                # Try banner grab on first attempt only
                                if grab_banners and attempt == 0:
                                    banner = self.detector.grab_banner(target_ip, port, timeout=timeout)
                                    if banner:
                                        service_name, version, conf = self.detector.identify_service(port, banner)
                                        results_queue.put(("banner", port, service_name, version, conf, banner))
                                        logger.info(f"[✓] {target_ip}:{port} OPEN - {service_name} (v{version or '?'})")
                                    else:
                                        logger.debug(f"[✓] {target_ip}:{port} OPEN (no banner)")
                                # Send RST to close
                                try:
                                    scapy.send(scapy.IP(dst=target_ip) / scapy.TCP(dport=port, flags="R"), verbose=0)
                                except:
                                    pass
                            
                            elif tcp_layer.flags & 0x14:  # RST = CLOSED
                                results_queue.put(("closed", port, attempt))
                    else:
                        # No response = likely filtered (firewall)
                        results_queue.put(("filtered", port, attempt))
                        logger.debug(f"[·] {target_ip}:{port} filtered/no-response")
                        
                except Exception as e:
                    results_queue.put(("error", port, attempt))
                    logger.debug(f"[!] Scan error {target_ip}:{port}: {e}")
        
        # Start threads
        threads = []
        logger.info(f"[🔍] TCP scanning {target_ip} ({len(ports)} ports, {self.max_threads} threads, {retries} retries)")
        for _ in range(min(self.max_threads, len(ports) * retries)):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join(timeout=30)
        
        # Collect results with majority voting
        port_results = {}
        while not results_queue.empty():
            result = results_queue.get()
            
            if result[0] == "banner":
                _, port, svc, ver, conf, raw = result
                banners_dict[port] = ServiceBanner(
                    port=port,
                    service_name=svc,
                    version=ver,
                    confidence=conf,
                    raw_banner=raw
                )
            elif result[0] in ("open", "closed", "filtered", "error"):
                status, port, attempt = result
                if port not in port_results:
                    port_results[port] = []
                port_results[port].append(status)
        
        # Majority vote on results
        for port, statuses in port_results.items():
            # Count occurrences
            from collections import Counter
            counts = Counter(statuses)
            final_status = counts.most_common(1)[0][0]
            
            if final_status == "open":
                open_ports.append(port)
            elif final_status == "closed":
                closed_ports.append(port)
            else:  # filtered or error
                filtered_ports.append(port)
        
        logger.info(f"[✅] TCP scan {target_ip}: {len(open_ports)} open, {len(closed_ports)} closed, {len(filtered_ports)} filtered/timeout")
        return open_ports, closed_ports, filtered_ports, banners_dict
    
    def guess_os_fingerprint(self, ttl: Optional[int], response_time_ms: float) -> Tuple[Optional[str], float]:
        """
        OS detection from TCP/IP fingerprinting (TTL + response time).
        
        Returns:
            (os_guess, confidence: 0-1)
            
        Confidence is low (0.3-0.6) because TTL can be spoofed/modified.
        """
        os_guess = None
        confidence = 0.0
        
        if ttl is None:
            return None, 0.0
        
        # TTL analysis (very weak fingerprinting)
        if ttl >= 120:
            os_guess = "Windows (TTL-based hint)"
            confidence = 0.3  # Very uncertain
        elif ttl >= 60:
            os_guess = "Linux/Unix (TTL-based hint)"
            confidence = 0.3
        else:
            os_guess = "Other/Modified TTL"
            confidence = 0.1
        
        # Confidence note
        logger.warning(f"[⚠️] OS detection is weak: {os_guess} (confidence={confidence:.1f}, TTL easily spoofed)")
        
        return os_guess, confidence
    
    def scan_network(
        self,
        subnet: str,
        scan_mode: str = "auto",
        grab_banners: bool = True,
        timeout: int = 2,
        persist: bool = True
    ) -> Dict[str, DiscoveredHost]:
        """
        Scan network using appropriate method.
        
        Args:
            subnet: Target subnet
            scan_mode: 'auto' (try ARP, fall back to ICMP), 'arp' (LAN only), 'icmp' (WAN)
            grab_banners: Attempt service banner grabbing
            timeout: Response timeout
            persist: Save scan results to disk (for drift detection)
            
        Returns:
            Dict of discovered hosts with services and OS hints
        """
        scan_start = time.time()
        logger.info(f"[🔍] === NETWORK SCAN: {subnet} (mode={scan_mode}) ===")
        
        discovered = {}
        
        # Discovery phase
        if scan_mode in ("auto", "arp"):
            try:
                discovered = self.discover_hosts_arp(subnet, timeout=timeout)
            except Exception as e:
                logger.warning(f"ARP failed: {e}")
                if scan_mode == "arp":
                    return discovered
        
        if not discovered and scan_mode in ("auto", "icmp"):
            discovered = self.discover_hosts_icmp(subnet, timeout=timeout)
        
        logger.info(f"[📊] Discovery phase: {len(discovered)} hosts found, starting port scans")
        
        # Port scanning + service detection
        for ip, host in list(discovered.items()):
            try:
                open_ports, closed_ports, filtered_ports, banners = self.scan_ports_tcp_concurrent(
                    ip,
                    timeout=timeout,
                    grab_banners=grab_banners,
                    retries=2
                )
                
                host.open_ports = open_ports
                host.closed_ports = closed_ports
                host.filtered_ports = filtered_ports
                host.banners = banners  # Store discovered banners
                
                # OS fingerprinting
                if host.ttl_value:
                    os_guess, os_conf = self.guess_os_fingerprint(host.ttl_value, host.response_time_ms)
                    host.os_guess = os_guess
                    host.os_confidence = os_conf
                
                # For LAN scanning with ARP, we know it's a real host
                if host.mac:
                    logger.info(f"[✓] {ip} ({host.mac}): {len(open_ports)} open, {len(filtered_ports)} filtered")
                else:
                    logger.info(f"[✓] {ip}: {len(open_ports)} open, {len(filtered_ports)} filtered (may be false hits)")
                    
            except Exception as e:
                logger.error(f"[!] Port scan failed for {ip}: {e}")
        
        # Record scan
        self.scan_history.append({
            "timestamp": datetime.now().isoformat(),
            "subnet": subnet,
            "scan_mode": scan_mode,
            "hosts_found": len(discovered),
            "scan_type": "real_hybrid_scan",
            "hosts": {
                ip: {
                    "mac": host.mac,
                    "open_ports": host.open_ports,
                    "closed_ports": host.closed_ports,
                    "filtered_ports": host.filtered_ports,
                    "os_guess": host.os_guess,
                    "os_confidence": host.os_confidence,
                    "ttl": host.ttl_value,
                    "response_time_ms": host.response_time_ms,
                    "total_scanned": host.total_ports_scanned
                }
                for ip, host in discovered.items()
            }
        })
        
        self.discovered_hosts.update(discovered)
        
        scan_duration = time.time() - scan_start
        logger.info(f"[✅] Network scan complete: {len(discovered)} hosts with {sum(len(h.open_ports) for h in discovered.values())} total services (took {scan_duration:.1f}s)")
        
        # Persist scan results for drift detection
        if persist and PERSISTENCE_AVAILABLE:
            try:
                persistence = ScanPersistence()
                filepath = persistence.save_scan(subnet, discovered, scan_duration)
                logger.info(f"[💾] Scan persisted to {filepath}")
            except Exception as e:
                logger.warning(f"[⚠️] Could not persist scan: {e}")
        
        return discovered


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    subnet = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.0/24"
    mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
    
    try:
        scanner = LiveNetworkScanner(max_threads=20)
        hosts = scanner.scan_network(subnet, scan_mode=mode, grab_banners=True)
        
        print("\n" + "=" * 100)
        print(f"RESULTS: Real Network Scan ({mode.upper()} mode)")
        print("=" * 100)
        print()
        
        if not hosts:
            print("[!] No hosts discovered")
            sys.exit(0)
        
        for ip, host in sorted(hosts.items()):
            print(f"[HOST] {ip}")
            
            if host.mac:
                print(f"  ✓ MAC: {host.mac} (verified via ARP)")
            else:
                print(f"  ? No MAC (WAN/unreachable)")
            
            if host.os_guess:
                print(f"  ⚠️ OS: {host.os_guess} (confidence: {host.os_confidence:.0%})")
            
            print(f"  Ports scanned: {host.total_ports_scanned}")
            
            if host.open_ports:
                print(f"    • OPEN ({len(host.open_ports)}): {sorted(host.open_ports)}")
                for port in sorted(host.open_ports):
                    if port in host.banners:
                        banner = host.banners[port]
                        print(f"        {port}: {banner.service_name} (v{banner.version or '?'}, confidence: {banner.confidence:.0%})")
                        if banner.raw_banner and len(banner.raw_banner) < 80:
                            print(f"             Banner: {banner.raw_banner}")
            
            if host.closed_ports:
                summary = f"{sorted(host.closed_ports)[:5]}"
                if len(host.closed_ports) > 5:
                    summary += f"... (+{len(host.closed_ports)-5} more)"
                print(f"    • CLOSED ({len(host.closed_ports)}): {summary}")
            
            if host.filtered_ports:
                print(f"    • FILTERED ({len(host.filtered_ports)}): {sorted(host.filtered_ports)}")
                print(f"      ⚠️ Firewall likely blocking → False negatives possible")
            
            print()
        
        # Summary
        print("=" * 100)
        total_hosts = len(hosts)
        total_open = sum(len(h.open_ports) for h in hosts.values())
        total_filtered = sum(len(h.filtered_ports) for h in hosts.values())
        
        print(f"SUMMARY: {total_hosts} hosts, {total_open} open ports, {total_filtered} filtered ports")
        if total_filtered > 0:
            print(f"⚠️ Firewall detected on multiple hosts - real open port count may be higher")
        
    except PermissionError:
        print("[!] ERROR: Administrator privileges required for raw socket access")
        print("    Run this script as Administrator")
        sys.exit(1)
    except Exception as e:
        logging.error(f"[!] Scan failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
   