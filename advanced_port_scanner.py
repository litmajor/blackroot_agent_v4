#!/usr/bin/env python3
"""
Advanced Port Scanner Methods - Enhanced Detection
Provides: SMB negotiation, Dynamic service probes, SYN stealth scanning

Import this module's methods into network_propagation_port_scanner.py for advanced features
"""

import socket
import logging
import time
import random
import struct
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger("AdvancedScan")

# Try imports for advanced features
try:
    from impacket.smbconnection import SMBConnection
    SMB_AVAILABLE = True
except ImportError:
    SMB_AVAILABLE = False
    logger.debug("SMB negotiation unavailable (impacket not found)")

try:
    from scapy.all import IP, TCP, sr1  # type: ignore
    SCAPY_AVAILABLE = True
except (ImportError, AttributeError):
    SCAPY_AVAILABLE = False
    logger.debug("SYN stealth unavailable (scapy not found)")


class AdvancedPortScannerMethods:
    """
    Advanced scanning methods to add to PortScanner class
    Handles: SMB negotiation, dynamic probes, SYN stealth
    """
    
    # Service detection probes (protocol-specific)
    SERVICE_PROBES = {
        'http': b'HEAD / HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n',
        'ssh': b'SSH-2.0-advanced-scanner\r\n',
        'smb': None,  # SMB handled separately
        'rdp': None,  # RDP handled separately
    }
    
    @staticmethod
    def _is_root() -> bool:
        """Check if running with admin/root privileges"""
        try:
            import os
            return os.geteuid() == 0 if hasattr(os, 'geteuid') else False  # type: ignore
        except (AttributeError, OSError):
            try:
                import ctypes
                return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore
            except:
                return False
    
    @staticmethod
    def _check_port_syn_stealth(target_ip: str, port: int, timeout: float = 2.0) -> Optional[Dict]:
        """
        SYN stealth scan (half-open connection)
        Requires: admin/root + scapy
        Returns: {'port': port, 'open': True/False, 'response_time': ms}
        """
        if not SCAPY_AVAILABLE:
            return None
        
        try:
            from scapy.all import IP, TCP, sr1  # type: ignore
            start_time = time.time()
            
            # Craft SYN packet with random ephemeral port
            sport = random.randint(49152, 65535)
            packet = IP(dst=target_ip) / TCP(sport=sport, dport=port, flags="S")
            
            # Send and listen for response
            response = sr1(packet, timeout=timeout, verbose=False)
            response_time = (time.time() - start_time) * 1000
            
            if response and response.haslayer(TCP):
                tcp_layer = response[TCP]
                # Check for SYN-ACK (0x12 = SYN | ACK)
                if tcp_layer.flags & 0x12:
                    # Port is open - send RST to close cleanly
                    try:
                        rst = IP(dst=target_ip) / TCP(sport=sport, dport=port, flags="R", seq=tcp_layer.ack)
                        sr1(rst, timeout=0.5, verbose=False)
                    except:
                        pass
                    
                    return {'port': port, 'open': True, 'response_time': response_time, 'method': 'SYN_STEALTH'}
                # RST response = port closed
                elif tcp_layer.flags & 0x04:  # RST flag
                    return {'port': port, 'open': False, 'response_time': response_time, 'method': 'SYN_STEALTH'}
            
            return {'port': port, 'open': False, 'response_time': response_time, 'method': 'SYN_STEALTH'}
        
        except Exception as e:
            logger.debug(f"SYN stealth failed: {e}")
            return None
    
    @staticmethod
    def _negotiate_smb_dialect(target_ip: str, port: int = 445, timeout: float = 2.0) -> Optional[str]:
        """
        Perform REAL SMB negotiation to detect dialect
        Returns: 'SMB1', 'SMB2', 'SMB3', or None
        Requires: impacket library
        """
        if not SMB_AVAILABLE or port != 445:
            return None
        
        try:
            if not SMB_AVAILABLE:
                return None
            conn = SMBConnection(target_ip, target_ip, sess_port=port, timeout=int(timeout))  # type: ignore
            
            # Get dialect version
            if hasattr(conn, 'getDialect'):
                dialect_code = conn.getDialect()
                if dialect_code == 1:
                    return "SMB1"
                elif dialect_code == 2:
                    return "SMB2"
                elif dialect_code in (3, 0x0202, 0x0210, 0x0300):
                    return "SMB3"
            
            conn.close()
        except Exception as e:
            logger.debug(f"SMB negotiation error: {e}")
        
        return None
    
    @staticmethod
    def _detect_service_by_probe(target_ip: str, port: int, timeout: float = 1.0) -> Optional[Tuple[str, str, str]]:
        """
        Send protocol-specific probes to detect service dynamically
        Bypasses port → service assumptions
        
        Returns: (service_name, product_name, version)
        """
        probes = [
            ('ssh', b'SSH-2.0-advanced-scanner\r\n', b'SSH-'),
            ('http', b'HEAD / HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n', b'HTTP/'),
            ('https_tls', b'\\x16\\x03\\x01\\x00K\\x01\\x00\\x00G\\x03\\x03', b'\\x16\\x03'),
        ]
        
        for probe_name, probe_bytes, expected_prefix in probes:
            try:
                sock = socket.create_connection((target_ip, port), timeout=timeout)
                sock.send(probe_bytes)
                response = sock.recv(2048)
                sock.close()
                
                if expected_prefix in response:
                    # Service detected!
                    if probe_name == 'ssh':
                        # Parse SSH banner
                        banner_line = response.decode('utf-8', errors='ignore').split('\n')[0]
                        if 'OpenSSH' in banner_line:
                            match = re.search(r'openssh[_\\s]+([0-9.]+)', banner_line, re.I)
                            version = match.group(1) if match else 'unknown'
                            return ('SSH', 'OpenSSH', version)
                    
                    elif probe_name == 'http':
                        # Parse HTTP response
                        resp_str = response.decode('utf-8', errors='ignore')
                        for line in resp_str.split('\\n'):
                            if line.lower().startswith('server:'):
                                server_header = line.split(':', 1)[1].strip()
                                # Parse version
                                match = re.search(r'([0-9]+\\.[0-9]+(?:\\.[0-9]+)?)', server_header)
                                version = match.group(1) if match else 'unknown'
                                return ('HTTP', server_header.split('/')[0], version)
                        return ('HTTP', 'Unknown-Web-Server', 'unknown')
                    
                    elif probe_name == 'https_tls':
                        return ('HTTPS', 'TLS/SSL', 'unknown')
            
            except socket.timeout:
                pass
            except Exception as e:
                logger.debug(f"Probe {probe_name} failed: {e}")
        
        return None
    
    @staticmethod
    def get_variable_banner_size() -> int:
        """Return random banner read size (avoid fixed 1024/4096 patterns)"""
        sizes = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
        return random.choice(sizes)
    
    @staticmethod
    def get_jittered_delay(mode: str = 'normal') -> float:
        """Get randomized jitter delay (NOT predictable)"""
        if mode == 'normal':
            # Wide ranging: 1-300ms for per-port jitter
            if random.random() < 0.15:  # 15% chance of longer delay
                return random.uniform(0.15, 0.8)  # 150-800ms occasional
            else:
                return random.uniform(0.001, 0.3)  # 1-300ms normal
        
        elif mode == 'beacon':
            # Inter-target beacon (NOT predictable, varies widely)
            rand = random.random()
            if rand < 0.25:  # 25% long delay
                return random.uniform(3.0, 15.0)  # 3-15s (scripted behavior)
            elif rand < 0.5:  # 25% medium delay
                return random.uniform(0.5, 3.0)  # 500ms-3s
            elif rand < 0.8:  # 30% short delay
                return random.uniform(0.05, 0.5)  # 50-500ms
            else:  # 20% rapid fire
                return random.uniform(0.01, 0.1)  # 10-100ms
        
        return random.uniform(0.001, 0.2)  # Default


if __name__ == '__main__':
    print("Advanced Port Scanner Methods Module")
    print("\nFeatures:")
    print("  1. SMB Negotiation - Detect SMB dialect (SMB1/2/3)")
    print("  2. Dynamic Probes - Protocol-specific service detection")
    print("  3. SYN Stealth - Half-open connections (requires admin + scapy)")
    print("  4. Variable Timings - Unpredictable jitter patterns")
    print("\nUsage:")
    print("  from advanced_port_scanner import AdvancedPortScannerMethods")
    print("  methods = AdvancedPortScannerMethods()")