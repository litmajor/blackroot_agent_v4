#!/usr/bin/env python3
"""
Scout Real Scanner Integration
Replaces simulation with REAL network scanning via the advanced port scanner
"""

import logging
from typing import Dict, Optional, Tuple
import socket
import time

logger = logging.getLogger("ScoutRealScanner")


class RealNetworkScanner:
    """
    Real scanner integration for Scout agent
    Uses network_propagation_port_scanner with advanced methods
    Replaces all simulated scanning
    """
    
    def __init__(self):
        """Initialize with real scanner"""
        try:
            from network_propagation_port_scanner import PortScanner, MultiTargetScanner
            self.PortScanner = PortScanner
            self.MultiTargetScanner = MultiTargetScanner
            self.scanner_available = True
            logger.info("[✅] Real port scanner available - LIVE mode")
        except ImportError:
            self.scanner_available = False
            logger.warning("[⚠️] Port scanner unavailable - falling back to simulation")
    
    def real_service_scan(self, target_ip: str, ports: Optional[list] = None) -> Dict[int, str]:
        """
        Perform REAL port scan instead of simulation
        Returns: {port: service_name, ...}
        """
        if not self.scanner_available:
            logger.warning(f"[⚠️] Real scanner not available for {target_ip}")
            return {}
        
        if ports is None:
            ports = [
                21, 22, 23, 25, 53, 80, 110, 135, 139, 143,
                443, 445, 993, 995, 1433, 3306, 3389, 5432, 5985, 5986
            ]
        
        try:
            scanner = self.PortScanner(timeout=2.0, max_workers=10, jitter_enabled=True)
            result = scanner.scan(target_ip, ports=ports)
            
            # Convert to service mapping
            services = {}
            for port_status in result.open_ports:
                service_name = port_status.service.value.lower()
                services[port_status.port] = service_name
                logger.info(f"[+] REAL: {target_ip}:{port_status.port} = {service_name}")
            
            return services
        
        except Exception as e:
            logger.error(f"[!] Real scan failed: {e}")
            return {}
    
    def real_network_scan(self, network_range: str) -> Dict[str, Dict[int, str]]:
        """
        Perform REAL network scan instead of simulation
        Returns: {ip: {port: service_name, ...}, ...}
        """
        if not self.scanner_available:
            logger.warning(f"[⚠️] Real scanner not available for {network_range}")
            return {}
        
        try:
            scanner = self.MultiTargetScanner(jitter_enabled=True, max_workers=5)
            results = scanner.scan_range(network_range)
            
            # Convert to service mapping
            network_services = {}
            for ip, scan_result in results.items():
                services = {}
                for port_status in scan_result.open_ports:
                    service_name = port_status.service.value.lower()
                    services[port_status.port] = service_name
                
                if services:
                    network_services[ip] = services
                    logger.info(f"[+] REAL: {ip} has {len(services)} open ports")
            
            return network_services
        
        except Exception as e:
            logger.error(f"[!] Real network scan failed: {e}")
            return {}
    
    def get_smb_dialect(self, target_ip: str, port: int = 445) -> Optional[str]:
        """
        Detect SMB dialect (SMB1, SMB2, SMB3) via real negotiation
        """
        try:
            from advanced_port_scanner import AdvancedPortScannerMethods
            return AdvancedPortScannerMethods._negotiate_smb_dialect(target_ip, port, timeout=2.0)
        except Exception as e:
            logger.debug(f"SMB dialect detection failed: {e}")
            return None
    
    def detect_service_by_probe(self, target_ip: str, port: int) -> Optional[Tuple[str, str, str]]:
        """
        Send protocol-specific probes to detect service
        Returns: (service_type, product_name, version)
        """
        try:
            from advanced_port_scanner import AdvancedPortScannerMethods
            return AdvancedPortScannerMethods._detect_service_by_probe(target_ip, port, timeout=1.0)
        except Exception as e:
            logger.debug(f"Service probe failed: {e}")
            return None


def create_real_scanner_integration():
    """Factory function to get scanner integration"""
    return RealNetworkScanner()


# Monkey-patch Scout agent's _simulate_service_scan method
def patch_scout_agent():
    """
    Patch Scout agent to use real scanning instead of simulation
    Call this during Scout agent initialization
    """
    try:
        from agents.scout import ScoutAgent
        real_scanner = RealNetworkScanner()
        
        # Replace simulation with real scanning
        original_simulate = ScoutAgent._simulate_service_scan  # type: ignore
        
        def patched_service_scan(self, node: str) -> Dict[int, str]:
            """Patched version that uses real scanning"""
            if real_scanner.scanner_available:
                logger.info(f"[🔄] Using REAL scanner instead of simulation for {node}")
                return real_scanner.real_service_scan(node)
            else:
                logger.warning(f"[⚠️] Real scanner unavailable, falling back to simulation")
                return original_simulate(self, node)
        
        # Apply patch
        ScoutAgent._simulate_service_scan = patched_service_scan  # type: ignore
        logger.info("[✅] Scout agent patched - now using REAL network scanning")
        
    except ImportError:
        logger.warning("Scout agent not available for patching")


if __name__ == '__main__':
    print("Scout Real Scanner Integration Module")
    print("\nFeatures:")
    print("  - Replace Scout simulated scanning with REAL port scanning")
    print("  - SMB dialect detection (SMB1/2/3)")
    print("  - Dynamic service identification via probes")
    print("  - Network range scanning with jitter evasion")
    print("\nUsage:")
    print("  # Option 1: Direct integration")
    print("  from scout_real_scanner_integration import RealNetworkScanner")
    print("  scanner = RealNetworkScanner()")
    print("  services = scanner.real_service_scan('192.168.1.50')")
    print("")
    print("  # Option 2: Automatic Scout patching")
    print("  from scout_real_scanner_integration import patch_scout_agent")
    print("  patch_scout_agent()  # Scout will now use REAL scanning")
