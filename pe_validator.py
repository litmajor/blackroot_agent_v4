#!/usr/bin/env python3
"""
PE File Validator - Verifies generated PE files are structurally correct
Checks: MZ header, PE signature, COFF header, Optional header, sections
"""

import struct
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("PEValidator")


class PEValidator:
    """Validates PE file structure and reports errors"""
    
    def __init__(self, pe_data: bytes):
        self.pe_data = pe_data
        self.errors = []
        self.warnings = []
        self.info = []
    
    def validate(self) -> Dict:
        """
        Comprehensive PE validation
        
        Returns:
            Dictionary with validation results
        """
        
        log.info("=" * 70)
        log.info("PE FILE VALIDATOR")
        log.info("=" * 70)
        
        result = {
            'valid': True,
            'file_size': len(self.pe_data),
            'errors': [],
            'warnings': [],
            'structure': {}
        }
        
        # 1. Check MZ header
        if not self._validate_mz_header():
            result['valid'] = False
        
        # 2. Check PE signature and location
        if not self._validate_pe_signature():
            result['valid'] = False
        
        # 3. Parse and validate COFF header
        if not self._validate_coff_header():
            result['valid'] = False
        
        # 4. Parse and validate Optional header
        if not self._validate_optional_header():
            result['valid'] = False
        
        # 5. Validate section headers
        if not self._validate_section_headers():
            result['valid'] = False
        
        # Compile results
        result['errors'] = self.errors
        result['warnings'] = self.warnings
        result['structure'] = self._get_structure_summary()
        
        # Print report
        self._print_report(result)
        
        return result
    
    def _validate_mz_header(self) -> bool:
        """Validate DOS/MZ header"""
        
        log.info("\n🔍 Validating MZ Header...")
        
        if len(self.pe_data) < 64:
            self.errors.append(f"File too small: {len(self.pe_data)} bytes (minimum 64)")
            return False
        
        # Check MZ magic
        magic = struct.unpack('<H', self.pe_data[0:2])[0]
        if magic != 0x5A4D:
            self.errors.append(f"Invalid MZ magic: {hex(magic)} (expected 0x5A4D)")
            return False
        
        self.info.append(f"✅ MZ magic: {hex(magic)}")
        
        # Check e_lfanew (offset to PE header)
        e_lfanew = struct.unpack('<I', self.pe_data[0x3C:0x40])[0]
        if e_lfanew < 64 or e_lfanew > 0x10000:
            self.errors.append(f"Invalid e_lfanew: {hex(e_lfanew)} (must be 64-0x10000)")
            return False
        
        if e_lfanew >= len(self.pe_data):
            self.errors.append(f"e_lfanew points past EOF: {hex(e_lfanew)} (file size: {len(self.pe_data)})")
            return False
        
        self.info.append(f"✅ e_lfanew points to offset: {hex(e_lfanew)}")
        
        return True
    
    def _validate_pe_signature(self) -> bool:
        """Validate PE signature location and content"""
        
        log.info("\n🔍 Validating PE Signature...")
        
        # Get PE offset from MZ header
        e_lfanew = struct.unpack('<I', self.pe_data[0x3C:0x40])[0]
        
        if e_lfanew + 4 > len(self.pe_data):
            self.errors.append(f"PE signature extends past EOF")
            return False
        
        # Check PE signature
        pe_sig = self.pe_data[e_lfanew:e_lfanew+4]
        if pe_sig != b'PE\x00\x00':
            self.errors.append(f"Invalid PE signature: {pe_sig} (expected b'PE\\x00\\x00')")
            return False
        
        self.info.append(f"✅ PE signature found at offset {hex(e_lfanew)}: {pe_sig}")
        
        return True
    
    def _validate_coff_header(self) -> bool:
        """Validate COFF header"""
        
        log.info("\n🔍 Validating COFF Header...")
        
        e_lfanew = struct.unpack('<I', self.pe_data[0x3C:0x40])[0]
        coff_offset = e_lfanew + 4  # After PE signature
        
        if coff_offset + 20 > len(self.pe_data):
            self.errors.append(f"COFF header extends past EOF")
            return False
        
        # Parse COFF header
        coff_data = self.pe_data[coff_offset:coff_offset+20]
        machine, num_sections, timestamp, sym_ptr, num_syms, opt_hdr_size, characteristics = struct.unpack(
            '<HHIIIHH', coff_data
        )
        
        # Validate machine type
        valid_machines = {0x8664: "x64", 0x014c: "i386", 0xAA64: "ARM64"}
        if machine not in valid_machines:
            self.warnings.append(f"Unusual machine type: {hex(machine)}")
        else:
            self.info.append(f"✅ Machine: {valid_machines[machine]} ({hex(machine)})")
        
        # Validate section count
        if num_sections == 0:
            self.errors.append(f"No sections in PE file")
            return False
        elif num_sections > 32:
            self.warnings.append(f"Unusual section count: {num_sections} (typically <= 8)")
        
        self.info.append(f"✅ Number of sections: {num_sections}")
        
        # Validate optional header size
        if opt_hdr_size == 0:
            self.errors.append(f"Optional header size is 0")
            return False
        
        self.info.append(f"✅ Optional header size: {opt_hdr_size}")
        
        return True
    
    def _validate_optional_header(self) -> bool:
        """Validate Optional header"""
        
        log.info("\n🔍 Validating Optional Header...")
        
        e_lfanew = struct.unpack('<I', self.pe_data[0x3C:0x40])[0]
        opt_offset = e_lfanew + 4 + 20  # After PE sig + COFF header
        
        if opt_offset + 26 > len(self.pe_data):
            self.errors.append(f"Optional header extends past EOF")
            return False
        
        # Parse first part of optional header
        magic = struct.unpack('<H', self.pe_data[opt_offset:opt_offset+2])[0]
        
        valid_magics = {0x010B: "PE32", 0x020B: "PE32+"}
        if magic not in valid_magics:
            self.errors.append(f"Invalid optional header magic: {hex(magic)}")
            return False
        
        self.info.append(f"✅ Optional header format: {valid_magics[magic]} ({hex(magic)})")
        
        # For PE32+, check addresses
        if magic == 0x020B:  # PE32+
            image_base = struct.unpack('<Q', self.pe_data[opt_offset+24:opt_offset+32])[0]
            self.info.append(f"✅ Image base: {hex(image_base)}")
        
        return True
    
    def _validate_section_headers(self) -> bool:
        """Validate section headers"""
        
        log.info("\n🔍 Validating Section Headers...")
        
        e_lfanew = struct.unpack('<I', self.pe_data[0x3C:0x40])[0]
        coff_offset = e_lfanew + 4
        
        # Get number of sections
        num_sections = struct.unpack('<H', self.pe_data[coff_offset+2:coff_offset+4])[0]
        opt_hdr_size = struct.unpack('<H', self.pe_data[coff_offset+16:coff_offset+18])[0]
        
        section_offset = coff_offset + 20 + opt_hdr_size
        
        for i in range(num_sections):
            sec_hdr_offset = section_offset + (i * 40)
            
            if sec_hdr_offset + 40 > len(self.pe_data):
                self.errors.append(f"Section {i} header extends past EOF")
                return False
            
            # Parse section header
            sec_data = self.pe_data[sec_hdr_offset:sec_hdr_offset+40]
            sec_name = sec_data[0:8].rstrip(b'\x00').decode('ascii', errors='ignore')
            virtual_size = struct.unpack('<I', sec_data[8:12])[0]
            virtual_addr = struct.unpack('<I', sec_data[12:16])[0]
            raw_size = struct.unpack('<I', sec_data[16:20])[0]
            raw_ptr = struct.unpack('<I', sec_data[20:24])[0]
            characteristics = struct.unpack('<I', sec_data[36:40])[0]
            
            self.info.append(f"✅ Section {i}: {sec_name:8s} VA={hex(virtual_addr)} Size={virtual_size}")
            
            # Validate raw data doesn't exceed file
            if raw_ptr + raw_size > len(self.pe_data):
                self.warnings.append(f"Section {sec_name} raw data extends past EOF")
        
        return True
    
    def _get_structure_summary(self) -> Dict:
        """Get summary of PE structure"""
        
        summary = {
            'file_size': len(self.pe_data),
            'mz_header_offset': 0,
            'pe_signature_offset': struct.unpack('<I', self.pe_data[0x3C:0x40])[0],
        }
        
        return summary
    
    def _print_report(self, result: Dict):
        """Print validation report"""
        
        log.info("\n" + "=" * 70)
        log.info("VALIDATION REPORT")
        log.info("=" * 70)
        
        log.info(f"\n📊 Overall Status: {'✅ VALID' if result['valid'] else '❌ INVALID'}")
        log.info(f"📦 File Size: {result['file_size']} bytes")
        
        if self.info:
            log.info("\n📋 Information:")
            for info in self.info:
                log.info(f"   {info}")
        
        if self.warnings:
            log.info(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                log.info(f"   ⚠️  {warn}")
        
        if self.errors:
            log.info(f"\n❌ Errors ({len(self.errors)}):")
            for err in self.errors:
                log.info(f"   ❌ {err}")
        
        log.info("\n" + "=" * 70 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: pe_validator.py <pe_file>")
        sys.exit(1)
    
    pe_file = sys.argv[1]
    
    with open(pe_file, "rb") as f:
        pe_data = f.read()
    
    validator = PEValidator(pe_data)
    result = validator.validate()
    
    sys.exit(0 if result['valid'] else 1)
