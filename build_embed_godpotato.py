#!/usr/bin/env python3
"""
Automate GodPotato binary embedding into privesc.py

Usage:
    python build_embed_godpotato.py <path_to_godpotato.exe>

Output:
    - Generates base64-encoded constant
    - Injects into WindowsPrivesc class in agents/privesc.py
    - Backs up original privesc.py to privesc.py.bak
"""

import base64
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def encode_binary(exe_path: str) -> str:
    """Read binary file and return base64-encoded string."""
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Binary not found: {exe_path}")
    
    with open(exe_path, 'rb') as f:
        binary_data = f.read()
    
    size_kb = len(binary_data) / 1024
    logger.info(f"Read {exe_path}: {size_kb:.1f} KB")
    
    # Encode to base64
    b64_encoded = base64.b64encode(binary_data).decode('ascii')
    return b64_encoded

def split_b64_lines(b64_string: str, line_length: int = 80) -> str:
    """Split long base64 string into readable lines with proper formatting."""
    lines = []
    for i in range(0, len(b64_string), line_length):
        line = b64_string[i:i+line_length]
        if i == 0:
            lines.append(f'        "{line}"')
        else:
            lines.append(f'        "{line}"')
    return '\n        +\n'.join(lines)

def generate_constant_code(b64_encoded: str, exe_path: str) -> str:
    """Generate Python code for the class constant."""
    timestamp = datetime.now().isoformat()
    size_kb = len(base64.b64decode(b64_encoded)) / 1024
    
    code = f'''    # GodPotato v2.0+ ({timestamp})
    # Source: {os.path.basename(exe_path)} ({size_kb:.1f} KB decoded)
    # Encoded via: build_embed_godpotato.py
    GODPOTATO_BINARY = base64.b64decode(
        {split_b64_lines(b64_encoded)}
    )
'''
    return code

def inject_into_privesc(constant_code: str, privesc_path: str = 'agents/privesc.py'):
    """Inject the constant into WindowsPrivesc class."""
    if not os.path.exists(privesc_path):
        raise FileNotFoundError(f"privesc.py not found at {privesc_path}")
    
    # Backup original
    backup_path = f"{privesc_path}.bak"
    shutil.copy2(privesc_path, backup_path)
    logger.info(f"Backed up original to {backup_path}")
    
    with open(privesc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find insertion point: after MITRE_MAPPING definition
    insertion_marker = "MITRE_MAPPING = {"
    
    # Find the end of MITRE_MAPPING dict
    start_idx = content.find(insertion_marker)
    if start_idx == -1:
        raise ValueError("Could not find MITRE_MAPPING in privesc.py")
    
    # Find closing brace of MITRE_MAPPING
    brace_count = 0
    found_start = False
    insert_pos = start_idx
    
    for i, char in enumerate(content[start_idx:], start=start_idx):
        if char == '{':
            brace_count += 1
            found_start = True
        elif char == '}':
            brace_count -= 1
            if found_start and brace_count == 0:
                insert_pos = i + 1
                # Find next newline
                insert_pos = content.find('\n', insert_pos)
                break
    
    if insert_pos == -1 or insert_pos == start_idx:
        raise ValueError("Could not find end of MITRE_MAPPING")
    
    # Remove old GODPOTATO_BINARY if it exists
    old_start = content.find("    # GodPotato v")
    if old_start != -1:
        old_end = content.find('\n    def ', old_start)
        if old_end != -1:
            content = content[:old_start] + content[old_end+1:]
            insert_pos = old_start
            logger.info("Removed old GODPOTATO_BINARY constant")
    
    # Inject new constant
    new_content = content[:insert_pos] + '\n' + constant_code + '\n' + content[insert_pos:]
    
    with open(privesc_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    logger.info(f"Injected GODPOTATO_BINARY into {privesc_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_embed_godpotato.py <path_to_godpotato.exe> [privesc_path]")
        print("\nExample:")
        print("  python build_embed_godpotato.py C:\\Tools\\GodPotato.exe")
        print("  python build_embed_godpotato.py GodPotato.exe agents/privesc.py")
        sys.exit(1)
    
    exe_path = sys.argv[1]
    privesc_path = sys.argv[2] if len(sys.argv) > 2 else 'agents/privesc.py'
    
    # Make path relative to current directory if needed
    if not os.path.isabs(exe_path):
        exe_path = os.path.join(os.getcwd(), exe_path)
    
    try:
        logger.info("=" * 70)
        logger.info("GodPotato Binary Embedding Tool")
        logger.info("=" * 70)
        
        # Encode
        logger.info(f"\n[1/3] Encoding {exe_path}...")
        b64_encoded = encode_binary(exe_path)
        logger.info(f"[✓] Encoded: {len(b64_encoded)} characters in base64")
        
        # Generate code
        logger.info(f"\n[2/3] Generating Python constant code...")
        constant_code = generate_constant_code(b64_encoded, exe_path)
        logger.info(f"[✓] Generated:\n{constant_code[:200]}...")
        
        # Inject
        logger.info(f"\n[3/3] Injecting into {privesc_path}...")
        inject_into_privesc(constant_code, privesc_path)
        logger.info(f"[✓] Successfully embedded!")
        
        logger.info("\n" + "=" * 70)
        logger.info("NEXT STEPS:")
        logger.info("1. Verify agents/privesc.py has GODPOTATO_BINARY constant")
        logger.info("2. Add 'import base64' at top of privesc.py if not present")
        logger.info("3. Run test: python test_propagation_pipeline.py")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"[✗] Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
