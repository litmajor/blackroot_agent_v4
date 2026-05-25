#!/usr/bin/env python3
"""
Raw Vault Decryption Browser - Direct file decryption without vault abstraction
"""

import os
import json
import sys
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import unpad
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_vault import BlackVault, CRYPTO_IMPORTED

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(title):
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^80}{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}GB"

def raw_decrypt_file(filepath, vault):
    """Decrypt file directly from disk using vault's key"""
    try:
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
        
        # Try GCM decryption first
        if CRYPTO_IMPORTED and len(encrypted_data) > 32:
            try:
                from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                from cryptography.hazmat.backends import default_backend
                
                # Extract IV (first 16 bytes) and tag (last 16 bytes)
                iv = encrypted_data[:16]
                tag = encrypted_data[-16:]
                ciphertext = encrypted_data[16:-16]
                
                cipher = Cipher(
                    algorithms.AES(vault.key),
                    modes.GCM(iv, tag),
                    backend=default_backend()
                )
                decryptor = cipher.decryptor()
                decrypted = decryptor.update(ciphertext) + decryptor.finalize()
                return decrypted
            except:
                pass
        
        # Try CBC decryption
        try:
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            
            cipher = AES.new(vault.key, AES.MODE_CBC, iv)
            decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
            return decrypted
        except:
            pass
        
        return None
    except Exception as e:
        return None

def main():
    print_header("RAW VAULT DECRYPTION BROWSER")
    
    vault = BlackVault()
    vault_dir = vault.vault_dir
    
    # List all .dat files
    all_files = sorted([f for f in os.listdir(vault_dir) if f.endswith('.dat')])
    
    print(f"{Colors.YELLOW}Found {len(all_files)} encrypted artifacts{Colors.END}\n")
    
    # Find largest file (likely main credentials)
    largest = None
    largest_size = 0
    
    for filename in all_files:
        filepath = os.path.join(vault_dir, filename)
        size = os.path.getsize(filepath)
        if size > largest_size:
            largest_size = size
            largest = filename
    
    if largest:
        print(f"{Colors.GREEN}✓ Largest artifact (7.1KB - likely credentials): {largest}{Colors.END}\n")
        print(f"{Colors.BOLD}Attempting decryption...{Colors.END}\n")
        
        filepath = os.path.join(vault_dir, largest)
        decrypted = raw_decrypt_file(filepath, vault)
        
        if decrypted:
            print(f"{Colors.GREEN}✓ Successfully decrypted!{Colors.END}\n")
            
            try:
                parsed = json.loads(decrypted.decode('utf-8'))
                
                # Display structure
                print(f"{Colors.BOLD}Artifact Structure:{Colors.END}")
                print(json.dumps(parsed, indent=2)[:3000])
                
                if len(json.dumps(parsed, indent=2)) > 3000:
                    print(f"\n{Colors.YELLOW}... (truncated){Colors.END}")
                
                # Analyze credentials
                if 'findings' in parsed:
                    findings = parsed['findings']
                    print(f"\n{Colors.BOLD}Credential Summary:{Colors.END}")
                    for phase_key in ['phase_1_extracted', 'phase_2_extracted', 
                                      'phase_3_extracted', 'phase_4_extracted']:
                        items = findings.get(phase_key, [])
                        print(f"  • {phase_key}: {len(items)} items")
            except Exception as e:
                print(f"{Colors.RED}✗ JSON parse error: {e}{Colors.END}")
                print(f"\n{Colors.BOLD}Raw Content:{Colors.END}")
                print(decrypted.decode('utf-8', errors='ignore')[:2000])
        else:
            print(f"{Colors.RED}✗ Decryption failed{Colors.END}")
    
    # Show all artifacts
    print(f"\n\n{Colors.BOLD}All Encrypted Artifacts:{Colors.END}\n")
    for idx, filename in enumerate(all_files, 1):
        filepath = os.path.join(vault_dir, filename)
        size = os.path.getsize(filepath)
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        marker = " ◄ CREDENTIALS" if filename == largest else ""
        print(f"{idx:2}. {filename} ({format_size(size):>8}) {mtime.strftime('%H:%M:%S')}{marker}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
