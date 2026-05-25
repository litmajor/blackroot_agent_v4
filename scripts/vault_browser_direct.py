#!/usr/bin/env python3
"""
Direct Vault Browser - Simple file-based artifact explorer
Access vault artifacts directly from filesystem
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_vault import BlackVault

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(title, char="="):
    print(f"\n{Colors.CYAN}{char * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^80}{Colors.END}")
    print(f"{Colors.CYAN}{char * 80}{Colors.END}\n")

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def list_vault_files():
    """List all files in vault """
    print_header("VAULT CONTENTS", "█")
    
    vault = BlackVault()
    vault_dir = vault.vault_dir
    
    if not os.path.isdir(vault_dir):
        print(f"{Colors.RED}✗ Vault directory not found: {vault_dir}{Colors.END}")
        return {}
    
    try:
        all_files = os.listdir(vault_dir)
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to list vault: {e}{Colors.END}")
        return {}
    
    system_files = {'.key', '.salt', '.sealed', '.gitkeep'}
    other_files = {'.key': True, '.salt': True, '.sealed': True, 'vault.keymeta': True}
    
    data_files = {}
    
    print(f"{Colors.YELLOW}System Files:{Colors.END}")
    for f in sorted(all_files):
        if f in system_files or f in other_files:
            filepath = os.path.join(vault_dir, f)
            size = os.path.getsize(filepath)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"  • {f:40} {format_size(size):>10}  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{Colors.YELLOW}Encrypted Data Artifacts ({len([f for f in all_files if f.endswith('.dat')])} found):{Colors.END}")
    
    for f in sorted(all_files):
        if f.endswith('.dat'):
            filepath = os.path.join(vault_dir, f)
            try:
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                data_files[f] = {
                    'path': filepath,
                    'size': size,
                    'mtime': mtime
                }
                print(f"  • {f:40} {format_size(size):>10}  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print(f"  • {f:40} {Colors.RED}Error: {e}{Colors.END}")
    
    total_data_size = sum(f['size'] for f in data_files.values())
    print(f"\n{Colors.YELLOW}Total Data Artifacts: {len(data_files)} ({format_size(total_data_size)}){Colors.END}")
    
    return data_files

def analyze_data_files(data_files):
    """Analyze data files for credential content"""
    print_header("ARTIFACT ANALYSIS", "🔐")
    
    if not data_files:
        print(f"{Colors.YELLOW}No data files to analyze{Colors.END}")
        return
    
    vault = BlackVault()
    
    credentials_artifacts = []
    
    for filename, info in data_files.items():
        try:
            # Try to retrieve using vault
            data = vault.retrieve(filename)
            
            # Try to parse as JSON
            try:
                parsed = json.loads(data.decode('utf-8'))
                
                # Check if it looks like credentials
                has_phases = any(key in parsed for key in [
                    'phase_1_extracted', 'phase_2_extracted', 
                    'phase_3_extracted', 'phase_4_extracted', 'findings'
                ])
                
                if has_phases:
                    findings = parsed.get('findings', parsed)
                    phase_counts = {
                        'Phase 1': len(findings.get('phase_1_extracted', [])),
                        'Phase 2': len(findings.get('phase_2_extracted', [])),
                        'Phase 3': len(findings.get('phase_3_extracted', [])),
                        'Phase 4': len(findings.get('phase_4_extracted', [])),
                    }
                    total = sum(phase_counts.values())
                    
                    if total > 0:
                        credentials_artifacts.append({
                            'filename': filename,
                            'size': info['size'],
                            'phases': phase_counts,
                            'total': total,
                            'data': parsed
                        })
            except:
                pass
        except Exception as e:
            pass
    
    if credentials_artifacts:
        print(f"{Colors.GREEN}Found {len(credentials_artifacts)} credential artifact(s):{Colors.END}\n")
        
        for artifact in credentials_artifacts:
            print(f"{Colors.BOLD}{artifact['filename']}{Colors.END}")
            print(f"  Size: {format_size(artifact['size'])}")
            print(f"  Total Credentials: {artifact['total']}")
            for phase, count in artifact['phases'].items():
                if count > 0:
                    print(f"    • {phase}: {count}")
            print()
            
            # Display sample credentials
            if len(credentials_artifacts) <= 2:  # Only for small sets
                findings = artifact['data'].get('findings', artifact['data'])
                for phase_name in ['phase_1_extracted', 'phase_2_extracted', 'phase_3_extracted', 'phase_4_extracted']:
                    items = findings.get(phase_name, [])
                    if items:
                        print(f"    {Colors.BOLD}Sample from {phase_name}:{Colors.END}")
                        for item in items[:2]:  # Show first 2 of each phase
                            if isinstance(item, dict):
                                for k, v in list(item.items())[:2]:  # Show first 2 fields
                                    print(f"      • {k}: {str(v)[:50]}")
                            else:
                                print(f"      • {str(item)[:80]}")
                        if len(items) > 2:
                            print(f"      ... and {len(items) - 2} more")
                print()
    else:
        print(f"{Colors.YELLOW}No credential artifacts found with expected structure{Colors.END}")

def display_artifact(filename, max_lines=50):
    """Display artifact contents"""
    print_header(f"ARTIFACT: {filename}", "🔓")
    
    vault = BlackVault()
    
    try:
        data = vault.retrieve(filename)
        print(f"{Colors.GREEN}✓ Successfully decrypted{Colors.END}\n")
        
        # Try JSON
        try:
            parsed = json.loads(data.decode('utf-8'))
            output = json.dumps(parsed, indent=2)
            lines = output.split('\n')
            print('\n'.join(lines[:max_lines]))
            if len(lines) > max_lines:
                print(f"\n{Colors.YELLOW}... ({len(lines) - max_lines} more lines){Colors.END}")
        except:
            # Raw display
            text = data.decode('utf-8', errors='ignore')
            lines = text.split('\n')
            print('\n'.join(lines[:max_lines]))
            if len(lines) > max_lines:
                print(f"\n{Colors.YELLOW}... ({len(lines) - max_lines} more lines){Colors.END}")
                
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to decrypt: {e}{Colors.END}")

def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'DIRECT VAULT BROWSER':^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    # List vault files
    data_files = list_vault_files()
    
    if not data_files:
        print(f"{Colors.RED}✗ No data files found in vault{Colors.END}")
        return
    
    # Analyze for credentials
    print()
    analyze_data_files(data_files)
    
    # Interactive browsing
    print(f"\n{Colors.YELLOW}Available artifacts:{Colors.END}")
    file_list = sorted(data_files.keys())
    
    for idx, f in enumerate(file_list, 1):
        info = data_files[f]
        print(f"  {idx}. {f} ({format_size(info['size'])})")
    
    print(f"\n{Colors.BLUE}Enter artifact number to view (0 to exit):{Colors.END}")
    
    try:
        while True:
            choice = input(f"{Colors.CYAN}>>> {Colors.END}").strip()
            
            if choice == '0':
                print(f"{Colors.GREEN}✓ Exiting...{Colors.END}")
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(file_list):
                    print()
                    display_artifact(file_list[idx])
                    print(f"\n{Colors.BLUE}Press Enter to continue...{Colors.END}")
                    input()
                else:
                    print(f"{Colors.RED}✗ Invalid selection{Colors.END}")
            except ValueError:
                print(f"{Colors.RED}✗ Enter a valid number{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Browser interrupted{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

if __name__ == "__main__":
    main()
