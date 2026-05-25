#!/usr/bin/env python3
"""
Vault Browser - Interactive artifact explorer and cipher analysis tool
Browse vault contents, display metadata, decrypt and analyze stored artifacts
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_vault import BlackVault

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title, char="="):
    """Print formatted section header"""
    width = 80
    print(f"\n{Colors.CYAN}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^{width}}{Colors.END}")
    print(f"{Colors.CYAN}{char * width}{Colors.END}\n")

def print_success(msg):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    """Print error message"""
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def list_vault_artifacts():
    """List all artifacts in vault with metadata"""
    print_header("VAULT ARTIFACT INVENTORY", "█")
    
    vault = BlackVault()
    vault_dir = vault.vault_dir
    
    if not os.path.isdir(vault_dir):
        print_error(f"Vault directory not found: {vault_dir}")
        return []
    
    try:
        artifacts = os.listdir(vault_dir)
    except Exception as e:
        print_error(f"Failed to list vault: {e}")
        return []
    
    # Separate system files from data artifacts
    system_files = {'.key', '.salt', '.sealed'}
    data_artifacts = [a for a in artifacts if a not in system_files]
    
    # Build artifact table
    artifact_data = []
    for artifact in sorted(data_artifacts):
        filepath = os.path.join(vault_dir, artifact)
        try:
            stat_info = os.stat(filepath)
            size = stat_info.st_size
            mtime = datetime.fromtimestamp(stat_info.st_mtime)
            
            artifact_data.append([
                artifact,
                format_size(size),
                mtime.strftime("%Y-%m-%d %H:%M:%S"),
                "Data artifact" if artifact.endswith('.dat') else "System file"
            ])
        except Exception as e:
            artifact_data.append([artifact, "Error", "Unknown", str(e)])
    
    if not artifact_data:
        print_info("Vault is empty")
        return []
    
    headers = ["Artifact Name", "Size", "Modified", "Type"]
    print(tabulate(artifact_data, headers=headers, tablefmt="grid"))
    print(f"\n{Colors.YELLOW}Total artifacts: {len(data_artifacts)}{Colors.END}")
    
    # System files info
    print(f"\n{Colors.YELLOW}System Files:{Colors.END}")
    for sys_file in sorted(system_files):
        filepath = os.path.join(vault_dir, sys_file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  • {sys_file:15} {format_size(size):>10}")
    
    return data_artifacts

def get_artifact_info(artifact_name):
    """Get detailed information about an artifact"""
    vault = BlackVault()
    vault_dir = vault.vault_dir
    filepath = os.path.join(vault_dir, artifact_name)
    
    if not os.path.exists(filepath):
        print_error(f"Artifact not found: {artifact_name}")
        return None
    
    try:
        stat_info = os.stat(filepath)
        info = {
            'name': artifact_name,
            'path': filepath,
            'size': stat_info.st_size,
            'created': datetime.fromtimestamp(stat_info.st_ctime),
            'modified': datetime.fromtimestamp(stat_info.st_mtime),
            'is_encrypted': artifact_name.endswith('.dat')
        }
        return info
    except Exception as e:
        print_error(f"Failed to get info: {e}")
        return None

def decrypt_artifact(artifact_name):
    """Decrypt and retrieve artifact contents"""
    vault = BlackVault()
    
    try:
        print_info(f"Attempting to retrieve: {artifact_name}")
        data = vault.retrieve(artifact_name)
        print_success(f"Successfully decrypted {artifact_name}")
        return data
    except FileNotFoundError:
        print_error(f"Artifact not found: {artifact_name}")
        return None
    except ValueError as e:
        print_error(f"Integrity check failed: {e}")
        return None
    except Exception as e:
        print_error(f"Decryption failed: {e}")
        return None

def display_artifact_contents(artifact_name, max_length=500):
    """Display decrypted artifact contents"""
    print_header(f"ARTIFACT: {artifact_name}", "🔓")
    
    # Show metadata
    info = get_artifact_info(artifact_name)
    if info:
        print(f"{Colors.YELLOW}Location:{Colors.END} {info['path']}")
        print(f"{Colors.YELLOW}Size:{Colors.END} {format_size(info['size'])}")
        print(f"{Colors.YELLOW}Modified:{Colors.END} {info['modified'].isoformat()}")
        print(f"{Colors.YELLOW}Encrypted:{Colors.END} {'Yes (AES-GCM-256)' if info['is_encrypted'] else 'No'}")
        print()
    
    # Decrypt and display
    data = decrypt_artifact(artifact_name)
    if data is None:
        return
    
    # Try to parse as JSON
    try:
        parsed = json.loads(data.decode('utf-8'))
        print(f"{Colors.GREEN}Format: JSON{Colors.END}\n")
        print(json.dumps(parsed, indent=2)[:max_length])
        if len(json.dumps(parsed, indent=2)) > max_length:
            print(f"\n{Colors.YELLOW}... (truncated){Colors.END}")
    except:
        # Fall back to raw display
        print(f"{Colors.GREEN}Format: Raw/Binary{Colors.END}\n")
        try:
            text = data.decode('utf-8', errors='ignore')
            print(text[:max_length])
            if len(text) > max_length:
                print(f"\n{Colors.YELLOW}... (truncated){Colors.END}")
        except:
            print(f"{Colors.YELLOW}[Binary data - {len(data)} bytes]{Colors.END}")

def analyze_credentials_artifact():
    """Analyze the main credentials extraction artifact"""
    print_header("CREDENTIALS EXTRACTION ANALYSIS", "🔐")
    
    vault = BlackVault()
    
    # List all data artifacts
    artifacts = os.listdir(vault.vault_dir)
    data_artifacts = [a for a in artifacts if a.endswith('.dat')]
    
    print(f"{Colors.YELLOW}Scanning {len(data_artifacts)} encrypted artifacts...{Colors.END}\n")
    
    credentials_found = {}
    
    for artifact in sorted(data_artifacts):
        try:
            data = vault.retrieve(artifact)
            parsed = json.loads(data.decode('utf-8'))
            
            # Check if this looks like credential data
            if 'phase_1_extracted' in parsed or 'phase_2_extracted' in parsed or \
               'phase_3_extracted' in parsed or 'phase_4_extracted' in parsed or \
               'findings' in parsed:
                
                phase_counts = {}
                if 'findings' in parsed:
                    findings = parsed['findings']
                    phase_counts['Phase 1'] = len(findings.get('phase_1_extracted', []))
                    phase_counts['Phase 2'] = len(findings.get('phase_2_extracted', []))
                    phase_counts['Phase 3'] = len(findings.get('phase_3_extracted', []))
                    phase_counts['Phase 4'] = len(findings.get('phase_4_extracted', []))
                else:
                    phase_counts['Phase 1'] = len(parsed.get('phase_1_extracted', []))
                    phase_counts['Phase 2'] = len(parsed.get('phase_2_extracted', []))
                    phase_counts['Phase 3'] = len(parsed.get('phase_3_extracted', []))
                    phase_counts['Phase 4'] = len(parsed.get('phase_4_extracted', []))
                
                total = sum(phase_counts.values())
                if total > 0:
                    credentials_found[artifact] = {
                        'size': os.path.getsize(os.path.join(vault.vault_dir, artifact)),
                        'phases': phase_counts,
                        'total': total
                    }
        except:
            pass
    
    if not credentials_found:
        print_info("No credential artifacts found")
        return
    
    print(f"{Colors.GREEN}Found {len(credentials_found)} credential artifact(s):{Colors.END}\n")
    
    for artifact, info in credentials_found.items():
        print(f"{Colors.BOLD}{artifact}{Colors.END}")
        print(f"  Size: {format_size(info['size'])}")
        print(f"  Total Credentials: {info['total']}")
        for phase, count in info['phases'].items():
            if count > 0:
                print(f"    • {phase}: {count} credentials")
        print()

def interactive_browser():
    """Interactive artifact browser"""
    print_header("VAULT BROWSER - INTERACTIVE MODE", "🔐")
    
    vault = BlackVault()
    artifacts = os.listdir(vault.vault_dir)
    data_artifacts = sorted([a for a in artifacts if a.endswith('.dat')])
    
    if not data_artifacts:
        print_info("No data artifacts found in vault")
        return
    
    print(f"Found {len(data_artifacts)} artifact(s):\n")
    
    for idx, artifact in enumerate(data_artifacts, 1):
        size = os.path.getsize(os.path.join(vault.vault_dir, artifact))
        print(f"  {idx}. {artifact} ({format_size(size)})")
    
    print(f"\n{Colors.YELLOW}Enter artifact number to view (0 to exit):{Colors.END}")
    
    while True:
        try:
            choice = input(f"{Colors.BLUE}>>> {Colors.END}").strip()
            
            if choice == '0':
                print_success("Exiting vault browser")
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(data_artifacts):
                display_artifact_contents(data_artifacts[idx], max_length=2000)
                print(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
                input()
            else:
                print_error("Invalid selection")
        except ValueError:
            print_error("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n" + Colors.YELLOW + "Exiting..." + Colors.END)
            break
        except Exception as e:
            print_error(f"Error: {e}")

def main():
    """Main vault browser"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'BLACKVAULT BROWSER - ARTIFACT EXPLORER':^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    # List all artifacts
    artifacts = list_vault_artifacts()
    
    if not artifacts:
        print_error("No artifacts available")
        return
    
    # Analyze credentials
    print()
    analyze_credentials_artifact()
    
    # Show main credential artifact
    print_header("MAIN CREDENTIALS ARTIFACT", "🔑")
    
    vault = BlackVault()
    data_artifacts = sorted([a for a in artifacts if a.endswith('.dat')])
    
    if data_artifacts:
        # Try to find the main credentials artifact
        for artifact in reversed(data_artifacts):  # Try most recent first
            try:
                info = get_artifact_info(artifact)
                if info and info['size'] > 100:  # Skip tiny artifacts
                    print_success(f"Loading: {artifact}")
                    display_artifact_contents(artifact, max_length=1000)
                    break
            except:
                continue
    
    # Offer interactive mode
    print(f"\n{Colors.YELLOW}Start interactive browser? (y/n): {Colors.END}", end='')
    try:
        response = input().strip().lower()
        if response == 'y':
            interactive_browser()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Browser terminated by user{Colors.END}")
    except Exception as e:
        print_error(f"Browser error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
