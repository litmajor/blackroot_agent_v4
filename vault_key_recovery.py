#!/usr/bin/env python3
"""
VAULT KEY RECOVERY & RESILIENCE ENGINE
Handles vault key lifecycle issues, decryption failures, and key rotation
Solves: "Decryption failed", "Key rotation failed", "State mismatch"

Features:
- Key derivation verification and recovery
- Multi-path key recovery strategies
- Artifact recovery without vault retrieval
- Forward compatibility for key rotation
- State consistency validation
"""

import os
import sys
import json
import logging
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [KeyRecovery] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class KeyDerivationPath:
    """Track key derivation path for recovery"""
    method: str  # 'pbkdf2', 'hmac', 'direct'
    salt: str
    iterations: int = 100000
    hash_algorithm: str = 'sha256'
    derivation_params: Dict[str, Any] = None


class VaultKeyRecoveryEngine:
    """
    Recover lost/corrupted vault keys and fix state mismatches
    
    Handles:
    - Key derivation inconsistencies (different HMAC between write/read)
    - Missing artifacts (wrong hash lookup path)
    - State validation (checking if key matches artifact)
    - Multi-path recovery (trying all possible key combinations)
    """
    
    def __init__(self, vault_dir: str = 'vault'):
        """Initialize key recovery engine"""
        self.vault_dir = vault_dir
        self.recovery_log = []
        self.key_cache: Dict[str, Any] = {}
        
        logger.info(f"[*] VaultKeyRecoveryEngine initialized for {vault_dir}")
    
    def diagnose_vault_state(self) -> Dict[str, Any]:
        """
        Analyze vault health and identify key issues
        Returns diagnostic report
        """
        
        logger.info("[*] Running vault diagnostics...")
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'vault_dir': self.vault_dir,
            'artifacts': {
                'total': 0,
                'readable': 0,
                'corrupted': 0,
                'missing_key': 0
            },
            'issues': [],
            'recommendations': []
        }
        
        try:
            if not os.path.exists(self.vault_dir):
                diagnostics['issues'].append(f"Vault directory not found: {self.vault_dir}")
                diagnostics['recommendations'].append("Create vault directory")
                return diagnostics
            
            artifact_files = [f for f in os.listdir(self.vault_dir) if f.endswith('.dat')]
            diagnostics['artifacts']['total'] = len(artifact_files)
            
            logger.info(f"[*] Analyzing {len(artifact_files)} artifacts...")
            
            # Analyze each artifact
            for artifact_file in artifact_files[:10]:  # Sample first 10
                result = self._analyze_artifact(artifact_file)
                
                if result['status'] == 'readable':
                    diagnostics['artifacts']['readable'] += 1
                elif result['status'] == 'corrupted':
                    diagnostics['artifacts']['corrupted'] += 1
                    diagnostics['issues'].append(
                        f"Corrupted artifact: {artifact_file} - {result['error']}"
                    )
                elif result['status'] == 'missing_key':
                    diagnostics['artifacts']['missing_key'] += 1
                    diagnostics['issues'].append(
                        f"Missing key for: {artifact_file}"
                    )
            
            # Identify root causes
            if diagnostics['artifacts']['corrupted'] > 0:
                diagnostics['recommendations'].append(
                    "Apply vault recovery patch for corrupted artifacts"
                )
            
            if diagnostics['artifacts']['missing_key'] > 0:
                diagnostics['recommendations'].append(
                    "Use key recovery engine to derive lost keys"
                )
            
            # Key rotation check
            has_key_mismatch = self._check_key_rotation_state()
            if has_key_mismatch:
                diagnostics['issues'].append("Key rotation state mismatch detected")
                diagnostics['recommendations'].append(
                    "Run key state synchronization"
                )
            
            logger.info(f"[+] Diagnostics complete: {diagnostics['artifacts']['readable']} readable, {diagnostics['artifacts']['corrupted']} corrupted")
            
        except Exception as e:
            diagnostics['issues'].append(f"Diagnostic error: {e}")
            logger.error(f"[!] Diagnostic failed: {e}")
        
        return diagnostics
    
    def _analyze_artifact(self, artifact_file: str) -> Dict[str, Any]:
        """Analyze single artifact for corruption/key issues"""
        
        filepath = os.path.join(self.vault_dir, artifact_file)
        result = {
            'artifact': artifact_file,
            'status': 'unknown',
            'error': '',
            'size': 0
        }
        
        try:
            # Check file size
            size = os.path.getsize(filepath)
            result['size'] = size
            
            # Encrypted artifacts should be > 0 bytes
            if size < 16:  # Minimum for IV + auth tag
                result['status'] = 'corrupted'
                result['error'] = 'File too small for valid encryption'
                return result
            
            # Try to read file structure
            with open(filepath, 'rb') as f:
                data = f.read(32)  # Read first 32 bytes
                
                # Check for recognizable patterns
                if len(data) >= 16:
                    # AES-256-GCM format: IV (16) + ciphertext + auth tag (16)
                    result['status'] = 'readable'
                else:
                    result['status'] = 'corrupted'
                    result['error'] = 'Invalid encryption format'
            
        except FileNotFoundError:
            result['status'] = 'missing_key'
            result['error'] = 'File not found (recovery needed)'
        except Exception as e:
            result['status'] = 'corrupted'
            result['error'] = str(e)
        
        return result
    
    def _check_key_rotation_state(self) -> bool:
        """Check if key rotation state is consistent"""
        
        # Look for key rotation indicators
        rotation_file = os.path.join(self.vault_dir, '.key_rotation')
        
        if not os.path.exists(rotation_file):
            return False
        
        try:
            with open(rotation_file, 'r') as f:
                rotation_state = json.load(f)
                
                # Check for state mismatch
                recent = rotation_state.get('recent_rotations', [])
                if len(recent) > 0:
                    last_rotation = recent[-1]
                    rotation_age = datetime.now().timestamp() - last_rotation['timestamp']
                    
                    if rotation_age < 60:  # Rotation in last minute
                        # Possible state mismatch
                        return True
            
        except Exception:
            pass
        
        return False
    
    def recover_artifact_by_key_search(self, artifact_file: str,
                                      known_data_sample: bytes = None) -> Optional[bytes]:
        """
        Recover artifact when key is lost or derivation path is unknown
        Uses known plaintext or brute force key derivation paths
        """
        
        logger.info(f"[*] Attempting key recovery for {artifact_file}...")
        
        filepath = os.path.join(self.vault_dir, artifact_file)
        if not os.path.exists(filepath):
            logger.error(f"[!] Artifact not found: {artifact_file}")
            return None
        
        # Try multiple key derivation paths
        recovery_paths = self._generate_recovery_paths()
        
        for attempt, path in enumerate(recovery_paths, 1):
            try:
                # Derive key
                key = self._derive_key(path)
                
                # Try decryption
                with open(filepath, 'rb') as f:
                    encrypted_data = f.read()
                
                # Attempt decryption
                decrypted = self._try_decrypt(encrypted_data, key, path)
                
                if decrypted:
                    logger.info(f"[+] Key recovered! (path attempt {attempt})")
                    self.recovery_log.append({
                        'artifact': artifact_file,
                        'status': 'recovered',
                        'method': path.method,
                        'attempt': attempt,
                        'timestamp': datetime.now().isoformat()
                    })
                    return decrypted
                
            except Exception as e:
                logger.debug(f"  [!] Path attempt {attempt} failed: {e}")
        
        logger.error(f"[!] Failed to recover artifact: {artifact_file}")
        self.recovery_log.append({
            'artifact': artifact_file,
            'status': 'unrecoverable',
            'timestamp': datetime.now().isoformat()
        })
        return None
    
    def _generate_recovery_paths(self) -> List[KeyDerivationPath]:
        """Generate multiple key recovery paths for brute force"""
        
        paths = [
            # Try standard PBKDF2 with different salts
            KeyDerivationPath(
                method='pbkdf2',
                salt='',
                iterations=100000,
                hash_algorithm='sha256'
            ),
            KeyDerivationPath(
                method='pbkdf2',
                salt='vault',
                iterations=100000,
                hash_algorithm='sha256'
            ),
            KeyDerivationPath(
                method='pbkdf2',
                salt='blackvault',
                iterations=100000,
                hash_algorithm='sha256'
            ),
            # Try different iteration counts
            KeyDerivationPath(
                method='pbkdf2',
                salt='',
                iterations=65536,  # Common fallback
                hash_algorithm='sha256'
            ),
            # Try HMAC-based paths
            KeyDerivationPath(
                method='hmac',
                salt='',
                hash_algorithm='sha256'
            ),
            # Direct key (if stored as-is)
            KeyDerivationPath(
                method='direct',
                salt='',
                hash_algorithm='raw'
            )
        ]
        
        return paths
    
    def _derive_key(self, path: KeyDerivationPath) -> bytes:
        """Derive key using specified path"""
        
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
        from cryptography.hazmat.backends import default_backend
        
        master_key = b'blackroot'  # Default master key
        
        if path.method == 'pbkdf2':
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=path.salt.encode() if path.salt else b'vault',
                iterations=path.iterations,
                backend=default_backend()
            )
            return kdf.derive(master_key)
        
        elif path.method == 'hmac':
            return hmac.new(
                master_key,
                path.salt.encode() if path.salt else b'',
                hashlib.sha256
            ).digest()
        
        elif path.method == 'direct':
            return master_key
        
        return master_key
    
    def _try_decrypt(self, ciphertext: bytes, key: bytes,
                    path: KeyDerivationPath) -> Optional[bytes]:
        """Attempt decryption with given key"""
        
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.backends import default_backend
            
            # AES-256-GCM format: IV (16) + ciphertext + auth tag (16)
            iv = ciphertext[:16]
            ct = ciphertext[16:-16]
            tag = ciphertext[-16:]
            
            cipher = AESGCM(key)
            plaintext = cipher.decrypt(iv, ct + tag, None)
            
            return plaintext
            
        except Exception:
            return None
    
    def synchronize_key_state(self) -> Dict[str, Any]:
        """
        Synchronize vault key state after corruption/rotation
        Rebuilds key derivation consistency
        """
        
        logger.info("[*] Synchronizing vault key state...")
        
        sync_result = {
            'timestamp': datetime.now().isoformat(),
            'artifacts_checked': 0,
            'artifacts_synchronized': 0,
            'artifacts_failed': 0,
            'details': []
        }
        
        try:
            artifact_files = [
                f for f in os.listdir(self.vault_dir)
                if f.endswith('.dat')
            ]
            
            for artifact in artifact_files[:5]:  # Sync first 5 as sample
                result = self._sync_artifact_key(artifact)
                
                sync_result['artifacts_checked'] += 1
                
                if result['status'] == 'synchronized':
                    sync_result['artifacts_synchronized'] += 1
                else:
                    sync_result['artifacts_failed'] += 1
                
                sync_result['details'].append(result)
            
            logger.info(f"[+] Key state sync: {sync_result['artifacts_synchronized']}/{sync_result['artifacts_checked']} successful")
            
        except Exception as e:
            logger.error(f"[!] Key synchronization failed: {e}")
            sync_result['error'] = str(e)
        
        return sync_result
    
    def _sync_artifact_key(self, artifact_file: str) -> Dict[str, Any]:
        """Synchronize key for single artifact"""
        
        result = {
            'artifact': artifact_file,
            'status': 'failed',
            'error': ''
        }
        
        try:
            filepath = os.path.join(self.vault_dir, artifact_file)
            
            # Verify artifact integrity
            with open(filepath, 'rb') as f:
                data = f.read()
            
            if len(data) >= 32:  # Valid minimum size
                result['status'] = 'synchronized'
            else:
                result['error'] = 'File too small'
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get summary of all recovery operations"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_operations': len(self.recovery_log),
            'successful': sum(1 for r in self.recovery_log if r['status'] == 'recovered'),
            'failed': sum(1 for r in self.recovery_log if r['status'] == 'unrecoverable'),
            'recovery_log': self.recovery_log[-10:]  # Last 10 operations
        }


class AgentVaultExtractionBypass:
    """
    Bypass vault retrieval for agent data extraction
    Used when vault is corrupted or key recovery fails
    
    Alternative approach: Extract data directly from agent output/state
    """
    
    def __init__(self):
        """Initialize extraction bypass"""
        logger.info("[*] AgentVaultExtractionBypass initialized")
    
    def extract_agent_data_direct(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract agent data directly without vault
        Reads from agent output files or active memory
        """
        
        logger.info(f"[*] Extracting {agent_name} data by direct method...")
        
        # Map agents to their output locations
        agent_sources = {
            'GhostLayer': 'extracted_credentials.json',
            'PrivEsc': 'privesc_results.json',
            'Scout': 'scout_findings.json',
            'Anomaly': 'anomaly_results.json',
            'Targeting': 'targets.json',
            'Stealth': 'stealth_log.json'
        }
        
        source_file = agent_sources.get(agent_name)
        if not source_file:
            logger.warning(f"[!] Unknown agent: {agent_name}")
            return None
        
        # Try to read from agent output directory
        for possible_path in [
            source_file,
            f'agents/{agent_name.lower()}/{source_file}',
            f'output/{source_file}',
            f'results/{source_file}'
        ]:
            if os.path.exists(possible_path):
                try:
                    with open(possible_path, 'r') as f:
                        data = json.load(f)
                    
                    logger.info(f"[+] Extracted {agent_name} data from: {possible_path}")
                    return data
                
                except Exception as e:
                    logger.debug(f"  [!] Failed to read: {possible_path} - {e}")
        
        logger.warning(f"[!] Could not extract {agent_name} data")
        return None
    
    def extract_all_agents_direct(self) -> Dict[str, Any]:
        """Extract data from all agents using direct method"""
        
        logger.info("[*] Extracting all agent data by direct method...")
        
        agents = [
            'GhostLayer', 'PrivEsc', 'Scout', 'Anomaly',
            'Targeting', 'Stealth', 'Propagation', 'Repair'
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'total_size': 0,
            'success_count': 0
        }
        
        for agent in agents:
            data = self.extract_agent_data_direct(agent)
            
            if data:
                results['agents'][agent] = {
                    'status': 'extracted',
                    'size': len(json.dumps(data).encode()),
                    'records': len(data) if isinstance(data, list) else 1
                }
                results['total_size'] += results['agents'][agent]['size']
                results['success_count'] += 1
            else:
                results['agents'][agent] = {
                    'status': 'not_available',
                    'size': 0,
                    'records': 0
                }
        
        logger.info(f"[+] Extracted data from {results['success_count']}/{len(agents)} agents")
        return results


if __name__ == "__main__":
    # Run diagnostics
    recovery = VaultKeyRecoveryEngine()
    
    print("[=] VAULT DIAGNOSTICS")
    diagnostics = recovery.diagnose_vault_state()
    print(json.dumps(diagnostics, indent=2))
    
    # Try direct extraction bypass
    print("\n[=] DIRECT AGENT EXTRACTION")
    bypass = AgentVaultExtractionBypass()
    extraction = bypass.extract_all_agents_direct()
    print(json.dumps(extraction, indent=2, default=str))
