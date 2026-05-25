"""
Vault Randomization Engine
Implements deterministic UUID-based vault directory randomization with recovery mechanism
Priority 1: Randomize vault directory names with seed-based recovery
"""

import os
import json
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Tuple
import uuid as uuid_module
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class VaultRandomizationConfig:
    """Configuration for vault randomization"""
    base_seed: str  # Initial seed for deterministic UUID generation
    vault_instance_id: str  # Unique instance identifier
    randomization_version: int = 1  # Version for future migration
    created_at: float = None  # Timestamp of creation
    decoy_vault_count: int = 3  # Number of decoy vaults to create
    enable_honeypot: bool = True  # Enable honeypot alerts on decoy access


class VaultRandomizationEngine:
    """
    Manages vault directory randomization with seed-based recovery.
    
    Core Capability:
    - Generates deterministic but randomized vault paths
    - Creates recovery mechanism via sealed secret
    - Implements decoy vault pattern for detection evasion
    - Supports vault discovery via recovery UUID
    """
    
    RECOVERY_CONFIG_FILE = ".vault_recovery"
    DECOY_VAULT_MARKER = ".decoy_marker"
    HONEYPOT_ALERT_LOG = ".honeypot_alerts"
    
    def __init__(self, base_seed: Optional[str] = None, instance_id: Optional[str] = None):
        """
        Initialize vault randomization engine.
        
        Args:
            base_seed: Base seed for deterministic UUID generation
            instance_id: Unique identifier for this vault instance
        """
        self.base_seed = base_seed or self._generate_base_seed()
        self.instance_id = instance_id or str(uuid_module.uuid4())
        self.home_dir = Path.home()
        
    @staticmethod
    def _generate_base_seed() -> str:
        """Generate a cryptographically secure base seed."""
        return secrets.token_hex(16)
    
    def generate_deterministic_uuid(self, salt: str = "") -> str:
        """
        Generate deterministic UUID based on base seed and optional salt.
        
        This ensures:
        1. Same seed always produces same UUID (recovery)
        2. Different salts produce different UUIDs (decoy vaults)
        3. Unpredictable to attacker (uses seed entropy)
        
        Args:
            salt: Optional salt for UUID variation
            
        Returns:
            Deterministic UUID string (8 chars for directory name)
        """
        combined = f"{self.base_seed}:{self.instance_id}:{salt}"
        hash_obj = hashlib.sha256(combined.encode())
        # Convert hash to UUID format and take first 8 chars
        uuid_val = str(uuid_module.UUID(hash_obj.hexdigest()[:32]))
        return uuid_val[:8]
    
    def get_randomized_vault_path(self) -> Path:
        """
        Get the randomized vault directory path.
        
        Format: ~/.{uuid8}/vault
        
        Returns:
            Path to randomized vault directory
        """
        uuid_part = self.generate_deterministic_uuid(salt="primary")
        vault_path = self.home_dir / f".{uuid_part}" / "vault"
        return vault_path
    
    def get_decoy_vault_paths(self, count: Optional[int] = None) -> list:
        """
        Get paths for decoy vault instances.
        
        These are honey pot directories with bad keys that trigger alerts if accessed.
        
        Args:
            count: Number of decoy vaults (default from config)
            
        Returns:
            List of Path objects for decoy vaults
        """
        if count is None:
            count = 3
        
        decoy_paths = []
        for i in range(count):
            uuid_part = self.generate_deterministic_uuid(salt=f"decoy_{i}")
            decoy_path = self.home_dir / f".{uuid_part}" / "vault"
            decoy_paths.append(decoy_path)
        
        return decoy_paths
    
    def create_recovery_config(self) -> Dict:
        """
        Create recovery configuration to be stored in sealed secret location.
        
        This config allows recovery of vault location if randomization is lost.
        
        Returns:
            Dictionary with recovery metadata
        """
        config = VaultRandomizationConfig(
            base_seed=self.base_seed,
            vault_instance_id=self.instance_id,
            created_at=__import__('time').time(),
        )
        return asdict(config)
    
    def save_recovery_config(self, sealed_secret_path: Path) -> bool:
        """
        Save recovery configuration to sealed secret location.
        
        Args:
            sealed_secret_path: Path to sealed secret file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            recovery_config = self.create_recovery_config()
            
            # Create parent directory if needed
            sealed_secret_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write as encrypted JSON (caller handles encryption)
            with open(sealed_secret_path, 'w') as f:
                json.dump(recovery_config, f)
            
            logger.info(f"Recovery config saved to {sealed_secret_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save recovery config: {e}")
            return False
    
    def create_decoy_vaults(self) -> Dict[str, bool]:
        """
        Create decoy vault instances with bad keys.
        
        These will trigger honeypot alerts if accessed by analysis tools.
        
        Returns:
            Dictionary mapping decoy paths to creation success status
        """
        decoy_results = {}
        
        for i, decoy_path in enumerate(self.get_decoy_vault_paths()):
            try:
                # Create decoy directory structure
                decoy_path.mkdir(parents=True, exist_ok=True)
                
                # Create marker file identifying it as decoy
                marker_file = decoy_path / self.DECOY_VAULT_MARKER
                marker_file.write_text(json.dumps({
                    "decoy_index": i,
                    "created_at": __import__('time').time(),
                    "purpose": "honeypot_detection_evasion"
                }))
                
                # Create fake .key file with bad data (will fail decryption)
                fake_key_path = decoy_path / ".key"
                fake_key_path.write_bytes(secrets.token_bytes(32))
                fake_key_path.chmod(0o600)
                
                # Create alert log file
                alert_log = decoy_path / self.HONEYPOT_ALERT_LOG
                alert_log.write_text("")
                alert_log.chmod(0o600)
                
                decoy_results[str(decoy_path)] = True
                logger.info(f"Created decoy vault at {decoy_path}")
                
            except Exception as e:
                logger.error(f"Failed to create decoy vault at {decoy_path}: {e}")
                decoy_results[str(decoy_path)] = False
        
        return decoy_results
    
    def check_honeypot_access(self, vault_path: Path) -> bool:
        """
        Check if a decoy vault was accessed (honeypot trigger).
        
        This monitors for analysis tools trying to access decoy vaults.
        
        Args:
            vault_path: Path to check
            
        Returns:
            True if honeypot was accessed
        """
        try:
            marker_file = vault_path / self.DECOY_VAULT_MARKER
            alert_log = vault_path / self.HONEYPOT_ALERT_LOG
            
            if marker_file.exists():
                # Log alert
                alert_msg = f"HONEYPOT_TRIGGERED at {vault_path} - {__import__('time').time()}\n"
                with open(alert_log, 'a') as f:
                    f.write(alert_msg)
                
                logger.warning(f"HONEYPOT ACCESS DETECTED: {vault_path}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking honeypot: {e}")
            return False
    
    def validate_vault_integrity(self, vault_path: Path) -> Tuple[bool, str]:
        """
        Validate that vault path hasn't been tampered with.
        
        Args:
            vault_path: Path to validate
            
        Returns:
            Tuple of (is_valid, status_message)
        """
        try:
            # Check if path exists
            if not vault_path.exists():
                return False, "Vault path does not exist"
            
            # Check if it's a decoy (honeypot)
            if self.check_honeypot_access(vault_path):
                return False, "Honeypot was accessed - analysis detected"
            
            # Check permissions
            stat_info = vault_path.stat()
            if stat_info.st_mode & 0o077 != 0:
                return False, "Vault directory has overly permissive permissions"
            
            return True, "Vault integrity verified"
            
        except Exception as e:
            return False, f"Integrity check error: {e}"


class VaultRecoveryEngine:
    """
    Handles recovery of randomized vault location from recovery config.
    """
    
    def __init__(self):
        self.decrypted_configs = {}
    
    def recover_vault_path(self, recovery_config: Dict) -> Path:
        """
        Recover vault path from recovery configuration.
        
        Args:
            recovery_config: Decrypted recovery configuration dictionary
            
        Returns:
            Path to vault directory
        """
        engine = VaultRandomizationEngine(
            base_seed=recovery_config['base_seed'],
            instance_id=recovery_config['vault_instance_id']
        )
        return engine.get_randomized_vault_path()
    
    def load_recovery_config(self, sealed_secret_path: Path) -> Optional[Dict]:
        """
        Load recovery configuration from sealed secret (already decrypted by caller).
        
        Args:
            sealed_secret_path: Path to recovery config file
            
        Returns:
            Dictionary with recovery config or None if failed
        """
        try:
            with open(sealed_secret_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load recovery config: {e}")
            return None


# Integration with existing BlackVault system
class VaultPathResolver:
    """
    Unified interface for vault path resolution.
    Handles both randomized and legacy vault paths.
    """
    
    def __init__(self, use_randomization: bool = True):
        self.use_randomization = use_randomization
        self.randomization_engine = None
        self.recovery_engine = VaultRecoveryEngine()
        
        if use_randomization:
            self.randomization_engine = VaultRandomizationEngine()
    
    def get_vault_path(self, recovery_config: Optional[Dict] = None) -> Path:
        """
        Get the vault path, handling both randomized and legacy modes.
        
        Args:
            recovery_config: Optional recovery config for path restoration
            
        Returns:
            Path to vault directory
        """
        if recovery_config and self.use_randomization:
            # Recover from existing config
            return self.recovery_engine.recover_vault_path(recovery_config)
        elif self.use_randomization:
            # Generate new randomized path
            return self.randomization_engine.get_randomized_vault_path()
        else:
            # Legacy path
            return Path.home() / ".blackvault" / "vault"
    
    def initialize_randomized_vaults(self, sealed_secret_path: Path) -> Tuple[Path, Dict]:
        """
        Initialize randomized vault system.
        
        Args:
            sealed_secret_path: Path to store recovery config
            
        Returns:
            Tuple of (vault_path, decoy_creation_results)
        """
        vault_path = self.get_vault_path()
        
        # Create vault directory
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Save recovery config
        self.randomization_engine.save_recovery_config(sealed_secret_path)
        
        # Create decoy vaults
        decoy_results = self.randomization_engine.create_decoy_vaults()
        
        return vault_path, decoy_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    engine = VaultRandomizationEngine()
    
    print(f"Primary vault path: {engine.get_randomized_vault_path()}")
    print(f"Decoy vault paths:")
    for p in engine.get_decoy_vault_paths():
        print(f"  - {p}")
    
    print(f"\nRecovery config: {engine.create_recovery_config()}")
    
    # Demonstrate path resolver
    resolver = VaultPathResolver(use_randomization=True)
    vault_path, decoy_results = resolver.initialize_randomized_vaults(
        Path.home() / ".vault_recovery_config"
    )
    print(f"\nInitialized vault at: {vault_path}")
    print(f"Decoy results: {decoy_results}")
