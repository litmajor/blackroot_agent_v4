"""
Vault Directory Randomization with Seed-Based Recovery
Implements deterministic UUID generation for vault paths with honeypot decoy instances
"""

import os
import uuid
import hashlib
import json
import secrets
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger('VaultRandomization')

class VaultRandomizationEngine:
    """
    Implements vault path randomization with seed-based recovery mechanism.
    
    Strategy:
    1. Generate deterministic UUID from machine hardware fingerprint + seed
    2. Use first 8 bytes of UUID as vault directory name
    3. Store UUID mapping in sealed_secret for recovery
    4. Create decoy vault instances (honeypot pattern) at predictable paths
    """
    
    @staticmethod
    def get_machine_fingerprint() -> str:
        """
        Generate deterministic machine fingerprint from hardware identifiers.
        Linux: CPU model from /proc/cpuinfo, MAC address
        Windows: CPU serial from WMI, adapter MAC addresses
        macOS: Hardware UUID from system_profiler
        """
        import platform
        system = platform.system()
        fingerprint_data = []
        
        try:
            if system == "Linux":
                # Try to read CPU info
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                fingerprint_data.append(line.split(":")[1].strip())
                                break
                except:
                    pass
                
                # Get MAC address
                try:
                    import uuid as uuid_lib
                    mac = uuid_lib.getnode()
                    fingerprint_data.append(str(mac))
                except:
                    pass
                    
            elif system == "Windows":
                # Get Windows drive serial
                try:
                    import ctypes
                    import struct
                    
                    kernel32 = ctypes.windll.kernel32
                    GetVolumeInformation = kernel32.GetVolumeInformationW
                    serial_num = ctypes.c_uint()
                    GetVolumeInformation(
                        "C:\\", None, None, ctypes.byref(serial_num), None, None, None, 0
                    )
                    fingerprint_data.append(str(serial_num.value))
                except:
                    pass
                
                # Try to get network adapter MAC
                try:
                    import uuid as uuid_lib
                    mac = uuid_lib.getnode()
                    if mac != 0:
                        fingerprint_data.append(str(mac))
                except:
                    pass
                    
            elif system == "Darwin":  # macOS
                # Get hardware UUID
                try:
                    import subprocess
                    result = subprocess.check_output(
                        ["system_profiler", "SPHardwareDataType"],
                        text=True
                    )
                    for line in result.split("\n"):
                        if "Hardware UUID" in line:
                            fingerprint_data.append(line.split(":")[1].strip())
                            break
                except:
                    pass
        except Exception as e:
            logger.warning(f"Error generating machine fingerprint: {e}")
        
        # Fallback to hostname if no hardware info available
        if not fingerprint_data:
            try:
                fingerprint_data.append(platform.node())
            except:
                pass
        
        # Hash the fingerprint data
        combined = "|".join(fingerprint_data)
        if not combined:
            combined = "unknown_machine"
        
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def generate_vault_uuid(seed: Optional[str] = None) -> str:
        """
        Generate deterministic vault UUID from machine fingerprint + optional seed.
        
        Args:
            seed: Optional additional seed for uniqueness (e.g., deployment ID, org ID)
        
        Returns:
            Deterministic UUID string
        """
        machine_fp = VaultRandomizationEngine.get_machine_fingerprint()
        
        # Combine with optional seed
        hash_input = machine_fp
        if seed:
            hash_input = f"{machine_fp}:{seed}"
        
        # Generate UUID from hash
        hash_obj = hashlib.sha256(hash_input.encode())
        # Create UUID from SHA256 hash (take first 16 bytes)
        vault_uuid = str(uuid.UUID(bytes=hash_obj.digest()[:16]))
        
        return vault_uuid
    
    @staticmethod
    def get_randomized_vault_path(seed: Optional[str] = None) -> str:
        """
        Get randomized vault directory path using deterministic UUID.
        
        Args:
            seed: Optional seed for vault UUID generation
        
        Returns:
            Randomized vault path: ~/.{uuid_8_chars}/vault
        """
        vault_uuid = VaultRandomizationEngine.generate_vault_uuid(seed)
        # Use first 8 characters of UUID
        uuid_short = vault_uuid.replace("-", "")[:8]
        return os.path.expanduser(f"~/.{uuid_short}/vault")
    
    @staticmethod
    def store_vault_recovery_info(vault_dir: str, recovery_file: Optional[str] = None) -> Dict[str, str]:
        """
        Store vault UUID and path recovery information in sealed location.
        
        Args:
            vault_dir: The vault directory path
            recovery_file: Optional custom recovery file location
        
        Returns:
            Recovery information dictionary
        """
        recovery_file = recovery_file or os.path.expanduser("~/.blackvault_recovery")
        
        # Extract vault UUID from path
        vault_path = os.path.expanduser(vault_dir)
        parent_dir = os.path.dirname(vault_path)
        vault_id = os.path.basename(parent_dir)  # 8-char uuid
        
        recovery_info = {
            "vault_id": vault_id,
            "vault_path": vault_path,
            "created": str(__import__('datetime').datetime.now().isoformat()),
            "recovery_seed": None  # Will be set by caller if needed
        }
        
        # Store recovery info with restricted permissions
        try:
            os.makedirs(os.path.dirname(recovery_file), exist_ok=True)
            with open(recovery_file, 'w') as f:
                json.dump(recovery_info, f, indent=2)
            os.chmod(recovery_file, 0o600)  # Restrict to owner only
            logger.info(f"Stored vault recovery info at {recovery_file}")
        except Exception as e:
            logger.error(f"Failed to store vault recovery info: {e}")
        
        return recovery_info
    
    @staticmethod
    def recover_vault_path(recovery_file: Optional[str] = None) -> Optional[str]:
        """
        Recover vault path from recovery file if vault directory is missing.
        
        Args:
            recovery_file: Optional custom recovery file location
        
        Returns:
            Vault path if found, None otherwise
        """
        recovery_file = recovery_file or os.path.expanduser("~/.blackvault_recovery")
        
        if not os.path.exists(recovery_file):
            return None
        
        try:
            with open(recovery_file, 'r') as f:
                recovery_info = json.load(f)
            
            vault_path = recovery_info.get("vault_path")
            if vault_path and os.path.exists(vault_path):
                logger.info(f"Recovered vault path: {vault_path}")
                return vault_path
        except Exception as e:
            logger.error(f"Failed to recover vault path: {e}")
        
        return None


class DecoyVaultHoneypot:
    """
    Creates decoy vault instances to attract/detect unauthorized access.
    
    Strategy:
    1. Create 2-3 decoy vault directories at predictable paths
    2. Populate with fake encrypted artifacts (bad keys)
    3. Log/trigger alert on any access attempt
    4. Redirect explorers to honeypot data
    """
    
    DECOY_VAULT_COUNT = 3
    DECOY_PATHS = [
        "~/.blackvault",           # Classic honeypot location
        "~/.vault",                # Alternative name
        "~/.config/vault",         # Common config location
    ]
    
    @staticmethod
    def get_decoy_paths() -> list:
        """Get list of decoy vault paths."""
        return [os.path.expanduser(p) for p in DecoyVaultHoneypot.DECOY_PATHS[:DecoyVaultHoneypot.DECOY_VAULT_COUNT]]
    
    @staticmethod
    def create_decoy_vault(decoy_path: str, trap_callback: Optional[callable] = None) -> bool:
        """
        Create a decoy vault with fake encrypted data.
        
        Args:
            decoy_path: Directory path for decoy vault
            trap_callback: Optional callback function on honeypot access
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create decoy directory structure
            os.makedirs(decoy_path, mode=0o700, exist_ok=True)
            
            # Create fake .key file (wrong key - will fail decryption)
            fake_key_path = os.path.join(decoy_path, ".key")
            fake_key = secrets.token_bytes(32)  # Random 32-byte key
            with open(fake_key_path, 'wb') as f:
                f.write(fake_key)
            os.chmod(fake_key_path, 0o600)
            
            # Create fake artifacts (will fail on decryption)
            for i in range(3):
                artifact_name = f"artifact_{i:03d}.dat"
                artifact_path = os.path.join(decoy_path, artifact_name)
                fake_artifact = secrets.token_bytes(1024)  # 1KB fake encrypted data
                with open(artifact_path, 'wb') as f:
                    f.write(fake_artifact)
            
            # Create honeypot marker file
            marker_file = os.path.join(decoy_path, ".honeypot_trap")
            trap_info = {
                "created": str(__import__('datetime').datetime.now().isoformat()),
                "access_log": [],
                "trap_callback": "alert_on_access"
            }
            with open(marker_file, 'w') as f:
                json.dump(trap_info, f)
            
            logger.info(f"Created decoy vault at {decoy_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create decoy vault at {decoy_path}: {e}")
            return False
    
    @staticmethod
    def create_all_decoy_vaults(trap_callback: Optional[callable] = None) -> int:
        """
        Create all decoy vault instances.
        
        Args:
            trap_callback: Optional callback on honeypot access
        
        Returns:
            Number of successfully created decoy vaults
        """
        created_count = 0
        for decoy_path in DecoyVaultHoneypot.get_decoy_paths():
            if DecoyVaultHoneypot.create_decoy_vault(decoy_path, trap_callback):
                created_count += 1
        
        logger.info(f"Created {created_count} decoy vault instances")
        return created_count
    
    @staticmethod
    def detect_honeypot_access(vault_path: str) -> bool:
        """
        Detect if honeypot marker exists in vault (indicates decoy vault access attempt).
        
        Args:
            vault_path: Path to check
        
        Returns:
            True if honeypot marker detected, False otherwise
        """
        marker_file = os.path.join(vault_path, ".honeypot_trap")
        if os.path.exists(marker_file):
            logger.warning(f"Honeypot trap detected at {vault_path}")
            
            # Log access attempt
            try:
                with open(marker_file, 'r') as f:
                    trap_info = json.load(f)
                
                access_entry = {
                    "timestamp": str(__import__('datetime').datetime.now().isoformat()),
                    "source": "access_detected"
                }
                trap_info["access_log"].append(access_entry)
                
                with open(marker_file, 'w') as f:
                    json.dump(trap_info, f, indent=2)
            except:
                pass
            
            return True
        
        return False
