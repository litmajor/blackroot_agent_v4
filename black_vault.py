import os
from typing import Optional
import json
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from threading import Lock
import logging
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# --- Swarm, Identity, Capabilities stubs for integration ---
class Swarm:
    @staticmethod
    def broadcast_capabilities(replica_id, capabilities):
        print(f"[Stub] Swarm.broadcast_capabilities({replica_id}, {capabilities})")
    class evolution:
        @staticmethod
        def update_composition(module, keys):
            print(f"[Stub] Swarm.evolution.update_composition({module}, {keys})")

class Identity:
    replica_id = "standalone-replica"

capabilities = ['mesh_networking', 'command_execution', 'anomaly_classifier']

# For compatibility with 'from black_vault import swarm, identity, capabilities'
swarm = Swarm
identity = Identity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

# Global configuration
VAULT_DIR = os.path.expanduser("~/.blackvault")
AES_BLOCK_SIZE = 16

class BlackVault:
    """A secure, encrypted, and authenticated local storage system for artifacts."""
    
    def __init__(self, password: Optional[str] = None, rotate_days: int = 7, vault_path: Optional[str] = None):
        """
        Initializes the vault with secure key derivation and directory setup.
        
        Args:
            password (str, optional): Password for key derivation. If None, generates a random key.
            rotate_days (int): Number of days before key rotation.
            vault_path (str, optional): Custom path for vault directory (overrides default).
        """
        self.logger = logging.getLogger('BlackVault')
        self.vault_dir = vault_path or VAULT_DIR
        self.lock = Lock()  # Thread safety for file operations
        self.rotate_days = max(1, rotate_days)  # Ensure positive rotation period
        
        # Create vault directory
        try:
            os.makedirs(self.vault_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create vault directory {self.vault_dir}: {e}")
            raise
        
        # Derive encryption and HMAC keys
        self.key = hashlib.sha256(get_random_bytes(32) if not password else password.encode()).digest()
        self.hmac_key = hashlib.sha256(self.key + b"-hmac").digest()
        
        # Initialize key rotation
        self._check_key_rotation()
        self.logger.info(f"Initialized BlackVault at {self.vault_dir}")

    def _vault_path(self, name: str) -> str:
        """
        Generates a hashed file path for an artifact.
        
        Args:
            name (str): Artifact name.
        
        Returns:
            str: Hashed file path.
        """
        hashed = hashlib.sha256(name.encode()).hexdigest()
        return os.path.join(self.vault_dir, f"{hashed}.dat")

    def _encrypt(self, data: bytes, expire_minutes: Optional[int] = None) -> bytes:
        """
        Encrypts data with AES-256-CBC and signs with HMAC-SHA256.
        
        Args:
            data (bytes): Data to encrypt.
            expire_minutes (int, optional): Expiration time in minutes.
        
        Returns:
            bytes: IV + ciphertext + HMAC.
        """
        try:
            iv = get_random_bytes(16)
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            expiry_timestamp = int((datetime.utcnow() + timedelta(minutes=expire_minutes or 0)).timestamp())
            data = expiry_timestamp.to_bytes(8, 'big') + data
            ct = cipher.encrypt(pad(data, AES_BLOCK_SIZE))
            hmac_digest = hmac.new(self.hmac_key, iv + ct, hashlib.sha256).digest()
            return iv + ct + hmac_digest
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise

    def _decrypt(self, encrypted: bytes) -> bytes:
        """
        Verifies, decrypts, and checks expiration of encrypted data.
        
        Args:
            encrypted (bytes): IV + ciphertext + HMAC.
        
        Returns:
            bytes: Decrypted data (excluding expiry timestamp).
        
        Raises:
            ValueError: If HMAC or expiration check fails.
        """
        try:
            if len(encrypted) < 48:  # Minimum: 16 (IV) + 16 (min padded block) + 32 (HMAC)
                raise ValueError("Invalid encrypted data length")
            iv, ct, hmac_received = encrypted[:16], encrypted[16:-32], encrypted[-32:]
            if not hmac.compare_digest(hmac_received, hmac.new(self.hmac_key, iv + ct, hashlib.sha256).digest()):
                raise ValueError("HMAC integrity check failed")
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            data = unpad(cipher.decrypt(ct), AES_BLOCK_SIZE)
            expiry_timestamp = int.from_bytes(data[:8], 'big')
            if expiry_timestamp and time.time() > expiry_timestamp:
                raise ValueError("Artifact has expired")
            return data[8:]
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise

    def store(self, name: str, data: bytes, expire_minutes: Optional[int] = None):
        """
        Encrypts and stores an artifact in the vault.
        
        Args:
            name (str): Artifact name.
            data (bytes): Data to store.
            expire_minutes (int, optional): Expiration time in minutes.
        """
        with self.lock:
            try:
                encrypted = self._encrypt(data, expire_minutes)
                with open(self._vault_path(name), "wb") as f:
                    f.write(encrypted)
                self.logger.info(f"Stored encrypted artifact: {name}")
            except (IOError, ValueError) as e:
                self.logger.error(f"Failed to store artifact {name}: {e}")
                raise

    def retrieve(self, name: str) -> bytes:
        """
        Retrieves and decrypts an artifact from the vault.
        
        Args:
            name (str): Artifact name.
        
        Returns:
            bytes: Decrypted data.
        
        Raises:
            FileNotFoundError: If artifact doesn't exist.
            ValueError: If decryption or HMAC fails.
        """
        with self.lock:
            try:
                with open(self._vault_path(name), "rb") as f:
                    encrypted = f.read()
                data = self._decrypt(encrypted)
                self.logger.info(f"Retrieved artifact: {name}")
                return data
            except (FileNotFoundError, ValueError) as e:
                self.logger.error(f"Failed to retrieve artifact {name}: {e}")
                raise

    def retrieve_for_ghostlayer(self, name: str) -> bytes:
        """
        Retrieves an artifact for GhostLayer with specific logging.
        
        Args:
            name (str): Artifact name.
        
        Returns:
            bytes: Decrypted data.
        """
        self.logger.info(f"GhostLayer fetching payload: {name}")
        return self.retrieve(name)

    def list_artifacts(self) -> list:
        """
        Lists all artifact files in the vault directory.
        
        Returns:
            list: List of artifact file names (hashed).
        """
        try:
            return [f for f in os.listdir(self.vault_dir) if f.endswith(".dat")]
        except OSError as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []

    def delete(self, name: str):
        """
        Deletes an artifact from the vault.
        
        Args:
            name (str): Artifact name.
        """
        with self.lock:
            path = self._vault_path(name)
            try:
                if os.path.exists(path):
                    os.remove(path)
                    self.logger.info(f"Deleted artifact: {name}")
                else:
                    self.logger.warning(f"Artifact not found: {name}")
            except OSError as e:
                self.logger.error(f"Failed to delete artifact {name}: {e}")

    def clean_expired(self):
        """
        Removes expired artifacts from the vault.
        """
        with self.lock:
            for f in self.list_artifacts():
                try:
                    with open(os.path.join(self.vault_dir, f), 'rb') as af:
                        self._decrypt(af.read())
                except ValueError as e:
                    if "expired" in str(e).lower():
                        try:
                            os.remove(os.path.join(self.vault_dir, f))
                            self.logger.info(f"Removed expired artifact: {f}")
                        except OSError as e:
                            self.logger.error(f"Failed to remove expired artifact {f}: {e}")

    def auto_wipe_if_debugger(self):
        """
        Wipes the vault if a debugger is detected.
        """
        try:
            if os.name == "nt":
                import ctypes
                if ctypes.windll.kernel32.IsDebuggerPresent():
                    self.wipe_all()
                    self.logger.warning("Debugger detected, vault wiped")
            elif os.getenv("DEBUG") == "1":
                self.wipe_all()
                self.logger.warning("Debug environment detected, vault wiped")
        except Exception as e:
            self.logger.error(f"Debugger check failed: {e}")

    def _check_key_rotation(self):
        """
        Checks and performs key rotation if needed, re-encrypting all artifacts.
        """
        key_meta_path = os.path.join(self.vault_dir, "vault.keymeta")
        now = int(time.time())
        with self.lock:
            try:
                meta = {"last_rotated": now}
                if os.path.exists(key_meta_path):
                    with open(key_meta_path, 'rb') as f:
                        meta = json.loads(self._decrypt(f.read()).decode())
                    if now - meta.get("last_rotated", 0) >= self.rotate_days * 86400:
                        self.logger.info("Rotating encryption key...")
                        old_key, old_hmac_key = self.key, self.hmac_key
                        self.key = hashlib.sha256(get_random_bytes(32)).digest()
                        self.hmac_key = hashlib.sha256(self.key + b"-hmac").digest()
                        for f in self.list_artifacts():
                            try:
                                with open(os.path.join(self.vault_dir, f), 'rb') as af:
                                    self.key, self.hmac_key = old_key, old_hmac_key
                                    data = self._decrypt(af.read())
                                    self.key = hashlib.sha256(self.key).digest()
                                    self.hmac_key = hashlib.sha256(self.key + b"-hmac").digest()
                                with open(os.path.join(self.vault_dir, f), 'wb') as af:
                                    af.write(self._encrypt(data))
                            except Exception as e:
                                self.logger.error(f"Failed to re-encrypt {f}: {e}")
                        meta["last_rotated"] = now
                with open(key_meta_path, 'wb') as f:
                    f.write(self._encrypt(json.dumps(meta).encode()))
            except Exception as e:
                self.logger.error(f"Key rotation failed: {e}")

    def wipe_all(self):
        """
        Deletes all files in the vault directory.
        """
        with self.lock:
            try:
                for f in os.listdir(self.vault_dir):
                    os.remove(os.path.join(self.vault_dir, f))
                self.logger.info("Vault wiped due to trigger condition")
            except OSError as e:
                self.logger.error(f"Failed to wipe vault: {e}")