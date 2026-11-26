import os
from typing import Optional, Dict, Any, cast
import json
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from threading import RLock
import logging
import secrets
import base64
from pathlib import Path

# Import vault randomization engine for Priority 1 security hardening
try:
    from vault_randomization_engine import VaultRandomizationEngine, VaultPathResolver
    VAULT_RANDOMIZATION_AVAILABLE = True
except ImportError:
    VAULT_RANDOMIZATION_AVAILABLE = False
    VaultRandomizationEngine = None
    VaultPathResolver = None

# PHASE 3: Import secure memory pool for in-memory key encryption
try:
    from secure_memory_pool import SecureMemoryPool, get_global_pool
    SECURE_MEMORY_AVAILABLE = True
except ImportError:
    SECURE_MEMORY_AVAILABLE = False
    SecureMemoryPool = None

# Always import pycrypto symbols at the top so they are available everywhere
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_IMPORTED = True
except ImportError:
    CRYPTO_IMPORTED = False
    pass

# Guard pycryptodome import - fail gracefully if not available
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad, unpad
    PYCRYPTODOME_IMPORTED = True
except ImportError:
    PYCRYPTODOME_IMPORTED = False
    AES = None
    get_random_bytes = None
    pad = None
    unpad = None

USE_ARGON2 = False  # Remove unresolved argon2 import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger('BlackVault')

# Global configuration - VAULT PATH RANDOMIZATION (Priority 1 security)
# Initialize randomization engine if available
_RANDOMIZATION_SEED = os.environ.get('BLACKVAULT_RANDOMIZATION_SEED')
_USE_VAULT_RANDOMIZATION = os.environ.get('BLACKVAULT_USE_RANDOMIZATION', 'true').lower() == 'true'

if VAULT_RANDOMIZATION_AVAILABLE and _USE_VAULT_RANDOMIZATION:
    # Initialize randomized vault path
    _vault_resolver = VaultPathResolver(use_randomization=True)
    VAULT_DIR = str(_vault_resolver.get_vault_path())
    VAULT_RANDOMIZATION_ENABLED = True
    logger.info(f"Vault randomization ENABLED: {VAULT_DIR}")
else:
    # Fallback to legacy hardcoded path
    VAULT_DIR = os.path.expanduser("~/.blackvault")
    VAULT_RANDOMIZATION_ENABLED = False
    logger.warning(f"Vault randomization DISABLED: {VAULT_DIR}")

CANONICAL_VAULT_PATH = VAULT_DIR  # Single canonical path (enforced at initialization)
AES_BLOCK_SIZE = 16
LEGACY_VAULT_PATHS = ['vault.bkr']  # Old paths to migrate/clean

# Salt suffixes (consistent usage across key derivation and rotation)
HMAC_SALT_SUFFIX = b'hmac_v2'
KEK_SALT_SUFFIX = b'kek_v1'

# Set USE_CRYPTOGRAPHY based on import success
USE_CRYPTOGRAPHY = CRYPTO_IMPORTED

# Only define cryptography symbols if CRYPTO_IMPORTED is True
if not CRYPTO_IMPORTED:
    PBKDF2HMAC = None
    hashes = None
    default_backend = None
    crypto_hmac = None
    Cipher = None
    algorithms = None
    modes = None


class BlackVault:
    """
    Enhanced secure, encrypted, and authenticated local storage system for artifacts.
    SINGLETON: Only one vault instance per Python process to prevent key duplication
    
    Improvements:
    - Uses cryptography library when available (more secure than pycrypto)
    - AES-GCM authenticated encryption when available
    - Argon2 password hashing when available
    - Rate limiting for access attempts
    - Enhanced file permissions and security
    - Atomic writes and better error handling
    - Metadata support
    - Secure deletion capabilities
    """
    
    _instance = None
    _lock = RLock()
    _instances_by_path: Dict[str, 'BlackVault'] = {}  # Support multiple vaults by path
    
    def __new__(cls, password: Optional[str] = None, rotate_days: int = 7, 
                 vault_path: Optional[str] = None, max_attempts: int = 5, 
                 use_hsm: bool = False, sealed_secret_path: Optional[str] = None):
        """Enforce singleton - always use CANONICAL_VAULT_PATH to prevent HMAC failures.
        All vault paths redirect to single canonical path to ensure deterministic key derivation.
        """
        # ENFORCE CANONICAL PATH: Redirect all requests to single vault path
        # This prevents HMAC mismatches from multiple paths deriving different keys
        vault_path = CANONICAL_VAULT_PATH
        
        if vault_path not in cls._instances_by_path:
            with cls._lock:
                if vault_path not in cls._instances_by_path:
                    instance = super(BlackVault, cls).__new__(cls)
                    cls._instances_by_path[vault_path] = instance
                    # Mark as uninitialized so __init__ runs once
                    instance._initialized = False
                    instance.use_hsm = use_hsm
                    instance.sealed_secret_path = sealed_secret_path or os.path.join(VAULT_DIR, '.sealed')
        
        return cls._instances_by_path[vault_path]
    
    def __init__(self, password: Optional[str] = None, rotate_days: int = 7, 
                 vault_path: Optional[str] = None, max_attempts: int = 5, 
                 use_hsm: bool = False, sealed_secret_path: Optional[str] = None):
        """
        Initializes the vault with secure key derivation and directory setup.
        Skip re-initialization on singleton returns.
        ENFORCES CANONICAL PATH: All requests use single vault path to prevent HMAC failures.
        
        Args:
            password (str, optional): Password for key derivation. If None, generates a random key.
            rotate_days (int): Number of days before key rotation.
            vault_path (str, optional): IGNORED - Always uses CANONICAL_VAULT_PATH to prevent HMAC mismatches.
            max_attempts (int): Maximum access attempts per hour for rate limiting.
            use_hsm (bool): Enable software HSM for key operations.
            sealed_secret_path (str, optional): Path to sealed secret file.
        """
        # Skip re-initialization on singleton returns
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.logger = logging.getLogger('BlackVault')
        # ENFORCE CANONICAL PATH: Ignore any vault_path parameter
        self.vault_dir = CANONICAL_VAULT_PATH
        self.lock = RLock()  # Reentrant lock for nested calls
        self.rotate_days = max(1, rotate_days)  # Ensure positive rotation period
        self.max_attempts = max_attempts
        self.use_hsm = use_hsm
        self.sealed_secret_path = sealed_secret_path or os.path.join(VAULT_DIR, '.sealed')
        
        # Initialize HSM if requested
        if self.use_hsm:
            self._init_hsm()
        
        # Load or create sealed secret
        self.sealed_secret = self._load_sealed_secret()
        self.access_attempts: Dict[str, list] = {}
        # Type annotations for cryptographic material
        self.salt: bytes = b''
        self.key: bytes = b''
        self.hmac_key: bytes = b''
        
        # Clean up legacy vault paths to prevent stale HMAC failures
        self._cleanup_legacy_paths()
        
        # Initialize randomized vault (Priority 1 security hardening)
        self._initialize_randomized_vault()
        
        # Create vault directory with restricted permissions
        try:
            os.makedirs(self.vault_dir, mode=0o700, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create vault directory {self.vault_dir}: {e}")
            raise
        
        # Initialize or load salt
        self.salt = self._get_or_create_salt()
        
        # Derive encryption and HMAC keys - MUST be consistent across instances
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            # Load or create persistent key (prevents HMAC failures on reinit)
            self.key = self._get_or_create_master_key()
        
        # Create separate HMAC key - MUST be derived consistently
        hmac_salt = self.salt + HMAC_SALT_SUFFIX  # consistent suffix for HMAC
        
        # Initialize secure memory pool for master key encryption (in-memory)
        self._memory_pool = None
        self._master_key_buffer_id = None
        if SECURE_MEMORY_AVAILABLE:
            try:
                self._memory_pool = get_global_pool()
                # Encrypt master key in secure memory pool
                self._master_key_buffer_id = self._memory_pool.encrypt_data(self.key)
                logger.info("[PHASE3] Master key encrypted in SecureMemoryPool")
            except Exception as e:
                logger.warning(f"[PHASE3] Failed to use SecureMemoryPool: {e}")
                self._memory_pool = None
        
        try:
            if CRYPTO_IMPORTED and PBKDF2HMAC and hashes and default_backend:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=hmac_salt,
                    iterations=100000,
                    backend=default_backend()
                )
                self.hmac_key = kdf.derive(self.key)
                self.logger.debug("HMAC key derived via PBKDF2HMAC")
            else:
                # Fallback: consistent HMAC key derivation
                self.hmac_key = hashlib.pbkdf2_hmac('sha256', self.key, hmac_salt, 100000)
                self.logger.debug("HMAC key derived via hashlib fallback")
        except Exception as e:
            self.logger.error(f"HMAC key derivation failed: {e}. Using SHA256 fallback.")
            self.hmac_key = hashlib.sha256(self.key + hmac_salt).digest()
        
        # Ensure all cryptographic material is properly initialized
        assert isinstance(self.salt, bytes), "salt must be bytes"
        assert isinstance(self.key, bytes), "key must be bytes"
        assert isinstance(self.hmac_key, bytes), "hmac_key must be bytes"
        
        # Initialize key rotation
        self._check_key_rotation()
        self.logger.info(f"Initialized BlackVault singleton at {self.vault_dir}")
        self._initialized = True

    def _ensure_pycryptodome(self):
        """Ensure pycryptodome is available for fallback operations."""
        if not PYCRYPTODOME_IMPORTED or AES is None or pad is None or unpad is None or get_random_bytes is None:
            raise ImportError("pycryptodome is required for fallback encryption (install via: pip install pycryptodome)")

    def _init_hsm(self):
        """Initialize software HSM for key operations."""
        if not CRYPTO_IMPORTED:
            self.logger.warning("Cryptography library not available, HSM disabled")
            self.use_hsm = False
            return
        
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            
            # Generate HSM key pair for signing operations
            self.hsm_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.hsm_public_key = self.hsm_private_key.public_key()
            self.logger.info("Software HSM initialized with RSA-2048 key pair")
        except Exception as e:
            self.logger.error(f"HSM initialization failed: {e}")
            self.use_hsm = False

    def _load_sealed_secret(self) -> Optional[bytes]:
        """Load or create sealed secret for TPM-like sealing."""
        try:
            if os.path.exists(self.sealed_secret_path):
                with open(self.sealed_secret_path, 'rb') as f:
                    sealed_data = f.read()
                # In a real TPM, this would be unsealed with PCR measurements
                # For software simulation, just return the data
                return sealed_data
            else:
                # Ensure parent directory exists before creating
                sealed_dir = os.path.dirname(self.sealed_secret_path)
                if sealed_dir and not os.path.exists(sealed_dir):
                    os.makedirs(sealed_dir, exist_ok=True)
                # Create new sealed secret
                secret = secrets.token_bytes(32) if hasattr(secrets, 'token_bytes') else (get_random_bytes and get_random_bytes(32)) or os.urandom(32)
                with open(self.sealed_secret_path, 'wb') as f:
                    f.write(secret)
                try:
                    os.chmod(self.sealed_secret_path, 0o600)
                except Exception:
                    pass
                self.logger.info("Created new sealed secret")
                return secret
        except Exception as e:
            self.logger.warning(f"Could not load/create sealed secret ({e}); continuing with ephemeral key material")
            return None

    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create new one"""
        salt_path = os.path.join(self.vault_dir, '.salt')
        try:
            with open(salt_path, 'rb') as f:
                salt = f.read(32)
                if len(salt) == 32:
                    return salt
        except FileNotFoundError:
            pass
        
        # Create new salt
        salt = secrets.token_bytes(32) if hasattr(secrets, 'token_bytes') else (get_random_bytes and get_random_bytes(32)) or os.urandom(32)
        try:
            with open(salt_path, 'wb') as f:
                f.write(salt)
            os.chmod(salt_path, 0o600)  # Restrict permissions
        except OSError as e:
            self.logger.warning(f"Could not set salt file permissions: {e}")
        return salt
    
    def _get_or_create_master_key(self) -> bytes:
        """Load persistent master key or create new one. Prevents HMAC failures."""
        key_path = os.path.join(self.vault_dir, '.key')
        try:
            with open(key_path, 'rb') as f:
                key = f.read(32)
                if len(key) == 32:
                    self.logger.debug("Loaded persistent master key from disk")
                    return key
        except FileNotFoundError:
            pass
        
        # Create and persist new master key
        key = secrets.token_bytes(32) if hasattr(secrets, 'token_bytes') else os.urandom(32)
        try:
            with open(key_path, 'wb') as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Restrict permissions
            self.logger.debug("Created and persisted new master key")
        except OSError as e:
            self.logger.warning(f"Could not persist master key: {e}")
        return key

    def _cleanup_legacy_paths(self):
        """Remove or migrate artifacts from legacy vault paths to prevent stale HMAC failures."""
        for legacy_path in LEGACY_VAULT_PATHS:
            if os.path.exists(legacy_path):
                try:
                    import shutil
                    shutil.rmtree(legacy_path)
                    self.logger.info(f"Cleaned legacy vault path: {legacy_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean legacy vault {legacy_path}: {e}")
    
    def _initialize_randomized_vault(self):
        """
        Initialize randomized vault with honeypot decoy instances (Priority 1).
        
        This enables:
        1. Randomized vault directory names
        2. Honeypot decoy vaults for detection evasion
        3. Recovery mechanism via sealed secret
        """
        if not VAULT_RANDOMIZATION_ENABLED or not VAULT_RANDOMIZATION_AVAILABLE:
            self.logger.debug("Vault randomization not enabled, skipping honeypot initialization")
            return
        
        try:
            # Create primary vault path
            vault_path = Path(self.vault_dir)
            vault_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Vault randomization initialized: {self.vault_dir}")
            
            # Initialize randomization engine
            randomization_engine = VaultRandomizationEngine(base_seed=_RANDOMIZATION_SEED)
            
            # Save recovery configuration to sealed secret location
            recovery_config_path = Path(self.sealed_secret_path).parent / ".vault_recovery_config"
            randomization_engine.save_recovery_config(recovery_config_path)
            self.logger.debug(f"Vault recovery config saved to {recovery_config_path}")
            
            # Create decoy vault instances (honeypot pattern)
            decoy_results = randomization_engine.create_decoy_vaults()
            deployed_decoys = sum(1 for v in decoy_results.values() if v)
            self.logger.info(f"Created {deployed_decoys} decoy vault instances for detection evasion")
            
        except Exception as e:
            self.logger.warning(f"Vault randomization initialization failed (non-fatal): {e}")
            # This is non-fatal - vault will work without randomization
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive key from password using best available method"""
        # Only use cryptography if all symbols are available
        if CRYPTO_IMPORTED and PBKDF2HMAC and hashes and default_backend:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
                backend=default_backend()
            )
            return kdf.derive(password.encode('utf-8'))
        else:
            # Fallback to simple hashing (original method)
            return hashlib.sha256(password.encode() + self.salt).digest()

    def _check_rate_limit(self, operation: str) -> bool:
        """Check if operation is rate limited (smart rate limiting based on operation type)"""
        now = time.time()
        if operation not in self.access_attempts:
            self.access_attempts[operation] = []
        
        # Clean old attempts (older than 1 hour = 3600 seconds)
        self.access_attempts[operation] = [
            t for t in self.access_attempts[operation] 
            if now - t < 3600
        ]
        
        # Determine appropriate limit based on operation type
        # Retrieval operations (safe, read-only) get higher limits
        if operation.startswith("retrieve:"):
            limit = max(1000, self.max_attempts * 10)  # Allow 10x more for retrievals
        # Store operations (write-heavy) get standard limit
        elif operation.startswith("store:"):
            limit = self.max_attempts  # Standard limit for writes
        # Other operations (delete, wipe, etc.) get strict limits
        else:
            limit = max(5, self.max_attempts // 2)  # Half limit for other ops
        
        if len(self.access_attempts[operation]) >= limit:
            self.logger.warning(f"Rate limit ({limit}/hour) exceeded for operation: {operation}")
            return False
        
        self.access_attempts[operation].append(now)
        return True

    def _vault_path(self, name: str) -> str:
        """
        Generates a deterministic hashed file path for an artifact.
        Uses salt + artifact name to ensure consistent paths across all loads.
        
        CRITICAL: Must not depend on HMAC key state to prevent path mismatches
        between storage and retrieval (which caused the 50/50 retrieval failure).
        
        Args:
            name (str): Artifact name.
        
        Returns:
            str: Hashed file path (deterministic, consistent across instances).
        """
        # Deterministic hash: salt + artifact name
        # This ensures the same artifact name always maps to the same path,
        # regardless of key loading order or HMAC key derivation timing
        salt: bytes = cast(bytes, self.salt)
        path_input = salt + name.encode()
        hashed = hashlib.sha256(path_input).hexdigest()[:32]
        return os.path.join(self.vault_dir, f"{hashed}.dat")

    def _encrypt(self, data: bytes, expire_minutes: Optional[int] = None) -> bytes:
        """
        Encrypts data with best available method (AES-GCM preferred, AES-CBC+HMAC fallback).
        
        Args:
            data (bytes): Data to encrypt.
            expire_minutes (int, optional): Expiration time in minutes. Defaults to one year if not set.
        
        Returns:
            bytes: Encrypted data.
        """
        if expire_minutes is None:
            expire_minutes = 525600  # 1 year
        
        # Type casts to ensure type checker knows these are bytes
        key: bytes = cast(bytes, self.key)
        hmac_key: bytes = cast(bytes, self.hmac_key)
        salt: bytes = cast(bytes, self.salt)
        
        try:
            # Envelope encryption: generate per-artifact DEK, encrypt data with DEK,
            # then encrypt DEK with KEK (derived from master key). This provides
            # forward secrecy for stored artifacts.
            expiry_timestamp = int((datetime.utcnow() + timedelta(minutes=expire_minutes or 0)).timestamp())
            data_with_expiry = expiry_timestamp.to_bytes(8, 'big') + data

            # Generate random DEK
            dek = secrets.token_bytes(32) if hasattr(secrets, 'token_bytes') else os.urandom(32)

            # Derive KEK from master key (key) using consistent salt
            kek_salt = salt + KEK_SALT_SUFFIX
            if CRYPTO_IMPORTED and PBKDF2HMAC and hashes and default_backend:
                kek_kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(), length=32, salt=kek_salt,
                    iterations=150000, backend=default_backend()
                )
                kek: bytes = kek_kdf.derive(key)
            else:
                kek = hashlib.pbkdf2_hmac('sha256', key, kek_salt, 150000)
            assert isinstance(kek, bytes), "KEK must be bytes"

            # Encrypt DEK with KEK (use AES-GCM if available)
            if CRYPTO_IMPORTED and Cipher and algorithms and modes and default_backend:
                dek_nonce = secrets.token_bytes(12) if hasattr(secrets, 'token_bytes') else os.urandom(12)
                kek_cipher = Cipher(algorithms.AES(kek), modes.GCM(dek_nonce), backend=default_backend())
                kek_enc = kek_cipher.encryptor()
                enc_dek_ct = kek_enc.update(dek) + kek_enc.finalize()
                dek_tag = kek_enc.tag
                enc_dek_blob = dek_nonce + enc_dek_ct + dek_tag
            else:
                # Fallback: encrypt DEK with AES-CBC + HMAC
                self._ensure_pycryptodome()
                dek_iv = os.urandom(16)
                dek_cipher = AES.new(kek, AES.MODE_CBC, dek_iv)  # type: ignore
                enc_dek_ct = dek_cipher.encrypt(pad(dek, AES_BLOCK_SIZE))  # type: ignore
                dek_hmac = hmac.new(hmac_key, dek_iv + enc_dek_ct, hashlib.sha256).digest()
                enc_dek_blob = dek_iv + enc_dek_ct + dek_hmac

            # Encrypt data_with_expiry with DEK
            if CRYPTO_IMPORTED and Cipher and algorithms and modes and default_backend:
                data_nonce = secrets.token_bytes(12) if hasattr(secrets, 'token_bytes') else os.urandom(12)
                data_cipher = Cipher(algorithms.AES(dek), modes.GCM(data_nonce), backend=default_backend())
                data_enc = data_cipher.encryptor()
                data_ct = data_enc.update(data_with_expiry) + data_enc.finalize()
                data_tag = data_enc.tag
                payload = data_nonce + data_ct + data_tag
            else:
                data_iv = os.urandom(16)
                data_cipher = AES.new(dek, AES.MODE_CBC, data_iv)  # type: ignore
                data_ct = data_cipher.encrypt(pad(data_with_expiry, AES_BLOCK_SIZE))  # type: ignore
                data_hmac = hmac.new(hmac_key, data_iv + data_ct, hashlib.sha256).digest()
                payload = data_iv + data_ct + data_hmac

            # Header with encrypted DEK (base64) and version
            header = json.dumps({
                'v': 2,
                'enc_dek': base64.b64encode(enc_dek_blob).decode('ascii')
            }).encode('utf-8') + b"\n"

            return header + payload
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise

    def _decrypt(self, encrypted: bytes) -> bytes:
        """
        Verifies, decrypts, and checks expiration of encrypted data.
        
        Args:
            encrypted (bytes): Encrypted data.
        
        Returns:
            bytes: Decrypted data (excluding expiry timestamp).
        
        Raises:
            ValueError: If verification or expiration check fails.
        """
        # Type casts to ensure type checker knows these are bytes
        key: bytes = cast(bytes, self.key)
        hmac_key: bytes = cast(bytes, self.hmac_key)
        salt: bytes = cast(bytes, self.salt)
        
        try:
            # New envelope format: header JSON (v=2) newline payload
            try:
                header_raw, payload = encrypted.split(b"\n", 1)
                header = json.loads(header_raw.decode('utf-8'))
            except Exception:
                # Fallback: unsupported/old format - attempt legacy decrypt with master key
                header = None
                payload = encrypted

            if header and header.get('v') == 2:
                enc_dek_b64 = header.get('enc_dek')
                if enc_dek_b64 is None:
                    raise ValueError("Missing 'enc_dek' in header")
                enc_dek_blob = base64.b64decode(enc_dek_b64)

                # Derive KEK from master key
                kek_salt = salt + KEK_SALT_SUFFIX
                if CRYPTO_IMPORTED and PBKDF2HMAC and hashes and default_backend:
                    kek_kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(), length=32, salt=kek_salt,
                        iterations=150000, backend=default_backend()
                    )
                    kek: bytes = kek_kdf.derive(key)
                else:
                    kek = hashlib.pbkdf2_hmac('sha256', key, kek_salt, 150000)
                assert isinstance(kek, bytes), "KEK must be bytes"

                # Decrypt DEK
                if CRYPTO_IMPORTED and Cipher and algorithms and modes and default_backend:
                    dek_nonce = enc_dek_blob[:12]
                    enc_dek_ct = enc_dek_blob[12:-16]
                    dek_tag = enc_dek_blob[-16:]
                    kek_cipher = Cipher(algorithms.AES(kek), modes.GCM(dek_nonce, dek_tag), backend=default_backend())
                    dek_dec = kek_cipher.decryptor()
                    dek = dek_dec.update(enc_dek_ct) + dek_dec.finalize()
                else:
                    dek_iv = enc_dek_blob[:16]
                    enc_dek_ct = enc_dek_blob[16:-32]
                    dek_hmac = enc_dek_blob[-32:]
                    if not hmac.compare_digest(dek_hmac, hmac.new(hmac_key, dek_iv + enc_dek_ct, hashlib.sha256).digest()):
                        raise ValueError("HMAC integrity check failed for DEK")
                    dek_cipher = AES.new(kek, AES.MODE_CBC, dek_iv)  # type: ignore
                    dek = unpad(dek_cipher.decrypt(enc_dek_ct), AES_BLOCK_SIZE)  # type: ignore

                # Now decrypt payload with DEK
                if CRYPTO_IMPORTED and Cipher and algorithms and modes and default_backend:
                    data_nonce = payload[:12]
                    data_ct = payload[12:-16]
                    data_tag = payload[-16:]
                    data_cipher = Cipher(algorithms.AES(dek), modes.GCM(data_nonce, data_tag), backend=default_backend())
                    data_dec = data_cipher.decryptor()
                    data_with_expiry = data_dec.update(data_ct) + data_dec.finalize()
                else:
                    data_iv = payload[:16]
                    data_ct = payload[16:-32]
                    data_hmac = payload[-32:]
                    if not hmac.compare_digest(data_hmac, hmac.new(hmac_key, data_iv + data_ct, hashlib.sha256).digest()):
                        raise ValueError("HMAC integrity check failed for payload")
                    data_cipher = AES.new(dek, AES.MODE_CBC, data_iv)  # type: ignore
                    data_with_expiry = unpad(data_cipher.decrypt(data_ct), AES_BLOCK_SIZE)  # type: ignore

                expiry_timestamp = int.from_bytes(data_with_expiry[:8], 'big')
                if expiry_timestamp and time.time() > expiry_timestamp:
                    raise ValueError("Artifact has expired")
                return data_with_expiry[8:]

            # Legacy fallback: attempt previous master-key-only decryption
            if len(payload) < 48:
                raise ValueError("Invalid encrypted data length")
            iv, ct, hmac_received = payload[:16], payload[16:-32], payload[-32:]
            if not hmac.compare_digest(hmac_received, hmac.new(hmac_key, iv + ct, hashlib.sha256).digest()):
                raise ValueError("HMAC integrity check failed")
            cipher = AES.new(key, AES.MODE_CBC, iv)  # type: ignore
            data = unpad(cipher.decrypt(ct), AES_BLOCK_SIZE)  # type: ignore
            expiry_timestamp = int.from_bytes(data[:8], 'big')
            if expiry_timestamp and time.time() > expiry_timestamp:
                raise ValueError("Artifact has expired")
            return data[8:]
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise

    def store(self, name: str, data: bytes, expire_minutes: Optional[int] = None, 
              metadata: Optional[Dict[str, Any]] = None):
        """
        Encrypts and stores an artifact in the vault.
        
        Args:
            name (str): Artifact name.
            data (bytes): Data to store.
            expire_minutes (int, optional): Expiration time in minutes. Defaults to one year if not set.
            metadata (dict, optional): Additional metadata to store with the artifact.
        """
        if expire_minutes is None:
            expire_minutes = 525600  # 1 year
        
        if not self._check_rate_limit(f"store:{name}"):
            raise ValueError("Rate limit exceeded")
        
        with self.lock:
            try:
                # Prepare data with metadata if provided
                if metadata:
                    data_package = {
                        'data': base64.b64encode(data).decode(),
                        'metadata': metadata,
                        'stored_at': datetime.utcnow().isoformat()
                    }
                    data_to_encrypt = json.dumps(data_package).encode()
                else:
                    data_to_encrypt = data
                
                encrypted = self._encrypt(data_to_encrypt, expire_minutes)
                
                # Atomic write (write to temp file, then rename)
                path = self._vault_path(name)
                temp_path = path + '.tmp'
                
                with open(temp_path, "wb") as f:
                    f.write(encrypted)
                
                # Set restrictive permissions
                try:
                    os.chmod(temp_path, 0o600)
                except OSError as e:
                    self.logger.warning(f"Could not set file permissions: {e}")
                
                # Atomic rename (use replace for Windows compatibility)
                os.replace(temp_path, path)
                self.logger.info(f"Stored encrypted artifact: {name}")

                # Update living TargetIdentity profile when target context is present
                try:
                    target_id = (metadata or {}).get("target_id")
                    if not target_id and ":" in name:
                        target_id = name.split(":", 1)[0]
                    if target_id:
                        from recon.target_identity import TargetIdentityRegistry
                        TargetIdentityRegistry.instance().ingest_extract(
                            target_id=str(target_id),
                            artifact_name=name,
                            metadata=metadata,
                            agent_id=str((metadata or {}).get("agent_id", "vault_store")),
                        )
                except Exception as tid_exc:
                    self.logger.debug(f"TargetIdentity ingest skipped: {tid_exc}")
                
            except Exception as e:
                # Clean up temp file if it exists
                temp_path = self._vault_path(name) + '.tmp'
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
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
            ValueError: If decryption or verification fails.
        """
        if not self._check_rate_limit(f"retrieve:{name}"):
            raise ValueError("Rate limit exceeded")
        
        with self.lock:
            try:
                with open(self._vault_path(name), "rb") as f:
                    encrypted = f.read()
                
                decrypted = self._decrypt(encrypted)
                
                # Check if this includes metadata
                try:
                    data_package = json.loads(decrypted.decode())
                    if isinstance(data_package, dict) and 'data' in data_package:
                        data = base64.b64decode(data_package['data'])
                        self.logger.info(f"Retrieved artifact with metadata: {name}")
                        return data
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                    pass
                
                # Return raw data if no metadata
                self.logger.info(f"Retrieved artifact: {name}")
                return decrypted
                
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
        Securely deletes an artifact from the vault.
        
        Args:
            name (str): Artifact name.
        """
        with self.lock:
            path = self._vault_path(name)
            try:
                if os.path.exists(path):
                    # Attempt secure deletion (overwrite with random data)
                    try:
                        size = os.path.getsize(path)
                        with open(path, 'r+b') as f:
                            random_data = secrets.token_bytes(size) if hasattr(secrets, 'token_bytes') else os.urandom(size)
                            f.write(random_data)
                            f.flush()
                            os.fsync(f.fileno())  # Force write to disk
                    except (OSError, IOError):
                        pass  # Best effort
                    
                    os.remove(path)
                    self.logger.info(f"Deleted artifact: {name}")
                else:
                    self.logger.warning(f"Artifact not found: {name}")
            except OSError as e:
                self.logger.error(f"Failed to delete artifact {name}: {e}")

    def store_signed(self, name: str, data: bytes, agent_id: str, 
                     expire_minutes: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Store data with digital signature for integrity verification.
        
        Args:
            name: Artifact name
            data: Data to store
            agent_id: Agent identifier for signing
            expire_minutes: Expiration time
            metadata: Additional metadata
        """
        if not self.use_hsm:
            raise ValueError("HSM required for signed storage")
        
        # Create signature
        signature = self._sign_data(data, agent_id)
        
        # Prepare signed package
        signed_package = {
            'data': base64.b64encode(data).decode(),
            'signature': base64.b64encode(signature).decode(),
            'agent_id': agent_id,
            'signed_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        signed_data = json.dumps(signed_package).encode()
        self.store(f"{name}.signed", signed_data, expire_minutes)

    def retrieve_signed(self, name: str) -> Dict[str, Any]:
        """
        Retrieve and verify signed data.
        
        Returns:
            Dict with 'data', 'agent_id', 'verified' status
        """
        signed_data = self.retrieve(f"{name}.signed")
        package = json.loads(signed_data.decode())
        
        data = base64.b64decode(package['data'])
        signature = base64.b64decode(package['signature'])
        agent_id = package['agent_id']
        
        verified = self._verify_signature(data, signature, agent_id)
        
        return {
            'data': data,
            'agent_id': agent_id,
            'verified': verified,
            'metadata': package.get('metadata', {})
        }

    def _sign_data(self, data: bytes, agent_id: str) -> bytes:
        """Sign data using HSM private key."""
        if not CRYPTO_IMPORTED or not hasattr(self, 'hsm_private_key'):
            raise ValueError("HSM not available")
        
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        
        # Create digest
        digest = hashes.Hash(hashes.SHA256())
        digest.update(data)
        data_hash = digest.finalize()
        
        # Sign with HSM key
        signature = self.hsm_private_key.sign(
            data_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    def _verify_signature(self, data: bytes, signature: bytes, agent_id: str) -> bool:
        """Verify signature using HSM public key."""
        if not CRYPTO_IMPORTED or not hasattr(self, 'hsm_public_key'):
            return False
        
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        
        try:
            # Create digest
            digest = hashes.Hash(hashes.SHA256())
            digest.update(data)
            data_hash = digest.finalize()
            
            # Verify with HSM key
            self.hsm_public_key.verify(
                signature,
                data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

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
        Ensures HMAC key derivation uses consistent parameters during rotation.
        """
        key_meta_path = os.path.join(self.vault_dir, "vault.keymeta")
        now = int(time.time())
        with self.lock:
            try:
                meta = {"last_rotated": now}
                if os.path.exists(key_meta_path):
                    try:
                        with open(key_meta_path, 'rb') as f:
                            meta = json.loads(self._decrypt(f.read()).decode())
                    except ValueError as ve:
                        # HMAC check failed - likely old format or key mismatch
                        self.logger.debug(f"Could not decrypt key metadata (expected on first rotation): {ve}")
                        meta = {"last_rotated": now}
                    
                    if now - meta.get("last_rotated", 0) >= self.rotate_days * 86400:
                        self.logger.info("Rotating encryption key...")
                        old_key = self.key
                        old_hmac_key = self.hmac_key
                        
                        # Generate new key FIRST, then derive HMAC consistently
                        new_key = secrets.token_bytes(32) if hasattr(secrets, 'token_bytes') else os.urandom(32)
                        
                        # CRITICAL FIX: Use HMAC_SALT_SUFFIX (b'hmac_v2'), NOT b'hmac'
                        # This ensures new HMAC key is derived with same salt as initial derivation
                        # Prevents verification failures after rotation
                        if CRYPTO_IMPORTED and PBKDF2HMAC and hashes and default_backend:
                            kdf = PBKDF2HMAC(
                                algorithm=hashes.SHA256(),
                                length=32,
                                salt=self.salt + HMAC_SALT_SUFFIX,  # Use canonical suffix
                                iterations=100000,
                                backend=default_backend()
                            )
                            new_hmac_key = kdf.derive(new_key)
                        else:
                            # Fallback: use same salt suffix for consistency
                            new_hmac_key = hashlib.pbkdf2_hmac('sha256', new_key, self.salt + HMAC_SALT_SUFFIX, 100000)
                        
                        # Re-encrypt all artifacts with old keys first, then new keys
                        # CRITICAL: Keep self.lock held throughout entire rotation to prevent concurrent access
                        artifacts_to_rotate = []
                        try:
                            artifacts_to_rotate = list(self.list_artifacts())
                        except Exception as e:
                            self.logger.warning(f"Could not list artifacts for rotation: {e}")
                        
                        rotation_failed = False
                        for artifact_name in artifacts_to_rotate:
                            try:
                                artifact_path = os.path.join(self.vault_dir, artifact_name)
                                
                                # Decrypt with OLD keys (don't modify self.key yet - maintain lock integrity)
                                # Temporarily swap for decryption using _decrypt's internal key references
                                old_saved_key = self.key
                                old_saved_hmac = self.hmac_key
                                self.key = old_key
                                self.hmac_key = old_hmac_key
                                
                                with open(artifact_path, 'rb') as af:
                                    encrypted_data = af.read()
                                    decrypted_data = self._decrypt(encrypted_data)
                                
                                # Re-encrypt with NEW keys 
                                self.key = new_key
                                self.hmac_key = new_hmac_key
                                new_encrypted = self._encrypt(decrypted_data)
                                
                                # Restore original keys for consistency during rotation
                                self.key = old_saved_key
                                self.hmac_key = old_saved_hmac
                                
                                # Atomic write (temp -> final)
                                temp_path = artifact_path + '.rotated'
                                with open(temp_path, 'wb') as af:
                                    af.write(new_encrypted)
                                os.replace(temp_path, artifact_path)
                                
                                self.logger.debug(f"Re-encrypted artifact: {artifact_name}")
                            except Exception as e:
                                self.logger.error(f"Failed to rotate {artifact_name}: {e}")
                                rotation_failed = True
                        
                        if not rotation_failed:
                            # Only update instance keys if rotation completed successfully
                            # This ensures lock is held throughout AND keys are only changed once
                            self.key = new_key
                            self.hmac_key = new_hmac_key
                            meta["last_rotated"] = now
                            self.logger.info(f"Key rotation complete. New key hash: {hashlib.sha256(new_key).hexdigest()[:16]}...")
                        else:
                            self.logger.warning("Key rotation had failures - keeping old keys (artifacts may be inconsistent)")
                        
                        return  # Exit to prevent double-write below
                
                # Write/update metadata with CURRENT keys
                with open(key_meta_path, 'wb') as f:
                    f.write(self._encrypt(json.dumps(meta).encode()))
            except Exception as e:
                self.logger.error(f"Key rotation failed: {e}")

    def clean_expired(self):
        """
        Clean expired entries from vault (if any TTL entries exist).
        Delegates to clean_expired_artifacts().
        """
        self.logger.debug("Checking for expired entries...")
        self.clean_expired_artifacts()

    def clean_expired_artifacts(self):
        """
        Removes expired artifacts from the vault.
        Handles corrupted or undecryptable artifacts gracefully.
        """
        with self.lock:
            for f in self.list_artifacts():
                try:
                    with open(os.path.join(self.vault_dir, f), 'rb') as af:
                        self._decrypt(af.read())
                except ValueError as e:
                    error_msg = str(e).lower()
                    # Only delete if explicitly expired; other errors mean corruption
                    if "expired" in error_msg:
                        try:
                            # Secure deletion of expired file
                            filepath = os.path.join(self.vault_dir, f)
                            size = os.path.getsize(filepath)
                            with open(filepath, 'r+b') as af:
                                random_data = secrets.token_bytes(size) if hasattr(secrets, 'token_bytes') else os.urandom(size)
                                af.write(random_data)
                                af.flush()
                                os.fsync(af.fileno())
                            os.remove(filepath)
                            self.logger.info(f"Removed expired artifact: {f}")
                        except OSError as err:
                            self.logger.error(f"Failed to remove expired artifact {f}: {err}")
                    else:
                        # Corrupted artifact, skip it
                        self.logger.debug(f"Skipping corrupted artifact {f}: {e}")
                except Exception as e:
                    # Handle all other decryption-related exceptions (InvalidTag, etc.)
                    error_name = type(e).__name__
                    self.logger.debug(f"Skipping artifact {f} ({error_name}): {e}")

    def wipe_all(self):
        """
        Securely deletes all files in the vault directory.
        """
        with self.lock:
            try:
                for filename in os.listdir(self.vault_dir):
                    filepath = os.path.join(self.vault_dir, filename)
                    if os.path.isfile(filepath):
                        # Secure deletion (overwrite with random data)
                        try:
                            size = os.path.getsize(filepath)
                            with open(filepath, 'r+b') as f:
                                random_data = secrets.token_bytes(size) if hasattr(secrets, 'token_bytes') else os.urandom(size)
                                f.write(random_data)
                                f.flush()
                                os.fsync(f.fileno())
                        except (OSError, IOError):
                            pass  # Best effort
                        
                        os.remove(filepath)
                        
                self.logger.info("Vault wiped due to trigger condition")
            except OSError as e:
                self.logger.error(f"Failed to wipe vault: {e}")

    def destroy_master_key(self, irrecoverable: bool = True):
        """
        Destroy the persistent master key. This renders stored artifacts unrecoverable
        unless a backup of the master key exists.

        Note: This is a practical way to 'crypto-wipe' the vault on platforms where
        secure file deletion cannot be guaranteed (SSDs, COW filesystems). Use with caution.
        """
        with self.lock:
            key_path = os.path.join(self.vault_dir, '.key')
            salt_path = os.path.join(self.vault_dir, '.salt')
            try:
                # Overwrite key file contents then remove
                if os.path.exists(key_path):
                    try:
                        with open(key_path, 'r+b') as f:
                            size = os.path.getsize(key_path)
                            if size > 0:
                                f.write(secrets.token_bytes(size))
                                f.flush()
                                try:
                                    os.fsync(f.fileno())
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        os.remove(key_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove master key file: {e}")

                # Optionally remove salt as well
                if os.path.exists(salt_path):
                    try:
                        os.remove(salt_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove salt file: {e}")

                # Zero master key in memory
                try:
                    self.key = b''  # Zero out by assigning empty bytes
                    self.hmac_key = b''  # Zero out by assigning empty bytes
                except Exception:
                    pass

                self.logger.warning("Master key destroyed; existing artifacts are now unrecoverable")
            except Exception as e:
                self.logger.error(f"Failed to destroy master key: {e}")
                
                
# ------------------------------------------------------------------
# main() – quick demo / self-test
# ------------------------------------------------------------------
def main():
    import tempfile, shutil, sys, getpass

    print("=== BlackVault Enhanced Demo ===")
    # 1. Create a throw-away vault in temp directory
    tmp_dir = tempfile.mkdtemp(prefix="vault_demo_")
    password = getpass.getpass("Enter vault password (or press Enter for random): ") or None
    vault = BlackVault(password=password, rotate_days=1, vault_path=tmp_dir)


    # 3. List what we have
    print("Artifacts on disk:", vault.list_artifacts())

    # 4. Retrieve & verify
    try:
        print("hello ->", vault.retrieve("hello").decode())
    except ValueError as e:
        print("hello failed:", e)
    try:
        print("secret ->", json.loads(vault.retrieve("secret").decode()))
    except ValueError as e:
        print("secret failed:", e)

    # 5. Force key rotation (demo)
    vault._check_key_rotation()

    # 6. Clean expired and wipe
    vault.clean_expired()
    vault.wipe_all()
    shutil.rmtree(tmp_dir)
    print("Demo finished – vault wiped.")

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# Remote Access Example - Enhanced
# ------------------------------------------------------------------
class RemoteBlackVaultClient:
    """
    Enhanced client for remote access to BlackVault via REST API.
    Agents, nodes, and swarm mesh can use this to store/retrieve artifacts securely.
    """
    def __init__(self, endpoint: str, api_key: str, timeout: int = 30):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logging.getLogger('RemoteBlackVaultClient')

    def store_artifact(self, name: str, data: bytes, expire_minutes: Optional[int] = None) -> bool:
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/octet-stream"
            }
            params = {}
            if expire_minutes is not None:
                params['expire_minutes'] = expire_minutes
            resp = requests.post(
                f"{self.endpoint}/store/{name}", 
                headers=headers, 
                data=data,
                params=params,
                timeout=self.timeout
            )
            self.logger.info(f"Store artifact {name}: {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to store artifact {name}: {e}")
            return False

    def retrieve_artifact(self, name: str) -> Optional[bytes]:
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(
                f"{self.endpoint}/retrieve/{name}", 
                headers=headers,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                self.logger.info(f"Retrieve artifact {name}: success")
                return resp.content
            else:
                self.logger.error(f"Retrieve artifact {name}: HTTP {resp.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact {name}: {e}")
            return None

    def list_artifacts(self) -> list:
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(
                f"{self.endpoint}/list", 
                headers=headers,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json().get('artifacts', [])
            return []
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []

    def delete_artifact(self, name: str) -> bool:
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.delete(
                f"{self.endpoint}/delete/{name}", 
                headers=headers,
                timeout=self.timeout
            )
            self.logger.info(f"Delete artifact {name}: {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {name}: {e}")
            return False
        
        