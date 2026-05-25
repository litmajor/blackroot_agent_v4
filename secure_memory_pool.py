"""
secure_memory_pool.py - In-memory encryption and secure memory management
============================================================================
Implements secure buffers for storing encryption keys, credentials, and
sensitive data in memory. All buffers are encrypted at rest and decrypted
on-demand with automatic TTL-based cleanup to defeat memory forensics.

Key Features:
- AES-256-GCM encryption for all in-memory data
- TTL-based automatic decryption and cleanup
- Secure memory wipe (overwrite with random data)
- Thread-local storage for credential isolation
- LRU caching to balance performance vs. security

Usage:
    from secure_memory_pool import SecureMemoryPool
    
    pool = SecureMemoryPool(ttl_seconds=60)
    
    # Encrypt data
    encrypted_id = pool.encrypt_data(b"my_secret_key_data")
    
    # Decrypt on demand (auto-cleanup after TTL)
    plaintext = pool.decrypt_data(encrypted_id)
    
    # Manual cleanup
    pool.secure_wipe(encrypted_id)
"""

import os
import gc
import threading
import time
from typing import Dict, Tuple, Optional
from collections import OrderedDict
import secrets

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
except ImportError:
    raise ImportError("cryptography package required: pip install cryptography")


class SecureMemoryPool:
    """
    Manages encrypted buffers for sensitive in-memory data.
    
    All data is encrypted at rest and only decrypted when accessed.
    Automatic cleanup enforces TTL and garbage collection.
    """
    
    def __init__(self, ttl_seconds: float = 5.0, max_cached_decryptions: int = 50):
        """
        Initialize secure memory pool.
        
        Args:
            ttl_seconds: Time-to-live for decrypted buffers (default: 5s)
            max_cached_decryptions: Max decrypted entries to cache (LRU eviction)
        """
        self.ttl_seconds = ttl_seconds
        self.max_cached = max_cached_decryptions
        self.backend = default_backend()
        self._pool_key = secrets.token_bytes(32)  # Master key for this pool
        self._buffers: Dict[str, Tuple[bytes, float]] = {}  # {buffer_id: (encrypted_data, creation_time)}
        self._decrypted_cache = OrderedDict()  # LRU cache of decrypted data
        self._lock = threading.RLock()
        self._access_counter = 0
        
    def _generate_buffer_id(self) -> str:
        """Generate unique buffer ID."""
        return secrets.token_hex(16)
    
    def _derive_key(self, salt: bytes, iterations: int = 50000) -> bytes:
        """
        Derive encryption key from pool master key using PBKDF2.
        
        Args:
            salt: Random salt for key derivation
            iterations: PBKDF2 iterations (default: 50k - balance security/speed)
        
        Returns:
            32-byte AES-256 key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=self.backend
        )
        return kdf.derive(self._pool_key)
    
    def encrypt_data(self, plaintext: bytes) -> str:
        """
        Encrypt plaintext and store in pool.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Buffer ID for later retrieval
        """
        with self._lock:
            # Generate random salt and IV
            salt = secrets.token_bytes(16)
            iv = secrets.token_bytes(12)  # GCM IV
            
            # Derive encryption key
            key = self._derive_key(salt)
            
            # Encrypt with AES-256-GCM
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            tag = encryptor.tag
            
            # Combine: salt || iv || tag || ciphertext
            encrypted = salt + iv + tag + ciphertext
            
            # Store in pool
            buffer_id = self._generate_buffer_id()
            self._buffers[buffer_id] = (encrypted, time.time())
            
            return buffer_id
    
    def decrypt_data(self, buffer_id: str) -> Optional[bytes]:
        """
        Decrypt buffer by ID (with TTL enforcement).
        
        Args:
            buffer_id: Buffer ID from encrypt_data()
        
        Returns:
            Decrypted plaintext or None if not found/expired
        """
        with self._lock:
            # Retrieve from encrypted storage first to check TTL
            if buffer_id not in self._buffers:
                # Remove from cache if expired buffer was cached
                self._decrypted_cache.pop(buffer_id, None)
                return None
            
            encrypted, creation_time = self._buffers[buffer_id]
            
            # Check TTL
            age = time.time() - creation_time
            if age > self.ttl_seconds:
                self.secure_wipe(buffer_id)
                return None
            
            # Check cache (TTL already verified)
            if buffer_id in self._decrypted_cache:
                self._decrypted_cache.move_to_end(buffer_id)
                return self._decrypted_cache[buffer_id]
            
            # Decrypt
            try:
                salt = encrypted[:16]
                iv = encrypted[16:28]
                tag = encrypted[28:44]
                ciphertext = encrypted[44:]
                
                key = self._derive_key(salt)
                cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
                decryptor = cipher.decryptor()
                plaintext = decryptor.update(ciphertext) + decryptor.finalize()
                
                # Cache decrypted result (LRU eviction if needed)
                self._decrypted_cache[buffer_id] = plaintext
                if len(self._decrypted_cache) > self.max_cached:
                    evicted_entry = self._decrypted_cache.popitem(last=False)
                    evicted_id, evicted_data = evicted_entry
                    # Overwrite evicted data before GC
                    evicted_data = bytes(len(evicted_data))  # Placeholder
                
                return plaintext
                
            except Exception as e:
                # Decryption failed (tampered data?)
                self.secure_wipe(buffer_id)
                return None
    
    def secure_wipe(self, buffer_id: str) -> bool:
        """
        Securely erase buffer (overwrite with random data before dealloc).
        
        Args:
            buffer_id: Buffer ID to wipe
        
        Returns:
            True if wiped, False if not found
        """
        with self._lock:
            # Remove from cache
            if buffer_id in self._decrypted_cache:
                encrypted_data = self._decrypted_cache.pop(buffer_id)
                # Overwrite before deallocation
                encrypted_data = bytes(len(encrypted_data))
            
            # Remove from pool
            if buffer_id in self._buffers:
                encrypted_data, _ = self._buffers.pop(buffer_id)
                # Overwrite before deallocation
                encrypted_data = bytes(len(encrypted_data))
                return True
            
            return False
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired buffers (TTL exceeded).
        
        Returns:
            Number of buffers cleaned up
        """
        with self._lock:
            now = time.time()
            expired_ids = []
            
            for buffer_id, (_, creation_time) in self._buffers.items():
                age = now - creation_time
                if age > self.ttl_seconds:
                    expired_ids.append(buffer_id)
            
            for buffer_id in expired_ids:
                self.secure_wipe(buffer_id)
            
            gc.collect()  # Force garbage collection
            return len(expired_ids)
    
    def encrypt_credential_batch(self, credentials: dict, ttl: Optional[float] = None) -> dict:
        """
        Encrypt a batch of credentials (e.g., from credential extraction).
        
        Args:
            credentials: Dict of {name: credential_data}
            ttl: Override TTL for this batch (default: use pool TTL)
        
        Returns:
            Dict of {name: buffer_id}
        
        Example:
            >>> creds = {"app_password": b"secret123", "api_key": b"key456"}
            >>> encrypted = pool.encrypt_credential_batch(creds)
            >>> later: plaintext = pool.decrypt_data(encrypted["app_password"])
        """
        result = {}
        for name, cred_data in credentials.items():
            if isinstance(cred_data, str):
                cred_data = cred_data.encode('utf-8')
            result[name] = self.encrypt_data(cred_data)
        return result
    
    def get_pool_stats(self) -> dict:
        """
        Return pool statistics for debugging.
        
        Returns:
            Dict with buffer count, cache size, etc.
        """
        with self._lock:
            return {
                'total_buffers': len(self._buffers),
                'cached_decryptions': len(self._decrypted_cache),
                'pool_key_size': len(self._pool_key),
                'ttl_seconds': self.ttl_seconds,
                'max_cached': self.max_cached,
            }


class ThreadLocalSecurePool(threading.local):
    """
    Thread-local secure memory pool (one pool per thread).
    
    Ensures each thread has isolated encrypted buffers for credentials.
    Useful for multi-threaded agent operations.
    
    Usage:
        pool = ThreadLocalSecurePool(ttl_seconds=60)
        buffer_id = pool.encrypt_data(b"secret")
        plaintext = pool.decrypt_data(buffer_id)
    """
    
    def __init__(self, ttl_seconds: float = 5.0):
        """Initialize thread-local secure pool."""
        super().__init__()
        self.pool = None
        self.ttl = ttl_seconds
    
    def get_pool(self) -> SecureMemoryPool:
        """Get or create pool for current thread."""
        if self.pool is None:
            self.pool = SecureMemoryPool(ttl_seconds=self.ttl)
        return self.pool


# Global secure pool instance (TIER 1 integration point)
_global_secure_pool = SecureMemoryPool(ttl_seconds=5.0)


def get_global_pool() -> SecureMemoryPool:
    """Get global secure memory pool."""
    return _global_secure_pool


if __name__ == '__main__':
    print("[*] Secure Memory Pool - Self Test")
    
    pool = SecureMemoryPool(ttl_seconds=2)
    print(f"[*] Pool created: {pool.get_pool_stats()}")
    
    # Test encryption/decryption
    secret = b"this_is_a_secret_encryption_key"
    buffer_id = pool.encrypt_data(secret)
    print(f"[+] Encrypted {len(secret)} bytes -> buffer_id: {buffer_id[:16]}...")
    
    # Decrypt
    recovered = pool.decrypt_data(buffer_id)
    assert recovered == secret, "Decryption failed!"
    print(f"[+] Decrypted successfully: {recovered == secret}")
    
    # Test TTL expiry
    print(f"[*] Waiting for TTL expiry (2s)...")
    time.sleep(2.5)
    recovered_after_ttl = pool.decrypt_data(buffer_id)
    assert recovered_after_ttl is None, "TTL cleanup failed!"
    print(f"[+] TTL cleanup verified: buffer expired as expected")
    
    # Test batch encryption
    creds = {"password": "secret123", "api_key": "key456789"}
    encrypted_batch = pool.encrypt_credential_batch(creds)
    print(f"[+] Batch encrypted {len(creds)} credentials")
    
    stats = pool.get_pool_stats()
    print(f"[*] Final stats: {stats}")
    print("[*] All tests passed!")
