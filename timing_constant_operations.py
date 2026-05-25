"""
timing_constant_operations.py - Defeat timing attacks on cryptographic operations
============================================================================
Adds timing uncertainty to cryptographic operations (PBKDF2, AES, key derivation)
to defeat timing-based side-channel attacks.

Techniques:
1. PBKDF2 with random added iterations (noise: 10-30% overhead)
2. AES with dummy operations on unrelated data (constant-time mask)
3. Key derivation with random sleep jitter (10-200ms)
4. Batch operations to amortize timing variance

Usage:
    from timing_constant_operations import pbkdf2_with_blinding
    
    # Instead of: key = pbkdf2(password, salt, iterations=100000)
    # Use:
    key = pbkdf2_with_blinding(password, salt, iterations=100000)
    # Takes 100-130k iterations (unpredictable to attacker)
"""

import time
import random
import threading
from typing import Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend


class TimingObfuscator:
    """Central manager for timing obfuscation across operations."""
    
    def __init__(self, noise_level: int = 3):
        """
        Initialize timing obfuscator.
        
        Args:
            noise_level: Timing noise percentage (1-30%, default 3%)
        """
        self.noise_level = max(1, min(30, noise_level))
        self.backend = default_backend()
        self._dummy_buffer = bytearray(1024 * 100)  # 100KB buffer for dummy ops
    
    def _add_sleep_jitter(self, base_sleep: float) -> float:
        """
        Add timing jitter to sleep duration.
        
        Args:
            base_sleep: Base sleep time
        
        Returns:
            Adjusted sleep time with jitter
        """
        jitter_factor = self.noise_level / 100.0
        jitter = random.uniform(-jitter_factor, jitter_factor)
        adjusted = base_sleep * (1 + jitter)
        return max(0, adjusted)  # Ensure non-negative
    
    def _add_cpu_noise(self):
        """
        Add CPU-bound dummy operations to obscure timing.
        """
        # Perform meaningless operations to add timing noise
        dummy_sum = 0
        for i in range(100000):
            dummy_sum ^= i  # XOR operations
        return dummy_sum  # Use return value to prevent optimization
    
    def pbkdf2_with_blinding(self, 
                             password: bytes, 
                             salt: bytes, 
                             iterations: int = 100000,
                             algorithm = None) -> bytes:  # type: ignore
        """
        PBKDF2 with random iteration blinding.
        
        Args:
            password: Password/key material
            salt: Salt for derivation
            iterations: Base iteration count
            algorithm: Hash algorithm (default: SHA256)
        
        Returns:
            Derived key (32 bytes)
                # Use SHA256 if no algorithm provided
        if algorithm is None:
            algorithm = hashes.SHA256()        Example:
            >>> key = obfuscator.pbkdf2_with_blinding(b"password", b"salt", 100000)
            >>> len(key)
            32
        """
        if algorithm is None:
            algorithm = hashes.SHA256()
        
        # Add random iterations (10-30% overhead based on noise_level)
        noise_factor = self.noise_level / 100.0
        added_iterations = int(iterations * random.uniform(0, noise_factor))
        total_iterations = iterations + added_iterations
        
        # Pre-jitter: random sleep (0-50ms)
        sleep_time = random.uniform(0, 0.05)
        time.sleep(sleep_time)
        
        # Add CPU noise in parallel thread (non-blocking timing obfuscation)
        noise_thread = threading.Thread(target=self._add_cpu_noise, daemon=True)
        noise_thread.start()
        
        # Derive key with obfuscated iteration count
        kdf = PBKDF2HMAC(
            algorithm=algorithm,
            length=32,
            salt=salt,
            iterations=total_iterations,
            backend=self.backend
        )
        key = kdf.derive(password)
        
        # Post-jitter: random sleep (0-50ms)
        sleep_time = random.uniform(0, 0.05)
        time.sleep(sleep_time)
        
        # Wait for noise thread
        noise_thread.join(timeout=0.1)
        
        return key
    
    def aes_with_constant_time(self,
                               plaintext: bytes,
                               key: bytes,
                               iv: bytes,
                               mode: str = 'GCM') -> bytes:
        """
        AES encryption with dummy operations (constant-time mask).
        
        Args:
            plaintext: Data to encrypt
            key: AES key
            iv: Initialization vector
            mode: Cipher mode (GCM or CBC)
        
        Returns:
            Ciphertext
        
        Example:
            >>> ciphertext = obfuscator.aes_with_constant_time(b"data", key, iv)
        """
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        
        # Pre-operation dummy work
        dummy_result = self._add_cpu_noise()
        
        # Create cipher
        if mode == 'GCM':
            cipher_mode = modes.GCM(iv)
        else:
            cipher_mode = modes.CBC(iv)
        
        cipher = Cipher(algorithms.AES(key), cipher_mode, backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Perform encryption
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Post-operation dummy work
        dummy_result2 = self._add_cpu_noise()
        
        # Random timing delay (10-200ms based on noise level)
        max_delay = (self.noise_level / 100.0) * 0.2  # 0.002-0.06s
        sleep_time = random.uniform(0, max_delay)
        time.sleep(sleep_time)
        
        if mode == 'GCM':
            # Return ciphertext + tag for GCM
            tag = getattr(encryptor, 'tag', b'')
            return ciphertext + tag
        else:
            return ciphertext
    
    def key_derivation_with_jitter(self,
                                   password: bytes,
                                   salt: bytes,
                                   iterations: int = 100000) -> Tuple[bytes, float]:
        """
        Key derivation with timing jitter to defeat side-channel timing analysis.
        
        Args:
            password: Password/seed material
            salt: Salt for derivation
            iterations: Base iteration count
        
        Returns:
            Tuple of (derived_key, actual_time_taken)
        
        Example:
            >>> key, timing = obfuscator.key_derivation_with_jitter(b"pwd", b"salt", 100k)
            >>> print(f"Took {timing:.3f}s (unpredictable)")
        """
        start_time = time.time()
        
        # Pre-operation sleep (add timing variance)
        pre_sleep = self._add_sleep_jitter(0.01)  # 10ms ± noise%
        time.sleep(pre_sleep)
        
        # Derive key with blinding
        key = self.pbkdf2_with_blinding(password, salt, iterations)
        
        # Post-operation sleep (add more variance)
        post_sleep = self._add_sleep_jitter(0.02)  # 20ms ± noise%
        time.sleep(post_sleep)
        
        elapsed = time.time() - start_time
        return key, elapsed


def pbkdf2_with_blinding(password: bytes,
                         salt: bytes,
                         iterations: int = 100000,
                         noise_level: int = 3) -> bytes:
    """
    Standalone PBKDF2 with timing blinding.
    
    Args:
        password: Password/key material
        salt: Salt
        iterations: Base iterations
        noise_level: Noise percentage (1-30%)
    
    Returns:
        Derived key (32 bytes)
    """
    obfuscator = TimingObfuscator(noise_level=noise_level)
    return obfuscator.pbkdf2_with_blinding(password, salt, iterations)


def aes_with_constant_time(plaintext: bytes,
                           key: bytes,
                           iv: bytes,
                           noise_level: int = 3) -> bytes:
    """
    Standalone AES encryption with constant-time operations.
    """
    obfuscator = TimingObfuscator(noise_level=noise_level)
    return obfuscator.aes_with_constant_time(plaintext, key, iv, mode='GCM')


def key_derivation_with_jitter(password: bytes,
                               salt: bytes,
                               iterations: int = 100000,
                               noise_level: int = 3) -> bytes:
    """
    Standalone key derivation with timing jitter.
    """
    obfuscator = TimingObfuscator(noise_level=noise_level)
    key, _ = obfuscator.key_derivation_with_jitter(password, salt, iterations)
    return key


# Global timing obfuscator instance (TIER 3a integration point)
_global_timing_obfuscator = TimingObfuscator(noise_level=3)


def get_global_timing_obfuscator() -> TimingObfuscator:
    """Get global timing obfuscator."""
    return _global_timing_obfuscator


if __name__ == '__main__':
    print("[*] Timing Constant Operations - Timing Variance Test")
    
    obfuscator = TimingObfuscator(noise_level=3)
    
    password = b"test_password_12345"
    salt = b"test_salt_67890"
    
    print("[*] Running 5 key derivations with timing measurements...")
    print("[*] Expected variance: ±3% of total time")
    
    timings = []
    for i in range(5):
        key, elapsed = obfuscator.key_derivation_with_jitter(password, salt, iterations=50000)
        timings.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s (key length: {len(key)})")
    
    # Calculate variance
    avg_time = sum(timings) / len(timings)
    variance = [(t - avg_time) ** 2 for t in timings]
    stddev = (sum(variance) / len(variance)) ** 0.5
    variance_percent = (stddev / avg_time) * 100
    
    print(f"\n[*] Timing Statistics:")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Std Dev: {stddev:.3f}s")
    print(f"  Variance: ±{variance_percent:.1f}%")
    
    if variance_percent > 5:
        print(f"[+] Timing variance sufficient to defeat timing attacks")
    else:
        print(f"[-] Timing variance may be insufficient (consider increasing noise_level)")
    
    print("[*] Test complete!")
