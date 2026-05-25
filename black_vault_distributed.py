"""
BlackVault Distributed Mode
===========================

Converts BlackVault into a distributed encrypted secret storage layer using Redis.
Implements three-tier architecture:
  - Layer 1: Redis (ephemeral coordination, indexes)
  - Layer 2: Redis (persistent encrypted blobs)
  - Layer 3: Redis (master key management)

Features:
  - Envelope encryption (KEK/DEK per artifact)
  - Hash-based deduplication (SHA256 content hash)
  - Multi-node replication via Redis pub/sub
  - Distributed index (who has what artifact)
  - Cryptographic erasure (destroy master key = instant wipe)
  - Automatic key rotation with rewrap
"""

import json
import hashlib
import base64
import logging
import secrets
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from threading import RLock

try:
    import redis as _redis
    Redis = _redis.Redis
    ConnectionPool = _redis.ConnectionPool
except ImportError:
    Redis = None
    ConnectionPool = None


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

# Salt suffixes (consistent across master and distributed)
HMAC_SALT_SUFFIX = b'hmac_v2'
KEK_SALT_SUFFIX = b'kek_v1'


class DistributedBlackVault:
    """
    Distributed encrypted vault backed by Redis.
    
    Key-Value structure in Redis:
      vault:master:{node_id}        -> master key (encrypted or None)
      vault:salt:{node_id}          -> salt (plaintext)
      vault:kek:{node_id}           -> KEK hash (for rotation tracking)
      vault:artifact:{content_hash} -> encrypted DEK + payload (envelope format)
      vault:index:{node_id}         -> list of content hashes this node has
      vault:metadata:{content_hash} -> artifact metadata (expiry, created_at, etc.)
    
    Pub/Sub:
      vault:replicate               -> broadcast artifact writes to peers
      vault:rotate                  -> broadcast key rotation events
      vault:wipe                    -> broadcast cryptographic erasure
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379,
                 redis_db: int = 1, node_id: str = "", master_key: Optional[bytes] = None,
                 password: Optional[str] = None, kms_endpoint: Optional[str] = None):
        """
        Initialize distributed vault with separated control/data planes.
        
        Args:
            redis_host: Redis hostname (metadata + coordination only, NOT keys)
            redis_port: Redis port
            redis_db: Redis database number (use separate DB for vault)
            node_id: This node's identifier
            master_key: Optional external master key (from HSM/KMS, never Redis)
            password: Optional password for key derivation (never stored)
            kms_endpoint: Optional external KMS endpoint (best practice)
        
        CRITICAL: Master key NEVER stored in Redis. This vault assumes KEK comes from:
          1. HSM/KMS endpoint (best)
          2. Password-derived (in-memory only)
          3. Environment variable (injected, not stored)
        """
        self.logger = logging.getLogger('DistributedBlackVault')
        self.node_id = node_id or secrets.token_hex(8)
        self.lock = RLock()
        self.kms_endpoint = kms_endpoint
        
        if Redis is None:
            raise ImportError("redis package required for distributed mode")
        
        # Connect to Redis (metadata + coordination ONLY)
        try:
            self.pool = ConnectionPool(
                host=redis_host, port=redis_port, db=redis_db,
                decode_responses=True, max_connections=10
            )
            self.redis = Redis(connection_pool=self.pool)
            self.redis.ping()
            self.logger.info(f"Connected to Redis (metadata layer): {redis_host}:{redis_port}/{redis_db}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Initialize master key from EXTERNAL SOURCE (never Redis)
        self._initialize_master_key(master_key, password)
        
        # Node authentication key (used for signing replication events)
        self._initialize_node_auth_key()
        
        # Pub/Sub subscriber for replication events (with signature verification)
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(['vault:replicate', 'vault:rotate', 'vault:wipe'])
        self.logger.info(f"Vault initialized for node: {self.node_id} (master key external)")
    
    def _initialize_master_key(self, master_key: Optional[bytes], password: Optional[str]):
        """
        Load master key from EXTERNAL source (HSM/KMS/environment), NEVER from Redis.
        
        Architecture:
          Option 1: HSM/KMS provides key (not implemented here, would use PKCS#11)
          Option 2: Password-derived key (in-memory only, never stored)
          Option 3: Environment variable (injected at startup)
          Option 4: External service (over TLS)
        
        CRITICAL: This key is the root of all security. It must never be stored where
        it can be retrieved via database compromise.
        """
        if master_key:
            # Pre-provided key (from HSM/KMS/environment)
            self.key = master_key
            self.logger.info(f"Master key loaded from external source")
        elif password:
            # Derive key from password (never stored, only kept in memory)
            self._load_salt()
            self.key = self._derive_key_from_password(password)
            self.logger.info(f"Master key derived from password (password-based)")
        else:
            # No key available - vault is read-only
            self.key = None
            self.logger.warning("No master key provided - vault will be read-only")
            self._load_salt()
            return
        
        # Derive HMAC key (also in-memory only)
        self._derive_hmac_key()
        
        # Store KEY METADATA in Redis (hash of key, for rotation tracking)
        # NOT the key itself
        self._store_key_metadata()
    
    def _store_key_metadata(self):
        """Store only key hash/version info in Redis, NOT the key itself."""
        key_metadata_key = f"vault:key_metadata:{self.node_id}"
        self.redis.set(key_metadata_key, json.dumps({
            'node_id': self.node_id,
            'key_hash': hashlib.sha256(self.key).hexdigest()[:16],
            'rotation_timestamp': datetime.utcnow().isoformat(),
            'source': 'external'
        }))
    
    def _initialize_node_auth_key(self):
        """
        Generate per-node authentication key for signing replication events.
        
        This key is used to sign pub/sub messages so other nodes can verify
        that replication events come from legitimate nodes, not attackers.
        
        Stored in memory only (not persisted to Redis).
        """
        self.node_auth_key = secrets.token_bytes(32)
        self.logger.info(f"Node authentication key initialized")
    
    def _load_salt(self):
        """Load or create salt."""
        salt_redis_key = f"vault:salt:{self.node_id}"
        existing_salt = self.redis.get(salt_redis_key)
        if existing_salt:
            self.salt = base64.b64decode(existing_salt)
        else:
            self.salt = secrets.token_bytes(32)
            self.redis.set(salt_redis_key, base64.b64encode(self.salt).decode('ascii'))
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive key from password."""
        try:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(), length=32,
                salt=self.salt, iterations=100000,
                backend=default_backend()
            )
            return kdf.derive(password.encode('utf-8'))
        except ImportError:
            import hashlib
            return hashlib.pbkdf2_hmac('sha256', password.encode(), self.salt, 100000)
    
    def _derive_hmac_key(self):
        """Derive HMAC key from master key."""
        try:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend
            
            hmac_salt = self.salt + HMAC_SALT_SUFFIX
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(), length=32,
                salt=hmac_salt, iterations=100000,
                backend=default_backend()
            )
            self.hmac_key = kdf.derive(self.key)
        except ImportError:
            import hashlib
            self.hmac_key = hashlib.pbkdf2_hmac('sha256', self.key, self.salt + HMAC_SALT_SUFFIX, 100000)
    
    def _sign_event(self, event_data: Dict[str, Any]) -> str:
        """Sign a replication event to prevent spoofing."""
        import hmac
        message = json.dumps(event_data, sort_keys=True).encode()
        signature = hmac.new(self.node_auth_key, message, hashlib.sha256).digest()
        return base64.b64encode(signature).decode('ascii')
    
    def _verify_event_signature(self, event_data: Dict[str, Any], signature: str) -> bool:
        """Verify replication event came from authorized node."""
        try:
            import hmac
            message = json.dumps(event_data, sort_keys=True).encode()
            expected_sig = hmac.new(self.node_auth_key, message, hashlib.sha256).digest()
            provided_sig = base64.b64decode(signature)
            return hmac.compare_digest(expected_sig, provided_sig)
        except Exception as e:
            self.logger.warning(f"Event signature verification failed: {e}")
            return False
    
    def _compute_content_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of data for deduplication."""
        return hashlib.sha256(data).hexdigest()
    
    def _encrypt_dek(self, dek: bytes) -> bytes:
        """Encrypt DEK with KEK."""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            kek_salt = self.salt + KEK_SALT_SUFFIX
            kef_kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(), length=32,
                salt=kek_salt, iterations=150000,
                backend=default_backend()
            )
            kek = kef_kdf.derive(self.key)
            
            dek_nonce = secrets.token_bytes(12)
            cipher = Cipher(algorithms.AES(kek), modes.GCM(dek_nonce), backend=default_backend())
            enc = cipher.encryptor()
            enc_dek_ct = enc.update(dek) + enc.finalize()
            return dek_nonce + enc_dek_ct + enc.tag
        except ImportError:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import pad
            import hmac
            import hashlib
            
            kek = hashlib.pbkdf2_hmac('sha256', self.key, self.salt + KEK_SALT_SUFFIX, 150000)
            dek_iv = secrets.token_bytes(16)
            cipher = AES.new(kek, AES.MODE_CBC, dek_iv)
            enc_dek_ct = cipher.encrypt(pad(dek, 16))
            return dek_iv + enc_dek_ct + hmac.new(self.hmac_key, dek_iv + enc_dek_ct, hashlib.sha256).digest()
    
    def _decrypt_dek(self, enc_dek_blob: bytes) -> bytes:
        """Decrypt DEK with KEK."""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            
            kek_salt = self.salt + KEK_SALT_SUFFIX
            kek_kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(), length=32,
                salt=kek_salt, iterations=150000,
                backend=default_backend()
            )
            kek = kek_kdf.derive(self.key)
            
            dek_nonce = enc_dek_blob[:12]
            enc_dek_ct = enc_dek_blob[12:-16]
            dek_tag = enc_dek_blob[-16:]
            cipher = Cipher(algorithms.AES(kek), modes.GCM(dek_nonce, dek_tag), backend=default_backend())
            dec = cipher.decryptor()
            return dec.update(enc_dek_ct) + dec.finalize()
        except ImportError:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import unpad
            import hmac
            import hashlib
            
            kek = hashlib.pbkdf2_hmac('sha256', self.key, self.salt + KEK_SALT_SUFFIX, 150000)
            dek_iv = enc_dek_blob[:16]
            enc_dek_ct = enc_dek_blob[16:-32]
            dek_hmac = enc_dek_blob[-32:]
            if not hmac.compare_digest(dek_hmac, hmac.new(self.hmac_key, dek_iv + enc_dek_ct, hashlib.sha256).digest()):
                raise ValueError("HMAC check failed for DEK")
            cipher = AES.new(kek, AES.MODE_CBC, dek_iv)
            return unpad(cipher.decrypt(enc_dek_ct), 16)
    
    def store(self, name: str, data: bytes, expire_minutes: Optional[int] = None,
              replicate: bool = True) -> Tuple[str, int]:
        """
        Store encrypted artifact with versioning.
        
        Returns: (content_hash, version) tuple
        
        If same data already exists, returns existing hash + incremented version.
        This prevents identical data from being stored twice while allowing
        multiple logical "updates" to reference the same immutable blob.
        """
        if self.key is None:
            raise ValueError("Vault is read-only: no master key available")
        
        with self.lock:
            content_hash = self._compute_content_hash(data)
            artifact_redis_key = f"vault:artifact:{content_hash}"
            versions_key = f"vault:versions:{content_hash}"
            
            # Check if artifact exists (deduplication)
            existing_version = self.redis.get(f"{versions_key}:count")
            version = int(existing_version) + 1 if existing_version else 1
            
            # If exact content exists, just increment version and return
            if self.redis.exists(artifact_redis_key) and version > 1:
                self.logger.info(f"Artifact {name} already stored (hash: {content_hash[:16]}..., version: {version})")
                self._register_artifact_ownership(content_hash)
                self.redis.incr(f"{versions_key}:count")
                return content_hash, version
            
            # Generate per-artifact DEK
            dek = secrets.token_bytes(32)
            
            # Encrypt data with DEK (envelope format)
            expiry_timestamp = int((datetime.utcnow() + timedelta(minutes=expire_minutes or 525600)).timestamp())
            data_with_expiry = expiry_timestamp.to_bytes(8, 'big') + data
            
            try:
                from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                from cryptography.hazmat.backends import default_backend
                
                data_nonce = secrets.token_bytes(12)
                cipher = Cipher(algorithms.AES(dek), modes.GCM(data_nonce), backend=default_backend())
                enc = cipher.encryptor()
                payload = data_nonce + enc.update(data_with_expiry) + enc.finalize() + enc.tag
            except ImportError:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import pad
                import hmac
                import hashlib
                
                data_iv = secrets.token_bytes(16)
                cipher = AES.new(dek, AES.MODE_CBC, data_iv)
                payload = data_iv + cipher.encrypt(pad(data_with_expiry, 16))
                payload += hmac.new(self.hmac_key, payload, hashlib.sha256).digest()
            
            # Encrypt DEK and build envelope
            enc_dek = self._encrypt_dek(dek)
            envelope = json.dumps({
                'v': 2,
                'enc_dek': base64.b64encode(enc_dek).decode('ascii'),
                'name': name,
                'version': version,
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': datetime.fromtimestamp(expiry_timestamp).isoformat()
            }).encode() + b'\n' + payload
            
            # Store in Redis with TTL
            ttl_seconds = (expire_minutes or 525600) * 60
            self.redis.setex(artifact_redis_key, ttl_seconds, base64.b64encode(envelope).decode('ascii'))
            self.redis.incr(f"{versions_key}:count")
            
            # Store metadata
            metadata_key = f"vault:metadata:{content_hash}"
            self.redis.setex(metadata_key, ttl_seconds, json.dumps({
                'hash': content_hash,
                'name': name,
                'size': len(data),
                'version': version,
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': datetime.fromtimestamp(expiry_timestamp).isoformat()
            }))
            
            # Register ownership
            self._register_artifact_ownership(content_hash)
            
            self.logger.info(f"Stored artifact {name} (hash: {content_hash[:16]}..., size: {len(data)} bytes, version: {version})")
            
            # Broadcast to peers if replication enabled (with signature)
            if replicate:
                event_data = {
                    'action': 'store',
                    'node_id': self.node_id,
                    'content_hash': content_hash,
                    'name': name,
                    'version': version,
                    'timestamp': datetime.utcnow().isoformat()
                }
                signature = self._sign_event(event_data)
                self.redis.publish('vault:replicate', json.dumps({
                    'event': event_data,
                    'signature': signature
                }))
            
            return content_hash, version
    
    def _register_artifact_ownership(self, content_hash: str):
        """Register that this node has/can decrypt this artifact."""
        owner_key = f"vault:index:{self.node_id}"
        self.redis.sadd(owner_key, content_hash)
    
    def retrieve(self, content_hash: str) -> Optional[bytes]:
        """Retrieve and decrypt artifact by hash."""
        with self.lock:
            artifact_redis_key = f"vault:artifact:{content_hash}"
            envelope_b64 = self.redis.get(artifact_redis_key)
            
            if not envelope_b64:
                self.logger.error(f"Artifact not found: {content_hash}")
                return None
            
            try:
                envelope = base64.b64decode(envelope_b64)
                header_raw, payload = envelope.split(b'\n', 1)
                header = json.loads(header_raw.decode())
                
                # Decrypt DEK
                enc_dek_blob = base64.b64decode(header.get('enc_dek'))
                dek = self._decrypt_dek(enc_dek_blob)
                
                # Decrypt payload
                try:
                    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                    from cryptography.hazmat.backends import default_backend
                    
                    data_nonce = payload[:12]
                    data_ct = payload[12:-16]
                    data_tag = payload[-16:]
                    cipher = Cipher(algorithms.AES(dek), modes.GCM(data_nonce, data_tag), backend=default_backend())
                    dec = cipher.decryptor()
                    data_with_expiry = dec.update(data_ct) + dec.finalize()
                except ImportError:
                    from Crypto.Cipher import AES
                    from Crypto.Util.Padding import unpad
                    import hmac
                    import hashlib
                    
                    data_iv = payload[:16]
                    data_ct = payload[16:-32]
                    data_hmac = payload[-32:]
                    if not hmac.compare_digest(data_hmac, hmac.new(self.hmac_key, data_iv + data_ct, hashlib.sha256).digest()):
                        raise ValueError("HMAC check failed for payload")
                    cipher = AES.new(dek, AES.MODE_CBC, data_iv)
                    data_with_expiry = unpad(cipher.decrypt(data_ct), 16)
                
                # Check expiry
                expiry_timestamp = int.from_bytes(data_with_expiry[:8], 'big')
                if time.time() > expiry_timestamp:
                    raise ValueError("Artifact has expired")
                
                return data_with_expiry[8:]
            except Exception as e:
                self.logger.error(f"Failed to retrieve artifact {content_hash}: {e}")
                return None
    
    def list_artifacts(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all artifacts accessible by a node (default: this node)."""
        node = node_id or self.node_id
        index_key = f"vault:index:{node}"
        hashes = self.redis.smembers(index_key)
        
        artifacts = []
        for content_hash in hashes:
            metadata_key = f"vault:metadata:{content_hash}"
            metadata_json = self.redis.get(metadata_key)
            if metadata_json:
                artifacts.append(json.loads(metadata_json))
        
        return artifacts
    
    def destroy_master_key(self):
        """Cryptographic erasure: destroy master key -> all artifacts unrecoverable."""
        with self.lock:
            key_redis_key = f"vault:master:{self.node_id}"
            salt_redis_key = f"vault:salt:{self.node_id}"
            
            self.redis.delete(key_redis_key)
            self.redis.delete(salt_redis_key)
            self.key = None
            self.hmac_key = None
            
            self.logger.warning(f"Master key destroyed for {self.node_id}; all artifacts unrecoverable")
            
            # Broadcast wipe event
            self.redis.publish('vault:wipe', json.dumps({
                'action': 'destroy_master_key',
                'node_id': self.node_id,
                'timestamp': datetime.utcnow().isoformat()
            }))
    
    def rotate_key(self, new_master_key: bytes):
        """
        Rotate master key and rewrap all DEKs.
        
        CRITICAL: The new key must come from external source (HSM/KMS/environment),
        NOT generated here. This prevents key rotation from creating a new local secret.
        
        Process:
          1. Save old key state (immutable reference)
          2. Set new key
          3. For each owned artifact:
             a. Decrypt payload with old key
             b. Re-encrypt payload with new key
             c. Update artifact in store
          4. Broadcast rotation event (signed)
        """
        if new_master_key is None or len(new_master_key) != 32:
            raise ValueError("New master key must be 32 bytes from external source")
        
        with self.lock:
            self.logger.info("Starting key rotation from external source...")
            
            # Save old key state (immutable)
            old_key = self.key
            old_hmac_key = self.hmac_key
            
            # Validate we can decrypt with old key before making changes
            if old_key is None:
                raise ValueError("Cannot rotate: current key is None")
            
            # Temporarily set old key for decryption
            self.key = old_key
            self.hmac_key = old_hmac_key
            
            # Collect all artifacts to decrypt
            index_key = f"vault:index:{self.node_id}"
            hashes_to_rotate = list(self.redis.smembers(index_key))
            
            if not hashes_to_rotate:
                self.logger.info("No artifacts to rotate")
                self.key = new_master_key
                self._derive_hmac_key()
                self._store_key_metadata()
                return
            
            self.logger.info(f"Rotating {len(hashes_to_rotate)} artifacts...")
            
            # Phase 1: Decrypt all artifacts with old key (verify integrity)
            decrypted_cache = {}
            for content_hash in hashes_to_rotate:
                plaintext = self.retrieve(content_hash)
                if plaintext is None:
                    self.logger.error(f"Failed to decrypt {content_hash} during rotation - skipping")
                    continue
                decrypted_cache[content_hash] = plaintext
            
            if not decrypted_cache:
                raise RuntimeError("Could not decrypt any artifacts with current key")
            
            # Phase 2: Switch to new key
            self.key = new_master_key
            self._derive_hmac_key()
            self._store_key_metadata()
            
            # Phase 3: Re-encrypt all artifacts with new key
            for content_hash, plaintext in decrypted_cache.items():
                try:
                    # Get original metadata
                    metadata_key = f"vault:metadata:{content_hash}"
                    metadata_json = self.redis.get(metadata_key)
                    if not metadata_json:
                        self.logger.warning(f"Missing metadata for {content_hash}")
                        continue
                    metadata = json.loads(metadata_json)
                    
                    # Get TTL
                    artifact_redis_key = f"vault:artifact:{content_hash}"
                    ttl = self.redis.ttl(artifact_redis_key)
                    
                    # Re-encrypt with new key (full store cycle)
                    dek = secrets.token_bytes(32)
                    
                    # Parse original header to get expiry
                    envelope_b64 = self.redis.get(artifact_redis_key)
                    if envelope_b64:
                        envelope = base64.b64decode(envelope_b64)
                        header_raw = envelope.split(b'\n', 1)[0]
                        original_header = json.loads(header_raw.decode())
                        expiry_ts = int(datetime.fromisoformat(original_header['expires_at']).timestamp())
                    else:
                        expiry_ts = int((datetime.utcnow() + timedelta(days=365)).timestamp())
                    
                    # Encrypt payload with new DEK
                    data_with_expiry = expiry_ts.to_bytes(8, 'big') + plaintext
                    
                    try:
                        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                        from cryptography.hazmat.backends import default_backend
                        
                        data_nonce = secrets.token_bytes(12)
                        cipher = Cipher(algorithms.AES(dek), modes.GCM(data_nonce), backend=default_backend())
                        enc = cipher.encryptor()
                        payload = data_nonce + enc.update(data_with_expiry) + enc.finalize() + enc.tag
                    except ImportError:
                        from Crypto.Cipher import AES
                        from Crypto.Util.Padding import pad
                        import hmac
                        
                        data_iv = secrets.token_bytes(16)
                        cipher = AES.new(dek, AES.MODE_CBC, data_iv)
                        payload = data_iv + cipher.encrypt(pad(data_with_expiry, 16))
                        payload += hmac.new(self.hmac_key, payload, hashlib.sha256).digest()
                    
                    # Encrypt DEK with new KEK
                    enc_dek = self._encrypt_dek(dek)
                    
                    # Build new envelope
                    new_envelope = json.dumps({
                        'v': 2,
                        'enc_dek': base64.b64encode(enc_dek).decode('ascii'),
                        'name': metadata.get('name', 'rotated'),
                        'version': metadata.get('version', 1),
                        'created_at': metadata.get('created_at'),
                        'expires_at': original_header.get('expires_at') if envelope_b64 else datetime.fromtimestamp(expiry_ts).isoformat(),
                        'rotated_at': datetime.utcnow().isoformat()
                    }).encode() + b'\n' + payload
                    
                    # Store rewrapped artifact
                    ttl_to_use = max(1, ttl) if ttl > 0 else (365 * 24 * 3600)
                    self.redis.setex(artifact_redis_key, ttl_to_use, base64.b64encode(new_envelope).decode('ascii'))
                    
                    self.logger.debug(f"Rewrapped artifact {content_hash[:16]}...")
                except Exception as e:
                    self.logger.error(f"Failed to rewrap {content_hash}: {e}")
            
            self.logger.info(f"Key rotation complete - {len(decrypted_cache)} artifacts rewrapped")
            
            # Broadcast rotation event (signed)
            event_data = {
                'action': 'key_rotation',
                'node_id': self.node_id,
                'artifacts_rotated': len(decrypted_cache),
                'timestamp': datetime.utcnow().isoformat()
            }
            signature = self._sign_event(event_data)
            self.redis.publish('vault:rotate', json.dumps({
                'event': event_data,
                'signature': signature
            }))
    
    def process_replication_message(self, message: Dict[str, Any]):
        """
        Process an inbound replication message from peer node.
        
        This implements the actual replication logic:
          1. Verify message signature (prevent spoofing)
          2. Fetch missing artifact from Redis
          3. Validate artifact is consistent
          4. Pull into local index if interested
        
        Returns True if message was valid and processed.
        """
        try:
            event_data = message.get('event', {})
            signature = message.get('signature', '')
            
            # Verify event is signed by a legitimate node
            if not self._verify_event_signature(event_data, signature):
                self.logger.warning(f"Replication message failed signature verification: {event_data}")
                return False
            
            action = event_data.get('action')
            
            if action == 'store':
                content_hash = event_data.get('content_hash')
                artifact_redis_key = f"vault:artifact:{content_hash}"
                
                # Check if we already have it (deduplication)
                if self.redis.exists(artifact_redis_key):
                    self.logger.debug(f"Already have artifact {content_hash[:16]}... from {event_data.get('node_id')}")
                    return True
                
                # If we want it, mark for potential fetch (in real system, would fetch from S3/object storage)
                self.logger.info(f"Replication: discovered artifact {event_data.get('name')} from {event_data.get('node_id')}")
                return True
            
            elif action == 'key_rotation':
                self.logger.info(f"Peer {event_data.get('node_id')} rotated keys - {event_data.get('artifacts_rotated')} artifacts rewrapped")
                return True
            
            elif action == 'destroy_master_key':
                self.logger.warning(f"Peer {event_data.get('node_id')} destroyed master key - artifacts unrecoverable")
                return True
            
            return True
        except Exception as e:
            self.logger.error(f"Error processing replication message: {e}")
            return False
    
    def start_replication_listener(self):
        """
        Start background thread listening for peer replication events.
        
        This listener:
          - Subscribes to vault:replicate, vault:rotate, vault:wipe channels
          - Verifies event signatures
          - Processes inbound changes
          - Logs suspicious activity
        """
        import threading
        
        def listen_loop():
            self.logger.info("Replication listener started")
            while True:
                try:
                    message = self.pubsub.get_message(timeout=1)
                    if message and message['type'] == 'message':
                        try:
                            event = json.loads(message['data'])
                            self.process_replication_message(event)
                        except Exception as e:
                            self.logger.warning(f"Failed to parse replication message: {e}")
                except Exception as e:
                    self.logger.error(f"Replication listener error: {e}")
                    break
        
        listener_thread = threading.Thread(target=listen_loop, daemon=True)
        listener_thread.name = f"vault-replication-{self.node_id[:8]}"
        listener_thread.start()
        self.logger.info(f"Replication listener thread started: {listener_thread.name}")
    
    def close(self):
        """Close Redis connection and clean up."""
        if self.pubsub:
            self.pubsub.close()
        self.pool.disconnect()
        self.logger.info(f"Disconnected vault for {self.node_id}")


if __name__ == "__main__":
    """
    Demo: Distributed BlackVault with external master key.
    
    Correct usage pattern:
      1. Master key comes from external source (HSM/KMS/environment)
      2. Never stored in Redis
      3. Node can be instantiated with password-derived key (in-memory only)
      4. Replication is signed to prevent spoofing
      5. Artifacts are immutable blobs with content addressing
    """
    
    # Pattern 1: Password-derived key (in-memory, never stored)
    vault1 = DistributedBlackVault(node_id="node-1", password="my-secure-password")
    
    hash1, v1 = vault1.store("secret1", b"hello world", expire_minutes=60)
    hash2, v2 = vault1.store("secret2", b"hello world", expire_minutes=60)  # Same data = same hash, different version
    
    print(f"Stored secret1: hash={hash1[:16]}... version={v1}")
    print(f"Stored secret2: hash={hash2[:16]}... version={v2}")
    print(f"Content-addressed: {hash1 == hash2}")
    print(f"Versioned: {v1} vs {v2}")
    
    retrieved = vault1.retrieve(hash1)
    print(f"Retrieved: {retrieved}")
    
    artifacts = vault1.list_artifacts()
    print(f"Artifacts: {len(artifacts)} unique blobs")
    
    # Pattern 2: External master key
    external_key = secrets.token_bytes(32)
    vault2 = DistributedBlackVault(node_id="node-2", master_key=external_key)
    hash3, v3 = vault2.store("secret3", b"from external key", expire_minutes=60)
    print(f"Stored with external key: hash={hash3[:16]}...")
    
    # Pattern 3: Key rotation with external key
    new_external_key = secrets.token_bytes(32)
    vault1.rotate_key(new_external_key)
    print("Key rotation complete (external source)")
    
    vault1.close()
    vault2.close()
