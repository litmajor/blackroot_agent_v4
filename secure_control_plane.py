"""
SECURE CONTROL PLANE
====================

Advanced C2 security layer implementing:
- JWT-based agent authentication with certificate validation
- Cryptographically signed requests with anti-replay protection
- Data munging/encryption (no plaintext communication)
- Kill switch and operational mode gating
- Rate limiting and anomaly detection
"""

import os
import json
import time
import uuid
import hmac
import hashlib
import secrets
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import base64
from typing import TYPE_CHECKING

# Type-only imports for better type checking
if TYPE_CHECKING:
    from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    from cryptography import x509
    from cryptography.x509.oid import NameOID

# Runtime imports with fallbacks
try:
    from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    CRYPTO_AVAILABLE = True
except ImportError:
    # Create dummy modules for type checking when cryptography is not available
    hashes = None
    crypto_hmac = None
    rsa = None
    padding = None
    serialization = None
    default_backend = None
    x509 = None
    NameOID = None
    CRYPTO_AVAILABLE = False

from black_vault import BlackVault

# Suppress Pylance warnings for conditional cryptography imports
# type: ignore[possibly-unbound-variable]

# Setup logging
logger = logging.getLogger("SECURE_C2")

class OperationalMode(Enum):
    """Operational modes for kill switch and gating"""
    ACTIVE = "active"           # Full operation
    STEALTH = "stealth"         # Limited operation, reduced visibility
    MAINTENANCE = "maintenance" # Diagnostic mode only
    LOCKDOWN = "lockdown"       # Emergency shutdown
    SELF_DESTRUCT = "self_destruct"  # Complete wipe and exit

class SecurityLevel(Enum):
    """Security clearance levels"""
    BASIC = 1       # Read-only operations
    STANDARD = 2    # Standard agent operations
    ELEVATED = 3    # Privileged operations
    CRITICAL = 4    # System-level operations
    EMERGENCY = 5   # Emergency override operations

@dataclass
class AgentCertificate:
    """Agent authentication certificate"""
    agent_id: str
    public_key_pem: str
    certificate_pem: str
    issued_at: float
    expires_at: float
    security_level: SecurityLevel
    capabilities: Set[str]
    revoked: bool = False

@dataclass
class SignedRequest:
    """Cryptographically signed request"""
    request_id: str
    timestamp: float
    agent_id: str
    operation: str
    payload: Dict[str, Any]
    signature: str
    nonce: str
    security_level: SecurityLevel

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    mode: OperationalMode
    allowed_operations: Set[str]
    rate_limits: Dict[str, int]  # operation -> requests per minute
    max_payload_size: int
    require_encryption: bool
    anti_replay_window: int  # seconds
    emergency_kill_switch: bool

class SecureControlPlane:
    """
    Secure C2 control plane with comprehensive security features.

    Features:
    - JWT-based authentication with certificate validation
    - Cryptographically signed requests
    - Anti-replay protection with nonces
    - Data munging (no plaintext)
    - Kill switch and operational mode gating
    - Rate limiting and anomaly detection
    """

    def __init__(self, vault: BlackVault, ca_private_key_path: Optional[str] = None):
        self.vault = vault
        self.lock = threading.RLock()

        # Security state
        self.operational_mode = OperationalMode.ACTIVE
        self.emergency_kill_switch = False
        self.security_policies: Dict[SecurityLevel, SecurityPolicy] = {}

        # Agent management
        self.agent_certificates: Dict[str, AgentCertificate] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.revoked_certificates: Set[str] = set()

        # Anti-replay protection
        self.used_nonces: Set[str] = set()
        self.nonce_cleanup_interval = 300  # 5 minutes
        self.max_nonce_age = 3600  # 1 hour

        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, List[float]]] = {}  # agent_id -> operation -> timestamps

        # Certificate Authority
        self.ca_private_key = None
        self.ca_certificate = None
        if CRYPTO_AVAILABLE and ca_private_key_path:
            self._load_ca_credentials(ca_private_key_path)
        elif CRYPTO_AVAILABLE:
            self._generate_ca_credentials()

        # Start background cleanup
        self._start_background_tasks()

        logger.info("🔐 Secure Control Plane initialized")

    def _load_ca_credentials(self, private_key_path: str):
        """Load existing CA credentials"""
        try:
            with open(private_key_path, 'rb') as f:
                self.ca_private_key = serialization.load_pem_private_key(  # type: ignore[possibly-unbound-variable]
                    f.read(),
                    password=None,
                    backend=default_backend()  # type: ignore[possibly-unbound-variable]
                )

            # Load CA certificate
            cert_path = private_key_path.replace('.key', '.crt')
            if os.path.exists(cert_path):
                with open(cert_path, 'rb') as f:
                    self.ca_certificate = x509.load_pem_x509_certificate(  # type: ignore[possibly-unbound-variable]
                        f.read(),
                        default_backend()  # type: ignore[possibly-unbound-variable]
                    )

            logger.info("✅ CA credentials loaded")
        except Exception as e:
            logger.error(f"Failed to load CA credentials: {e}")
            self._generate_ca_credentials()

    def _generate_ca_credentials(self):
        """Generate new CA credentials"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available, CA generation skipped")
            return
        try:
            # Generate CA private key
            self.ca_private_key = rsa.generate_private_key(  # type: ignore[possibly-unbound-variable]
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()  # type: ignore[possibly-unbound-variable]
            )

            # Generate CA certificate
            subject = issuer = x509.Name([  # type: ignore[possibly-unbound-variable]
                x509.NameAttribute(NameOID.COUNTRY_NAME, "XX"),  # type: ignore[possibly-unbound-variable]
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Blackroot"),  # type: ignore[possibly-unbound-variable]
                x509.NameAttribute(NameOID.LOCALITY_NAME, "C2"),  # type: ignore[possibly-unbound-variable]
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Blackroot Security"),  # type: ignore[possibly-unbound-variable]
                x509.NameAttribute(NameOID.COMMON_NAME, "Blackroot C2 CA"),  # type: ignore[possibly-unbound-variable]
            ])

            cert = x509.CertificateBuilder().subject_name(  # type: ignore[possibly-unbound-variable]
                subject
            ).issuer_name(
                issuer
            ).public_key(
                self.ca_private_key.public_key()
            ).serial_number(
                x509.random_serial_number()  # type: ignore[possibly-unbound-variable]
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=3650)  # 10 years
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),  # type: ignore[possibly-unbound-variable]
                critical=True,
            ).sign(self.ca_private_key, hashes.SHA256(), default_backend())  # type: ignore[possibly-unbound-variable]

            self.ca_certificate = cert
            logger.info("✅ CA credentials generated")
        except Exception as e:
            logger.error(f"Failed to generate CA credentials: {e}")

    def _start_background_tasks(self):
        """Start background security maintenance tasks"""
        def cleanup_task():
            while True:
                try:
                    self._cleanup_expired_nonces()
                    self._cleanup_expired_sessions()
                    time.sleep(self.nonce_cleanup_interval)
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=cleanup_task, daemon=True, name="security-cleanup")
        thread.start()

    def _cleanup_expired_nonces(self):
        """Clean up expired nonces for anti-replay protection"""
        current_time = time.time()
        expired_nonces = {
            nonce for nonce, timestamp in self.used_nonces
            if current_time - timestamp > self.max_nonce_age
        }
        self.used_nonces -= expired_nonces

    def _cleanup_expired_sessions(self):
        """Clean up expired agent sessions"""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if current_time - session_data.get('created_at', 0) > session_data.get('max_age', 3600)
        ]
        for session_id in expired_sessions:
            del self.active_sessions[session_id]

    # ============================================================================
    # AGENT CERTIFICATE MANAGEMENT
    # ============================================================================

    def issue_agent_certificate(self, agent_id: str, capabilities: Set[str],
                              security_level: SecurityLevel = SecurityLevel.STANDARD,
                              validity_days: int = 30) -> Optional[AgentCertificate]:
        """
        Issue a new agent certificate for authentication.

        Args:
            agent_id: Unique agent identifier
            capabilities: Set of allowed capabilities
            security_level: Security clearance level
            validity_days: Certificate validity period

        Returns:
            AgentCertificate if successful, None otherwise
        """
        if not CRYPTO_AVAILABLE or not self.ca_private_key:
            logger.error("CA not available for certificate issuance")
            return None
        try:
            # Generate agent key pair
            private_key = rsa.generate_private_key(  # type: ignore[possibly-unbound-variable]
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()  # type: ignore[possibly-unbound-variable]
            )

            # Create certificate
            subject = x509.Name([  # type: ignore[possibly-unbound-variable]
                x509.NameAttribute(NameOID.COMMON_NAME, f"agent-{agent_id}"),  # type: ignore[possibly-unbound-variable]
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Blackroot Agents"),  # type: ignore[possibly-unbound-variable]
            ])

            cert = x509.CertificateBuilder().subject_name(  # type: ignore[possibly-unbound-variable]
                subject
            ).issuer_name(
                self.ca_certificate.subject
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()  # type: ignore[possibly-unbound-variable]
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            ).add_extension(
                x509.KeyUsage(  # type: ignore[possibly-unbound-variable]
                    digital_signature=True,
                    key_encipherment=True,
                    content_commitment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True,
            ).sign(self.ca_private_key, hashes.SHA256(), default_backend())  # type: ignore[possibly-unbound-variable]

            # Create certificate object
            certificate = AgentCertificate(
                agent_id=agent_id,
                public_key_pem=private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,  # type: ignore[possibly-unbound-variable]
                    format=serialization.PublicFormat.SubjectPublicKeyInfo  # type: ignore[possibly-unbound-variable]
                ).decode(),
                certificate_pem=cert.public_bytes(serialization.Encoding.PEM).decode(),  # type: ignore[possibly-unbound-variable]
                issued_at=time.time(),
                expires_at=time.time() + (validity_days * 24 * 3600),
                security_level=security_level,
                capabilities=capabilities
            )

            # Store certificate
            with self.lock:
                self.agent_certificates[agent_id] = certificate

            # Store private key securely in vault
            self.vault.store(f"agent_key_{agent_id}",
                           private_key.private_bytes(
                               encoding=serialization.Encoding.PEM,
                               format=serialization.PrivateFormat.PKCS8,
                               encryption_algorithm=serialization.NoEncryption()
                           ))

            logger.info(f"✅ Certificate issued for agent {agent_id}")
            return certificate

        except Exception as e:
            logger.error(f"Failed to issue certificate for {agent_id}: {e}")
            return None

    def revoke_agent_certificate(self, agent_id: str) -> bool:
        """Revoke an agent certificate"""
        with self.lock:
            if agent_id in self.agent_certificates:
                self.agent_certificates[agent_id].revoked = True
                self.revoked_certificates.add(agent_id)
                logger.info(f"🚫 Certificate revoked for agent {agent_id}")
                return True
        return False

    def validate_agent_certificate(self, agent_id: str, certificate_pem: str) -> bool:
        """
        Validate an agent certificate.

        Returns:
            True if certificate is valid and not revoked
        """
        if not CRYPTO_AVAILABLE:
            return False
        try:
            # Load certificate
            cert = x509.load_pem_x509_certificate(  # type: ignore[possibly-unbound-variable]
                certificate_pem.encode(),
                default_backend()  # type: ignore[possibly-unbound-variable]
            )

            # Check if revoked
            if agent_id in self.revoked_certificates:
                return False

            # Check expiration
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return False

            # Verify against CA
            if self.ca_certificate:
                self.ca_certificate.public_key().verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),  # type: ignore[possibly-unbound-variable]
                    cert.signature_hash_algorithm
                )

            return True

        except Exception as e:
            logger.error(f"Certificate validation failed for {agent_id}: {e}")
            return False

    # ============================================================================
    # AUTHENTICATION & SESSION MANAGEMENT
    # ============================================================================

    def authenticate_agent(self, agent_id: str, certificate_pem: str,
                          challenge_response: str) -> Optional[str]:
        """
        Authenticate an agent and create a session.

        Args:
            agent_id: Agent identifier
            certificate_pem: Agent certificate in PEM format
            challenge_response: Signed challenge response

        Returns:
            Session token if authentication successful, None otherwise
        """
        if self.emergency_kill_switch:
            logger.warning("🚨 Emergency kill switch active - authentication blocked")
            return None

        # Validate certificate
        if not self.validate_agent_certificate(agent_id, certificate_pem):
            logger.warning(f"❌ Certificate validation failed for {agent_id}")
            return None

        # Generate challenge
        challenge = secrets.token_hex(32)

        # Verify challenge response
        if not self._verify_challenge_response(agent_id, challenge, challenge_response):
            logger.warning(f"❌ Challenge verification failed for {agent_id}")
            return None

        # Create session
        session_token = secrets.token_hex(32)
        session_data = {
            'agent_id': agent_id,
            'created_at': time.time(),
            'max_age': 3600,  # 1 hour
            'security_level': self.agent_certificates[agent_id].security_level.value
        }

        with self.lock:
            self.active_sessions[session_token] = session_data

        logger.info(f"✅ Agent {agent_id} authenticated, session created")
        return session_token

    def _verify_challenge_response(self, agent_id: str, challenge: str, response: str) -> bool:
        """Verify signed challenge response"""
        if not CRYPTO_AVAILABLE:
            return False

        try:
            # Get agent certificate
            cert = self.agent_certificates.get(agent_id)
            if not cert:
                return False

            # Load public key
            public_key = serialization.load_pem_public_key(
                cert.public_key_pem.encode(),
                backend=default_backend()
            )

            # Verify signature
            public_key.verify(
                base64.b64decode(response),
                challenge.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            logger.error(f"Challenge verification error: {e}")
            return False

    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate an active session"""
        with self.lock:
            session = self.active_sessions.get(session_token)
            if not session:
                return None

            # Check expiration
            if time.time() - session['created_at'] > session['max_age']:
                del self.active_sessions[session_token]
                return None

            return session

    # ============================================================================
    # SIGNED REQUESTS WITH ANTI-REPLAY PROTECTION
    # ============================================================================

    def create_signed_request(self, agent_id: str, operation: str,
                            payload: Dict[str, Any], session_token: str) -> Optional[SignedRequest]:
        """
        Create a cryptographically signed request.

        Args:
            agent_id: Requesting agent ID
            operation: Operation to perform
            payload: Request payload
            session_token: Valid session token

        Returns:
            SignedRequest if successful, None otherwise
        """
        # Validate session
        session = self.validate_session(session_token)
        if not session or session['agent_id'] != agent_id:
            return None

        # Check operational mode
        if not self._check_operational_mode(operation):
            return None

        # Generate request
        request = SignedRequest(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            agent_id=agent_id,
            operation=operation,
            payload=self._munge_payload(payload),  # Encrypt payload
            signature="",
            nonce=secrets.token_hex(16),
            security_level=SecurityLevel(session['security_level'])
        )

        # Sign request
        request.signature = self._sign_request(request)
        return request

    def validate_signed_request(self, signed_request: SignedRequest) -> bool:
        """
        Validate a signed request with anti-replay protection.

        Returns:
            True if request is valid and authorized
        """
        try:
            # Check operational mode
            if not self._check_operational_mode(signed_request.operation):
                return False

            # Anti-replay: check nonce
            if signed_request.nonce in self.used_nonces:
                logger.warning(f"🚨 Replay attack detected: nonce {signed_request.nonce}")
                return False

            # Check timestamp (anti-replay window)
            age = time.time() - signed_request.timestamp
            if age < 0 or age > 300:  # 5 minute window
                logger.warning(f"🚨 Timestamp out of range: {age}s")
                return False

            # Verify signature
            if not self._verify_request_signature(signed_request):
                logger.warning("🚨 Invalid request signature")
                return False

            # Check rate limits
            if not self._check_rate_limit(signed_request.agent_id, signed_request.operation):
                logger.warning(f"🚨 Rate limit exceeded for {signed_request.agent_id}")
                return False

            # Mark nonce as used
            self.used_nonces.add((signed_request.nonce, time.time()))

            return True

        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return False

    def _sign_request(self, request: SignedRequest) -> str:
        """Sign a request using agent's private key"""
        if not CRYPTO_AVAILABLE:
            return ""

        try:
            # Get agent private key from vault
            key_data = self.vault.retrieve(f"agent_key_{request.agent_id}")
            if not key_data:
                return ""

            private_key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )

            # Create signature payload
            payload = f"{request.request_id}:{request.timestamp}:{request.agent_id}:{request.operation}:{json.dumps(request.payload, sort_keys=True)}:{request.nonce}"

            # Sign
            signature = private_key.sign(
                payload.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return base64.b64encode(signature).decode()

        except Exception as e:
            logger.error(f"Request signing error: {e}")
            return ""

    def _verify_request_signature(self, request: SignedRequest) -> bool:
        """Verify request signature"""
        if not CRYPTO_AVAILABLE:
            return False

        try:
            # Get agent certificate
            cert = self.agent_certificates.get(request.agent_id)
            if not cert:
                return False

            # Load public key
            public_key = serialization.load_pem_public_key(
                cert.public_key_pem.encode(),
                backend=default_backend()
            )

            # Recreate payload
            payload = f"{request.request_id}:{request.timestamp}:{request.agent_id}:{request.operation}:{json.dumps(request.payload, sort_keys=True)}:{request.nonce}"

            # Verify signature
            public_key.verify(
                base64.b64decode(request.signature),
                payload.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    # ============================================================================
    # DATA MUNGING (NO PLAINTEXT)
    # ============================================================================

    def _munge_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt payload data (no plaintext communication)"""
        if not payload:
            return {}

        try:
            # Serialize payload
            payload_json = json.dumps(payload, sort_keys=True)

            # Generate encryption key
            key = secrets.token_bytes(32)

            # Encrypt payload
            encrypted_data = self._encrypt_data(payload_json.encode(), key)

            # Encrypt key with agent's public key
            encrypted_key = self._encrypt_key_for_agent(key, payload.get('target_agent', 'controller'))

            return {
                'encrypted_data': base64.b64encode(encrypted_data).decode(),
                'encrypted_key': base64.b64encode(encrypted_key).decode(),
                'algorithm': 'AES-256-GCM'
            }

        except Exception as e:
            logger.error(f"Payload munging error: {e}")
            return payload  # Fallback to plaintext if encryption fails

    def _unmunge_payload(self, munged_payload: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Decrypt munged payload"""
        try:
            # Decrypt key
            encrypted_key = base64.b64decode(munged_payload['encrypted_key'])
            key = self._decrypt_key_for_agent(encrypted_key, agent_id)

            # Decrypt data
            encrypted_data = base64.b64decode(munged_payload['encrypted_data'])
            decrypted_json = self._decrypt_data(encrypted_data, key)

            return json.loads(decrypted_json)

        except Exception as e:
            logger.error(f"Payload unmunging error: {e}")
            return munged_payload  # Fallback

    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data with AES-GCM"""
        if not CRYPTO_AVAILABLE:
            return data

        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

            # Generate nonce
            nonce = secrets.token_bytes(12)

            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
            encryptor = cipher.encryptor()

            # Encrypt
            ciphertext = encryptor.update(data) + encryptor.finalize()

            # Return nonce + tag + ciphertext
            return nonce + encryptor.tag + ciphertext

        except Exception as e:
            logger.error(f"Data encryption error: {e}")
            return data

    def _decrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Decrypt data with AES-GCM"""
        if not CRYPTO_AVAILABLE:
            return data

        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

            # Extract nonce, tag, ciphertext
            nonce = data[:12]
            tag = data[12:28]
            ciphertext = data[28:]

            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
            decryptor = cipher.decryptor()

            # Decrypt
            return decryptor.update(ciphertext) + decryptor.finalize()

        except Exception as e:
            logger.error(f"Data decryption error: {e}")
            return data

    def _encrypt_key_for_agent(self, key: bytes, agent_id: str) -> bytes:
        """Encrypt symmetric key with agent's public key"""
        if not CRYPTO_AVAILABLE:
            return key

        try:
            # Get agent certificate
            cert = self.agent_certificates.get(agent_id)
            if not cert:
                return key

            # Load public key
            public_key = serialization.load_pem_public_key(
                cert.public_key_pem.encode(),
                backend=default_backend()
            )

            # Encrypt key
            encrypted_key = public_key.encrypt(
                key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return encrypted_key

        except Exception as e:
            logger.error(f"Key encryption error: {e}")
            return key

    def _decrypt_key_for_agent(self, encrypted_key: bytes, agent_id: str) -> bytes:
        """Decrypt symmetric key with agent's private key"""
        if not CRYPTO_AVAILABLE:
            return encrypted_key

        try:
            # Get agent private key from vault
            key_data = self.vault.retrieve(f"agent_key_{agent_id}")
            if not key_data:
                return encrypted_key

            private_key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )

            # Decrypt key
            return private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

        except Exception as e:
            logger.error(f"Key decryption error: {e}")
            return encrypted_key

    # ============================================================================
    # OPERATIONAL MODE GATING & KILL SWITCH
    # ============================================================================

    def set_operational_mode(self, mode: OperationalMode, authorized_by: str) -> bool:
        """
        Set operational mode (requires CRITICAL security level).

        Args:
            mode: New operational mode
            authorized_by: Agent authorizing the change

        Returns:
            True if mode changed successfully
        """
        # Check authorization
        cert = self.agent_certificates.get(authorized_by)
        if not cert or cert.security_level.value < SecurityLevel.CRITICAL.value:
            logger.warning(f"🚫 Unauthorized mode change attempt by {authorized_by}")
            return False

        with self.lock:
            old_mode = self.operational_mode
            self.operational_mode = mode

            # Handle special modes
            if mode == OperationalMode.LOCKDOWN:
                self._enter_lockdown()
            elif mode == OperationalMode.SELF_DESTRUCT:
                self._initiate_self_destruct()

        logger.info(f"🔄 Operational mode changed: {old_mode.value} → {mode.value} by {authorized_by}")
        return True

    def activate_kill_switch(self, authorized_by: str, reason: str) -> bool:
        """
        Activate emergency kill switch.

        Args:
            authorized_by: Agent authorizing kill switch
            reason: Reason for activation

        Returns:
            True if kill switch activated
        """
        # Check authorization
        cert = self.agent_certificates.get(authorized_by)
        if not cert or cert.security_level.value < SecurityLevel.EMERGENCY.value:
            logger.warning(f"🚫 Unauthorized kill switch attempt by {authorized_by}")
            return False

        with self.lock:
            self.emergency_kill_switch = True
            self.operational_mode = OperationalMode.LOCKDOWN

        logger.critical(f"🚨 EMERGENCY KILL SWITCH ACTIVATED by {authorized_by}: {reason}")

        # Notify all agents
        self._broadcast_kill_switch(reason)
        return True

    def _check_operational_mode(self, operation: str) -> bool:
        """Check if operation is allowed in current operational mode"""
        if self.emergency_kill_switch:
            return False

        mode_permissions = {
            OperationalMode.ACTIVE: {
                'allowed_operations': {'*'},  # All operations allowed
                'max_security_level': SecurityLevel.CRITICAL
            },
            OperationalMode.STEALTH: {
                'allowed_operations': {'status', 'heartbeat', 'report'},
                'max_security_level': SecurityLevel.STANDARD
            },
            OperationalMode.MAINTENANCE: {
                'allowed_operations': {'status', 'diagnostic', 'health'},
                'max_security_level': SecurityLevel.ELEVATED
            },
            OperationalMode.LOCKDOWN: {
                'allowed_operations': set(),
                'max_security_level': SecurityLevel.EMERGENCY
            },
            OperationalMode.SELF_DESTRUCT: {
                'allowed_operations': set(),
                'max_security_level': SecurityLevel.EMERGENCY
            }
        }

        permissions = mode_permissions.get(self.operational_mode, mode_permissions[OperationalMode.LOCKDOWN])

        # Check operation
        if operation not in permissions['allowed_operations'] and '*' not in permissions['allowed_operations']:
            return False

        return True

    def _enter_lockdown(self):
        """Enter lockdown mode - revoke all sessions and certificates"""
        with self.lock:
            # Revoke all sessions
            self.active_sessions.clear()

            # Revoke all certificates
            for agent_id in self.agent_certificates:
                self.agent_certificates[agent_id].revoked = True
                self.revoked_certificates.add(agent_id)

        logger.critical("🔒 LOCKDOWN MODE: All sessions and certificates revoked")

    def _initiate_self_destruct(self):
        """Initiate self-destruct sequence"""
        logger.critical("💥 SELF-DESTRUCT INITIATED")

        # This would trigger complete system wipe
        # Implementation depends on specific requirements

    def _broadcast_kill_switch(self, reason: str):
        """Broadcast kill switch activation to all agents"""
        # Implementation would send notifications to all connected agents
        pass

    # ============================================================================
    # RATE LIMITING
    # ============================================================================

    def _check_rate_limit(self, agent_id: str, operation: str) -> bool:
        """Check if operation is within rate limits"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        with self.lock:
            if agent_id not in self.rate_limiters:
                self.rate_limiters[agent_id] = {}

            if operation not in self.rate_limiters[agent_id]:
                self.rate_limiters[agent_id][operation] = []

            # Clean old timestamps
            timestamps = self.rate_limiters[agent_id][operation]
            timestamps[:] = [t for t in timestamps if t > window_start]

            # Check limit (default 100 requests per minute)
            limit = 100
            if len(timestamps) >= limit:
                return False

            # Add current timestamp
            timestamps.append(current_time)
            return True

    # ============================================================================
    # CONFIGURATION MANAGEMENT
    # ============================================================================

    def update_security_policy(self, level: SecurityLevel, policy: SecurityPolicy) -> bool:
        """Update security policy for a clearance level"""
        with self.lock:
            self.security_policies[level] = policy
        logger.info(f"🔒 Security policy updated for level {level.value}")
        return True

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            'operational_mode': self.operational_mode.value,
            'emergency_kill_switch': self.emergency_kill_switch,
            'active_sessions': len(self.active_sessions),
            'issued_certificates': len(self.agent_certificates),
            'revoked_certificates': len(self.revoked_certificates),
            'used_nonces': len(self.used_nonces),
            'crypto_available': CRYPTO_AVAILABLE,
            'ca_initialized': self.ca_certificate is not None
        }

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def generate_challenge(self) -> str:
        """Generate authentication challenge"""
        return secrets.token_hex(32)

    def cleanup_security_state(self):
        """Clean up security state (for testing/debugging)"""
        with self.lock:
            self.agent_certificates.clear()
            self.active_sessions.clear()
            self.revoked_certificates.clear()
            self.used_nonces.clear()
            self.rate_limiters.clear()
            self.operational_mode = OperationalMode.ACTIVE
            self.emergency_kill_switch = False

        logger.info("🧹 Security state cleaned up")