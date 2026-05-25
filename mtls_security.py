"""
PKI-based Mutual TLS (mTLS) for Node Communication
- Certificate Authority for distributed trust
- Node certificate provisioning and rotation
- TLS connection authentication
- Certificate revocation support
"""

import ssl
import logging
import threading
import json
import time
import os
from typing import Dict, Optional, Tuple, Any, cast
from dataclasses import dataclass, field
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timedelta

log = logging.getLogger("MTLS-SECURITY")


@dataclass
class CertificateMetadata:
    """Metadata for issued certificate."""
    cert_id: str
    node_id: str
    issued_at: float
    expires_at: float
    issuer_cn: str
    subject_cn: str
    fingerprint: str
    is_revoked: bool = False
    revoked_at: Optional[float] = None


@dataclass
class Certificate:
    """Container for certificate and private key."""
    cert_pem: str
    key_pem: str
    metadata: CertificateMetadata


class CertificateAuthority:
    """
    Root CA for distributed PKI.
    
    Responsibilities:
    - Issue node certificates
    - Manage certificate lifecycle
    - Handle revocation
    """

    def __init__(self, ca_dir: str = "./pki", organization: str = "BlackRoot"):
        """
        Initialize Certificate Authority.
        
        Args:
            ca_dir: Directory for CA cert/key storage
            organization: Organization name
        """
        self.ca_dir = ca_dir
        self.organization = organization
        self.ca_cert_path = os.path.join(ca_dir, "ca.crt")
        self.ca_key_path = os.path.join(ca_dir, "ca.key")

        os.makedirs(ca_dir, exist_ok=True)

        self.ca_cert = None
        self.ca_key = None
        self.issued_certs: Dict[str, CertificateMetadata] = {}
        self.revoked_certs: Dict[str, float] = {}  # cert_id -> revoked_at
        self.lock = threading.RLock()

        self._load_or_create_ca()
        log.info(f"Initialized PKI with CA: {organization}")

    def _load_or_create_ca(self):
        """Load or create root CA."""
        with self.lock:
            if os.path.exists(self.ca_cert_path) and os.path.exists(self.ca_key_path):
                self._load_ca_from_disk()
            else:
                self._create_root_ca()

    def _create_root_ca(self):
        """Create root CA certificate and key."""
        try:
            # Generate CA private key
            ca_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )

            # Create CA certificate
            ca_cert = x509.CertificateBuilder().add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True,
            ).subject_name(
                x509.Name([
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, self.organization),
                    x509.NameAttribute(NameOID.COMMON_NAME, f"{self.organization} Root CA"),
                ])
            ).issuer_name(
                x509.Name([
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, self.organization),
                    x509.NameAttribute(NameOID.COMMON_NAME, f"{self.organization} Root CA"),
                ])
            ).public_key(
                ca_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=3650)  # 10 years
            ).sign(ca_key, hashes.SHA256(), default_backend())

            # Save CA cert
            with open(self.ca_cert_path, 'wb') as f:
                f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

            # Save CA key
            with open(self.ca_key_path, 'wb') as f:
                f.write(ca_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))

            self.ca_cert = ca_cert
            self.ca_key = ca_key
            log.info(f"Created new Root CA in {self.ca_dir}")
        except Exception as e:
            log.error(f"Failed to create root CA: {e}")
            raise

    def _load_ca_from_disk(self):
        """Load existing CA from disk."""
        try:
            with open(self.ca_cert_path, 'rb') as f:
                self.ca_cert = x509.load_pem_x509_certificate(
                    f.read(), default_backend()
                )

            with open(self.ca_key_path, 'rb') as f:
                self.ca_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )

            log.info(f"Loaded Root CA from {self.ca_dir}")
        except Exception as e:
            log.error(f"Failed to load root CA: {e}")
            raise

    def issue_node_certificate(self, node_id: str, 
                              san_hosts: Optional[list] = None,
                              validity_days: int = 365) -> Certificate:
        """
        Issue certificate for node.
        
        Args:
            node_id: Unique node identifier
            san_hosts: Subject Alternative Names (domain names)
            validity_days: Certificate validity period
            
        Returns:
            Certificate object with cert/key
        """
        if not self.ca_cert or not self.ca_key:
            raise RuntimeError("CA not initialized")

        with self.lock:
            try:
                # Generate node private key
                node_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )

                # Build SAN extension
                san_list = [x509.DNSName(node_id)]
                if san_hosts:
                    san_list.extend([x509.DNSName(h) for h in san_hosts])

                # Create node certificate
                node_cert = x509.CertificateBuilder() \
                    .add_extension(
                        x509.SubjectAlternativeName(san_list),
                        critical=False,
                    ) \
                    .add_extension(
                        x509.KeyUsage(
                            digital_signature=True,
                            key_encipherment=True,
                            key_cert_sign=False,
                            crl_sign=False,
                            content_commitment=False,
                            data_encipherment=False,
                            key_agreement=False,
                            encipher_only=False,
                            decipher_only=False
                        ),
                        critical=True,
                    ) \
                    .add_extension(
                        x509.ExtendedKeyUsage([
                            ExtendedKeyUsageOID.SERVER_AUTH,
                            ExtendedKeyUsageOID.CLIENT_AUTH,
                        ]),
                        critical=True,
                    ) \
                    .subject_name(
                        x509.Name([
                            x509.NameAttribute(NameOID.ORGANIZATION_NAME, self.organization),
                            x509.NameAttribute(NameOID.COMMON_NAME, node_id),
                        ])
                    ) \
                    .issuer_name(self.ca_cert.subject) \
                    .public_key(node_key.public_key()) \
                    .serial_number(x509.random_serial_number()) \
                    .not_valid_before(datetime.utcnow()) \
                    .not_valid_after(
                        datetime.utcnow() + timedelta(days=validity_days)
                    ) \
                    .sign(self.ca_key, hashes.SHA256())  # type: ignore

                # Serialize to PEM
                cert_pem = node_cert.public_bytes(
                    serialization.Encoding.PEM
                ).decode('utf-8')

                key_pem = node_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode('utf-8')

                # Compute fingerprint
                fingerprint = node_cert.fingerprint(hashes.SHA256()).hex()

                # Create metadata
                cert_id = f"node_{node_id}_{int(time.time())}"
                metadata = CertificateMetadata(
                    cert_id=cert_id,
                    node_id=node_id,
                    issued_at=time.time(),
                    expires_at=(datetime.utcnow() + timedelta(days=validity_days)).timestamp(),
                    issuer_cn=self.organization,
                    subject_cn=node_id,
                    fingerprint=fingerprint
                )

                # Store metadata
                self.issued_certs[cert_id] = metadata

                log.info(f"Issued certificate for node {node_id}: {cert_id}")

                return Certificate(cert_pem=cert_pem, key_pem=key_pem, metadata=metadata)

            except Exception as e:
                log.error(f"Failed to issue node certificate: {e}")
                raise

    def revoke_certificate(self, cert_id: str) -> bool:
        """Revoke certificate."""
        with self.lock:
            if cert_id not in self.issued_certs:
                return False

            metadata = self.issued_certs[cert_id]
            metadata.is_revoked = True
            metadata.revoked_at = time.time()
            self.revoked_certs[cert_id] = metadata.revoked_at

            log.info(f"Revoked certificate: {cert_id}")
            return True

    def is_certificate_valid(self, cert_id: str) -> bool:
        """Check certificate validity."""
        with self.lock:
            if cert_id not in self.issued_certs:
                return False

            metadata = self.issued_certs[cert_id]

            # Check revocation
            if metadata.is_revoked:
                return False

            # Check expiration
            if time.time() > metadata.expires_at:
                return False

            return True

    def get_ca_cert_pem(self) -> str:
        """Get CA certificate in PEM format."""
        if not self.ca_cert:
            return ""

        return self.ca_cert.public_bytes(
            serialization.Encoding.PEM
        ).decode('utf-8')

    def get_status(self) -> Dict[str, Any]:
        """Get PKI status."""
        with self.lock:
            valid_certs = sum(
                1 for cert in self.issued_certs.values()
                if not cert.is_revoked and time.time() < cert.expires_at
            )

            return {
                'organization': self.organization,
                'total_issued': len(self.issued_certs),
                'valid_certs': valid_certs,
                'revoked_certs': len(self.revoked_certs),
                'ca_path': self.ca_cert_path
            }


class MTLSConnectionManager:
    """
    Manages TLS connections between nodes.
    
    Features:
    - Authenticate both client and server
    - Certificate pinning
    - Session management
    """

    def __init__(self, ca_pem: str, node_cert_pem: str, node_key_pem: str):
        """
        Initialize mTLS connection manager.
        
        Args:
            ca_pem: Root CA certificate
            node_cert_pem: Node certificate
            node_key_pem: Node private key
        """
        self.ca_pem = ca_pem
        self.node_cert_pem = node_cert_pem
        self.node_key_pem = node_key_pem
        self.pinned_certs: Dict[str, str] = {}  # node_id -> cert_fingerprint
        self.lock = threading.RLock()

    def create_ssl_context(self, server_mode: bool = False) -> ssl.SSLContext:
        """
        Create SSL context for TLS connections.
        
        Args:
            server_mode: True for server, False for client
            
        Returns:
            Configured SSLContext
        """
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server_mode else ssl.PROTOCOL_TLS_CLIENT)

        # Load node certificate and key
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(self.node_cert_pem)
            cert_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(self.node_key_pem)
            key_path = f.name

        try:
            context.load_cert_chain(cert_path, key_path)

            # Load CA certificate
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
                f.write(self.ca_pem)
                ca_path = f.name

            try:
                context.load_verify_locations(ca_path)
            finally:
                os.unlink(ca_path)

            # Require mutual TLS
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True

            # Strong ciphers only
            context.set_ciphers('HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA')

            # Disable insecure protocols
            context.minimum_version = ssl.TLSVersion.TLSv1_2

            return context
        finally:
            os.unlink(cert_path)
            os.unlink(key_path)

    def pin_certificate(self, node_id: str, cert_fingerprint: str):
        """Pin certificate for node (for additional security)."""
        with self.lock:
            self.pinned_certs[node_id] = cert_fingerprint
            log.info(f"Pinned certificate for {node_id}")

    def verify_pinned_certificate(self, node_id: str, cert_fingerprint: str) -> bool:
        """Verify certificate pin."""
        with self.lock:
            if node_id not in self.pinned_certs:
                return True  # Not pinned, accept

            return self.pinned_certs[node_id] == cert_fingerprint

    def get_status(self) -> Dict[str, Any]:
        """Get connection manager status."""
        with self.lock:
            return {
                'pinned_certs': len(self.pinned_certs),
                'tls_version': 'TLSv1.2+'
            }
