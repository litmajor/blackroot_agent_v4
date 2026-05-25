"""
Unified Vault Adapter
=====================

Provides a common interface for both local (file-based) and distributed (Redis-backed) vaults.
Allows seamless switching between modes via configuration.

Usage:
    # Local mode (file-based)
    vault = UnifiedVault(mode='local')
    
    # Distributed mode (Redis-backed)
    vault = UnifiedVault(mode='distributed', redis_host='localhost', password='secret')

Both modes implement:
    - store(name: str, data: bytes) -> artifact_id
    - retrieve(artifact_id: str) -> bytes
    - delete(artifact_id: str) -> bool
    - list_artifacts(node_id=None) -> List[Dict]
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger('UnifiedVault')


class UnifiedVault:
    """
    Adapter providing unified interface for local and distributed vaults.
    
    Modes:
        'local': File-based vault (BlackVault)
        'distributed': Redis-backed vault (DistributedBlackVault)
    """
    
    def __init__(self, mode: str = 'local', redis_host: str = 'localhost', 
                 redis_port: int = 6379, redis_db: int = 1, node_id: Optional[str] = None,
                 password: Optional[str] = None, **kwargs):
        """
        Initialize unified vault.
        
        Args:
            mode: 'local' or 'distributed'
            redis_host: Redis hostname (distributed mode)
            redis_port: Redis port (distributed mode)
            redis_db: Redis database number (distributed mode)
            node_id: Node identifier (distributed mode)
            password: Password for key derivation (distributed mode)
            **kwargs: Additional arguments passed to underlying vault
        """
        self.mode = mode.lower()
        self.logger = logging.getLogger(f'UnifiedVault[{self.mode}]')
        
        if self.mode == 'distributed':
            self._init_distributed(redis_host, redis_port, redis_db, node_id, password, **kwargs)
        elif self.mode == 'local':
            self._init_local(**kwargs)
        else:
            raise ValueError(f"Unknown vault mode: {self.mode}. Use 'local' or 'distributed'")
        
        self.logger.info(f"Vault initialized in {self.mode} mode")
    
    def _init_local(self, **kwargs):
        """Initialize local file-based vault."""
        try:
            from black_vault import BlackVault
            self.vault = BlackVault(**kwargs)
            self.is_distributed = False
        except ImportError:
            raise ImportError("black_vault module not available for local mode")
    
    def _init_distributed(self, redis_host: str, redis_port: int, redis_db: int,
                         node_id: Optional[str], password: Optional[str], **kwargs):
        """Initialize distributed Redis-backed vault."""
        try:
            from black_vault_distributed import DistributedBlackVault
            self.vault = DistributedBlackVault(
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=redis_db,
                node_id=node_id or os.environ.get('BLACKROOT_NODE_ID', 'agent-node'),
                password=password or os.environ.get('VAULT_PASSWORD'),
                **kwargs
            )
            self.is_distributed = True
        except ImportError:
            raise ImportError("black_vault_distributed module not available for distributed mode")
    
    # ============ Unified Interface ============
    
    def store(self, name: str, data: bytes, expire_minutes: Optional[int] = None,
              replicate: bool = True) -> str:
        """
        Store artifact and return identifier.
        
        Args:
            name: Human-readable artifact name
            data: Bytes to encrypt and store
            expire_minutes: TTL in minutes (distributed mode)
            replicate: Broadcast replication event (distributed mode)
        
        Returns:
            Artifact ID (content hash for distributed, path for local)
        """
        try:
            if self.is_distributed:
                # Returns (hash, version) tuple
                artifact_id, version = self.vault.store(name, data, expire_minutes, replicate)
                self.logger.info(f"Stored artifact {name} (ID: {artifact_id[:16]}..., version: {version})")
                return str(artifact_id)
            else:
                # Local vault returns path
                artifact_id = self.vault.store(data)
                self.logger.info(f"Stored artifact {name} (ID: {artifact_id})")
                return str(artifact_id)
        except Exception as e:
            self.logger.error(f"Failed to store {name}: {e}")
            raise
    
    def retrieve(self, artifact_id: str) -> Optional[bytes]:
        """
        Retrieve artifact by ID.
        
        Args:
            artifact_id: Artifact identifier
        
        Returns:
            Decrypted artifact bytes, or None if not found
        """
        try:
            plaintext = self.vault.retrieve(artifact_id)
            if plaintext:
                self.logger.debug(f"Retrieved artifact {artifact_id[:16] if len(artifact_id) > 16 else artifact_id}... ({len(plaintext)} bytes)")
            else:
                self.logger.warning(f"Artifact not found: {artifact_id}")
            return plaintext
        except Exception as e:
            self.logger.error(f"Failed to retrieve {artifact_id}: {e}")
            return None
    
    def delete(self, artifact_id: str) -> bool:
        """
        Delete artifact by ID.
        
        Args:
            artifact_id: Artifact identifier
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            if self.is_distributed:
                # Distributed vault uses cryptographic erasure (master key destruction)
                # Not per-artifact deletion. Log as unsupported but not error.
                self.logger.info(f"Distributed vault: artifact {artifact_id} is immutable. Use destroy_master_key() for cryptographic erasure.")
                return False
            else:
                success = self.vault.delete(artifact_id)
                if success:
                    self.logger.info(f"Deleted artifact {artifact_id}")
                else:
                    self.logger.warning(f"Failed to delete artifact {artifact_id}")
                return success
        except Exception as e:
            self.logger.error(f"Error deleting {artifact_id}: {e}")
            return False
    
    def list_artifacts(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all artifacts (metadata).
        
        Args:
            node_id: Filter by node (distributed mode only)
        
        Returns:
            List of artifact metadata dicts
        """
        try:
            if self.is_distributed:
                artifacts = self.vault.list_artifacts(node_id)
            else:
                # Local vault: emulate list via directory listing
                try:
                    list_method = getattr(self.vault, 'list', None)
                    if callable(list_method):
                        artifacts = list_method()
                    else:
                        artifacts = []
                except Exception:
                    artifacts = []
            
            self.logger.info(f"Listed {len(artifacts)} artifacts")
            return artifacts
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []
    
    def get_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for an artifact.
        
        Args:
            artifact_id: Artifact identifier
        
        Returns:
            Metadata dict if found, None otherwise
        """
        try:
            if self.is_distributed:
                # Get from Redis metadata key
                artifacts = self.vault.list_artifacts()
                for art in artifacts:
                    if art.get('hash') == artifact_id:
                        return art
                return None
            else:
                # Try local vault metadata method
                metadata_method = getattr(self.vault, 'get_metadata', None)
                if callable(metadata_method):
                    return metadata_method(artifact_id)
                return None
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {artifact_id}: {e}")
            return None
    
    def destroy_master_key(self):
        """
        Cryptographic erasure: destroy master key (distributed mode only).
        All artifacts become unrecoverable immediately.
        """
        if self.is_distributed:
            try:
                self.vault.destroy_master_key()
                self.logger.warning("Master key destroyed - all artifacts unrecoverable")
            except Exception as e:
                self.logger.error(f"Failed to destroy master key: {e}")
                raise
        else:
            self.logger.warning("Local vault: destroy_master_key not applicable")
    
    def rotate_key(self, new_master_key: Optional[bytes] = None):
        """
        Rotate encryption key (distributed mode only).
        
        Args:
            new_master_key: New 32-byte master key (from external source)
        """
        if self.is_distributed:
            if new_master_key is None:
                raise ValueError("Distributed vault requires external master key for rotation")
            try:
                self.vault.rotate_key(new_master_key)
                self.logger.info("Master key rotated - all artifacts rewrapped")
            except Exception as e:
                self.logger.error(f"Failed to rotate master key: {e}")
                raise
        else:
            self.logger.warning("Local vault: rotate_key not applicable")
    
    def close(self):
        """Close vault and cleanup resources."""
        try:
            if hasattr(self.vault, 'close'):
                self.vault.close()
            self.logger.info("Vault closed")
        except Exception as e:
            self.logger.error(f"Error closing vault: {e}")
    
    # ============ Mode-Specific Methods ============
    
    def start_replication_listener(self):
        """
        Start background replication listener (distributed mode only).
        Listens for pub/sub events from peer nodes.
        """
        if self.is_distributed:
            try:
                self.vault.start_replication_listener()
                self.logger.info("Replication listener started")
            except Exception as e:
                self.logger.error(f"Failed to start replication listener: {e}")
                raise
        else:
            self.logger.debug("Local vault: no replication listener")
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


def create_vault_from_config(config: Dict[str, Any]) -> UnifiedVault:
    """
    Factory function to create vault from configuration dictionary.
    
    Config format:
        {
            "vault": {
                "mode": "distributed",  # or "local"
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 1
                },
                "node_id": "agent-01",
                "password": "..."  # from environment preferred
            }
        }
    
    Args:
        config: Configuration dictionary
    
    Returns:
        UnifiedVault instance
    """
    vault_config = config.get('vault', {})
    mode = vault_config.get('mode', 'local')
    
    if mode == 'distributed':
        redis_config = vault_config.get('redis', {})
        return UnifiedVault(
            mode='distributed',
            redis_host=redis_config.get('host', 'localhost'),
            redis_port=redis_config.get('port', 6379),
            redis_db=redis_config.get('db', 1),
            node_id=vault_config.get('node_id'),
            password=os.environ.get('VAULT_PASSWORD')
        )
    else:
        return UnifiedVault(mode='local')


if __name__ == "__main__":
    """Demo: both modes side-by-side"""
    print("=" * 70)
    print("UnifiedVault Demo: Local vs Distributed")
    print("=" * 70)
    
    # Demo 1: Local mode
    print("\n1. LOCAL MODE (File-based)")
    print("-" * 70)
    try:
        vault_local = UnifiedVault(mode='local')
        artifact_id = vault_local.store("demo_local", b"Hello from local vault")
        print(f"✓ Stored artifact: {artifact_id}")
        data = vault_local.retrieve(artifact_id)
        print(f"✓ Retrieved: {data}")
        artifacts = vault_local.list_artifacts()
        print(f"✓ Listed {len(artifacts)} artifacts")
        vault_local.close()
    except Exception as e:
        print(f"✗ Local mode error: {e}")
    
    # Demo 2: Distributed mode
    print("\n2. DISTRIBUTED MODE (Redis-backed)")
    print("-" * 70)
    try:
        vault_dist = UnifiedVault(
            mode='distributed',
            node_id='demo-node',
            password='demo-password'
        )
        artifact_id, version = vault_dist.vault.store("demo_dist", b"Hello from distributed vault")
        print(f"✓ Stored artifact: {artifact_id} (version {version})")
        data = vault_dist.retrieve(artifact_id)
        print(f"✓ Retrieved: {data}")
        artifacts = vault_dist.list_artifacts()
        print(f"✓ Listed {len(artifacts)} artifacts")
        print(f"✓ Master key available: {vault_dist.vault.key is not None}")
        vault_dist.close()
    except Exception as e:
        print(f"✗ Distributed mode error: {e} (Redis may not be running)")
    
    print("\n" + "=" * 70)
    print("Demo complete")
    print("=" * 70)
