"""
Unified Redis configuration and client factory for Blackroot architecture.

Provides:
- Centralized Redis configuration (host, port, db, pool settings)
- Factory methods for getting properly configured clients
- Key namespace management (prevent collisions)
- Sync/async client support
- Environment-based configuration override
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RedisDB(Enum):
    """Redis database indices for logical separation."""
    DEFAULT = 0           # General-purpose operations
    MESH_COMM = 1         # SwarmMesh peer communication
    SECURE_CHANNEL = 2    # Secure channel messaging
    IDENTITY = 3          # Identity registry
    MISSION_BUFFER = 4    # Mission queue buffer
    THREAT_INTEL = 5      # Threat intelligence cache
    ECONOMIC = 6          # Economic/wallet data
    KERNEL_STATE = 7      # Kernel event bus & heartbeat
    CACHE = 8             # Temporary results/caching
    TELEMETRY = 9         # Metrics & monitoring data


class RedisKeyPrefix(Enum):
    """Key prefix namespacing to prevent collisions."""
    # Mesh communication
    MESH_PEER = "mesh:peer:"
    MESH_TOPOLOGY = "mesh:topo:"
    MESH_DISCOVERY = "mesh:discovery:"
    
    # Secure channels
    SECURE_MSG = "secure:msg:"
    SECURE_CHANNEL = "secure:chan:"
    
    # Identity & agents
    IDENTITY = "ident:"
    AGENT_STATE = "agent:state:"
    AGENT_WALLET = "agent:wallet:"
    
    # Missions & tasks
    MISSION_QUEUE = "mission:queue:"
    MISSION_RESULT = "mission:result:"
    TASK_LOCK = "task:lock:"
    
    # Reconnaissance
    RECON_RESULT = "recon:result:"
    RECON_CACHE = "recon:cache:"
    THREAT_INTEL = "ti:"
    
    # Economic layer
    LEDGER = "ledger:"
    TRANSACTION = "txn:"
    WALLET = "wallet:"
    
    # Kernel & telemetry
    KERNEL_EVENT = "kernel:event:"
    KERNEL_HEARTBEAT = "kernel:hb:"
    TELEMETRY = "telemetry:"
    STEALTH_METRICS = "stealth:"


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    decode_responses: bool = True
    
    # Pooling
    max_connections: int = 20
    
    # Resilience (ResilientRedisPool)
    max_retries: int = 3
    initial_backoff: float = 0.5
    circuit_breaker_threshold: int = 5
    half_open_timeout: float = 30.0
    
    # PubSub (for SwarmMesh, secure channels)
    pubsub_ignore_subscribe_messages: bool = True
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            password=os.getenv('REDIS_PASSWORD'),
            ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true',
            socket_timeout=float(os.getenv('REDIS_TIMEOUT', '5.0')),
            max_connections=int(os.getenv('REDIS_MAX_CONN', '20')),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for Redis client initialization."""
        return {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'password': self.password,
            'ssl': self.ssl,
            'socket_timeout': self.socket_timeout,
            'socket_connect_timeout': self.socket_connect_timeout,
            'socket_keepalive': self.socket_keepalive,
            'decode_responses': self.decode_responses,
            'max_connections': self.max_connections,
        }
    
    @property
    def redis_url(self) -> str:
        """Generate redis:// URL."""
        protocol = 'rediss' if self.ssl else 'redis'
        auth = f':{self.password}@' if self.password else ''
        return f'{protocol}://{auth}{self.host}:{self.port}/{self.db}'
