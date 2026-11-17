"""
Factory and singleton pattern for Redis client management.

Centralized access to Redis with proper pooling, error handling, and namespace support.
"""

import logging
import json
from typing import Optional, Any, Dict, List, Callable, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Wrapper around Redis client with namespace support and safety features.
    
    Usage:
        client = RedisClient(db=RedisDB.MESH_COMM)
        client.set_namespaced(RedisKeyPrefix.MESH_PEER, "node1", data)
        data = client.get_namespaced(RedisKeyPrefix.MESH_PEER, "node1")
    """
    
    def __init__(self, redis_client, config):
        """Initialize wrapper with Redis client and config."""
        self.redis = redis_client
        self.config = config
        self.logger = logging.getLogger(f"RedisClient[db={config.db}]")
    
    def set_namespaced(self, prefix, key: str, value: Any, 
                      ttl: Optional[int] = None, serializer: Callable = json.dumps) -> bool:
        """Set a namespaced key with optional TTL."""
        try:
            full_key = f"{prefix.value}{key}"
            serialized = serializer(value) if isinstance(value, (dict, list)) else value
            
            if ttl:
                self.redis.setex(full_key, ttl, serialized)
            else:
                self.redis.set(full_key, serialized)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set {prefix.value}{key}: {e}")
            return False
    
    def get_namespaced(self, prefix, key: str,
                      deserializer: Callable = json.loads) -> Optional[Any]:
        """Get a namespaced key with auto-deserialization."""
        try:
            full_key = f"{prefix.value}{key}"
            value = self.redis.get(full_key)
            
            if value is None:
                return None
            
            # Try deserializing if it looks like JSON
            if isinstance(value, str):
                try:
                    return deserializer(value)
                except (json.JSONDecodeError, ValueError):
                    return value
            
            return value
        except Exception as e:
            self.logger.error(f"Failed to get {prefix.value}{key}: {e}")
            return None
    
    def delete_namespaced(self, prefix, key: str) -> bool:
        """Delete a namespaced key."""
        try:
            full_key = f"{prefix.value}{key}"
            self.redis.delete(full_key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete {prefix.value}{key}: {e}")
            return False
    
    def exists_namespaced(self, prefix, key: str) -> bool:
        """Check if namespaced key exists."""
        try:
            full_key = f"{prefix.value}{key}"
            return bool(self.redis.exists(full_key))
        except Exception as e:
            self.logger.error(f"Failed to check existence of {prefix.value}{key}: {e}")
            return False
    
    def incr_namespaced(self, prefix, key: str, amount: int = 1) -> Optional[int]:
        """Atomically increment a namespaced counter."""
        try:
            full_key = f"{prefix.value}{key}"
            return self.redis.incrby(full_key, amount)
        except Exception as e:
            self.logger.error(f"Failed to increment {prefix.value}{key}: {e}")
            return None
    
    def lpush_namespaced(self, prefix, key: str, *values) -> Optional[int]:
        """Push to a namespaced list."""
        try:
            full_key = f"{prefix.value}{key}"
            serialized = [
                json.dumps(v) if isinstance(v, (dict, list)) else v
                for v in values
            ]
            return self.redis.lpush(full_key, *serialized)
        except Exception as e:
            self.logger.error(f"Failed to lpush {prefix.value}{key}: {e}")
            return None
    
    def lrange_namespaced(self, prefix, key: str, 
                         start: int = 0, end: int = -1) -> List[Any]:
        """Get range from a namespaced list."""
        try:
            full_key = f"{prefix.value}{key}"
            values = self.redis.lrange(full_key, start, end)
            result = []
            for v in values:
                if isinstance(v, str):
                    try:
                        result.append(json.loads(v))
                    except (json.JSONDecodeError, ValueError):
                        result.append(v)
                else:
                    result.append(v)
            return result
        except Exception as e:
            self.logger.error(f"Failed to lrange {prefix.value}{key}: {e}")
            return []
    
    def publish_namespaced(self, prefix, key: str, message: Any) -> int:
        """Publish to a namespaced channel."""
        try:
            full_key = f"{prefix.value}{key}"
            payload = json.dumps(message) if isinstance(message, (dict, list)) else message
            return self.redis.publish(full_key, payload)
        except Exception as e:
            self.logger.error(f"Failed to publish to {prefix.value}{key}: {e}")
            return 0
    
    def hset_namespaced(self, prefix, key: str, **mapping) -> int:
        """Set hash fields for a namespaced key."""
        try:
            full_key = f"{prefix.value}{key}"
            serialized = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in mapping.items()
            }
            return self.redis.hset(full_key, mapping=serialized)
        except Exception as e:
            self.logger.error(f"Failed to hset {prefix.value}{key}: {e}")
            return 0
    
    def hget_namespaced(self, prefix, key: str, field: str) -> Optional[Any]:
        """Get a hash field from namespaced key."""
        try:
            full_key = f"{prefix.value}{key}"
            value = self.redis.hget(full_key, field)
            
            if value is None:
                return None
            
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    return value
            
            return value
        except Exception as e:
            self.logger.error(f"Failed to hget {prefix.value}{key}.{field}: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            return self.redis.ping()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Global client registry
_clients: Dict[int, RedisClient] = {}
_config: Optional[Any] = None


def initialize_redis(config=None) -> None:
    """Initialize Redis configuration and pools."""
    global _config
    
    if config is None:
        from .redis_config import RedisConfig
        config = RedisConfig.from_env()
    
    _config = config
    
    # Initialize resilient pool (shared for all clients)
    try:
        from resilience.redis_pool import initialize_global_pool
        initialize_global_pool(
            host=config.host,
            port=config.port,
            db=config.db,
            max_retries=config.max_retries,
            initial_backoff=config.initial_backoff,
            circuit_breaker_threshold=config.circuit_breaker_threshold,
            half_open_timeout=config.half_open_timeout,
        )
    except Exception as e:
        logger.error(f"Failed to initialize resilient pool: {e}")
    
    logger.info(f"[💾] Redis initialized: {config.host}:{config.port}")
    logger.info(f"    Pool settings: max_connections={config.max_connections}, max_retries={config.max_retries}")


def get_redis_client(db: Union[int, Any, None] = None) -> RedisClient:
    """Get or create a RedisClient for a specific database."""
    global _config, _clients
    
    if db is None:
        from .redis_config import RedisDB
        db = RedisDB.DEFAULT
    
    if _config is None:
        initialize_redis()
    
    # Explicitly convert db to int with proper type narrowing
    try:
        # Try accessing .value for enum members
        db_num: int = int(getattr(db, 'value', db))
    except (TypeError, AttributeError):
        db_num = int(db)
    
    if db_num not in _clients:
        # Create new client for this DB
        try:
            from redis import Redis
            
            if _config is None:
                raise RuntimeError("Redis config not initialized")
            
            config_dict = _config.to_dict()
            config_dict['db'] = db_num
            
            # Use resilient pool if available
            try:
                from resilience.redis_pool import get_global_pool
                pool = get_global_pool()
                if pool and pool.pool:
                    redis_client = Redis(connection_pool=pool.pool)
                else:
                    redis_client = Redis(**config_dict)
            except Exception:
                redis_client = Redis(**config_dict)
            
            _clients[db_num] = RedisClient(redis_client, _config)
            logger.debug(f"Created RedisClient for DB {db_num}")
        except Exception as e:
            logger.error(f"Failed to create RedisClient for DB {db_num}: {e}")
            # Return a dummy client that fails gracefully
            return _create_dummy_client(_config)
    
    return _clients[db_num]


def _create_dummy_client(config) -> RedisClient:
    """Create a dummy Redis client that handles None gracefully."""
    class DummyPubSub:
        def subscribe(self, *args, **kwargs):
            pass
        def unsubscribe(self, *args, **kwargs):
            pass
        def listen(self):
            return iter([])
        def get_message(self, *args, **kwargs):
            return None
        def close(self):
            pass

    class DummyRedis:
        is_dummy = True

        def pubsub(self):
            return DummyPubSub()

        def ping(self):
            return False
        def get(self, key):
            return None
        def set(self, key, value, ex=None):
            return False
        def delete(self, key):
            return False
        def exists(self, key):
            return False
        def lpush(self, key, *values):
            return 0
        def lrange(self, key, start, end):
            return []
        def publish(self, channel, message):
            return 0
        def incrby(self, key, amount):
            return 0
        def hset(self, key, mapping=None, **kwargs):
            return 0
        def hget(self, key, field):
            return None
        def setex(self, key, time, value):
            return False
    
    return RedisClient(DummyRedis(), config)


@contextmanager
def redis_context(db=None):
    """Context manager for Redis operations."""
    client = get_redis_client(db)
    try:
        yield client
    except Exception as e:
        logger.error(f"Redis operation failed: {e}")
        raise


def get_redis_health() -> Dict[str, Any]:
    """Get health status of all Redis connections."""
    try:
        from resilience.redis_pool import get_global_pool
        pool = get_global_pool()
    except Exception:
        pool = None
    
    health_info = {
        "pool_healthy": pool is not None,
        "databases": {}
    }
    
    if pool:
        pool_health = pool.health_check()
        health_info.update(pool_health)
    
    # Check each active database
    for db_num, client in _clients.items():
        health_info["databases"][db_num] = {
            "healthy": client.health_check()
        }
    
    return health_info
