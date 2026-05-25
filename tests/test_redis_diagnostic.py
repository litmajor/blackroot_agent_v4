#!/usr/bin/env python3
"""
Redis Diagnostic Test
Verifies Redis connectivity and integration with Blackroot C2 API
"""

import sys
import time
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_raw_redis_connection() -> bool:
    """Test raw Redis connection using redis-py."""
    logger.info("=" * 60)
    logger.info("TEST 1: Raw Redis Connection")
    logger.info("=" * 60)
    
    try:
        import redis
        logger.info("✓ redis-py imported successfully")
        
        # Connect to Redis
        client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            decode_responses=True
        )
        
        # Ping Redis
        pong = client.ping()
        logger.info(f"✓ Redis PING: {pong}")
        
        # Get info
        info = client.info('server')
        logger.info(f"✓ Redis Server Version: {info.get('redis_version')}")
        logger.info(f"✓ Redis Uptime: {info.get('uptime_in_seconds')} seconds")
        
        return True
    except Exception as e:
        logger.error(f"✗ Raw Redis connection failed: {e}")
        return False


def test_redis_config() -> bool:
    """Test Redis configuration loading."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Redis Config Loading")
    logger.info("=" * 60)
    
    try:
        from core.redis_config import RedisConfig, RedisDB, RedisKeyPrefix
        
        config = RedisConfig.from_env()
        logger.info(f"✓ Config loaded from environment")
        logger.info(f"  Host: {config.host}")
        logger.info(f"  Port: {config.port}")
        logger.info(f"  DB: {config.db}")
        logger.info(f"  URL: {config.redis_url}")
        logger.info(f"  Timeout: {config.socket_timeout}s")
        logger.info(f"  Max Connections: {config.max_connections}")
        
        # Verify RedisDB enum
        logger.info(f"\n✓ RedisDB databases registered:")
        for db in RedisDB:
            logger.info(f"    {db.name} = {db.value}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Redis config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_redis_client_factory() -> bool:
    """Test RedisClient factory."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: RedisClient Factory")
    logger.info("=" * 60)
    
    try:
        from core.redis_client import initialize_redis, get_redis_client
        from core.redis_config import RedisDB, RedisKeyPrefix
        
        # Initialize
        initialize_redis()
        logger.info("✓ Redis initialized")
        
        # Get default client
        client = get_redis_client()
        logger.info(f"✓ Default client created: {type(client).__name__}")
        
        # Test basic operations with namespace
        test_key = "diagnostic_test_key"
        test_value = "hello_world"
        prefix = RedisKeyPrefix.TELEMETRY  # Use proper prefix enum
        
        # Set
        client.set_namespaced(prefix, test_key, test_value, ttl=60)
        logger.info(f"✓ Set operation: {prefix.value}{test_key} = {test_value}")
        
        # Get
        retrieved = client.get_namespaced(prefix, test_key)
        if retrieved == test_value:
            logger.info(f"✓ Get operation: {retrieved}")
        else:
            logger.error(f"✗ Get operation: expected '{test_value}', got '{retrieved}'")
            return False
        
        # Cleanup
        client.delete_namespaced(prefix, test_key)
        logger.info(f"✓ Delete operation: {prefix.value}{test_key}")
        
        return True
    except Exception as e:
        logger.error(f"✗ RedisClient factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_swarm_mesh_redis() -> bool:
    """Test SwarmMesh Redis integration."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: SwarmMesh Redis Integration")
    logger.info("=" * 60)
    
    try:
        from swarm_mesh import SwarmMesh
        
        # SwarmMesh requires a node_id
        node_id = "diagnostic_node_test"
        swarm = SwarmMesh(node_id=node_id)
        logger.info(f"✓ SwarmMesh created with node_id: {node_id}")
        
        if hasattr(swarm, 'redis') and swarm.redis is not None:
            logger.info(f"✓ SwarmMesh.redis is configured: {type(swarm.redis).__name__}")
            
            # Test SwarmMesh Redis operations
            if hasattr(swarm.redis, 'ping'):
                try:
                    pong = swarm.redis.ping()
                    logger.info(f"✓ SwarmMesh Redis PING: {pong}")
                    return True
                except Exception as e:
                    logger.error(f"✗ SwarmMesh Redis PING failed: {e}")
                    return False
            else:
                logger.warning("⚠ SwarmMesh.redis doesn't have ping method")
                return True
        else:
            logger.warning("⚠ SwarmMesh.redis is not configured")
            return False
            
    except Exception as e:
        logger.error(f"✗ SwarmMesh Redis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_redis_integration() -> bool:
    """Test that the C2 API can access Redis."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: C2 API Redis Integration")
    logger.info("=" * 60)
    
    try:
        # Check what the API module initializes
        logger.info("Checking blackroot_c2_api.py initialization...")
        
        import redis
        from redis import Redis
        
        # Simulate what the API does
        REDIS_URL = "redis://localhost:6379"
        try:
            sync_redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            logger.info(f"✓ Sync Redis client created from URL: {REDIS_URL}")
            
            # Test the connection
            pong = sync_redis_client.ping()
            logger.info(f"✓ Sync Redis client PING: {pong}")
        except Exception as e:
            logger.error(f"✗ Sync Redis client failed: {e}")
            return False
        
        try:
            import aioredis
            # Note: aioredis v2.x is actually just redis with async support
            async_redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            logger.info(f"✓ Async Redis client created from URL: {REDIS_URL}")
        except ImportError:
            logger.warning("⚠ aioredis not available (async support disabled)")
        except Exception as e:
            logger.error(f"✗ Async Redis client failed: {e}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"✗ C2 API Redis integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_redis_data_in_db() -> bool:
    """Inspect what's currently in Redis."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Redis Database Contents")
    logger.info("=" * 60)
    
    try:
        import redis
        
        client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Get all keys
        keys = client.keys('*')
        logger.info(f"✓ Total keys in Redis DB 0: {len(keys)}")
        
        if keys:
            logger.info("\nKey sample (first 20):")
            for i, key in enumerate(keys[:20]):
                try:
                    key_type = client.type(key)
                    ttl = client.ttl(key)
                    ttl_str = f"TTL: {ttl}s" if ttl > 0 else ("No TTL" if ttl == -1 else "Expired")
                    
                    if key_type == 'string':
                        val = client.get(key)
                        if len(str(val)) > 50:
                            val = str(val)[:47] + "..."
                        logger.info(f"  [{i+1}] {key} ({key_type}) = {val} [{ttl_str}]")
                    else:
                        logger.info(f"  [{i+1}] {key} ({key_type}) [{ttl_str}]")
                except Exception as e:
                    logger.error(f"  Error reading key {key}: {e}")
        else:
            logger.info("  (Redis is empty)")
        
        # Memory stats
        info = client.info('memory')
        logger.info(f"\nMemory Usage: {info.get('used_memory_human')}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Redis database inspection failed: {e}")
        return False


def main():
    """Run all diagnostic tests."""
    logger.info("\n" + "🔍 REDIS DIAGNOSTIC SUITE 🔍".center(60))
    logger.info("Verifying Redis wiring for Blackroot C2 Agent\n")
    
    results: Dict[str, bool] = {}
    
    # Run tests
    results['Raw Connection'] = test_raw_redis_connection()
    results['Config Loading'] = test_redis_config()
    results['Client Factory'] = test_redis_client_factory()
    results['SwarmMesh Integration'] = test_swarm_mesh_redis()
    results['C2 API Integration'] = test_api_redis_integration()
    results['Database Contents'] = test_redis_data_in_db()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ Redis is properly wired and operational!")
        return 0
    else:
        logger.info(f"\n✗ {total - passed} test(s) failed - see details above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
