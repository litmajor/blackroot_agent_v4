import json, time, logging
from typing import Dict, Any, Optional
from redis import Redis

class IdentityMap:
    """
    Distributed identity registry:
    - Redis hash per key (TTL auto-cleanup)
    - SwarmMesh broadcast on every set/del
    - local in-memory cache for ultra-fast reads
    """
    def __init__(self,
                 redis: Redis,
                 channel: str = "identity_events",
                 ttl: int = 3600):
        self.redis   = redis
        self.channel = channel
        self.ttl     = ttl
        self.logger  = logging.getLogger("IdentityMap")
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ---------- public API ----------
    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self.redis.hset("identities", key, json.dumps(value))
        self.redis.expire("identities", self.ttl)
        self._cache[key] = value
        # Value is always a dict, so just publish as-is
        self.redis.publish(self.channel, json.dumps({"op": "set", "key": key, "value": value}))

    def __getitem__(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._cache:
            return self._cache[key]
        raw = self.redis.hget("identities", key)
        # Handle both sync and async Redis clients
        import inspect, asyncio
        if inspect.isawaitable(raw):
            raw = asyncio.get_event_loop().run_until_complete(raw)
        if raw:
            if isinstance(raw, bytes):
                raw = raw.decode()
            val = json.loads(raw)
            self._cache[key] = val
            return val
        return None

    def __delitem__(self, key: str) -> None:
        self.redis.hdel("identities", key)
        self._cache.pop(key, None)
        self.redis.publish(self.channel, json.dumps({"op": "del", "key": key}))

    def keys(self):
        keys = self.redis.hkeys("identities")
        import inspect, asyncio
        if inspect.isawaitable(keys):
            keys = asyncio.get_event_loop().run_until_complete(keys)
        return [k.decode() if isinstance(k, bytes) else k for k in keys]

    def sync(self):
        """Refresh local cache from Redis."""
        hgetall_result = self.redis.hgetall("identities")
        import inspect, asyncio
        if inspect.isawaitable(hgetall_result):
            hgetall_result = asyncio.get_event_loop().run_until_complete(hgetall_result)
        for k, v in hgetall_result.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            try:
                self._cache[key] = json.loads(val)
            except Exception as e:
                self.logger.error(f"Failed to decode identity for key {key}: {e}")