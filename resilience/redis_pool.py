"""
Redis connection pooling and resilience patterns for Blackroot.

Provides:
- Connection pooling with configurable limits
- Exponential backoff retry logic
- Circuit breaker pattern for graceful degrada
- Connection health checks
- Automatic reconnection with jitter
"""

import logging
import time
from typing import Optional, Any, Dict
from redis import Redis, ConnectionPool, exceptions
from threading import Lock, Event
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for Redis connection."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class ResilientRedisPool:
    """
    Thread-safe Redis connection pool with resilience patterns.
    
    Features:
    - Exponential backoff retry (0.5s → 32s)
    - Circuit breaker (open after 5 failures, half-open after 30s)
    - Health checks before use
    - Jittered reconnection
    - Graceful degradation
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        circuit_breaker_threshold: int = 5,
        half_open_timeout: float = 30.0
    ):
        self.host = host
        self.port = port
        self.db = db
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.half_open_timeout = half_open_timeout
        
        # Circuit breaker state
        self.circuit_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.circuit_lock = Lock()
        
        # Connection pool
        self.pool: Optional[ConnectionPool] = None
        self._pool_lock = Lock()
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                max_connections=10,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 1,  # TCP_KEEPINTVL  
                    3: 1,  # TCP_KEEPCNT
                },
                decode_responses=True,
                health_check_interval=30
            )
            logger.info(f"[💾] Redis pool initialized: {self.host}:{self.port}/{self.db}")
        except Exception as e:
            logger.error(f"[❌] Failed to initialize Redis pool: {e}")
            self.pool = None
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open; transition if needed."""
        with self.circuit_lock:
            current_time = time.time()
            
            # If open, check if we should try half-open
            if self.circuit_state == CircuitBreakerState.OPEN:
                if (current_time - self.last_failure_time) > self.half_open_timeout:
                    self.circuit_state = CircuitBreakerState.HALF_OPEN
                    self.failure_count = 0
                    logger.warning("[⚠️] Circuit breaker HALF_OPEN: attempting recovery")
                    return True  # Allow attempt
                else:
                    logger.debug(f"[🔴] Circuit OPEN: blocking requests for {self.half_open_timeout}s")
                    return False  # Reject
            
            # If half-open, allow single attempt
            if self.circuit_state == CircuitBreakerState.HALF_OPEN:
                return True
            
            # If closed, normal operation
            return True
    
    def _record_failure(self) -> None:
        """Record failure and transition circuit state if needed."""
        with self.circuit_lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.circuit_breaker_threshold:
                self.circuit_state = CircuitBreakerState.OPEN
                logger.error(f"[❌] Circuit breaker OPEN: {self.failure_count} failures")
    
    def _record_success(self) -> None:
        """Record successful operation; reset circuit state."""
        with self.circuit_lock:
            if self.circuit_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("[✅] Circuit breaker CLOSED: Redis recovered")
            elif self.circuit_state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # Reset failure counter on success
    
    def get_connection(self, timeout: Optional[float] = None) -> Optional[Redis]:
        """
        Get a Redis connection with retry and circuit breaker logic.
        
        Args:
            timeout: Max time to wait for connection
            
        Returns:
            Redis client or None if unavailable
        """
        if not self._check_circuit_breaker():
            return None
        
        backoff = self.initial_backoff
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.pool is None:
                    self._initialize_pool()
                
                if self.pool is None:
                    raise ConnectionError("Redis pool initialization failed")
                
                # Get connection from pool
                client = Redis(connection_pool=self.pool)
                
                # Test connection with ping
                client.ping()
                
                self._record_success()
                logger.debug(f"[✅] Redis connection successful (attempt {attempt + 1})")
                return client
                
            except (exceptions.ConnectionError, exceptions.TimeoutError, ConnectionError) as e:
                self._record_failure()
                
                if attempt < self.max_retries:
                    # Calculate backoff with jitter: backoff * (1 ± 0.1)
                    jittered_backoff = backoff * (0.9 + 0.2 * (time.time() % 1))
                    
                    elapsed = time.time() - start_time
                    if timeout and (elapsed + jittered_backoff) > timeout:
                        logger.warning(f"[⏱️] Redis connection timeout after {elapsed:.1f}s")
                        return None
                    
                    logger.warning(
                        f"[⚠️] Redis connection failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {jittered_backoff:.2f}s..."
                    )
                    time.sleep(jittered_backoff)
                    backoff = min(backoff * 2, 32.0)  # Exponential backoff, max 32s
                else:
                    logger.error(f"[❌] Redis connection failed after {self.max_retries + 1} attempts: {e}")
                    return None
            
            except Exception as e:
                logger.error(f"[❌] Unexpected error connecting to Redis: {e}")
                self._record_failure()
                return None
        
        return None
    
    @contextmanager
    def get_connection_context(self, timeout: Optional[float] = None):
        """
        Context manager for Redis connections.
        
        Usage:
            with redis_pool.get_connection_context() as redis_client:
                redis_client.set("key", "value")
        """
        client = self.get_connection(timeout=timeout)
        try:
            yield client
        finally:
            if client:
                client.close()
    
    def execute_with_retry(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with Redis connection retry logic.
        
        Args:
            func: Callable that takes Redis client as first arg
            *args: Additional positional args
            max_retries: Override max retries for this call
            **kwargs: Additional keyword args
            
        Returns:
            Result of func or None if all retries failed
        """
        max_retries = max_retries or self.max_retries
        backoff = self.initial_backoff
        
        for attempt in range(max_retries + 1):
            client = self.get_connection()
            if client is None:
                if attempt < max_retries:
                    jittered_backoff = backoff * (0.9 + 0.2 * (time.time() % 1))
                    time.sleep(jittered_backoff)
                    backoff = min(backoff * 2, 32.0)
                continue
            
            try:
                result = func(client, *args, **kwargs)
                self._record_success()
                return result
            except (exceptions.ConnectionError, exceptions.TimeoutError) as e:
                logger.warning(f"[⚠️] Redis operation failed: {e}. Retry {attempt + 1}/{max_retries + 1}")
                self._record_failure()
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 32.0)
            except Exception as e:
                logger.error(f"[❌] Unexpected error in Redis operation: {e}")
                return None
            finally:
                client.close()
        
        logger.error(f"[❌] Redis operation failed after {max_retries + 1} attempts")
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis pool health.
        
        Returns:
            Dict with connectivity, circuit state, failure count
        """
        client = self.get_connection()
        is_healthy = client is not None
        
        if client:
            client.close()
        
        return {
            "is_healthy": is_healthy,
            "circuit_state": self.circuit_state.value,
            "failure_count": self.failure_count,
            "host": self.host,
            "port": self.port,
            "last_failure_time": self.last_failure_time
        }
    
    def close(self) -> None:
        """Disconnect the pool."""
        if self.pool:
            self.pool.disconnect()
            logger.info("[💾] Redis pool closed")


# Global singleton pool
_redis_pool: Optional[ResilientRedisPool] = None
_pool_lock = Lock()


def initialize_global_pool(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    **kwargs
) -> ResilientRedisPool:
    """Initialize or get the global Redis pool."""
    global _redis_pool
    
    with _pool_lock:
        if _redis_pool is None:
            _redis_pool = ResilientRedisPool(host=host, port=port, db=db, **kwargs)
        return _redis_pool


def get_global_pool() -> Optional[ResilientRedisPool]:
    """Get the global Redis pool."""
    return _redis_pool
