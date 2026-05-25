"""
Backpressure & Circuit Breaker for Resource Management
- Queue depth limits and overflow handling
- Circuit breaker pattern for fault isolation
- Adaptive rate limiting
- Resource usage tracking
"""

import time
import threading
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

log = logging.getLogger("BACKPRESSURE")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"          # Normal operation
    OPEN = "OPEN"              # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"    # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: int = 60           # Time before trying half-open
    failure_rate_threshold: float = 0.5 # Failure % to open


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_call_time: float = field(default_factory=time.time)
    last_failure_time: float = field(default_factory=time.time)

    def get_failure_rate(self) -> float:
        """Get failure rate percentage."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitBreaker:
    """
    Circuit breaker for fault isolation.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject requests
    - HALF_OPEN: Testing recovery
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Breaker name
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.lock = threading.RLock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Raises:
            CircuitBreakerOpenException: If circuit is open
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if timeout expired (try half-open)
                if time.time() - self.metrics.last_failure_time > self.config.timeout_seconds:
                    self.state = CircuitBreakerState.HALF_OPEN
                    log.info(f"Circuit {self.name} entering HALF_OPEN state")
                else:
                    self.metrics.rejected_calls += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit {self.name} is OPEN"
                    )

            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise

    def _record_success(self):
        """Record successful call."""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            # In half-open, count successes to close
            if self.metrics.successful_calls >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                log.info(f"Circuit {self.name} CLOSED after successful test")

    def _record_failure(self):
        """Record failed call."""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            log.warning(f"Circuit {self.name} OPEN after test failure")
        elif self.state == CircuitBreakerState.CLOSED:
            # Check thresholds
            failure_rate = self.metrics.get_failure_rate()

            if self.metrics.failed_calls >= self.config.failure_threshold or \
               failure_rate >= self.config.failure_rate_threshold:
                self.state = CircuitBreakerState.OPEN
                log.warning(f"Circuit {self.name} OPEN: {self.metrics.failed_calls} failures, "
                           f"rate {failure_rate:.1%}")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.metrics.total_calls,
                'successful_calls': self.metrics.successful_calls,
                'failed_calls': self.metrics.failed_calls,
                'rejected_calls': self.metrics.rejected_calls,
                'failure_rate': f"{self.metrics.get_failure_rate():.1%}"
            }


class CircuitBreakerOpenException(Exception):
    """Raised when circuit is open."""
    pass


@dataclass
class QueueStats:
    """Statistics for bounded queue."""
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_dropped: int = 0
    peak_size: int = 0

    def get_drop_rate(self) -> float:
        """Get drop rate percentage."""
        if self.total_enqueued == 0:
            return 0.0
        return self.total_dropped / self.total_enqueued


class BoundedQueue:
    """
    Queue with depth limit to prevent unbounded growth.
    
    Features:
    - Maximum depth
    - Drop oldest/newest on overflow
    - Priority support
    """

    def __init__(self, name: str, max_size: int = 1000,
                 overflow_strategy: str = "drop_oldest"):
        """
        Initialize bounded queue.
        
        Args:
            name: Queue name
            max_size: Maximum queue size
            overflow_strategy: "drop_oldest", "drop_newest", "block"
        """
        self.name = name
        self.max_size = max_size
        self.overflow_strategy = overflow_strategy
        self.queue: deque = deque(maxlen=None)
        self.stats = QueueStats()
        self.lock = threading.RLock()

    def enqueue(self, item: Any, priority: int = 0) -> bool:
        """
        Enqueue item.
        
        Args:
            item: Item to enqueue
            priority: Priority (higher = more important)
            
        Returns:
            True if enqueued, False if dropped
        """
        with self.lock:
            if len(self.queue) >= self.max_size:
                if self.overflow_strategy == "drop_oldest":
                    # Remove lowest priority item
                    if self.queue:
                        self.queue.popleft()
                    self.stats.total_dropped += 1
                elif self.overflow_strategy == "drop_newest":
                    self.stats.total_dropped += 1
                    return False
                elif self.overflow_strategy == "block":
                    # Caller should handle blocking
                    self.stats.total_dropped += 1
                    return False

            self.queue.append((item, priority, time.time()))
            self.stats.total_enqueued += 1
            self.stats.peak_size = max(self.stats.peak_size, len(self.queue))

            return True

    def dequeue(self) -> Optional[Any]:
        """Dequeue highest priority item."""
        with self.lock:
            if not self.queue:
                return None

            # Sort by priority (descending), then by time (ascending)
            sorted_items = sorted(self.queue, key=lambda x: (-x[1], x[2]))
            best_item = sorted_items[0]

            self.queue.remove(best_item)
            self.stats.total_dequeued += 1

            return best_item[0]

    def get_size(self) -> int:
        """Get current queue size."""
        with self.lock:
            return len(self.queue)

    def get_fill_percentage(self) -> float:
        """Get queue fill percentage."""
        with self.lock:
            return (len(self.queue) / self.max_size) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                'queue_name': self.name,
                'current_size': len(self.queue),
                'max_size': self.max_size,
                'fill_percent': f"{self.get_fill_percentage():.1f}%",
                'total_enqueued': self.stats.total_enqueued,
                'total_dequeued': self.stats.total_dequeued,
                'total_dropped': self.stats.total_dropped,
                'drop_rate': f"{self.stats.get_drop_rate():.1%}",
                'peak_size': self.stats.peak_size
            }


class AdaptiveRateLimiter:
    """
    Rate limiter that adapts based on system load.
    
    Uses token bucket with dynamic refill rate.
    """

    def __init__(self, name: str, base_rate: float = 1000):
        """
        Initialize rate limiter.
        
        Args:
            name: Limiter name
            base_rate: Base requests per second
        """
        self.name = name
        self.base_rate = base_rate
        self.tokens = base_rate
        self.max_tokens = base_rate
        self.last_refill = time.time()
        self.load_factor = 1.0  # Adapt based on load
        self.allowed_count = 0
        self.denied_count = 0
        self.lock = threading.RLock()

    def allow_request(self) -> bool:
        """Check if request is allowed."""
        with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_refill

            # Refill tokens
            refill_rate = self.base_rate * self.load_factor
            tokens_to_add = time_passed * refill_rate
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = current_time

            if self.tokens >= 1:
                self.tokens -= 1
                self.allowed_count += 1
                return True
            else:
                self.denied_count += 1
                return False

    def update_load_factor(self, system_load: float):
        """
        Update load factor (0.0-2.0).
        
        0.5 = half rate, 1.0 = normal, 2.0 = double rate
        """
        with self.lock:
            self.load_factor = max(0.1, min(2.0, system_load))
            log.debug(f"Rate limiter {self.name} load factor: {self.load_factor:.2f}")

    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        with self.lock:
            return {
                'name': self.name,
                'base_rate': self.base_rate,
                'effective_rate': f"{self.base_rate * self.load_factor:.0f}",
                'current_tokens': f"{self.tokens:.1f}",
                'allowed': self.allowed_count,
                'denied': self.denied_count
            }


class BackpressureManager:
    """
    Manages overall backpressure and resource limits.
    
    Coordinates:
    - Circuit breakers
    - Bounded queues
    - Rate limiting
    """

    def __init__(self):
        """Initialize backpressure manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bounded_queues: Dict[str, BoundedQueue] = {}
        self.rate_limiters: Dict[str, AdaptiveRateLimiter] = {}
        self.system_load = 0.5  # 0-1.0
        self.lock = threading.RLock()

    def create_circuit_breaker(self, name: str, 
                              config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create or get circuit breaker."""
        with self.lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, config)
            return self.circuit_breakers[name]

    def create_bounded_queue(self, name: str, max_size: int = 1000) -> BoundedQueue:
        """Create or get bounded queue."""
        with self.lock:
            if name not in self.bounded_queues:
                self.bounded_queues[name] = BoundedQueue(name, max_size)
            return self.bounded_queues[name]

    def create_rate_limiter(self, name: str, base_rate: float = 1000) -> AdaptiveRateLimiter:
        """Create or get rate limiter."""
        with self.lock:
            if name not in self.rate_limiters:
                self.rate_limiters[name] = AdaptiveRateLimiter(name, base_rate)
            return self.rate_limiters[name]

    def update_system_load(self, load: float):
        """Update system load (0-1.0)."""
        with self.lock:
            self.system_load = max(0.0, min(1.0, load))

            # Adapt rate limiters based on load
            for limiter in self.rate_limiters.values():
                # Reduce rate as load increases
                limiter.update_load_factor(1.0 - (self.system_load * 0.5))

    def get_status(self) -> Dict[str, Any]:
        """Get overall backpressure status."""
        with self.lock:
            return {
                'system_load': f"{self.system_load:.1%}",
                'circuit_breakers': len(self.circuit_breakers),
                'bounded_queues': len(self.bounded_queues),
                'rate_limiters': len(self.rate_limiters),
                'circuit_breaker_status': [cb.get_status() for cb in self.circuit_breakers.values()],
                'queue_status': [q.get_stats() for q in self.bounded_queues.values()]
            }
