# resilience.py
import hashlib, time, random, logging, threading
from urllib.parse import urlparse
from typing import Dict, Optional

LOG = logging.getLogger("Resilience")


class CircuitBreaker:
    """
    Simple state machine: OK → OPEN after N failures.
    OPEN → HALF_OPEN after cooldown.
    """
    def __init__(self, failure_limit: int = 5, timeout: int = 60):
        self.failure_limit = failure_limit
        self.timeout = timeout
        self.failures = 0
        self.state = "OK"
        self.last_failure = 0.0
        self._lock = threading.Lock()

    def call(self, func, *args, **kw):
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure > self.timeout:
                    self.state = "HALF_OPEN"
                    self.failures = 0
                else:
                    raise RuntimeError("Circuit OPEN")

        try:
            result = func(*args, **kw)
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "OK"
                    LOG.info("[CircuitBreaker] Recovered → OK")
            return result
        except Exception as exc:
            with self._lock:
                self.failures += 1
                self.last_failure = time.time()
                if self.failures >= self.failure_limit:
                    if self.state != "OPEN":
                        LOG.warning(
                            "[CircuitBreaker] OPEN after %d failures (last: %s)",
                            self.failures, exc,
                        )
                    self.state = "OPEN"
            raise exc


# ------------------------------------------------------------------
# Global named-breaker registry
# ------------------------------------------------------------------
class CircuitBreakerRegistry:
    """
    Process-singleton registry of named CircuitBreaker instances.
    Call CircuitBreakerRegistry.get("shodan") anywhere to obtain the
    shared breaker for that logical channel — state persists across calls.
    """
    _lock = threading.Lock()
    _breakers: Dict[str, "CircuitBreaker"] = {}

    @classmethod
    def get(
        cls,
        name: str,
        failure_limit: int = 5,
        timeout: int = 60,
    ) -> "CircuitBreaker":
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(
                    failure_limit=failure_limit, timeout=timeout
                )
                LOG.debug("[CBRegistry] Registered breaker '%s'", name)
            return cls._breakers[name]

    @classmethod
    def status(cls) -> Dict[str, str]:
        """Return {name: state} for all registered breakers."""
        with cls._lock:
            return {n: b.state for n, b in cls._breakers.items()}


# ------------------------------------------------------------------
# Retry with exponential back-off + jitter
# ------------------------------------------------------------------
class RetryWithBackoff:
    """
    Wraps a callable with exponential back-off and optional jitter.
    Retries on ConnectionError, Timeout, and HTTP 429.
    """
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.3,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def call(self, func, *args, **kw):
        import requests  # local import — avoid hard dep at module level

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kw)
                # Treat certain HTTP status codes as transient failures
                status = getattr(result, "status_code", None)
                if status in self._RETRYABLE_STATUS:
                    raise requests.exceptions.HTTPError(
                        f"Retryable HTTP {status}", response=result
                    )
                return result
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                delay *= 1 + self.jitter * (random.random() - 0.5) * 2
                LOG.debug(
                    "[RetryBackoff] attempt %d/%d failed (%s); retry in %.1fs",
                    attempt + 1, self.max_retries, exc, delay,
                )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]


# ------------------------------------------------------------------
# Canary generator
# ------------------------------------------------------------------
def make_canary(domain: str) -> str:
    return f"https://canary.blackroot/{int(time.time())}-{random.randint(1000,9999)}.{domain}"


# ------------------------------------------------------------------
# Checksum cache
# ------------------------------------------------------------------
class ChecksumCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}  # url → sha256

    def seen(self, url: str, content: bytes) -> bool:
        h = hashlib.sha256(content).hexdigest()
        if h == self._cache.get(url):
            return True
        self._cache[url] = h
        return False