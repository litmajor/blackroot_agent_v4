# resilience.py
import hashlib, time, random, logging
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

    def call(self, func, *args, **kw):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
                self.failures = 0
            else:
                raise RuntimeError("Circuit OPEN")

        try:
            result = func(*args, **kw)
            if self.state == "HALF_OPEN":
                self.state = "OK"
            return result
        except Exception as exc:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.failure_limit:
                self.state = "OPEN"
            raise exc

# ------------------------------------------------------------------
# Canary generator
def make_canary(domain: str) -> str:
    return f"https://canary.blackroot/{int(time.time())}-{random.randint(1000,9999)}.{domain}"

# ------------------------------------------------------------------
# Checksum cache
class ChecksumCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}  # url → sha256

    def seen(self, url: str, content: bytes) -> bool:
        h = hashlib.sha256(content).hexdigest()
        if h == self._cache.get(url):
            return True
        self._cache[url] = h
        return False