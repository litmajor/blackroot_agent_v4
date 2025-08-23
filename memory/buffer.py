import json, logging, threading, time
from typing import Any, Dict, List
from redis import Redis

class MissionBuffer:
    """
    Thread-safe mission queue:
    - local in-memory list for ultra-fast enqueue
    - background thread flushes to Redis every N ms
    - SwarmMesh broadcast on every push
    """
    def __init__(self,
                 redis: Redis,
                 channel: str = "mission_buffer",
                 flush_interval: float = 0.5):
        self.redis      = redis
        self.channel    = channel
        self.flush_interval = flush_interval
        self.logger     = logging.getLogger("MissionBuffer")
        self._tasks: List[Dict[str, Any]] = []
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()

        # background flusher
        self._thread = threading.Thread(target=self._flusher, daemon=True)
        self._thread.start()

    def push(self, task: Dict[str, Any]):
        """Non-blocking enqueue."""
        with self._lock:
            self._tasks.append(task)
        # real-time broadcast to SwarmMesh
        self.redis.publish(self.channel, json.dumps(task))

    def pop_all(self) -> List[Dict[str, Any]]:
        """Atomic drain."""
        with self._lock:
            out, self._tasks = self._tasks[:], []
        return out

    def sync(self):
        """Manual flush (blocks until Redis ACK)."""
        batch = self.pop_all()
        if not batch:
            return
        try:
            # store entire batch as one key for replay
            self.redis.lpush("mission_log", json.dumps(batch))
            self.logger.info("Synced %d tasks", len(batch))
        except Exception as e:
            self.logger.error("Sync failed: %s", e)

    def _flusher(self):
        """Auto-sync every flush_interval seconds."""
        while not self._stop_evt.is_set():
            time.sleep(self.flush_interval)
            self.sync()

    def shutdown(self):
        """Graceful stop."""
        self._stop_evt.set()
        self._thread.join()
        self.sync()  # final drain