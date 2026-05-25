import json, logging, threading, time
from typing import Any, Dict, List, Optional

from core.redis_client import get_redis_client
from core.redis_config import RedisDB

class MissionBuffer:
    """
    Thread-safe mission queue with size limits:
    - local in-memory list for ultra-fast enqueue
    - background thread flushes to Redis every N ms
    - SwarmMesh broadcast on every push
    - Enforces max queue size with overflow handling
    """
    def __init__(self,
                 redis=None,
                 channel: str = "mission_buffer",
                 flush_interval: float = 0.5,
                 max_queue_size: int = 10000,
                 overflow_policy: str = "drop_old"):
        """
        Initialize mission buffer.
        
        Args:
            redis: Optional Redis client (uses factory if None)
            channel: Pub/sub channel for broadcasts
            flush_interval: Seconds between Redis flushes
            max_queue_size: Maximum missions in buffer before overflow
            overflow_policy: "drop_old" (remove oldest) or "drop_new" (reject new)
        """
        # Use factory if no redis client provided
        if redis is None:
            redis_wrapper = get_redis_client(RedisDB.MISSION_BUFFER)
            self.redis = redis_wrapper.redis
            self.redis_client_wrapper = redis_wrapper
        else:
            self.redis = redis
            self.redis_client_wrapper = None
        
        self.channel    = channel
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        self.overflow_policy = overflow_policy
        self.logger     = logging.getLogger("MissionBuffer")
        self._tasks: List[Dict[str, Any]] = []
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        
        # Metrics
        self.overflow_count = 0
        self.total_queued = 0

        # background flusher
        self._thread = threading.Thread(target=self._flusher, daemon=True)
        self._thread.start()

    def push(self, task: Dict[str, Any]) -> bool:
        """
        Non-blocking enqueue with size limit enforcement.
        
        Args:
            task: Mission to queue
            
        Returns:
            True if queued, False if rejected by overflow policy
        """
        with self._lock:
            # Check size limit
            if len(self._tasks) >= self.max_queue_size:
                self.overflow_count += 1
                
                if self.overflow_policy == "drop_old":
                    # Remove oldest (FIFO)
                    dropped = self._tasks.pop(0)
                    self.logger.warning(
                        f"[🚨] Queue overflow (size: {len(self._tasks) + 1}). "
                        f"Dropping oldest mission: {dropped.get('mission_id', 'unknown')}"
                    )
                elif self.overflow_policy == "drop_new":
                    # Reject new
                    self.logger.warning(
                        f"[🚨] Queue full (size: {self.max_queue_size}). "
                        f"Rejecting new mission: {task.get('mission_id', 'unknown')}"
                    )
                    return False
            
            self._tasks.append(task)
            self.total_queued += 1
        
        # real-time broadcast to SwarmMesh
        try:
            self.redis.publish(self.channel, json.dumps(task))
        except Exception as e:
            self.logger.error(f"Failed to publish mission: {e}")
        
        return True

    def pop_all(self) -> List[Dict[str, Any]]:
        """Atomic drain."""
        with self._lock:
            out, self._tasks = self._tasks[:], []
        return out

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._tasks)

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

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._tasks),
                "max_size": self.max_queue_size,
                "overflow_count": self.overflow_count,
                "total_queued": self.total_queued,
                "overflow_policy": self.overflow_policy
            }

    def shutdown(self):
        """Graceful stop."""
        self._stop_evt.set()
        self._thread.join()
        self.sync()  # final drain