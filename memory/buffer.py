import json, logging, os, threading, time
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
    - Disk WAL fallback when Redis is offline / unreachable
    """
    def __init__(self,
                 redis=None,
                 channel: str = "mission_buffer",
                 flush_interval: float = 0.5,
                 max_queue_size: int = 10000,
                 overflow_policy: str = "drop_old",
                 wal_path: Optional[str] = "mission_wal.jsonl"):
        """
        Initialize mission buffer.
        
        Args:
            redis: Optional Redis client (uses factory if None)
            channel: Pub/sub channel for broadcasts
            flush_interval: Seconds between Redis flushes
            max_queue_size: Maximum missions in buffer before overflow
            overflow_policy: "drop_old" (remove oldest) or "drop_new" (reject new)
            wal_path: File path for local disk WAL fallback
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
        self.wal_path   = wal_path
        self._wal_lock  = threading.Lock()
        self.logger     = logging.getLogger("MissionBuffer")
        self._tasks: List[Dict[str, Any]] = []
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        
        # Metrics
        self.overflow_count = 0
        self.total_queued = 0
        self.wal_entries_count = 0

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

    def _is_redis_healthy(self) -> bool:
        """Check if Redis connection is live and not a dummy mock."""
        try:
            if getattr(self.redis, 'is_dummy', False):
                return False
            return bool(self.redis.ping())
        except Exception:
            return False

    def _append_to_wal(self, batch: List[Dict[str, Any]]):
        """Append tasks to local WAL on Redis failure."""
        if not self.wal_path or not batch:
            return
        with self._wal_lock:
            try:
                with open(self.wal_path, "a", encoding="utf-8") as f:
                    for item in batch:
                        f.write(json.dumps(item) + "\n")
                self.wal_entries_count += len(batch)
                self.logger.warning(
                    f"[WAL] Persisted {len(batch)} tasks to disk fallback: {self.wal_path}"
                )
            except Exception as e:
                self.logger.error(f"[WAL] Failed writing to WAL: {e}")

    def _replay_wal(self) -> int:
        """Replay tasks stored in local WAL to Redis."""
        if not self.wal_path or not os.path.exists(self.wal_path):
            return 0
        with self._wal_lock:
            try:
                with open(self.wal_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    return 0

                replayed_tasks = []
                for line in lines:
                    try:
                        replayed_tasks.append(json.loads(line))
                    except Exception:
                        pass

                if replayed_tasks:
                    self.redis.lpush("mission_log", json.dumps(replayed_tasks))
                    self.logger.info(
                        f"[WAL] Replayed {len(replayed_tasks)} WAL tasks to Redis"
                    )

                # Clear WAL after successful replay
                with open(self.wal_path, "w", encoding="utf-8") as f:
                    pass
                count = len(replayed_tasks)
                self.wal_entries_count = max(0, self.wal_entries_count - count)
                return count
            except Exception as e:
                self.logger.error(f"[WAL] Failed to replay WAL: {e}")
                return 0

    def sync(self):
        """
        Flush queued tasks.
        
        Attempts to write batch to Redis with exponential backoff.
        If Redis is unreachable or fails all retries, the batch is safely
        persisted to disk via local WAL so no missions are lost.
        """
        # First try replaying any pending WAL entries if Redis is healthy
        if self._is_redis_healthy():
            self._replay_wal()

        batch = self.pop_all()
        if not batch:
            return

        # Attempt Redis push with retries
        success = False
        if self._is_redis_healthy():
            for attempt in range(1, 4):
                try:
                    res = self.redis.lpush("mission_log", json.dumps(batch))
                    if res:
                        self.logger.info("Synced %d tasks to Redis", len(batch))
                        success = True
                        break
                except Exception as e:
                    self.logger.warning("Sync attempt %d failed: %s", attempt, e)
                    time.sleep(0.05 * (2 ** attempt))

        # Fallback to WAL if Redis unavailable or failed
        if not success:
            self._append_to_wal(batch)

    def _flusher(self):
        """Auto-sync every flush_interval seconds."""
        while not self._stop_evt.wait(self.flush_interval):
            self.sync()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._tasks),
                "max_size": self.max_queue_size,
                "overflow_count": self.overflow_count,
                "total_queued": self.total_queued,
                "overflow_policy": self.overflow_policy,
                "wal_entries": self.wal_entries_count
            }

    def shutdown(self):
        """Graceful stop."""
        self._stop_evt.set()
        self._thread.join()
        self.sync()  # final drain