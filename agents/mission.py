import time, logging, json, uuid, random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional
from agents.base import BaseAgent

log = logging.getLogger("MIS-EXECUTOR")


class MissionAgent(BaseAgent):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__("MIS-EXECUTOR")
        cfg = cfg or {}
        self.workers = max(1, cfg.get("workers", 2))
        self.retry_limit = cfg.get("retry_limit", 3)
        self.backoff_base = cfg.get("backoff_base", 1.0)  # seconds
        self.log_every = cfg.get("log_every", 60)        # seconds
        self.last_log = datetime.utcnow()
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.workers)

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("Mission processor started (workers=%s)", self.workers)
        while self.running:
            if getattr(self, "priority", None) == "low":
                log.debug("Low priority — sleeping")
                time.sleep(3)
                continue

            buffer = getattr(getattr(self, "kernel", None), "mission_buffer", None)
            if buffer is None or not hasattr(buffer, "tasks"):
                time.sleep(1)
                continue

            # Pop missions and submit to thread-pool
            while buffer.tasks:
                mission = buffer.tasks.pop(0)
                self.executor.submit(self._execute_with_retry, mission)

            time.sleep(0.2)  # small sleep to reduce CPU spin

    # ------------------------------------------------------------------
    def _execute_with_retry(self, mission: Dict[str, Any]):
        task = mission.get("task", "unknown")
        log.debug("Executing task: %s", task)

        # SwarmMesh start telemetry
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis:
            redis.publish(
                "mission_start",
                json.dumps({"id": str(uuid.uuid4()), "task": task, "ts": datetime.utcnow().isoformat()}, separators=(",", ":")),
            )

        for attempt in range(1, self.retry_limit + 1):
            try:
                self._dispatch(task, mission)
                # Swarm success telemetry
                if redis:
                    redis.publish("mission_done", json.dumps({"task": task, "status": "done"}))
                return  # success
            except Exception as exc:
                log.warning("Task %s failed (attempt %s/%s): %s", task, attempt, self.retry_limit, exc)
                if attempt < self.retry_limit:
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)) * random.uniform(0.5, 1.5))
                else:
                    log.error("Task %s permanently failed.", task)
                    if redis:
                        redis.publish("mission_error", json.dumps({"task": task, "error": str(exc)}))

    # ------------------------------------------------------------------
    def _dispatch(self, task: str, mission: Dict[str, Any]):
        handler = {
            "deep_scan": self._deep_scan,
            "monitor_changes": self._monitor_changes,
            "apply_patch": self._apply_patch,
        }.get(task, self._unknown_task)
        handler(mission)

    # ------------------------------------------------------------------
    def _deep_scan(self, _):
        log.debug("Deep scan stub executed")
        time.sleep(2)

    def _monitor_changes(self, _):
        log.debug("Monitor stub executed")
        time.sleep(2)

    def _apply_patch(self, _):
        log.debug("Patch stub executed")
        time.sleep(2)

    def _unknown_task(self, mission):
        log.error("Unknown task: %s", mission)

    # ------------------------------------------------------------------
    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        self.executor.shutdown(wait=True)
        log.info("Mission agent shut down.")