import time, logging, json, uuid, random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional
from agents.base import BaseAgent
# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event

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
                m = buffer.tasks.pop(0)
                # Accept both dict and Mission
                mission = m if isinstance(m, Mission) else Mission(
                    name=m.get("task", "unknown"),
                    objectives=m.get("objectives", []),
                    parameters=m,
                    status=m.get("status", "pending")
                )
                self.executor.submit(self._execute_with_retry, mission)

            time.sleep(0.2)  # small sleep to reduce CPU spin

    # ------------------------------------------------------------------
    def _execute_with_retry(self, mission: Mission):
        log.debug("Executing mission: %s", mission.name)
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        # Emit Event for mission start
        event_start = Event(event_type="mission_start", payload={"mission_id": mission.mission_id, "name": mission.name, "ts": datetime.utcnow().isoformat()})
        if redis:
            redis.publish("mission_start", json.dumps(event_start.payload))
        for attempt in range(1, self.retry_limit + 1):
            try:
                self._dispatch(mission.name, mission)
                # Emit Event for mission done
                event_done = Event(event_type="mission_done", payload={"mission_id": mission.mission_id, "name": mission.name, "status": "done"})
                if redis:
                    redis.publish("mission_done", json.dumps(event_done.payload))
                return
            except Exception as exc:
                log.warning("Mission %s failed (attempt %s/%s): %s", mission.name, attempt, self.retry_limit, exc)
                if attempt < self.retry_limit:
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)) * random.uniform(0.5, 1.5))
                else:
                    log.error("Mission %s permanently failed.", mission.name)
                    event_err = Event(event_type="mission_error", payload={"mission_id": mission.mission_id, "name": mission.name, "error": str(exc)})
                    if redis:
                        redis.publish("mission_error", json.dumps(event_err.payload))

    # ------------------------------------------------------------------
    def _dispatch(self, task: str, mission: Mission):
        handler = {
            "deep_scan": self._deep_scan,
            "monitor_changes": self._monitor_changes,
            "apply_patch": self._apply_patch,
        }.get(task, self._unknown_task)
        handler(mission)

    # ------------------------------------------------------------------
    def _deep_scan(self, mission: Mission):
        log.debug("Deep scan stub executed for mission %s", mission.mission_id)
        time.sleep(2)

    def _monitor_changes(self, mission: Mission):
        log.debug("Monitor stub executed for mission %s", mission.mission_id)
        time.sleep(2)

    def _apply_patch(self, mission: Mission):
        log.debug("Patch stub executed for mission %s", mission.mission_id)
        time.sleep(2)

    def _unknown_task(self, mission: Mission):
        log.error("Unknown task: %s", mission)

    # ------------------------------------------------------------------
    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        self.executor.shutdown(wait=True)
        log.info("Mission agent shut down.")