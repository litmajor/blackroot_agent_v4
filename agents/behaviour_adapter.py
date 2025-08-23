import logging, json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

log = logging.getLogger("BEHAVIOR-ADAPTER")


class BehaviorAdapter:
    def __init__(self, kernel, cfg: Optional[Dict[str, Any]] = None):
        self.kernel = kernel
        cfg = cfg or {}
        self.threshold = cfg.get("escalation_threshold", 5)
        self.log_every = cfg.get("log_every", 300)  # seconds
        self.last_log = datetime.utcnow()
        self.adaptation_log = []

    # ------------------------------------------------------------------
    def adapt_based_on_beliefs(self, beliefs: Dict[str, Any]) -> None:
        for key, value in beliefs.items():
            if not key.startswith("learned_pattern::"):
                continue
            atype = key.split("::")[1]
            frequency = int(value) if str(value).isdigit() else 0
            if frequency < self.threshold:
                continue

            self._escalate(atype, frequency)
            self._log_and_broadcast(f"Escalated agents due to {atype} ({frequency}x)")

    # ------------------------------------------------------------------
    def _escalate(self, anomaly_type: str, frequency: int) -> None:
        agents = getattr(self.kernel, "agents", [])
        for agent in agents:
            codename = getattr(agent, "codename", "").upper()
            if "DEFENDER" in codename:
                agent.priority = "high"
                log.debug("Escalated %s → high priority", agent.codename)

    # ------------------------------------------------------------------
    def _log_and_broadcast(self, message: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "source": "BEHAVIOR-ADAPTER"
        }
        self.adaptation_log.append(entry)

        # 1. Persist via kernel.storage (if available)
        storage = getattr(self.kernel, "storage", None)
        if storage:
            storage.persist(entry)

        # 2. SwarmMesh broadcast
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis:
            redis.publish("behavior_events", json.dumps(entry, separators=(",", ":")))

        # 3. Rate-limited console log
        now = datetime.utcnow()
        if (now - self.last_log).total_seconds() >= self.log_every:
            log.info(message)
            self.last_log = now