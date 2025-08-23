import time, logging, json, uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from agents.base import BaseAgent

log = logging.getLogger("LEARN-MIND")


class LearningAgent(BaseAgent):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__("LEARN-MIND")
        cfg = cfg or {}
        self.window_minutes = cfg.get("window_minutes", 60)
        self.threshold = cfg.get("threshold", 5)
        self.interval = cfg.get("interval", 60)
        self.stats: Dict[str, int] = {}
        self.last_log = datetime.utcnow()

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("Learning loop started (window %s min, threshold %s)", self.window_minutes, self.threshold)
        while True:
            logs = self._fetch_logs()
            if logs:
                self._analyze(logs)
                self._adapt()
            time.sleep(self.interval)

    # ------------------------------------------------------------------
    def _fetch_logs(self) -> List[Dict]:
        storage = getattr(getattr(self, "kernel", None), "storage", None)
        if storage is None:
            return []
        try:
            return storage.query_last(n=100)  # adjust as needed
        except Exception as e:
            log.warning("Storage query error: %s", e)
            return []

    # ------------------------------------------------------------------
    def _analyze(self, logs: List[Dict]) -> None:
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        self.stats.clear()
        for entry in logs:
            ts = entry.get("timestamp")
            if ts and datetime.fromisoformat(ts) < cutoff:
                continue
            atype = entry.get("anomaly_class", {}).get("type")
            if atype:
                self.stats[atype] = self.stats.get(atype, 0) + 1

    # ------------------------------------------------------------------
    def _adapt(self) -> None:
        for atype, count in self.stats.items():
            if count < self.threshold:
                continue

            log.debug("Learned pattern '%s' (%s hits)", atype, count)
            self._inject_learnings(atype, count)

        self._maybe_log()

    # ------------------------------------------------------------------
    def _inject_learnings(self, atype: str, count: int):
        # 1. MirrorCore beliefs / emotions / missions
        mc = getattr(self.kernel, "mirrorcore", None)
        if mc is not None:
            mc.inject_beliefs({f"learned::{atype}": count})
            mc.inject_emotions(["vigilant"])
            mc.dispatch_mission({"task": f"preempt_{atype}", "reason": "learned", "confidence": "high"})

        # 2. SwarmMesh telemetry
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis:
            redis.publish(
                "learned_anomalies",
                json.dumps({"atype": atype, "count": count, "ts": datetime.utcnow().isoformat()}, separators=(",", ":")),
            )

        # 3. Flag defender agents (placeholder)
        agents = getattr(self.kernel, "agents", [])
        for agent in agents:
            if "defender" in str(getattr(agent, "codename", "")).lower():
                log.debug("Flagged %s for escalation", agent.codename)

    # ------------------------------------------------------------------
    def _maybe_log(self):
        now = datetime.utcnow()
        if (now - self.last_log).total_seconds() >= 300:  # 5 min
            log.info("Current patterns: %s", self.stats)
            self.last_log = now