import time, logging, json, uuid, random
from datetime import datetime, timedelta
from agents.base import BaseAgent

log = logging.getLogger("REFLECT-DELTA")


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg: dict | None = None):
        super().__init__("REFLECT-DELTA")
        cfg = cfg or {}
        self.interval = max(1, cfg.get("interval", 60))
        self.jitter = cfg.get("jitter", 0.1)          # ±10 %
        self.log_every = cfg.get("log_every", 300)    # seconds
        self.last_log = datetime.utcnow()
        self.running = True

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("Reflection cycle started (interval %ss, jitter %s)", self.interval, self.jitter)
        while self.running:
            self._single_cycle()
            time.sleep(self._jittered_sleep())

    # ------------------------------------------------------------------
    def _single_cycle(self):
        """One atomic reflection loop."""
        mirrorcore = getattr(self.kernel, "mirrorcore", None)
        storage = getattr(self.kernel, "storage", None)
        memory = getattr(self.kernel, "memory", None)

        # Data gathering
        peer_scores = self._score_peers(memory)
        conflicts = self._detect_conflicts(mirrorcore)
        self._prune_outdated(mirrorcore)
        self._refine_missions(mirrorcore)

        # Build snapshot
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": self.codename,
            "peer_scores": peer_scores,
            "conflicts": conflicts,
            "beliefs": getattr(mirrorcore, "beliefs", {}) if mirrorcore else {},
            "missions": getattr(mirrorcore, "mission_queue", []) if mirrorcore else [],
        }

        # Persist
        if storage:
            storage.persist(snapshot)

        # SwarmMesh broadcast
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis:
            redis.publish("reflection", json.dumps(snapshot, separators=(",", ":")))

        self._maybe_log(snapshot)

    # ------------------------------------------------------------------
    def _score_peers(self, memory) -> dict[str, int]:
        if memory is None:
            return {}
        mem = getattr(memory, "memory", {})
        scores: dict[str, int] = {}
        for rec in mem.values():
            src = rec.get("source")
            if src:
                scores[src] = scores.get(src, 0) + 1
        return scores

    def _detect_conflicts(self, mirrorcore) -> list[str]:
        if mirrorcore is None:
            return []
        beliefs = getattr(mirrorcore, "beliefs", {})
        conflicts = []

        cpu = beliefs.get("sensor.cpu_percent", {}).get("value", 0)
        load = beliefs.get("operational_load")
        if load == "high" and cpu < 50:
            beliefs["operational_load"] = "normal"
            beliefs["last_adjusted_by"] = self.codename
            beliefs["timestamp"] = time.time()
            conflicts.append("CPU low but belief high-load")
        return conflicts

    def _prune_outdated(self, mirrorcore):
        if mirrorcore is None:
            return
        beliefs = getattr(mirrorcore, "beliefs", {})
        cutoff = time.time() - 3600
        stale = [k for k, v in beliefs.items() if isinstance(v, dict) and v.get("timestamp", 0) < cutoff]
        for k in stale:
            del beliefs[k]

    def _refine_missions(self, mirrorcore):
        if mirrorcore is None:
            return
        queue = getattr(mirrorcore, "mission_queue", [])
        cutoff = time.time() - 3600
        fresh = [m for m in queue if not m.get("completed") and m.get("timestamp", 0) > cutoff]
        mirrorcore.mission_queue = fresh

    # ------------------------------------------------------------------
    def _jittered_sleep(self) -> float:
        base = self.interval
        delta = base * self.jitter
        return base + random.uniform(-delta, delta)

    def _maybe_log(self, snapshot: dict):
        now = datetime.utcnow()
        if (now - self.last_log).total_seconds() >= self.log_every:
            log.info("Reflection snapshot: %s", snapshot)
            self.last_log = now