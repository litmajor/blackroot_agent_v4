import time, logging, json, uuid, random
from datetime import datetime, timedelta
from agents.base import BaseAgent
# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event

log = logging.getLogger("REFLECT-DELTA")


class ReflectionAgent(BaseAgent):
    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])
    def __init__(self, cfg: dict | None = None):
        super().__init__("REFLECT-DELTA")
        cfg = cfg or {}
        self.agent_id = AgentID("REFLECT-DELTA")
        self.agent_status = AgentStatus.ACTIVE
        self.interval = max(1, cfg.get("interval", 60))
        self.jitter = cfg.get("jitter", 0.1)          # ±10 %
        self.log_every = cfg.get("log_every", 300)    # seconds
        self.last_log = datetime.utcnow()
        self.running = True
        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('85'))  # Initial budget
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant for ReflectionAgent: {e}")

    def _deduct_reflection_cost(self):
        """Deduct Root for reflection operation (0.3 Root per cycle)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.agent_id), Decimal('0.3'), "Reflection cost")

    def _reward_reflection_success(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.agent_id)].balance += Decimal('0.6')
            self.accountant.treasury_balance -= Decimal('0.6')
            log.info(f"Rewarded 0.6 Root to ReflectionAgent for successful reflection.")

    def _burn_on_reflection_failure(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.45')
            self.accountant.agent_wallets[str(self.agent_id)].balance -= penalty
            log.info(f"Burned 0.45 Root from ReflectionAgent for failed reflection.")

    def get_status(self):
        return {
            "agent_id": str(self.agent_id),
            "status": self.agent_status.value,
            "interval": self.interval,
            "jitter": self.jitter,
            "log_every": self.log_every,
            "running": self.running
        }

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("Reflection cycle started (interval %ss, jitter %s)", self.interval, self.jitter)
        while self.running:
            self._deduct_reflection_cost()
            try:
                self._single_cycle()
                self._reward_reflection_success()
            except Exception as e:
                self._burn_on_reflection_failure()
                log.warning(f"Reflection cycle failed: {e}")
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
        mission_queue = getattr(mirrorcore, "mission_queue", []) if mirrorcore else []
        # Convert to Mission objects if not already
        mission_objs = [m if isinstance(m, Mission) else Mission(
            name=m.get("task", m.get("name", "unknown")),
            objectives=m.get("objectives", []),
            parameters=m,
            status=m.get("status", "pending")
        ) for m in mission_queue]
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": self.codename,
            "peer_scores": peer_scores,
            "conflicts": conflicts,
            "beliefs": getattr(mirrorcore, "beliefs", {}) if mirrorcore else {},
            "missions": [m.__dict__ for m in mission_objs],
        }

        # Emit Event for snapshot
        event = Event(event_type="reflection_snapshot", payload=snapshot)

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
        # Convert to Mission objects if not already
        mission_objs = [m if isinstance(m, Mission) else Mission(
            name=m.get("task", m.get("name", "unknown")),
            objectives=m.get("objectives", []),
            parameters=m,
            status=m.get("status", "pending")
        ) for m in queue]
        fresh = [m for m in mission_objs if not getattr(m, "completed", False) and getattr(m, "timestamp", time.time()) > cutoff]
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