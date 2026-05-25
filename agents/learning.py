import time, logging, json, uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from agents.base import BaseAgent

log = logging.getLogger("LEARN-MIND")


class LearningAgent(BaseAgent):
    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__("LEARN-MIND")
        cfg = cfg or {}
        self.window_minutes = int(cfg.get("window_minutes", 60))
        self.threshold = int(cfg.get("threshold", 5))
        self.interval = int(cfg.get("interval", 60))
        self.stats: Dict[str, int] = {}
        self.last_log = datetime.utcnow()
        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.codename), Decimal('80'))  # Initial budget
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant for LearningAgent: {e}")

    def _deduct_learning_cost(self):
        """Deduct Root for learning operation (0.3 Root per learning cycle)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.codename), Decimal('0.3'), "Learning cost")

    def _reward_learning_success(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.codename)].balance += Decimal('0.6')
            self.accountant.treasury_balance -= Decimal('0.6')
            log.info(f"Rewarded 0.6 Root to LearningAgent for successful learning.")

    def _burn_on_learning_failure(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.45')
            self.accountant.agent_wallets[str(self.codename)].balance -= penalty
            log.info(f"Burned 0.45 Root from LearningAgent for failed learning.")

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("Learning loop started (window %s min, threshold %s)", self.window_minutes, self.threshold)
        while True:
            self._deduct_learning_cost()
            try:
                logs = self._fetch_logs()
                if logs:
                    self._analyze(logs)
                    self._adapt()
                self._reward_learning_success()
            except Exception as e:
                self._burn_on_learning_failure()
                log.warning(f"Learning cycle failed: {e}")
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
            # Defensive: handle both ISO and float timestamps
            dt = None
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts)
                    else:
                        dt = datetime.utcfromtimestamp(float(ts))
                except Exception:
                    dt = None
            if dt and dt < cutoff:
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