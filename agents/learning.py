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
        kernel = getattr(self, "kernel", None)
        storage = getattr(kernel, "storage", None)
        if storage is None and kernel is not None:
            storage = getattr(getattr(kernel, "mirrorcore", None), "storage", None)
        if storage is None:
            try:
                from mirrorcore.storage import MirrorStorage
                storage = MirrorStorage()
            except Exception:
                storage = None
        if storage is None or not hasattr(storage, "query_last"):
            return []
        try:
            return storage.query_last(n=100)
        except Exception as e:
            log.warning("Storage query error: %s", e)
            return []

    # ------------------------------------------------------------------
    def _analyze(self, logs: List[Dict]) -> None:
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        self.stats.clear()
        for entry in logs:
            if not entry or not isinstance(entry, dict):
                continue
            ts = entry.get("timestamp")
            dt = None
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.utcfromtimestamp(float(ts))
                except Exception:
                    dt = None
            if dt and dt < cutoff:
                continue
            # Extract anomaly type from multiple known schemas
            atype = (
                entry.get("anomaly_class", {}).get("type")
                or entry.get("anomaly_type")
                or entry.get("event_type")
            )
            if atype:
                self.stats[atype] = self.stats.get(atype, 0) + 1

    # ------------------------------------------------------------------
    def _adapt(self) -> None:
        if not hasattr(self, "learned_patterns"):
            self.learned_patterns = {}

        for atype, count in list(self.stats.items()):
            if count < self.threshold:
                continue

            log.info("Learned pattern '%s' (%s hits >= threshold %s)", atype, count, self.threshold)
            self.learned_patterns[atype] = count
            self._inject_learnings(atype, count)

        self._maybe_log()

    # ------------------------------------------------------------------
    def _inject_learnings(self, atype: str, count: int):
        beliefs_payload = {
            f"learned_pattern::{atype}": f"{count}_hits",
            f"learned::{atype}": count
        }

        # 1. MirrorCore beliefs / emotions / missions
        mc = getattr(getattr(self, "kernel", None), "mirrorcore", None)
        if mc is not None:
            try:
                mc.inject_beliefs(beliefs_payload)
                if hasattr(mc, "inject_emotions") and callable(mc.inject_emotions):
                    mc.inject_emotions(["vigilant"])
                if hasattr(mc, "dispatch_mission") and callable(mc.dispatch_mission):
                    mc.dispatch_mission({"task": f"preempt_{atype}", "reason": "learned", "confidence": "high"})
            except Exception as e:
                log.warning(f"MirrorCore belief injection failed: {e}")

        # 2. BehaviorAdapter synchronization (escalate defender priority)
        behavior = getattr(getattr(self, "kernel", None), "behavior", None) or getattr(getattr(self, "kernel", None), "behaviour", None)
        if behavior is not None and hasattr(behavior, "adapt_based_on_beliefs"):
            try:
                behavior.adapt_based_on_beliefs(beliefs_payload)
            except Exception as e:
                log.warning(f"BehaviorAdapter adaptation failed: {e}")

        # 3. Direct agent priority escalation for defense/repair units
        agents = getattr(getattr(self, "kernel", None), "agents", [])
        for agent in agents:
            cname = str(getattr(agent, "codename", "")).upper()
            if any(k in cname for k in ["DEFENDER", "REPAIR", "CLOAK", "ENFORCER"]):
                agent.priority = "high"
                log.debug("Escalated %s → high priority due to %s", getattr(agent, "codename", ""), atype)

        # 4. SwarmMesh telemetry broadcast (fault-tolerant)
        try:
            redis = getattr(getattr(getattr(self, "kernel", None), "swarm", None), "redis", None)
            if redis and hasattr(redis, "publish"):
                redis.publish(
                    "learned_anomalies",
                    json.dumps({"atype": atype, "count": count, "ts": datetime.utcnow().isoformat()}, separators=(",", ":")),
                )
        except Exception as e:
            log.debug(f"Swarm telemetry publish skipped: {e}")

    # ------------------------------------------------------------------
    def _maybe_log(self):
        now = datetime.utcnow()
        if (now - self.last_log).total_seconds() >= 300:  # 5 min
            log.info("Current patterns: %s", self.stats)
            self.last_log = now