import time, psutil, logging, json, sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
try:
    from .base import BaseAgent
except ImportError:
    # Add parent directory to path for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.base import BaseAgent
# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event

log = logging.getLogger("SENS-DAEMON")


class SensorAgent(BaseAgent):
    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])
    def __init__(self, cfg: Optional[dict] = None, agent_id: AgentID = AgentID(value="SENS-DAEMON")):
        self.agent_id: AgentID = agent_id
        super().__init__(str(self.agent_id))
        cfg = cfg or {}
        self.interval = max(1, cfg.get("interval", 5))
        self.jitter = cfg.get("jitter", 0.2)          # ±20 % for stealth
        self.log_every = cfg.get("log_every", 60)     # seconds
        self.last_log = datetime.utcnow()
        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('75'))  # Initial budget
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant for SensorAgent: {e}")

    def _deduct_sensor_cost(self):
        """Deduct Root for sensing operation (0.2 Root per sample)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.agent_id), Decimal('0.2'), "Sensor sample cost")

    def _reward_sensor_success(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.agent_id)].balance += Decimal('0.4')
            self.accountant.treasury_balance -= Decimal('0.4')
            log.info(f"Rewarded 0.4 Root to SensorAgent for successful sensing.")

    def _burn_on_sensor_failure(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.3')
            self.accountant.agent_wallets[str(self.agent_id)].balance -= penalty
            log.info(f"Burned 0.3 Root from SensorAgent for failed sensing.")

    # ------------------------------------------------------------------
    def run(self, mission: Mission = Mission(name="default", objectives=[], parameters={})):
        super().run()
        log.info("System sensing started (interval %ss, jitter %s)", self.interval, self.jitter)
        while True:
            self._deduct_sensor_cost()
            try:
                if getattr(self, "priority", None) == "low":
                    log.debug("Low priority → sleeping")
                    time.sleep(self.interval)
                    continue
                data = self._sample()
                self._dispatch(data)
                self._emit_event(Event(event_type="sensor_sample", payload=data, sender_id=str(self.agent_id)))
                self._maybe_log(data)
                self._reward_sensor_success()
            except Exception as e:
                self._burn_on_sensor_failure()
                log.warning(f"Sensor cycle failed: {e}")
            time.sleep(self._jittered_sleep())

    # ------------------------------------------------------------------
    def _sample(self) -> Dict:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "active_ports": sorted({c.laddr[1] for c in psutil.net_connections(kind='inet')
                                  if c.status == 'LISTEN' and len(c.laddr) > 1}),
            "top_processes": self._top_processes(5),
            "timestamp": datetime.utcnow().isoformat()
        }

    def _top_processes(self, n: int) -> List[dict]:
        procs = [(p.pid, p.name(), p.cpu_percent()) for p in psutil.process_iter(['name'])]
        procs.sort(key=lambda x: x[2], reverse=True)
        return [{"pid": pid, "name": name, "cpu": cpu} for pid, name, cpu in procs[:n]]

    # ------------------------------------------------------------------
    def _dispatch(self, data: Dict):
        anomaly = self._classify(data)
        data["anomaly"] = anomaly

        # 1. Store in kernel memory if available
        kernel = getattr(self, "kernel", None)
        memory = getattr(kernel, "memory", None) if kernel is not None else None
        if memory is not None and hasattr(memory, "update_sensors"):
            memory.update_sensors(data)

        # 2. Feed MirrorCore beliefs / emotions / missions
        mc = getattr(self.kernel, "mirrorcore", None)
        if mc is not None:
            beliefs, emotions, missions = self._build_beliefs(anomaly, data)
            mc.inject_beliefs(beliefs)
            mc.inject_emotions(emotions)
            for m in missions:
                # Convert to Mission if not already
                mission = m if isinstance(m, Mission) else Mission(
                    name=m.get("task", m.get("name", "unknown")),
                    objectives=m.get("objectives", []),
                    parameters=m,
                    status=m.get("status", "pending")
                )
                mc.dispatch_mission(mission)

        # 3. Exfil via SwarmMesh Redis
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis is not None:
            redis.publish("sensor_data", json.dumps(data, separators=(",", ":")))

    def _emit_event(self, event: Event):
        # Stub for event emission (to be integrated with event bus or kernel)
        log.info(f"Event emitted: {event.event_type} for agent {getattr(event, 'sender_id', None)}")

    # ------------------------------------------------------------------
    def _classify(self, d: Dict) -> Optional[dict]:
        alerts = []
        if d["cpu_percent"] > 95 and d["memory_percent"] > 90:
            alerts.append(("resource_exhaustion", "critical"))
        if d["disk_percent"] > 95:
            alerts.append(("disk_full", "high"))
        if any(p > 1024 for p in d["active_ports"]):
            alerts.append(("suspicious_ports", "medium"))

        if not alerts:
            return None
        alerts.sort(key=lambda x: {"low": 0, "medium": 1, "high": 2, "critical": 3}[x[1]])
        return {"type": alerts[0][0], "severity": alerts[0][1]}

    def _build_beliefs(self, anomaly: Optional[dict], d: Dict):
        beliefs, emotions, missions = {}, [], []

        if anomaly:
            beliefs["anomaly_type"] = anomaly["type"]
            beliefs["anomaly_severity"] = anomaly["severity"]
            emotions.append("alarmed")
            if anomaly["type"] == "suspicious_ports":
                missions.append({"task": "deep_scan"})
            elif anomaly["type"] == "resource_exhaustion":
                missions.append({"task": "contain_intrusion"})

        if d["cpu_percent"] > 85 or d["memory_percent"] > 90:
            beliefs["operational_load"] = "high"
            emotions.append("overwhelmed")
            missions.append({"task": "monitor_changes"})

        if d["disk_percent"] > 90:
            beliefs["resource_alert"] = "disk"
            emotions.append("stressed")

        return beliefs, emotions, missions

    # ------------------------------------------------------------------
    def _jittered_sleep(self) -> float:
        base = self.interval
        delta = base * self.jitter
        return base + (time.time() % (2 * delta)) - delta

    def _maybe_log(self, data: Dict):
        now = datetime.utcnow()
        if (now - self.last_log).total_seconds() >= self.log_every:
            log.info("Sensor sample: %s", data)
            self.last_log = now
            
if __name__ == "__main__":
    SensorAgent().run()