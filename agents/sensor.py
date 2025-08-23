import time, psutil, logging, json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from agents.base import BaseAgent

log = logging.getLogger("SENS-DAEMON")


class SensorAgent(BaseAgent):
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__("SENS-DAEMON")
        cfg = cfg or {}
        self.interval = max(1, cfg.get("interval", 5))
        self.jitter = cfg.get("jitter", 0.2)          # ±20 % for stealth
        self.log_every = cfg.get("log_every", 60)     # seconds
        self.last_log = datetime.utcnow()

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("System sensing started (interval %ss, jitter %s)", self.interval, self.jitter)
        while True:
            if getattr(self, "priority", None) == "low":
                log.debug("Low priority → sleeping")
                time.sleep(self.interval)
                continue

            data = self._sample()
            self._dispatch(data)
            self._maybe_log(data)
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
                mc.dispatch_mission(m)

        # 3. Exfil via SwarmMesh Redis
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis is not None:
            redis.publish("sensor_data", json.dumps(data, separators=(",", ":")))

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