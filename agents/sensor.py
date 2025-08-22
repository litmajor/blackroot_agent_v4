import time
import psutil
from agents.base import BaseAgent
from datetime import datetime

class SensorAgent(BaseAgent):
    def __init__(self):
        super().__init__('SENS-DAEMON')

    def run(self):
        super().run()
        print("[SENS-DAEMON] Starting system sensing...")
        while True:
            if self.priority == 'low':
                print("[SENS-DAEMON] Priority low — sleeping...")
                time.sleep(3)
                continue

            sensors = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "active_ports": self._get_open_ports(),
                "top_processes": self._get_top_processes(5)
            }

            anomaly_score = self._calculate_anomaly_score(sensors)
            anomaly_class = self._classify_anomaly(sensors)
            sensors["anomaly_score"] = anomaly_score
            sensors["anomaly_class"] = anomaly_class

            if hasattr(self, 'kernel') and self.kernel is not None and hasattr(self.kernel, 'memory') and self.kernel.memory is not None:
                self.kernel.memory.update_sensors(sensors)

            # Feed sensor-based beliefs to MirrorCore (if available)
            mirrorcore = getattr(self.kernel, 'mirrorcore', None)
            if mirrorcore is not None:
                beliefs: dict[str, object] = {"anomaly_score": anomaly_score}
                emotions = []
                missions = []
                cpu = sensors["cpu_percent"]
                ram = sensors["memory_percent"]
                disk = sensors["disk_percent"]
                ports = sensors["active_ports"]

                if anomaly_class:
                    beliefs["anomaly_type"] = anomaly_class["type"]
                    beliefs["anomaly_severity"] = anomaly_class["severity"]
                    emotions.append("alarmed")
                    if anomaly_class["type"] == "suspicious_ports":
                        missions.append({"task": "deep_scan"})
                    elif anomaly_class["type"] == "resource_exhaustion":
                        missions.append({"task": "contain_intrusion"})

                if cpu > 85 or ram > 90:
                    beliefs["operational_load"] = "high"  # Accept string value
                    emotions.append("overwhelmed")
                    missions.append({"task": "monitor_changes"})

                if disk > 90:
                    beliefs["resource_alert"] = "disk"  # Accept string value
                    emotions.append("stressed")

                if any(p > 1024 for p in ports):
                    beliefs["suspicious_ports"] = True
                    if "alert" not in emotions:
                        emotions.append("alert")

                mirrorcore.inject_beliefs(beliefs)
                mirrorcore.inject_emotions(emotions)
                for mission in missions:
                    mirrorcore.dispatch_mission(mission)

                # Log emotions and mission triggers to storage
                if hasattr(self, 'kernel') and self.kernel is not None and hasattr(self.kernel, 'storage') and self.kernel.storage is not None:
                    self.kernel.storage.persist({
                        "timestamp": datetime.now().isoformat(),
                        "source": self.codename,
                        "anomaly_score": anomaly_score,
                        "anomaly_class": anomaly_class,
                        "emotions": emotions,
                        "beliefs": beliefs,
                        "missions": missions,
                        "sensors": sensors
                    })

            time.sleep(5)

    def _get_open_ports(self):
        connections = psutil.net_connections(kind='inet')
        ports = []
        for conn in connections:
            if conn.status == 'LISTEN' and hasattr(conn, 'laddr') and isinstance(conn.laddr, tuple) and len(conn.laddr) > 1:
                ports.append(conn.laddr[1])
        return sorted(set(ports))

    def _get_top_processes(self, count=5):
        procs = [(p.pid, p.name(), p.cpu_percent(interval=0.1)) for p in psutil.process_iter(['name'])]
        procs.sort(key=lambda x: x[2], reverse=True)
        return [{"pid": pid, "name": name, "cpu": cpu} for pid, name, cpu in procs[:count]]

    def _calculate_anomaly_score(self, sensors):
        score = 0
        if sensors["cpu_percent"] > 85:
            score += 2
        if sensors["memory_percent"] > 90:
            score += 2
        if sensors["disk_percent"] > 90:
            score += 2
        if any(p > 1024 for p in sensors["active_ports"]):
            score += 2
        if sensors["cpu_percent"] > 95 or sensors["memory_percent"] > 95:
            score += 2  # extreme load
        return min(score, 10)

    def _classify_anomaly(self, sensors):
        alerts = []
        cpu = sensors["cpu_percent"]
        mem = sensors["memory_percent"]
        disk = sensors["disk_percent"]
        ports = sensors["active_ports"]

        if cpu > 95 and mem > 90:
            alerts.append(("resource_exhaustion", "critical", "CPU and Memory critically high."))

        if disk > 95:
            alerts.append(("disk_full", "high", "Disk space critically low."))

        if any(p > 1024 for p in ports):
            alerts.append(("suspicious_ports", "medium", "High ports open. Possible intrusion vector."))

        if not alerts:
            return None

        alerts.sort(key=lambda x: {"low":0, "medium":1, "high":2, "critical":3}[x[1]], reverse=True)
        t, s, m = alerts[0]
        return {"type": t, "severity": s, "message": m}
