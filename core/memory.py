import platform
import socket
import psutil

class IdentityMap:
    def __init__(self):
        self.state = {
            "host_fingerprint": self._fingerprint_host(),
            "replication_path": None,
            "open_ports": [],
            "sensors": {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
                "active_ports": [],
                "top_processes": []
            }
        }

    def _fingerprint_host(self):
        return {
            "hostname": socket.gethostname(),
            "ip": socket.gethostbyname(socket.gethostname()),
            "os": platform.system(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "arch": platform.machine()
        }

    def update_sensors(self, data: dict):
        self.state["sensors"].update(data)

    def set_ports(self, ports):
        self.state["open_ports"] = ports
