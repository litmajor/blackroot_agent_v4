
import os
import json
import random
import hashlib
import statistics
import psutil
import shutil
import time
from datetime import datetime
import threading

MAX_CATEGORIES = 20

# ---------- JSON encoder ----------
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

# === Runtime Registration Hook (Deferred) ===
# Avoid circular imports - register_module will be called from kernel after initialization
def register_with_kernel():
    """Register this module with kernel when called (avoids circular imports)"""
    try:
        from kernel.core import register_module
        register_module("anomaly_classifier", lambda: AnomalyClassifierDaemon())
    except ImportError:
        pass  # Standalone mode

# === Swarm Integration Stub ===
class SwarmAnomalySync:
    def __init__(self, classifier):
        self.classifier = classifier

    def broadcast_profile(self):
        payload = json.dumps(self.classifier.history, cls=DateTimeEncoder).encode()
        print("[📡] Broadcasting anomaly profile to swarm...")
        try:
            # Try to import a global SwarmMesh instance
            from swarm_mesh import SwarmMesh
            import builtins
            swarm = getattr(builtins, 'swarm', None)
            if swarm is not None and isinstance(swarm, SwarmMesh):
                node_id = getattr(swarm, 'node_id', 'anomaly_node')
                capabilities = getattr(swarm, '_get_node_capabilities', lambda: [])()
                swarm.broadcast_capabilities(node_id, list(capabilities) + ["anomaly_classifier"])
            else:
                print("[⚠️] No active SwarmMesh instance found for broadcast.")
        except Exception as e:
            print(f"[⚠️] Swarm broadcast failed: {e}")

    def sync_from_peer(self, profile_data):
        try:
            peer_data = json.loads(profile_data.decode())
            for k, v in peer_data.items():
                if k not in self.classifier.history:
                    self.classifier.history[k] = v
                else:
                    combined = (self.classifier.history[k] + v)[-50:]
                    self.classifier.history[k] = combined
            print("[🔁] Anomaly profile synchronized from peer.")
        except Exception as e:
            print(f"[!] Sync error: {e}")

        try:
            from black_vault import swarm  # type: ignore
            if hasattr(swarm, 'evolution'):
                swarm.evolution.update_composition("anomaly_classifier", list(self.classifier.history.keys()))
        except Exception:
            pass

class AnomalyClassifier:
    def __init__(self):
        self.behavior_profiles = {}
        self.outlier_threshold = 2.5
        self.history = {}

    def observe(self, category: str, value: float):
        if category not in self.history:
            self.history[category] = []
        self.history[category].append((time.time(), value))
        if len(self.history[category]) > 50:
            self.history[category] = self.history[category][-50:]

        if len(self.history) > MAX_CATEGORIES:
            oldest = list(self.history.keys())[0]
            del self.history[oldest]

        self._evaluate(category)

    def _evaluate(self, category: str):
        values = [v for _, v in self.history[category]]
        if len(values) < 10:
            return
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        latest = values[-1]
        z_score = abs(latest - mean) / stdev if stdev > 0 else 0
        if z_score > self.outlier_threshold:
            self._flag_anomaly(category, latest, z_score)

    def _flag_anomaly(self, category: str, value: float, z_score: float):
        print(f"[🧠] Anomaly detected in '{category}': value={value:.2f}, z-score={z_score:.2f}")
        self.adapt_strategy(category, value, z_score)

    def adapt_strategy(self, category: str, value: float, z_score: float):
        print(f"[🔄] Adapting tactics for '{category}' due to anomaly...")
        try:
            from swarm_mesh import SwarmMesh
            import builtins
            swarm = getattr(builtins, 'swarm', None)
            if swarm is not None and isinstance(swarm, SwarmMesh):
                swarm.broadcast_capabilities("anomaly_response", [f"anomaly:{category}"])
            else:
                print("[⚠️] No active SwarmMesh instance found for anomaly response.")
        except Exception as e:
            print(f"[⚠️] Swarm anomaly response failed: {e}")
        try:
            from ghost_layer import GhostLayer
            gl = GhostLayer(
                vault="anomaly_classifier"
            )
            gl.mutate_shellcode()
        except Exception as e:
            print(f"[⚠️] GhostLayer mutation failed: {e}")

    def debug_memory(self):
        import sys
        total = sum(sys.getsizeof(v) for v in self.history.values())
        print(f"[🧠] History size: {len(self.history)} categories, approx {total} bytes")

    def export_model(self, path="anomaly_model.json"):
        with open(path, 'w') as f:
            json.dump(self.history, f, cls=DateTimeEncoder)
        print(f"[💾] Behavior model exported to {path}")

    def import_model(self, path="anomaly_model.json"):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.history = json.load(f)
            print(f"[📥] Behavior model imported from {path}")

class AnomalyClassifierDaemon:
    def __init__(self, interval=60):
        self.classifier = AnomalyClassifier()
        self.sync = SwarmAnomalySync(self.classifier)
        self.interval = interval

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def run(self):
        while True:
            self.classifier.observe("cpu_usage", psutil.cpu_percent())
            self.classifier.observe("memory_percent", psutil.virtual_memory().percent)
            net = psutil.net_io_counters()
            self.classifier.observe("bytes_sent", net.bytes_sent / 1024 / 1024)  # MB
            self.classifier.observe("bytes_recv", net.bytes_recv / 1024 / 1024)

            if random.random() < 0.1:
                self.classifier.observe("cpu_usage", 90.0)

            self.sync.broadcast_profile()
            if random.random() < 0.05:
                self.classifier.export_model()
            self.classifier.debug_memory()
            time.sleep(self.interval)

# Example Usage
if __name__ == "__main__":
    daemon = AnomalyClassifierDaemon()
    daemon.start()
    while True:
        time.sleep(300)
