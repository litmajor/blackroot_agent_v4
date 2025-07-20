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

# === Runtime Registration Hook ===
try:
    import blackroot_core
    blackroot_core.register_module("anomaly_classifier", lambda: AnomalyClassifierDaemon().start())
    print("[🔗] AnomalyClassifier auto-deployed into Blackroot runtime.")
except ImportError:
    print("[⚠️] Blackroot core runtime not found. Standalone mode engaged.")

# === Swarm Integration Stub ===
class SwarmAnomalySync:
    def __init__(self, classifier):
        self.classifier = classifier

    def broadcast_profile(self):
        payload = json.dumps(self.classifier.history).encode()
        print("[📡] Broadcasting anomaly profile to swarm...")
        try:
            from blackvault_storage_module import swarm, identity, capabilities
            swarm.broadcast_capabilities(identity.replica_id, capabilities + ["anomaly_classifier"])
        except:
            pass

    def sync_from_peer(self, profile_data):
        try:
            peer_data = json.loads(profile_data.decode())
            for k, v in peer_data.items():
                if k not in self.classifier.history:
                    self.classifier.history[k] = v
                else:
                    self.classifier.history[k].extend(v)
                    self.classifier.history[k] = self.classifier.history[k][-50:]
            print("[🔁] Anomaly profile synchronized from peer.")
        except Exception as e:
            print(f"[!] Sync error: {e}")

        try:
            from blackvault_storage_module import swarm
            if hasattr(swarm, 'evolution'):
                swarm.evolution.update_composition("anomaly_classifier", list(self.classifier.history.keys()))
        except:
            pass

class AnomalyClassifier:
    def __init__(self):
        self.behavior_profiles = {}
        self.outlier_threshold = 2.5
        self.history = {}

    def observe(self, category: str, value: float):
        if category not in self.history:
            self.history[category] = []
        self.history[category].append((datetime.utcnow(), value))
        if len(self.history[category]) > 50:
            self.history[category] = self.history[category][-50:]
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
            from blackvault_storage_module import swarm
            swarm.broadcast_capabilities("anomaly_response", [f"anomaly:{category}"])
        except:
            pass
        try:
            from ghost_layer_module import GhostLayer
            gl = GhostLayer()
            gl.mutate_shellcode()
        except:
            pass

    def export_model(self, path="anomaly_model.json"):
        with open(path, 'w') as f:
            json.dump(self.history, f)
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
            self.classifier.export_model()
            time.sleep(self.interval)

# Example Usage
if __name__ == "__main__":
    daemon = AnomalyClassifierDaemon()
    daemon.start()
    while True:
        time.sleep(300)
