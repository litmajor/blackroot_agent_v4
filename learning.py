import time
from datetime import datetime
from threading import Lock
from agents.base import BaseAgent

class LearningAgent(BaseAgent):
    def __init__(self, analysis_interval=60, threshold=5, log_limit=100):
        super().__init__('LEARN-MIND')
        self.anomaly_stats = {}
        self.stats_lock = Lock()
        self.analysis_interval = analysis_interval
        self.threshold = threshold
        self.log_limit = log_limit
        self.stop_flag = False

    def run(self):
        super().run()
        print("[LEARN-MIND] Beginning cognitive reflection...")
        while not self.stop_flag:
            try:
                if hasattr(self.kernel, 'storage'):
                    logs = self.kernel.storage.query_last(n=self.log_limit)
                    self._analyze(logs)
                    self._adjust_response_model()
                time.sleep(self.analysis_interval)
            except Exception as e:
                print(f"[LEARN-MIND] Error in run loop: {e}")
                time.sleep(self.analysis_interval)

    # In LearningAgent._analyze
    def _analyze(self, logs):
        with self.stats_lock:
            for entry in logs:
                if not entry:
                    continue
                anomaly_class = entry.get("anomaly_class", {})
                anomaly_type = anomaly_class.get("type")
                if anomaly_type:
                    self.anomaly_stats[anomaly_type] = self.anomaly_stats.get(anomaly_type, 0) + 1
            try:
                for artifact in self.kernel.black_vault.list_artifacts():
                    if any(x in artifact for x in ["recon_", "injection_", "form_", "xss_"]):
                        data = self.kernel.black_vault.retrieve(artifact)
                        recon = json.loads(data.decode())
                        if recon.get('secrets'):
                           self.anomaly_stats['secrets_found'] += len(recon['secrets'])
                        if recon.get('xss_results'):
                            self.anomaly_stats['xss_vulnerabilities'] += sum(1 for r in recon['xss_results'].values() if r)
                        self.logger.info(f"Analyzed artifact: {artifact}")
            except Exception as e:
                self.logger.error(f"Failed to analyze artifacts: {e}")

    def _adjust_response_model(self):
        print("[LEARN-MIND] Adapting based on observed trends:")
        with self.stats_lock:
            with open("anomaly_log.txt", "a") as f:
                f.write(f"{datetime.now()}: Analyzed {len(self.anomaly_stats)} anomaly types\n")
            for atype, count in self.anomaly_stats.items():
                if count >= self.threshold:
                    print(f"  → Anomaly '{atype}' occurred {count} times. Suggesting strategic boost.")
                    mirrorcore = getattr(self.kernel, 'mirrorcore', None)
                    if mirrorcore:
                        try:
                            mirrorcore.inject vehicular_beliefs({
                                f"learned_pattern::{atype}": f"{count}_hits"
                            })
                        except Exception as e:
                            print(f"[LEARN-MIND] Failed to inject beliefs: {e}")
            self.anomaly_stats.clear()

    def stop(self):
        self.stop_flag = True