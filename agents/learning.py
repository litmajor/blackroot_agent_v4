import time
from datetime import datetime
from agents.base import BaseAgent

class LearningAgent(BaseAgent):
    def __init__(self):
        super().__init__('LEARN-MIND')
        self.anomaly_stats = {}
        self.analysis_interval = 60  # seconds
        self.threshold = 5  # minimum frequency for response adaptation

    def run(self):
        super().run()
        print("[LEARN-MIND] Beginning cognitive reflection...")

        while True:
            if hasattr(self.kernel, 'storage'):
                logs = self.kernel.storage.query_last(n=100)
                self._analyze(logs)
                self._adjust_response_model()
            time.sleep(self.analysis_interval)

    def _analyze(self, logs):
        for entry in logs:
            if not entry:
                continue
            anomaly_class = entry.get("anomaly_class", {})
            anomaly_type = anomaly_class.get("type")
            if anomaly_type:
                self.anomaly_stats[anomaly_type] = self.anomaly_stats.get(anomaly_type, 0) + 1

    def _adjust_response_model(self):
        print("[LEARN-MIND] Adapting based on observed trends:")
        for atype, count in self.anomaly_stats.items():
            if count >= self.threshold:
                print(f"  → Anomaly '{atype}' occurred {count} times. Suggesting strategic boost.")

                # MirrorCore belief injection and memory
                mirrorcore = getattr(self.kernel, 'mirrorcore', None)
                if mirrorcore:
                    # Belief injection
                    mirrorcore.inject_beliefs({
                        f"learned_pattern::{atype}": f"{count}_hits"
                    })

                    # Emotional state shift
                    mirrorcore.inject_emotions(["vigilant"])

                    # Mission suggestion
                    mirrorcore.dispatch_mission({
                        "task": f"preempt::{atype}",
                        "reason": "learned_pattern",
                        "confidence": "learned"
                    })

                # Flag relevant agents for future priority upgrade
                if hasattr(self.kernel, 'agents'):
                    for agent in self.kernel.agents:
                        if hasattr(agent, 'codename') and 'DEFENDER' in agent.codename.upper():
                            print(f"  → Flagging {agent.codename} for priority escalation (future logic).")

        # Clear stats after learning cycle
        self.anomaly_stats.clear()
