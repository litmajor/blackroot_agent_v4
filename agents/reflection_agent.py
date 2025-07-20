import time
from agents.base import BaseAgent
from datetime import datetime

class ReflectionAgent(BaseAgent):
    def __init__(self):
        super().__init__('REFLECT-DELTA')
        self.interval = 60  # seconds

    def run(self):
        super().run()
        print("[REFLECT-DELTA] Reflection cycle started.")

        while self.running:
            self.peer_scores = self.score_peers()
            self.conflict_flags = self.detect_conflicts()
            self.audit_beliefs()
            self.refine_missions()
            self.tune_behavior()

            if hasattr(self.kernel, 'storage'):
                self.kernel.storage.persist({
                    "timestamp": datetime.now().isoformat(),
                    "source": self.codename,
                    "cycle": "reflection",
                    "peer_scores": self.peer_scores,
                    "beliefs": getattr(self.kernel.mirrorcore, "beliefs", {}),
                    "missions": getattr(self.kernel.mirrorcore, "mission_queue", []),
                    "conflicts_detected": self.conflict_flags,
                })

            time.sleep(self.interval)

    def audit_beliefs(self):
        if not hasattr(self.kernel, 'mirrorcore'):
            return

        beliefs = self.kernel.mirrorcore.beliefs
        print("[REFLECT-DELTA] Auditing beliefs:", beliefs)

        outdated = [k for k, v in beliefs.items() if isinstance(v, dict) and v.get('timestamp', 0) < time.time() - 3600]
        for key in outdated:
            print(f"[REFLECT-DELTA] Belief '{key}' is outdated — considering removal.")
            # Optionally prune or revise

    def refine_missions(self):
        if not hasattr(self.kernel, 'mirrorcore'):
            return

        missions = self.kernel.mirrorcore.mission_queue
        active = [m for m in missions if not m.get('completed')]
        print(f"[REFLECT-DELTA] {len(active)} active missions found.")

        now = time.time()
        new_queue = []
        for m in active:
            if 'timestamp' in m and now - m['timestamp'] > 3600:
                print(f"[REFLECT-DELTA] Mission stale: {m}")
                continue
            new_queue.append(m)

        self.kernel.mirrorcore.mission_queue = new_queue

    def score_peers(self):
        if not hasattr(self.kernel, 'memory'):
            return {}

        print("[REFLECT-DELTA] Scoring peers based on memory contributions...")
        peer_scores = {}
        for k, record in self.kernel.memory.memory.items():
            src = record.get("source")
            if not src:
                continue
            peer_scores[src] = peer_scores.get(src, 0) + 1

        for peer, score in peer_scores.items():
            print(f"[REFLECT-DELTA] Peer '{peer}' score: {score}")
        return peer_scores

    def detect_conflicts(self):
        beliefs = getattr(self.kernel, 'mirrorcore', {}).beliefs
        memory = getattr(self.kernel, 'memory', {}).memory
        flags = []

        if 'operational_load' in beliefs and 'sensor.cpu_percent' in memory:
            cpu = memory['sensor.cpu_percent']['value']
            load = beliefs['operational_load']
            if load == 'high' and cpu < 50:
                msg = "Conflict: CPU low but belief is high load."
                print(f"[REFLECT-DELTA] {msg}")
                beliefs['operational_load'] = 'normal'
                beliefs['last_adjusted_by'] = self.codename
                beliefs['timestamp'] = time.time()
                flags.append(msg)

        return flags

    def tune_behavior(self):
        print("[REFLECT-DELTA] Behavioral tuning (placeholder)...")
        # Future: self-kernel tuning logic
