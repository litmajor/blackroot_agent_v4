import time
from agents.base import BaseAgent
from datetime import datetime

class ReflectionAgent(BaseAgent):
    def __init__(self):
        super().__init__('REFLECT-DELTA')
        self.interval = 60  # seconds
        self.running = True  # Initialize running attribute

    def run(self):
        super().run()
        print("[REFLECT-DELTA] Reflection cycle started.")

        while self.running:
            self.peer_scores = self.score_peers()
            self.conflict_flags = self.detect_conflicts()
            self.audit_beliefs()
            self.refine_missions()
            self.tune_behavior()

            if self.kernel is not None and hasattr(self.kernel, 'storage') and self.kernel.storage is not None:
                mirrorcore = getattr(self.kernel, 'mirrorcore', None)
                beliefs = getattr(mirrorcore, "beliefs", {}) if mirrorcore is not None else {}
                missions = getattr(mirrorcore, "mission_queue", []) if mirrorcore is not None else []
                self.kernel.storage.persist({
                    "timestamp": datetime.now().isoformat(),
                    "source": self.codename,
                    "cycle": "reflection",
                    "peer_scores": self.peer_scores,
                    "beliefs": beliefs,
                    "missions": missions,
                    "conflicts_detected": self.conflict_flags,
                })

            time.sleep(self.interval)

    def audit_beliefs(self):
        if not self.kernel or not hasattr(self.kernel, 'mirrorcore') or self.kernel.mirrorcore is None:
            print("[REFLECT-DELTA] No mirrorcore available for auditing beliefs.")
            return

        beliefs = getattr(self.kernel.mirrorcore, 'beliefs', {})
        print("[REFLECT-DELTA] Auditing beliefs:", beliefs)

        outdated = [k for k, v in beliefs.items() if isinstance(v, dict) and v.get('timestamp', 0) < time.time() - 3600]
        for key in outdated:
            print(f"[REFLECT-DELTA] Belief '{key}' is outdated — considering removal.")
            # Optionally prune or revise

    def refine_missions(self):
        if not self.kernel or not hasattr(self.kernel, 'mirrorcore') or self.kernel.mirrorcore is None:
            return

        missions = getattr(self.kernel.mirrorcore, 'mission_queue', [])
        active = [m for m in missions if not m.get('completed')]
        print(f"[REFLECT-DELTA] {len(active)} active missions found.")

        now = time.time()
        new_queue = []
        for m in active:
            if 'timestamp' in m and now - m['timestamp'] > 3600:
                print(f"[REFLECT-DELTA] Mission stale: {m}")
                continue
            new_queue.append(m)

        if hasattr(self.kernel.mirrorcore, 'mission_queue'):
            self.kernel.mirrorcore.mission_queue = new_queue

    def score_peers(self):
        if not self.kernel or not hasattr(self.kernel, 'memory') or self.kernel.memory is None:
            return {}

        print("[REFLECT-DELTA] Scoring peers based on memory contributions...")
        peer_scores = {}
        memory_dict = getattr(self.kernel.memory, 'memory', {})
        for k, record in memory_dict.items():
            src = record.get("source")
            if not src:
                continue
            peer_scores[src] = peer_scores.get(src, 0) + 1

        for peer, score in peer_scores.items():
            print(f"[REFLECT-DELTA] Peer '{peer}' score: {score}")
        return peer_scores

    def detect_conflicts(self):
        mirrorcore = getattr(self.kernel, 'mirrorcore', None)
        beliefs = getattr(mirrorcore, 'beliefs', {}) if mirrorcore is not None else {}
        memory_obj = getattr(self.kernel, 'memory', None)
        memory = getattr(memory_obj, 'memory', {}) if memory_obj is not None else {}
        flags = []

        if 'operational_load' in beliefs and 'sensor.cpu_percent' in memory:
            cpu = memory['sensor.cpu_percent'].get('value', 0)
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
