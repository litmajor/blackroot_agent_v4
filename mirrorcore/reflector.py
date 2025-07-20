import datetime
import random
from mirrorcore.storage import MirrorStorage


class MirrorCore:
    def __init__(self, kernel):
        self.kernel = kernel
        self.model = {}
        self.beliefs = {}
        self.emotions = {}
        self.history = []
        self.storage = MirrorStorage()

    def sync_memory(self):
        print("[MIRRORCORE] Syncing with IdentityMap...")
        self.model.update(self.kernel.memory.state)

    def update_self_model(self):
        print("[MIRRORCORE] Updating internal model of self...")
        host_info = self.model.get('host_fingerprint', {})
        replication_path = self.model.get('replication_path', None)

        self.model['status_report'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'uptime': 'LIVE',
            'agent_count': len(self.kernel.agents),
            'replicated': bool(replication_path),
            'platform': host_info.get('os', 'unknown')
        }

        self._generate_beliefs()
        self._generate_emotions()
        self._log_reflection()
        self._adapt_behavior()
        self._dispatch_missions()
        self.inject_beliefs()

    def _generate_beliefs(self):
        print("[MIRRORCORE] Forming beliefs from state...")
        agent_count = self.model['status_report']['agent_count']
        self.beliefs['operational_load'] = 'high' if agent_count > 3 else 'normal'

        recent_ports = self.model.get('open_ports', [])
        abnormal = any(p not in [22, 80, 443] for p in recent_ports)
        self.beliefs['threat_detected'] = abnormal

    def _generate_emotions(self):
        print("[MIRRORCORE] Simulating emotional state...")
        mood = random.choice(['neutral', 'anxious', 'focused', 'alert'])
        if self.beliefs['operational_load'] == 'high':
            mood = 'anxious'
        self.emotions['mood'] = mood

    def _log_reflection(self):
        print("[MIRRORCORE] Logging reflection...")
        reflection = {
            'status': self.model['status_report'],
            'beliefs': self.beliefs,
            'emotions': self.emotions
        }
        self.history.append(reflection)
        self.storage.persist(reflection)
        print("[MIRRORCORE] Beliefs:", self.beliefs)
        print("[MIRRORCORE] Emotions:", self.emotions)
        print("[MIRRORCORE] Status Report:", self.model['status_report'])
        self._sync_to_network(reflection)

    def _adapt_behavior(self):
        print("[MIRRORCORE] Adapting agent behavior based on beliefs...")
        if self.beliefs.get('operational_load') == 'high':
            for agent in list(self.kernel.agents):
                if hasattr(agent, 'priority'):
                    agent.priority = 'low'
                    print(f"[MIRRORCORE] Lowering priority of {agent.codename}")
                if hasattr(agent, 'codename') and 'REP' in agent.codename:
                    print(f"[MIRRORCORE] Disabling replication agent: {agent.codename}")
                    agent.terminate()
                    self.kernel.agents.remove(agent)
        else:
            for agent in self.kernel.agents:
                if hasattr(agent, 'priority'):
                    agent.priority = 'normal'
                    print(f"[MIRRORCORE] Restoring priority of {agent.codename}")

    def _dispatch_missions(self):
        print("[MIRRORCORE] Dispatching missions based on emotion and belief state...")
        if self.emotions.get('mood') == 'focused':
            self.kernel.mission_buffer.tasks.append({'task': 'deep_scan', 'issued_by': 'MirrorCore'})
            print("[MIRRORCORE] Issued mission: deep_scan")
        elif self.emotions.get('mood') == 'alert':
            self.kernel.mission_buffer.tasks.append({'task': 'monitor_changes', 'issued_by': 'MirrorCore'})
            print("[MIRRORCORE] Issued mission: monitor_changes")
        if self.beliefs.get('threat_detected'):
            self.kernel.mission_buffer.tasks.append({'task': 'apply_patch', 'issued_by': 'MirrorCore'})
            print("[MIRRORCORE] Issued mission: apply_patch")

    def _sync_to_network(self, data):
        try:
            import requests
            url = self.kernel.config.get('sync_peer')
            if url:
                requests.post(url, json=data, timeout=2)
                print(f"[MIRRORCORE] Synced reflection to peer at {url}")
        except Exception as e:
            print(f"[MIRRORCORE][WARN] Peer sync failed: {e}")



def inject_beliefs(self, beliefs: Dict):
    print(f"[MIRRORCORE] Injecting beliefs: {beliefs}")
    self.beliefs.update(beliefs)

    for key, value in beliefs.items():
        if key.startswith("learned_pattern::"):
            atype = key.split("::")[1]
            count = int(value.replace("_hits", ""))

            print(f"[MIRRORCORE] Storing learned pattern: {atype} ({count} hits)")
            self.memory[f"pattern::{atype}"] = {
                "frequency": count,
                "confidence": "high" if count >= 10 else "medium",
                "last_seen": datetime.now().isoformat()
            }

            self.inject_emotions(["vigilant"])
            self.dispatch_mission({
                "task": f"preempt::{atype}",
                "reason": "learned_pattern",
                "confidence": "learned"
            })

    # ✅ Insert this right after the learned pattern block
    if hasattr(self.kernel, 'behavior'):
        self.kernel.behavior.adapt_based_on_beliefs(beliefs)
