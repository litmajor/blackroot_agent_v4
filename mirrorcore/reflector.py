import datetime
import random
import time
from typing import Dict, Optional, List, Any
from mirrorcore.storage import MirrorStorage

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    HAS_REQUESTS = False


class MirrorCore:
    def dispatch_mission(self, mission):
        print(f"[MIRRORCORE] Dispatching mission: {mission}")
        # You can implement actual mission dispatch logic here if needed
    def __init__(self, kernel):
        self.kernel = kernel
        self.model = {}
        self.beliefs = {}
        self.emotions = {}
        self.history = []
        self.storage = MirrorStorage()

    def sync_memory(self):
        """Sync memory with kernel state and load recent reflections for continuity"""
        print("[MIRRORCORE] Syncing with IdentityMap...")
        self.model.update(self.kernel.memory.state)
        
        # ✅ ISSUE #4: Load recent memory for continuity
        self._load_recent_memory()
        
        # ✅ ISSUE #2: Map agent identities to artifacts
        self._map_agent_identities()

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
        """
        ✅ ISSUE #5: Robust network sync with retry logic and proper error handling
        """
        if not HAS_REQUESTS:
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = self.kernel.config.get('sync_peer')
                if url:
                    # ✅ Retry with exponential backoff
                    response = requests.post(url, json=data, timeout=2)
                    if response.status_code == 200:
                        print(f"[MIRRORCORE] Synced reflection to peer at {url}")
                        return
                    else:
                        print(f"[MIRRORCORE][WARN] Sync failed with status {response.status_code}, retrying...")
            except Exception as e:
                print(f"[MIRRORCORE][WARN] Peer sync failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
        
        print(f"[MIRRORCORE][ERROR] Failed to sync after {max_retries} attempts")
    
    def _load_recent_memory(self):
        """
        ✅ ISSUE #4: Load past reflections for continuity of identity
        Agent system should remember what it believed/did recently
        """
        try:
            dates = self.storage.list_log_dates()
            if dates:
                latest_date = dates[-1]
                logs = self.storage.search_by_date(latest_date)
                if logs:
                    # Keep last 5 reflections for context
                    self.model['recent_reflections'] = logs[-5:]
                    print(f"[MIRRORCORE] Loaded {len(logs[-5:])} recent reflections from {latest_date}")
                    
                    # ✅ Extract previous beliefs for continuity
                    if logs[-1].get('beliefs'):
                        self.model['previous_beliefs'] = logs[-1]['beliefs']
                        print(f"[MIRRORCORE] Restored previous beliefs: {logs[-1]['beliefs']}")
        except Exception as e:
            print(f"[MIRRORCORE][WARN] Failed to load recent memory: {e}")
    
    def _map_agent_identities(self):
        """
        ✅ ISSUE #2: Bind artifacts to agent identities
        Prevents "Unknown" agent labeling by mapping artifacts to actual agents
        """
        try:
            artifacts = self.model.get("artifacts", [])
            agent_map = {}
            
            for agent in self.kernel.agents:
                codename = getattr(agent, "codename", "UNKNOWN")
                agent_id = getattr(agent, "id", None)
                role = getattr(agent, "role", "unknown")
                agent_type = getattr(agent, "type", "generic")
                
                agent_map[codename] = {
                    "id": agent_id,
                    "role": role,
                    "type": agent_type,
                    "artifact_count": len([a for a in artifacts if getattr(a, 'owner', None) == codename])
                }
            
            self.model['agent_registry'] = agent_map
            print(f"[MIRRORCORE] Mapped {len(agent_map)} agent identities")
            
        except Exception as e:
            print(f"[MIRRORCORE][WARN] Failed to map agent identities: {e}")
    
    def ingest_mission_outcome(self, mission_result: Dict[str, Any]):
        """
        ✅ ISSUE #3: Feedback loop - beliefs update based on mission results
        System learns if actions are effective (state → belief → action → outcome → updated belief)
        """
        try:
            success = mission_result.get("success", False)
            task = mission_result.get("task", "unknown")
            
            if success:
                self.beliefs["last_action_effective"] = True
                self.beliefs[f"learned_success::{task}"] = True
                print(f"[MIRRORCORE] Mission successful: {task} - updating beliefs for next cycle")
            else:
                self.beliefs["last_action_effective"] = False
                self.beliefs[f"learned_failure::{task}"] = True
                print(f"[MIRRORCORE] Mission failed: {task} - adjusting strategy")
            
            # Update emotional state based on outcomes
            if success and self.beliefs.get("threat_detected"):
                self.emotions['mood'] = 'confident'
            elif not success and self.beliefs.get("threat_detected"):
                self.emotions['mood'] = 'anxious'
            
            print(f"[MIRRORCORE] Updated beliefs after mission outcome: {mission_result}")
            
        except Exception as e:
            print(f"[MIRRORCORE][ERROR] Failed to ingest mission outcome: {e}")


    def inject_beliefs(self, beliefs: Optional[Dict] = None):
        """
        ✅ ISSUE #6: Behavioral hook moved OUTSIDE loop for efficiency
        """
        if beliefs is None:
            beliefs = self.beliefs
        if beliefs is None:
            beliefs = {}
        print(f"[MIRRORCORE] Injecting beliefs: {beliefs}")
        self.beliefs.update(beliefs)

        for key, value in beliefs.items():
            if key.startswith("learned_pattern::") or key.startswith("learned::"):
                atype = key.split("::")[1]
                clean_val = str(value).replace("_hits", "").strip()
                count = int(clean_val) if clean_val.isdigit() else 0

                print(f"[MIRRORCORE] Storing learned pattern: {atype} ({count} hits)")
                if not hasattr(self, 'memory'):
                    self.memory = {}
                self.memory[f"pattern::{atype}"] = {
                    "frequency": count,
                    "confidence": "high" if count >= 10 else "medium",
                    "last_seen": datetime.datetime.now().isoformat()
                }

                if hasattr(self, 'inject_emotions') and callable(getattr(self, 'inject_emotions', None)):
                    self.inject_emotions(["vigilant"])
                if hasattr(self, 'dispatch_mission') and callable(getattr(self, 'dispatch_mission', None)):
                    self.dispatch_mission({
                        "task": f"preempt::{atype}",
                        "reason": "learned_pattern",
                        "confidence": "learned"
                    })
        
        # ✅ ISSUE #6: Behavioral hook OUTSIDE loop - runs once per belief injection
        behavior_module = getattr(self.kernel, 'behavior', None) or getattr(self.kernel, 'behaviour', None)
        if behavior_module and hasattr(behavior_module, 'adapt_based_on_beliefs'):
            behavior_module.adapt_based_on_beliefs(self.beliefs)

    
    def inject_emotions(self, emotions):
        print(f"[MIRRORCORE] Injecting emotions: {emotions}")
        if isinstance(emotions, list):
            self.emotions['injected'] = emotions
        else:
            self.emotions['injected'] = [emotions]
