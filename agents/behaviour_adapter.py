from datetime import datetime

class BehaviorAdapter:
    def __init__(self, kernel):
        self.kernel = kernel
        self.adaptation_log = []

    def adapt_based_on_beliefs(self, beliefs):
        print("[BEHAVIOR] Reflecting on belief triggers...")
        for key, value in beliefs.items():
            if key.startswith("learned_pattern::"):
                atype = key.split("::")[1]
                frequency = int(value.replace("_hits", ""))

                if frequency >= 5:
                    self._prioritize_agents(atype)
                    self._log_adaptation(f"Escalated agents due to {atype} ({frequency}x)")

    def _prioritize_agents(self, anomaly_type):
        for agent in self.kernel.agents:
            if hasattr(agent, 'codename') and 'DEFENDER' in agent.codename.upper():
                print(f"[BEHAVIOR] Escalating {agent.codename} to high priority in response to {anomaly_type}")
                agent.priority = 'high'

    def _log_adaptation(self, message):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        self.adaptation_log.append(entry)
        if hasattr(self.kernel, 'storage'):
            self.kernel.storage.persist({
                "source": "BEHAVIOR-ADAPTER",
                **entry
            })
