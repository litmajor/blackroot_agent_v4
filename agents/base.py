class BaseAgent:
    def __init__(self, codename: str):
        self.codename = codename
        self.kernel = None
        self.priority = 'normal'

    def should_activate(self, config):
        return self.codename in config.get('agent_whitelist', [])

    def attach_kernel(self, kernel):
        self.kernel = kernel

    def run(self):
        print(f"[AGENT:{self.codename}] Running with priority {self.priority}")

    def terminate(self):
        print(f"[AGENT:{self.codename}] Terminated")
