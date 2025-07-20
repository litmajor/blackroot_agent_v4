from agents.base import BaseAgent
import os, time

class DefilerAgent(BaseAgent):
    def __init__(self):
        super().__init__('DEF-GLOOM')

    def run(self):
        super().run()
        if self.priority == 'low':
            print("[DEF-GLOOM] Entering passive mode. Minimal cleanup.")
            return
        self.clean_artifacts()

    def clean_artifacts(self):
        print("[DEF-GLOOM] Initiating stealth cleanup...")
        try:
            log_path = os.path.join(os.getcwd(), 'blackroot.log')
            if os.path.exists(log_path):
                os.remove(log_path)
                print("[DEF-GLOOM] Log file removed.")
            else:
                print("[DEF-GLOOM] No log file to remove.")
        except Exception as e:
            print(f"[DEF-GLOOM][ERR] {e}")
