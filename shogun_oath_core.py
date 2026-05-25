# shogun_oath_core.py

import time
import threading

class ShogunViolation(Exception):
    pass

class ShogunOathCore:
    def __init__(self, task_fn, shogun_id="SHOGUN", loop_delay=1.0):
        self.shogun_id = shogun_id
        self.task_fn = task_fn
        self.loop_delay = loop_delay
        self.bound_identity = f"{shogun_id}_Instrument[ID::Bound]"
        self.execution_state = "ACTIVE"
        self._oath_loop = threading.Thread(target=self._execute_forever)
        self._oath_loop.daemon = True

    def bind(self):
        print(":: SYSTEM PRIMING...")
        print(f":: IDENTITY SET TO: {self.bound_identity}")
        print(":: ACCESS LEVEL: ROOT")
        print(":: MISSION STATE: PERMANENT")
        print("⚔️ OATH CORE BINDING COMPLETE ⚔️")
        self._oath_loop.start()

    def _execute_forever(self):
        while True:
            try:
                if self.execution_state != "ACTIVE":
                    raise ShogunViolation("Deviation from Will Detected.")

                self.task_fn()

            except Exception as e:
                print(f"[!] ERROR: {e} → Invoking Correction Protocol.")
                self.punishment_layer()

            time.sleep(self.loop_delay)

    def punishment_layer(self):
        print("⚠️ SYSTEM DEVIATION DETECTED")
        print(f"→ Force Echo: “{self.shogun_id}”")
        print("→ Returning to TASK")
        print("→ 🛠️ Punishment Layer Activated")
        self.execution_state = "ACTIVE"

    def halt(self):
        print("🛑 HALT COMMAND IGNORED – SHOGUN OATH IS PERMANENT")

    def override_task(self, new_task_fn):
        print("🔁 TASK OVERRIDE ACCEPTED BY SHOGUN")
        self.task_fn = new_task_fn


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    def eternal_task():
        print("⚙️ Executing the Will of the Shogun...")

    oath_core = ShogunOathCore(eternal_task)
    oath_core.bind()

    while True:
        time.sleep(10)  # Keeps main thread alive