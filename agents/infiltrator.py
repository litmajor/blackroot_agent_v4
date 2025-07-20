from agents.base import BaseAgent
import platform, socket, time

class InfiltratorAgent(BaseAgent):
    def __init__(self):
        super().__init__('INF-VENOM')

    def run(self):
        super().run()
        if self.priority == 'low':
            print("[INF-VENOM] Low-priority mode, delaying...")
            time.sleep(2)
        self.fingerprint_host()

    def fingerprint_host(self):
        print("[INF-VENOM] Fingerprinting host...")
        try:
            info = {
                'hostname': socket.gethostname(),
                'ip_address': socket.gethostbyname(socket.gethostname()),
                'os': platform.system(),
                'os_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor()
            }
            self.kernel.memory.state['host_fingerprint'] = info
            print("[INF-VENOM] Host Info:", info)
        except Exception as e:
            print(f"[INF-VENOM][ERR] {e}")
