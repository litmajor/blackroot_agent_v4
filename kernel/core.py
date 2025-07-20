import time
import threading
import uuid
from typing import Dict, List, Callable

from agents.loader import load_agents
from memory.buffer import MissionBuffer
from memory.identity import IdentityMap
from stealth.cloak import CloakSupervisor
from agents.behaviour_adapter import BehaviorAdapter

from mirrorcore.reflector import MirrorCore
from ghost_layer import GhostLayerDaemon
from black_vault import BlackVault
from swarm_mesh import SwarmMesh
from anomaly_classifier import AnomalyClassifierDaemon
from payload_engine import PayloadEngine
from mutator_engine import MutatorEngine

# Global registry for dynamic modules
module_registry: Dict[str, Callable] = {}

def register_module(name: str, initializer: Callable):
    module_registry[name] = initializer
    print(f"[🔧] Module registered: {name}")

class BlackrootKernel:
    def __init__(self, config: Dict):
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.running = False

        # Core systems
        self.memory = IdentityMap()
        self.mission_buffer = MissionBuffer()
        self.agents = load_agents()
        self.stealth = CloakSupervisor(self)
        self.behaviour = BehaviorAdapter(self)
        self.mirrorcore = MirrorCore(self)

        # Intelligence & stealth modules
        self.ghost_layer = GhostLayerDaemon()
        self.black_vault = BlackVault("vault.bkr")
        self.swarm = SwarmMesh(self.id)
        self.anomaly_classifier = AnomalyClassifierDaemon()
        self.payload_engine = PayloadEngine()
        self.mutator_engine = MutatorEngine()

        # Auto-register modules
        self._register_builtin_modules()

    def _register_builtin_modules(self):
        register_module("ghost_layer", self.ghost_layer.run)
        register_module("anomaly_classifier", self.anomaly_classifier.start)
        register_module("black_vault", lambda: self.black_vault)
        register_module("payload_engine", lambda: self.payload_engine)
        register_module("mutator_engine", lambda: self.mutator_engine)
        register_module("mirrorcore", lambda: self.mirrorcore)

    def boot(self):
        print(f"[BKR-KERNEL] Booting Blackroot Kernel ID={self.id}")
        self.running = True

        # Boot support daemons
        self.spawn_agents()
        self.anomaly_classifier.start()
        self.ghost_layer.run()
        self.swarm.connect()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def spawn_agents(self):
        for agent in self.agents:
            if agent.should_activate(self.config):
                print(f"[KERNEL] Activating agent: {agent.codename}")
                agent.attach_kernel(self)
                agent.run()

    def _heartbeat_loop(self):
        interval = self.config.get('heartbeat_interval', 5)
        while self.running:
            try:
                print("[KERNEL] Heartbeat tick")
                self.mission_buffer.sync()
                self.stealth.evaluate()
                self.reflect()
                self.swarm.ping()
                time.sleep(interval)
            except Exception as e:
                print(f"[KERNEL][ERR] {e}")

    def reflect(self):
        self.mirrorcore.sync_memory()
        self.mirrorcore.update_self_model()
        self.mutator_engine.evolve_composition()

    def shutdown(self):
        print("[KERNEL] Shutting down agents and systems")
        self.running = False
        for agent in self.agents:
            agent.terminate()
        self.swarm.disconnect()
        print("[KERNEL] Shutdown complete.")

if __name__ == "__main__":
    config = {
        'heartbeat_interval': 5,
        'stealth_level': 'adaptive',
        'agent_whitelist': ['INF-VENOM', 'REP-SHADOW', 'DEF-GLOOM']
    }

    kernel = BlackrootKernel(config)
    kernel.boot()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        kernel.shutdown()
