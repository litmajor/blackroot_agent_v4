import time
import threading
import uuid
from typing import Dict, List, Callable
import logging

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
from recon import ReconModule

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

# Thread-safe module registry
module_registry: Dict[str, Callable] = {}
registry_lock = threading.Lock()

def register_module(name: str, initializer: Callable):
    with registry_lock:
        module_registry[name] = initializer
        logging.getLogger('BKR-KERNEL').info(f"Module registered: {name}")

class BlackrootKernel:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger('BKR-KERNEL')
        if not config or not isinstance(config, dict):
            raise ValueError("Config must be a non-empty dictionary")
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.running = False

        # Core systems
        try:
            self.memory = IdentityMap()
            self.mission_buffer = MissionBuffer()
            self.agents = load_agents()
            self.stealth = CloakSupervisor(self, stealth_threshold=config.get('stealth_threshold', 0.8))
            self.behaviour = BehaviorAdapter(self)
            self.mirrorcore = MirrorCore(self)
            self.anomaly_classifier = AnomalyClassifierDaemon()
            self.payload_engine = PayloadEngine()
            self.mutator_engine = MutatorEngine()

            self.black_vault = BlackVault(
password=self.config.get('vault_password', get_random_bytes(32).hex()),
 rotate_days=self.config.get('vault_rotate_days', 7),
    vault_path=self.config.get('vault_path', 'vault.bkr')
)
            self.swarm = SwarmMesh("controller-node")
            self.swarm.redis = Redis(host="localhost", port=6379, decode_responses=True)
            self.recon_module = ReconModule(self.black_vault, self.swarm, self.swarm.redis)
            self.ghost_layer = GhostLayer(self.black_vault)
            self.ghost_layer_daemon = GhostLayerDaemon(self.ghost_layer, self.black_vault)
self.self_learning_injection = SelfLearningInjection(self.black_vault)
            self.ghost_hive = GhostHive([GhostLayer(self.black_vault) for _ in range(5)], self.black_vault)
            self.register_module("recon", )

        self._register_builtin_modules()

    def _register_builtin_modules(self):
        register_module("ghost_layer", self.ghost_layer.run)
        register_module("anomaly_classifier", self.anomaly_classifier.start)
        register_module("black_vault", lambda: self.black_vault)
        register_module("payload_engine", lambda: self.payload_engine)
        register_module("mutator_engine", lambda: self.mutator_engine)
        register_module("mirrorcore", lambda: self.mirrorcore)

    def boot(self):
        self.logger.info(f"Booting Blackroot Kernel ID={self.id}")
        self.running = True
        try:
            self.spawn_agents()
            self.anomaly_classifier.start()
            self.ghost_layer.run()
            self.swarm.connect()
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        except Exception as e:
            self.logger.error(f"Boot failed: {e}")
            self.shutdown()

    def spawn_agents(self):
        for agent in self.agents:
            if agent.should_activate(self.config):
                self.logger.info(f"Activating agent: {agent.codename}")
                try:
                    agent.attach_kernel(self)
                    threading.Thread(target=agent.run, daemon=True).start()
                except AttributeError as e:
                    self.logger.error(f"Failed to activate {agent.codename}: {e}")

    def _heartbeat_loop(self):
        interval = self.config.get('heartbeat_interval', 5)
        if interval <= 0:
            self.logger.error("Invalid heartbeat interval, using default: 5")
            interval = 5
        while self.running:
            try:
                self.logger.info("Heartbeat tick")
                self.mission_buffer.sync()
                self.stealth.evaluate()
                self.reflect()
                self.swarm.ping()
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
            time.sleep(interval)

    def reflect(self):
        try:
            self.mirrorcore.sync_memory()
            self.mirrorcore.update_self_model()
            self.mutator_engine.evolve_composition()
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")

    def shutdown(self):
        self.logger.info("Shutting down agents and systems")
        self.running = False
        for agent in self.agents:
            try:
                agent.terminate()
            except AttributeError:
                self.logger.warning(f"Agent {agent.codename} lacks terminate method")
        try:
            self.swarm.disconnect()
        except Exception as e:
            self.logger.error(f"Swarm disconnect failed: {e}")
        self.logger.info("Shutdown complete.")

if __name__ == "__main__":
    config = {
        'heartbeat_interval': 5,
        'stealth_level': 'adaptive',
        'stealth_threshold': 0.8,
        'agent_whitelist': ['INF-VENOM', 'REP-SHADOW', 'DEF-GLOOM']
    }
    try:
        kernel = BlackrootKernel(config)
        kernel.boot()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        kernel.shutdown()
    except Exception as e:
        logging.getLogger('BKR-KERNEL').error(f"Main loop failed: {e}")