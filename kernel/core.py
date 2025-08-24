

# ---------- soft-fail fallbacks ----------
try:
    from ghost_layer import GhostLayer, GhostHive, GhostLayerDaemon
except ImportError:
    class GhostLayer:
        def __init__(self, vault): pass
        def run(self): pass
    class GhostHive:
        def __init__(self, layers, vault): pass
        def run(self): pass
    class GhostLayerDaemon:
        def __init__(self, ghost_layer, vault): pass

try:
    from learning import LearningAgent
    SelfLearningInjection = LearningAgent
except ImportError:
    class SelfLearningInjection:
        def __init__(self, vault): pass

import time
import threading
import uuid
from typing import Dict, List, Callable
import logging
try:
    from agents.loader import load_agents
    from memory.buffer import MissionBuffer
    from memory.identity import IdentityMap
    from stealth.cloak import CloakSupervisor
    from agents.behaviour_adapter import BehaviorAdapter
    from mirrorcore.reflector import MirrorCore
    from black_vault import BlackVault
    from swarm_mesh import SwarmMesh
    from anomaly_classifier import AnomalyClassifierDaemon
    from payload_engine import PayloadEngine
    from mutator_engine import MutatorEngine
    from recon.scanner import ReconModule
except ModuleNotFoundError:
    from agents.loader import load_agents
    from memory.buffer import MissionBuffer
    from memory.identity import IdentityMap
    from stealth.cloak import CloakSupervisor
    from agents.behaviour_adapter import BehaviorAdapter
    from mirrorcore.reflector import MirrorCore
    from black_vault import BlackVault
    from swarm_mesh import SwarmMesh
    from anomaly_classifier import AnomalyClassifierDaemon
    from payload_engine import PayloadEngine
    from mutator_engine import MutatorEngine
    from recon.scanner import ReconModule
from Crypto.Random import get_random_bytes
import json
from redis import Redis


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
        self.memory = IdentityMap(
            redis=Redis(host="localhost", port=6379, decode_responses=True)
        )
        self.mission_buffer = MissionBuffer(
            redis=Redis(host="localhost", port=6379, decode_responses=True)
        )
        self.agents = load_agents()
        # --- Integrate ScoutAgent (Recon/Mapping) ---
        try:
            from agents.scout import ScoutAgent
            scout_targets = self.config.get('scout_targets', [])
            command_center = self.config.get('command_center', 'localhost:8888')
            self.scout_agent = ScoutAgent(
                agent_id=self.id,
                initial_target_range=scout_targets,
                command_center_address=command_center,
                log_level=logging.INFO,
                black_vault=self.black_vault
            )
            register_module("scout_agent", lambda: self.scout_agent)
            self.logger.info("ScoutAgent integrated and registered as 'scout_agent' module.")
        except Exception as e:
            self.logger.warning(f"ScoutAgent integration failed: {e}")
        # --- Integrate RepairAgent (Immune System) ---
        try:
            from agents.repair import RepairAgent
            # Build node_registry from agents (or other system state)
            self.node_registry = {a.codename: {'health': getattr(a, 'health', None), 'state': {}, 'configuration': {}, 'dependencies': getattr(a, 'dependencies', []), 'metrics': {}} for a in self.agents if hasattr(a, 'codename')}
            self.repair_agent = RepairAgent(agent_id=self.id, node_registry=self.node_registry)
            # Register as a module for global access
            register_module("repair_agent", lambda: self.repair_agent)
            self.logger.info("RepairAgent integrated and registered as 'repair_agent' module.")
        except Exception as e:
            self.logger.warning(f"RepairAgent integration failed: {e}")
        self.stealth = CloakSupervisor(self, stealth_threshold=config.get('stealth_threshold', 0.8))
        self.behaviour = BehaviorAdapter(self)
        self.mirrorcore = MirrorCore(self)
        self.anomaly_classifier = AnomalyClassifierDaemon()
        self.payload_engine = PayloadEngine()
        self.mutator_engine = MutatorEngine(source_path="mutator_engine.py")

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
        try:
            self.ghost_hive = GhostHive([GhostLayer(self.black_vault) for _ in range(5)], self.black_vault)
        except Exception as e:
            self.logger.warning("GhostHive unavailable: %s", e)
            self.ghost_hive = None
        register_module("recon", self.recon_module.advanced_recon)
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
            # Start RepairAgent (immune system) in background
            try:
                import asyncio
                threading.Thread(target=lambda: asyncio.run(self.repair_agent.start()), daemon=True).start()
                self.logger.info("RepairAgent started in background.")
            except Exception as e:
                self.logger.warning(f"RepairAgent start failed: {e}")
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
    def execute_command(self, command):
        command_id = command.get("command_id", get_random_bytes(16).hex())
        if command.get("type") == "recon":
            target = command.get("target")
            max_depth = command.get("max_depth", 3)
            brute_subdomains = command.get("brute_subdomains", True)
            result = self.recon_module.advanced_recon(target, max_depth, None, brute_subdomains, command_id)
            # Ensure channel exists
            channel = getattr(self.swarm, 'channel', 'recon_channel')
            self.swarm.redis.publish(channel, json.dumps({
                "command_id": command_id,
                "type": "recon_report",
                "data": result
            }))

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
        # Stop RepairAgent
        try:
            import asyncio
            if hasattr(self, 'repair_agent'):
                asyncio.run(self.repair_agent.stop())
                self.logger.info("RepairAgent stopped.")
        except Exception as e:
            self.logger.warning(f"RepairAgent stop failed: {e}")
        self.logger.info("Shutdown complete.")

if __name__ == "__main__":


    config = {
        'heartbeat_interval': 5,
        'stealth_level': 'adaptive',
        'stealth_threshold': 0.8,
        'agent_whitelist': ['INF-VENOM', 'REP-SHADOW', 'DEF-GLOOM']
    }
    kernel = None
    try:
        kernel = BlackrootKernel(config)
        kernel.boot()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if kernel:
            kernel.shutdown()
    except Exception as e:
        logging.getLogger('BKR-KERNEL').error(f"Main loop failed: {e}")
        if kernel:
            kernel.shutdown()