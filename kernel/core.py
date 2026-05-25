

# ---------- soft-fail fallbacks ----------
try:
    from ghost_layer import GhostLayer as GhostLayerImpl, GhostHive as GhostHiveImpl, GhostLayerDaemon as GhostLayerDaemonImpl  # type: ignore
except ImportError:
    class GhostLayerImpl:  # type: ignore
        def __init__(self, vault): pass
        def run(self): pass
    class GhostHiveImpl:  # type: ignore
        def __init__(self, layers, vault): pass
        def run(self): pass
    class GhostLayerDaemonImpl:  # type: ignore
        def __init__(self, ghost_layer, vault): pass

# Alias for use throughout kernel
GhostLayer = GhostLayerImpl
GhostHive = GhostHiveImpl
GhostLayerDaemon = GhostLayerDaemonImpl

try:
    from agents.privesc import PrivescEngine as PrivescEngineImpl  # type: ignore
except ImportError:
    class PrivescEngineImpl:  # type: ignore
        def __init__(self, vault, swarm=None): pass
        def assess_environment(self): return {'threat_level': 0, 'techniques': []}
        def attempt_escalation(self, technique=None): return {'success': False, 'error': 'PrivescEngine unavailable'}

PrivescEngine = PrivescEngineImpl

try:
    from agents.learning import LearningAgent
    SelfLearningInjection = LearningAgent
except ImportError:
    class SelfLearningInjection:  # type: ignore
        def __init__(self, vault): pass

import time
import threading
import uuid
from typing import Dict, List, Callable, Optional, Any
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
    from agents.phone_agent_wallet import PhoneWalletAgent
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
from redis import Redis
from decimal import Decimal
import json
from typing import TYPE_CHECKING


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

# Thread-safe module registry with duplicate prevention
module_registry: Dict[str, Callable] = {}
registered_module_hashes: set = set()  # Track by canonical module hash to prevent duplicates
registry_lock = threading.Lock()
networking_available = False  # Capability flag for networking

def _compute_module_hash(name: str, agent_class=None) -> str:
    """Compute canonical hash for module identity (class name + module path)."""
    if agent_class is not None:
        # Use class name + module path for canonical identity
        module_path = getattr(agent_class, '__module__', '')
        class_name = getattr(agent_class, '__name__', '')
        return f"{module_path}:{class_name}"
    return name

def register_module(name: str, initializer: Callable, agent_class=None):
    """Register module with idempotency checking. Only first registration logs (no duplicates)."""
    with registry_lock:
        # Compute canonical module hash for deduplication
        module_hash = _compute_module_hash(name, agent_class)
        
        # Check if this exact module already registered
        if module_hash in registered_module_hashes:
            logging.getLogger('BKR-KERNEL').debug(f"Module {name} already registered by hash {module_hash[:40]} (idempotent skip)")
            return False  # Already registered - don't initialize again
        
        # First-time registration: record hash, register initializer
        registered_module_hashes.add(module_hash)
        module_registry[name] = initializer
        logging.getLogger('BKR-KERNEL').info(f"Module registered: {name}")
        return True  # First-time registration

class BlackrootKernel:
    """Thread-safe singleton kernel to prevent duplicate initialization."""
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, config: dict):
        """Enforce singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BlackrootKernel, cls).__new__(cls)
        return cls._instance
    
    def _autodiscover_and_register_agents(self):
        """
        Discover and register agents with lenient interface contracts.
        FAILURE ISOLATION: Failed agents skip permanently, logged separately.
        Only strictly enforce interface for BaseAgent subclasses.
        """
        import importlib
        import os
        import inspect
        from agents.base import BaseAgent
        
        agent_dir = os.path.join(os.path.dirname(__file__), '..', 'agents')
        agent_files = [f for f in os.listdir(agent_dir) if f.endswith('.py') and not f.startswith('__')]
        
        failed_agents = []
        registered_count = 0
        
        for file in agent_files:
            module_name = f"agents.{file[:-3]}"
            try:
                module = importlib.import_module(module_name)
                # Find agent classes (ending with 'Agent')
                for attr in dir(module): 
                    if attr.endswith('Agent') and attr[0].isupper():
                        agent_cls = getattr(module, attr)
                        if inspect.isabstract(agent_cls):
                            continue
                        
                        # For BaseAgent subclasses, check for run/stop/status methods
                        # For standalone agents, try instantiation directly
                        is_base_agent = issubclass(agent_cls, BaseAgent) if inspect.isclass(agent_cls) else False
                        
                        if is_base_agent:
                            # BaseAgent subclasses MUST have run (and now stop/status from base)
                            if not hasattr(agent_cls, 'run') or (hasattr(agent_cls.run, '__isabstractmethod__') and agent_cls.run.__isabstractmethod__):
                                error_msg = f"{attr} (BaseAgent) missing run() implementation"
                                self.logger.warning(f"FAILURE DOMAIN: {error_msg}")
                                failed_agents.append((attr, error_msg))
                                continue
                        
                        # Networking-dependent agents: skip if networking unavailable
                        networking_agents = {'PeerLinkerAgent', 'EnhancedPeerLinkerAgent'}
                        if attr in networking_agents and not networking_available:
                            self.logger.debug(f"Skipping {attr}: networking not available")
                            continue
                        
                        # Try to instantiate with special handling for known agents
                        try:
                            if attr == 'ScoutAgent':
                                scout_targets = self.config.get('scout_targets', [])
                                command_center = self.config.get('command_center', 'localhost:8888')
                                agent_instance = agent_cls(
                                    agent_id=self.id,
                                    initial_target_range=scout_targets,
                                    command_center_address=command_center,
                                    log_level=logging.INFO
                                )
                            elif attr == 'RepairAgent':
                                # Build node_registry from agents (or other system state)
                                if not hasattr(self, 'node_registry'):
                                    self.node_registry = {a.codename: {'health': getattr(a, 'health', None), 'state': {}, 'configuration': {}, 'dependencies': getattr(a, 'dependencies', []), 'metrics': {}} for a in self.agents if hasattr(a, 'codename')}
                                agent_instance = agent_cls(
                                    agent_id=self.id,
                                    node_registry=self.node_registry
                                )
                            elif attr == 'ExfiltratorAgent':
                                agent_instance = agent_cls(mission=None, agent_id=self.id)
                            elif attr == 'EnforcerAgent':
                                agent_instance = agent_cls(
                                    enforcement_engine=None,
                                    economic_layer=None,
                                    command_control=None,
                                    constitution_reference=None
                                )
                            elif attr == 'ArchivistAgent':
                                agent_instance = agent_cls(
                                    agent_id=self.id,
                                    secret_key='default_secret_key'
                                )
                            else:
                                # Try no-args constructor
                                try:
                                    agent_instance = agent_cls()
                                except TypeError:
                                    # If no-args fails, try with agent_id
                                    agent_instance = agent_cls(agent_id=self.id)
                            
                            # Register with idempotency check (pass agent_cls for deduplication)
                            if register_module(attr.lower(), lambda inst=agent_instance: inst, agent_class=agent_cls):
                                self.logger.info(f"✓ Agent registered: {attr}")
                                registered_count += 1
                            # else: already registered, skip logging
                            
                        except TypeError as te:
                            # Constructor argument mismatch - can't instantiate
                            error_msg = f"Constructor error: {str(te)[:60]}"
                            self.logger.debug(f"FAILURE DOMAIN: {attr} - {error_msg}")
                            failed_agents.append((attr, error_msg))
                            # Skip this agent permanently
                        except Exception as e:
                            error_msg = f"Instantiation error: {str(e)[:60]}"
                            self.logger.debug(f"FAILURE DOMAIN: {attr} - {error_msg}")
                            failed_agents.append((attr, error_msg))
                            # Skip this agent permanently
                            
            except Exception as e:
                self.logger.warning(f"FAILURE DOMAIN: Module {module_name} import failed: {e}")
                failed_agents.append((module_name, str(e)))
        
        # Summary
        if registered_count > 0:
            self.logger.info(f"✓✓✓ Agent autodiscovery: {registered_count} agents registered")
        if failed_agents:
            self.logger.warning(f"⚠ Failure domain isolated: {len(failed_agents)} modules skipped permanently")
            for agent_name, error in failed_agents[:5]:  # Show first 5 failures
                self.logger.debug(f"  - {agent_name}: {error}")
    def __init__(self, config: dict):
        """Initialize kernel once. Subsequent calls return existing instance."""
        if BlackrootKernel._initialized:
            return  # Already initialized
        
        BlackrootKernel._initialized = True
        import uuid
        self.id = config.get("node_id", str(uuid.uuid4()))
        if not config or not isinstance(config, dict):
            raise ValueError("Config must be a non-empty dictionary")
        self.config = config
        self.black_vault = config.get("black_vault", None)
        self.vault: Optional[Any] = None  # Unified vault (from vault_adapter), can be UnifiedVault or None
        self.logger = logging.getLogger('BKR-KERNEL')
        self.logger.info(f"Initializing singleton kernel (ID={self.id})")
        self.running = False
        
        # Core enhancements attributes (initialized in main.py)
        self.lifecycle_manager: Optional[Any] = None
        self.threat_feedback: Optional[Any] = None
        self.plugin_discovery: Optional[Any] = None
        self.config_security: Optional[Any] = None
        self.edr_orchestrator: Optional[Any] = None

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
                    log_level=logging.INFO
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
        
        # Initialize ACC-LEDGER singleton (economic subsystem) - only once per kernel
        try:
            from agents.accountant import ACCLedger
            self.ledger = ACCLedger(creator_id=self.id, initial_treasury=Decimal(1000))
            register_module("ledger", lambda: self.ledger)
            self.logger.info("ACC-LEDGER singleton registered (prevents duplicate initialization)")
        except Exception as e:
            self.logger.warning(f"ACC-LEDGER initialization failed: {e}")
            self.ledger = None
        
        self.privesc_engine = PrivescEngine(vault=self.black_vault)
        self.self_learning_injection = SelfLearningInjection(self.config)
        try:
            self.ghost_hive = GhostHive([GhostLayer(self.black_vault) for _ in range(5)], self.black_vault)
        except Exception as e:
            self.logger.warning("GhostHive unavailable: %s", e)
            self.ghost_hive = None
        register_module("recon", self.recon_module.advanced_recon)
        self._register_builtin_modules()
        self._autodiscover_and_register_agents()

    
    def _register_builtin_modules(self):
        register_module("ghost_layer", self.ghost_layer.run)
        register_module("privesc_engine", lambda: self.privesc_engine)
        register_module("anomaly_classifier", self.anomaly_classifier.start)
        register_module("black_vault", lambda: self.black_vault)
        register_module("payload_engine", lambda: self.payload_engine)
        register_module("mutator_engine", lambda: self.mutator_engine)
        register_module("mirrorcore", lambda: self.mirrorcore)
        register_module("ghost_hive", lambda: self.ghost_hive)
        register_module("self_learning_injection", lambda: self.self_learning_injection)
        try:
            from agents.phone_agent_wallet import PhoneWalletAgent
            register_module("phone_wallet_agent", lambda: PhoneWalletAgent())
        except ImportError:
            self.logger.warning("PhoneWalletAgent unavailable: import failed.")
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