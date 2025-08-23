# passive_plus.py
import dns.resolver
import requests
import logging
import re
import os
from typing import List, Dict, Any

LOG = logging.getLogger("PassivePlus")
logging.basicConfig(level=logging.INFO)

try:
    import nuclei
except ImportError:
    nuclei = None
except Exception as exc:
    nuclei = None
    LOG.warning("Failed to import Nuclei: %s", exc)


class PassivePlus:
    """
    Passive reconnaissance helper.
    Provides wrappers for DNS, CSP, security.txt, and optional Nuclei scanning.
    """

    @staticmethod
    def nuclei_scan(target: str, templates: str = "default") -> List[Dict[str, Any]]:
        """Run Nuclei on a target; returns JSON list of findings if possible."""
        if nuclei is None:
            LOG.warning("Nuclei Python bindings not available → skipping scan")
            return []

        if not any(os.path.exists(p) for p in ["/usr/local/bin/nuclei", "/usr/bin/nuclei"]):
            LOG.warning("Nuclei binary not found in PATH → skipping scan")
            return []

        import subprocess
        import json
        try:
            nuclei_path = None
            for p in ["/usr/local/bin/nuclei", "/usr/bin/nuclei"]:
                if os.path.exists(p):
                    nuclei_path = p
                    break
            if not nuclei_path:
                LOG.warning("Nuclei binary not found in PATH → skipping scan")
                return []
            cmd = [
                nuclei_path,
                "-u", target,
                "-t", templates if templates != "default" else "",
                "-json"
            ]
            # Remove empty template argument if default
            cmd = [c for c in cmd if c]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            findings = []
            for line in result.stdout.splitlines():
                try:
                    findings.append(json.loads(line))
                except Exception:
                    continue
            return findings
        except Exception as exc:
            LOG.warning("Nuclei error: %s", exc)
            return []

    @staticmethod
    def csp_domains(target: str) -> List[str]:
        """Extract all whitelisted domains from CSP header of target."""
        try:
            r = requests.head(target, timeout=5, allow_redirects=True)
            csp = r.headers.get("Content-Security-Policy", "")
            return re.findall(r"[\w.-]+\.\w+", csp)
        except Exception as exc:
            LOG.debug("CSP parsing error: %s", exc)
            return []

    @staticmethod
    def dns_records(domain: str) -> Dict[str, Any]:
        """Return DNS records: A, AAAA, CNAME, TXT, MX, NS."""
        records = {}
        for qtype in ("A", "AAAA", "CNAME", "TXT", "MX", "NS"):
            try:
                ans = dns.resolver.resolve(domain, qtype)
                records[qtype] = [str(r) for r in ans]
            except Exception:
                records[qtype] = []
        return records

    @staticmethod
    def security_txt(target: str) -> Dict[str, str]:
        """Parse /.well-known/security.txt if present on target."""
        url = target.rstrip("/") + "/.well-known/security.txt"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = {}
                for line in r.text.splitlines():
                    if ":" in line and not line.startswith("#"):
                        k, v = line.split(":", 1)
                        data[k.strip()] = v.strip()
                return data
        except Exception as exc:
            LOG.debug("security.txt fetch error: %s", exc)
        return {}
