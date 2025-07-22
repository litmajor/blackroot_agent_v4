import os
import requests
import random
import time
import logging
import threading
import json
import pickle
import re
import socket
import concurrent.futures
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import Optional, List, Dict
import dnstwist
from Crypto.Random import get_random_bytes
from fastapi import FastAPI
from redis import Redis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

RECON_OUTPUT_DIR = "recon_results"
FORM_CACHE_FILE = "form_cache.pkl"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "curl/7.68.0",
    "BLACKROOT.Recon-Agent"
]
PROXY_POOL = [
    # Add proxy servers here, e.g., {"http": "http://proxy:port", "https": "https://proxy:port"}
]

class ReconModule:
    """Performs advanced web reconnaissance with BlackVault and SwarmMesh integration."""
    
    def __init__(self, vault, swarm: 'SwarmMesh', redis: Redis, logger: logging.Logger = None):
        """
        Initializes the recon module with BlackVault and SwarmMesh.
        
        Args:
            vault: BlackVault instance for storing results.
            swarm: SwarmMesh instance for command distribution.
            redis: Redis instance for Pub/Sub.
            logger: Optional custom logger (defaults to 'ReconModule').
        """
        self.vault = vault
        self.swarm = swarm
        self.redis = redis
        self.logger = logger or logging.getLogger('ReconModule')
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BLACKROOT.Recon-Agent"})
        self.form_cache = self._load_form_cache()
        self.command_id = None

    def _save_form_cache(self):
        """Saves form cache to BlackVault."""
        try:
            self.vault.store("form_cache", pickle.dumps(self.form_cache))
            self.logger.info("Form cache saved to BlackVault")
        except Exception as e:
            self.logger.error(f"Failed to save form cache: {e}")

    def _load_form_cache(self) -> Dict:
        """Loads form cache from BlackVault."""
        try:
            data = self.vault.retrieve("form_cache")
            return pickle.loads(data)
        except Exception:
            self.logger.info("No form cache found, initializing new cache")
            return {}

    def _get_random_headers(self) -> Dict[str, str]:
        """Returns random headers with User-Agent."""
        return {"User-Agent": random.choice(USER_AGENTS)}

    def _get_random_proxy(self) -> Optional[Dict[str, str]]:
        """Returns a random proxy from the pool."""
        return random.choice(PROXY_POOL) if PROXY_POOL else None

    def _random_query_pad(self, url: str) -> str:
        """Adds random query parameters to evade caching."""
        pad = random.choice([
            f"?v={random.randint(1000, 9999)}",
            f"?cache={random.randint(100, 999)}",
            f"?token={random.randint(10000, 99999)}"
        ])
        return url + (pad.replace("?", "&", 1) if "?" in url else pad)

    def get_subdomains(self, domain: str, wordlist: Optional[List[str]] = None) -> List[str]:
        """
        Enumerates subdomains using brute-forcing, crt.sh, certspotter, and dnstwist.
        
        Args:
            domain: Target domain.
            wordlist: Optional list of subdomains to brute-force.
        
        Returns:
            List[str]: Discovered subdomains.
        """
        if not wordlist:
            wordlist = ["www", "mail", "dev", "test", "admin", "api", "staging", "beta", "portal", "vpn"]
        
        found = set()
        def check_subdomain(sub: str):
            try:
                socket.gethostbyname(f"{sub}.{domain}")
                with self.lock:
                    found.add(f"{sub}.{domain}")
            except Exception:
                pass

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(check_subdomain, wordlist)
            found.update(fetch_crtsh_subdomains(domain))
            found.update(fetch_certspotter_subdomains(domain))
            found.update(dnstwist_permutations(domain))
            
            self.vault.store(f"subdomains_{domain}_{self.command_id}", json.dumps(list(found)).encode())
            self.redis.publish(self.swarm.channel, json.dumps({
                "command_id": self.command_id,
                "type": "subdomains",
                "data": list(found)
            }))
            self.logger.info(f"Found {len(found)} subdomains for {domain}")
            return list(found)
        except Exception as e:
            self.logger.error(f"Subdomain enumeration failed for {domain}: {e}")
            self.vault.store(f"subdomain_error_{time.time()}", 
                            json.dumps({"domain": domain, "error": str(e), "command_id": self.command_id}).encode())
            return []

    def brute_force_paths(self, target: str, wordlist: Optional[List[str]] = None, threads: int = 10) -> List[str]:
        """
        Brute-forces paths on the target URL.
        
        Args:
            target: Target URL.
            wordlist: Optional list of paths to check.
            threads: Number of concurrent threads.
        
        Returns:
            List[str]: Discovered paths.
        """
        if not wordlist:
            wordlist = ["admin", "login", "dashboard", "api", "uploads", "backup", "config", 
                        ".git", ".env", "test", "dev", "staging", "private", "data", "db"]
        
        found = []
        base_len = None
        try:
            base = self.session.get(target, headers=self._get_random_headers(), timeout=8, 
                                  proxies=self._get_random_proxy(), verify=False)
            base_len = len(base.content)
        except Exception:
            self.logger.warning(f"Failed to get base content length for {target}")

        def check_path(path: str) -> Optional[str]:
            url = urljoin(target, f"/{path}")
            try:
                r = self.session.get(url, headers=self._get_random_headers(), timeout=8, 
                                   proxies=self._get_random_proxy(), verify=False)
                if base_len and abs(len(r.content) - base_len) < 32:
                    return None
                if r.status_code < 400:
                    self.logger.info(f"Found: {url} (status {r.status_code}, size {len(r.content)})")
                    return url
                return None
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            found = [r for r in executor.map(check_path, wordlist) if r]
        
        self.vault.store(f"paths_{target}_{self.command_id}", json.dumps(found).encode())
        self.redis.publish(self.swarm.channel, json.dumps({
            "command_id": self.command_id,
            "type": "paths",
            "data": found
        }))
        self.logger.info(f"Found {len(found)} paths for {target}")
        return found

    def harvest_and_inject_forms(self, endpoints: List[str], base_url: Optional[str] = None) -> tuple[Dict, Dict]:
        """
        Harvests forms and tests with safe payloads.
        
        Args:
            endpoints: List of URLs to check.
            base_url: Optional base URL for relative actions.
        
        Returns:
            tuple: (injection results, form map).
        """
        FORM_PAYLOADS = ["test_input", f"user_{random.randint(1000, 9999)}", "data"]
        results = {}
        
        with self.lock:
            for url in endpoints:
                try:
                    r = self.session.get(url, headers=self._get_random_headers(), timeout=8, 
                                       proxies=self._get_random_proxy())
                    soup = BeautifulSoup(r.text, "html.parser")
                    forms = [f for f in soup.find_all("form") if isinstance(f, Tag)]
                    for idx, form in enumerate(forms):
                        action = str(form.get("action") or url)
                        method = str(form.get("method", "get")).lower()
                        fields = {inp.get("name"): inp.get("type", "text") 
                                 for inp in form.find_all(["input", "textarea"]) 
                                 if isinstance(inp, Tag) and inp.get("name")}
                        form_id = f"{url}::form{idx}"
                        self.form_cache[form_id] = {"action": action, "method": method, "fields": fields}
                        inj_results = []
                        for payload in FORM_PAYLOADS:
                            data = {k: payload for k in fields}
                            try:
                                target_url = urljoin(base_url or url, action)
                                if method == "post":
                                    resp = self.session.post(target_url, data=data, 
                                                           headers=self._get_random_headers(), 
                                                           timeout=8, proxies=self._get_random_proxy())
                                else:
                                    resp = self.session.get(target_url, params=data, 
                                                          headers=self._get_random_headers(), 
                                                          timeout=8, proxies=self._get_random_proxy())
                                inj_results.append({
                                    "payload": payload, 
                                    "status": resp.status_code, 
                                    "error": resp.status_code >= 400 or any(e in resp.text.lower() 
                                                                          for e in ["error", "exception"])
                                })
                            except Exception as e:
                                inj_results.append({"payload": payload, "status": None, "error": True, 
                                                  "exception": str(e)})
                        results[form_id] = inj_results
                except Exception as e:
                    self.logger.error(f"Form harvest error at {url}: {e}")
                    self.vault.store(f"form_error_{time.time()}", 
                                    json.dumps({"url": url, "error": str(e), "command_id": self.command_id}).encode())
        
        self._save_form_cache()
        self.vault.store(f"form_results_{time.time()}_{self.command_id}", json.dumps(results).encode())
        self.redis.publish(self.swarm.channel, json.dumps({
            "command_id": self.command_id,
            "type": "forms",
            "data": results
        }))
        self.logger.info(f"Harvested {len(self.form_cache)} forms, tested {len(results)}")
        return results, self.form_cache

    def scan_for_xss(self, target: str, endpoints: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Scans for XSS vulnerabilities with safe payloads.
        
        Args:
            target: Target URL.
            endpoints: Optional list of endpoints to scan.
        
        Returns:
            Dict[str, bool]: XSS scan results.
        """
        XSS_PAYLOADS = ["<test>", f"user_{random.randint(1000, 9999)}", "data"]
        results = {}
        
        def scan_url(url: str) -> bool:
            try:
                r = self.session.get(url, headers=self._get_random_headers(), timeout=8, 
                                   proxies=self._get_random_proxy())
                for payload in XSS_PAYLOADS:
                    if payload in r.text:
                        self.logger.info(f"Potential XSS found at {url} with payload: {payload}")
                        return True
                return False
            except Exception as e:
                self.logger.error(f"XSS scan failed for {url}: {e}")
                return False

        endpoints = endpoints or [target]
        for url in endpoints:
            results[url] = scan_url(url)
        
        self.vault.store(f"xss_results_{time.time()}_{self.command_id}", json.dumps(results).encode())
        self.redis.publish(self.swarm.channel, json.dumps({
            "command_id": self.command_id,
            "type": "xss",
            "data": results
        }))
        self.logger.info(f"XSS scan completed for {len(endpoints)} endpoints")
        return results

    def parse_robots(self, content: str, base: str) -> List[str]:
        """Parses robots.txt for disallowed paths."""
        paths = []
        for line in content.splitlines():
            if line.startswith("Disallow:"):
                path = line.split(":", 1)[1].strip()
                paths.append(urljoin(base, path))
        return paths

    def parse_sitemap(self, content: str) -> List[str]:
        """Parses sitemap.xml for URLs."""
        return re.findall(r'<loc>(.*?)</loc>', content)

    def extract_emails(self, text: str) -> List[str]:
        """Extracts email addresses from text."""
        return re.findall(r'[\w\.-]+@[\w\.-]+', text)

    def extract_secrets(self, text: str) -> List[str]:
        """Extracts potential secrets (e.g., API keys)."""
        patterns = [r'AKIA[0-9A-Z]{16}', r'sk_live_[0-9a-zA-Z]{24,}', r'AIza[0-9A-Za-z\-_]{35}']
        return [s for pat in patterns for s in re.findall(pat, text)]

    def fingerprint_tech(self, headers: Dict, html: str) -> List[str]:
        """Identifies technologies from headers and HTML."""
        tech = set()
        if 'x-powered-by' in headers:
            tech.add(headers['x-powered-by'])
        if 'server' in headers:
            tech.add(headers['server'])
        if 'wp-content' in html or 'wp-includes' in html:
            tech.add('WordPress')
        if 'Drupal.settings' in html:
            tech.add('Drupal')
        if 'Set-Cookie' in headers and 'PHPSESSID' in headers.get('Set-Cookie', ''):
            tech.add('PHP')
        if 'csrfmiddlewaretoken' in html:
            tech.add('Django')
        if 'laravel_session' in html:
            tech.add('Laravel')
        return list(tech)

    def advanced_recon(self, target: str, max_depth: int = 3, scope_domains: Optional[List[str]] = None, 
                      brute_subdomains: bool = True, command_id: Optional[str] = None) -> Dict:
        """
        Performs advanced reconnaissance with BlackVault and SwarmMesh integration.
        
        Args:
            target: Target URL.
            max_depth: Maximum crawl depth.
            scope_domains: List of domains to scope the crawl.
            brute_subdomains: Enable subdomain brute-forcing.
            command_id: Optional command ID from C2.
        
        Returns:
            Dict: Recon report.
        """
        self.command_id = command_id or get_random_bytes(16).hex()
        visited = set()
        endpoints = set()
        forms = []
        js_endpoints = set()
        emails = set()
        secrets = set()
        tech = set()
        subdomains = set()
        report = {}

        if not scope_domains:
            parsed = urlparse(target)
            scope_domains = [parsed.netloc]

        if brute_subdomains:
            self.logger.info("Enumerating subdomains...")
            for dom in scope_domains:
                subdomains.update(self.get_subdomains(dom))
            report['subdomains'] = list(subdomains)

        self.logger.info("Brute-forcing common directories...")
        report['brute_dirs'] = self.brute_force_paths(target)

        def crawl(url: str, depth: int):
            if url in visited or depth > max_depth:
                return
            url_padded = self._random_query_pad(url)
            headers = self._get_random_headers()
            with self.lock:
                visited.add(url)
            backoff = 0
            try:
                start = time.time()
                r = self.session.get(url_padded, headers=headers, timeout=8, 
                                   proxies=self._get_random_proxy(), verify=False)
                elapsed = time.time() - start
                if r.status_code in (403, 503) or elapsed > 10:
                    self.logger.warning(f"WAF anomaly: {url} status={r.status_code} delay={elapsed:.1f}s")
                    backoff = random.uniform(2, 10)
                if r.headers.get('Retry-After'):
                    backoff = max(backoff, float(r.headers.get('Retry-After')))
                    self.logger.info(f"Rate limit detected. Backing off for {backoff:.1f}s")
                elif r.headers.get('X-RateLimit-Reset'):
                    reset_time = float(r.headers.get('X-RateLimit-Reset'))
                    backoff = max(backoff, max(0, reset_time - time.time()))
                    self.logger.info(f"Rate limit detected. Backing off for {backoff:.1f}s")
                if backoff > 0:
                    time.sleep(backoff + random.uniform(0.5, 2.5))
                
                soup = BeautifulSoup(r.text, "html.parser")
                html = r.text
                endpoints.add(url)
                emails.update(self.extract_emails(html))
                secrets.update(self.extract_secrets(html))
                tech.update(self.fingerprint_tech(r.headers, html))
                
                if url.endswith("robots.txt"):
                    for p in self.parse_robots(html, url):
                        crawl(p, depth + 1)
                if url.endswith("sitemap.xml"):
                    for loc in self.parse_sitemap(html):
                        crawl(loc, depth + 1)
                
                for form in soup.find_all("form"):
                    if not isinstance(form, Tag):
                        continue
                    action = form.get("action") or url
                    method = str(form.get("method", "get")).lower()
                    inputs = [inp for inp in form.find_all("input") if isinstance(inp, Tag)]
                    forms.append({
                        'url': url,
                        'action': action,
                        'method': method,
                        'inputs': [inp.get('name') for inp in inputs if inp.get('name')]
                    })
                
                for script in soup.find_all("script"):
                    if isinstance(script, Tag):
                        src = script.get("src")
                        if src and isinstance(src, str) and src.startswith("http"):
                            js_endpoints.add(src)
                        if script.string:
                            js_endpoints.update(re.findall(r'https?://[\w\./\-_%]+', script.string))
                
                for link in soup.find_all("a"):
                    if not isinstance(link, Tag):
                        continue
                    href = link.get("href")
                    if href:
                        absolute = urljoin(url, str(href))
                        if any(domain in absolute for domain in scope_domains):
                            crawl(absolute, depth + 1)
            except Exception as e:
                self.logger.error(f"Crawl failed for {url}: {e}")
                self.vault.store(f"crawl_error_{time.time()}", 
                                json.dumps({"url": url, "error": str(e), "command_id": self.command_id}).encode())

        for path in ["admin", "login", "dashboard", "api", "robots.txt", "sitemap.xml"]:
            crawl(urljoin(target, path), 1)
        crawl(target, 1)

        form_results, form_map = self.harvest_and_inject_forms(list(endpoints), target)
        xss_results = self.scan_for_xss(target, list(endpoints))
        
        report['endpoints'] = list(endpoints)
        report['forms'] = forms
        report['js_endpoints'] = list(js_endpoints)
        report['emails'] = list(emails)
        report['secrets'] = list(secrets)
        report['technologies'] = list(tech)
        report['form_results'] = form_results
        report['xss_results'] = xss_results
        report['command_id'] = self.command_id
        
        try:
            self.vault.store(f"recon_{target}_{self.command_id}", json.dumps(report).encode())
            self.redis.publish(self.swarm.channel, json.dumps({
                "command_id": self.command_id,
                "type": "recon_report",
                "data": report
            }))
            self.logger.info(f"Recon complete: {len(endpoints)} endpoints, {len(forms)} forms, {len(js_endpoints)} JS endpoints")
        except Exception as e:
            self.logger.error(f"Failed to store recon report: {e}")
        
        return report

def fetch_crtsh_subdomains(domain: str) -> List[str]:
    """Fetches subdomains from crt.sh."""
    try:
        r = requests.get(f"https://crt.sh/?q=%25.{domain}&output=json", timeout=10)
        if r.status_code == 200:
            return list({sub.strip() for entry in r.json() 
                        for sub in entry.get('name_value', '').split('\n') 
                        if sub.endswith(domain)})
        return []
    except Exception as e:
        logging.getLogger('ReconModule').error(f"crt.sh error: {e}")
        return []

def fetch_certspotter_subdomains(domain: str) -> List[str]:
    """Fetches subdomains from certspotter."""
    try:
        r = requests.get(f"https://api.certspotter.com/v1/issuances?domain={domain}&include_subdomains=true&expand=dns_names", 
                        timeout=10)
        if r.status_code == 200:
            return list({sub.strip() for entry in r.json() 
                        for sub in entry.get('dns_names', []) if sub.endswith(domain)})
        return []
    except Exception as e:
        logging.getLogger('ReconModule').error(f"certspotter error: {e}")
        return []

def dnstwist_permutations(domain: str) -> List[str]:
    """Generates domain permutations using dnstwist."""
    try:
        fuzzer = dnstwist.Fuzzer(domain)
        fuzzer.generate()
        return [x['domain'] for x in fuzzer.domains]
    except Exception as e:
        logging.getLogger('ReconModule').error(f"dnstwist error: {e}")
        return []