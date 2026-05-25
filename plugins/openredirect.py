import requests, re
def scan(base):
    redirs = ["/go?url=", "/out?link=", "/redirect?to="]
    found = []
    for r in redirs:
        probe = base + r + "https://evil.com"
        try:
            resp = requests.get(probe, allow_redirects=False, timeout=5)
            if "evil.com" in resp.headers.get("Location", ""):
                found.append(base + r + "<script>alert(1)</script>")
        except: pass
    return found