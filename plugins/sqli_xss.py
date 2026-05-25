import requests
def scan(base):
    payloads = ["/search?q='\"><script>alert(1)</script>"]
    found = []
    for p in payloads:
        try:
            r = requests.get(base + p, timeout=5)
            if "<script>" in r.text and ("mysql" in r.text.lower() or "syntax" in r.text.lower()):
                found.append(base + p)
        except: pass
    return found