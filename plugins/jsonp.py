import requests, re
def scan(base):
    found = []
    try:
        r = requests.get(base + "/api?callback=test", timeout=5)
        if re.search(r'test\([^)]+\)', r.text):
            found.append(base + "/api?callback=<script>alert(1)</script>")
    except: pass
    return found