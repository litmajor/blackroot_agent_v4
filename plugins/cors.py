import requests
def scan(base):
    found = []
    try:
        r = requests.get(base + "/user", headers={"Origin": "evil.com"}, timeout=5)
        if "evil.com" in r.headers.get("Access-Control-Allow-Origin", ""):
            found.append(f"{base}/user  [CORS wildcard]")
    except: pass
    return found