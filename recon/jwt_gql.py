# jwt_gql.py
import json, re, jwt, logging, requests
from typing import Dict, Any, List, Optional

LOG = logging.getLogger("JWT_GQL")

# --- JWT helpers ----------------------------------------------------
TOP_1000 = [line.strip() for line in open("wordlists/rockyou-1000.txt")][:1000] \
           if open("wordlists/rockyou-1000.txt") else \
           ["secret", "123456", "password", "jwt", "admin"]

def _extract_tokens(text: str) -> List[str]:
    # JWT regex (header.payload.signature)
    return re.findall(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', text)

def _weak_secret(token: str) -> Optional[str]:
    for secret in TOP_1000:
        try:
            jwt.decode(token, secret, algorithms=["HS256"])
            return secret
        except Exception:
            continue
    return None

def jwt_audit(responses: List[str]) -> Dict[str, Any]:
    findings: Dict[str, Any] = {"tokens": [], "issues": []}
    for resp in responses:
        for tok in _extract_tokens(resp):
            try:
                h = jwt.get_unverified_header(tok)
                p = jwt.decode(tok, options={"verify_signature": False})
                issues = []
                if h.get("alg") == "none":
                    issues.append("alg-none")
                secret = _weak_secret(tok)
                if secret:
                    issues.append(f"weak-secret:{secret}")
                findings["tokens"].append({"header": h, "payload": p})
                if issues:
                    findings["issues"].append({"token": tok, "issues": issues})
            except Exception:
                continue
    return findings

# --- GraphQL helpers -----------------------------------------------
INTRO_QUERY = """
{ __schema { types { name fields { name } } mutationType { name fields { name } } } }
"""

def gql_introspection(base: str) -> Dict[str, Any]:
    paths = ["/graphql", "/api/graphql", "/v1/graphql", "/graph"]
    for p in paths:
        url = base.rstrip("/") + p
        try:
            r = requests.post(
                url,
                json={"query": INTRO_QUERY},
                headers={"Content-Type": "application/json"},
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json().get("data", {})
                dangerous = [
                    f["name"]
                    for f in data.get("mutationType", {}).get("fields", [])
                    if any(k in f["name"].lower() for k in ["delete", "drop", "reset", "exec"])
                ]
                return {"url": url, "schema": data, "dangerous_mutations": dangerous}
        except Exception as e:
            LOG.debug("GraphQL probe %s failed: %s", url, e)
    return {}