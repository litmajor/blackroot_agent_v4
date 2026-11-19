# passive.py
import os
import logging
import requests
from typing import List, Dict, Any
from requests.adapters import HTTPAdapter, Retry

# waybackpy has an unstable API across versions; guard tightly
try:
    # waybackpy >= 3.x uses CDXApi / availability API
    from waybackpy import WaybackMachineCDXServerAPI as _CDXApi  # type: ignore
    _WAYBACK_MODE = "cdx"
except ImportError:
    try:
        # waybackpy 2.x used Url() with .snapshots()
        from waybackpy import Url as _WaybackUrl  # type: ignore
        _WAYBACK_MODE = "legacy"
    except ImportError:
        _CDXApi = None  # type: ignore
        _WaybackUrl = None  # type: ignore
        _WAYBACK_MODE = "unavailable"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOGGER = logging.getLogger("Passive")
LOGGER.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Shodan setup
# ---------------------------------------------------------------------
SHODAN_API_KEY = os.getenv("SHODAN_API_KEY", "")

# Reusable requests session with retry logic
_session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=retries))


def shodan_lookup(domain: str) -> List[Dict[str, Any]]:
    """
    Query Shodan for open ports and banners related to a given domain.

    Returns:
        List of match dictionaries from Shodan API (may be empty if no results).
    """
    if not SHODAN_API_KEY:
        LOGGER.debug("SHODAN_API_KEY not set → skipping Shodan lookup")
        return []

    try:
        resp = _session.get(
            "https://api.shodan.io/shodan/host/search",
            params={"key": SHODAN_API_KEY, "query": f"hostname:{domain}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        LOGGER.info("Shodan returned %d results for %s", len(matches), domain)
        return matches
    except requests.exceptions.RequestException as exc:
        LOGGER.warning("Shodan network error for %s: %s", domain, exc)
        return []
    except ValueError as exc:  # JSON decoding
        LOGGER.warning("Shodan response parse error: %s", exc)
        return []


def wayback_urls(domain: str, limit: int = 500, dedupe: bool = True) -> List[str]:
    """
    Retrieve historical archive URLs for a given domain from the Wayback Machine.

    Uses CDX Server API (waybackpy >= 3.x) with graceful fallback to legacy
    Url().snapshots() (waybackpy 2.x) and finally a direct CDX HTTP call when
    the waybackpy package is unavailable entirely.

    Args:
        domain: target domain (e.g., 'example.com')
        limit: maximum number of URLs to return
        dedupe: remove duplicates if True

    Returns:
        List of archived URLs (strings).
    """
    urls: List[str] = []

    if _WAYBACK_MODE == "unavailable":
        # Direct CDX API call — no waybackpy needed
        try:
            resp = _session.get(
                "http://web.archive.org/cdx/search/cdx",
                params={"url": f"*.{domain}", "output": "text", "fl": "original",
                        "collapse": "urlkey", "limit": limit},
                timeout=15,
            )
            resp.raise_for_status()
            raw = [line.strip() for line in resp.text.splitlines() if line.strip()]
            urls = raw[:limit]
        except Exception as exc:
            LOGGER.warning("Wayback CDX HTTP fallback error for %s: %s", domain, exc)
            return []

    elif _WAYBACK_MODE == "cdx":
        try:
            cdx = _CDXApi(domain, user_agent="Mozilla/5.0")
            for i, snap in enumerate(cdx.snapshots()):
                if i >= limit:
                    break
                url = getattr(snap, "archive_url", None) or getattr(snap, "original", None)
                if url:
                    urls.append(str(url).strip())
        except Exception as exc:
            LOGGER.warning("Wayback CDX error for %s: %s", domain, exc)
            return []

    else:  # legacy
        try:
            api = _WaybackUrl(domain, user_agent="Mozilla/5.0")
            for i, snapshot in enumerate(api.snapshots()):
                if i >= limit:
                    break
                url = getattr(snapshot, "archive_url", None)
                if url:
                    urls.append(url.strip())
        except Exception as exc:
            LOGGER.warning("Wayback legacy error for %s: %s", domain, exc)
            return []

    if dedupe:
        urls = list(dict.fromkeys(urls))

    LOGGER.info("Wayback returned %d URLs for %s", len(urls), domain)
    return urls