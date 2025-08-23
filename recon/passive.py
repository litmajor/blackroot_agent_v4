# passive.py
import os
import logging
import requests
from typing import List, Dict, Any
from waybackpy import Url
from requests.adapters import HTTPAdapter, Retry


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

    Args:
        domain: target domain (e.g., 'example.com')
        limit: maximum number of URLs to return
        dedupe: remove duplicates if True

    Returns:
        List of archived URLs (strings).
    """
    urls: List[str] = []
    try:
        api = Url(domain, user_agent="Mozilla/5.0")
        snapshots = api.snapshots()  # generator of snapshot objects

        for i, snapshot in enumerate(snapshots):
            if i >= limit:
                break

            # Each snapshot has `.archive_url` (archived link) and `.timestamp`
            url = getattr(snapshot, "archive_url", None)
            if url:
                urls.append(url.strip())

        if dedupe:
            # preserve order while deduping
            urls = list(dict.fromkeys(urls))

        LOGGER.info("Wayback returned %d URLs for %s", len(urls), domain)
        return urls

    except Exception as exc:
        LOGGER.warning("Wayback error for %s: %s", domain, exc)
        return []