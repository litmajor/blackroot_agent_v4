"""
tests/test_ghost_layer_financial.py
====================================
Unit tests for ghost_layer_financial.py Phase 5 collectors
and new ArtifactType entries in ArtifactValuationTable.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ghost_layer_financial import (
    _collect_browser_web_storage,
    _collect_electron_sessions,
    _collect_erp_artifacts,
    _collect_banking_files,
    _collect_web3_developer_secrets,
    _collect_payment_gateway_keys,
    _collect_hw_wallet_artifacts,
    _collect_financial_documents,
    run_phase5_financial,
)
from recon.target_identity import ArtifactType, ArtifactValuationTable


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_tree(base: Path, paths: list[str]) -> None:
    for rel in paths:
        full = base / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(b"test")


# ── 1. ArtifactValuationTable coverage ────────────────────────────────────────

def test_phase5_types_in_valuation_table():
    table = ArtifactValuationTable()
    phase5_types = [
        ArtifactType.AZURE_PRT_CACHE,
        ArtifactType.BROWSER_WEB_STORAGE,
        ArtifactType.ELECTRON_SESSION_TOKEN,
        ArtifactType.WEB3_DEVELOPER_SECRET,
        ArtifactType.PAYMENT_GATEWAY_KEY,
        ArtifactType.BANKING_BATCH_FILE,
        ArtifactType.ERP_ACCOUNTING_CONFIG,
        ArtifactType.FINANCIAL_DOCUMENT,
    ]
    for at in phase5_types:
        spec = table.get_spec(at)
        assert spec.base_root_value > 0
        assert spec.sensitivity_tier in ("critical", "high", "medium", "low")

    # Ordering: Azure PRT > payment gateway key > banking file > ERP > financial doc
    assert table.get_spec(ArtifactType.AZURE_PRT_CACHE).base_root_value > \
           table.get_spec(ArtifactType.ERP_ACCOUNTING_CONFIG).base_root_value
    assert table.get_spec(ArtifactType.PAYMENT_GATEWAY_KEY).base_root_value > \
           table.get_spec(ArtifactType.FINANCIAL_DOCUMENT).base_root_value


def test_classify_phase5_types():
    table = ArtifactValuationTable()
    cases = [
        ("azure_prt_cache_data", ArtifactType.AZURE_PRT_CACHE),
        ("chrome_indexeddb_stripe_folder", ArtifactType.BROWSER_WEB_STORAGE),
        ("electron_session_slack", ArtifactType.ELECTRON_SESSION_TOKEN),
        ("solana/id.json_backup", ArtifactType.WEB3_DEVELOPER_SECRET),
        ("sk_live_key_config", ArtifactType.PAYMENT_GATEWAY_KEY),
        ("payroll_nacha_export.ach", ArtifactType.BANKING_BATCH_FILE),
        ("saplogon_ini_backup", ArtifactType.ERP_ACCOUNTING_CONFIG),
        ("wire_instruction_vendor.pdf", ArtifactType.FINANCIAL_DOCUMENT),
        ("ledger_live_app_json", ArtifactType.WALLET_SEED),
    ]
    for name, expected in cases:
        result = table.classify(name)
        assert result == expected, f"classify({name!r}) = {result!r}, expected {expected!r}"


# ── 2. ERP collector with synthetic files ─────────────────────────────────────

def test_erp_collector_finds_quickbooks(tmp_path, monkeypatch):
    # Create a fake QB company file
    qb_root = tmp_path / "QuickBooks"
    qb_root.mkdir()
    (qb_root / "company.qbw").write_bytes(b"QBW")
    (qb_root / "backup.qbb").write_bytes(b"QBB")

    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_QUICKBOOKS_ROOTS", [qb_root])
    results = glf._collect_erp_artifacts()
    extensions = {r["extension"] for r in results}
    assert ".qbw" in extensions
    assert ".qbb" in extensions


def test_erp_collector_finds_sap(tmp_path, monkeypatch):
    sap_file = tmp_path / "saplogon.ini"
    sap_file.write_bytes(b"[SAPLogon]")
    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_SAP_PATHS", [sap_file])
    results = glf._collect_erp_artifacts()
    assert any("saplogon" in r["file"].lower() for r in results)


# ── 3. Banking file collector ─────────────────────────────────────────────────

def test_banking_collector_finds_ach_files(tmp_path, monkeypatch):
    dl = tmp_path / "Downloads"
    dl.mkdir()
    (dl / "payroll_export.ach").write_bytes(b"1 ACH DATA")
    (dl / "wire_transfer_march.csv").write_bytes(b"routing,amount")
    (dl / "unrelated.txt").write_bytes(b"nothing")

    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_BANKING_SEARCH_ROOTS", [dl])
    results = glf._collect_banking_files()
    filenames = [Path(r["file"]).name for r in results]
    assert "payroll_export.ach" in filenames
    assert "wire_transfer_march.csv" in filenames
    assert "unrelated.txt" not in filenames


def test_banking_collector_detects_sepa_xml(tmp_path, monkeypatch):
    dl = tmp_path / "Downloads"
    dl.mkdir()
    sepa_xml = dl / "payment.xml"
    sepa_xml.write_text(
        '<?xml version="1.0"?><Document><CdtTrfTxInf><IBAN>DE89370400440532013000</IBAN></CdtTrfTxInf></Document>',
        encoding="utf-8"
    )
    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_BANKING_SEARCH_ROOTS", [dl])
    results = glf._collect_banking_files()
    assert any("payment.xml" in r["file"] for r in results)


# ── 4. Web3 developer secrets ─────────────────────────────────────────────────

def test_web3_collector_finds_env_with_infura(tmp_path, monkeypatch):
    repo = tmp_path / "myproject"
    repo.mkdir()
    env_file = repo / ".env"
    env_file.write_text("INFURA_API_KEY=abc123def456abc123def456abc123de\nFOO=bar", encoding="utf-8")

    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_ENV_FILE_SEARCH_ROOTS", [tmp_path])
    results = glf._collect_web3_developer_secrets()
    assert any(".env" in r["file"] for r in results)
    assert all(r["artifact_type"] == "web3_developer_secret" for r in results)


def test_web3_collector_finds_solana_keypair(tmp_path, monkeypatch):
    sol_key = tmp_path / "id.json"
    sol_key.write_text(json.dumps(list(range(64))), encoding="utf-8")
    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_SOLANA_KEYPAIR", sol_key)
    monkeypatch.setattr(glf, "_ENV_FILE_SEARCH_ROOTS", [])
    monkeypatch.setattr(glf, "_FOUNDRY_KEYSTORES", tmp_path / "nonexistent_foundry")
    results = glf._collect_web3_developer_secrets()
    assert any("Solana" in r["source"] for r in results)


# ── 5. Payment gateway key scanner ────────────────────────────────────────────

def test_payment_gateway_finds_stripe_live_key(tmp_path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    dummy_key = "".join(["sk_", "live_", "dummytoken1234567890abcdef"])
    (src / "config.py").write_text(f'STRIPE_SECRET = "{dummy_key}"', encoding="utf-8")

    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_SOURCE_SEARCH_ROOTS", [tmp_path])
    results = glf._collect_payment_gateway_keys()
    assert any(r["key_type"] == "stripe_secret" for r in results)


# ── 6. Financial document scanner ─────────────────────────────────────────────

def test_financial_docs_found(tmp_path, monkeypatch):
    dl = tmp_path / "Downloads"
    dl.mkdir()
    (dl / "wire_instruction_vendor_A.pdf").write_bytes(b"%PDF")
    (dl / "cap_table_2026.xlsx").write_bytes(b"PK")
    (dl / "meeting_notes.docx").write_bytes(b"PK")

    import ghost_layer_financial as glf
    monkeypatch.setattr(glf, "_DOC_SEARCH_ROOTS", [dl])
    results = glf._collect_financial_documents()
    filenames = [r["filename"] for r in results]
    assert "wire_instruction_vendor_A.pdf" in filenames
    assert "cap_table_2026.xlsx" in filenames
    assert "meeting_notes.docx" not in filenames


# ── 7. run_phase5_financial smoke test ────────────────────────────────────────

def test_run_phase5_financial_no_vault(monkeypatch):
    # Patch all search roots to empty tmp dirs so the scan completes instantly
    import ghost_layer_financial as glf
    from pathlib import Path
    import tempfile
    empty = Path(tempfile.mkdtemp())
    monkeypatch.setattr(glf, "_QUICKBOOKS_ROOTS", [empty])
    monkeypatch.setattr(glf, "_SAP_PATHS", [])
    monkeypatch.setattr(glf, "_BANKING_SEARCH_ROOTS", [empty])
    monkeypatch.setattr(glf, "_ENV_FILE_SEARCH_ROOTS", [empty])
    monkeypatch.setattr(glf, "_SOURCE_SEARCH_ROOTS", [empty])
    monkeypatch.setattr(glf, "_DOC_SEARCH_ROOTS", [empty])
    monkeypatch.setattr(glf, "_WAM_PATHS", [empty])
    monkeypatch.setattr(glf, "BROWSER_PROFILE_ROOTS", [empty])
    monkeypatch.setattr(glf, "ELECTRON_APPS", {})
    monkeypatch.setattr(glf, "_HW_WALLET_PATHS", {})
    monkeypatch.setattr(glf, "_SOLANA_KEYPAIR", empty / "nonexistent.json")
    monkeypatch.setattr(glf, "_FOUNDRY_KEYSTORES", empty / "nonexistent")

    result = run_phase5_financial(vault=None)
    for key in ("browser_web_storage", "electron_sessions", "azure_prt",
                "erp_artifacts", "banking_files", "web3_secrets",
                "payment_gateway_keys", "hw_wallets", "financial_documents"):
        assert key in result
        assert isinstance(result[key], list)


# ── 8. TargetIdentity records phase5 artifact types ──────────────────────────

def test_target_identity_records_azure_prt():
    from recon.target_identity import TargetIdentityRegistry
    reg = TargetIdentityRegistry()
    evt = reg.ingest_extract(
        target_id="corp-dc-02",
        artifact_name="azure_prt_cache_msal",
        metadata={"privilege_level": "admin", "domain_role": "domain_controller"},
    )
    assert evt.artifact_type == ArtifactType.AZURE_PRT_CACHE.value
    assert evt.root_value > 300.0  # admin + DC multipliers push it above base
    ident = reg.get("corp-dc-02")
    assert ident is not None
    assert ident.has_cloud_tokens is True


def test_target_identity_records_web3_dev_secret():
    from recon.target_identity import TargetIdentityRegistry
    reg = TargetIdentityRegistry()
    evt = reg.ingest_extract(
        target_id="dev-ws-01",
        artifact_name="solana/id.json_backup",
    )
    assert evt.artifact_type == ArtifactType.WEB3_DEVELOPER_SECRET.value
    ident = reg.get("dev-ws-01")
    assert ident is not None
    assert ident.has_wallets is True
