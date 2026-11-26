"""
tests/test_target_identity.py
=============================
Unit & integration tests for ArtifactValuationTable, TargetIdentity,
BlackVault -> TargetIdentity hook, BEL credit, and CausalGraph prior updates.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recon.target_identity import (
    ArtifactType,
    ArtifactValuationTable,
    TargetIdentity,
    TargetIdentityRegistry,
)


def test_artifact_classification_and_pricing():
    table = ArtifactValuationTable()

    assert table.classify("corp-dc-01:metamask_seed") == ArtifactType.WALLET_SEED
    assert table.classify("aws_iam_sts_token") == ArtifactType.CLOUD_IAM_TOKEN
    assert table.classify("id_ed25519_prod") == ArtifactType.SSH_PRIVATE_KEY
    assert table.classify("chrome_passwords.db") == ArtifactType.BROWSER_PASSWORD
    assert table.classify("custom", metadata={"artifact_type": "db_credential"}) == ArtifactType.DB_CREDENTIAL

    # Base wallet value > cloud > ssh > browser
    w_base = table.price_artifact(ArtifactType.WALLET_SEED)
    c_base = table.price_artifact(ArtifactType.CLOUD_IAM_TOKEN)
    s_base = table.price_artifact(ArtifactType.SSH_PRIVATE_KEY)
    b_base = table.price_artifact(ArtifactType.BROWSER_PASSWORD)
    assert w_base > c_base > s_base > b_base

    # Admin + domain_controller multiplier increases ROOT value
    dc_admin = TargetIdentity(
        target_id="10.0.0.5",
        privilege_level="admin",
        domain_role="domain_controller",
    )
    assert table.price_artifact(ArtifactType.WALLET_SEED, dc_admin) > w_base


def test_target_identity_weighting_and_ranking():
    reg = TargetIdentityRegistry()

    # Low-value workstation with heavy EDR
    reg.ingest_extract(
        target_id="ws-01",
        artifact_name="email_imap_creds",
        metadata={"privilege_level": "user", "domain_role": "workstation", "edr_products": ["CrowdStrike", "Defender"]},
    )

    # High-value cloud VM with wallet + cloud token + admin
    reg.ingest_extract(
        target_id="cloud-01",
        artifact_name="metamask_vault_seed",
        metadata={"privilege_level": "admin", "domain_role": "cloud_vm"},
    )
    reg.ingest_extract(
        target_id="cloud-01",
        artifact_name="aws_iam_root_key",
    )

    ranked = reg.ranked_targets()
    assert [t.target_id for t in ranked] == ["cloud-01", "ws-01"]
    cloud = reg.get("cloud-01")
    assert cloud is not None
    assert cloud.has_wallets is True
    assert cloud.has_cloud_tokens is True
    assert cloud.total_extracted_root_value > 500.0


def test_causal_graph_prior_updates_on_extract():
    from causal_graph import create_credential_extraction_graph

    graph = create_credential_extraction_graph()
    before = graph.query_effects("extract_chrome")["data_value"]

    reg = TargetIdentityRegistry(causal_graph=graph)
    # Ingest a chrome extract; EMA blends toward 0.58 from 0.70
    reg.ingest_extract(
        target_id="192.168.1.20",
        artifact_name="chrome_login_data",
    )
    after = graph.query_effects("extract_chrome")["data_value"]
    assert after != before
    assert 0.0 < after <= 1.0


@pytest.mark.asyncio
async def test_ingest_and_credit_bel():
    ledger = MagicMock()
    ledger.credit = AsyncMock()

    reg = TargetIdentityRegistry()
    evt = await reg.ingest_and_credit_bel(
        ledger=ledger,
        target_id="10.10.10.5",
        artifact_name="id_rsa_deploy",
        metadata={"privilege_level": "admin", "domain_role": "member_server"},
        agent_id="SCOUT-01",
    )

    assert evt.artifact_type == "ssh_private_key"
    assert evt.root_value > 120.0
    ledger.credit.assert_awaited_once_with(
        "SCOUT-01",
        evt.root_value,
        "vault_extract:ssh_private_key:10.10.10.5",
    )


def test_blackvault_store_updates_target_identity():
    from black_vault import BlackVault

    vault = BlackVault()
    reg = TargetIdentityRegistry.instance()

    vault.store(
        "10.20.30.40:metamask_seed",
        b"seed-phrase-bytes",
        metadata={
            "target_id": "10.20.30.40",
            "privilege_level": "admin",
            "domain_role": "workstation",
            "agent_id": "INFILTRATOR-01",
        },
    )

    ident = reg.get("10.20.30.40")
    assert ident is not None
    assert ident.has_wallets is True
    assert ident.artifact_counts.get("wallet_seed", 0) >= 1
