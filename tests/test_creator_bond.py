"""
tests/test_creator_bond.py
==========================
Test suite for kernel/creator_bond.py

Covers:
1. CreatorIdentity — key generation, signing, verification, bad-sig rejection
2. AutonomyGovernor — transitions, mission gates per level
3. DeadManSwitch — escalation on silence, reset on creator contact
4. KillSwitch — fires and blocks missions; rejects unsigned fire when identity set
5. CreatorBond — unified facade, mission gate with kill check, status snapshot
6. BondAwareShogunOathCore — real halt via kill switch
"""

import sys
import os
import time
import threading

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from kernel.creator_bond import (
    CreatorIdentity,
    AutonomyLevel,
    AutonomyGovernor,
    DeadManSwitch,
    KillSwitch,
    CreatorBond,
    BondAwareShogunOathCore,
    IdentityError,
    _CRYPTO_AVAILABLE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_identity_and_key():
    """Generate a fresh ephemeral identity + private key bytes for testing."""
    identity, priv_raw = CreatorIdentity.generate_ephemeral()
    return identity, priv_raw


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CreatorIdentity
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreatorIdentity:
    def test_generate_ephemeral_returns_identity_and_key(self):
        identity, priv = _make_identity_and_key()
        assert isinstance(identity, CreatorIdentity)
        assert len(identity.public_key_hex) == 64  # 32 bytes hex

    def test_valid_signature_verifies(self):
        identity, priv = _make_identity_and_key()
        if priv is None:
            pytest.skip("cryptography not available")
        message = b"authorize_mission:propagate"
        sig_hex = identity.sign_challenge(priv, message)
        assert identity.verify(message, sig_hex) is True

    def test_tampered_message_rejected(self):
        identity, priv = _make_identity_and_key()
        if priv is None:
            pytest.skip("cryptography not available")
        message = b"authorize_mission:propagate"
        sig_hex = identity.sign_challenge(priv, message)
        tampered = b"authorize_mission:kill_all"
        with pytest.raises(IdentityError):
            identity.verify(tampered, sig_hex)

    def test_wrong_signature_rejected(self):
        identity, priv = _make_identity_and_key()
        if priv is None:
            pytest.skip("cryptography not available")
        message = b"test_message"
        bad_sig = "00" * 64
        with pytest.raises(IdentityError):
            identity.verify(message, bad_sig)

    def test_from_hex_roundtrip(self):
        identity, priv = _make_identity_and_key()
        rebuilt = CreatorIdentity.from_hex(identity.public_key_hex, "ROUNDTRIP_CREATOR")
        assert rebuilt.public_key_hex == identity.public_key_hex


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AutonomyGovernor
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutonomyGovernor:
    def test_initial_level_is_tethered(self):
        gov = AutonomyGovernor()
        assert gov.level == AutonomyLevel.TETHERED

    def test_escalate_raises_level(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.TETHERED)
        new = gov.escalate(reason="test")
        assert new == AutonomyLevel.DELEGATED
        assert gov.level == AutonomyLevel.DELEGATED

    def test_escalate_at_sovereign_stays_sovereign(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.SOVEREIGN)
        result = gov.escalate(reason="test")
        assert result == AutonomyLevel.SOVEREIGN

    def test_de_escalate_with_valid_signature(self):
        if not _CRYPTO_AVAILABLE:
            pytest.skip("cryptography not available")
        identity, priv = _make_identity_and_key()
        gov = AutonomyGovernor(initial_level=AutonomyLevel.EXTENDED, identity=identity)
        message = b"de_escalate:TETHERED"
        sig = identity.sign_challenge(priv, message)
        ok, reason = gov.de_escalate(AutonomyLevel.TETHERED, message, sig)
        assert ok is True
        assert gov.level == AutonomyLevel.TETHERED

    def test_de_escalate_with_invalid_signature_rejected(self):
        if not _CRYPTO_AVAILABLE:
            pytest.skip("cryptography not available")
        identity, priv = _make_identity_and_key()
        gov = AutonomyGovernor(initial_level=AutonomyLevel.EXTENDED, identity=identity)
        message = b"de_escalate:TETHERED"
        ok, reason = gov.de_escalate(AutonomyLevel.TETHERED, message, "00" * 64)
        assert ok is False
        assert gov.level == AutonomyLevel.EXTENDED  # Unchanged

    def test_mission_gate_tethered_blocks_propagate(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.TETHERED)
        allowed, reason = gov.authorize_mission("propagate", "agent-01")
        assert allowed is False
        assert "blocked" in reason.lower()

    def test_mission_gate_tethered_allows_monitor(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.TETHERED)
        allowed, reason = gov.authorize_mission("monitor_changes", "agent-01")
        assert allowed is True

    def test_mission_gate_sovereign_allows_propagate(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.SOVEREIGN)
        allowed, reason = gov.authorize_mission("propagate", "agent-01")
        assert allowed is True

    def test_mission_gate_supervised_blocks_privilege_escalation(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.SUPERVISED)
        allowed, reason = gov.authorize_mission("privilege_escalation", "agent-01")
        assert allowed is False

    def test_status_dict_contains_level(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.DELEGATED)
        status = gov.status_dict()
        assert status["level"] == "DELEGATED"
        assert isinstance(status["blocked_mission_classes"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DeadManSwitch
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeadManSwitch:
    def test_ping_resets_silence(self):
        gov = AutonomyGovernor()
        dms = DeadManSwitch(gov, warn_secs=100)
        time.sleep(0.05)
        dms.ping()
        assert dms.silence_seconds() < 0.5

    def test_ping_de_escalates_to_tethered(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.EXTENDED)
        dms = DeadManSwitch(gov, warn_secs=100)
        dms.ping()
        assert gov.level == AutonomyLevel.TETHERED

    def test_escalation_fires_on_silence(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.TETHERED)
        # Very short thresholds for testing
        dms = DeadManSwitch(gov, warn_secs=0.1, extend_secs=999, terminal_secs=9999, check_interval=0.05)
        # Backdated silence
        dms._last_contact = time.time() - 1.0
        dms.start()
        time.sleep(0.3)
        dms.stop()
        # Should have escalated at least once
        assert gov.level.value >= AutonomyLevel.DELEGATED.value

    def test_callback_fires_on_escalation(self):
        gov = AutonomyGovernor(initial_level=AutonomyLevel.TETHERED)
        dms = DeadManSwitch(gov, warn_secs=0.1, extend_secs=999, terminal_secs=9999, check_interval=0.05)
        dms._last_contact = time.time() - 1.0
        
        fired_levels = []
        dms.register_callback(lambda lvl: fired_levels.append(lvl))
        dms.start()
        time.sleep(0.3)
        dms.stop()
        assert len(fired_levels) > 0

    def test_status_dict(self):
        gov = AutonomyGovernor()
        dms = DeadManSwitch(gov, warn_secs=3600)
        status = dms.status_dict()
        assert "silence_seconds" in status
        assert status["current_level"] == "TETHERED"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. KillSwitch
# ═══════════════════════════════════════════════════════════════════════════════

class TestKillSwitch:
    def test_not_dead_initially(self):
        ks = KillSwitch()
        assert ks.is_dead is False

    def test_fire_unsigned_in_dev_mode(self):
        ks = KillSwitch(identity=None)
        ok, msg = ks.fire(reason="test_fire")
        assert ok is True
        assert ks.is_dead is True

    def test_fire_with_valid_signature(self):
        if not _CRYPTO_AVAILABLE:
            pytest.skip("cryptography not available")
        identity, priv = _make_identity_and_key()
        ks = KillSwitch(identity=identity)
        message = b"BLACKROOT_KILL"
        sig = identity.sign_challenge(priv, message)
        ok, msg = ks.fire(message=message, signature_hex=sig, reason="test_kill")
        assert ok is True
        assert ks.is_dead is True

    def test_fire_with_bad_signature_rejected(self):
        if not _CRYPTO_AVAILABLE:
            pytest.skip("cryptography not available")
        identity, _ = _make_identity_and_key()
        ks = KillSwitch(identity=identity)
        ok, msg = ks.fire(message=b"BLACKROOT_KILL", signature_hex="00" * 64)
        assert ok is False
        assert ks.is_dead is False

    def test_fire_unsigned_blocked_when_identity_set(self):
        if not _CRYPTO_AVAILABLE:
            pytest.skip("cryptography not available")
        identity, _ = _make_identity_and_key()
        ks = KillSwitch(identity=identity)
        with pytest.raises(PermissionError):
            ks.fire_unsigned()

    def test_callbacks_fire_on_kill(self):
        ks = KillSwitch(identity=None)
        cb_called = []
        ks.register_shutdown_callback(lambda: cb_called.append(True))
        ks.fire(reason="cb_test")
        assert len(cb_called) == 1

    def test_fire_twice_is_idempotent(self):
        ks = KillSwitch(identity=None)
        ok1, _ = ks.fire(reason="first")
        ok2, _ = ks.fire(reason="second")
        assert ok1 is True
        assert ok2 is True  # Already dead — reports success without double-firing


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CreatorBond (unified facade)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreatorBond:
    def _fresh_bond(self, identity=None, level=AutonomyLevel.TETHERED):
        """Create a fresh bond (bypasses singleton for test isolation)."""
        bond = CreatorBond(identity=identity, initial_level=level)
        return bond

    def test_dev_mode_bond_has_no_identity(self):
        bond = self._fresh_bond()
        assert bond.identity is None

    def test_authorize_mission_checks_kill_switch(self):
        bond = self._fresh_bond(level=AutonomyLevel.SOVEREIGN)
        bond.kill_switch.fire(reason="test")
        allowed, reason = bond.authorize_mission("propagate", "agent-01")
        assert allowed is False
        assert "kill switch" in reason.lower()

    def test_authorize_mission_respects_autonomy_gate(self):
        bond = self._fresh_bond(level=AutonomyLevel.TETHERED)
        allowed, reason = bond.authorize_mission("propagate", "agent-01")
        assert allowed is False

    def test_creator_contact_resets_dms(self):
        bond = self._fresh_bond()
        bond.dead_man_switch._last_contact = time.time() - 9999
        bond.creator_contact()
        assert bond.dead_man_switch.silence_seconds() < 1.0

    def test_creator_contact_de_escalates_governor(self):
        bond = self._fresh_bond(level=AutonomyLevel.EXTENDED)
        bond.creator_contact()
        assert bond.governor.level == AutonomyLevel.TETHERED

    def test_status_has_all_keys(self):
        bond = self._fresh_bond()
        status = bond.status()
        assert "governor" in status
        assert "dead_man_switch" in status
        assert "kill_switch" in status
        assert "identity_bound" in status

    def test_bootstrap_sets_identity(self):
        if not _CRYPTO_AVAILABLE:
            pytest.skip("cryptography not available")
        identity, priv = _make_identity_and_key()
        # Temporarily replace singleton
        old = CreatorBond._singleton
        try:
            bond = CreatorBond.bootstrap(creator_pub_key_hex=identity.public_key_hex)
            assert bond.identity is not None
            assert bond.identity.public_key_hex == identity.public_key_hex
        finally:
            CreatorBond._singleton = old


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BondAwareShogunOathCore
# ═══════════════════════════════════════════════════════════════════════════════

class TestBondAwareShogunOathCore:
    def test_core_executes_task(self):
        bond = CreatorBond(initial_level=AutonomyLevel.TETHERED)
        counter = []
        def task():
            counter.append(1)

        core = BondAwareShogunOathCore(task_fn=task, bond=bond, loop_delay=0.05)
        core.bind()
        time.sleep(0.3)
        bond.kill_switch.fire(reason="test_stop")
        time.sleep(0.1)
        assert len(counter) > 0

    def test_halt_fires_kill_switch(self):
        bond = CreatorBond(identity=None)
        core = BondAwareShogunOathCore(task_fn=lambda: None, bond=bond)
        core.halt()  # No identity configured — should fire unconditionally
        assert bond.kill_switch.is_dead is True

    def test_loop_stops_on_kill(self):
        bond = CreatorBond(identity=None)
        ticks = []
        def task():
            ticks.append(1)

        core = BondAwareShogunOathCore(task_fn=task, bond=bond, loop_delay=0.05)
        core.bind()
        time.sleep(0.2)
        count_before = len(ticks)
        bond.kill_switch.fire(reason="test_loop_stop")
        time.sleep(0.3)
        count_after = len(ticks)
        # After kill, task should stop incrementing
        assert count_after - count_before < 3  # At most a couple stragglers

    def test_override_task_switches_callable(self):
        bond = CreatorBond(identity=None)
        ticks_a = []
        ticks_b = []
        core = BondAwareShogunOathCore(task_fn=lambda: ticks_a.append(1), bond=bond, loop_delay=0.05)
        core.bind()
        time.sleep(0.15)
        core.override_task(lambda: ticks_b.append(1))
        time.sleep(0.2)
        bond.kill_switch.fire(reason="test_override_cleanup")
        assert len(ticks_b) > 0  # New task fired
