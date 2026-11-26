"""
tests/test_creator_bond_enforcer_integration.py
===============================================
Integration test suite for:
- CreatorBond integration as Gate 0 in MissionApprovalEngine
- CreatorBond integration in EnforcerAgent (mission evaluation + anomaly interception)
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import MagicMock
from decimal import Decimal

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernel.creator_bond import CreatorBond, AutonomyLevel
from laws.mission_approval import MissionApprovalEngine, ApprovalStatus, get_mission_approval_engine
from laws.rule_engine import RuleEngine
from agents.enforcer import EnforcerAgent, AuthorityLevel


@pytest.fixture
def clean_bond_tethered():
    """Create a fresh CreatorBond in TETHERED mode."""
    return CreatorBond(initial_level=AutonomyLevel.TETHERED)


@pytest.fixture
def clean_bond_sovereign():
    """Create a fresh CreatorBond in SOVEREIGN mode."""
    return CreatorBond(initial_level=AutonomyLevel.SOVEREIGN)


def test_mission_approval_engine_gate0_blocks_tethered(clean_bond_tethered):
    """Verify that MissionApprovalEngine blocks forbidden mission in TETHERED mode at Gate 0."""
    rule_engine = RuleEngine()
    engine = MissionApprovalEngine(rule_engine=rule_engine, bond=clean_bond_tethered)

    decision = engine.evaluate_mission(
        mission_id="m-prop-01",
        mission_name="propagate",
        agent_id="propagation_agent",
        mission_parameters={"target": "192.168.1.50"}
    )

    assert decision.status == ApprovalStatus.DENIED
    assert decision.approver_id == "CREATOR-BOND"
    assert "CreatorBond Gate Rejection" in decision.justification
    assert "blocked at autonomy level TETHERED" in decision.justification


def test_mission_approval_engine_gate0_allows_safe_mission_tethered(clean_bond_tethered):
    """Verify that safe mission passes Gate 0 in TETHERED mode and proceeds to rule evaluation."""
    rule_engine = RuleEngine()
    engine = MissionApprovalEngine(rule_engine=rule_engine, bond=clean_bond_tethered)

    decision = engine.evaluate_mission(
        mission_id="m-mon-01",
        mission_name="monitor_changes",
        agent_id="sensor_agent",
        mission_parameters={"target": "system_logs"}
    )

    assert decision.status == ApprovalStatus.APPROVED
    assert decision.approver_id == "ENFORCER-AGENT"


def test_mission_approval_engine_gate0_allows_propagate_in_sovereign(clean_bond_sovereign):
    """Verify that propagate mission passes Gate 0 when autonomy reaches SOVEREIGN."""
    rule_engine = RuleEngine()
    engine = MissionApprovalEngine(rule_engine=rule_engine, bond=clean_bond_sovereign)

    decision = engine.evaluate_mission(
        mission_id="m-prop-02",
        mission_name="propagate",
        agent_id="propagation_agent",
        mission_parameters={"target": "10.0.0.5"}
    )

    # Allowed by CreatorBond gate; status depends on constitutional risk/rules
    assert decision.approver_id != "CREATOR-BOND"
    assert decision.status in [ApprovalStatus.APPROVED, ApprovalStatus.CONDITIONAL_APPROVAL, ApprovalStatus.REQUIRES_REVIEW]


def test_mission_approval_engine_gate0_kill_switch_active(clean_bond_sovereign):
    """Verify that all missions are denied at Gate 0 when KillSwitch has fired."""
    clean_bond_sovereign.kill_switch.fire(reason="Emergency operator termination")

    engine = MissionApprovalEngine(bond=clean_bond_sovereign)
    decision = engine.evaluate_mission(
        mission_id="m-any-01",
        mission_name="monitor_changes",
        agent_id="sensor_agent",
        mission_parameters={}
    )

    assert decision.status == ApprovalStatus.DENIED
    assert decision.approver_id == "CREATOR-BOND"
    assert "Kill switch is active" in decision.justification


@pytest.mark.asyncio
async def test_enforcer_agent_evaluate_mission_for_approval_gate0(clean_bond_tethered):
    """Verify that EnforcerAgent.evaluate_mission_for_approval enforces CreatorBond gate."""
    mock_ee = MagicMock()
    mock_el = MagicMock()
    mock_cc = MagicMock()
    constitution = {"blackroot_constitution": {"governance": {}, "agents": []}}

    enforcer = EnforcerAgent(
        enforcement_engine=mock_ee,
        economic_layer=mock_el,
        command_control=mock_cc,
        constitution_reference=constitution,
        bond=clean_bond_tethered
    )

    try:
        decision = await enforcer.evaluate_mission_for_approval(
            mission_id="enf-prop-01",
            mission_name="propagate",
            agent_id="rogue_propagator",
            mission_parameters={"target": "192.168.1.1"}
        )

        assert decision["status"] == "denied"
        assert decision["approver_id"] == "CREATOR-BOND"
        assert "CreatorBond Gate Rejection" in decision["justification"]
        assert len(enforcer.denial_log) > 0
        assert enforcer.denial_log[-1]["mission_id"] == "enf-prop-01"
    finally:
        enforcer.shutdown(timeout=1.0)


def test_enforcer_agent_anomaly_creator_bond_breach(clean_bond_tethered):
    """Verify that EnforcerAgent._evaluate_anomaly_sync detects and sanctions CreatorBond breaches."""
    mock_ee = MagicMock()
    mock_el = MagicMock()
    mock_cc = MagicMock()
    constitution = {"blackroot_constitution": {"governance": {}, "agents": []}}

    enforcer = EnforcerAgent(
        enforcement_engine=mock_ee,
        economic_layer=mock_el,
        command_control=mock_cc,
        constitution_reference=constitution,
        bond=clean_bond_tethered
    )

    try:
        anomaly = {
            "agent_id": "rogue_infiltrator",
            "action": "privilege_escalation",
            "context": {"target": "SYSTEM"}
        }

        enforcer._evaluate_anomaly_sync(anomaly)

        # Should record a violation or sanction
        assert len(enforcer.violation_log) > 0 or len(enforcer.sanction_log) > 0
    finally:
        enforcer.shutdown(timeout=1.0)


def test_cli_bond_generate_identity_and_status(tmp_path, capsys):
    """Verify blackroot.cmd_bond --generate-identity and --status work cleanly."""
    import argparse
    import json
    import blackroot

    args_gen = argparse.Namespace(generate_identity=True, out_dir=str(tmp_path), status=False)
    blackroot.cmd_bond(args_gen)
    out = capsys.readouterr().out
    assert "Ed25519 Creator Identity Generated" in out

    pub_file = tmp_path / "creator_identity.json"
    priv_file = tmp_path / "creator_private.key"
    assert pub_file.exists()
    assert priv_file.exists()
    pub_data = json.loads(pub_file.read_text(encoding="utf-8"))
    assert len(pub_data["public_key_hex"]) == 64

    args_status = argparse.Namespace(generate_identity=False, status=True)
    blackroot.cmd_bond(args_status)
    out_status = capsys.readouterr().out
    assert "Autonomy Level:" in out_status
    assert "Kill Switch:" in out_status


def test_cli_start_oath_exits_on_kill_switch(monkeypatch, capsys):
    """Verify cmd_start with --oath binds BondAwareShogunOathCore and exits when kill_switch fires."""
    import argparse
    import threading
    import time
    import blackroot
    from kernel.creator_bond import CreatorBond

    bond = CreatorBond(initial_level=AutonomyLevel.TETHERED)
    monkeypatch.setattr(CreatorBond, "instance", classmethod(lambda cls: bond))

    def _fire_soon():
        time.sleep(0.15)
        bond.kill_switch.fire(reason="test_shutdown")

    threading.Thread(target=_fire_soon, daemon=True).start()

    args = argparse.Namespace(
        all=False, c2=False, dns=False, agents=False, orchestrate=False, oath=True
    )
    blackroot.cmd_start(args)
    out = capsys.readouterr().out
    assert "BondAwareShogunOathCore bound" in out
    assert "KillSwitch triggered" in out
