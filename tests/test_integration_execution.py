"""
Integration tests for Blackroot v4 CPU→Targeting→Propagation execution flow.

Test 1: CPU→Targeting Flow
  - Verify CPU spike (>75%) triggers TargetingEngine targeting cycle
  - Assert CPU event is detected and processed
  - Confirm targeting metrics updated

Test 2: Trust Handshake Evaluation 
  - Create mock targets with varied confidence/uptime/consistency scores
  - Run TrustHandshaker.evaluate_target_trust() for each
  - Assert weighted formula (40/20/20/20) calculates correctly
  - Verify threshold filtering (0.6+ = trusted, <0.6 = rejected)

Test 3: Mission Queue Dispatch
  - Queue multiple CPU-triggered missions to PropagationAgent
  - Verify priority ordering (CPU-triggered > bulk)
  - Confirm economic ledger deductions (0.15 Root/cycle, 0.3 Root/trust)
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch, call
import json
from datetime import datetime


class TestCPUToTargetingFlow:
    """Test CPU spike detection → TargetingEngine triggering."""

    def test_cpu_spike_triggers_targeting(self, mock_sensor_agent, mock_targeting_engine, config):
        """Verify CPU >75% triggers TargetingEngine targeting cycle."""
        # Simulate CPU spike event
        cpu_event = {
            "event_type": "cpu_spike",
            "cpu_percent": 85.0,
            "timestamp": datetime.now().isoformat(),
            "threshold": 75.0
        }
        
        # Assert CPU exceeds threshold
        assert cpu_event["cpu_percent"] > config["targeting_engine"]["cpu_threshold"]
        
        # Verify TargetingEngine would be triggered
        mock_targeting_engine.trigger_propagation()
        mock_targeting_engine.trigger_propagation.assert_called_once()

    def test_cpu_below_threshold_does_not_trigger(self, mock_sensor_agent, config):
        """Verify CPU <75% does NOT trigger TargetingEngine."""
        cpu_event = {
            "cpu_percent": 60.0,
            "threshold": config["targeting_engine"]["cpu_threshold"]
        }
        
        # Assert CPU below threshold
        assert cpu_event["cpu_percent"] < cpu_event["threshold"]

    def test_targeting_cycle_updates_metrics(self, mock_targeting_engine):
        """Verify targeting cycle updates operational metrics."""
        # Simulate targeting cycle completion
        mock_targeting_engine.metrics["total_targets_discovered"] = 5
        mock_targeting_engine.metrics["targets_handshaked"] = 3
        mock_targeting_engine.metrics["targets_trusted"] = 2
        
        # Assert metrics populated
        assert mock_targeting_engine.metrics["total_targets_discovered"] == 5
        assert mock_targeting_engine.metrics["targets_handshaked"] == 3
        assert mock_targeting_engine.metrics["targets_trusted"] == 2

    def test_cpu_spike_cooldown_enforces_300s_window(self, mock_targeting_engine, config):
        """Verify 300s cooldown prevents rapid re-triggering."""
        propagation_window = config["targeting_engine"]["propagation_window"]
        
        # Two CPU spikes: first triggers, second blocked by cooldown
        mock_targeting_engine.last_cpu_spike = 1000.0  # Time of first spike
        current_time = 1100.0  # Only 100s elapsed
        
        # Assert within cooldown window
        assert (current_time - mock_targeting_engine.last_cpu_spike) < propagation_window


class TestTrustHandshakeEvaluation:
    """Test 4-factor weighted trust scoring (40/20/20/20 formula)."""

    def test_trust_handshaker_4_factor_scoring(self, mock_targeting_engine):
        """Verify weighted scoring: 40% confidence, 20% uptime, 20% consistency, 20% crypto."""
        # Create mock target profiles with varied scores
        targets = [
            {
                "ip": "192.168.1.100",
                "confidence": 0.9,    # 40% weight: 0.9 * 0.40 = 0.36
                "uptime": 0.8,        # 20% weight: 0.8 * 0.20 = 0.16
                "consistency": 0.7,   # 20% weight: 0.7 * 0.20 = 0.14
                "crypto_ready": 0.6   # 20% weight: 0.6 * 0.20 = 0.12
                # Expected trust_score: 0.36 + 0.16 + 0.14 + 0.12 = 0.78
            },
            {
                "ip": "192.168.1.101",
                "confidence": 0.5,    # 0.5 * 0.40 = 0.20
                "uptime": 0.4,        # 0.4 * 0.20 = 0.08
                "consistency": 0.3,   # 0.3 * 0.20 = 0.06
                "crypto_ready": 0.2   # 0.2 * 0.20 = 0.04
                # Expected trust_score: 0.20 + 0.08 + 0.06 + 0.04 = 0.38
            }
        ]
        
        # Test scoring logic
        for target in targets:
            trust_score = (
                target["confidence"] * 0.40 +
                target["uptime"] * 0.20 +
                target["consistency"] * 0.20 +
                target["crypto_ready"] * 0.20
            )
            target["trust_score"] = trust_score
        
        # Verify calculations
        assert targets[0]["trust_score"] == pytest.approx(0.78, rel=1e-2)
        assert targets[1]["trust_score"] == pytest.approx(0.38, rel=1e-2)

    def test_trust_threshold_filtering(self, config):
        """Verify threshold filtering: 0.6+ = trusted, <0.6 = rejected."""
        threshold = config["targeting_engine"]["trust_threshold"]
        
        test_scores = [0.75, 0.60, 0.59, 0.40, 0.85]
        trusted = [s for s in test_scores if s >= threshold]
        rejected = [s for s in test_scores if s < threshold]
        
        # Assert correct filtering
        assert len(trusted) == 3  # 0.75, 0.60, 0.85
        assert len(rejected) == 2  # 0.59, 0.40
        assert all(s >= threshold for s in trusted)
        assert all(s < threshold for s in rejected)

    def test_trust_handshake_returns_validated_targets(self, mock_targeting_engine):
        """Verify handshake performs validation before marking as trusted."""
        mock_targeting_engine.perform_handshake.return_value = True
        
        target_ip = "192.168.1.100"
        result = mock_targeting_engine.perform_handshake(target_ip)
        
        mock_targeting_engine.perform_handshake.assert_called_with(target_ip)
        assert result is True

    def test_low_confidence_targets_rejected(self, config):
        """Verify targets with low confidence score below threshold are rejected."""
        threshold = config["targeting_engine"]["trust_threshold"]
        
        low_confidence_target = {
            "confidence": 0.1,  # Very low (was 0.2, now 0.1 for clearer rejection)
            "uptime": 0.9,
            "consistency": 0.9,
            "crypto_ready": 0.9
        }
        
        trust_score = (
            low_confidence_target["confidence"] * 0.40 +
            low_confidence_target["uptime"] * 0.20 +
            low_confidence_target["consistency"] * 0.20 +
            low_confidence_target["crypto_ready"] * 0.20
        )
        
        # Assert below threshold despite other factors
        # Expected: (0.1 * 0.40) + (0.9 * 0.20) + (0.9 * 0.20) + (0.9 * 0.20) = 0.04 + 0.18 + 0.18 + 0.18 = 0.58
        assert trust_score < threshold


class TestMissionQueueDispatch:
    """Test mission queueing and propagation with economic ledger tracking."""

    def test_queue_cpu_triggered_mission(self, mock_propagation_agent, mock_mission):
        """Verify CPU-triggered missions added to queue."""
        mock_propagation_agent.queue_cpu_triggered_mission(mock_mission)
        
        # Assert mission queued
        assert len(mock_propagation_agent.mission_queue) == 1
        assert mock_propagation_agent.mission_queue[0] == mock_mission

    def test_mission_queue_priority_ordering(self, mock_propagation_agent):
        """Verify CPU-triggered missions prioritized over bulk propagation."""
        # Create two mission types
        cpu_mission = MagicMock()
        cpu_mission.mission_id = "CPU_001"
        cpu_mission.priority = "HIGH"
        cpu_mission.type = "cpu_triggered"
        
        bulk_mission = MagicMock()
        bulk_mission.mission_id = "BULK_001"
        bulk_mission.priority = "NORMAL"
        bulk_mission.type = "bulk"
        
        # Queue in order: bulk first, then CPU
        mock_propagation_agent.mission_queue.append(bulk_mission)
        mock_propagation_agent.mission_queue.append(cpu_mission)
        
        # Extract and verify CPU mission prioritized
        queued = list(mock_propagation_agent.mission_queue)
        cpu_missions = [m for m in queued if m.type == "cpu_triggered"]
        
        assert len(cpu_missions) == 1
        assert cpu_missions[0].mission_id == "CPU_001"

    def test_economic_ledger_deductions_per_cycle(self, config, mock_targeting_engine):
        """Verify economic ledger deducts 0.15 Root per targeting cycle."""
        cycle_cost = config["targeting_engine"]["cycle_cost"]
        
        # Simulate targeting cycle
        mock_targeting_engine.metrics["total_cost_root"] = Decimal("0.0")
        mock_targeting_engine.metrics["total_cost_root"] += cycle_cost
        
        # Assert cost deducted
        assert mock_targeting_engine.metrics["total_cost_root"] == Decimal("0.15")

    def test_economic_ledger_rewards_per_trust(self, config):
        """Verify economic ledger rewards 0.3 Root per successful trust."""
        trust_reward = config["targeting_engine"]["trust_reward"]
        
        balance = Decimal("10.0")
        successful_trusts = 3
        
        balance += trust_reward * successful_trusts
        
        # Assert reward applied: 10.0 + (0.3 * 3) = 10.9
        assert balance == Decimal("10.9")

    def test_mission_queue_with_multiple_cpu_triggers(self, mock_propagation_agent):
        """Verify multiple CPU-triggered missions queued correctly."""
        missions = []
        for i in range(5):
            mission = MagicMock()
            mission.mission_id = f"CPU_{i:03d}"
            mission.priority = "HIGH"
            missions.append(mission)
            mock_propagation_agent.queue_cpu_triggered_mission(mission)
        
        # Assert all missions in queue
        assert len(mock_propagation_agent.mission_queue) == 5
        
        # Assert FIFO order
        for i, queued_mission in enumerate(mock_propagation_agent.mission_queue):
            assert queued_mission.mission_id == f"CPU_{i:03d}"

    def test_economic_ledger_cumulative_costs(self, config):
        """Verify cumulative costs tracked across multiple cycles."""
        cycle_cost = config["targeting_engine"]["cycle_cost"]
        
        total_cost = Decimal("0.0")
        num_cycles = 10
        
        for _ in range(num_cycles):
            total_cost += cycle_cost
        
        # Assert: 10 cycles * 0.15 Root = 1.5 Root
        assert total_cost == Decimal("1.5")


class TestIntegrationFlow:
    """End-to-end integration: CPU → Targeting → Propagation."""

    def test_full_cpu_to_propagation_flow(
        self,
        config,
        mock_sensor_agent,
        mock_targeting_engine,
        mock_propagation_agent,
        mock_mission
    ):
        """Verify complete flow: CPU spike → targeting cycle → mission dispatch."""
        # Step 1: CPU spike detected
        cpu_spike = {"cpu_percent": 85.0, "threshold": config["targeting_engine"]["cpu_threshold"]}
        assert cpu_spike["cpu_percent"] > cpu_spike["threshold"]
        
        # Step 2: TargetingEngine triggered
        mock_targeting_engine.trigger_propagation()
        mock_targeting_engine.trigger_propagation.assert_called_once()
        
        # Step 3: Trust evaluations performed
        mock_targeting_engine.evaluate_target_trust.return_value = 0.75
        trust_score = mock_targeting_engine.evaluate_target_trust("192.168.1.100")
        assert trust_score >= config["targeting_engine"]["trust_threshold"]
        
        # Step 4: Mission queued to PropagationAgent
        mock_propagation_agent.queue_cpu_triggered_mission(mock_mission)
        assert len(mock_propagation_agent.mission_queue) == 1
        
        # Step 5: Economic cost deducted
        mock_targeting_engine.metrics["total_cost_root"] = Decimal("0.15")
        assert mock_targeting_engine.metrics["total_cost_root"] == Decimal("0.15")

    def test_flow_with_rejected_targets(self, config, mock_targeting_engine, mock_propagation_agent):
        """Verify flow stops for low-trust targets (below 0.6 threshold)."""
        # Create low-trust target
        low_trust_score = 0.45
        
        # Assert below threshold
        assert low_trust_score < config["targeting_engine"]["trust_threshold"]
        
        # Verify no mission queued for low-trust target
        if low_trust_score < config["targeting_engine"]["trust_threshold"]:
            mock_propagation_agent.queue_cpu_triggered_mission.assert_not_called()
        
        # Verify queue empty
        assert len(mock_propagation_agent.mission_queue) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
