"""
Pytest configuration and shared fixtures for Blackroot integration tests.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from decimal import Decimal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing without actual Redis."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.incr.return_value = 1
    redis_mock.publish.return_value = 1
    redis_mock.ping.return_value = True
    return redis_mock


@pytest.fixture
def mock_kernel(mock_redis):
    """Mock BlackrootKernel with core components."""
    from unittest.mock import MagicMock
    kernel = MagicMock()
    kernel.redis_client = mock_redis
    kernel.black_vault = MagicMock()
    kernel.swarm = MagicMock()
    kernel.swarm.agents = []
    kernel.swarm.channel = "blackroot_swarm"
    return kernel


@pytest.fixture
def config():
    """Test configuration."""
    return {
        "black_vault": {"enabled": True, "path": "/tmp/vault"},
        "swarm": {"enabled": True, "peers": []},
        "redis": {"host": "localhost", "port": 6379},
        "agent_whitelist": ["SensorAgent", "TargetingEngine", "PropagationAgent"],
        "targeting_engine": {
            "cpu_threshold": 75.0,
            "trust_threshold": 0.6,
            "propagation_window": 300,
            "cycle_cost": Decimal("0.15"),
            "trust_reward": Decimal("0.3")
        }
    }


@pytest.fixture
def mock_sensor_agent():
    """Mock SensorAgent for CPU monitoring tests."""
    from unittest.mock import MagicMock
    sensor = MagicMock()
    sensor.agent_id = "sensor-agent-001"
    sensor.get_cpu_usage.return_value = {"cpu_percent": 85.0, "timestamp": 1676000000}
    return sensor


@pytest.fixture
def mock_targeting_engine():
    """Mock TargetingEngine for targeting tests."""
    from unittest.mock import MagicMock
    from collections import deque
    
    targeting = MagicMock()
    targeting.agent_id = "targeting-engine-001"
    targeting.targets = {}
    targeting.last_cpu_spike = None
    targeting.metrics = {
        "total_targets_discovered": 0,
        "targets_handshaked": 0,
        "targets_trusted": 0,
        "propagation_triggered": 0,
        "total_cost_root": Decimal("0.0"),
        "avg_trust_score": 0.0
    }
    
    # Add mock methods
    targeting.evaluate_target_trust = MagicMock(return_value=0.75)
    targeting.perform_handshake = MagicMock(return_value=True)
    targeting.trigger_propagation = MagicMock(return_value=True)
    
    return targeting


@pytest.fixture
def mock_propagation_agent():
    """Mock PropagationAgent for mission queue tests."""
    from unittest.mock import MagicMock
    from collections import deque
    from threading import Lock
    
    propagation = MagicMock()
    propagation.agent_id = "propagation-agent-001"
    propagation.mission_queue = deque()
    propagation.mission_queue_lock = Lock()
    propagation.missions_executed = []
    
    # Add mock methods
    propagation.queue_cpu_triggered_mission = MagicMock(return_value=True)
    propagation.run = MagicMock()
    
    def queue_mission(mission):
        with propagation.mission_queue_lock:
            propagation.mission_queue.append(mission)
        return True
    
    propagation.queue_cpu_triggered_mission.side_effect = queue_mission
    
    return propagation


@pytest.fixture
def mock_mission():
    """Mock Mission object for testing."""
    from unittest.mock import MagicMock
    mission = MagicMock()
    mission.mission_id = "CPU_MISSION_001"
    mission.target_ips = ["192.168.1.100", "192.168.1.101"]
    mission.priority = "HIGH"
    mission.cost_root = Decimal("0.15")
    mission.trust_required = "MEDIUM"
    return mission
