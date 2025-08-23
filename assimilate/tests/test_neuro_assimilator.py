import pytest
import os
from src.neuro_assimilator import NeuroAssimilatorAgent
from src.security.security_manager import SecurityManager
from src.reliability.reliability_monitor import ReliabilityMonitor
from src.performance.performance_profiler import PerformanceProfiler
from src.extensibility.plugin_manager import PluginManager
from src.rust_integration.rust_handler import RustScriptHandler

from src.neuro_assimilator import TrustMatrix

@pytest.fixture
def setup_agent():
    trust_matrix = TrustMatrix()
    agent_codex = {}
    agent = NeuroAssimilatorAgent(trust_matrix, agent_codex)
    return agent

def test_agent_initialization(setup_agent):
    agent = setup_agent
    assert agent is not None
    assert isinstance(agent.trust_matrix, TrustMatrix)
    assert isinstance(agent.codex, dict)

def test_agent_observe(setup_agent):
    agent = setup_agent
    observation = agent.observe({'cpu_usage': 30.0, 'memory_usage': 50.0})
    assert 'cpu_usage' in observation
    assert 'memory_usage' in observation
    assert observation['cpu_usage'] == 30.0
    assert observation['memory_usage'] == 50.0

def test_agent_decide(setup_agent):
    agent = setup_agent
    observation = {'cpu_usage': 30.0, 'memory_usage': 50.0, 'performance_score': 0.8}
    action = agent.decide(observation)
    assert action == 'idle'

def test_agent_act_execute_code(setup_agent):
    agent = setup_agent
    python_code = {
        'name': 'test_script',
        'type': 'python_script',
        'source': 'def test(): return "Hello, World!"'
    }
    agent.discover_and_assimilate(python_code)
    context = {'pid': os.getpid()}
    result = agent.act('execute_code', context)
    assert 'test_script' in result

def test_security_manager_integration():
    security_manager = SecurityManager()
    assert security_manager is not None

def test_reliability_monitor_integration():
    reliability_monitor = ReliabilityMonitor()
    assert reliability_monitor is not None

def test_performance_profiler_integration():
    performance_profiler = PerformanceProfiler()
    assert performance_profiler is not None

def test_plugin_manager_integration():
    plugin_manager = PluginManager(manifest_path="/tmp/test_manifest.json")
    assert plugin_manager is not None

def test_rust_script_handler_integration():
    rust_handler = RustScriptHandler()
    assert rust_handler is not None

def test_agent_adapt_reflex_tree(setup_agent):
    agent = setup_agent
    for _ in range(6):
        agent.observe({'cpu_usage': 90.0, 'memory_usage': 70.0, 'performance_score': 0.3})
    agent.adapt_reflex_tree()
    assert len(agent.reflex_tree) > 0  # Ensure reflex tree has been adapted