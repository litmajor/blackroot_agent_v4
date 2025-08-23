
from agents.infiltrator import InfiltratorAgent
from agents.replicator import ReplicatorAgent
from agents.defiler import DefilerAgent
from agents.sensor import SensorAgent
from agents.learning import LearningAgent
from agents.mission import MissionAgent
from agents.reflection_agent import ReflectionAgent
from agents.propagation_agents import PropagationAgent
try:
    from agents.peer_linker import PeerLinkerAgent, EnhancedPeerLinkerAgent
except ImportError:
    PeerLinkerAgent = None
    EnhancedPeerLinkerAgent = None


def load_agents():
    agents = [
        InfiltratorAgent(),
        ReplicatorAgent(),
        DefilerAgent(),
        SensorAgent(),
        LearningAgent(),
        MissionAgent(),
        ReflectionAgent(),
        PropagationAgent(),
    ]
    if PeerLinkerAgent:
        agents.append(PeerLinkerAgent())
    if EnhancedPeerLinkerAgent:
        agents.append(EnhancedPeerLinkerAgent())
    return agents
