from agents.infiltrator import InfiltratorAgent
from agents.replicator import ReplicatorAgent
from agents.defiler import DefilerAgent 
from agents.sensor import SensorAgent
from agents.learning import LearningAgent
from agents.mission import MissionAgent
from agents.reflection_agent import ReflectionAgent


def load_agents():
    return [
        InfiltratorAgent(),
        ReplicatorAgent(),
        DefilerAgent(),
        SensorAgent(),
        LearningAgent(),
        MissionAgent(),
        ReflectionAgent()

    ]
