
from agents.infiltrator import InfiltratorAgent
from agents.replicator import ReplicatorAgent
from agents.defiler import DefilerAgent
from agents.sensor import SensorAgent
from agents.learning import LearningAgent
from agents.mission import MissionAgent
from agents.reflection_agent import ReflectionAgent
from agents.propagation_agents import PropagationAgent
from agents.phone_agent_wallet import PhoneWalletAgent

try:
    from agents.peer_linker import PeerLinkerAgent, EnhancedPeerLinkerAgent
except ImportError:
    PeerLinkerAgent = None
    EnhancedPeerLinkerAgent = None

try:
    from agents.shogun_recon import ShogunReconAgent
except ImportError:
    ShogunReconAgent = None

try:
    from agents.targeting_engine import TargetingEngine
except ImportError:
    TargetingEngine = None


def load_agents(config=None):
    agents = [
        InfiltratorAgent(),
        ReplicatorAgent(),
        DefilerAgent(),
        SensorAgent(),
        LearningAgent(),
        MissionAgent(),
        ReflectionAgent(),
        PropagationAgent(),
        PhoneWalletAgent(),
    ]
    if PeerLinkerAgent:
        agents.append(PeerLinkerAgent())
    if EnhancedPeerLinkerAgent:
        agents.append(EnhancedPeerLinkerAgent())
    if ShogunReconAgent:
        agents.append(ShogunReconAgent())
    if TargetingEngine:
        agents.append(TargetingEngine())

    # Add ArchivistAgent if config is provided and archivist config exists
    if config and 'agents' in config and 'archivist' in config['agents']:
        archivist_cfg = config['agents']['archivist']
        try:
            from agents.archivist import ArchivistAgent
            agent_id = archivist_cfg.get('agent_id', 'ARCHIVIST-001')
            secret_key = archivist_cfg.get('secret_key', 'dev_secret')
            governance_key = archivist_cfg.get('governance_key', '')
            db_path = archivist_cfg.get('db_path', 'blackroot_archive.db')
            agents.append(ArchivistAgent(agent_id, secret_key, governance_key, db_path))
        except Exception as e:
            import logging
            logging.getLogger('AGENT-LOADER').warning(f"Failed to instantiate ArchivistAgent: {e}")
    return agents
