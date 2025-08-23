# --- BlackrootHost: Real Runtime Host Abstraction ---
import os
import uuid
import psutil
import socket as pysocket
import base64
import secrets
from typing import Set
import json
import time
import random
import hashlib
import threading
import socket
from typing import Dict, List, Callable, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from redis import Redis

class BlackrootHost:
    # --- Mesh compatibility fields (dynamic, for peer tracking) ---
    # These are set dynamically for mesh peer objects
    @property
    def last_seen(self):
        return getattr(self, '_last_seen', None)
    @last_seen.setter
    def last_seen(self, value):
        self._last_seen = value

    @property
    def status(self):
        return getattr(self, '_status', NodeStatus.CONNECTED)
    @status.setter
    def status(self, value):
        self._status = value

    @property
    def port(self):
        return getattr(self, '_port', 0)
    @port.setter
    def port(self, value):
        self._port = value

    @property
    def capabilities(self):
        # Prefer explicit capabilities if set, else loaded_agents
        return getattr(self, '_capabilities', list(self.loaded_agents))
    @capabilities.setter
    def capabilities(self, value):
        self._capabilities = value

    def is_alive(self, timeout: float = 90.0) -> bool:
        from datetime import datetime
        if self.last_seen is not None:
            return (datetime.now() - self.last_seen).total_seconds() < timeout
        return True
    """
    Represents a real Blackroot runtime host (node) in the mesh.
    Encapsulates identity, capacity, roles, and loaded agents (abilities).
    """
    def __init__(self, node_id: Optional[str] = None, roles: Optional[Set[str]] = None):
        self.node_id = node_id or self._generate_node_id()
        self.identity = self._generate_identity()
        self.capacity = self._get_capacity()
        self.roles = set(roles) if roles else set()
        self.loaded_agents = set()  # Set of agent/ability names
        self.hostname = pysocket.gethostname()
        self.address = self._get_ip_address()
        self.signature = self._generate_signature()

    def _generate_node_id(self):
        return f"host_{uuid.uuid4().hex[:12]}"

    def _generate_identity(self):
        # For demo: base64-encoded random 32 bytes
        return base64.b64encode(secrets.token_bytes(32)).decode()

    def _generate_signature(self):
        # For demo: hash of identity
        return hashlib.sha256(self.identity.encode()).hexdigest()

    def _get_capacity(self):
        # Use psutil to get CPU/mem/IO
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'disk_gb': round(psutil.disk_usage(os.getcwd()).total / (1024**3), 2),
        }

    def _get_ip_address(self):
        try:
            return pysocket.gethostbyname(self.hostname)
        except Exception:
            return '127.0.0.1'

    def add_role(self, role: str):
        self.roles.add(role)

    def load_agent(self, agent_name: str):
        self.loaded_agents.add(agent_name)

    def unload_agent(self, agent_name: str):
        self.loaded_agents.discard(agent_name)

    def to_dict(self):
        return {
            'node_id': self.node_id,
            'identity': self.identity,
            'signature': self.signature,
            'capacity': self.capacity,
            'roles': list(self.roles),
            'loaded_agents': list(self.loaded_agents),
            'hostname': self.hostname,
            'address': self.address,
        }

# Define missing CommandType enum
class CommandType(Enum):
    EXECUTE = "execute"
    QUERY = "query"
    UPDATE = "update"
    DELETE = "delete"
    # Add more as needed

# Define missing MetricsCollector stub if not present
class MetricsCollector:
    def __init__(self, mesh_node):
        self.mesh_node = mesh_node
        self.collection_interval = 10
    def collect_metrics(self):
        pass


# NetworkMessage dataclass (must be after MessageType and datetime are defined)
from dataclasses import dataclass

@dataclass
class NetworkMessage:
    message_id: str
    message_type: 'MessageType'
    sender: str
    recipient: str
    payload: dict
    timestamp: 'datetime'

    def to_dict(self) -> dict:
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender': self.sender,
            'recipient': self.recipient,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NetworkMessage':
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            sender=data['sender'],
            recipient=data['recipient'],
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
import logging
from datetime import datetime, timedelta
from secure_channel import SecureChannel  # Assume this exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a SecureChannelBase for type consistency
class SecureChannelBase:
    def __init__(self, node_id: str):
        self.node_id = node_id

    def pull_messages(self) -> List[Dict]:
        raise NotImplementedError

    def send_message(self, target: str, message: Dict) -> bool:
        raise NotImplementedError

    def add_incoming_message(self, message: Dict):
        raise NotImplementedError

# Import secure channel module (assumed to exist)
try:
    from secure_channel import SecureChannel as ImportedSecureChannel

except ImportError:
    logger.warning("SecureChannel module not found. Using enhanced mock implementation.")

    class SecureChannel(SecureChannelBase):
        """Enhanced mock implementation of SecureChannel with encryption simulation"""
        def __init__(self, node_id: str):
            super().__init__(node_id)
            self.message_queue = []
            self.encryption_key = hashlib.sha256(node_id.encode()).hexdigest()[:32]
            self.message_counter = 0

        def pull_messages(self) -> List[Dict]:
            messages = self.message_queue[:]
            self.message_queue = []
            return messages

        def send_message(self, target: str, message: Dict) -> bool:
            try:
                # Simulate message encryption/signing
                encrypted_msg = self._encrypt_message(message)
                logger.debug(f"Encrypted message to {target}: {len(str(encrypted_msg))} bytes")

                # In real implementation, this would send over network
                # For mock, we'll simulate delivery with some probability
                delivery_success = random.random() > 0.05  # 95% delivery rate

                if delivery_success:
                    self.message_counter += 1
                    return True
                else:
                    logger.warning(f"Simulated message delivery failure to {target}")
                    return False

            except Exception as e:
                logger.error(f"Failed to send message to {target}: {e}")
                return False

        def _encrypt_message(self, message: Dict) -> str:
            """Simulate message encryption"""
            msg_str = json.dumps(message, sort_keys=True)
            # Simple mock encryption using base64 + key
            import base64
            encrypted = base64.b64encode(
                (self.encryption_key + msg_str).encode()
            ).decode()
            return encrypted

        def add_incoming_message(self, message: Dict):
            """Add an incoming message to the queue (for testing)"""
            self.message_queue.append(message)


class NodeStatus(Enum):
    """Enumeration of possible node states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LEADER = "leader"
    ERROR = "error"


class MessageType(Enum):
    """Enumeration of message types for mesh communication"""
    HEARTBEAT = "heartbeat"
    IDENTITY_BROADCAST = "identity_broadcast"
    PEER_REQUEST = "peer_request"
    PEER_RESPONSE = "peer_response"
    COMMAND = "command"
    COMMAND_RESPONSE = "command_response"
    ELECTION_ANNOUNCE = "election_announce"
    ELECTION_VOTE = "election_vote"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    ARTIFACT_TRANSFER = "artifact_transfer"  # New for payload mobility


@dataclass


@dataclass
class Command:
    """Data class for command structure"""
    command_id: str
    command_type: CommandType
    target_module: str
    action: str
    args: List[Any]
    kwargs: Dict[str, Any]
    sender: str
    timestamp: datetime
    priority: int = 1
    timeout: float = 30.0
    
    def to_dict(self) -> Dict:
        """Convert command to dictionary for serialization"""
        data = asdict(self)
        data['command_type'] = self.command_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Command':
        """Create command from dictionary"""
        data['command_type'] = CommandType(data['command_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


# Global registry for dynamic modules with metadata
module_registry: Dict[str, Dict[str, Any]] = {}


def register_module(name: str, 


                   initializer: Callable,
                   capabilities: Optional[List[str]] = None,
                   version: str = "1.0.0",
                   description: str = ""):
    """
    Register a module in the global registry with metadata
    
    Args:
        name: Unique module identifier
        initializer: Callable that initializes the module
        capabilities: List of capabilities this module provides
        version: Module version string
        description: Human-readable description
    """
    module_registry[name] = {
        'initializer': initializer,
        'capabilities': capabilities or [],
        'version': version,
        'description': description,
        'registered_at': datetime.now(),
        'call_count': 0,
        'last_called': None
    }
    logger.info(f"Module registered: {name} v{version}")



def get_module_info(name: str) -> Optional[Dict[str, Any]]:
    # Register NeuroAssimilatorAgent as a module after get_module_info is defined
    try:
        from assimilate.src.neuro_assimilator import NeuroAssimilatorAgent
        neuro_agent = NeuroAssimilatorAgent()
        # Use a callable method from the agent as initializer, e.g., agent.execute or similar
        # If no such method, fallback to a lambda that returns the agent
        initializer = getattr(neuro_agent, 'execute', None)
        if not callable(initializer):
            initializer = lambda *args, **kwargs: neuro_agent
        register_module(
            name="payload_engine",
            initializer=initializer,
            capabilities=["payload_management", "neuro_assimilation"],
            version="1.0.0",
            description="Advanced payload and neuro-assimilation agent"
        )
    except Exception as e:
        logger.warning(f"Failed to register NeuroAssimilatorAgent: {e}")
    """Get information about a registered module"""
    return module_registry.get(name)

# Register NeuroAssimilatorAgent as a module after get_module_info is defined
try:
    from assimilate.src.neuro_assimilator import NeuroAssimilatorAgent
    neuro_agent = NeuroAssimilatorAgent()
    # Use a callable method from the agent as initializer, e.g., agent.execute or similar
    # If no such method, fallback to a lambda that returns the agent
    initializer = getattr(neuro_agent, 'execute', None)
    if not callable(initializer):
        initializer = lambda *args, **kwargs: neuro_agent
    register_module(
        name="payload_engine",
        initializer=initializer,
        capabilities=["payload_management", "neuro_assimilation"],
        version="1.0.0",
        description="Advanced payload and neuro-assimilation agent"
    )
except Exception as e:
    logger.warning(f"Failed to register NeuroAssimilatorAgent: {e}")


class LoadBalancer:
    """Load balancer for distributing commands across mesh nodes"""
    
    def __init__(self, mesh_node: 'SwarmMesh'):
        self.mesh_node = mesh_node
        self.load_metrics: Dict[str, float] = {}  # node_id -> load score
        self.last_update = datetime.now()
    
    def select_target_node(self, 
                          command: Command,
                          required_capabilities: Optional[List[str]] = None) -> Optional[str]:
        """
        Select the best node to execute a command
        
        Args:
            command: Command to be executed
            required_capabilities: List of required capabilities
            
        Returns:
            Node ID of selected target or None if no suitable node found
        """
        # Get all available nodes including self
        candidates = [self.mesh_node.node_id]
        
        with self.mesh_node.lock:
            for peer_id, peer_info in self.mesh_node.peers.items():
                if peer_info.is_alive() and peer_info.status == NodeStatus.CONNECTED:
                    candidates.append(peer_id)
        
        # Filter by capabilities if specified
        if required_capabilities:
            filtered_candidates = []
            for node_id in candidates:
                if self._node_has_capabilities(node_id, required_capabilities):
                    filtered_candidates.append(node_id)
            candidates = filtered_candidates
        
        if not candidates:
            return None
        
        # Select based on load balancing strategy
        return self._select_least_loaded_node(candidates)
    
    def _node_has_capabilities(self, node_id: str, required_caps: List[str]) -> bool:
        """Check if a node has required capabilities"""
        if node_id == self.mesh_node.node_id:
            node_caps = self.mesh_node._get_node_capabilities()
        else:
            # In real implementation, this would query peer capabilities
            # For mock, assume all peers have basic capabilities
            node_caps = ['mesh_networking', 'command_execution']
        
        return all(cap in node_caps for cap in required_caps)
    
    def _select_least_loaded_node(self, candidates: List[str]) -> str:
        """Select the node with lowest load"""
        self._update_load_metrics()
        
        # Find node with minimum load
        best_node = candidates[0]
        min_load = self.load_metrics.get(best_node, 0.0)
        
        for node_id in candidates[1:]:
            node_load = self.load_metrics.get(node_id, 0.0)
            if node_load < min_load:
                min_load = node_load
                best_node = node_id
        
        return best_node
    
    def _update_load_metrics(self):
        """Update load metrics for all known nodes"""
        now = datetime.now()
        if (now - self.last_update).total_seconds() < 10:  # Update every 10 seconds
            return
        
        # Update own load
        own_load = self._calculate_node_load(self.mesh_node.node_id)
        self.load_metrics[self.mesh_node.node_id] = own_load
        
        # For peers, we'd normally request load info
        # For mock, use random values
        with self.mesh_node.lock:
            for peer_id in self.mesh_node.peers.keys():
                if peer_id not in self.load_metrics:
                    self.load_metrics[peer_id] = random.uniform(0.1, 0.9)
        
        self.last_update = now
    
    def _calculate_node_load(self, node_id: str) -> float:
        """Calculate current load for a node (0.0 = no load, 1.0 = fully loaded)"""
        if node_id != self.mesh_node.node_id:
            return self.load_metrics.get(node_id, 0.5)  # Default for remote nodes
        
        # Calculate load based on various factors
        factors = []
        
        # Command queue length
        with self.mesh_node.lock:
            queue_load = min(len(self.mesh_node.pending_commands) / 10.0, 1.0)
            factors.append(queue_load)
        
        # Thread activity (mock calculation)
        thread_load = min(len(self.mesh_node.threads) / 10.0, 1.0)
        factors.append(thread_load)
        
        # Recent command execution rate
        recent_commands = self.mesh_node.stats['commands_executed']
        execution_load = min(recent_commands / 100.0, 1.0)
        factors.append(execution_load)
        
        # Calculate weighted average
        weights = [0.5, 0.2, 0.3]  # Queue=50%, Threads=20%, Execution=30%
        return sum(f * w for f, w in zip(factors, weights))


class ConsistencyManager:
    """Manages data consistency across the mesh network"""
    
    def __init__(self, mesh_node: 'SwarmMesh'):
        self.mesh_node = mesh_node
        self.version_vector: Dict[str, int] = {}  # node_id -> version
        self.pending_syncs: Dict[str, datetime] = {}  # node_id -> last_sync_request
    
    def sync_with_peers(self, force: bool = False):
        """Synchronize state with all peers"""
        now = datetime.now()
        
        with self.mesh_node.lock:
            for peer_id, peer_info in self.mesh_node.peers.items():
                if not peer_info.is_alive():
                    continue
                
                last_sync = self.pending_syncs.get(peer_id)
                if not force and last_sync and (now - last_sync).total_seconds() < 60:
                    continue  # Don't sync too frequently
                
                self._request_sync(peer_id)
                self.pending_syncs[peer_id] = now
    
    def _request_sync(self, peer_id: str):
        """Request synchronization with a specific peer"""
        sync_msg = NetworkMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.SYNC_REQUEST,
            sender=self.mesh_node.node_id,
            recipient=peer_id,
            payload={
                'version_vector': self.version_vector.copy(),
                'requesting_modules': list(module_registry.keys())
            },
            timestamp=datetime.now()
        )
        
        try:
            self.mesh_node.secure_channel.send_message(peer_id, sync_msg.to_dict())
            logger.debug(f"Sync request sent to {peer_id}")
        except Exception as e:
            logger.error(f"Failed to send sync request to {peer_id}: {e}")
    
    def handle_sync_request(self, message: NetworkMessage) -> Dict[str, Any]:
        """Handle incoming sync request and return response data"""
        peer_vector = message.payload.get('version_vector', {})
        requested_modules = message.payload.get('requesting_modules', [])
        
        response_data = {
            'version_vector': self.version_vector.copy(),
            'module_updates': {},
            'peer_list': {}
        }
        
        # Check which modules need updates
        for module_name in requested_modules:
            if module_name in module_registry:
                peer_version = peer_vector.get(f"{self.mesh_node.node_id}:{module_name}", 0)
                our_version = self.version_vector.get(f"{self.mesh_node.node_id}:{module_name}", 1)
                
                if our_version > peer_version:
                    response_data['module_updates'][module_name] = {
                        'version': our_version,
                        'metadata': module_registry[module_name].copy()
                    }
                    # Remove the actual initializer function for transmission
                    if 'initializer' in response_data['module_updates'][module_name]['metadata']:
                        del response_data['module_updates'][module_name]['metadata']['initializer']
        
        # Share peer information
        with self.mesh_node.lock:
            for peer_id, peer_info in self.mesh_node.peers.items():
                if peer_info.is_alive():
                    response_data['peer_list'][peer_id] = {
                        'address': peer_info.address,
                        'port': peer_info.port,
                        'capabilities': peer_info.capabilities,
                        'last_seen': peer_info.last_seen.isoformat() if peer_info.last_seen else ''
                    }
        
        return response_data
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return hashlib.md5(
            f"{self.mesh_node.node_id}_{time.time()}_{random.randint(0, 10000)}".encode()
        ).hexdigest()[:16]


class SecurityManager:
    """Enhanced security manager for the mesh network"""
    
    def __init__(self, mesh_node: 'SwarmMesh'):
        self.mesh_node = mesh_node
        self.trusted_nodes: Set[str] = set()
        self.blacklisted_nodes: Set[str] = set()
        self.rate_limits: Dict[str, Dict[str, Any]] = {}  # node_id -> rate limit info
        self.security_events: List[Dict[str, Any]] = []
    
    def authenticate_node(self, node_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate a node attempting to join the mesh"""
        # In a real implementation, this would verify certificates, tokens, etc.
        # For demo, we'll use simple validation
        
        if node_id in self.blacklisted_nodes:
            self._log_security_event("auth_blocked", node_id, "Node is blacklisted")
            return False
        
        # Check if credentials are valid
        expected_token = hashlib.sha256(f"secret_{node_id}".encode()).hexdigest()
        provided_token = credentials.get('token', '')
        
        if provided_token != expected_token:
            self._log_security_event("auth_failed", node_id, "Invalid credentials")
            return False
        
        self.trusted_nodes.add(node_id)
        self._log_security_event("auth_success", node_id, "Node authenticated successfully")
        return True
    
    def authorize_command(self, command: Command) -> bool:
        """Authorize command execution"""
        sender = command.sender
        
        # Check if sender is trusted
        if sender not in self.trusted_nodes and sender != self.mesh_node.node_id:
            self._log_security_event("cmd_unauthorized", sender, f"Untrusted sender: {command.command_id}")
            return False
        
        # Check rate limits
        if not self._check_rate_limit(sender, command.command_type):
            self._log_security_event("rate_limited", sender, f"Rate limit exceeded: {command.command_id}")
            return False
        
        # Check command-specific permissions
        if not self._check_command_permissions(sender, command):
            self._log_security_event("permission_denied", sender, f"Permission denied: {command.command_id}")
            return False
        
        return True
    
    def _check_rate_limit(self, node_id: str, command_type: CommandType) -> bool:
        """Check if node is within rate limits"""
        now = datetime.now()
        
        if node_id not in self.rate_limits:
            self.rate_limits[node_id] = {
                'commands': [],
                'last_reset': now
            }
        
        rate_info = self.rate_limits[node_id]
        
        # Reset rate limit window every minute
        if (now - rate_info['last_reset']).total_seconds() > 60:
            rate_info['commands'] = []
            rate_info['last_reset'] = now
        
        # Remove old commands (older than 1 minute)
        cutoff_time = now - timedelta(minutes=1)
        rate_info['commands'] = [
            cmd_time for cmd_time in rate_info['commands']
            if cmd_time > cutoff_time
        ]
        
        # Check limit (e.g., 100 commands per minute)
        if len(rate_info['commands']) >= 100:
            return False
        
        # Record this command attempt
        rate_info['commands'].append(now)
        return True
    
    def _check_command_permissions(self, sender: str, command: Command) -> bool:
        """Check if sender has permission to execute this command"""
        # Define permission rules (in real implementation, this would be more sophisticated)
        restricted_modules = {'system_critical', 'security_config'}
        admin_nodes = {self.mesh_node.node_id}  # Only local node has admin rights for demo
        
        if command.target_module in restricted_modules and sender not in admin_nodes:
            return False
        
        return True
    
    def _log_security_event(self, event_type: str, node_id: str, description: str):
        """Log a security event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'node_id': node_id,
            'description': description
        }
        
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        logger.warning(f"Security event [{event_type}] {node_id}: {description}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report"""
        return {
            'trusted_nodes': list(self.trusted_nodes),
            'blacklisted_nodes': list(self.blacklisted_nodes),
            'recent_events': self.security_events[-50:],  # Last 50 events
            'rate_limit_status': {
                node_id: {
                    'current_rate': len(info['commands']),
                    'last_reset': info['last_reset'].isoformat()
                }
                for node_id, info in self.rate_limits.items()
            }
        }


class SwarmMesh:

    def get_artifact_metadata(self, artifact_id: str) -> Optional[dict]:
        """
        Retrieve metadata for an artifact from BlackVault or memory.
        Args:
            artifact_id: Artifact identifier
        Returns:
            Metadata dict if found, else None
        """
        try:
            from black_vault import BlackVault
            if not hasattr(self, '_artifact_vault'):
                self._artifact_vault = BlackVault()
            # Try 'get_metadata' or 'metadata' method
            try:
                get_metadata = getattr(self._artifact_vault, 'get_metadata', None)
                if callable(get_metadata):
                    result = get_metadata(str(artifact_id))
                    if isinstance(result, dict):
                        return result
                    else:
                        return None
            except Exception:
                pass
            try:
                metadata_method = getattr(self._artifact_vault, 'metadata', None)
                if callable(metadata_method):
                    result = metadata_method(str(artifact_id))
                    if isinstance(result, dict):
                        return result
                    else:
                        return None
            except Exception:
                pass
        except Exception:
            pass
        # Fallback to in-memory
        if hasattr(self, '_received_artifacts') and artifact_id in self._received_artifacts:
            return self._received_artifacts[artifact_id].get('metadata')
        return None

    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact locally (from BlackVault or memory).
        Args:
            artifact_id: Artifact identifier
        Returns:
            True if deleted, False otherwise
        """
        deleted = False
        try:
            from black_vault import BlackVault
            if not hasattr(self, '_artifact_vault'):
                self._artifact_vault = BlackVault()
            # Try 'delete' or 'remove' method
            if hasattr(self._artifact_vault, 'delete'):
                self._artifact_vault.delete(str(artifact_id))
                deleted = True
            else:
                try:
                    remove_method = getattr(self._artifact_vault, 'remove', None)
                    if callable(remove_method):
                        remove_method(str(artifact_id))
                        deleted = True
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback to in-memory
        if not deleted and hasattr(self, '_received_artifacts') and artifact_id in self._received_artifacts:
            del self._received_artifacts[artifact_id]
            deleted = True
        return deleted

    def remote_delete_artifact(self, peer_id: str, artifact_id: str) -> bool:
        """
        Request a remote peer to delete an artifact.
        Args:
            peer_id: Target peer node ID
            artifact_id: Artifact identifier
        Returns:
            True if request sent, False otherwise
        """
        delete_msg = {
            'type': 'artifact_delete_request',
            'artifact_id': artifact_id,
            'requester': self.node_id,
            'timestamp': datetime.now().isoformat(),
        }
        try:
            self.secure_channel.send_message(peer_id, delete_msg)
            logger.info(f"Sent remote delete request for artifact {artifact_id} to {peer_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send remote delete request to {peer_id}: {e}")
            return False

    def list_artifacts(self) -> list:
        """
        List all known artifact IDs (from BlackVault if available, else memory).
        Returns:
            List of artifact IDs (str)
        """
        # Try BlackVault first
        try:
            from black_vault import BlackVault
            if not hasattr(self, '_artifact_vault'):
                self._artifact_vault = BlackVault()
            try:
                list_method = getattr(self._artifact_vault, 'list', None)
                if callable(list_method):
                    result = list_method()
                    if isinstance(result, list):
                        return result
                    else:
                        return []
            except Exception:
                pass
            try:
                list_payloads_method = getattr(self._artifact_vault, 'list_payloads', None)
                if callable(list_payloads_method):
                    result = list_payloads_method()
                    if isinstance(result, list):
                        return result
                    else:
                        return []
            except Exception:
                pass
        except Exception:
            pass
        # Fallback to in-memory
        if hasattr(self, '_received_artifacts'):
            return list(self._received_artifacts.keys())
        return []

    def broadcast_artifact(self, artifact_id: str, metadata: Optional[dict] = None):
        """
        Broadcast an artifact to all peers in the mesh.
        Args:
            artifact_id: Artifact identifier
            metadata: Optional metadata to include/override
        """
        artifact_data = self.get_artifact(artifact_id)
        if artifact_data is None:
            logger.warning(f"Cannot broadcast artifact {artifact_id}: not found.")
            return
        with self.lock:
            peer_ids = list(self.peers.keys())
        for peer_id in peer_ids:
            self.send_artifact(peer_id, artifact_id, artifact_data, metadata)
        logger.info(f"Broadcasted artifact {artifact_id} to {len(peer_ids)} peers.")

    def fetch_artifact_from_peer(self, peer_id: str, artifact_id: str, timeout: float = 30.0) -> Optional[bytes]:
        """
        Request an artifact from a peer. Sends a fetch request and waits for response.
        Args:
            peer_id: Peer node ID to request from
            artifact_id: Artifact identifier
            timeout: How long to wait for response (seconds)
        Returns:
            Artifact bytes if received, else None
        """
        # Send a fetch request message
        fetch_msg = {
            'type': 'artifact_fetch_request',
            'artifact_id': artifact_id,
            'requester': self.node_id,
            'timestamp': datetime.now().isoformat(),
        }
        try:
            self.secure_channel.send_message(peer_id, fetch_msg)
            logger.info(f"Sent artifact fetch request for {artifact_id} to {peer_id}")
        except Exception as e:
            logger.error(f"Failed to send fetch request to {peer_id}: {e}")
            return None

        # Wait for artifact to arrive (polling _received_artifacts or BlackVault)
        start = time.time()
        while time.time() - start < timeout:
            data = self.get_artifact(artifact_id)
            if data is not None:
                logger.info(f"Fetched artifact {artifact_id} from {peer_id}")
                return data
            time.sleep(0.5)
        logger.warning(f"Timeout waiting for artifact {artifact_id} from {peer_id}")
        return None

    def send_artifact(self, peer_id: str, artifact_id: str, artifact_data: bytes, metadata: Optional[dict] = None) -> bool:
        """
        Send an artifact (payload) to a peer node.
        Args:
            peer_id: Target peer node ID
            artifact_id: Unique identifier for the artifact
            artifact_data: Raw bytes of the artifact (should be base64-encoded for transport)
            metadata: Optional metadata dict (e.g., type, description)
        Returns:
            True if sent successfully, False otherwise
        """
        import base64
        payload = {
            'artifact_id': artifact_id,
            'artifact_data': base64.b64encode(artifact_data).decode(),
            'metadata': metadata or {},
            'sender': self.node_id,
            'timestamp': datetime.now().isoformat(),
        }
        msg = NetworkMessage(
            message_id=hashlib.md5(f"artifact_{artifact_id}_{time.time()}".encode()).hexdigest()[:16],
            message_type=MessageType.ARTIFACT_TRANSFER,
            sender=self.node_id,
            recipient=peer_id,
            payload=payload,
            timestamp=datetime.now()
        )
        try:
            self.secure_channel.send_message(peer_id, msg.to_dict())
            logger.info(f"Sent artifact {artifact_id} to {peer_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send artifact {artifact_id} to {peer_id}: {e}")
            return False

    def receive_artifact(self, message: NetworkMessage, relay: bool = False, relay_targets: Optional[list] = None):
        """
        Handle an incoming artifact transfer message. Optionally relay to other peers.
        Args:
            message: NetworkMessage of type ARTIFACT_TRANSFER
            relay: If True, relay artifact to other peers
            relay_targets: Optional list of peer_ids to relay to (default: all except sender)
        """
        import base64
        payload = message.payload
        artifact_id = payload.get('artifact_id')
        artifact_data_b64 = payload.get('artifact_data')
        metadata = payload.get('metadata', {})
        sender = payload.get('sender')
        try:
            if not artifact_data_b64:
                logger.error(f"Received artifact {artifact_id} from {sender} missing data (artifact_data_b64 is None or empty)")
                return
            artifact_data = base64.b64decode(artifact_data_b64)
            # Store in BlackVault if available, else in-memory
            stored = False
            try:
                from black_vault import BlackVault
                if not hasattr(self, '_artifact_vault'):
                    self._artifact_vault = BlackVault()
                # Try both possible method names for storing
                artifact_id_str = str(artifact_id) if artifact_id is not None else None
                if artifact_id_str is None:
                    raise ValueError("artifact_id is None")
                if hasattr(self._artifact_vault, 'store'):
                    try:
                        self._artifact_vault.store(artifact_id_str, artifact_data, metadata)
                        logger.info(f"Stored artifact {artifact_id_str} in BlackVault (store).")
                        stored = True
                    except Exception as e:
                        logger.warning(f"store failed: {e}")
                else:
                    logger.warning("No store method on BlackVault; storing in memory.")
            except Exception as e:
                logger.warning(f"BlackVault unavailable or failed, storing artifact {artifact_id} in memory: {e}")
            if not stored:
                if not hasattr(self, '_received_artifacts'):
                    self._received_artifacts = {}
                self._received_artifacts[artifact_id] = {
                    'data': artifact_data,
                    'metadata': metadata,
                    'sender': sender,
                    'received_at': datetime.now(),
                }
            logger.info(f"Received artifact {artifact_id} from {sender} (size: {len(artifact_data)} bytes)")
            # Optionally relay artifact to other peers
            if relay and artifact_id is not None:
                self.relay_artifact(str(artifact_id), artifact_data, metadata, exclude=[sender] if sender else None, targets=relay_targets)
        except Exception as e:
            logger.error(f"Failed to process received artifact {artifact_id} from {sender}: {e}")

    def relay_artifact(self, artifact_id: str, artifact_data: bytes, metadata: Optional[dict] = None, exclude: Optional[list] = None, targets: Optional[list] = None):
        """
        Relay an artifact to other peers (except those in exclude list).
        Args:
            artifact_id: Artifact identifier
            artifact_data: Raw bytes
            metadata: Optional metadata
            exclude: List of peer_ids to exclude (e.g., sender)
            targets: If provided, only relay to these peer_ids
        """
        with self.lock:
            peer_ids = set(self.peers.keys())
            if exclude:
                peer_ids -= set(exclude)
            if targets:
                peer_ids &= set(targets)
        for peer_id in peer_ids:
            self.send_artifact(peer_id, artifact_id, artifact_data, metadata)

    def get_artifact(self, artifact_id: str) -> Optional[bytes]:
        """
        Retrieve an artifact by ID from BlackVault or memory.
        Args:
            artifact_id: Artifact identifier
        Returns:
            Artifact bytes if found, else None
        """
        # Try BlackVault first
        try:
            from black_vault import BlackVault
            if not hasattr(self, '_artifact_vault'):
                self._artifact_vault = BlackVault()
            artifact_id_str = str(artifact_id) if artifact_id is not None else None
            if artifact_id_str is None:
                return None
            if hasattr(self._artifact_vault, 'retrieve'):
                try:
                    return self._artifact_vault.retrieve(artifact_id_str)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback to in-memory
        if hasattr(self, '_received_artifacts') and artifact_id in self._received_artifacts:
            return self._received_artifacts[artifact_id]['data']
        return None

    def broadcast_capabilities(self, node_id: str, capabilities: list):
        """
        Broadcast this node's capabilities to all peers (for anomaly integration).
        """
        msg = {
            'type': 'capabilities_broadcast',
            'node_id': node_id,
            'capabilities': capabilities,
            'timestamp': datetime.now().isoformat()
        }
        with self.lock:
            for peer_id in self.peers:
                try:
                    self.secure_channel.send_message(peer_id, msg)
                    logger.info(f"Broadcasted capabilities to {peer_id}: {capabilities}")
                except Exception as e:
                    logger.warning(f"Failed to broadcast capabilities to {peer_id}: {e}")
    """
    Enhanced SwarmMesh implementation for distributed peer-to-peer networking
    with secure communication, leader election, and dynamic module execution.
    """
    
    def __init__(self, 
                 node_id: str, 
                 listen_port: int = 0,
                 discovery_port: int = 9999,
                 max_peers: int = 50,
                 roles: Optional[Set[str]] = None):
        """
        Initialize SwarmMesh node as a real BlackrootHost.
        """
        self.host = BlackrootHost(node_id=node_id, roles=roles)
        self.node_id = self.host.node_id
        self.listen_port = listen_port
        self.discovery_port = discovery_port
        self.max_peers = max_peers
        self.redis = Redis(host="localhost", port=6379, decode_responses=True)

        # Network state
        self.status = NodeStatus.DISCONNECTED
        self.peers: Dict[str, BlackrootHost] = {}  # peer_id -> BlackrootHost
        self.leader_id: Optional[str] = None
        self.election_in_progress = False

        # Command handling
        self.pending_commands: List[Command] = []
        self.command_history: List[Command] = []
        self.max_history = 1000

        # Security and communication
        self.secure_channel = SecureChannel(self.node_id)
        self.auth_tokens: Dict[str, str] = {}  # peer_id -> token
        self.channel = 'blackroot_swarm'  # For compatibility with existing code
        # Threading and lifecycle
        self.running = False
        self.threads: List[threading.Thread] = []
        self.lock = threading.RLock()

        # Statistics
        self.stats = {
            'commands_executed': 0,
            'commands_failed': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'uptime_start': datetime.now()
        }

        # Enhanced components
        self.load_balancer = LoadBalancer(self)
        self.consistency_manager = ConsistencyManager(self)
        self.security_manager = SecurityManager(self)
        self.metrics_collector = MetricsCollector(self)

        logger.info(f"SwarmMesh host {self.node_id} initialized with roles: {self.host.roles}")

    def connect(self, bootstrap_peers: Optional[List[Tuple[str, int]]] = None):
        """
        Connect to the swarm mesh network
        
        Args:
            bootstrap_peers: List of (address, port) tuples for initial connections
        """
        if self.status != NodeStatus.DISCONNECTED:
            logger.warning(f"Node {self.node_id} already connected")
            return
            
        logger.info(f"Connecting to mesh as node {self.node_id}")
        
        try:
            self.status = NodeStatus.CONNECTING
            self.running = True
            
            # Start background threads
            self._start_background_threads()
            
            # Bootstrap with provided peers
            if bootstrap_peers:
                self._bootstrap_with_peers(bootstrap_peers)
            
            # Start peer discovery
            self.broadcast_identity()
            
            self.status = NodeStatus.CONNECTED
            logger.info(f"Node {self.node_id} successfully connected to mesh")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.status = NodeStatus.ERROR
            self.disconnect()

    def disconnect(self):
        """Disconnect from the swarm mesh and cleanup resources"""
        logger.info(f"Node {self.node_id} disconnecting from mesh")
        
        self.running = False
        self.status = NodeStatus.DISCONNECTED
        
        # Stop all background threads
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.threads.clear()
        self.peers.clear()
        self.leader_id = None
        
        logger.info(f"Node {self.node_id} disconnected")

    def _start_background_threads(self):
        """Start background threads for mesh operations"""
        threads_config = [
            (self._heartbeat_loop, "Heartbeat"),
            (self._peer_discovery_loop, "PeerDiscovery"),
            (self._command_processor_loop, "CommandProcessor"),
            (self._cleanup_loop, "Cleanup"),
            (self._metrics_collection_loop, "MetricsCollection"),
            (self._consistency_sync_loop, "ConsistencySync")
        ]
        
        for target, name in threads_config:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)
            logger.debug(f"Started {name} thread")

    def _bootstrap_with_peers(self, bootstrap_peers: List[Tuple[str, int]]):
        """Connect to bootstrap peers to join the network"""
        for address, port in bootstrap_peers:
            try:
                self._connect_to_peer(address, port)
                logger.info(f"Connected to bootstrap peer {address}:{port}")
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {address}:{port}: {e}")

    def _connect_to_peer(self, address: str, port: int) -> bool:
        """Attempt to establish connection with a specific peer"""
        # This would contain actual socket connection logic
        # For now, we'll simulate the connection
        peer_id = f"peer_{address}_{port}"
        
        with self.lock:
            if len(self.peers) >= self.max_peers:
                logger.warning("Maximum peer limit reached")
                return False
                
            peer = BlackrootHost(node_id=peer_id)
            peer.address = address
            peer.port = port
            from datetime import datetime
            peer.last_seen = datetime.now()
            peer.status = NodeStatus.CONNECTED
            peer.capabilities = []
            self.peers[peer_id] = peer
        
        logger.debug(f"Established connection with peer {peer_id}")
        return True

    def broadcast_identity(self):
        """Broadcast this node's identity to discover peers"""
        identity_message = {
            'type': 'identity_broadcast',
            'node_id': self.node_id,
            'capabilities': self._get_node_capabilities(),
            'timestamp': datetime.now().isoformat(),
            'discovery_port': self.discovery_port
        }
        
        try:
            # In a real implementation, this would use UDP multicast or a discovery service
            logger.info("Broadcasting identity for peer discovery")
            self._simulate_peer_discovery()
            
        except Exception as e:
            logger.error(f"Failed to broadcast identity: {e}")

    def _simulate_peer_discovery(self):
        """Simulate peer discovery for demonstration purposes"""
        # Add some simulated peers
        simulated_peers = [
            ("192.168.1.100", 8001),
            ("192.168.1.101", 8002),
            ("192.168.1.102", 8003)
        ]
        
        for addr, port in simulated_peers:
            if random.random() < 0.7:  # 70% chance of discovery
                self._connect_to_peer(addr, port)

    def _get_node_capabilities(self) -> List[str]:
        """Get list of capabilities this node provides"""
        capabilities = ['mesh_networking', 'command_execution']
        
        # Add capabilities from registered modules
        for module_info in module_registry.values():
            capabilities.extend(module_info.get('capabilities', []))
        
        return list(set(capabilities))  # Remove duplicates

    def ping(self) -> Dict[str, bool]:
        """
        Ping all connected peers to check connectivity
        
        Returns:
            Dictionary mapping peer_id to ping success status
        """
        results = {}
        
        with self.lock:
            peers_to_ping = list(self.peers.keys())
        
        logger.info(f"Pinging {len(peers_to_ping)} peers")
        
        for peer_id in peers_to_ping:
            try:
                # Simulate ping with random success/failure
                success = random.random() > 0.1  # 90% success rate
                results[peer_id] = success
                
                if success:
                    with self.lock:
                        if peer_id in self.peers:
                            self.peers[peer_id].last_seen = datetime.now()
                else:
                    logger.warning(f"Ping failed for peer {peer_id}")
                    
            except Exception as e:
                logger.error(f"Error pinging peer {peer_id}: {e}")
                results[peer_id] = False
        
        return results

    def _heartbeat_loop(self):
        """Background thread for sending periodic heartbeats"""
        while self.running:
            try:
                self._send_heartbeats()
                time.sleep(10)  # Send heartbeat every 10 seconds
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    def _send_heartbeats(self):
        """Send heartbeat messages to all peers"""
        heartbeat_msg = {
            'type': 'heartbeat',
            'node_id': self.node_id,
            'timestamp': datetime.now().isoformat(),
            'status': self.status.value
        }
        
        with self.lock:
            for peer_id in list(self.peers.keys()):
                try:
                    self.secure_channel.send_message(peer_id, heartbeat_msg)
                    self.stats['messages_sent'] += 1
                except Exception as e:
                    logger.debug(f"Failed to send heartbeat to {peer_id}: {e}")

    def _peer_discovery_loop(self):
        """Background thread for continuous peer discovery"""
        while self.running:
            try:
                # Periodically rediscover peers
                if random.random() < 0.3:  # 30% chance every cycle
                    self.broadcast_identity()
                time.sleep(30)  # Discovery every 30 seconds
            except Exception as e:
                logger.error(f"Peer discovery loop error: {e}")

    def _command_processor_loop(self):
        """Background thread for processing commands, artifacts, fetch and delete requests"""
        while self.running:
            try:
                # Pull all messages (commands, artifacts, fetch/delete requests)
                raw_msgs = self.secure_channel.pull_messages()
                commands = []
                for raw_msg in raw_msgs:
                    # If it's a command dict
                    if isinstance(raw_msg, dict) and 'command_type' in raw_msg:
                        try:
                            command = Command.from_dict(raw_msg)
                            commands.append(command)
                            self.stats['messages_received'] += 1
                        except Exception as e:
                            logger.error(f"Failed to parse command: {e}")
                    # If it's a NetworkMessage dict of type ARTIFACT_TRANSFER
                    elif isinstance(raw_msg, dict) and raw_msg.get('message_type') == MessageType.ARTIFACT_TRANSFER.value:
                        try:
                            msg_obj = NetworkMessage.from_dict(raw_msg)
                            self.receive_artifact(msg_obj)
                            self.stats['messages_received'] += 1
                        except Exception as e:
                            logger.error(f"Failed to process artifact message: {e}")
                    # If it's an artifact fetch request
                    elif isinstance(raw_msg, dict) and raw_msg.get('type') == 'artifact_fetch_request':
                        artifact_id = raw_msg.get('artifact_id')
                        requester = raw_msg.get('requester')
                        if artifact_id and requester:
                            artifact_data = self.get_artifact(artifact_id)
                            if artifact_data is not None:
                                self.send_artifact(requester, artifact_id, artifact_data)
                                logger.info(f"Responded to artifact fetch request for {artifact_id} from {requester}")
                    # If it's an artifact delete request
                    elif isinstance(raw_msg, dict) and raw_msg.get('type') == 'artifact_delete_request':
                        artifact_id = raw_msg.get('artifact_id')
                        requester = raw_msg.get('requester')
                        if artifact_id:
                            deleted = self.delete_artifact(artifact_id)
                            logger.info(f"Artifact {artifact_id} delete request from {requester}: {'deleted' if deleted else 'not found'}")
                # Add any pending commands from local injection
                with self.lock:
                    commands.extend(self.pending_commands)
                    self.pending_commands.clear()
                for command in commands:
                    self._process_command_async(command)
                time.sleep(1)  # Check for messages every second
            except Exception as e:
                logger.error(f"Command processor loop error: {e}")

    def _cleanup_loop(self):
        """Background thread for periodic cleanup tasks"""
        while self.running:
            try:
                self._cleanup_dead_peers()
                self._cleanup_command_history()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    def _metrics_collection_loop(self):
        """Background thread for metrics collection"""
        while self.running:
            try:
                self.metrics_collector.collect_metrics()
                time.sleep(self.metrics_collector.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")

    def _consistency_sync_loop(self):
        """Background thread for data consistency synchronization"""
        while self.running:
            try:
                self.consistency_manager.sync_with_peers()
                time.sleep(120)  # Sync every 2 minutes
            except Exception as e:
                logger.error(f"Consistency sync loop error: {e}")

    def _cleanup_dead_peers(self):
        """Remove peers that haven't been seen recently"""
        with self.lock:
            dead_peers = []
            for peer_id, peer_info in self.peers.items():
                if not peer_info.is_alive():
                    dead_peers.append(peer_id)
            
            for peer_id in dead_peers:
                del self.peers[peer_id]
                logger.info(f"Removed dead peer: {peer_id}")
                
            # Trigger leader election if leader is dead
            if self.leader_id in dead_peers:
                self.leader_id = None
                self.elect_leader()

    def _cleanup_command_history(self):
        """Clean up old commands from history"""
        with self.lock:
            if len(self.command_history) > self.max_history:
                excess = len(self.command_history) - self.max_history
                self.command_history = self.command_history[excess:]

    def receive_commands(self) -> List[Command]:
        """
        Pull and process commands from secure channel
        
        Returns:
            List of Command objects ready for execution
        """
        try:
            raw_commands = self.secure_channel.pull_messages()
            commands = []
            
            for raw_cmd in raw_commands:
                try:
                    if isinstance(raw_cmd, dict) and 'command_type' in raw_cmd:
                        command = Command.from_dict(raw_cmd)
                        commands.append(command)
                        self.stats['messages_received'] += 1
                except Exception as e:
                    logger.error(f"Failed to parse command: {e}")
            
            # Add any pending commands from local injection
            with self.lock:
                commands.extend(self.pending_commands)
                self.pending_commands.clear()
            
            return commands
            
        except Exception as e:
            logger.error(f"Error receiving commands: {e}")
            return []

    def _process_command_async(self, command: Command):
        """Process a single command asynchronously"""
        try:
            result = self.execute_command(command)
            
            with self.lock:
                self.command_history.append(command)
                
            if result is not None:
                self.stats['commands_executed'] += 1
                logger.debug(f"Command {command.command_id} executed successfully")
            else:
                self.stats['commands_failed'] += 1
                logger.warning(f"Command {command.command_id} failed")
                
        except Exception as e:
            logger.error(f"Error processing command {command.command_id}: {e}")
            self.stats['commands_failed'] += 1

    def execute_command(self, command: Command) -> Any:
        """
        Execute a command with enhanced security and validation
        
        Args:
            command: Command object to execute
            
        Returns:
            Result of command execution or None if failed
        """
        try:
            # Validate command
            if not self._validate_command(command):
                logger.warning(f"Command validation failed: {command.command_id}")
                return None
            
            target = command.target_module
            action = command.action
            
            if target not in module_registry:
                logger.error(f"Unknown module: {target}")
                return None
            
            module_info = module_registry[target]
            handler = module_info['initializer']
            
            # Update module statistics
            module_info['call_count'] += 1
            module_info['last_called'] = datetime.now()
            
            logger.info(f"Executing {action} on {target} (Command ID: {command.command_id})")
            
            if callable(handler):
                # Execute with timeout protection
                return self._execute_with_timeout(
                    handler, 
                    command.args, 
                    command.kwargs,
                    command.timeout
                )
            else:
                return handler
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return None

    def _validate_command(self, command: Command) -> bool:
        """
        Validate command for security and correctness
        
        Args:
            command: Command to validate
            
        Returns:
            True if command is valid, False otherwise
        """
        # Check command age
        age = (datetime.now() - command.timestamp).total_seconds()
        if age > 300:  # 5 minutes
            logger.warning(f"Command too old: {age} seconds")
            return False
        
        # Check if module exists and is safe
        if command.target_module not in module_registry:
            logger.warning(f"Unknown target module: {command.target_module}")
            return False
        
        # Add more validation rules as needed
        # - Check sender authorization
        # - Validate argument types
        # - Check resource limits
        
        return True

    def _execute_with_timeout(self, 
                            func: Callable, 
                            args: List[Any], 
                            kwargs: Dict[str, Any],
                            timeout: float) -> Any:
        """Execute function with timeout protection"""
        # In a real implementation, this would use proper timeout mechanisms
        # For now, we'll just execute directly
        return func(*args, **kwargs)

    def inject_command(self, 
                      target_module: str,
                      action: str,
                      args: Optional[List[Any]] = None,
                      kwargs: Optional[Dict[str, Any]] = None,
                      priority: int = 1,
                      timeout: float = 30.0) -> str:
        """
        Inject a command for local execution
        
        Args:
            target_module: Module to target
            action: Action to perform
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Command priority (higher = more urgent)
            timeout: Execution timeout in seconds
            
        Returns:
            Command ID for tracking
        """
        command_id = hashlib.md5(
            f"{self.node_id}_{time.time()}_{random.randint(0, 10000)}".encode()
        ).hexdigest()[:12]
        
        command = Command(
            command_id=command_id,
            command_type=CommandType.EXECUTE,
            target_module=target_module,
            action=action,
            args=args or [],
            kwargs=kwargs or {},
            sender=self.node_id,
            timestamp=datetime.now(),
            priority=priority,
            timeout=timeout
        )
        
        with self.lock:
            # Insert based on priority
            inserted = False
            for i, existing_cmd in enumerate(self.pending_commands):
                if command.priority > existing_cmd.priority:
                    self.pending_commands.insert(i, command)
                    inserted = True
                    break
            
            if not inserted:
                self.pending_commands.append(command)
        
        logger.info(f"Injected command {command_id} for {target_module}.{action}")
        return command_id

    def elect_leader(self) -> Optional[str]:
        """
        Perform leader election using enhanced algorithm
        
        Returns:
            ID of elected leader or None if election failed
        """
        if self.election_in_progress:
            logger.info("Election already in progress")
            return self.leader_id
        
        self.election_in_progress = True
        
        try:
            with self.lock:
                # Get all alive nodes including self
                candidates = [self.node_id]
                for peer_id, peer_info in self.peers.items():
                    if peer_info.is_alive() and peer_info.status == NodeStatus.CONNECTED:
                        candidates.append(peer_id)
            
            if not candidates:
                logger.warning("No candidates for leader election")
                return None
            
            # Enhanced election: consider node capabilities and load
            best_candidate = self._select_best_leader(candidates)
            
            self.leader_id = best_candidate
            
            if best_candidate == self.node_id:
                self.status = NodeStatus.LEADER
                logger.info(f"Elected as leader: {self.node_id}")
            else:
                self.status = NodeStatus.CONNECTED
                logger.info(f"Leader elected: {self.leader_id}")
            
            return self.leader_id
            
        finally:
            self.election_in_progress = False

    def _select_best_leader(self, candidates: List[str]) -> str:
        """
        Select the best leader candidate based on various criteria
        
        Args:
            candidates: List of candidate node IDs
            
        Returns:
            ID of the best candidate
        """
        # For now, use a simple lexicographic ordering
        # In a real implementation, this could consider:
        # - Node capabilities
        # - Current load
        # - Network connectivity
        # - Historical performance
        
        return min(candidates)

    def get_network_status(self) -> Dict[str, Any]:
        """
        Get comprehensive network status information
        
        Returns:
            Dictionary containing network status details
        """
        with self.lock:
            uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()
            
            return {
                'node_id': self.node_id,
                'status': self.status.value,
                'leader_id': self.leader_id,
                'peer_count': len(self.peers),
                'pending_commands': len(self.pending_commands),
                'uptime_seconds': uptime,
                'statistics': self.stats.copy(),
                'capabilities': self._get_node_capabilities(),
                'peers': {
                    peer_id: {
                        'address': peer.address,
                        'port': peer.port,
                        'status': peer.status.value,
                        'last_seen': peer.last_seen.isoformat() if peer.last_seen else '',
                        'alive': peer.is_alive()
                    }
                    for peer_id, peer in self.peers.items()
                }
            }

    def get_command_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent command execution history
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of command dictionaries
        """
        with self.lock:
            recent_commands = self.command_history[-limit:]
            return [cmd.to_dict() for cmd in recent_commands]


# Example module implementations
def example_math_module(*args, **kwargs) -> float:
    """Example math operations module"""
    operation = kwargs.get('operation', 'add')
    numbers = args
    
    if operation == 'add':
        return sum(numbers)
    elif operation == 'multiply':
        result = 1
        for num in numbers:
            result *= num
        return result
    elif operation == 'average':
        return sum(numbers) / len(numbers) if numbers else 0
    else:
        raise ValueError(f"Unknown operation: {operation}")


def example_system_module(*args, **kwargs) -> Dict[str, Any]:
    """Example system information module"""
    info_type = kwargs.get('info_type', 'basic')
    
    if info_type == 'basic':
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time(),
            'random_value': random.randint(1, 1000)
        }
    elif info_type == 'detailed':
        return {
            'timestamp': datetime.now().isoformat(),
            'process_id': 12345,  # Mock PID
            'memory_usage': random.randint(100, 1000),  # Mock memory
            'cpu_usage': random.uniform(0, 100)  # Mock CPU
        }
    else:
        raise ValueError(f"Unknown info type: {info_type}")


# Register example modules
register_module(
    name="math_ops",
    initializer=example_math_module,
    capabilities=["arithmetic", "statistics"],
    version="1.0.0",
    description="Basic mathematical operations and statistics"
)

register_module(
    name="system_info",
    initializer=example_system_module,
    capabilities=["monitoring", "diagnostics"],
    version="1.0.0",
    description="System information and monitoring"
)


# Example usage and testing
def main():
    """Example usage of the enhanced SwarmMesh"""
    # Create a mesh node
    node = SwarmMesh("node_001", listen_port=8000)
    
    try:
        # Connect to the mesh
        node.connect()
        
        # Wait a moment for initialization
        time.sleep(2)
        
        # Inject some test commands
        cmd_id1 = node.inject_command(
            target_module="math_ops",
            action="calculate",
            args=[1, 2, 3, 4, 5],
            kwargs={'operation': 'add'}
        )
        
        cmd_id2 = node.inject_command(
            target_module="system_info",
            action="get_info",
            kwargs={'info_type': 'basic'}
        )
        
        # Let commands execute
        time.sleep(3)
        
        # Check network status
        status = node.get_network_status()
        print("\nNetwork Status:")
        print(json.dumps(status, indent=2, default=str))
        
        # Check command history
        history = node.get_command_history(10)
        print(f"\nCommand History ({len(history)} commands):")
        for cmd in history:
            print(f"- {cmd['command_id']}: {cmd['target_module']}.{cmd['action']}")
        
        # Perform leader election
        leader = node.elect_leader()
        print(f"\nElected leader: {leader}")
        
        # Test ping
        ping_results = node.ping()
        print(f"\nPing results: {ping_results}")
        
        # Keep running for a bit
        print("\nNode running... (Press Ctrl+C to stop)")
        time.sleep(20)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.disconnect()
        print("Node shutdown complete")


if __name__ == "__main__":
    main() 