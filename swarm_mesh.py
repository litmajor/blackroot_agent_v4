import json
import time
import random
from typing import Dict, List, Callable

from secure_channel import SecureChannel  # module for encrypted messaging


# Global registry for dynamic modules
module_registry: Dict[str, Callable] = {}

def register_module(name: str, initializer: Callable):
    module_registry[name] = initializer
    print(f"[🔧] Module registered: {name}")

class SwarmMesh:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.peers: Dict[str, str] = {}  # peer_id -> address
        self.pending_commands: List[Dict] = []
        self.secure_channel = SecureChannel(self.node_id)

    def connect(self):
        print(f"[SWARM] Connecting to mesh as node {self.node_id}")
        self.broadcast_identity()

    def disconnect(self):
        print("[SWARM] Disconnected from mesh")

    def broadcast_identity(self):
        print("[SWARM] Broadcasting presence to peers")
        # Placeholder: This would normally contact a discovery service or use multicast

    def ping(self):
        print(f"[SWARM] Pinging peers: {list(self.peers.keys())}")

    def receive_commands(self) -> List[Dict]:
        # Pull commands from a secure channel or memory inbox
        commands = self.secure_channel.pull_messages()
        if commands:
            self.pending_commands.extend(commands)
        return self.flush_commands()

    def flush_commands(self) -> List[Dict]:
        commands = self.pending_commands[:]
        self.pending_commands = []
        return commands

    def execute_command(self, command: Dict):

        target = command.get("target_module")
        action = command.get("action")
        args = command.get("args", [])

        if target in module_registry:
            handler = module_registry[target]
            print(f"[C2] Executing {action} on {target}")
            if callable(handler):
                return handler(*args)
            return handler
        else:
            print(f"[C2] Unknown module: {target}")
            return None

    def inject_command(self, command: Dict):
        self.pending_commands.append(command)

    def elect_leader(self):
        # Example naive election
        candidates = list(self.peers.keys()) + [self.node_id]
        self.leader_id = min(candidates)
        print(f"[SWARM] Leader elected: {self.leader_id}")
