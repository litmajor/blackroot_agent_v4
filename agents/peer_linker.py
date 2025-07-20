import zmq
import threading
import time
import json
import socket
from cryptography.fernet import Fernet
from agents.base import BaseAgent

class PeerLinkerAgent(BaseAgent):
    def __init__(self):
        super().__init__('PEER-LINKER')
        self.listen_port = 5555
        self.discovery_port = 5558
        self.known_peers = set()
        self.peer_keys = {}        # peer_addr -> Fernet key
        self.peer_last_seen = {}   # peer_addr -> last timestamp
        self.context = zmq.Context()
        self.running = True

        self.secret_key = Fernet.generate_key()
        self.cipher = Fernet(self.secret_key)
        self.real_ip = socket.gethostbyname(socket.gethostname())

    def run(self):
        super().run()
        print("[PEER-LINKER] Launching peer sync + discovery threads...")
        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._sync_loop, daemon=True).start()
        threading.Thread(target=self._discovery_loop, daemon=True).start()
        threading.Thread(target=self._peer_cleanup_loop, daemon=True).start()

        while self.running:
            time.sleep(1)

    def _listen_loop(self):
        socket_sub = self.context.socket(zmq.SUB)
        socket_sub.bind(f"tcp://*:{self.listen_port}")
        socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[PEER-LINKER] Listening on port {self.listen_port}...")

        while self.running:
            try:
                msg = socket_sub.recv()
                decrypted = self.cipher.decrypt(msg).decode()
                data = json.loads(decrypted)
                print(f"[PEER-LINKER] Received from peer: {data}")
                self._handle_peer_data(data)
            except Exception as e:
                print(f"[PEER-LINKER] Listen error: {e}")

    def _sync_loop(self):
        socket_pub = self.context.socket(zmq.PUB)
        for peer in self.known_peers:
            try:
                socket_pub.connect(f"tcp://{peer}")
            except Exception as e:
                print(f"[PEER-LINKER] Could not connect to peer {peer}: {e}")

        while self.running:
            payload = {
                "source": self.codename,
                "ip": self.real_ip,
                "timestamp": time.time(),
                "beliefs": getattr(self.kernel, 'mirrorcore', {}).beliefs,
                "emotions": getattr(self.kernel, 'mirrorcore', {}).emotions,
                "missions": getattr(self.kernel, 'mirrorcore', {}).mission_queue,
                "memory": self.kernel.memory.recall() if hasattr(self.kernel, 'memory') else {},
                "agent_status": {
                    k: str(v.status)
                    for k, v in self.kernel.agents.items()
                } if hasattr(self.kernel, 'agents') else {}
            }

            for peer, key in self.peer_keys.items():
                peer_cipher = Fernet(key.encode())
                try:
                    encrypted = peer_cipher.encrypt(json.dumps(payload).encode())
                    socket_pub.send(encrypted)
                    print(f"[PEER-LINKER] Sent encrypted sync to {peer}")
                except Exception as e:
                    print(f"[PEER-LINKER] Sync to {peer} failed: {e}")

            time.sleep(10)

    def _discovery_loop(self):
        disc_socket = self.context.socket(zmq.REQ)
        disc_socket.RCVTIMEO = 2000
        for port in range(5555, 5560):
            if port == self.listen_port:
                continue
            try:
                peer_addr = f"{self.real_ip}:{port}"
                disc_socket.connect(f"tcp://{peer_addr}")
                disc_socket.send_string(json.dumps({"type": "peer_hello", "from": self.codename}))
                reply = disc_socket.recv_string()
                data = json.loads(reply)
                if data.get("type") == "peer_ack":
                    self.known_peers.add(peer_addr)
                    if 'key' in data:
                        self.peer_keys[peer_addr] = data['key']
                    print(f"[PEER-LINKER] Discovered peer at {peer_addr}")
            except Exception:
                continue

    def _handle_peer_data(self, data):
        peer_ip = data.get("ip")
        if peer_ip:
            self.peer_last_seen[peer_ip] = time.time()

        if hasattr(self.kernel, 'mirrorcore'):
            if 'beliefs' in data:
                self.kernel.mirrorcore.inject_beliefs(data['beliefs'])
            if 'emotions' in data:
                self.kernel.mirrorcore.inject_emotions(data['emotions'])
            if 'missions' in data:
                self.kernel.mirrorcore.import_peer_missions(data['missions'])

        if 'memory' in data and hasattr(self.kernel, 'memory'):
            peer_id = data.get("source", "unknown")
            self.kernel.memory.import_peer_memory(data['memory'], source=peer_id)

        if 'agent_status' in data:
            print(f"[PEER-LINKER] Peer agent status: {data['agent_status']}")

    def _peer_cleanup_loop(self):
        while self.running:
            now = time.time()
            expired_peers = [peer for peer, ts in self.peer_last_seen.items() if now - ts > 180]
            for peer in expired_peers:
                print(f"[PEER-LINKER] Peer {peer} expired. Removing.")
                self.known_peers.discard(peer)
                self.peer_keys.pop(peer, None)
                self.peer_last_seen.pop(peer, None)
            time.sleep(60)

    import time

def import_peer_memory(self, peer_data: dict, source: str = "unknown_peer"):
    new_keys = 0
    for key, value in peer_data.items():
        if key not in self.memory:
            self.memory[key] = {
                "value": value,
                "source": source,
                "timestamp": time.time()
            }
            new_keys += 1
        else:
            # Overwrite if peer version is newer or different
            existing = self.memory[key]
            if existing["value"] != value:
                self.memory[key] = {
                    "value": value,
                    "source": source,
                    "timestamp": time.time()
                }
                new_keys += 1
    if new_keys > 0:
        print(f"[MEMORY] Imported {new_keys} peer memory items from {source}")

