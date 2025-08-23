import json
import time
import base64
import threading
import redis
from typing import List, Dict
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import queue

class SecureChannel:
    def send_message(self, target: str, message: Dict) -> bool:
        """Send a message to a target node (wrapper for push_message)."""
        # For compatibility with SwarmMesh, target is ignored (uses self.channel)
        try:
            self.push_message(message)
            return True
        except Exception as e:
            print(f"[SECURE] Failed to send message to {target}: {e}")
            return False
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.shared_key = self._derive_key("blackroot_mesh_secret")
        self.inbox = queue.Queue()
        self.history = []
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.channel = f"secure_channel:{self.node_id}"
        self.discovery_channel = "blackroot:discovery"
        self._start_listener()
        self._start_discovery_broadcast()
        self.known_peers = set()

    def _derive_key(self, passphrase: str) -> bytes:
        return pad(passphrase.encode(), 16)[:16]  # AES-128

    def encrypt(self, data: Dict) -> str:
        raw = json.dumps(data).encode()
        iv = get_random_bytes(16)
        cipher = AES.new(self.shared_key, AES.MODE_CBC, iv)
        ct = cipher.encrypt(pad(raw, 16))
        return base64.b64encode(iv + ct).decode()

    def decrypt(self, encrypted: str) -> Dict:
        blob = base64.b64decode(encrypted.encode())
        iv = blob[:16]
        ct = blob[16:]
        cipher = AES.new(self.shared_key, AES.MODE_CBC, iv)
        raw = unpad(cipher.decrypt(ct), 16)
        return json.loads(raw.decode())

    def push_message(self, message: Dict):
        encoded = self.encrypt(message)
        self.redis.publish(self.channel, encoded)
        self.history.append(("outgoing", message))

    def pull_messages(self) -> List[Dict]:
        results = []
        while not self.inbox.empty():
            try:
                raw = self.inbox.get()
                decoded = self.decrypt(raw)
                results.append(decoded)
                self.history.append(("incoming", decoded))
            except Exception as e:
                print(f"[SECURE] Failed to decrypt message: {e}")
        return results

    def simulate_incoming(self, message: Dict):
        # For test/diagnostic use
        self.inbox.put(self.encrypt(message))

    def _start_listener(self):
        def listen():
            pubsub = self.redis.pubsub()
            pubsub.subscribe(self.channel)
            print(f"[SECURE] Listening on {self.channel}")
            for message in pubsub.listen():
                if message['type'] == 'message':
                    self.inbox.put(message['data'])

        threading.Thread(target=listen, daemon=True).start()

    def _start_discovery_broadcast(self):
        def announce():
            while True:
                payload = json.dumps({"node_id": self.node_id})
                self.redis.publish(self.discovery_channel, payload)
                time.sleep(10)

        def receive():
            pubsub = self.redis.pubsub()
            pubsub.subscribe(self.discovery_channel)
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        sender_id = data.get("node_id")
                        if sender_id and sender_id != self.node_id:
                            self.known_peers.add(sender_id)
                            print(f"[DISCOVERY] Peer discovered: {sender_id}")
                    except Exception as e:
                        print(f"[DISCOVERY] Error: {e}")

        threading.Thread(target=announce, daemon=True).start()
        threading.Thread(target=receive, daemon=True).start()

    def get_known_peers(self) -> List[str]:
        return list(self.known_peers)
