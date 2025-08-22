import zmq
import threading
import time
import json
import socket
import secrets
import struct
import hashlib
from datetime import datetime, timedelta
from urllib.parse import urlparse
from contextlib import contextmanager
import logging

import nacl.utils
import nacl.bindings as b
from cryptography.fernet import Fernet  # legacy fallback only
from agents.base import BaseAgent

try:
    import miniupnpc
except ImportError:
    miniupnpc = None
try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    _zeroconf = Zeroconf()
    _ServiceListenerBase = ServiceListener
except ImportError:
    class ServiceListener:
        def add_service(self, zc, type_, name):
            pass
        def remove_service(self, zc, type_, name):
            pass
        def update_service(self, zc, type_, name):
            pass
    ServiceBrowser = None
    Zeroconf = None
    _zeroconf = None
    _ServiceListenerBase = ServiceListener

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter to prevent spam and DoS attacks"""
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}  # peer_id -> [timestamps]
        self.lock = threading.Lock()
    
    def is_allowed(self, peer_id):
        now = time.time()
        with self.lock:
            if peer_id not in self.requests:
                self.requests[peer_id] = []
            
            # Clean old requests
            cutoff = now - self.time_window
            self.requests[peer_id] = [
                ts for ts in self.requests[peer_id] if ts > cutoff
            ]
            
            # Check rate limit
            if len(self.requests[peer_id]) >= self.max_requests:
                return False
            
            # Record this request
            self.requests[peer_id].append(now)
            return True


class PeerLinkerAgent(BaseAgent):
    LISTEN_PORT         = 5555
    DISCOVERY_STUN_HOST = "stun.l.google.com"
    DISCOVERY_STUN_PORT = 19302
    DHT_K               = 8
    ID_BITS             = 160
    KEY_ROTATE_MSGS     = 100
    KEY_ROTATE_SECONDS  = 600
    UPnP_LEASE_SEC      = 3600
    MAX_MESSAGE_SIZE    = 1024 * 1024  # 1MB
    SOCKET_TIMEOUT      = 5000  # 5 seconds

    def __init__(self):
        super().__init__('PEER-LINKER')

        # ---- Crypto ----
        self.id_sk, self.id_pk = b.crypto_sign_keypair()
        self.node_id = hashlib.sha256(self.id_pk).digest()[:20]
        self.ephemeral_sk, self.ephemeral_pk = b.crypto_kx_keypair()
        self.session_keys = {}     # peer_id -> {'tx','rx','msg_cnt','last'}

        # ---- Threading & Safety ----
        self._session_lock = threading.RLock()
        self._peers_lock = threading.RLock()
        self.running = True
        self.rate_limiter = RateLimiter(max_requests=50, time_window=60)

        # ---- Networking ----
        self.real_ip = self._get_real_ip()
        self.external_addr = None
        self.known_peers = set()
        self.kbuckets = KBucketTable(self.node_id, k=self.DHT_K)
        self.context = zmq.Context()
        self.socket_pool = {}  # (socket_type, identity) -> socket

        # Initialize network
        self._map_port()
        self._discover_external()

    # ------------------------------------------------------------------
    #  Thread-safe context managers
    # ------------------------------------------------------------------
    @contextmanager
    def _session_context(self):
        """Context manager for thread-safe session operations"""
        with self._session_lock:
            yield
            
    @contextmanager
    def _peers_context(self):
        """Context manager for thread-safe peer operations"""
        with self._peers_lock:
            yield

    # ------------------------------------------------------------------
    #  Public lifecycle
    # ------------------------------------------------------------------
    def run(self):
        super().run()
        logger.info("Launching enhanced peer sync + discovery...")
        
        # Start all background threads
        threads = [
            threading.Thread(target=self._listen_loop, daemon=True, name="PeerLinker-Listen"),
            threading.Thread(target=self._sync_loop, daemon=True, name="PeerLinker-Sync"),
            threading.Thread(target=self._discovery_loop, daemon=True, name="PeerLinker-Discovery"),
            threading.Thread(target=self._peer_cleanup_loop, daemon=True, name="PeerLinker-Cleanup"),
            threading.Thread(target=self._key_rotation_loop, daemon=True, name="PeerLinker-KeyRotation")
        ]
        
        for thread in threads:
            thread.start()
            logger.info(f"Started {thread.name}")
        
        # Main loop
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.shutdown()

    def shutdown(self):
        """Clean shutdown with proper resource cleanup"""
        logger.info("Initiating shutdown...")
        self.running = False
        
        # Close all pooled sockets
        for sock in self.socket_pool.values():
            try:
                sock.close()
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
        
        # Terminate ZMQ context
        try:
            self.context.term()
        except Exception as e:
            logger.warning(f"Error terminating ZMQ context: {e}")
        
        logger.info("Clean shutdown completed")

    # ------------------------------------------------------------------
    #  Networking helpers
    # ------------------------------------------------------------------
    def _get_real_ip(self):
        try:
            # Try to connect to a remote address to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except socket.gaierror:
                return "127.0.0.1"

    def _map_port(self):
        """Attempt UPnP port mapping"""
        if not miniupnpc:
            logger.warning("miniupnpc not available, skipping UPnP")
            return
            
        try:
            u = miniupnpc.UPnP()
            u.discover()
            u.selectigd()
            result = u.addportmapping(
                self.LISTEN_PORT, "TCP", self.real_ip,
                self.LISTEN_PORT, "PeerLinker", self.UPnP_LEASE_SEC
            )
            if result:
                logger.info(f"UPnP mapped TCP {self.LISTEN_PORT}")
            else:
                logger.warning("UPnP mapping failed")
        except Exception as e:
            logger.warning(f"UPnP failed: {e}")

    def _discover_external(self):
        """Discover external IP/port using STUN"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        try:
            stun_request = self._build_stun_binding()
            sock.sendto(stun_request, (self.DISCOVERY_STUN_HOST, self.DISCOVERY_STUN_PORT))
            data, _ = sock.recvfrom(2048)
            ip, port = self._parse_stun_response(data)
            self.external_addr = (ip, port)
            logger.info(f"External endpoint discovered: {ip}:{port}")
        except Exception as e:
            logger.warning(f"STUN discovery failed: {e}")
            self.external_addr = (self.real_ip, self.LISTEN_PORT)
        finally:
            sock.close()

    def _build_stun_binding(self):
        """Build STUN binding request"""
        tid = secrets.token_bytes(12)
        return b'\x00\x01\x00\x08' + tid + b'\x00' * 8

    def _parse_stun_response(self, data):
        """Parse STUN response to extract mapped address"""
        if len(data) < 20 or data[0:2] != b'\x01\x01':
            raise ValueError("Invalid STUN response")
            
        attr_start = 20
        while attr_start + 4 <= len(data):
            attr_type, attr_len = struct.unpack('!HH', data[attr_start:attr_start + 4])
            if attr_type == 0x0001:  # MAPPED-ADDRESS
                family = data[attr_start + 5]
                if family == 0x01:  # IPv4
                    ip = socket.inet_ntoa(data[attr_start + 8:attr_start + 12])
                    port = struct.unpack('!H', data[attr_start + 6:attr_start + 8])[0]
                    return ip, port
            attr_start += 4 + attr_len + ((4 - attr_len % 4) % 4)
        raise ValueError("No MAPPED-ADDRESS found in STUN response")

    def _get_or_create_socket(self, socket_type, identity=None, connect_addr=None):
        """Get or create a socket with proper configuration"""
        try:
            sock = self.context.socket(socket_type)
            sock.setsockopt(zmq.RCVTIMEO, self.SOCKET_TIMEOUT)
            sock.setsockopt(zmq.SNDTIMEO, self.SOCKET_TIMEOUT)
            sock.setsockopt(zmq.LINGER, 1000)
            
            if identity:
                sock.setsockopt(zmq.IDENTITY, identity)
            
            if connect_addr:
                sock.connect(connect_addr)
                
            return sock
        except Exception as e:
            logger.error(f"Failed to create socket: {e}")
            raise

    # ------------------------------------------------------------------
    #  Core loops
    # ------------------------------------------------------------------
    def _listen_loop(self):
        """Main listening loop for incoming connections"""
        sock = None
        try:
            sock = self._get_or_create_socket(zmq.REP)
            sock.bind(f"tcp://*:{self.LISTEN_PORT}")
            logger.info(f"Listening on port {self.LISTEN_PORT}")
            
            while self.running:
                try:
                    raw = sock.recv(zmq.NOBLOCK)
                    if len(raw) < 20:
                        sock.send(b"INVALID")
                        continue
                        
                    if len(raw) > self.MAX_MESSAGE_SIZE:
                        sock.send(b"TOOLARGE")
                        continue
                    
                    peer_id, payload = raw[:20], raw[20:]
                    
                    # Rate limiting
                    if not self.rate_limiter.is_allowed(peer_id):
                        sock.send(b"RATELIMIT")
                        continue
                    
                    # Handle message
                    if peer_id not in self.session_keys:
                        self._handle_handshake(sock, peer_id, payload)
                    else:
                        try:
                            plain = self._decrypt(peer_id, payload)
                            data = json.loads(plain.decode())
                            self._handle_peer_data(data)
                            sock.send(b"ACK")
                        except Exception as e:
                            logger.warning(f"Message handling error: {e}")
                            sock.send(b"ERROR")
                            
                except zmq.Again:
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Listen loop error: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal listen loop error: {e}")
        finally:
            if sock:
                sock.close()

    def _sync_loop(self):
        """Periodic sync with known peers"""
        while self.running:
            try:
                # Build sync payload
                payload = {
                    "source": self.codename,
                    "ip": self.external_addr[0] if self.external_addr else self.real_ip,
                    "port": self.external_addr[1] if self.external_addr else self.LISTEN_PORT,
                    "timestamp": time.time(),
                    "node_id": self.node_id.hex()
                }
                
                # Add kernel data if available
                if self.kernel is not None and hasattr(self.kernel, 'mirrorcore') and self.kernel.mirrorcore is not None:
                    payload.update({
                        "beliefs": getattr(self.kernel.mirrorcore, 'beliefs', {}),
                        "emotions": getattr(self.kernel.mirrorcore, 'emotions', {}),
                        "missions": getattr(self.kernel.mirrorcore, 'mission_queue', [])
                    })
                if self.kernel is not None and hasattr(self.kernel, 'memory') and self.kernel.memory is not None:
                    payload["memory"] = self.kernel.memory.recall()
                if self.kernel is not None and hasattr(self.kernel, 'agents') and self.kernel.agents is not None:
                    payload["agent_status"] = {
                        k: str(v.status) for k, v in self.kernel.agents.items()
                    }
                
                # Send to all known peers
                payload_bytes = json.dumps(payload).encode()
                with self._peers_context():
                    peer_list = list(self.known_peers)
                
                for peer_id in peer_list:
                    try:
                        self._send_encrypted(peer_id, payload_bytes)
                    except Exception as e:
                        logger.warning(f"Sync failed to peer {peer_id.hex()[:8]}: {e}")
                
                logger.debug(f"Synced with {len(peer_list)} peers")
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
            
            time.sleep(10)

    def _discovery_loop(self):
        """Peer discovery using mDNS and DHT"""
        # Start mDNS listener if available
        # Only enable mDNS peer discovery if zeroconf and ServiceListener are imported from zeroconf
        if _zeroconf and ServiceBrowser and _ServiceListenerBase.__module__ == 'zeroconf._services':
            listener = _MDNSListener(self)
            # Ensure listener is an instance of the actual ServiceListener
            if isinstance(listener, _ServiceListenerBase):
                try:
                    ServiceBrowser(_zeroconf, "_peerlinker._tcp.local.", listener)
                    logger.info("mDNS discovery started")
                except Exception as e:
                    logger.warning(f"ServiceBrowser instantiation failed: {e}")
            else:
                logger.info("mDNS peer discovery is disabled (listener type mismatch)")
        else:
            logger.info("mDNS peer discovery is disabled (zeroconf or ServiceListener not available)")

        while self.running:
            try:
                # Random DHT lookup for peer discovery
                target = secrets.token_bytes(20)
                nodes = self.kbuckets.find_node(target)
                
                for node_id, addr in nodes:
                    try:
                        self._ping_node(node_id, addr)
                    except Exception as e:
                        logger.debug(f"Ping failed to {addr}: {e}")
                
                # Periodic k-bucket maintenance
                self.kbuckets.refresh()
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
            
            time.sleep(30)

    def _peer_cleanup_loop(self):
        """Clean up expired peers and sessions"""
        while self.running:
            try:
                cutoff = datetime.utcnow() - timedelta(seconds=180)
                expired_peers = []
                
                # Identify expired peers
                with self._session_context():
                    expired_peers = [
                        pid for pid, meta in self.session_keys.items()
                        if meta.get('last', datetime.utcnow()) < cutoff
                    ]
                
                # Remove expired peers
                if expired_peers:
                    with self._peers_context():
                        for pid in expired_peers:
                            self.known_peers.discard(pid)
                            
                    with self._session_context():
                        for pid in expired_peers:
                            self.session_keys.pop(pid, None)
                    
                    # Also remove from k-buckets
                    for pid in expired_peers:
                        self.kbuckets.remove(pid)
                            
                    logger.info(f"Cleaned up {len(expired_peers)} expired peers")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
            
            time.sleep(60)

    def _key_rotation_loop(self):
        """Periodic key rotation for forward secrecy"""
        while self.running:
            try:
                with self._session_context():
                    peers_to_rotate = []
                    now = datetime.utcnow()
                    
                    for peer_id, meta in self.session_keys.items():
                        if (meta.get('msg_cnt', 0) >= self.KEY_ROTATE_MSGS or
                            now - meta.get('last', now) >= timedelta(seconds=self.KEY_ROTATE_SECONDS)):
                            peers_to_rotate.append(peer_id)
                
                for peer_id in peers_to_rotate:
                    try:
                        self._initiate_key_rotation(peer_id)
                    except Exception as e:
                        logger.warning(f"Key rotation failed for peer {peer_id.hex()[:8]}: {e}")
                
                if peers_to_rotate:
                    logger.info(f"Rotated keys for {len(peers_to_rotate)} peers")
                
            except Exception as e:
                logger.error(f"Key rotation loop error: {e}")
            
            time.sleep(300)  # Check every 5 minutes

    # ------------------------------------------------------------------
    #  Handshake & crypto
    # ------------------------------------------------------------------
    def _handle_handshake(self, sock, peer_id, payload):
        """Handle incoming handshake requests"""
        try:
            data = json.loads(payload.decode())
            if data.get("type") != "handshake_1":
                sock.send(b"INVALID_HANDSHAKE")
                return
                
            their_ephemeral = bytes.fromhex(data["ephemeral_pk"])
            their_sig = bytes.fromhex(data["sig"])

            # Verify signature
            try:
                from nacl.signing import VerifyKey
                verify_key = VerifyKey(peer_id)
                verify_key.verify(their_ephemeral, their_sig)
            except Exception:
                sock.send(b"BADSIG")
                return

            # Derive shared secret
            rx, tx = b.crypto_kx_server_session_keys(
                self.ephemeral_sk, self.ephemeral_pk, their_ephemeral
            )
            
            # Store session keys thread-safely
            with self._session_context():
                self.session_keys[peer_id] = {
                    "tx": tx, 
                    "rx": rx, 
                    "msg_cnt": 0, 
                    "last": datetime.utcnow()
                }
            
            # Add to known peers
            with self._peers_context():
                self.known_peers.add(peer_id)
                
            # Add to k-buckets
            self.kbuckets.insert(peer_id, (data["ip"], data["port"]))

            # Send handshake response
            from nacl.signing import SigningKey
            signing_key = SigningKey(self.id_sk)
            sig = signing_key.sign(self.ephemeral_pk).signature
            reply = json.dumps({
                "type": "handshake_2",
                "ephemeral_pk": self.ephemeral_pk.hex(),
                "sig": sig.hex(),
                "ip": self.external_addr[0] if self.external_addr else self.real_ip,
                "port": self.external_addr[1] if self.external_addr else self.LISTEN_PORT
            }).encode()
            sock.send(reply)
            
            logger.info(f"Handshake completed with peer {peer_id.hex()[:8]}")
            
        except Exception as e:
            logger.error(f"Handshake error: {e}")
            try:
                sock.send(b"HANDSHAKE_ERROR")
            except:
                pass

    def _send_encrypted(self, peer_id, plaintext: bytes):
        """Send encrypted message to peer"""
        with self._session_context():
            if peer_id not in self.session_keys:
                raise ValueError("No session key for peer")
            meta = self.session_keys[peer_id]
            
            # Encrypt message
            nonce = nacl.utils.random(24)
            cipher = b.crypto_secretbox(plaintext, nonce, meta["tx"])
            raw = peer_id + nonce + cipher
            
            # Update counters
            meta["msg_cnt"] += 1
            meta["last"] = datetime.utcnow()
        
        # Send message (outside lock to avoid deadlock)
        addr = self.kbuckets.addr_of(peer_id)
        if not addr:
            raise ValueError("No address for peer")
            
        sock = None
        try:
            sock = self._get_or_create_socket(zmq.PUSH)
            sock.connect(f"tcp://{addr[0]}:{addr[1]}")
            sock.send(raw)
        finally:
            if sock:
                sock.close()

    def _decrypt(self, peer_id, raw):
        """Decrypt message from peer"""
        with self._session_context():
            if peer_id not in self.session_keys:
                raise ValueError("No session key for peer")
            meta = self.session_keys[peer_id]
            
            if len(raw) < 24:
                raise ValueError("Message too short")
                
            nonce, cipher = raw[:24], raw[24:]
            return b.crypto_secretbox_open(cipher, nonce, meta["rx"])

    def _initiate_key_rotation(self, peer_id):
        """Initiate key rotation with a peer"""
        # Generate new ephemeral keys
        new_sk, new_pk = b.crypto_kx_keypair()
        
        # Sign new public key
        from nacl.signing import SigningKey
        signing_key = SigningKey(self.id_sk)
        sig = signing_key.sign(new_pk).signature

        # Send key rotation request
        rotation_msg = json.dumps({
            "type": "key_rotation",
            "new_ephemeral_pk": new_pk.hex(),
            "sig": sig.hex()
        }).encode()
        
        self._send_encrypted(peer_id, rotation_msg)
        
        # Update our keys (peer will respond with their new key)
        self.ephemeral_sk, self.ephemeral_pk = new_sk, new_pk
        
        logger.debug(f"Initiated key rotation with peer {peer_id.hex()[:8]}")

    # ------------------------------------------------------------------
    #  Data handling
    # ------------------------------------------------------------------
    def _handle_peer_data(self, data):
        """Process data received from peers"""
        try:
            # Update peer last seen time
            peer_source = data.get("source", "unknown")
            peer_ip = data.get("ip")
            
            if peer_ip:
                # Update session timestamp if we have this peer
                peer_node_id = bytes.fromhex(data.get("node_id", ""))[:20]
                if peer_node_id in self.session_keys:
                    with self._session_context():
                        self.session_keys[peer_node_id]["last"] = datetime.utcnow()
            
            # Handle different types of peer data
            if self.kernel is not None and hasattr(self.kernel, 'mirrorcore') and self.kernel.mirrorcore is not None:
                if 'beliefs' in data:
                    self.kernel.mirrorcore.inject_beliefs(data['beliefs'])
                if 'emotions' in data:
                    self.kernel.mirrorcore.inject_emotions(data['emotions'])
                if 'missions' in data:
                    self.kernel.mirrorcore.import_peer_missions(data['missions'])
            
            if (
                'memory' in data and 
                hasattr(self.kernel, 'memory') and 
                self.kernel is not None and 
                getattr(self.kernel, 'memory', None) is not None
            ):
                self.kernel.memory.import_peer_memory(data['memory'], source=peer_source)
            
            # Handle key rotation requests
            if data.get("type") == "key_rotation":
                self._handle_key_rotation(data)
            
            logger.debug(f"Processed data from peer {peer_source}")
            
        except Exception as e:
            logger.error(f"Error handling peer data: {e}")

    def _handle_key_rotation(self, data):
        """Handle incoming key rotation request"""
        try:
            new_ephemeral = bytes.fromhex(data["new_ephemeral_pk"])
            sig = bytes.fromhex(data["sig"])
            
            # Verify signature (we'd need the peer's ID key for this)
            # For now, just log the rotation request
            logger.info("Received key rotation request")
            
        except Exception as e:
            logger.error(f"Key rotation handling error: {e}")

    def _ping_node(self, node_id, addr):
        """Ping a node to check if it's alive"""
        sock = None
        try:
            sock = self._get_or_create_socket(zmq.REQ)
            sock.connect(f"tcp://{addr[0]}:{addr[1]}")
            
            ping_msg = node_id + json.dumps({"type": "ping"}).encode()
            sock.send(ping_msg)
            
            response = sock.recv()
            if response == b"PONG":
                logger.debug(f"Ping successful to {addr}")
            
        except Exception as e:
            logger.debug(f"Ping failed to {addr}: {e}")
            self.kbuckets.remove(node_id)
        finally:
            if sock:
                sock.close()

    # ------------------------------------------------------------------
    #  Public API methods
    # ------------------------------------------------------------------
    def get_peer_count(self):
        """Get number of known peers"""
        with self._peers_context():
            return len(self.known_peers)
    
    def get_network_info(self):
        """Get current network status"""
        with self._peers_context():
            peer_count = len(self.known_peers)
        
        with self._session_context():
            active_sessions = len(self.session_keys)
        
        return {
            "node_id": self.node_id.hex(),
            "external_addr": self.external_addr,
            "peer_count": peer_count,
            "active_sessions": active_sessions,
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }


# ------------------------------------------------------------------
#  KBucketTable (Kademlia k-buckets)
# ------------------------------------------------------------------
class KBucketTable:
    """Kademlia-style k-bucket table for DHT"""
    
    def __init__(self, node_id, k=8):
        self.node_id = node_id
        self.k = k
        self.buckets = [[] for _ in range(160)]  # 160 buckets for 160-bit space
        self.lock = threading.Lock()

    def _bucket_index(self, node_id):
        """Calculate bucket index based on XOR distance"""
        d = int.from_bytes(self._xor(self.node_id, node_id), 'big')
        return max(0, d.bit_length() - 1) if d else 0

    def _xor(self, a, b):
        """XOR two byte arrays"""
        return bytes(x ^ y for x, y in zip(a, b))

    def insert(self, node_id, addr):
        """Insert node into appropriate k-bucket"""
        if node_id == self.node_id:
            return  # Don't store ourselves
            
        with self.lock:
            idx = self._bucket_index(node_id)
            bucket = self.buckets[idx]
            
            # If node already exists, move to end (most recent)
            for i, (nid, _) in enumerate(bucket):
                if nid == node_id:
                    bucket.pop(i)
                    break
            
            # Add to end of bucket
            bucket.append((node_id, addr))
            
            # If bucket is full, remove oldest
            if len(bucket) > self.k:
                bucket.pop(0)

    def remove(self, node_id):
        """Remove node from k-bucket"""
        with self.lock:
            idx = self._bucket_index(node_id)
            self.buckets[idx] = [
                (nid, addr) for nid, addr in self.buckets[idx] 
                if nid != node_id
            ]

    def find_node(self, target_id):
        """Find closest nodes to target"""
        with self.lock:
            idx = self._bucket_index(target_id)
            
            # Start with target bucket and expand outward
            candidates = []
            for i in range(len(self.buckets)):
                # Check buckets in order of distance from target bucket
                bucket_idx = (idx + i) % len(self.buckets)
                candidates.extend(self.buckets[bucket_idx])
                
                if len(candidates) >= self.k:
                    break
            
            # Sort by XOR distance to target
            candidates.sort(key=lambda x: int.from_bytes(
                self._xor(x[0], target_id), 'big'
            ))
            
            return candidates[:self.k]

    def addr_of(self, node_id):
        """Get address of specific node"""
        with self.lock:
            idx = self._bucket_index(node_id)
            for nid, addr in self.buckets[idx]:
                if nid == node_id:
                    return addr
        return None

    def refresh(self):
        """Refresh k-buckets by removing stale entries"""
        with self.lock:
            # This is a simplified refresh - in a full implementation,
            # we'd ping nodes to check if they're still alive
            pass

    def get_all_nodes(self):
        """Get all stored nodes"""
        with self.lock:
            all_nodes = []
            for bucket in self.buckets:
                all_nodes.extend(bucket)
            return all_nodes


# ------------------------------------------------------------------
#  mDNS listener
# ------------------------------------------------------------------
class _MDNSListener(_ServiceListenerBase):
    """Zeroconf/mDNS service listener for local peer discovery"""
    
    def __init__(self, agent):
        self.agent = agent

    def add_service(self, zc, type_, name):
        """Handle discovered mDNS service"""
        try:
            info = zc.get_service_info(type_, name)
            if info and info.addresses:
                addr = (socket.inet_ntoa(info.addresses[0]), info.port)
                # Extract node ID from service properties
                node_id_bytes = info.properties.get(b"id", b"")
                node_id_hex = node_id_bytes.decode() if node_id_bytes else ""
                if node_id_hex:
                    node_id = bytes.fromhex(node_id_hex)
                    self.agent.kbuckets.insert(node_id, addr)
                    logger.info(f"Discovered peer via mDNS: {addr}")
        except Exception as e:
            logger.warning(f"mDNS service processing error: {e}")

    def remove_service(self, zc, type_, name):
        """Handle removed mDNS service"""
        pass

    def update_service(self, zc, type_, name):
        """Handle updated mDNS service"""
        # Treat updates as new services
        self.add_service(zc, type_, name)


# ------------------------------------------------------------------
#  Network Statistics and Monitoring
# ------------------------------------------------------------------
class NetworkStats:
    """Network statistics collection and monitoring"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'handshakes_completed': 0,
            'handshakes_failed': 0,
            'key_rotations': 0,
            'peers_discovered': 0,
            'connection_errors': 0
        }
    
    def increment(self, stat_name, amount=1):
        """Thread-safe increment of statistic"""
        with self.lock:
            self.stats[stat_name] = self.stats.get(stat_name, 0) + amount
    
    def get_stats(self):
        """Get current statistics snapshot"""
        with self.lock:
            uptime = time.time() - self.start_time
            stats_copy = self.stats.copy()
            stats_copy['uptime_seconds'] = int(uptime)
            stats_copy['messages_per_second'] = int(
                (stats_copy['messages_sent'] + stats_copy['messages_received']) / max(uptime, 1)
            )
            return stats_copy


# ------------------------------------------------------------------
#  Enhanced PeerLinkerAgent with monitoring and reliability
# ------------------------------------------------------------------
class EnhancedPeerLinkerAgent(PeerLinkerAgent):
    """Enhanced version with monitoring, reliability, and better error handling"""
    
    def __init__(self):
        super().__init__()
        self.stats = NetworkStats()
        self.connection_retry_delays = {}  # peer_id -> delay
        self.max_retry_delay = 300  # 5 minutes max retry delay
        self.message_queue = {}  # peer_id -> [messages] for offline peers
        self.max_queue_size = 100
        
        # Health check parameters
        self.last_successful_sync = time.time()
        self.health_check_interval = 30
        self.unhealthy_threshold = 300  # 5 minutes without successful sync
        
        # Start health monitoring
        threading.Thread(target=self._health_monitor_loop, daemon=True).start()
    
    def _health_monitor_loop(self):
        """Monitor network health and take corrective actions"""
        while self.running:
            try:
                current_time = time.time()
                time_since_sync = current_time - self.last_successful_sync
                
                # Check if we're healthy
                is_healthy = (
                    time_since_sync < self.unhealthy_threshold and
                    self.get_peer_count() > 0
                )
                
                if not is_healthy:
                    logger.warning(f"Network unhealthy: {time_since_sync:.1f}s since last sync, "
                                 f"{self.get_peer_count()} peers")
                    self._attempt_network_recovery()
                
                # Log periodic health status
                if current_time % 300 < 30:  # Every 5 minutes
                    stats = self.stats.get_stats()
                    logger.info(f"Network health: {self.get_peer_count()} peers, "
                              f"{stats['messages_per_second']:.1f} msg/s")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            
            time.sleep(self.health_check_interval)
    
    def _attempt_network_recovery(self):
        """Attempt to recover network connectivity"""
        try:
            # Re-discover external address
            self._discover_external()
            
            # Try to re-establish UPnP mapping
            self._map_port()
            
            # Clear failed connections
            self.connection_retry_delays.clear()
            
            # Try to reconnect to known peers
            with self._peers_context():
                peer_list = list(self.known_peers)
            
            for peer_id in peer_list[:5]:  # Try up to 5 peers
                try:
                    addr = self.kbuckets.addr_of(peer_id)
                    if addr:
                        self._ping_node(peer_id, addr)
                except Exception:
                    continue
            
            logger.info("Network recovery attempt completed")
            
        except Exception as e:
            logger.error(f"Network recovery failed: {e}")
    
    def _send_encrypted_with_retry(self, peer_id, plaintext: bytes):
        """Send encrypted message with retry logic"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                self._send_encrypted(peer_id, plaintext)
                self.stats.increment('messages_sent')
                self.stats.increment('bytes_sent', len(plaintext))
                
                # Reset retry delay on success
                self.connection_retry_delays.pop(peer_id, None)
                self.last_successful_sync = time.time()
                
                # Send queued messages if any
                self._send_queued_messages(peer_id)
                
                return
                
            except Exception as e:
                self.stats.increment('connection_errors')
                
                if attempt == max_retries - 1:
                    # Final attempt failed, queue message
                    self._queue_message(peer_id, plaintext)
                    logger.warning(f"Message queued for peer {peer_id.hex()[:8]} after {max_retries} attempts")
                else:
                    # Wait before retry
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    logger.debug(f"Retrying send to {peer_id.hex()[:8]}, attempt {attempt + 2}/{max_retries}")
    
    def _queue_message(self, peer_id, message: bytes):
        """Queue message for later delivery"""
        if peer_id not in self.message_queue:
            self.message_queue[peer_id] = []
        
        queue = self.message_queue[peer_id]
        queue.append(message)
        
        # Limit queue size
        if len(queue) > self.max_queue_size:
            queue.pop(0)  # Remove oldest message
    
    def _send_queued_messages(self, peer_id):
        """Send any queued messages for a peer"""
        if peer_id not in self.message_queue:
            return
        
        queue = self.message_queue[peer_id]
        sent_count = 0
        
        while queue and sent_count < 10:  # Limit burst sending
            message = None
            try:
                message = queue.pop(0)
                self._send_encrypted(peer_id, message)
                sent_count += 1
            except Exception as e:
                # Put message back and stop
                if message is not None:
                    queue.insert(0, message)
                break
        
        if sent_count > 0:
            logger.info(f"Sent {sent_count} queued messages to {peer_id.hex()[:8]}")
        
        # Clean up empty queues
        if not queue:
            del self.message_queue[peer_id]
    
    def _handle_handshake(self, sock, peer_id, payload):
        """Enhanced handshake handling with statistics"""
        try:
            super()._handle_handshake(sock, peer_id, payload)
            self.stats.increment('handshakes_completed')
            logger.info(f"Handshake completed with peer {peer_id.hex()[:8]}")
        except Exception as e:
            self.stats.increment('handshakes_failed')
            logger.error(f"Handshake failed with peer {peer_id.hex()[:8]}: {e}")
            raise
    
    def _handle_peer_data(self, data):
        """Enhanced peer data handling with statistics"""
        try:
            super()._handle_peer_data(data)
            self.stats.increment('messages_received')
            
            # Update last successful sync time
            self.last_successful_sync = time.time()
            
        except Exception as e:
            logger.error(f"Error handling peer data: {e}")
            raise
    
    def _initiate_key_rotation(self, peer_id):
        """Enhanced key rotation with statistics"""
        try:
            super()._initiate_key_rotation(peer_id)
            self.stats.increment('key_rotations')
        except Exception as e:
            logger.error(f"Key rotation failed for {peer_id.hex()[:8]}: {e}")
            raise
    
    def get_detailed_status(self):
        """Get detailed agent status including statistics"""
        base_info = self.get_network_info()
        stats = self.stats.get_stats()
        
        # Calculate health score
        time_since_sync = time.time() - self.last_successful_sync
        health_score = max(0, 100 - (time_since_sync / self.unhealthy_threshold * 100))
        
        return {
            **base_info,
            'statistics': stats,
            'health_score': health_score,
            'last_successful_sync': self.last_successful_sync,
            'queued_messages': sum(len(q) for q in self.message_queue.values()),
            'retry_delays': len(self.connection_retry_delays)
        }
    
    def export_peer_list(self):
        """Export current peer list for backup/restore"""
        all_nodes = self.kbuckets.get_all_nodes()
        return {
            'timestamp': time.time(),
            'node_id': self.node_id.hex(),
            'external_addr': self.external_addr,
            'peers': [
                {
                    'node_id': node_id.hex(),
                    'address': addr,
                    'bucket_index': self.kbuckets._bucket_index(node_id)
                }
                for node_id, addr in all_nodes
            ]
        }
    
    def import_peer_list(self, peer_data):
        """Import peer list from backup"""
        try:
            if not isinstance(peer_data, dict) or 'peers' not in peer_data:
                raise ValueError("Invalid peer data format")
            
            imported = 0
            for peer_info in peer_data['peers']:
                try:
                    node_id = bytes.fromhex(peer_info['node_id'])
                    addr = tuple(peer_info['address'])
                    self.kbuckets.insert(node_id, addr)
                    imported += 1
                except Exception as e:
                    logger.warning(f"Failed to import peer {peer_info}: {e}")
            
            logger.info(f"Imported {imported} peers from backup")
            return imported
            
        except Exception as e:
            logger.error(f"Peer import failed: {e}")
            return 0


# ------------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------------
def create_peer_linker_agent(enhanced=True):
    """Factory function to create PeerLinkerAgent"""
    if enhanced:
        return EnhancedPeerLinkerAgent()
    else:
        return PeerLinkerAgent()


def validate_node_id(node_id_hex):
    """Validate node ID format"""
    try:
        node_id = bytes.fromhex(node_id_hex)
        return len(node_id) == 20
    except ValueError:
        return False


def calculate_distance(node_id_a, node_id_b):
    """Calculate XOR distance between two node IDs"""
    if len(node_id_a) != len(node_id_b):
        raise ValueError("Node IDs must be same length")
    
    distance = 0
    for a, b in zip(node_id_a, node_id_b):
        distance = (distance << 8) | (a ^ b)
    
    return distance


# ------------------------------------------------------------------
#  Configuration and constants
# ------------------------------------------------------------------
class PeerLinkerConfig:
    """Configuration class for PeerLinker settings"""
    
    # Network settings
    LISTEN_PORT = 5555
    MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
    SOCKET_TIMEOUT = 5000  # 5 seconds
    
    # Discovery settings
    STUN_HOST = "stun.l.google.com"
    STUN_PORT = 19302
    UPNP_LEASE_SECONDS = 3600
    
    # DHT settings
    DHT_K = 8
    DHT_ALPHA = 3  # Parallelism factor
    DHT_REFRESH_INTERVAL = 3600  # 1 hour
    
    # Security settings
    KEY_ROTATE_MESSAGES = 100
    KEY_ROTATE_SECONDS = 600  # 10 minutes
    SESSION_TIMEOUT = 180  # 3 minutes
    
    # Performance settings
    MAX_PEERS = 1000
    MAX_QUEUE_SIZE = 100
    SYNC_INTERVAL = 10  # seconds
    DISCOVERY_INTERVAL = 30  # seconds
    
    # Monitoring settings
    HEALTH_CHECK_INTERVAL = 30  # seconds
    UNHEALTHY_THRESHOLD = 300  # 5 minutes
    STATS_LOG_INTERVAL = 300  # 5 minutes


# ------------------------------------------------------------------
#  Example usage and testing
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description='PeerLinker P2P Agent')
    parser.add_argument('--port', type=int, default=5555, help='Listen port')
    parser.add_argument('--bootstrap', help='Bootstrap peer address (host:port)')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced agent')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    # Create agent
    agent = create_peer_linker_agent(enhanced=args.enhanced)
    
    if args.port != 5555:
        agent.LISTEN_PORT = args.port

    # Graceful shutdown on SIGINT / Ctrl-C
    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        agent.shutdown()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.shutdown()
