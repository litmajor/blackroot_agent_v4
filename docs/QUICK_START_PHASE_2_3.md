# QUICK START: PHASE 2.3 SCOUT ML IMPLEMENTATION

**Status**: Ready to start immediately  
**Estimated Duration**: 4 days (Days 6-10 of Phase 2)  
**LOC to Add**: ~1000 (2 new files + Scout updates)

---

## What You Need to Do

### Step 1: Install Dependencies (1 hour)
```bash
cd e:\repos\litmajor\blackroot_agent_v4
pip install scikit-learn numpy pandas nvdlib requests
```

### Step 2: Create ThreatIntelligenceDB (Days 6-7)

**File to Create**: `recon/threat_intelligence.py`  
**Size**: 450 lines  
**Code**: See full implementation in [PHASE_2_3_SCOUT_ML.md](PHASE_2_3_SCOUT_ML.md) Part 1

**Quick Copy**:
- Class: `ThreatIntelligenceDB`
- Init: Loads from Redis, refreshes from sources
- Methods:
  - `load_from_redis()` - Cache management
  - `refresh_from_sources()` - Update threat data
  - `is_honeypot(ip, profile)` → (bool, confidence)
  - `is_malicious(ip, profile)` → (bool, confidence)

### Step 3: Create MLAnomalyDetector (Days 7-8)

**File to Create**: `recon/ml_anomaly_detector.py`  
**Size**: 350 lines  
**Code**: See full implementation in [PHASE_2_3_SCOUT_ML.md](PHASE_2_3_SCOUT_ML.md) Part 2

**Quick Copy**:
- Class: `MLAnomalyDetector`
- Uses: Isolation Forest (scikit-learn)
- Methods:
  - `extract_features(node_profile)` → 7-dim feature vector
  - `score_as_honeypot(profile)` → (float, explanations)
  - `train_model(training_data)` → Save to disk

### Step 4: Update Scout Agent (Days 8-9)

**File to Update**: [agents/scout.py](agents/scout.py)  
**Lines to Replace**: 582-592 (`_load_threat_intelligence`)  
**Code**: See [PHASE_2_3_SCOUT_ML.md](PHASE_2_3_SCOUT_ML.md) Part 3

**What Changes**:
```python
# OLD (stub):
def _load_threat_intelligence(self) -> Dict[str, Any]:
    return {'known_honeypots': [], 'malicious_ips': [], ...}

# NEW (real):
def _load_threat_intelligence(self) -> Dict[str, Any]:
    threat_db = ThreatIntelligenceDB(self.vault, self.config)
    threat_db.refresh_from_sources()
    ml_detector = MLAnomalyDetector(self.config.get("scout", {}))
    return {
        'threat_db': threat_db,
        'ml_detector': ml_detector,
        ...
    }
```

**New Method to Add**:
- `_detect_honeypots(targets)` → (safe_targets, honeypots)
  - Combines threat DB + ML scoring
  - Returns filtered targets with explanations

### Step 5: Test & Validate (Days 9-10)

**Run Tests**:
```bash
python -m pytest tests/test_scout_ml.py -v
```

**Validation Checklist**:
- [ ] ThreatIntelligenceDB loads from Redis
- [ ] MLAnomalyDetector creates feature vectors
- [ ] Scout._load_threat_intelligence() returns real data (not stub)
- [ ] _detect_honeypots() filters targets correctly
- [ ] Performance <100ms per target

---

## Configuration Update

**File**: [core/agent_config.json](core/agent_config.json)  
**Add Section**:
```json
"scout": {
  "enabled": true,
  "threat_db_refresh_interval": 86400,
  "ml_model_path": "models/honeypot_detector.pkl",
  "honeypot_confidence_threshold": 0.7,
  "ml_features": [
    "port_count", "service_count", "response_time_variance",
    "os_fingerprint_confidence", "uptime_days",
    "distinctive_service_combo", "response_consistency_score"
  ]
}
```

---

## Testing Strategy

### Unit Tests
```python
# Test 1: Threat Intelligence
threat_db.is_honeypot("192.168.1.100", honeypot_profile)
# Expected: (True, confidence > 0.7)

# Test 2: ML Detection
detector.score_as_honeypot(suspicious_profile)
# Expected: (float 0-1, list of explanations)

# Test 3: Scout Integration
scout._detect_honeypots(targets)
# Expected: (safe_targets, honeypots)
```

### Integration Test
```python
# Full pipeline: Shogun → Targeting → Scout
from agents.shogun_recon import ShogunReconAgent
from agents.targeting_engine import TargetingEngine
from agents.scout import Scout

shogun = ShogunReconAgent()
targets = shogun.discover_network()  # Real nmap

targeting = TargetingEngine()
candidates = targeting.validate(targets)  # Real trust scores

scout = Scout()
safe, honeypots = scout.filter(candidates)  # Real honeyopt detection

assert len(safe) > 0, "Should have safe targets"
assert len(honeypots) > 0, "Should detect honeypots"
```

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| [recon/network_mapper.py](recon/network_mapper.py) | Phase 2.1 | ✅ DONE |
| [agents/shogun_recon.py](agents/shogun_recon.py) | Phase 2.1 | ✅ DONE |
| [agents/targeting_engine.py](agents/targeting_engine.py) | Phase 2.2 | ✅ DONE |
| **recon/threat_intelligence.py** | Phase 2.3 | 📋 TO CREATE |
| **recon/ml_anomaly_detector.py** | Phase 2.3 | 📋 TO CREATE |
| **agents/scout.py** | Phase 2.3 | 🔄 TO UPDATE |
| [PHASE_2_3_SCOUT_ML.md](PHASE_2_3_SCOUT_ML.md) | Phase 2.3 Spec | ✅ READY |
| [PHASE_2_PROGRESS.md](PHASE_2_PROGRESS.md) | Phase 2 Status | ✅ READY |

---

## Architecture Reminder

```
ShogunReconAgent (DONE)
  ↓ Real nmap scans → Returns: {nodes with scores}
  
TargetingEngine (DONE)
  ↓ Query real Shogun → Returns: {targets with trust_score}
  
Scout (NEXT - Phase 2.3)
  ├─ ThreatIntelligenceDB
  │  ├─ Known honeypots list
  │  ├─ Malicious IPs
  │  └─ Service CVEs
  │
  ├─ MLAnomalyDetector
  │  ├─ Feature extraction (7 dimensions)
  │  └─ Isolation Forest scoring
  │
  └─ _detect_honeypots()
     ↓ Returns: (safe_targets, honeypots) with confidence scores
     
MissionAgent (Phase 2.4 - after Scout)
  ↓ Receives only safe targets → Executes real operations
```

---

## Success Criteria (When Done)

✅ **Phase 2.3 Complete When**:
1. `recon/threat_intelligence.py` created and tested
2. `recon/ml_anomaly_detector.py` created and tested
3. Scout._load_threat_intelligence() returns real data (not stub)
4. Scout._detect_honeypots() filters targets with >90% accuracy
5. All honeypots from test set identified, <5% false positives
6. Latency <100ms per target
7. Tests pass: `pytest tests/test_scout_ml.py -v`

---

## Common Issues & Solutions

### Issue: scikit-learn not installed
**Solution**: `pip install scikit-learn numpy`

### Issue: ML model not loading
**Solution**: Model will auto-create default Isolation Forest if file missing

### Issue: Threat data source unreachable
**Solution**: Gracefully fallback to Redis cache, manual lists

### Issue: Performance slow (>100ms/target)
**Solution**: Cache features, batch process targets, tune ML hyperparameters

---

## Next Phase (After 2.3)

Once Scout is complete, Phase 2.4 (MissionAgent) will:
- Receive only safe targets (no honeypots)
- Execute deep vulnerability scans
- Monitor for changes
- Apply patches

MissionAgent will use the same pattern:
1. Real nmap -sV -A scans (not simulated)
2. Actual file monitoring (not placeholder)
3. Safe patching with rollback (not fake)

---

## One-Command Verification

After implementation, verify everything works:

```bash
# Run validation suite
python PHASE_2_VALIDATION.py

# Should show:
# ✅ Passed: 6/6 (up from 4/6)
# → Phase 2.1 (ShogunReconAgent): READY FOR DEPLOYMENT ✓
# → Phase 2.2 (TargetingEngine): READY FOR DEPLOYMENT ✓
# → Phase 2.3 (Scout): READY FOR DEPLOYMENT ✓
# → Phase 2.4 onwards: Ready to implement
```

---

## Timeline Summary

| Days | Task | LOC | Status |
|------|------|-----|--------|
| 1-2 | Shogun real discovery | 550 | ✅ DONE |
| 3-4 | Targeting trust eval | 100 | ✅ DONE |
| 6-7 | Threat intelligence DB | 450 | 📋 NEXT |
| 7-8 | ML anomaly detector | 350 | → THEN |
| 8-9 | Scout integration | 150 | → THEN |
| 9-10 | Testing & validation | 100 | → THEN |
| 11-14 | MissionAgent execution | 400 | ⏳ NEXT PHASE |
| 15-16 | EnforcerAgent rules | 300 | ⏳ NEXT PHASE |

**Total Phase 2**:  
- LOC: ~2150
- Timeline: 3 weeks
- Goal: 95% production-ready

---

**Ready to start?** → Begin with Step 1: Install ML dependencies  
**Need details?** → See [PHASE_2_3_SCOUT_ML.md](PHASE_2_3_SCOUT_ML.md)  
**Previous work?** → See [PHASE_2_PROGRESS.md](PHASE_2_PROGRESS.md)

