# PHASE 2.3: SCOUT ML THREAT DETECTION

**Status**: 🔄 IN PROGRESS - READY TO IMPLEMENT  
**Owner**: Scout Agent (ml_classifier subsystem)  
**Scope**: ~600 lines of new code + ML training pipeline  
**Timeline**: Days 6-10 of Phase 2  

---

## Problem Statement

Scout's threat intelligence is currently stubbed:

```python
def _load_threat_intelligence(self) -> Dict[str, Any]:
    """Load threat intelligence database (stub implementation)."""
    return {
        'known_honeypots': [],
        'malicious_ips': [],
        'threat_signatures': [],
        'recent_incidents': []
    }
```

**Impact**: Scout receives zombie candidates from TargetingEngine but cannot:
- Detect honeypots/traps
- Filter malicious IPs
- Identify suspicious Services
- Block known compromised systems

**Goal**: Replace stub with real threat detection using:
1. YARA rules (malware signatures)
2. CVE data (known vulnerabilities)
3. DNS/IP blacklists
4. ML anomaly detection (Isolation Forest)

---

## Solution Architecture

### Data Flow

```
TargetingEngine (approved targets)
    ↓
    └→ Scout._evaluate_targets()
       ├─ _load_threat_intelligence()      ← REPLACE THIS
       ├─ _calculate_trap_likelihood()     ← Use real data
       ├─ _detect_honeypots()              ← Use ML
       └─ Returns: {safe_targets: [...], honeypots: [...], risky: [...]}
    ↓
MissionAgent (exec on safe targets only)
```

---

## Implementation: 4-Part Strategy

### Part 1: Threat Intelligence Database (Days 6-7)

**File**: `recon/threat_intelligence.py` (NEW - 450 lines)

```python
from typing import Dict, Set, List, Tuple
import json
import requests
from datetime import datetime, timedelta
import logging

class ThreatIntelligenceDB:
    """Central threat intelligence repository."""
    
    def __init__(self, vault, config):
        self.vault = vault
        self.config = config
        self.redis = vault.redis
        self.known_honeypots: Set[str] = set()
        self.malicious_ips: Set[str] = set()
        self.threat_signatures: Dict = {}
        self.last_update = None
        self.load_from_redis()
    
    def load_from_redis(self):
        """Load cached threat data from Redis."""
        # Honeypots
        honeypots_json = self.redis.get("threat_honeypots")
        if honeypots_json:
            self.known_honeypots = set(json.loads(honeypots_json))
        
        # Malicious IPs
        malicious_json = self.redis.get("threat_malicious_ips")
        if malicious_json:
            self.malicious_ips = set(json.loads(malicious_json))
        
        # Signatures
        sig_json = self.redis.get("threat_signatures")
        if sig_json:
            self.threat_signatures = json.loads(sig_json)
    
    def refresh_from_sources(self) -> bool:
        """
        Refresh threat data from external sources.
        Called periodically (every 24 hours in production).
        """
        try:
            # 1. Load honeypot signatures
            self._load_honeypot_patterns()
            
            # 2. Load CVE data
            self._load_cve_database()
            
            # 3. Load DNS/IP blacklists
            self._load_dns_blacklist()
            
            # 4. Cache in Redis with 24h TTL
            self.redis.setex(
                "threat_honeypots",
                86400,
                json.dumps(list(self.known_honeypots))
            )
            self.redis.setex(
                "threat_malicious_ips", 
                86400,
                json.dumps(list(self.malicious_ips))
            )
            self.redis.setex(
                "threat_signatures",
                86400,
                json.dumps(self.threat_signatures)
            )
            
            self.last_update = datetime.now()
            return True
        except Exception as e:
            logging.error(f"Failed to refresh threat intelligence: {e}")
            return False
    
    def _load_honeypot_patterns(self):
        """
        Load known honeypot signatures.
        Sources: Glastopf, MHN project, T-Pot patterns.
        """
        honeypot_indicators = {
            # Port-based indicators
            "ports": {
                "3389": "win-rm-honeypot",      # High RDP = honeypot
                "23": "telnet-trap",             # Enabled Telnet = trap
                "69": "tftp-trap",               # TFTP = rarely seen legitimately
            },
            # Service banners
            "banners": {
                "Honeypot": 1.0,
                "Trap": 1.0,
                "Glastopf": 1.0,
                "Cowrie": 0.95,                  # SSH honeypot
                "Dionaea": 0.95,                 # Malware honeypot
            },
            # OS patterns
            "os_patterns": {
                "honeyd-generated": 1.0,
                "emulated environment": 0.9,
                "docker-honeypot": 0.95,
                "virtual machine honeypot": 0.9,
            }
        }
        
        # Add to database
        self.threat_signatures['honeypot_patterns'] = honeypot_indicators
        
        # Known honeypot IPs (populate from MHN or public lists)
        # In production: Fetch from https://mhn.example.com/api/honeypots
        self.known_honeypots.add("honeypot.thinkst.com")  # Canary tokens
        self.known_honeypots.add("3.145.115.0/24")        # Example range
    
    def _load_cve_database(self):
        """
        Load CVE data for services.
        Uses NVD API (National Vulnerability Database).
        """
        try:
            # In production: pip install nvdlib
            # import nvdlib
            # results = nvdlib.searchCVE(keyword="ssh")
            
            # For now, static high-severity CVEs
            self.threat_signatures['cves'] = {
                "OpenSSH": {
                    "CVE-2023-38709": {"severity": "HIGH", "score": 9.8},
                    "CVE-2023-25136": {"severity": "HIGH", "score": 7.5},
                },
                "Apache": {
                    "CVE-2023-44487": {"severity": "HIGH", "score": 7.5},
                },
                "Windows": {
                    "CVE-2023-21674": {"severity": "CRITICAL", "score": 9.3},
                }
            }
        except Exception as e:
            logging.warning(f"CVE load failed (expected on offline): {e}")
    
    def _load_dns_blacklist(self):
        """
        Load DNS blacklist data for IP reputation.
        Sources: Spamhaus, Project Honey Pot, AbuseIPDB.
        """
        # In production: fetch from ABuseIPDB API
        # Example format:
        dns_bl_data = {
            "malware_ips": {
                "203.0.113.1": {"reports": 45, "confidence": 95},
                "198.51.100.5": {"reports": 89, "confidence": 99},
            },
            "botnet_ips": {
                "192.0.2.100": {"botnet": "Emotet", "confidence": 85},
            },
            "spam_sources": {
                "198.51.100.0/24": {"confidence": 80, "reports": 200},
            }
        }
        
        self.threat_signatures['dns_blacklist'] = dns_bl_data
        self.malicious_ips.update(dns_bl_data.get('malware_ips', {}).keys())
    
    def is_honeypot(self, ip: str, node_profile: Dict) -> Tuple[bool, float]:
        """
        Determine if IP is likely honeypot.
        Returns: (is_honeypot, confidence 0-1)
        """
        confidence = 0.0
        
        # Check known honeypot list
        if ip in self.known_honeypots:
            return True, 0.99
        
        # Check banner indicators
        banner = node_profile.get("banner", "").lower()
        for honeypot_indicator, weight in self.threat_signatures.get(
            'honeypot_patterns', {}).get('banners', {}).items():
            if honeypot_indicator.lower() in banner:
                confidence = max(confidence, weight)
        
        # Check OS patterns
        os = node_profile.get("os", "").lower()
        for pattern, weight in self.threat_signatures.get(
            'honeypot_patterns', {}).get('os_patterns', {}).items():
            if pattern.lower() in os:
                confidence = max(confidence, weight)
        
        # Check port patterns
        ports = node_profile.get("ports", {})
        honeypot_port_count = sum(
            1 for p in ports if str(p) in
            self.threat_signatures.get('honeypot_patterns', {}).get('ports', {})
        )
        if honeypot_port_count >= 3:
            confidence = max(confidence, 0.85)
        
        return confidence >= 0.6, confidence
    
    def is_malicious(self, ip: str, node_profile: Dict) -> Tuple[bool, float]:
        """
        Determine if IP is known malicious.
        Returns: (is_malicious, confidence 0-1)
        """
        # Check known malicious list
        if ip in self.malicious_ips:
            return True, 0.95
        
        # Check CVE severity
        services = node_profile.get("services", {})
        for service, version in services.items():
            cves = self.threat_signatures.get('cves', {}).get(service, {})
            if cves:
                # Has known CVEs with no patches = high risk
                return True, 0.75
        
        return False, 0.0
```

**Key Methods**:
- `load_from_redis()` - Cache management
- `refresh_from_sources()` - Update threat data (24h cycle)
- `is_honeypot(ip, profile)` - Honeypot detection → (bool, confidence)
- `is_malicious(ip, profile)` - Malicious IP detection → (bool, confidence)

---

### Part 2: ML Anomaly Detection (Days 7-8)

**File**: `recon/ml_anomaly_detector.py` (NEW - 350 lines)

```python
from typing import Dict, List, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta

class MLAnomalyDetector:
    """Machine learning-based anomaly detection for honeypot identification."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model_path = self.config.get("model_path", "models/honeypot_detector.pkl")
        self.model = None
        self.scaler = None
        self.feature_names = [
            "port_count",
            "service_count", 
            "response_time_variance",
            "os_fingerprint_confidence",
            "uptime_days",
            "distinctive_service_combo",
            "response_consistency_score"
        ]
        self.load_model()
    
    def load_model(self):
        """Load pre-trained Isolation Forest model."""
        import os
        if os.path.exists(self.model_path):
            import pickle
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logging.info(f"Loaded ML model from {self.model_path}")
            except Exception as e:
                logging.warning(f"Failed to load model: {e}, using default")
                self._create_default_model()
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default model if none exists."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Create model with typical honeypot detection parameters
            self.model = IsolationForest(
                contamination=0.1,  # Assume ~10% are honeypots
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
            logging.info("Created default Isolation Forest model")
        except ImportError:
            logging.warning("scikit-learn not available, ML detection disabled")
            self.model = None
    
    def extract_features(self, node_profile: Dict) -> np.ndarray:
        """
        Extract feature vector from node profile.
        Returns: 7-dimensional feature vector
        """
        features = []
        
        # Feature 1: Port count (honeypots often have few ports)
        ports = node_profile.get("ports", [])
        features.append(len(ports))
        
        # Feature 2: Service count
        services = node_profile.get("services", {})
        features.append(len(services))
        
        # Feature 3: Response time variance (honeypots = high variance or zero variance)
        response_times = node_profile.get("response_times", [1.0])
        if len(response_times) > 1:
            variance = np.var(response_times)
        else:
            variance = 0.0
        features.append(variance)
        
        # Feature 4: OS fingerprint confidence (low = suspicious)
        os_confidence = node_profile.get("os_confidence", 0.5)
        features.append(os_confidence)
        
        # Feature 5: Uptime in days (new systems = suspicious)
        uptime_seconds = node_profile.get("uptime_seconds", 86400)
        uptime_days = uptime_seconds / 86400
        features.append(uptime_days)
        
        # Feature 6: Distinctive service combo score
        # (honeypots often have odd service combinations)
        distinctive = self._score_service_combination(services)
        features.append(distinctive)
        
        # Feature 7: Response consistency (honeypots = too consistent or too varied)
        consistency = self._score_response_consistency(response_times)
        features.append(consistency)
        
        return np.array(features, dtype=float).reshape(1, -1)
    
    def _score_service_combination(self, services: Dict) -> float:
        """Score how unusual the service combination is."""
        if not services:
            return 1.0  # No services = suspicious
        
        # Check for unusual combinations
        service_names = set(services.keys())
        
        # Known suspicious combos (honeypots often have these)
        suspicious_combos = [
            {"ssh", "telnet"},  # Both shell access
            {"http", "https", "ftp", "smtp"},  # Everything exposed
            {"honeypot", "trap", "emulation"}  # Explicit honeypot tools
        ]
        
        for combo in suspicious_combos:
            if combo.issubset(service_names):
                return 0.8  # High suspicion
        
        # Score based on randomness
        return 0.3 if len(services) >= 3 else 0.1
    
    def _score_response_consistency(self, response_times: List[float]) -> float:
        """Score how consistent responses are (honeypots = extreme values)."""
        if not response_times or len(response_times) < 2:
            return 0.5
        
        response_times = np.array(response_times)
        
        # Perfect consistency (all same) = suspicious
        if np.std(response_times) < 0.001:
            return 0.9
        
        # Huge variance = suspicious
        cv = np.std(response_times) / (np.mean(response_times) + 0.0001)
        if cv > 5:
            return 0.8
        
        # Normal variation = 0-1 range
        return min(cv / 5, 1.0)
    
    def score_as_honeypot(self, node_profile: Dict) -> Tuple[float, List[str]]:
        """
        Score node as honeypot using ML model.
        Returns: (anomaly_score 0-1, explanation_list)
        
        Anomaly score: 0.0 = normal, 1.0 = definitely honeypot
        """
        explanations = []
        
        if self.model is None:
            return 0.0, ["ML model not available"]
        
        try:
            features = self.extract_features(node_profile)
            
            # Isolation Forest returns -1 (anomaly) or 1 (normal)
            prediction = self.model.predict(features)[0]
            
            # Get anomaly score (0-1, higher = more anomalous)
            if hasattr(self.model, 'score_samples'):
                raw_score = self.model.score_samples(features)[0]
                # Normalize to 0-1
                anomaly_score = 1.0 / (1.0 + np.exp(-raw_score))
            else:
                # Fallback: use prediction
                anomaly_score = 0.8 if prediction == -1 else 0.2
            
            # Generate explanations
            port_count = len(node_profile.get("ports", []))
            if port_count <= 2:
                explanations.append(f"Very few ports ({port_count})")
            
            service_count = len(node_profile.get("services", {}))
            if service_count > 10:
                explanations.append(f"Too many services ({service_count})")
            
            os_conf = node_profile.get("os_confidence", 0.5)
            if os_conf < 0.3:
                explanations.append(f"Low OS confidence ({os_conf:.1%})")
            
            uptime_days = node_profile.get("uptime_seconds", 86400) / 86400
            if uptime_days < 0.5:
                explanations.append("New system (recently booted)")
            
            return float(anomaly_score), explanations
            
        except Exception as e:
            logging.error(f"ML scoring failed: {e}")
            return 0.0, [f"ML error: {e}"]
    
    def train_model(self, training_data: List[Dict]):
        """
        Retrain model on historical data.
        
        Args:
            training_data: [{profile: {...}, is_honeypot: bool}, ...]
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            # Extract features from training data
            features = []
            for item in training_data:
                profile = item['profile']
                feat = self.extract_features(profile)
                features.append(feat[0])
            
            X = np.array(features)
            
            # Train model
            self.model = IsolationForest(
                contamination=0.1,  # Assume 10% honeypots
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X)
            
            # Save model
            import pickle
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logging.info(f"Trained ML model on {len(training_data)} samples")
            return True
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            return False
```

**Key Methods**:
- `extract_features()` - Convert node profile to 7-dim feature vector
- `score_as_honeypot()` - Isolation Forest anomaly score → (float, explanations)
- `train_model()` - Retrain on historical labeled data

---

### Part 3: Scout Integration (Days 8-9)

**File**: [agents/scout.py](agents/scout.py) - REPLACE STUB

**Current Stub** (lines 582-592):
```python
def _load_threat_intelligence(self) -> Dict[str, Any]:
    """Load threat intelligence database (stub implementation)."""
    return {
        'known_honeypots': [],
        'malicious_ips': [],
        'threat_signatures': [],
        'recent_incidents': []
    }
```

**New Implementation**:
```python
def _load_threat_intelligence(self) -> Dict[str, Any]:
    """Load real threat intelligence database with ML anomaly detection."""
    from recon.threat_intelligence import ThreatIntelligenceDB
    from recon.ml_anomaly_detector import MLAnomalyDetector
    
    # Initialize threat DB
    threat_db = ThreatIntelligenceDB(vault=self.vault, config=self.config)
    
    # Initialize ML detector
    ml_detector = MLAnomalyDetector(config=self.config.get("scout", {}))
    
    # Refresh threat data from sources (daily in production)
    threat_db.refresh_from_sources()
    
    return {
        'threat_db': threat_db,
        'ml_detector': ml_detector,
        'known_honeypots': threat_db.known_honeypots,
        'malicious_ips': threat_db.malicious_ips,
        'threat_signatures': threat_db.threat_signatures,
        'last_updated': datetime.now()
    }

def _detect_honeypots(self, targets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Real honeypot detection combining threat intelligence + ML.
    
    Returns:
        (safe_targets, honeypots)
    """
    safe_targets = []
    honeypots = []
    
    threat_db = self.threat_database.get('threat_db')
    ml_detector = self.threat_database.get('ml_detector')
    
    for target in targets:
        ip = target.get("ip")
        profile = target.copy()
        
        # Check 1: Known honeypot database
        is_honeypot_db, db_confidence = threat_db.is_honeypot(ip, profile)
        
        # Check 2: Known malicious IP
        is_malicious, mal_confidence = threat_db.is_malicious(ip, profile)
        
        # Check 3: ML anomaly detection
        ml_score, ml_explanations = ml_detector.score_as_honeypot(profile)
        
        # Combine scores
        max_honeypot_score = max(db_confidence if is_honeypot_db else 0.0, ml_score)
        
        if is_malicious:
            honeypots.append({
                **target,
                "reason": "Known malicious IP",
                "confidence": mal_confidence,
                "detection_method": "malicious_db"
            })
        elif max_honeypot_score >= 0.7:
            honeypots.append({
                **target,
                "reason": f"Honeypot (DB={db_confidence:.1%}, ML={ml_score:.1%})",
                "confidence": max_honeypot_score,
                "detection_method": "combined",
                "ml_explanations": ml_explanations
            })
        else:
            safe_targets.append({
                **target,
                "honeypot_score": max_honeypot_score,
                "risk_level": self._calculate_risk_level(profile)
            })
    
    return safe_targets, honeypots

def _calculate_risk_level(self, profile: Dict) -> str:
    """Calculate risk level (LOW/MEDIUM/HIGH/CRITICAL) for safe targets."""
    service_count = len(profile.get("services", {}))
    port_count = len(profile.get("ports", []))
    os_confidence = profile.get("os_confidence", 0.5)
    
    risk_score = 0.0
    
    # More services = more attack surface
    if service_count > 8:
        risk_score += 0.3
    
    # Open ports increase attack surface
    if port_count > 10:
        risk_score += 0.3
    
    # Low OS confidence = unknown target
    if os_confidence < 0.3:
        risk_score += 0.4
    
    if risk_score >= 0.8:
        return "CRITICAL"
    elif risk_score >= 0.5:
        return "HIGH"
    elif risk_score >= 0.2:
        return "MEDIUM"
    else:
        return "LOW"
```

**Integration Points**:
- `Scout.__init__()` calls `self._load_threat_intelligence()` ← NOW REAL
- `Scout._run()` processes targets through `_detect_honeypots()` ← NEW
- Returns filtered targets to MissionAgent

---

### Part 4: Configuration & Testing (Days 9-10)

**Update**: [core/agent_config.json](core/agent_config.json)

Add Scout section:
```json
"scout": {
  "enabled": true,
  "threat_db_refresh_interval": 86400,
  "ml_model_path": "models/honeypot_detector.pkl",
  "honeypot_confidence_threshold": 0.7,
  "ml_features": ["port_count", "service_count", "response_time_variance", ...],
  "external_threat_sources": [
    "https://mhn.example.com/api/honeypots",
    "https://abuse.ch",
    "https://api.abuseipdb.com"
  ]
}
```

**Create**: [tests/test_scout_ml.py](tests/test_scout_ml.py) (NEW - 200 lines)

```python
import pytest
from recon.threat_intelligence import ThreatIntelligenceDB
from recon.ml_anomaly_detector import MLAnomalyDetector

def test_threat_intelligence_honeypot_detection():
    """Test honeypot detection via threat DB."""
    db = ThreatIntelligenceDB(vault=mock_vault, config={})
    
    honeypot_profile = {
        "ip": "192.168.1.100",
        "banner": "Honeypot 1.0",
        "ports": [23, 69, 3389],
        "services": {"ssh": "fake", "http": "glastopf"},
    }
    
    is_honeypot, confidence = db.is_honeypot("192.168.1.100", honeypot_profile)
    assert is_honeypot
    assert confidence > 0.6

def test_ml_anomaly_detection():
    """Test ML-based anomaly scoring."""
    detector = MLAnomalyDetector()
    
    suspicious_profile = {
        "ports": [80],  # Only one port
        "services": {"http": "glastopf"},
        "response_times": [0.1, 0.1, 0.1, 0.1],  # Perfect consistency
        "os_confidence": 0.2,
        "uptime_seconds": 3600,
    }
    
    score, explanations = detector.score_as_honeypot(suspicious_profile)
    assert score > 0.6
    assert len(explanations) > 0

def test_scout_integration():
    """Test Scout honeypot filtering."""
    scout = Scout(vault=mock_vault)
    
    targets = [
        {"ip": "10.0.0.1", "ports": [22, 80], "services": {"ssh": "OpenSSH", "http": "Apache"}},
        {"ip": "10.0.0.2", "banner": "Honeypot", "ports": [23], "services": {}},
    ]
    
    safe, honeypots = scout._detect_honeypots(targets)
    
    assert len(safe) >= 1
    assert len(honeypots) >= 1
    assert any(h["ip"] == "10.0.0.2" for h in honeypots)
```

---

## Dependencies to Install

```bash
# ML & data processing
pip install scikit-learn numpy pandas

# Threat intelligence
pip install nvdlib requests

# YARA rules support (optional)
pip install yara-python
```

---

## Data Pipeline

### Training Data

For ML model training, need historical labeled data:

```json
{
  "training_samples": [
    {
      "profile": {
        "ip": "192.168.1.100",
        "ports": [80, 443],
        "services": {"http": "Apache", "https": "Apache"},
        "response_times": [0.05, 0.06, 0.04]
      },
      "is_honeypot": false,
      "source": "manual_validation"
    },
    {
      "profile": {
        "ip": "192.168.1.200",
        "ports": [23],
        "services": {"telnet": "Cowrie"},
        "response_times": [0.1, 0.1, 0.1]
      },
      "is_honeypot": true,
      "source": "known_honeypot_list"
    }
  ]
}
```

**Get Historical Data**:
1. Download from MHN project (Honeypot Database)
2. Use public honeypot IP lists
3. Manual validation during initial rollout
4. Build feedback loop: Scout predictions → Validation → Retraining

---

## Success Criteria (Phase 2.3)

✅ **Passed When**:
1. ThreatIntelligenceDB loaded successfully
   - `is_honeypot()` returns (bool, confidence)
   - `is_malicious()` returns (bool, confidence)
   - Redis caching works

2. MLAnomalyDetector trained and functional
   - Model saved to disk
   - `score_as_honeypot()` returns 0-1 scores
   - Feature extraction handles all node profiles

3. Scout integration complete
   - `_load_threat_intelligence()` returns real DB + ML detector
   - `_detect_honeypots()` filters targets
   - Returns: (safe_targets, honeypots) with explanations

4. Testing & validation
   - >90% accuracy on known honeypots
   - <5% false positives on real systems
   - <100ms latency per target

---

## Integration with Phase 2.4

Scout outputs to MissionAgent:
- `safe_targets` ← MissionAgent processes these
- `honeypots` ← Logged for intelligence
- `risk_level` (LOW/MEDIUM/HIGH/CRITICAL) ← Used for execution priority

---

## Rollout Plan

1. **Days 6-7**: Implement threat intelligence database
   - Load honey honeypot patterns
   - Load CVE data
   - Setup Redis caching

2. **Days 7-8**: Implement ML anomaly detector
   - Feature extraction
   - Isolation Forest model
   - Training pipeline

3. **Days 8-9**: Integrate with Scout agent
   - Replace `_load_threat_intelligence()` stub
   - Implement `_detect_honeypots()` method
   - Add configuration options

4. **Days 9-10**: Testing & validation
   - Unit tests for all components
   - Integration tests with TargetingEngine
   - Performance benchmarking

---

## Performance Metrics

| Metric | Target | Implementation |
|--------|--------|-----------------|
| Honeypot detection rate | >90% | Isolation Forest + signatures |
| False positive rate | <5% | Cross-validation on known systems |
| Detection latency | <100ms/target | Feature extraction + ML predict |
| Memory footprint | <500MB | Redis caching + model compression |
| Update frequency | 24h | Background refresh task |

---

## Next Phase (2.4)

Once Scout is complete, MissionAgent receives:
- Filtered targets (honeypots removed)
- Risk levels (CRITICAL/HIGH/MEDIUM/LOW)
- Execution priority queue

MissionAgent then:
1. Prioritize CRITICAL/HIGH risk targets
2. Execute real nmap scans
3. Monitor for changes
4. Apply patches safely

---

**Status**: Ready for implementation  
**Next Action**: Create `recon/threat_intelligence.py` with ThreatIntelligenceDB  
**ETA**: +4 days to Phase 2.3 completion

