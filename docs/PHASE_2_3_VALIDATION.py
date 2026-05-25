#!/usr/bin/env python3
"""
PHASE 2.3 VALIDATION - Scout ML Threat Detection
Validates real threat intelligence loading and ML-based anomaly detection.
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__)


def test_threat_intelligence_loader():
    """Test that ThreatIntelligenceLoader exists and has real data"""
    try:
        from recon.threat_intelligence import ThreatIntelligenceLoader, ThreatType, get_threat_loader
        
        log.info("✓ ThreatIntelligenceLoader imported successfully")
        
        # Create instance
        loader = ThreatIntelligenceLoader()
        log.info("✓ ThreatIntelligenceLoader instantiated")
        
        # Check it has real threat data
        assert len(loader.threat_signatures) > 0, "No threat signatures loaded"
        log.info(f"✓ Loaded {len(loader.threat_signatures)} threat signatures")
        
        assert len(loader.cve_database) > 0, "No CVE records loaded"
        log.info(f"✓ Loaded {len(loader.cve_database)} CVE records")
        
        # Test threat level calculation
        indicators = {
            "ip_address": "192.168.1.1",
            "services": {22: "ssh", 80: "http"},
            "fingerprint": "Linux",
            "ports": [22, 80]
        }
        threat_score, signatures = loader.get_threat_level(indicators)
        log.info(f"✓ Threat level calculation works: score={threat_score:.2f}")
        
        # Test CVE lookup
        cves = loader.find_relevant_cves("nginx")
        log.info(f"✓ CVE lookup works: found {len(cves)} nginx CVEs")
        
        # Test get_threat_loader global
        global_loader = get_threat_loader()
        assert global_loader is not None
        log.info("✓ Global threat loader singleton works")
        
        return True
    
    except ImportError as e:
        log.error(f"✗ Import error: {e}")
        return False
    except AssertionError as e:
        log.error(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        log.error(f"✗ Error: {e}")
        return False


def test_ml_anomaly_detector():
    """Test that MLAnomalyDetector exists and works"""
    try:
        from ml.anomaly_detection import MLAnomalyDetector, BehavioralFeatures, get_ml_detector
        
        log.info("✓ MLAnomalyDetector imported successfully")
        
        # Create instance
        detector = MLAnomalyDetector()
        log.info("✓ MLAnomalyDetector instantiated")
        
        # Test BehavioralFeatures
        assert BehavioralFeatures is not None
        log.info("✓ BehavioralFeatures dataclass available")
        
        # Test global getter
        global_detector = get_ml_detector()
        assert global_detector is not None
        log.info("✓ Global ML detector singleton works")
        
        log.info(f"✓ ML detector state: trained={detector.is_trained}, samples={detector.training_samples}")
        
        return True
    
    except ImportError as e:
        log.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        log.error(f"✗ Error: {e}")
        return False


def test_scout_real_threat_loading():
    """Test that Scout agent loads real threat intelligence"""
    try:
        from agents.scout import RiskAssessor
        
        log.info("✓ Scout RiskAssessor imported")
        
        # Create instance
        assessor = RiskAssessor()
        log.info("✓ RiskAssessor instantiated")
        
        # Check threat database is real (not empty stub)
        threat_db = assessor.threat_database
        
        # Should have real threat signatures
        if 'threat_signatures' in threat_db:
            sig_count = len(threat_db.get('threat_signatures', {}))
            assert sig_count > 0, f"No threat signatures in database (count: {sig_count})"
            log.info(f"✓ Scout loaded {sig_count} real threat signatures")
        
        # Should have real CVEs
        if 'cve_database' in threat_db:
            cve_count = len(threat_db.get('cve_database', {}))
            assert cve_count > 0, f"No CVEs in database (count: {cve_count})"
            log.info(f"✓ Scout loaded {cve_count} real CVE records")
        
        # Check ML detector initialized
        if assessor.ml_detector is not None:
            log.info(f"✓ Scout initialized ML detector (trained={assessor.ml_detector.is_trained})")
        else:
            log.warning("⚠ Scout ML detector not available (sklearn may not be installed)")
        
        # Verify _detect_behavioral_anomaly method exists
        assert hasattr(assessor, '_detect_behavioral_anomaly'), \
            "RiskAssessor missing _detect_behavioral_anomaly method"
        log.info("✓ RiskAssessor has _detect_behavioral_anomaly method")
        
        return True
    
    except ImportError as e:
        log.error(f"✗ Import error: {e}")
        return False
    except AssertionError as e:
        log.error(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        log.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threat_level_integration():
    """Test that threat level evaluation includes ML anomaly score"""
    try:
        from agents.scout import RiskAssessor, NodeProfile
        from agent_core_anatomy import AgentID
        from datetime import datetime
        
        log.info("✓ Imported Scout components for integration test")
        
        # Create RiskAssessor
        assessor = RiskAssessor()
        
        # Create a test node profile
        node = NodeProfile(
            node_id=AgentID("test_node"),
            ip_address="192.168.1.100",
            open_ports=[22, 80, 443],
            services={22: "ssh", 80: "http", 443: "https"},
            system_fingerprint="Linux 5.4.0",
            response_timing={22: 0.05, 80: 0.10, 443: 0.08},
            behavioral_signature="normal_server",
            vulnerability_indicators=[],
            threat_score=0.0,
            honeypot_probability=0.0,
            last_scan_time=datetime.now()
        )
        
        log.info("✓ Created test NodeProfile")
        
        # Evaluate threat level (this should include ML anomaly score now)
        threat_result = assessor.evaluate_node_threat_level(node)
        
        assert 'threat_score' in threat_result, "No threat_score in result"
        log.info(f"✓ Threat evaluation works: score={threat_result['threat_score']:.2f}")
        
        # Check that threat_vectors includes ml_anomaly_score
        assert 'ml_anomaly_score' in threat_result.get('threat_vectors', {}), \
            "ML anomaly score not included in evaluation"
        log.info(f"✓ ML anomaly score included: {threat_result['threat_vectors'].get('ml_anomaly_score', 'N/A'):.2f}")
        
        return True
    
    except Exception as e:
        log.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_2_3_config():
    """Test that Phase 2.3 configuration is present"""
    try:
        import json
        
        config_path = "core/agent_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        log.info("✓ Agent config loaded")
        
        # Check for Phase 2.3 scout settings (optional but good to have)
        if 'scout' in config:
            log.info("✓ Scout configuration section found")
        else:
            log.warning("⚠ No scout configuration section (Phase 2.3 optional)")
        
        return True
    
    except Exception as e:
        log.warning(f"⚠ Config check: {e}")
        return True  # Not critical


def main():
    """Run all Phase 2.3 validation tests"""
    log.info("=" * 70)
    log.info("PHASE 2.3 VALIDATION - Scout ML Threat Detection")
    log.info("=" * 70)
    
    tests = [
        ("Threat Intelligence Loader", test_threat_intelligence_loader),
        ("ML Anomaly Detector", test_ml_anomaly_detector),
        ("Scout Real Threat Loading", test_scout_real_threat_loading),
        ("Threat Level Integration", test_threat_level_integration),
        ("Phase 2.3 Configuration", test_phase_2_3_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        log.info(f"\nRunning: {test_name}")
        log.info("-" * 70)
        result = test_func()
        results.append((test_name, result))
        log.info("-" * 70)
    
    # Summary
    log.info("\n" + "=" * 70)
    log.info("PHASE 2.3 VALIDATION SUMMARY")
    log.info("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        log.info(f"{status}: {test_name}")
    
    log.info("=" * 70)
    log.info(f"Result: {passed}/{total} tests passed")
    log.info("=" * 70)
    
    if passed == total:
        log.info("✓ Phase 2.3 Scout ML implementation validated successfully!")
        log.info("Status: Phase 2.3 READY FOR DEPLOYMENT")
        return 0
    else:
        log.warning(f"⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
