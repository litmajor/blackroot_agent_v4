"""
test_edr_detector_integration.py - Verify EDR detector + behavioral evasion integration
=======================================================================================
Tests:
1. EDR threat detection (process scanning)
2. Threat level calculation
3. Adaptive evasion intensity mapping
4. Orchestrator autonomous operation
"""

import time
import sys
import logging
from edr_detector import EDRDetector, AdaptiveEvasionOrchestrator, ThreatLevel

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("Test")

def test_detector_initialization():
    """Test EDR detector initializes correctly."""
    log.info("[TEST 1] EDR Detector initialization...")
    detector = EDRDetector()
    assert detector is not None, "Detector failed to initialize"
    assert len(detector.THREAT_SIGNATURES) > 0, "No threat signatures loaded"
    log.info(f"✓ Detector initialized with {len(detector.THREAT_SIGNATURES)} signatures")
    return detector

def test_threat_scanning(detector: EDRDetector):
    """Test threat scanning."""
    log.info("\n[TEST 2] Threat scanning...")
    result = detector.scan_for_threats()
    assert result is not None, "Scan returned None"
    assert result.threat_level in ThreatLevel, "Invalid threat level"
    log.info(f"✓ Scan complete: {result.threat_level.name} (score={result.total_score})")
    log.info(f"  - Processes detected: {len(result.detected_processes)}")
    log.info(f"  - Drivers detected: {len(result.detected_drivers)}")
    log.info(f"  - Registry entries: {len(result.detected_registry)}")
    return result

def test_threat_level_calculation(detector: EDRDetector):
    """Test threat level calculation logic."""
    log.info("\n[TEST 3] Threat level calculation...")
    
    # Test mapping: score 0 = CLEAR
    from edr_detector import DetectionResult
    result = DetectionResult(total_score=0)
    level = detector.calculate_threat_level(result)
    assert level == ThreatLevel.CLEAR, f"Expected CLEAR, got {level.name}"
    
    # Test mapping: score 5 = MEDIUM
    result = DetectionResult(total_score=5)
    level = detector.calculate_threat_level(result)
    assert level == ThreatLevel.MEDIUM, f"Expected MEDIUM, got {level.name}"
    
    # Test mapping: score 20+ = CRITICAL
    result = DetectionResult(total_score=25)
    level = detector.calculate_threat_level(result)
    assert level == ThreatLevel.CRITICAL, f"Expected CRITICAL, got {level.name}"
    
    log.info("✓ Threat level calculation verified (CLEAR / MEDIUM / CRITICAL)")
    return True

def test_orchestrator_initialization():
    """Test orchestrator initializes correctly."""
    log.info("\n[TEST 4] Orchestrator initialization...")
    orchestrator = AdaptiveEvasionOrchestrator(scan_interval=5, auto_start=False)
    assert orchestrator is not None, "Orchestrator failed to initialize"
    assert orchestrator.detector is not None, "Detector not initialized"
    assert orchestrator.current_threat_level == ThreatLevel.CLEAR, "Wrong initial threat level"
    assert orchestrator.current_noise_intensity == 0, "Wrong initial intensity"
    log.info("✓ Orchestrator initialized correctly")
    return orchestrator

def test_intensity_mapping(orchestrator: AdaptiveEvasionOrchestrator):
    """Test evasion intensity mapping."""
    log.info("\n[TEST 5] Evasion intensity mapping...")
    
    test_cases = [
        (ThreatLevel.CLEAR, 0),
        (ThreatLevel.LOW, 2),
        (ThreatLevel.MEDIUM, 4),
        (ThreatLevel.HIGH, 7),
        (ThreatLevel.CRITICAL, 10),
    ]
    
    for threat_level, expected_intensity in test_cases:
        intensity = orchestrator._calculate_evasion_intensity(threat_level)
        assert intensity == expected_intensity, f"Expected {expected_intensity}%, got {intensity}%"
        log.info(f"✓ {threat_level.name:10} → {intensity}% intensity")
    
    return True

def test_autonomous_monitoring():
    """Test autonomous monitoring loop (5 seconds)."""
    log.info("\n[TEST 6] Autonomous monitoring (5 second test)...")
    
    orchestrator = AdaptiveEvasionOrchestrator(scan_interval=2, auto_start=False)
    
    try:
        orchestrator.start_monitoring()
        log.info("✓ Monitoring started")
        
        # Run for 5 seconds
        for i in range(5):
            time.sleep(1)
            summary = orchestrator.get_threat_summary()
            log.info(f"  [{i+1}s] Threat: {summary['current_threat_level']}, "
                    f"Intensity: {summary['evasion_intensity']}%")
        
        log.info("✓ Autonomous monitoring operational")
        
    finally:
        orchestrator.stop_monitoring()
        log.info("✓ Monitoring stopped")
    
    return True

def test_threat_summary():
    """Test threat summary generation."""
    log.info("\n[TEST 7] Threat summary generation...")
    
    orchestrator = AdaptiveEvasionOrchestrator(scan_interval=5, auto_start=False)
    
    # Trigger one scan
    orchestrator.detector.scan_for_threats()
    
    summary = orchestrator.get_threat_summary()
    assert 'current_threat_level' in summary, "Missing threat level in summary"
    assert 'current_threat_score' in summary, "Missing threat score in summary"
    assert 'evasion_intensity' in summary, "Missing evasion intensity in summary"
    assert 'detections' in summary, "Missing detection counts in summary"
    
    log.info("✓ Threat summary generated successfully")
    log.info(f"  - Threat level: {summary['current_threat_level']}")
    log.info(f"  - Threat score: {summary['current_threat_score']}")
    log.info(f"  - Evasion intensity: {summary['evasion_intensity']}%")
    log.info(f"  - Detections: {summary['detections']}")
    
    return True

def main():
    """Run all integration tests."""
    log.info("=" * 70)
    log.info("EDR DETECTOR + BEHAVIORAL EVASION INTEGRATION TEST")
    log.info("=" * 70)
    
    tests = [
        ("EDR Detector Initialization", lambda: test_detector_initialization()),
        ("EDR Threat Scanning", lambda: (detector := test_detector_initialization(), 
                                         test_threat_scanning(detector))[1]),
        ("Threat Level Calculation", lambda: test_threat_level_calculation(
                                         test_detector_initialization())),
        ("Orchestrator Initialization", lambda: test_orchestrator_initialization()),
        ("Evasion Intensity Mapping", lambda: test_intensity_mapping(
                                         test_orchestrator_initialization())),
        ("Threat Summary Generation", lambda: test_threat_summary()),
        ("Autonomous Monitoring", lambda: test_autonomous_monitoring()),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS"))
        except Exception as e:
            log.error(f"✗ FAILED: {e}")
            results.append((test_name, f"FAIL: {e}"))
    
    # Print summary
    log.info("\n" + "=" * 70)
    log.info("TEST SUMMARY")
    log.info("=" * 70)
    
    passed = sum(1 for _, result in results if result == "PASS")
    total = len(results)
    
    for test_name, result in results:
        status_icon = "✓" if result == "PASS" else "✗"
        log.info(f"{status_icon} {test_name:40} {result}")
    
    log.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        log.info("\n🎉 ALL TESTS PASSED - BEHAVIORAL EVASION MODULE READY")
        return 0
    else:
        log.error(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
