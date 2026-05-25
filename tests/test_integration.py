#!/usr/bin/env python3
"""
Integration Test: PrivescEngine with Blackroot Framework
Tests the wiring of PrivescEngine into the kernel and agents
"""

import sys
import logging
import json
import time

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('INTEGRATION-TEST')

def test_privesc_module():
    """Test PrivescEngine module independently"""
    logger.info("=" * 70)
    logger.info("TEST 1: PrivescEngine Module Load")
    logger.info("=" * 70)
    
    try:
        from agents.privesc import PrivescEngine
        logger.info("✓ PrivescEngine imported successfully")
        
        # Initialize engine
        engine = PrivescEngine()
        logger.info("✓ PrivescEngine instantiated")
        
        # Run assessment
        assessment = engine.assess_environment()
        threat_level = assessment.get('threat_level', 0)
        techniques = assessment.get('techniques', [])
        
        logger.info(f"✓ Assessment completed: {threat_level}/100 threat, {len(techniques)} techniques")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_kernel_integration():
    """Test PrivescEngine integration into kernel"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Kernel Integration (with fallback vault/swarm)")
    logger.info("=" * 70)
    
    try:
        # Mock vault and swarm for testing
        class MockVault:
            def store(self, key, value):
                pass
            def retrieve(self, key):
                return b""
        
        class MockSwarm:
            def __init__(self):
                self.redis = MockRedis()
        
        class MockRedis:
            def publish(self, channel, message):
                pass
        
        # Now test kernel
        from kernel.core import BlackrootKernel
        
        config = {
            'heartbeat_interval': 60,
            'stealth_level': 'adaptive',
            'stealth_threshold': 0.8,
            'agent_whitelist': ['INF-VENOM'],
            'vault_password': 'test_password'
        }
        
        logger.info("Initializing BlackrootKernel with config...")
        kernel = BlackrootKernel(config)
        logger.info("✓ Kernel initialized")
        
        # Check PrivescEngine
        if hasattr(kernel, 'privesc_engine'):
            logger.info("✓ PrivescEngine found in kernel.privesc_engine")
            
            # Test assessment through kernel
            assessment = kernel.privesc_engine.assess_environment()
            logger.info(f"✓ Kernel.privesc_engine.assess_environment() returned threat={assessment.get('threat_level', 0)}")
            return True
        else:
            logger.error("✗ PrivescEngine NOT found in kernel")
            return False
            
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_cli_flag():
    """Test that CLI flag was added"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: CLI Flag --enable-privesc")
    logger.info("=" * 70)
    
    try:
        with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if '--enable-privesc' in content:
            logger.info("✓ --enable-privesc flag found in main.py")
            
            if 'enable_privesc' in content:
                logger.info("✓ enable_privesc argument parsing found")
                return True
            else:
                logger.error("✗ enable_privesc argument parsing not found")
                return False
        else:
            logger.error("✗ --enable-privesc flag not found in main.py")
            return False
            
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


def test_infiltrator_hook():
    """Test that infiltrator has escalation hook"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Infiltrator Escalation Hook")
    logger.info("=" * 70)
    
    try:
        with open('agents/infiltrator.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if 'PRIVILEGE ESCALATION INTEGRATION' in content:
            logger.info("✓ Escalation integration marker found in infiltrator")
            
            if 'privesc_engine' in content and 'attempt_escalation' in content:
                logger.info("✓ privesc_engine escalation call found")
                logger.info("✓ Infiltrator will attempt escalation post-infiltration")
                return True
            else:
                logger.error("✗ privesc_engine call not found in infiltrator")
                return False
        else:
            logger.error("✗ Escalation integration marker not found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


def test_anti_analysis_hook():
    """Test that anti-analysis phase includes privesc assessment"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Anti-Analysis Phase Hook")
    logger.info("=" * 70)
    
    try:
        with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if 'privesc_assessment' in content:
            logger.info("✓ privesc_assessment found in main.py anti-analysis phase")
            
            if 'escalation_opportunities' in content:
                logger.info("✓ Escalation opportunity tracking found")
                return True
            else:
                logger.error("✗ Escalation opportunity tracking not found")
                return False
        else:
            logger.error("✗ privesc_assessment not found in main.py")
            return False
            
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    logger.info("\n")
    logger.info("#" * 70)
    logger.info("# BLACKROOT PRIVILEGE ESCALATION INTEGRATION TEST SUITE")
    logger.info("#" * 70)
    logger.info("\n")
    
    results = {
        'PrivescEngine Module': test_privesc_module(),
        'Kernel Integration': test_kernel_integration(),
        'CLI Flag': test_cli_flag(),
        'Infiltrator Hook': test_infiltrator_hook(),
        'Anti-Analysis Hook': test_anti_analysis_hook(),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} | {test_name}")
    
    logger.info("=" * 70)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED")
        logger.info("\nNext steps:")
        logger.info("1. Start Redis server (required for full runtime)")
        logger.info("2. Run: python main.py --api-mode --enable-privesc")
        logger.info("3. Run: python main.py --target example.com --enable-privesc --use-infiltrator")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
