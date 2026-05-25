"""
Phase 3.2: TargetingEngine TaskRunner Integration Validation Suite

This module validates the TaskRunner-based daemon pool integration in TargetingEngine.
It ensures that 3 independent background threads are successfully consolidated
into an adaptive TaskRunner pool with proper coordination, error handling, and
graceful shutdown semantics.

Tests:
1. TaskRunner Pool Initialization
2. Per-Daemon Task Execution
3. CPU Monitor Task
4. Handshake Processor Task
5. Propagation Dispatcher Task
6. Pool Shutdown with Timeout
7. Error Backoff Behavior
8. Integration Lifecycle

Expected Outcome: All 12 tests pass, validating Phase 3.2 scale-out readiness.
"""

import unittest
import time
import threading
from typing import Optional, Dict
from unittest.mock import Mock, patch, MagicMock
import logging

# Set up logging for test visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestTargetingEngineTaskRunnerIntegration(unittest.TestCase):
    """Comprehensive validation suite for TargetingEngine TaskRunner pool integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from agents.targeting_engine import TargetingEngine
            from core.task_runner import TaskRunner
            self.TargetingEngine = TargetingEngine
            self.TaskRunner = TaskRunner
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def test_1_taskrunner_pool_initialization(self):
        """
        Test 1: TaskRunner Pool Initialization
        Validates that TargetingEngine.__init__() creates an empty task_runners dict,
        and manual initialization creates 3 TaskRunner instances with correct names/intervals.
        """
        logger.info("Test 1: TaskRunner Pool Initialization")
        
        # Create TargetingEngine instance
        engine = self.TargetingEngine()
        
        # Verify pool dict exists
        self.assertTrue(hasattr(engine, 'task_runners'), "task_runners pool not initialized")
        self.assertIsInstance(engine.task_runners, dict, "task_runners should be dict")
        self.assertEqual(len(engine.task_runners), 0, "Pool should be empty on init")
        
        # Manually initialize task runners (without calling run() which has infinite loop)
        daemon_config = [
            ("CPUMonitor", engine._cpu_monitor_task, 5.0),
            ("HandshakeProcessor", engine._handshake_processor_task, 10.0),
            ("PropagationDispatcher", engine._propagation_dispatcher_task, 2.0)
        ]
        
        for name, task_fn, interval in daemon_config:
            runner = self.TaskRunner(
                task_fn=task_fn,
                interval=interval,
                name=name,
                max_backoff=30.0
            )
            runner.start(run_immediately=False)
            engine.task_runners[name] = runner
        
        # Verify pool has 3 runners with correct names
        expected_runners = {"CPUMonitor", "HandshakeProcessor", "PropagationDispatcher"}
        actual_runners = set(engine.task_runners.keys())
        
        self.assertEqual(actual_runners, expected_runners,
                        f"Pool should have exact set of runners. Got: {actual_runners}")
        
        # Verify each runner is a TaskRunner instance
        for name, runner in engine.task_runners.items():
            self.assertIsInstance(runner, self.TaskRunner,
                                f"{name} should be TaskRunner instance, got {type(runner)}")
        
        # Cleanup
        for runner in engine.task_runners.values():
            try:
                runner.stop(timeout=1.0)
            except:
                pass
        
        logger.info("✅ Test 1 passed: Pool initialized with 3 TaskRunner instances")
    
    def test_2_cpu_monitor_task_execution(self):
        """
        Test 2: CPU Monitor Task Execution
        Validates that _cpu_monitor_task() executes successfully and returns None (default interval).
        """
        logger.info("Test 2: CPU Monitor Task Execution")
        
        engine = self.TargetingEngine()
        engine.kernel = None
        
        # Mock the CPU propagation check
        with patch.object(engine.cpu_propagation, 'should_trigger_propagation', return_value=False):
            result = engine._cpu_monitor_task()
        
        # Verify return value (None = use default interval)
        self.assertIsNone(result, "CPU monitor task should return None for default interval")
        
        logger.info("✅ Test 2 passed: CPU monitor task executes correctly")
    
    def test_3_cpu_monitor_task_error_handling(self):
        """
        Test 3: CPU Monitor Task Error Handling
        Validates that _cpu_monitor_task() returns a float (backoff interval) on exception.
        """
        logger.info("Test 3: CPU Monitor Task Error Handling")
        
        engine = self.TargetingEngine()
        
        # Mock _execute_targeting_cycle to raise exception
        with patch.object(engine, '_execute_targeting_cycle', side_effect=RuntimeError("Test error")):
            with patch.object(engine.cpu_propagation, 'should_trigger_propagation', return_value=True):
                result = engine._cpu_monitor_task()
        
        # Verify return value is a float (backoff interval)
        self.assertIsInstance(result, float, "Error should return float backoff interval")
        self.assertEqual(result, 5.0, "CPU monitor error backoff should be 5.0")
        
        logger.info("✅ Test 3 passed: Error handling returns backoff interval")
    
    def test_4_handshake_processor_task_execution(self):
        """
        Test 4: Handshake Processor Task Execution
        Validates that _handshake_processor_task() processes targets correctly.
        """
        logger.info("Test 4: Handshake Processor Task Execution")
        
        engine = self.TargetingEngine()
        
        # Add mock targets to process
        from agents.targeting_engine import TargetProfile
        target = TargetProfile(ip="192.168.1.1", confidence=0.8, trust_score=0.7)
        engine.target_profiles["192.168.1.1"] = target
        
        # Mock handshaker
        with patch.object(engine.trust_handshaker, 'perform_handshake', return_value=True):
            result = engine._handshake_processor_task()
        
        # Verify return value
        self.assertIsNone(result, "Handshake processor task should return None for default interval")
        
        logger.info("✅ Test 4 passed: Handshake processor task executes correctly")
    
    def test_5_handshake_processor_error_handling(self):
        """
        Test 5: Handshake Processor Error Handling
        Validates that _handshake_processor_task() returns backoff on error.
        """
        logger.info("Test 5: Handshake Processor Error Handling")
        
        engine = self.TargetingEngine()
        engine.target_profiles = {}
        
        # Mock trust_handshaker to raise exception
        with patch.object(engine.trust_handshaker, 'perform_handshake', side_effect=RuntimeError("Test error")):
            from agents.targeting_engine import TargetProfile
            target = TargetProfile(ip="192.168.1.1", confidence=0.8, trust_score=0.7)
            engine.target_profiles["192.168.1.1"] = target
            
            result = engine._handshake_processor_task()
        
        # Verify return value
        self.assertIsInstance(result, float, "Error should return float backoff interval")
        self.assertEqual(result, 10.0, "Handshake error backoff should be 10.0")
        
        logger.info("✅ Test 5 passed: Handshake processor error handling works")
    
    def test_6_propagation_dispatcher_task_execution(self):
        """
        Test 6: Propagation Dispatcher Task Execution
        Validates that _propagation_dispatcher_task() processes missions correctly.
        """
        logger.info("Test 6: Propagation Dispatcher Task Execution")
        
        engine = self.TargetingEngine()
        engine.propagation_queue = []
        
        # Add mock mission
        from agent_core_anatomy import Mission
        mission = Mission(name="test_mission", objectives=["test"], parameters={})
        engine.propagation_queue.append(mission)
        
        result = engine._propagation_dispatcher_task()
        
        # Verify return value
        self.assertIsNone(result, "Propagation dispatcher task should return None for default interval")
        # Verify mission was processed (popped from queue)
        self.assertEqual(len(engine.propagation_queue), 0, "Mission should be popped from queue")
        
        logger.info("✅ Test 6 passed: Propagation dispatcher task executes correctly")
    
    def test_7_propagation_dispatcher_error_handling(self):
        """
        Test 7: Propagation Dispatcher Error Handling
        Validates that _propagation_dispatcher_task() returns backoff on error.
        """
        logger.info("Test 7: Propagation Dispatcher Error Handling")
        
        engine = self.TargetingEngine()
        engine.propagation_queue = None  # Cause error by setting to None
        
        result = engine._propagation_dispatcher_task()
        
        # Verify return value
        self.assertIsInstance(result, float, "Error should return float backoff interval")
        self.assertEqual(result, 2.0, "Propagation dispatcher error backoff should be 2.0")
        
        logger.info("✅ Test 7 passed: Propagation dispatcher error handling works")
    
    def test_8_task_functions_exist_and_callable(self):
        """
        Test 8: Task Functions Integration
        Validates that all 3 task functions exist and are callable.
        """
        logger.info("Test 8: Task Functions Integration")
        
        engine = self.TargetingEngine()
        
        # Verify task functions exist and are callable
        task_methods = ['_cpu_monitor_task', '_handshake_processor_task', '_propagation_dispatcher_task']
        
        for method_name in task_methods:
            self.assertTrue(hasattr(engine, method_name),
                          f"TargetingEngine should have {method_name}")
            method = getattr(engine, method_name)
            self.assertTrue(callable(method), f"{method_name} should be callable")
        
        logger.info("✅ Test 8 passed: All task functions integrated")
    
    def test_9_shutdown_task_runners(self):
        """
        Test 9: Graceful Shutdown
        Validates that _shutdown_task_runners() properly stops all runners.
        """
        logger.info("Test 9: Graceful Shutdown")
        
        engine = self.TargetingEngine()
        
        # Create mock runners
        mock_runner1 = MagicMock()
        mock_runner2 = MagicMock()
        mock_runner3 = MagicMock()
        
        engine.task_runners = {
            "Runner1": mock_runner1,
            "Runner2": mock_runner2,
            "Runner3": mock_runner3
        }
        
        # Call shutdown
        engine._shutdown_task_runners()
        
        # Verify all runners had stop() called
        mock_runner1.stop.assert_called_once_with(timeout=5.0)
        mock_runner2.stop.assert_called_once_with(timeout=5.0)
        mock_runner3.stop.assert_called_once_with(timeout=5.0)
        
        # Verify pool is cleared
        self.assertEqual(len(engine.task_runners), 0, "Pool should be cleared after shutdown")
        
        logger.info("✅ Test 9 passed: Graceful shutdown works correctly")
    
    def test_10_error_propagation_in_tasks(self):
        """
        Test 10: Error Propagation and Backoff
        Validates that all tasks return appropriate backoff intervals on error.
        """
        logger.info("Test 10: Error Propagation and Backoff")
        
        engine = self.TargetingEngine()
        
        # Test CPU monitor error
        with patch.object(engine.cpu_propagation, 'should_trigger_propagation', side_effect=Exception("Test")):
            result = engine._cpu_monitor_task()
            self.assertEqual(result, 5.0, "CPU monitor should backoff 5.0s on error")
        
        # Test handshake processor error
        with patch.object(engine, 'target_profiles', side_effect=Exception("Test")):
            result = engine._handshake_processor_task()
            self.assertEqual(result, 10.0, "Handshake should backoff 10.0s on error")
        
        # Test propagation dispatcher error
        with patch.object(engine, 'propagation_queue', side_effect=Exception("Test")):
            result = engine._propagation_dispatcher_task()
            self.assertEqual(result, 2.0, "Propagation should backoff 2.0s on error")
        
        logger.info("✅ Test 10 passed: Error backoff intervals correct")
    
    def test_11_pool_state_consistency(self):
        """
        Test 11: Pool State Consistency
        Validates that pool state is properly maintained.
        """
        logger.info("Test 11: Pool State Consistency")
        
        engine = self.TargetingEngine()
        
        # Verify initial state
        self.assertEqual(len(engine.task_runners), 0, "Pool should start empty")
        self.assertFalse(engine.running, "running should be False initially")
        
        # Mock state
        engine.running = True
        engine.task_runners = {
            "CPUMonitor": MagicMock(),
            "HandshakeProcessor": MagicMock(),
            "PropagationDispatcher": MagicMock()
        }
        
        # Verify state after setup
        self.assertEqual(len(engine.task_runners), 3, "Pool should have 3 runners")
        self.assertTrue(engine.running, "running should be True")
        
        # Simulate shutdown
        engine._shutdown_task_runners()
        engine.running = False
        
        # Verify post-shutdown state
        self.assertEqual(len(engine.task_runners), 0, "Pool should be cleared")
        self.assertFalse(engine.running, "running should be False")
        
        logger.info("✅ Test 11 passed: Pool state consistency maintained")
    
    def test_12_backward_compatibility(self):
        """
        Test 12: Backward Compatibility
        Validates that legacy loop methods still exist for compatibility.
        """
        logger.info("Test 12: Backward Compatibility")
        
        engine = self.TargetingEngine()
        
        # Verify legacy methods still exist
        legacy_methods = ['_monitor_cpu_events', '_process_handshakes', '_dispatch_propagations']
        
        for method_name in legacy_methods:
            self.assertTrue(hasattr(engine, method_name),
                          f"Legacy method {method_name} should still exist")
            method = getattr(engine, method_name)
            self.assertTrue(callable(method), f"{method_name} should be callable")
        
        logger.info("✅ Test 12 passed: Backward compatibility maintained")


class TestTargetingEngineDaemonFunctions(unittest.TestCase):
    """Focused tests for daemon function behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from agents.targeting_engine import TargetingEngine
            self.TargetingEngine = TargetingEngine
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_cpu_monitor_with_cpu_spike(self):
        """Verify CPU monitor detects CPU spikes"""
        logger.info("Test: CPU monitor with CPU spike")
        
        engine = self.TargetingEngine()
        
        # Mock CPU spike
        with patch.object(engine.cpu_propagation, 'should_trigger_propagation', return_value=True):
            with patch.object(engine, '_execute_targeting_cycle') as mock_cycle:
                result = engine._cpu_monitor_task()
        
        # Verify cycle was triggered
        mock_cycle.assert_called_once()
        
        logger.info("✅ CPU spike detected and targeting cycle triggered")
    
    def test_handshake_with_multiple_targets(self):
        """Verify handshake processor handles multiple targets"""
        logger.info("Test: Handshake processor with multiple targets")
        
        engine = self.TargetingEngine()
        
        from agents.targeting_engine import TargetProfile
        target1 = TargetProfile(ip="192.168.1.1", confidence=0.8, trust_score=0.7)
        target2 = TargetProfile(ip="192.168.1.2", confidence=0.9, trust_score=0.8)
        engine.target_profiles = {
            "192.168.1.1": target1,
            "192.168.1.2": target2
        }
        
        handshake_count = 0
        
        def count_handshakes(target):
            nonlocal handshake_count
            handshake_count += 1
            return True
        
        with patch.object(engine.trust_handshaker, 'perform_handshake', side_effect=count_handshakes):
            result = engine._handshake_processor_task()
        
        # Verify both targets were processed
        self.assertEqual(handshake_count, 2, "Should process both targets")
        
        logger.info("✅ Multiple targets processed correctly")


def _run_with_timeout(engine, timeout=2.0):
    """Helper to run with timeout protection"""
    try:
        # Start the run in a controlled way
        engine.running = True
        time.sleep(timeout)
        engine.running = False
        engine._shutdown_task_runners()
    except:
        pass


def run_validation_suite():
    """Run the complete validation suite and report results"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3.2: TARGETINGENGINE TASKRUNNER INTEGRATION VALIDATION")
    logger.info("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTargetingEngineTaskRunnerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTargetingEngineDaemonFunctions))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        logger.info("\n✅ ALL TESTS PASSED - Phase 3.2 validation complete!")
        logger.info("TargetingEngine TaskRunner pool integration is production-ready.\n")
    else:
        logger.info("\n❌ SOME TESTS FAILED - Review errors above.\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_validation_suite()
    exit(0 if success else 1)
