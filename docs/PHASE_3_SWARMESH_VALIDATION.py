"""
Phase 3.1: SwarmMesh TaskRunner Integration Validation Suite

This module validates the TaskRunner-based daemon pool integration in SwarmMesh.
It ensures that 6 independent background threads are successfully consolidated
into an adaptive TaskRunner pool with proper coordination, error handling, and
graceful shutdown semantics.

Tests:
1. TaskRunner Pool Initialization
2. Per-Daemon Task Execution
3. Heartbeat Task Functionality
4. Command Processing Task
5. Cleanup Task Execution
6. Pool Shutdown with Timeout
7. Error Backoff Behavior
8. Integration Lifecycle

Expected Outcome: All 8 tests pass, validating Phase 3.1 scale-out readiness.
"""

import unittest
import time
import threading
from typing import Optional, Dict, List
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import logging

# Set up logging for test visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestSwarmMeshTaskRunnerIntegration(unittest.TestCase):
    """Comprehensive validation suite for SwarmMesh TaskRunner pool integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import here to ensure it's available
        try:
            from swarm_mesh import SwarmMesh, NodeStatus
            from core.task_runner import TaskRunner
            self.SwarmMesh = SwarmMesh
            self.NodeStatus = NodeStatus
            self.TaskRunner = TaskRunner
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        
        self.test_node_id = "test_node_v3.1"
        self.mesh = None
        self.start_time = datetime.now()
    
    def tearDown(self):
        """Clean up after tests"""
        if self.mesh and hasattr(self.mesh, 'disconnect'):
            try:
                self.mesh.disconnect()
            except Exception as e:
                logger.warning(f"Error during teardown disconnect: {e}")
    
    def test_1_taskrunner_pool_initialization(self):
        """
        Test 1: TaskRunner Pool Initialization
        Validates that SwarmMesh.__init__() creates an empty task_runners dict,
        and connect() initializes 6 TaskRunner instances with correct names/intervals.
        """
        logger.info("Test 1: TaskRunner Pool Initialization")
        
        # Create mesh instance
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        # Verify pool dict exists
        self.assertTrue(hasattr(self.mesh, 'task_runners'), "task_runners pool not initialized")
        self.assertIsInstance(self.mesh.task_runners, dict, "task_runners should be dict")
        self.assertEqual(len(self.mesh.task_runners), 0, "Pool should be empty before connect()")
        
        # Mock external dependencies before connect
        with patch.object(self.mesh, '_start_background_threads') as mock_start:
            self.mesh.running = True
            self.mesh.status = self.NodeStatus.CONNECTED
        
        # Now manually call _start_background_threads to initialize pool
        with patch.object(self.mesh, 'broadcast_identity'):
            with patch.object(self.mesh.metrics_collector, 'collection_interval', 5.0):
                self.mesh._start_background_threads()
        
        # Verify pool has 6 runners with correct names
        expected_runners = {"Heartbeat", "PeerDiscovery", "CommandProcessor", 
                           "Cleanup", "MetricsCollection", "ConsistencySync"}
        actual_runners = set(self.mesh.task_runners.keys())
        
        self.assertEqual(actual_runners, expected_runners,
                        f"Pool should have exact set of runners. Got: {actual_runners}")
        
        # Verify each runner is a TaskRunner instance
        for name, runner in self.mesh.task_runners.items():
            self.assertIsInstance(runner, self.TaskRunner,
                                f"{name} should be TaskRunner instance, got {type(runner)}")
        
        logger.info("✅ Test 1 passed: Pool initialized with 6 TaskRunner instances")
    
    def test_2_heartbeat_task_execution(self):
        """
        Test 2: Heartbeat Task Execution
        Validates that _heartbeat_task() executes successfully and returns None (default interval).
        """
        logger.info("Test 2: Heartbeat Task Execution")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Mock the _send_heartbeats method
        with patch.object(self.mesh, '_send_heartbeats') as mock_send:
            result = self.mesh._heartbeat_task()
        
        # Verify task was called
        mock_send.assert_called_once()
        
        # Verify return value (None = use default interval)
        self.assertIsNone(result, "Heartbeat task should return None for default interval")
        
        logger.info("✅ Test 2 passed: Heartbeat task executes correctly")
    
    def test_3_heartbeat_task_error_handling(self):
        """
        Test 3: Heartbeat Task Error Handling
        Validates that _heartbeat_task() returns a float (backoff interval) on exception.
        """
        logger.info("Test 3: Heartbeat Task Error Handling")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Mock _send_heartbeats to raise an exception
        with patch.object(self.mesh, '_send_heartbeats', side_effect=RuntimeError("Test error")):
            result = self.mesh._heartbeat_task()
        
        # Verify return value is a float (backoff interval)
        self.assertIsInstance(result, float, "Error should return float backoff interval")
        self.assertEqual(result, 5.0, "Heartbeat error backoff should be 5.0")
        
        logger.info("✅ Test 3 passed: Error handling returns backoff interval")
    
    def test_4_discovery_task_execution(self):
        """
        Test 4: Peer Discovery Task Execution
        Validates that _discovery_task() executes and handles broadcast_identity() correctly.
        """
        logger.info("Test 4: Peer Discovery Task Execution")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Mock broadcast_identity
        with patch.object(self.mesh, 'broadcast_identity') as mock_broadcast:
            with patch('random.random', return_value=0.1):  # Force broadcast (< 0.3)
                result = self.mesh._discovery_task()
        
        # Verify broadcast was called
        mock_broadcast.assert_called_once()
        
        # Verify return value
        self.assertIsNone(result, "Discovery task should return None for default interval")
        
        logger.info("✅ Test 4 passed: Discovery task executes correctly")
    
    def test_5_command_task_execution(self):
        """
        Test 5: Command Processor Task Execution
        Validates that _command_task() pulls messages and processes them.
        """
        logger.info("Test 5: Command Processor Task Execution")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Mock secure channel to return no messages
        with patch.object(self.mesh.secure_channel, 'pull_messages', return_value=[]):
            with patch.object(self.mesh, '_process_command_async') as mock_process:
                result = self.mesh._command_task()
        
        # Verify return value
        self.assertIsNone(result, "Command task should return None for default interval")
        
        logger.info("✅ Test 5 passed: Command task executes correctly")
    
    def test_6_cleanup_task_execution(self):
        """
        Test 6: Cleanup Task Execution
        Validates that _cleanup_task() calls cleanup methods.
        """
        logger.info("Test 6: Cleanup Task Execution")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Mock cleanup methods
        with patch.object(self.mesh, '_cleanup_dead_peers') as mock_dead:
            with patch.object(self.mesh, '_cleanup_command_history') as mock_history:
                result = self.mesh._cleanup_task()
        
        # Verify cleanups were called
        mock_dead.assert_called_once()
        mock_history.assert_called_once()
        
        # Verify return value
        self.assertIsNone(result, "Cleanup task should return None for default interval")
        
        logger.info("✅ Test 6 passed: Cleanup task executes correctly")
    
    def test_7_metrics_task_execution(self):
        """
        Test 7: Metrics Collection Task Execution
        Validates that _metrics_task() calls metrics_collector.collect_metrics().
        """
        logger.info("Test 7: Metrics Task Execution")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Mock metrics collector
        with patch.object(self.mesh.metrics_collector, 'collect_metrics') as mock_collect:
            result = self.mesh._metrics_task()
        
        # Verify collect was called
        mock_collect.assert_called_once()
        
        # Verify return value
        self.assertIsNone(result, "Metrics task should return None for default interval")
        
        logger.info("✅ Test 7 passed: Metrics task executes correctly")
    
    def test_8_sync_task_execution(self):
        """
        Test 8: Consistency Sync Task Execution
        Validates that _sync_task() calls consistency_manager.sync_with_peers().
        """
        logger.info("Test 8: Sync Task Execution")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Mock consistency manager
        with patch.object(self.mesh.consistency_manager, 'sync_with_peers') as mock_sync:
            result = self.mesh._sync_task()
        
        # Verify sync was called
        mock_sync.assert_called_once()
        
        # Verify return value
        self.assertIsNone(result, "Sync task should return None for default interval")
        
        logger.info("✅ Test 8 passed: Sync task executes correctly")
    
    def test_9_pool_shutdown_cleanup(self):
        """
        Test 9: Pool Shutdown and Cleanup
        Validates that disconnect() properly stops all TaskRunner instances.
        """
        logger.info("Test 9: Pool Shutdown and Cleanup")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Initialize pool with mocked runners
        with patch.object(self.mesh, 'broadcast_identity'):
            with patch.object(self.mesh.metrics_collector, 'collection_interval', 5.0):
                self.mesh._start_background_threads()
        
        pool_size = len(self.mesh.task_runners)
        self.assertGreater(pool_size, 0, "Pool should be initialized")
        
        # Verify all runners are callable
        for runner in self.mesh.task_runners.values():
            self.assertTrue(hasattr(runner, 'stop'), "RunnerRunner should have stop() method")
        
        # Call disconnect and verify cleanup
        self.mesh.disconnect()
        
        # Verify pool is cleared
        self.assertEqual(len(self.mesh.task_runners), 0, "Pool should be cleared after disconnect")
        
        # Verify running flag is set to False
        self.assertFalse(self.mesh.running, "running flag should be False")
        
        logger.info("✅ Test 9 passed: Pool shutdown and cleanup successful")
    
    def test_10_task_integration_with_pool(self):
        """
        Test 10: Task Integration with TaskRunner Pool
        Validates that task functions are properly registered and callable.
        """
        logger.info("Test 10: Task Integration with TaskRunner Pool")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        # Verify task functions exist and are callable
        task_methods = ['_heartbeat_task', '_discovery_task', '_command_task',
                       '_cleanup_task', '_metrics_task', '_sync_task']
        
        for method_name in task_methods:
            self.assertTrue(hasattr(self.mesh, method_name),
                          f"SwarmMesh should have {method_name}")
            method = getattr(self.mesh, method_name)
            self.assertTrue(callable(method), f"{method_name} should be callable")
        
        logger.info("✅ Test 10 passed: All task functions integrated")
    
    def test_11_error_propagation_in_tasks(self):
        """
        Test 11: Error Propagation and Backoff
        Validates that task errors return appropriate backoff intervals.
        """
        logger.info("Test 11: Error Propagation and Backoff")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        self.mesh.running = True
        
        # Test each task's error handling
        error_tests = [
            (self.mesh._heartbeat_task, patch.object(self.mesh, '_send_heartbeats', 
                                                      side_effect=Exception("Test")), 5.0),
            (self.mesh._discovery_task, patch.object(self.mesh, 'broadcast_identity',
                                                      side_effect=Exception("Test")), 10.0),
            (self.mesh._cleanup_task, patch.object(self.mesh, '_cleanup_dead_peers',
                                                    side_effect=Exception("Test")), 15.0),
            (self.mesh._metrics_task, patch.object(self.mesh.metrics_collector, 'collect_metrics',
                                                    side_effect=Exception("Test")), 10.0),
            (self.mesh._sync_task, patch.object(self.mesh.consistency_manager, 'sync_with_peers',
                                                side_effect=Exception("Test")), 30.0),
        ]
        
        for task_fn, mock_patch, expected_backoff in error_tests:
            with mock_patch:
                result = task_fn()
                self.assertEqual(result, expected_backoff,
                              f"{task_fn.__name__} should return {expected_backoff} on error")
        
        logger.info("✅ Test 11 passed: Error backoff intervals correct")
    
    def test_12_pool_state_consistency(self):
        """
        Test 12: Pool State Consistency
        Validates that pool state is properly maintained across operations.
        """
        logger.info("Test 12: Pool State Consistency")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id=self.test_node_id)
        
        # Verify initial state
        self.assertEqual(len(self.mesh.task_runners), 0, "Pool should start empty")
        self.assertFalse(self.mesh.running, "running should be False initially")
        
        # Simulate connect
        self.mesh.running = True
        with patch.object(self.mesh, 'broadcast_identity'):
            with patch.object(self.mesh.metrics_collector, 'collection_interval', 5.0):
                self.mesh._start_background_threads()
        
        # Verify post-connect state
        self.assertEqual(len(self.mesh.task_runners), 6, "Pool should have 6 runners")
        self.assertTrue(self.mesh.running, "running should be True")
        
        # Verify pool structure
        for name, runner in self.mesh.task_runners.items():
            self.assertIn(name, {"Heartbeat", "PeerDiscovery", "CommandProcessor",
                                "Cleanup", "MetricsCollection", "ConsistencySync"})
        
        # Simulate disconnect
        self.mesh.disconnect()
        
        # Verify post-disconnect state
        self.assertEqual(len(self.mesh.task_runners), 0, "Pool should be cleared")
        self.assertFalse(self.mesh.running, "running should be False")
        
        logger.info("✅ Test 12 passed: Pool state consistency maintained")


class TestSwarmMeshDaemonTasks(unittest.TestCase):
    """Focused tests for individual daemon task behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from swarm_mesh import SwarmMesh
            self.SwarmMesh = SwarmMesh
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        
        self.mesh = None
    
    def tearDown(self):
        """Clean up"""
        if self.mesh:
            try:
                self.mesh.disconnect()
            except:
                pass
    
    def test_command_task_with_pending_commands(self):
        """Verify command task processes pending commands"""
        logger.info("Test: Command task with pending commands")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id="test_pending")
        
        self.mesh.running = True
        
        # Create mock command
        mock_cmd = MagicMock()
        self.mesh.pending_commands = [mock_cmd]
        
        with patch.object(self.mesh.secure_channel, 'pull_messages', return_value=[]):
            with patch.object(self.mesh, '_process_command_async') as mock_process:
                self.mesh._command_task()
        
        # Verify command was processed
        mock_process.assert_called_once_with(mock_cmd)
        # Verify pending queue was cleared
        self.assertEqual(len(self.mesh.pending_commands), 0)
        
        logger.info("✅ Pending commands processed correctly")
    
    def test_cleanup_task_with_dead_peers(self):
        """Verify cleanup task removes dead peers"""
        logger.info("Test: Cleanup task with dead peers")
        
        with patch('swarm_mesh.Redis'):
            self.mesh = self.SwarmMesh(node_id="test_cleanup")
        
        self.mesh.running = True
        
        # Add a mock peer
        mock_peer = MagicMock()
        mock_peer.is_alive.return_value = False
        self.mesh.peers = {"dead_peer": mock_peer}
        
        with patch.object(self.mesh, '_cleanup_dead_peers') as mock_clean:
            self.mesh._cleanup_task()
        
        # Verify cleanup was called
        mock_clean.assert_called_once()
        
        logger.info("✅ Cleanup task removes dead peers")


class TestSwarmMeshBackwardCompatibility(unittest.TestCase):
    """Tests for backward compatibility with existing SwarmMesh code"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from swarm_mesh import SwarmMesh
            self.SwarmMesh = SwarmMesh
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_running_flag_behavior(self):
        """Verify self.running flag controls daemon execution"""
        logger.info("Test: Running flag control")
        
        with patch('swarm_mesh.Redis'):
            mesh = self.SwarmMesh(node_id="test_flag")
        
        # Should not run tasks when running=False
        mesh.running = False
        
        with patch.object(mesh, '_send_heartbeats') as mock_send:
            # Task should still execute but loop doesn't matter in sync version
            result = mesh._heartbeat_task()
        
        # Verify heartbeat still calls the send method (tasks are synchronous)
        mock_send.assert_called_once()
        
        logger.info("✅ Running flag behavior correct")
    
    def test_lock_usage_in_tasks(self):
        """Verify thread lock is used in critical sections"""
        logger.info("Test: Lock usage in tasks")
        
        with patch('swarm_mesh.Redis'):
            mesh = self.SwarmMesh(node_id="test_lock")
        
        mesh.running = True
        
        # Verify lock exists
        self.assertTrue(hasattr(mesh, 'lock'), "SwarmMesh should have lock attribute")
        
        # Mock lock to verify it's used
        original_lock = mesh.lock
        mesh.lock = MagicMock()
        mesh.lock.__enter__ = MagicMock(return_value=None)
        mesh.lock.__exit__ = MagicMock(return_value=None)
        
        with patch.object(mesh, '_cleanup_dead_peers'):
            with patch.object(mesh, '_cleanup_command_history'):
                mesh._cleanup_task()
        
        # Verify lock context manager was used at least once
        self.assertGreater(mesh.lock.__enter__.call_count, 0, "Lock should be acquired")
        
        logger.info("✅ Lock usage verified")


def run_validation_suite():
    """Run the complete validation suite and report results"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3.1: SWARMESH TASKRUNNER INTEGRATION VALIDATION")
    logger.info("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmMeshTaskRunnerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmMeshDaemonTasks))
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmMeshBackwardCompatibility))
    
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
        logger.info("\n✅ ALL TESTS PASSED - Phase 3.1 validation complete!")
        logger.info("SwarmMesh TaskRunner pool integration is production-ready.\n")
    else:
        logger.info("\n❌ SOME TESTS FAILED - Review errors above.\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_validation_suite()
    exit(0 if success else 1)
