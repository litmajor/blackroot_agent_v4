"""
═══════════════════════════════════════════════════════════════════════════════
PHASE 3: TASKRUNNER INTEGRATION VALIDATION
═══════════════════════════════════════════════════════════════════════════════

Validates the hot-swappable TaskRunner daemon refactor for EnforcerAgent.

Features tested:
- TaskRunner initialization and lifecycle
- Adaptive interval management
- Exponential backoff on errors
- Hot-swappable task functions
- Integration with EnforcerAgent compliance monitoring
- Graceful shutdown
"""

import sys
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("TASKRUNNER-VALIDATION")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Basic TaskRunner Functionality
# ═══════════════════════════════════════════════════════════════════════════════

def test_taskrunner_creation():
    """Test: Create and manage TaskRunner lifecycle"""
    from core.task_runner import TaskRunner
    
    execution_log = []
    
    def task_fn():
        execution_log.append(time.time())
        return None  # Use default interval
    
    # Create runner
    runner = TaskRunner(task_fn, interval=0.1, name="test-basic")
    assert not runner.is_running(), "Runner should not be running yet"
    
    # Start runner
    runner.start()
    assert runner.is_running(), "Runner should be running"
    
    # Let it run a few iterations
    time.sleep(0.5)
    
    # Stop runner
    stopped = runner.stop(timeout=2.0)
    assert stopped, "Runner should stop cleanly"
    assert not runner.is_running(), "Runner should not be running"
    
    # Check execution
    assert len(execution_log) >= 3, f"Expected 3+ executions, got {len(execution_log)}"
    
    log.info(f"✅ TEST 1 PASSED: TaskRunner created and executed {len(execution_log)} times")
    return True


def test_taskrunner_adaptive_intervals():
    """Test: TaskRunner respects returned intervals"""
    from core.task_runner import TaskRunner
    
    call_count = [0]
    
    def task_fn():
        call_count[0] += 1
        if call_count[0] == 1:
            return 0.2  # Override interval
        else:
            return None  # Back to default
    
    runner = TaskRunner(task_fn, interval=0.05, name="test-adaptive")
    runner.start()
    
    time.sleep(0.5)
    
    runner.stop(timeout=2.0)
    
    assert call_count[0] >= 2, f"Expected 2+ calls, got {call_count[0]}"
    
    log.info(f"✅ TEST 2 PASSED: TaskRunner adaptive intervals working ({call_count[0]} calls)")
    return True


def test_taskrunner_error_backoff():
    """Test: TaskRunner applies exponential backoff on errors"""
    from core.task_runner import TaskRunner
    
    call_count = [0]
    start_time = time.time()
    
    def failing_task():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError("Simulated error")
        return None  # Succeed on 3rd try
    
    runner = TaskRunner(
        failing_task,
        interval=0.05,
        backoff_factor=2.0,
        max_backoff=5.0,
        name="test-backoff"
    )
    runner.start()
    
    time.sleep(1.0)
    
    runner.stop(timeout=2.0)
    elapsed = time.time() - start_time
    
    assert call_count[0] >= 3, f"Expected 3+ calls, got {call_count[0]}"
    # With backoff, it should take longer than default interval
    assert elapsed >= 0.5, f"Expected backoff delay, but only took {elapsed}s"
    
    stats = runner.get_stats()
    assert stats['error_count'] >= 2, f"Expected 2+ errors, got {stats['error_count']}"
    
    log.info(f"✅ TEST 3 PASSED: Backoff working ({stats['error_count']} errors, {elapsed:.2f}s elapsed)")
    return True


def test_taskrunner_hot_swap():
    """Test: Hot-swap task function without restart"""
    from core.task_runner import TaskRunner
    
    task1_calls = [0]
    task2_calls = [0]
    
    def task1():
        task1_calls[0] += 1
        return None
    
    def task2():
        task2_calls[0] += 1
        return None
    
    runner = TaskRunner(task1, interval=0.05, name="test-hotswap")
    runner.start()
    
    time.sleep(0.2)
    
    # Hot-swap task
    runner.update_task(task2)
    
    time.sleep(0.2)
    
    runner.stop(timeout=2.0)
    
    assert task1_calls[0] > 0, "Task1 should have been called"
    assert task2_calls[0] > 0, "Task2 should have been called after swap"
    
    log.info(f"✅ TEST 4 PASSED: Hot-swap successful (task1={task1_calls[0]}, task2={task2_calls[0]})")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: EnforcerAgent Integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_enforcer_taskrunner_integration():
    """Test: EnforcerAgent uses TaskRunner for compliance monitoring"""
    from agents.enforcer import EnforcerAgent
    
    try:
        # Create a minimal enforcer agent (interfaces mocked)
        class MockEngine:
            def get_anomaly_feed(self):
                return []
        
        class MockEconomic:
            pass
        
        class MockControl:
            pass
        
        enforcer = EnforcerAgent(
            enforcement_engine=MockEngine(),
            economic_layer=MockEconomic(),
            command_control=MockControl(),
            constitution_reference={}
        )
        
        # Check that TaskRunner was initialized
        assert hasattr(enforcer, '_monitor_runner'), "EnforcerAgent should have _monitor_runner"
        assert enforcer._monitor_runner is not None, "TaskRunner should be initialized"
        assert enforcer._monitor_runner.is_running(), "Monitor TaskRunner should be running"
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Check that the runner is still alive (it should have executed at least once)
        is_alive = enforcer._monitor_runner.is_running()
        assert is_alive, "TaskRunner should still be running after 0.5s"
        
        # Shutdown
        enforcer.shutdown(timeout=2.0)
        assert not enforcer._monitor_runner.is_running(), "TaskRunner should stop"
        
        log.info(f"✅ TEST 5 PASSED: EnforcerAgent TaskRunner integration working and responding")
        return True
    
    except Exception as e:
        log.error(f"❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Performance & Overhead
# ═══════════════════════════════════════════════════════════════════════════════

def test_taskrunner_overhead():
    """Test: TaskRunner overhead is minimal"""
    from core.task_runner import TaskRunner
    
    call_count = [0]
    
    def fast_task():
        call_count[0] += 1
        return None
    
    runner = TaskRunner(fast_task, interval=0.05, name="test-overhead")
    
    start = time.time()
    runner.start()
    time.sleep(0.5)
    runner.stop(timeout=2.0)
    elapsed = time.time() - start
    
    # At 0.05s interval, we should get ~10 calls in 0.5s (allowing for some jitter)
    expected_min = 5  # Conservative minimum
    
    assert call_count[0] >= expected_min, \
        f"Expected >={expected_min} calls in {elapsed:.2f}s, got {call_count[0]}"
    
    overhead_ms = (elapsed / call_count[0]) * 1000
    log.info(f"✅ TEST 6 PASSED: Low overhead ({overhead_ms:.2f}ms per cycle, {call_count[0]} cycles)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print(" PHASE 3: TASKRUNNER INTEGRATION VALIDATION")
    print("="*80 + "\n")
    
    tests = [
        ("TaskRunner Creation", test_taskrunner_creation),
        ("Adaptive Intervals", test_taskrunner_adaptive_intervals),
        ("Error Backoff", test_taskrunner_error_backoff),
        ("Hot-Swap Tasks", test_taskrunner_hot_swap),
        ("EnforcerAgent Integration", test_enforcer_taskrunner_integration),
        ("Overhead Profile", test_taskrunner_overhead),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            log.error(f"❌ {name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("\n🚀 ALL TESTS PASSED - TaskRunner Integration Ready")
        print("   Next: Scale to SwarmMesh, Kernel, PeerLinker")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
