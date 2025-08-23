import pytest
from src.performance.performance_profiler import PerformanceProfiler

def test_performance_profiler_initialization():
    profiler = PerformanceProfiler()
    assert profiler is not None
    assert profiler.performance_data == {}

def test_record_performance_data():
    profiler = PerformanceProfiler()
    profiler.record_performance_data('test_task', 0.5)
    assert 'test_task' in profiler.performance_data
    assert profiler.performance_data['test_task'] == 0.5

def test_optimize_performance():
    profiler = PerformanceProfiler()
    profiler.record_performance_data('test_task', 0.5)
    optimized_value = profiler.optimize_performance('test_task')
    assert optimized_value < 0.5  # Assuming optimization reduces the time

def test_monitor_system_performance(mocker):
    profiler = PerformanceProfiler()
    mocker.patch('src.performance.performance_profiler.psutil.cpu_percent', return_value=30.0)
    mocker.patch('src.performance.performance_profiler.psutil.virtual_memory', return_value=mocker.Mock(percent=50.0))
    
    observation = profiler.monitor_system_performance()
    assert observation['cpu_usage'] == 30.0
    assert observation['memory_usage'] == 50.0

def test_persistent_storage():
    profiler = PerformanceProfiler()
    profiler.record_performance_data('test_task', 0.5)
    profiler.save_to_storage()
    
    # Assuming there is a method to load data back
    loaded_data = profiler.load_from_storage()
    assert loaded_data['test_task'] == 0.5