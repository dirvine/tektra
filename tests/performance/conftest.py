#!/usr/bin/env python3
"""
Performance Testing Configuration

Provides fixtures and utilities for performance testing across the Tektra system.
Includes resource monitoring, benchmarking utilities, and performance regression detection.
"""

import asyncio
import gc
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class PerformanceMonitor:
    """Monitor system resources during test execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_metrics = {}
        self.end_metrics = {}
        self.peak_metrics = {}
        
    def start_monitoring(self):
        """Start performance monitoring."""
        # Force garbage collection for clean baseline
        gc.collect()
        
        # Record baseline metrics
        self.start_metrics = {
            'memory_rss': self.process.memory_info().rss,
            'memory_vms': self.process.memory_info().vms,
            'cpu_percent': self.process.cpu_percent(),
            'open_files': len(self.process.open_files()),
            'connections': len(self.process.net_connections()),
            'threads': self.process.num_threads(),
            'timestamp': time.time()
        }
        
        # Initialize peak tracking
        self.peak_metrics = self.start_metrics.copy()
        
    def update_peaks(self):
        """Update peak resource usage."""
        current = {
            'memory_rss': self.process.memory_info().rss,
            'memory_vms': self.process.memory_info().vms,
            'cpu_percent': self.process.cpu_percent(),
            'open_files': len(self.process.open_files()),
            'connections': len(self.process.net_connections()),
            'threads': self.process.num_threads(),
        }
        
        for key, value in current.items():
            if key in self.peak_metrics and value > self.peak_metrics[key]:
                self.peak_metrics[key] = value
                
    def stop_monitoring(self):
        """Stop monitoring and record final metrics."""
        self.end_metrics = {
            'memory_rss': self.process.memory_info().rss,
            'memory_vms': self.process.memory_info().vms,
            'cpu_percent': self.process.cpu_percent(),
            'open_files': len(self.process.open_files()),
            'connections': len(self.process.net_connections()),
            'threads': self.process.num_threads(),
            'timestamp': time.time()
        }
        
    def get_memory_delta_mb(self) -> float:
        """Get memory usage delta in MB."""
        delta_bytes = self.end_metrics['memory_rss'] - self.start_metrics['memory_rss']
        return delta_bytes / (1024 * 1024)
        
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_metrics['memory_rss'] / (1024 * 1024)
        
    def get_duration(self) -> float:
        """Get test duration in seconds."""
        return self.end_metrics['timestamp'] - self.start_metrics['timestamp']
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'duration_seconds': self.get_duration(),
            'memory_delta_mb': self.get_memory_delta_mb(),
            'peak_memory_mb': self.get_peak_memory_mb(),
            'start_memory_mb': self.start_metrics['memory_rss'] / (1024 * 1024),
            'end_memory_mb': self.end_metrics['memory_rss'] / (1024 * 1024),
            'file_descriptors_delta': self.end_metrics['open_files'] - self.start_metrics['open_files'],
            'connections_delta': self.end_metrics['connections'] - self.start_metrics['connections'],
            'threads_delta': self.end_metrics['threads'] - self.start_metrics['threads'],
            'peak_cpu_percent': max(self.start_metrics['cpu_percent'], self.end_metrics['cpu_percent'])
        }


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.monitor = PerformanceMonitor()
        self.results = []
        
    def __enter__(self):
        """Start performance monitoring."""
        self.monitor.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and store results."""
        self.monitor.stop_monitoring()
        summary = self.monitor.get_summary()
        summary['test_name'] = self.name
        summary['success'] = exc_type is None
        self.results.append(summary)
        
    async def measure_async_operation(self, operation, *args, **kwargs):
        """Measure an async operation with detailed timing."""
        start_time = time.perf_counter()
        self.monitor.update_peaks()
        
        try:
            result = await operation(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            
        end_time = time.perf_counter()
        self.monitor.update_peaks()
        
        return {
            'result': result,
            'duration': end_time - start_time,
            'success': success,
            'peak_memory_mb': self.monitor.get_peak_memory_mb()
        }
        
    def measure_sync_operation(self, operation, *args, **kwargs):
        """Measure a sync operation with detailed timing."""
        start_time = time.perf_counter()
        self.monitor.update_peaks()
        
        try:
            result = operation(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            
        end_time = time.perf_counter()
        self.monitor.update_peaks()
        
        return {
            'result': result,
            'duration': end_time - start_time,
            'success': success,
            'peak_memory_mb': self.monitor.get_peak_memory_mb()
        }


@pytest.fixture
def performance_monitor():
    """Provide a performance monitor for tests."""
    return PerformanceMonitor()


@pytest.fixture
def performance_benchmark():
    """Provide a performance benchmark utility."""
    def _benchmark(name: str):
        return PerformanceBenchmark(name)
    return _benchmark


@pytest.fixture
def temp_dir_performance():
    """Provide a temporary directory optimized for performance testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Ensure directory is on fast storage if available
        path = Path(temp_dir)
        yield path


@pytest.fixture
def memory_config_performance(temp_dir_performance):
    """Create optimized memory configuration for performance testing."""
    from tektra.memory.memory_config import MemoryConfig
    
    return MemoryConfig(
        storage_path=str(temp_dir_performance),
        database_name="performance_test.db",
        max_memories_per_user=10000,  # Large enough for performance testing
        use_memos=False,  # Disable for consistent performance
        enable_semantic_search=False,  # Disable for speed
        batch_size=100,  # Optimize for performance
        cache_size=1000,  # Enable caching
    )


@pytest.fixture
async def memory_manager_performance(memory_config_performance):
    """Create memory manager optimized for performance testing."""
    from tektra.memory.memory_manager import TektraMemoryManager
    
    manager = TektraMemoryManager(memory_config_performance)
    await manager.initialize()
    
    yield manager
    
    await manager.cleanup()


# Performance test markers
def pytest_configure(config):
    """Configure performance test markers."""
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "stress: marks tests as stress tests")
    config.addinivalue_line("markers", "integration_perf: marks integration performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for performance tests."""
    # Add performance marker to all tests in performance directory
    for item in items:
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


class PerformanceAssertion:
    """Performance assertion utilities."""
    
    @staticmethod
    def assert_memory_usage(actual_mb: float, expected_max_mb: float, operation: str):
        """Assert memory usage is within expected bounds."""
        assert actual_mb <= expected_max_mb, (
            f"{operation} used {actual_mb:.2f}MB memory, expected <= {expected_max_mb}MB"
        )
        
    @staticmethod
    def assert_duration(actual_seconds: float, expected_max_seconds: float, operation: str):
        """Assert operation duration is within expected bounds."""
        assert actual_seconds <= expected_max_seconds, (
            f"{operation} took {actual_seconds:.3f}s, expected <= {expected_max_seconds}s"
        )
        
    @staticmethod
    def assert_throughput(operations_count: int, duration_seconds: float, min_ops_per_second: float, operation: str):
        """Assert throughput meets minimum requirements."""
        actual_ops_per_second = operations_count / duration_seconds
        assert actual_ops_per_second >= min_ops_per_second, (
            f"{operation} achieved {actual_ops_per_second:.2f} ops/sec, expected >= {min_ops_per_second} ops/sec"
        )


@pytest.fixture
def perf_assert():
    """Provide performance assertion utilities."""
    return PerformanceAssertion()


# Global performance tracking
PERFORMANCE_RESULTS = []


def log_performance_result(result: Dict[str, Any]):
    """Log performance result for analysis."""
    PERFORMANCE_RESULTS.append(result)


@pytest.fixture(autouse=True)
def track_performance_results():
    """Automatically track performance results."""
    yield
    # Results are tracked via PerformanceBenchmark context manager


# Performance test configuration
PERFORMANCE_LIMITS = {
    'memory_system': {
        'single_operation_max_duration': 0.1,  # 100ms
        'single_operation_max_memory': 10,  # 10MB
        'bulk_operation_max_duration': 5.0,  # 5 seconds
        'bulk_operation_max_memory': 100,  # 100MB
    },
    'ai_backend': {
        'initialization_max_duration': 120,  # 2 minutes
        'initialization_max_memory': 8192,  # 8GB
        'inference_max_duration': 30,  # 30 seconds
        'inference_max_memory': 4096,  # 4GB
    },
    'integration': {
        'cross_component_max_duration': 1.0,  # 1 second
        'cross_component_max_memory': 50,  # 50MB
    }
}


@pytest.fixture
def performance_limits():
    """Provide performance limits configuration."""
    return PERFORMANCE_LIMITS