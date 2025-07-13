#!/usr/bin/env python3
"""
Tektra AI Assistant - Performance Benchmark Tests

Comprehensive performance benchmarking and load testing to validate
system performance under various conditions and loads.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with pytest,pytest-asyncio,pytest-benchmark,psutil,aiohttp python -m pytest test_performance_benchmarks.py -v
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest>=7.4.0",
#     "pytest-asyncio>=0.21.0",
#     "pytest-benchmark>=4.0.0",
#     "psutil>=5.9.0",
#     "aiohttp>=3.8.0",
#     "numpy>=1.24.0",
#     "loguru>=0.7.0",
# ]
# ///

import asyncio
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
import pytest
import psutil
import numpy as np

from loguru import logger

# Import Tektra components
from tektra.core.tektra_system import TektraSystem, TektraSystemConfig
from tektra.core.deployment_manager import DeploymentManager, DeploymentMetrics
from tektra.security.context import SecurityContext, SecurityLevel
from tektra.config.production_config import create_development_config, create_production_config
from tektra.performance.cache_manager import CacheManager, CacheLevel
from tektra.performance.task_scheduler import TaskScheduler, Priority
from tektra.performance.memory_manager import MemoryManager
from tektra.performance.performance_monitor import PerformanceMonitor


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    test_name: str
    duration_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    percentiles: Dict[str, float]
    metadata: Dict[str, Any]


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_monitor = psutil.Process()
    
    def record_result(self, result: BenchmarkResult):
        """Record a benchmark result."""
        self.results.append(result)
        logger.info(f"📊 {result.test_name}: {result.duration_ms:.2f}ms, "
                   f"{result.throughput_ops_per_sec:.2f} ops/sec, "
                   f"{result.success_rate:.1%} success")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics."""
        if not self.results:
            return {}
        
        return {
            "total_tests": len(self.results),
            "average_duration_ms": statistics.mean(r.duration_ms for r in self.results),
            "total_throughput": sum(r.throughput_ops_per_sec for r in self.results),
            "average_success_rate": statistics.mean(r.success_rate for r in self.results),
            "total_errors": sum(r.error_count for r in self.results),
            "peak_memory_mb": max(r.memory_usage_mb for r in self.results),
            "peak_cpu_percent": max(r.cpu_usage_percent for r in self.results)
        }


class TestSystemStartupPerformance:
    """Test system startup and initialization performance."""
    
    @pytest.fixture
    def benchmark_suite(self):
        return PerformanceBenchmarkSuite()
    
    async def test_cold_start_performance(self, benchmark_suite):
        """Test cold start performance from scratch."""
        logger.info("🧪 Testing cold start performance")
        
        config = create_development_config()
        tektra_config = config.to_tektra_config()
        
        # Measure cold start time
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        system = TektraSystem(tektra_config)
        init_success = await system.initialize()
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        assert init_success == True, "System should initialize successfully"
        
        duration_ms = (end_time - start_time) * 1000
        memory_usage = memory_after - memory_before
        
        result = BenchmarkResult(
            test_name="cold_start",
            duration_ms=duration_ms,
            throughput_ops_per_sec=1000 / duration_ms if duration_ms > 0 else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=0,  # Not measured for startup
            success_rate=1.0,
            error_count=0,
            percentiles={},
            metadata={"components_initialized": len(system.components)}
        )
        
        benchmark_suite.record_result(result)
        
        # Performance thresholds
        assert duration_ms < 30000, f"Cold start should complete within 30s, took {duration_ms/1000:.2f}s"
        assert memory_usage < 500, f"Memory usage should be < 500MB, used {memory_usage:.1f}MB"
        
        await system.shutdown()
        logger.info("✅ Cold start performance test passed")
    
    async def test_component_initialization_performance(self, benchmark_suite):
        """Test individual component initialization performance."""
        logger.info("🧪 Testing component initialization performance")
        
        config = create_development_config()
        tektra_config = config.to_tektra_config()
        
        system = TektraSystem(tektra_config)
        
        # Test individual component initialization times
        component_times = {}
        
        # Security components
        start_time = time.time()
        await system._initialize_security_components()
        component_times['security'] = (time.time() - start_time) * 1000
        
        # Performance components
        start_time = time.time()
        await system._initialize_performance_components()
        component_times['performance'] = (time.time() - start_time) * 1000
        
        # Agent components
        start_time = time.time()
        await system._initialize_agent_components()
        component_times['agents'] = (time.time() - start_time) * 1000
        
        # Record results for each component
        for component, duration_ms in component_times.items():
            result = BenchmarkResult(
                test_name=f"component_init_{component}",
                duration_ms=duration_ms,
                throughput_ops_per_sec=1000 / duration_ms if duration_ms > 0 else 0,
                memory_usage_mb=0,  # Not measured per component
                cpu_usage_percent=0,
                success_rate=1.0,
                error_count=0,
                percentiles={},
                metadata={"component": component}
            )
            benchmark_suite.record_result(result)
            
            # Each component should initialize quickly
            assert duration_ms < 10000, f"{component} component should initialize within 10s"
        
        await system.shutdown()
        logger.info("✅ Component initialization performance test passed")


class TestAgentPerformance:
    """Test agent creation and execution performance."""
    
    @pytest.fixture
    async def tektra_system(self):
        """Create Tektra system for testing."""
        config = create_development_config()
        tektra_config = config.to_tektra_config()
        system = TektraSystem(tektra_config)
        
        if await system.initialize():
            yield system
            await system.shutdown()
        else:
            pytest.fail("Failed to initialize Tektra system")
    
    @pytest.fixture
    def benchmark_suite(self):
        return PerformanceBenchmarkSuite()
    
    async def test_agent_creation_performance(self, tektra_system, benchmark_suite):
        """Test agent creation performance."""
        logger.info("🧪 Testing agent creation performance")
        
        security_context = SecurityContext(
            agent_id="perf_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="perf_session"
        )
        
        # Test single agent creation
        start_time = time.time()
        agent_id = await tektra_system.create_agent(
            agent_name="Performance Test Agent",
            agent_config={"model": "text_completion", "max_tokens": 100},
            security_context=security_context
        )
        duration_ms = (time.time() - start_time) * 1000
        
        assert agent_id is not None, "Agent should be created successfully"
        
        result = BenchmarkResult(
            test_name="single_agent_creation",
            duration_ms=duration_ms,
            throughput_ops_per_sec=1000 / duration_ms if duration_ms > 0 else 0,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=1.0,
            error_count=0,
            percentiles={},
            metadata={"agent_id": agent_id}
        )
        
        benchmark_suite.record_result(result)
        
        # Agent creation should be fast
        assert duration_ms < 5000, f"Agent creation should complete within 5s, took {duration_ms:.2f}ms"
        
        logger.info("✅ Agent creation performance test passed")
    
    async def test_concurrent_agent_creation(self, tektra_system, benchmark_suite):
        """Test concurrent agent creation performance."""
        logger.info("🧪 Testing concurrent agent creation performance")
        
        num_agents = 10
        security_context = SecurityContext(
            agent_id="concurrent_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="concurrent_session"
        )
        
        # Create agents concurrently
        start_time = time.time()
        tasks = []
        
        for i in range(num_agents):
            task = tektra_system.create_agent(
                agent_name=f"Concurrent Agent {i}",
                agent_config={"model": "text_completion", "max_tokens": 50},
                security_context=security_context
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration_ms = (time.time() - start_time) * 1000
        
        # Count successful creations
        successful_agents = [r for r in results if not isinstance(r, Exception)]
        error_count = len(results) - len(successful_agents)
        success_rate = len(successful_agents) / len(results)
        
        result = BenchmarkResult(
            test_name="concurrent_agent_creation",
            duration_ms=duration_ms,
            throughput_ops_per_sec=(len(successful_agents) * 1000) / duration_ms if duration_ms > 0 else 0,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=success_rate,
            error_count=error_count,
            percentiles={},
            metadata={"num_agents": num_agents, "successful": len(successful_agents)}
        )
        
        benchmark_suite.record_result(result)
        
        # Should create most agents successfully
        assert success_rate >= 0.8, f"Should create at least 80% of agents, got {success_rate:.1%}"
        assert duration_ms < 15000, f"Concurrent creation should complete within 15s"
        
        logger.info("✅ Concurrent agent creation performance test passed")
    
    async def test_agent_task_execution_performance(self, tektra_system, benchmark_suite):
        """Test agent task execution performance."""
        logger.info("🧪 Testing agent task execution performance")
        
        security_context = SecurityContext(
            agent_id="task_perf_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="task_perf_session"
        )
        
        # Create agent for testing
        agent_id = await tektra_system.create_agent(
            agent_name="Task Performance Agent",
            agent_config={"model": "text_completion", "max_tokens": 100},
            security_context=security_context
        )
        
        # Test different task complexities
        tasks = [
            ("Simple calculation", "Calculate 2 + 2"),
            ("Text processing", "Generate a list of 5 programming languages"),
            ("Complex reasoning", "Explain the benefits of asynchronous programming")
        ]
        
        for task_name, task_description in tasks:
            start_time = time.time()
            
            result = await tektra_system.execute_agent_task(
                agent_id=agent_id,
                task_description=task_description,
                context={"type": "performance_test"},
                security_context=security_context
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            success = result is not None and isinstance(result, dict)
            
            benchmark_result = BenchmarkResult(
                test_name=f"task_execution_{task_name.replace(' ', '_').lower()}",
                duration_ms=duration_ms,
                throughput_ops_per_sec=1000 / duration_ms if duration_ms > 0 else 0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success_rate=1.0 if success else 0.0,
                error_count=0 if success else 1,
                percentiles={},
                metadata={"task_type": task_name, "complexity": len(task_description)}
            )
            
            benchmark_suite.record_result(benchmark_result)
            
            # Task execution should complete reasonably quickly
            assert duration_ms < 30000, f"{task_name} should complete within 30s"
            assert success, f"{task_name} should execute successfully"
        
        logger.info("✅ Agent task execution performance test passed")


class TestCachePerformance:
    """Test cache system performance."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for testing."""
        from tektra.performance import create_cache_manager
        return create_cache_manager(l1_size_mb=64, l2_size_mb=128)
    
    @pytest.fixture
    def benchmark_suite(self):
        return PerformanceBenchmarkSuite()
    
    def test_cache_write_performance(self, cache_manager, benchmark_suite, benchmark):
        """Test cache write performance."""
        logger.info("🧪 Testing cache write performance")
        
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3, 4, 5]}
        
        def cache_write_operation():
            for i in range(1000):
                cache_manager.set(f"test_key_{i}", test_data, level=CacheLevel.L1)
        
        # Benchmark cache writes
        result = benchmark(cache_write_operation)
        
        duration_ms = result.stats.mean * 1000
        throughput = 1000 / result.stats.mean  # ops per second
        
        benchmark_result = BenchmarkResult(
            test_name="cache_write_performance",
            duration_ms=duration_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=1.0,
            error_count=0,
            percentiles={
                "p50": result.stats.median * 1000,
                "p95": np.percentile([t * 1000 for t in result.stats.data], 95),
                "p99": np.percentile([t * 1000 for t in result.stats.data], 99)
            },
            metadata={"operations": 1000, "data_size": len(str(test_data))}
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Cache writes should be fast
        assert duration_ms < 100, f"1000 cache writes should complete within 100ms"
        assert throughput > 10000, f"Should achieve >10k ops/sec, got {throughput:.0f}"
        
        logger.info("✅ Cache write performance test passed")
    
    def test_cache_read_performance(self, cache_manager, benchmark_suite, benchmark):
        """Test cache read performance."""
        logger.info("🧪 Testing cache read performance")
        
        # Pre-populate cache
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3, 4, 5]}
        for i in range(1000):
            cache_manager.set(f"test_key_{i}", test_data, level=CacheLevel.L1)
        
        def cache_read_operation():
            hits = 0
            for i in range(1000):
                value = cache_manager.get(f"test_key_{i}", level=CacheLevel.L1)
                if value is not None:
                    hits += 1
            return hits
        
        # Benchmark cache reads
        result = benchmark(cache_read_operation)
        
        duration_ms = result.stats.mean * 1000
        throughput = 1000 / result.stats.mean
        
        benchmark_result = BenchmarkResult(
            test_name="cache_read_performance",
            duration_ms=duration_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=1.0,
            error_count=0,
            percentiles={
                "p50": result.stats.median * 1000,
                "p95": np.percentile([t * 1000 for t in result.stats.data], 95),
                "p99": np.percentile([t * 1000 for t in result.stats.data], 99)
            },
            metadata={"operations": 1000, "cache_hits_expected": 1000}
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Cache reads should be very fast
        assert duration_ms < 50, f"1000 cache reads should complete within 50ms"
        assert throughput > 20000, f"Should achieve >20k ops/sec, got {throughput:.0f}"
        
        logger.info("✅ Cache read performance test passed")
    
    def test_cache_hit_ratio_performance(self, cache_manager, benchmark_suite):
        """Test cache hit ratio under load."""
        logger.info("🧪 Testing cache hit ratio performance")
        
        # Pre-populate with hot data
        hot_data = {"frequently_accessed": True, "priority": "high"}
        for i in range(100):
            cache_manager.set(f"hot_key_{i}", hot_data, level=CacheLevel.L1)
        
        # Mix of hot and cold access patterns
        start_time = time.time()
        hits = 0
        misses = 0
        
        # 80% hot data access, 20% cold data access
        for i in range(1000):
            if i % 5 == 0:  # 20% cold access
                value = cache_manager.get(f"cold_key_{i}", level=CacheLevel.L1)
            else:  # 80% hot access
                value = cache_manager.get(f"hot_key_{i % 100}", level=CacheLevel.L1)
            
            if value is not None:
                hits += 1
            else:
                misses += 1
        
        duration_ms = (time.time() - start_time) * 1000
        hit_ratio = hits / (hits + misses)
        
        benchmark_result = BenchmarkResult(
            test_name="cache_hit_ratio_performance",
            duration_ms=duration_ms,
            throughput_ops_per_sec=1000 * 1000 / duration_ms if duration_ms > 0 else 0,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=hit_ratio,
            error_count=misses,
            percentiles={},
            metadata={"hits": hits, "misses": misses, "hit_ratio": hit_ratio}
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Should have good hit ratio for hot data
        assert hit_ratio >= 0.75, f"Hit ratio should be >=75%, got {hit_ratio:.1%}"
        
        logger.info("✅ Cache hit ratio performance test passed")


class TestMemoryPerformance:
    """Test memory management performance."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create memory manager for testing."""
        from tektra.performance import create_memory_manager
        return create_memory_manager(max_memory_mb=1024, enable_pools=True)
    
    @pytest.fixture
    def benchmark_suite(self):
        return PerformanceBenchmarkSuite()
    
    def test_memory_allocation_performance(self, memory_manager, benchmark_suite, benchmark):
        """Test memory allocation performance."""
        logger.info("🧪 Testing memory allocation performance")
        
        def memory_allocation_operation():
            allocations = []
            for i in range(100):
                # Allocate 1MB chunks
                data = memory_manager.allocate(1024 * 1024, f"test_allocation_{i}")
                allocations.append(data)
            
            # Clean up
            for allocation in allocations:
                memory_manager.deallocate(allocation)
        
        result = benchmark(memory_allocation_operation)
        
        duration_ms = result.stats.mean * 1000
        throughput = 100 / result.stats.mean  # allocations per second
        
        benchmark_result = BenchmarkResult(
            test_name="memory_allocation_performance",
            duration_ms=duration_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=1.0,
            error_count=0,
            percentiles={
                "p50": result.stats.median * 1000,
                "p95": np.percentile([t * 1000 for t in result.stats.data], 95)
            },
            metadata={"allocations": 100, "size_mb": 1}
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Memory allocation should be efficient
        assert duration_ms < 1000, f"100 1MB allocations should complete within 1s"
        
        logger.info("✅ Memory allocation performance test passed")
    
    def test_memory_fragmentation_resistance(self, memory_manager, benchmark_suite):
        """Test resistance to memory fragmentation."""
        logger.info("🧪 Testing memory fragmentation resistance")
        
        start_time = time.time()
        
        # Create fragmentation pattern
        allocations = []
        
        # Allocate many small chunks
        for i in range(1000):
            size = 1024 * (i % 10 + 1)  # 1KB to 10KB
            allocation = memory_manager.allocate(size, f"frag_test_{i}")
            allocations.append(allocation)
        
        # Free every other allocation to create fragmentation
        for i in range(0, len(allocations), 2):
            memory_manager.deallocate(allocations[i])
        
        # Try to allocate larger chunks
        large_allocations = []
        successful_large = 0
        
        for i in range(100):
            try:
                allocation = memory_manager.allocate(50 * 1024, f"large_test_{i}")  # 50KB
                large_allocations.append(allocation)
                successful_large += 1
            except Exception:
                pass
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Clean up
        for allocation in allocations[1::2]:  # Remaining small allocations
            try:
                memory_manager.deallocate(allocation)
            except:
                pass
        
        for allocation in large_allocations:
            try:
                memory_manager.deallocate(allocation)
            except:
                pass
        
        success_rate = successful_large / 100
        
        benchmark_result = BenchmarkResult(
            test_name="memory_fragmentation_resistance",
            duration_ms=duration_ms,
            throughput_ops_per_sec=0,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=success_rate,
            error_count=100 - successful_large,
            percentiles={},
            metadata={
                "small_allocations": 1000,
                "large_allocations_attempted": 100,
                "large_allocations_successful": successful_large
            }
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Should handle fragmentation reasonably well
        assert success_rate >= 0.7, f"Should successfully allocate >=70% of large chunks after fragmentation"
        
        logger.info("✅ Memory fragmentation resistance test passed")


class TestConcurrencyPerformance:
    """Test concurrency and threading performance."""
    
    @pytest.fixture
    async def tektra_system(self):
        """Create Tektra system for testing."""
        config = create_development_config()
        tektra_config = config.to_tektra_config()
        system = TektraSystem(tektra_config)
        
        if await system.initialize():
            yield system
            await system.shutdown()
        else:
            pytest.fail("Failed to initialize Tektra system")
    
    @pytest.fixture
    def benchmark_suite(self):
        return PerformanceBenchmarkSuite()
    
    async def test_concurrent_operations_performance(self, tektra_system, benchmark_suite):
        """Test performance under concurrent operations."""
        logger.info("🧪 Testing concurrent operations performance")
        
        security_context = SecurityContext(
            agent_id="concurrent_ops_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="concurrent_ops_session"
        )
        
        # Create agent for testing
        agent_id = await tektra_system.create_agent(
            agent_name="Concurrent Operations Agent",
            agent_config={"model": "text_completion", "max_tokens": 50},
            security_context=security_context
        )
        
        # Test concurrent task execution
        num_concurrent_tasks = 20
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for i in range(num_concurrent_tasks):
            task = tektra_system.execute_agent_task(
                agent_id=agent_id,
                task_description=f"Task {i}: Generate a simple response",
                context={"type": "concurrent_test", "task_id": i},
                security_context=security_context
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Analyze results
        successful_tasks = [r for r in results if not isinstance(r, Exception)]
        error_count = len(results) - len(successful_tasks)
        success_rate = len(successful_tasks) / len(results)
        throughput = len(successful_tasks) * 1000 / duration_ms if duration_ms > 0 else 0
        
        benchmark_result = BenchmarkResult(
            test_name="concurrent_operations_performance",
            duration_ms=duration_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=success_rate,
            error_count=error_count,
            percentiles={},
            metadata={
                "concurrent_tasks": num_concurrent_tasks,
                "successful_tasks": len(successful_tasks)
            }
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Should handle concurrent operations well
        assert success_rate >= 0.8, f"Should complete >=80% of concurrent tasks, got {success_rate:.1%}"
        assert duration_ms < 60000, f"Concurrent operations should complete within 60s"
        
        logger.info("✅ Concurrent operations performance test passed")
    
    async def test_thread_pool_performance(self, benchmark_suite):
        """Test thread pool performance for CPU-bound tasks."""
        logger.info("🧪 Testing thread pool performance")
        
        def cpu_intensive_task(n: int) -> int:
            """Simulate CPU-intensive work."""
            result = 0
            for i in range(n):
                result += i * i
            return result
        
        # Test with different numbers of workers
        worker_counts = [1, 2, 4, 8]
        task_count = 100
        task_size = 10000
        
        for num_workers in worker_counts:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(cpu_intensive_task, task_size) for _ in range(task_count)]
                results = [future.result() for future in as_completed(futures)]
            
            duration_ms = (time.time() - start_time) * 1000
            throughput = task_count * 1000 / duration_ms if duration_ms > 0 else 0
            
            benchmark_result = BenchmarkResult(
                test_name=f"thread_pool_performance_{num_workers}_workers",
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success_rate=1.0,
                error_count=0,
                percentiles={},
                metadata={
                    "num_workers": num_workers,
                    "task_count": task_count,
                    "task_size": task_size
                }
            )
            
            benchmark_suite.record_result(benchmark_result)
            
            # More workers should generally improve performance (up to CPU limit)
            if num_workers <= psutil.cpu_count():
                assert len(results) == task_count, "All tasks should complete"
        
        logger.info("✅ Thread pool performance test passed")


class TestLoadTesting:
    """Comprehensive load testing scenarios."""
    
    @pytest.fixture
    async def tektra_system(self):
        """Create Tektra system for testing."""
        config = create_development_config()
        tektra_config = config.to_tektra_config()
        system = TektraSystem(tektra_config)
        
        if await system.initialize():
            yield system
            await system.shutdown()
        else:
            pytest.fail("Failed to initialize Tektra system")
    
    @pytest.fixture
    def benchmark_suite(self):
        return PerformanceBenchmarkSuite()
    
    async def test_sustained_load_performance(self, tektra_system, benchmark_suite):
        """Test performance under sustained load."""
        logger.info("🧪 Testing sustained load performance")
        
        security_context = SecurityContext(
            agent_id="sustained_load_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="sustained_load_session"
        )
        
        # Create agent
        agent_id = await tektra_system.create_agent(
            agent_name="Sustained Load Agent",
            agent_config={"model": "text_completion", "max_tokens": 100},
            security_context=security_context
        )
        
        # Run sustained load for a period
        duration_seconds = 30
        target_rps = 5  # 5 requests per second
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        completed_tasks = 0
        failed_tasks = 0
        response_times = []
        
        while time.time() < end_time:
            batch_start = time.time()
            
            # Execute a batch of tasks
            task_start = time.time()
            try:
                result = await tektra_system.execute_agent_task(
                    agent_id=agent_id,
                    task_description=f"Sustained load task {completed_tasks + 1}",
                    context={"type": "sustained_load", "timestamp": time.time()},
                    security_context=security_context
                )
                
                task_duration = time.time() - task_start
                response_times.append(task_duration)
                
                if result is not None:
                    completed_tasks += 1
                else:
                    failed_tasks += 1
                    
            except Exception as e:
                failed_tasks += 1
                logger.warning(f"Task failed during sustained load: {e}")
            
            # Rate limiting
            batch_duration = time.time() - batch_start
            sleep_time = (1.0 / target_rps) - batch_duration
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        total_duration_ms = (time.time() - start_time) * 1000
        total_tasks = completed_tasks + failed_tasks
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        actual_rps = completed_tasks / duration_seconds
        
        # Calculate percentiles
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50) * 1000,
                "p90": np.percentile(response_times, 90) * 1000,
                "p95": np.percentile(response_times, 95) * 1000,
                "p99": np.percentile(response_times, 99) * 1000
            }
        else:
            percentiles = {}
        
        benchmark_result = BenchmarkResult(
            test_name="sustained_load_performance",
            duration_ms=total_duration_ms,
            throughput_ops_per_sec=actual_rps,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=success_rate,
            error_count=failed_tasks,
            percentiles=percentiles,
            metadata={
                "target_rps": target_rps,
                "actual_rps": actual_rps,
                "duration_seconds": duration_seconds,
                "completed_tasks": completed_tasks
            }
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Should maintain good performance under sustained load
        assert success_rate >= 0.95, f"Should maintain >=95% success rate, got {success_rate:.1%}"
        assert actual_rps >= target_rps * 0.8, f"Should achieve >=80% of target RPS"
        
        logger.info("✅ Sustained load performance test passed")
    
    async def test_burst_load_performance(self, tektra_system, benchmark_suite):
        """Test performance under burst load conditions."""
        logger.info("🧪 Testing burst load performance")
        
        security_context = SecurityContext(
            agent_id="burst_load_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="burst_load_session"
        )
        
        # Create agent
        agent_id = await tektra_system.create_agent(
            agent_name="Burst Load Agent",
            agent_config={"model": "text_completion", "max_tokens": 50},
            security_context=security_context
        )
        
        # Generate burst of concurrent requests
        burst_size = 50
        
        start_time = time.time()
        
        # Create all tasks at once (burst)
        tasks = []
        for i in range(burst_size):
            task = tektra_system.execute_agent_task(
                agent_id=agent_id,
                task_description=f"Burst task {i}",
                context={"type": "burst_load", "task_id": i},
                security_context=security_context
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Analyze results
        successful_tasks = [r for r in results if not isinstance(r, Exception)]
        failed_tasks = len(results) - len(successful_tasks)
        success_rate = len(successful_tasks) / len(results)
        throughput = len(successful_tasks) * 1000 / total_duration_ms if total_duration_ms > 0 else 0
        
        benchmark_result = BenchmarkResult(
            test_name="burst_load_performance",
            duration_ms=total_duration_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            success_rate=success_rate,
            error_count=failed_tasks,
            percentiles={},
            metadata={
                "burst_size": burst_size,
                "successful_tasks": len(successful_tasks)
            }
        )
        
        benchmark_suite.record_result(benchmark_result)
        
        # Should handle burst load reasonably well
        assert success_rate >= 0.7, f"Should handle >=70% of burst load, got {success_rate:.1%}"
        assert total_duration_ms < 120000, f"Burst should complete within 2 minutes"
        
        logger.info("✅ Burst load performance test passed")


# Test execution and reporting
def pytest_configure(config):
    """Configure pytest for performance benchmarks."""
    logger.info("📊 Configuring Performance Benchmark Tests")


def pytest_sessionstart(session):
    """Start of test session."""
    logger.info("⚡ Starting Performance Benchmark Tests")


def pytest_sessionfinish(session, exitstatus):
    """End of test session."""
    if exitstatus == 0:
        logger.info("✅ All performance benchmark tests passed!")
    else:
        logger.error(f"❌ Some benchmark tests failed with exit status: {exitstatus}")


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    logger.info("📊 Running Performance Benchmark Tests")
    
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--benchmark-skip"  # Skip benchmark unless explicitly requested
    ])
    
    sys.exit(result.returncode)