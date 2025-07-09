#!/usr/bin/env python3
"""
Performance tests for the memory system.
"""

import pytest
import sys
import time
import asyncio
import tempfile
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tektra.memory.memory_manager import TektraMemoryManager
from tektra.memory.memory_types import MemoryType, MemoryEntry, MemoryContext
from tektra.memory.memory_config import MemoryConfig


class TestMemoryPerformance:
    """Performance tests for memory system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def performance_config(self, temp_dir):
        """Create a performance-optimized memory configuration."""
        return MemoryConfig(
            storage_path=str(temp_dir),
            database_name="performance_test.db",
            max_memories_per_user=10000,
            use_memos=False,
            enable_semantic_search=False,
            cache_size=1000,
            batch_size=100
        )
    
    @pytest.fixture
    def memory_manager(self, performance_config):
        """Create a memory manager for performance testing."""
        return TektraMemoryManager(performance_config)
    
    def create_test_entries(self, count: int, base_user_id: str = "perf_user") -> List[MemoryEntry]:
        """Create test memory entries for performance testing."""
        entries = []
        
        for i in range(count):
            entry = MemoryEntry(
                id=f"perf_entry_{i:06d}",
                content=f"This is performance test entry number {i}. It contains some text for testing purposes.",
                type=MemoryType.CONVERSATION if i % 2 == 0 else MemoryType.LEARNED_FACT,
                metadata={
                    "batch": i // 100,
                    "index": i,
                    "category": "performance_test"
                },
                importance=min(1.0, 0.1 + (i % 10) * 0.1),
                timestamp=datetime.now() - timedelta(hours=i % 24),
                user_id=f"{base_user_id}_{i % 5}",  # Distribute across 5 users
                session_id=f"session_{i % 10}"  # Distribute across 10 sessions
            )
            entries.append(entry)
        
        return entries
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function and return execution time."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, (end_time - start_time) * 1000  # Return time in milliseconds
    
    async def async_benchmark_function(self, func, *args, **kwargs):
        """Benchmark an async function and return execution time."""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, (end_time - start_time) * 1000  # Return time in milliseconds
    
    @pytest.mark.asyncio
    async def test_memory_insertion_performance(self, memory_manager):
        """Test memory insertion performance."""
        await memory_manager.initialize()
        
        # Test small batch (100 entries)
        small_entries = self.create_test_entries(100)
        
        # Benchmark individual insertions
        individual_times = []
        for entry in small_entries[:10]:  # Test first 10
            _, exec_time = await self.async_benchmark_function(
                memory_manager.add_memory, entry
            )
            individual_times.append(exec_time)
        
        # Benchmark batch insertion
        batch_entries = self.create_test_entries(100, "batch_user")
        start_time = time.perf_counter()
        
        for entry in batch_entries:
            await memory_manager.add_memory(entry)
        
        end_time = time.perf_counter()
        batch_time = (end_time - start_time) * 1000
        
        # Performance assertions
        avg_individual_time = statistics.mean(individual_times)
        assert avg_individual_time < 50, f"Individual insertion too slow: {avg_individual_time:.2f}ms"
        
        avg_batch_time = batch_time / len(batch_entries)
        assert avg_batch_time < 20, f"Batch insertion too slow: {avg_batch_time:.2f}ms per entry"
        
        print(f"Individual insertion: {avg_individual_time:.2f}ms avg")
        print(f"Batch insertion: {avg_batch_time:.2f}ms avg per entry")
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_retrieval_performance(self, memory_manager):
        """Test memory retrieval performance."""
        await memory_manager.initialize()
        
        # Insert test data
        test_entries = self.create_test_entries(500)
        for entry in test_entries:
            await memory_manager.add_memory(entry)
        
        # Test individual retrieval
        retrieval_times = []
        for i in range(0, 50, 5):  # Test every 5th entry
            entry_id = f"perf_entry_{i:06d}"
            _, exec_time = await self.async_benchmark_function(
                memory_manager.get_memory, entry_id
            )
            retrieval_times.append(exec_time)
        
        # Performance assertions
        avg_retrieval_time = statistics.mean(retrieval_times)
        assert avg_retrieval_time < 10, f"Memory retrieval too slow: {avg_retrieval_time:.2f}ms"
        
        print(f"Memory retrieval: {avg_retrieval_time:.2f}ms avg")
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, memory_manager):
        """Test memory search performance."""
        await memory_manager.initialize()
        
        # Insert test data with searchable content
        test_entries = []
        search_terms = ["python", "javascript", "artificial", "intelligence", "machine", "learning"]
        
        for i in range(1000):
            term = search_terms[i % len(search_terms)]
            entry = MemoryEntry(
                id=f"search_entry_{i:06d}",
                content=f"This is a {term} related entry with additional content for search testing.",
                type=MemoryType.LEARNED_FACT,
                metadata={"search_term": term},
                importance=0.5 + (i % 5) * 0.1,
                timestamp=datetime.now() - timedelta(hours=i % 100),
                user_id=f"search_user_{i % 3}"
            )
            test_entries.append(entry)
        
        # Insert entries
        for entry in test_entries:
            await memory_manager.add_memory(entry)
        
        # Test search performance
        search_times = []
        for term in search_terms:
            search_context = MemoryContext(
                query=term,
                max_results=10,
                user_id="search_user_0"
            )
            _, exec_time = await self.async_benchmark_function(
                memory_manager.search_memories, search_context
            )
            search_times.append(exec_time)
        
        # Performance assertions
        avg_search_time = statistics.mean(search_times)
        assert avg_search_time < 100, f"Search too slow: {avg_search_time:.2f}ms"
        
        print(f"Search performance: {avg_search_time:.2f}ms avg")
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, memory_manager):
        """Test concurrent memory operations performance."""
        await memory_manager.initialize()
        
        # Test concurrent insertions
        async def insert_batch(start_idx: int, count: int):
            entries = self.create_test_entries(count, f"concurrent_user_{start_idx}")
            for i, entry in enumerate(entries):
                entry.id = f"concurrent_entry_{start_idx}_{i:06d}"
                await memory_manager.add_memory(entry)
            return count
        
        # Run concurrent insertions
        start_time = time.perf_counter()
        
        tasks = [
            insert_batch(i * 100, 100) for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        concurrent_time = (end_time - start_time) * 1000
        
        total_inserted = sum(results)
        avg_concurrent_time = concurrent_time / total_inserted
        
        # Performance assertions
        assert avg_concurrent_time < 50, f"Concurrent operations too slow: {avg_concurrent_time:.2f}ms per entry"
        
        print(f"Concurrent insertions: {avg_concurrent_time:.2f}ms avg per entry")
        print(f"Total concurrent time: {concurrent_time:.2f}ms for {total_inserted} entries")
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, memory_manager):
        """Test performance with large datasets."""
        await memory_manager.initialize()
        
        # Insert large dataset
        large_dataset_size = 2000
        large_entries = self.create_test_entries(large_dataset_size, "large_user")
        
        # Benchmark large dataset insertion
        start_time = time.perf_counter()
        
        for entry in large_entries:
            await memory_manager.add_memory(entry)
        
        end_time = time.perf_counter()
        insertion_time = (end_time - start_time) * 1000
        
        # Test search performance on large dataset
        search_context = MemoryContext(
            query="performance test",
            max_results=50,
            user_id="large_user_0"
        )
        
        _, search_time = await self.async_benchmark_function(
            memory_manager.search_memories, search_context
        )
        
        # Test statistics performance
        _, stats_time = await self.async_benchmark_function(
            memory_manager.get_memory_stats
        )
        
        # Performance assertions
        avg_insertion_time = insertion_time / large_dataset_size
        assert avg_insertion_time < 25, f"Large dataset insertion too slow: {avg_insertion_time:.2f}ms per entry"
        assert search_time < 200, f"Large dataset search too slow: {search_time:.2f}ms"
        assert stats_time < 100, f"Statistics calculation too slow: {stats_time:.2f}ms"
        
        print(f"Large dataset insertion: {avg_insertion_time:.2f}ms avg per entry")
        print(f"Large dataset search: {search_time:.2f}ms")
        print(f"Statistics calculation: {stats_time:.2f}ms")
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_cleanup_performance(self, memory_manager):
        """Test memory cleanup performance."""
        await memory_manager.initialize()
        
        # Insert entries with different ages
        old_entries = []
        recent_entries = []
        
        for i in range(1000):
            # Half old entries (30+ days old)
            if i < 500:
                entry = MemoryEntry(
                    id=f"old_entry_{i:06d}",
                    content=f"Old entry {i}",
                    type=MemoryType.CONVERSATION,
                    importance=0.1,
                    timestamp=datetime.now() - timedelta(days=35 + i % 10),
                    user_id="cleanup_user"
                )
                old_entries.append(entry)
            else:
                # Half recent entries
                entry = MemoryEntry(
                    id=f"recent_entry_{i:06d}",
                    content=f"Recent entry {i}",
                    type=MemoryType.CONVERSATION,
                    importance=0.8,
                    timestamp=datetime.now() - timedelta(days=i % 5),
                    user_id="cleanup_user"
                )
                recent_entries.append(entry)
        
        # Insert all entries
        for entry in old_entries + recent_entries:
            await memory_manager.add_memory(entry)
        
        # Benchmark cleanup operation
        _, cleanup_time = await self.async_benchmark_function(
            memory_manager.cleanup_old_memories, 30  # Delete entries older than 30 days
        )
        
        # Performance assertions
        assert cleanup_time < 1000, f"Cleanup too slow: {cleanup_time:.2f}ms"
        
        print(f"Cleanup performance: {cleanup_time:.2f}ms")
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, memory_manager):
        """Test memory usage scaling with dataset size."""
        await memory_manager.initialize()
        
        # Test different dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]
        performance_metrics = []
        
        for size in dataset_sizes:
            # Clear previous data
            await memory_manager.cleanup_old_memories(0)  # Delete all
            
            # Insert dataset
            entries = self.create_test_entries(size, f"scale_user_{size}")
            
            start_time = time.perf_counter()
            for entry in entries:
                await memory_manager.add_memory(entry)
            end_time = time.perf_counter()
            
            insertion_time = (end_time - start_time) * 1000
            
            # Test search performance
            search_context = MemoryContext(
                query="test entry",
                max_results=10,
                user_id=f"scale_user_{size}_0"
            )
            
            _, search_time = await self.async_benchmark_function(
                memory_manager.search_memories, search_context
            )
            
            performance_metrics.append({
                'size': size,
                'insertion_time': insertion_time,
                'avg_insertion_time': insertion_time / size,
                'search_time': search_time
            })
        
        # Analyze scaling
        for i, metrics in enumerate(performance_metrics):
            print(f"Dataset size {metrics['size']}: "
                  f"Insertion {metrics['avg_insertion_time']:.2f}ms/entry, "
                  f"Search {metrics['search_time']:.2f}ms")
            
            # Performance should scale reasonably with size
            if i > 0:
                prev_metrics = performance_metrics[i-1]
                size_ratio = metrics['size'] / prev_metrics['size']
                time_ratio = metrics['avg_insertion_time'] / prev_metrics['avg_insertion_time']
                
                # Time should not increase more than linearly (allowing for improvements due to caching)
                assert time_ratio < size_ratio * 2.0, f"Performance degrades too quickly with size: {time_ratio:.2f}x for {size_ratio:.2f}x size"
        
        await memory_manager.cleanup()


class TestMemoryBenchmarks:
    """Benchmark tests for memory system components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def benchmark_config(self, temp_dir):
        """Create a benchmark memory configuration."""
        return MemoryConfig(
            storage_path=str(temp_dir),
            database_name="benchmark_test.db",
            max_memories_per_user=50000,
            use_memos=False,
            enable_semantic_search=False,
            cache_size=5000,
            batch_size=500
        )
    
    @pytest.fixture
    def memory_manager(self, benchmark_config):
        """Create a memory manager for benchmarking."""
        return TektraMemoryManager(benchmark_config)
    
    def benchmark_memory_operations(self, name: str, operation_func, iterations: int = 100):
        """Benchmark memory operations."""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            operation_func()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        p95_time = sorted(times)[int(0.95 * len(times))]
        
        print(f"{name} Benchmark:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Median: {median_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")
        
        return {
            'average': avg_time,
            'median': median_time,
            'p95': p95_time
        }
    
    def test_memory_entry_creation_benchmark(self):
        """Benchmark memory entry creation."""
        def create_entry():
            return MemoryEntry(
                content="Benchmark test entry with some content",
                type=MemoryType.CONVERSATION,
                metadata={"test": "benchmark"},
                importance=0.5,
                timestamp=datetime.now(),
                user_id="benchmark_user"
            )
        
        results = self.benchmark_memory_operations("Memory Entry Creation", create_entry, 1000)
        
        # Should be very fast
        assert results['average'] < 1.0, f"Entry creation too slow: {results['average']:.2f}ms"
    
    def test_memory_entry_serialization_benchmark(self):
        """Benchmark memory entry serialization."""
        entry = MemoryEntry(
            content="Benchmark test entry with some content for serialization testing",
            type=MemoryType.LEARNED_FACT,
            metadata={"test": "benchmark", "data": list(range(100))},
            importance=0.8,
            timestamp=datetime.now(),
            user_id="benchmark_user"
        )
        
        def serialize_entry():
            return entry.to_dict()
        
        def deserialize_entry():
            data = entry.to_dict()
            return MemoryEntry.from_dict(data)
        
        serialize_results = self.benchmark_memory_operations("Serialization", serialize_entry, 1000)
        deserialize_results = self.benchmark_memory_operations("Deserialization", deserialize_entry, 1000)
        
        # Should be fast
        assert serialize_results['average'] < 2.0, f"Serialization too slow: {serialize_results['average']:.2f}ms"
        assert deserialize_results['average'] < 3.0, f"Deserialization too slow: {deserialize_results['average']:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_database_operations_benchmark(self, memory_manager):
        """Benchmark database operations."""
        await memory_manager.initialize()
        
        # Prepare test data
        test_entries = []
        for i in range(1000):
            entry = MemoryEntry(
                id=f"db_bench_{i:06d}",
                content=f"Database benchmark entry {i}",
                type=MemoryType.CONVERSATION,
                importance=0.5,
                timestamp=datetime.now(),
                user_id="db_user"
            )
            test_entries.append(entry)
        
        # Insert entries first
        for entry in test_entries:
            await memory_manager.add_memory(entry)
        
        # Benchmark retrievals
        async def retrieve_entry():
            entry_id = f"db_bench_{(time.time_ns() % 1000):06d}"
            return await memory_manager.get_memory(entry_id)
        
        # Benchmark searches
        async def search_entries():
            context = MemoryContext(
                query="benchmark entry",
                max_results=10,
                user_id="db_user"
            )
            return await memory_manager.search_memories(context)
        
        # Run benchmarks
        retrieval_times = []
        search_times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            await retrieve_entry()
            end_time = time.perf_counter()
            retrieval_times.append((end_time - start_time) * 1000)
            
            start_time = time.perf_counter()
            await search_entries()
            end_time = time.perf_counter()
            search_times.append((end_time - start_time) * 1000)
        
        avg_retrieval = statistics.mean(retrieval_times)
        avg_search = statistics.mean(search_times)
        
        print(f"Database Retrieval: {avg_retrieval:.2f}ms avg")
        print(f"Database Search: {avg_search:.2f}ms avg")
        
        # Performance assertions
        assert avg_retrieval < 10, f"Database retrieval too slow: {avg_retrieval:.2f}ms"
        assert avg_search < 50, f"Database search too slow: {avg_search:.2f}ms"
        
        await memory_manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements