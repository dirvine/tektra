#!/usr/bin/env python3
"""
Memory System Performance Tests

Comprehensive performance testing for the Tektra memory system.
Tests memory operations, search performance, concurrency, and resource usage.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tektra.memory.memory_types import MemoryContext, MemoryEntry, MemoryType


@pytest.mark.performance
@pytest.mark.benchmark
class TestMemorySystemPerformance:
    """Test memory system performance characteristics."""

    async def test_single_memory_operations_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test performance of individual memory operations."""
        
        with performance_benchmark("single_memory_operations") as bench:
            # Test single memory creation
            test_entry = MemoryEntry(
                id="perf_test_001",
                content="Performance test memory entry with substantial content to simulate real usage patterns",
                type=MemoryType.CONVERSATION,
                metadata={"test": "performance", "size": "medium"},
                importance=0.8,
                timestamp=datetime.now(),
                user_id="perf_user",
                session_id="perf_session"
            )
            
            # Measure add operation
            add_result = await bench.measure_async_operation(
                memory_manager_performance.add_memory, test_entry
            )
            
            # Measure get operation
            get_result = await bench.measure_async_operation(
                memory_manager_performance.get_memory, test_entry.id
            )
            
            # Measure search operation
            search_context = MemoryContext(
                user_id="perf_user",
                query="performance test",
                max_results=10
            )
            search_result = await bench.measure_async_operation(
                memory_manager_performance.search_memories, search_context
            )
            
            # Measure delete operation
            delete_result = await bench.measure_async_operation(
                memory_manager_performance.delete_memory, test_entry.id
            )
        
        # Performance assertions
        perf_assert.assert_duration(add_result['duration'], 0.1, "Memory add operation")
        perf_assert.assert_duration(get_result['duration'], 0.05, "Memory get operation")
        perf_assert.assert_duration(search_result['duration'], 0.2, "Memory search operation")
        perf_assert.assert_duration(delete_result['duration'], 0.1, "Memory delete operation")
        
        # Verify operations succeeded
        assert add_result['success'], "Memory add operation failed"
        assert get_result['success'], "Memory get operation failed"
        assert search_result['success'], "Memory search operation failed"
        assert delete_result['success'], "Memory delete operation failed"

    async def test_bulk_memory_operations_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test performance of bulk memory operations."""
        
        # Generate test data
        test_entries = []
        for i in range(1000):
            entry = MemoryEntry(
                id=f"bulk_test_{i:04d}",
                content=f"Bulk test memory entry {i} with varied content length and complexity",
                type=MemoryType.CONVERSATION if i % 2 == 0 else MemoryType.LEARNED_FACT,
                metadata={"bulk_test": True, "index": i, "category": f"cat_{i % 10}"},
                importance=0.1 + (i % 10) * 0.1,
                timestamp=datetime.now() - timedelta(minutes=i),
                user_id=f"user_{i % 5}",
                session_id=f"session_{i % 20}"
            )
            test_entries.append(entry)
        
        with performance_benchmark("bulk_memory_operations") as bench:
            # Measure bulk insert performance
            start_time = time.perf_counter()
            for entry in test_entries:
                await memory_manager_performance.add_memory(entry)
            bulk_insert_duration = time.perf_counter() - start_time
            
            # Measure bulk search performance
            search_contexts = [
                MemoryContext(user_id=f"user_{i}", max_results=50)
                for i in range(5)
            ]
            
            search_start = time.perf_counter()
            search_results = []
            for context in search_contexts:
                result = await memory_manager_performance.search_memories(context)
                search_results.append(result)
            bulk_search_duration = time.perf_counter() - search_start
            
            # Measure statistics performance
            stats_result = await bench.measure_async_operation(
                memory_manager_performance.get_memory_stats
            )
        
        # Performance assertions
        perf_assert.assert_duration(bulk_insert_duration, 10.0, "Bulk insert (1000 entries)")
        perf_assert.assert_throughput(1000, bulk_insert_duration, 100, "Memory insert throughput")
        
        perf_assert.assert_duration(bulk_search_duration, 2.0, "Bulk search (5 users)")
        perf_assert.assert_throughput(5, bulk_search_duration, 2.5, "Memory search throughput")
        
        perf_assert.assert_duration(stats_result['duration'], 1.0, "Memory statistics operation")
        
        # Verify data integrity
        stats = stats_result['result']
        assert stats.total_memories >= 1000, f"Expected >= 1000 memories, got {stats.total_memories}"

    async def test_concurrent_memory_access_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test performance under concurrent access patterns."""
        
        with performance_benchmark("concurrent_memory_access") as bench:
            # Prepare test data
            async def create_test_memories(user_id: str, count: int):
                """Create test memories for a user."""
                for i in range(count):
                    entry = MemoryEntry(
                        id=f"concurrent_{user_id}_{i}",
                        content=f"Concurrent test memory {i} for {user_id}",
                        type=MemoryType.CONVERSATION,
                        user_id=user_id,
                        timestamp=datetime.now()
                    )
                    await memory_manager_performance.add_memory(entry)
            
            async def search_memories(user_id: str, iterations: int):
                """Search memories for a user multiple times."""
                context = MemoryContext(user_id=user_id, max_results=20)
                for _ in range(iterations):
                    await memory_manager_performance.search_memories(context)
            
            # Test concurrent operations
            concurrent_start = time.perf_counter()
            
            # Run concurrent tasks
            tasks = []
            
            # Multiple users creating memories concurrently
            for user_id in [f"concurrent_user_{i}" for i in range(5)]:
                tasks.append(create_test_memories(user_id, 50))
            
            # Multiple users searching concurrently
            for user_id in [f"search_user_{i}" for i in range(3)]:
                tasks.append(search_memories(user_id, 10))
            
            # Execute all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
            concurrent_duration = time.perf_counter() - concurrent_start
        
        # Performance assertions
        perf_assert.assert_duration(concurrent_duration, 15.0, "Concurrent operations")
        
        # Verify system stability
        final_stats = await memory_manager_performance.get_memory_stats()
        assert final_stats.total_memories >= 250, "Expected memories from concurrent operations"

    async def test_memory_search_performance_scaling(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test how search performance scales with database size."""
        
        # Create datasets of different sizes
        dataset_sizes = [100, 500, 1000, 2500]
        search_times = []
        
        for size in dataset_sizes:
            # Clear previous data
            await memory_manager_performance.cleanup()
            await memory_manager_performance.initialize()
            
            # Create dataset
            for i in range(size):
                entry = MemoryEntry(
                    id=f"scale_test_{size}_{i}",
                    content=f"Scaling test entry {i} with search terms: python programming machine learning artificial intelligence data science",
                    type=MemoryType.LEARNED_FACT,
                    user_id="scale_test_user",
                    importance=0.5 + (i % 100) * 0.005,  # Varied importance
                    timestamp=datetime.now() - timedelta(hours=i)
                )
                await memory_manager_performance.add_memory(entry)
            
            # Measure search performance
            with performance_benchmark(f"search_scaling_{size}") as bench:
                search_context = MemoryContext(
                    user_id="scale_test_user",
                    query="python programming",
                    max_results=50
                )
                
                search_result = await bench.measure_async_operation(
                    memory_manager_performance.search_memories, search_context
                )
                
                search_times.append(search_result['duration'])
        
        # Analyze scaling characteristics
        # Search time should scale sub-linearly (ideally logarithmically)
        max_scaling_factor = search_times[-1] / search_times[0]
        dataset_scaling_factor = dataset_sizes[-1] / dataset_sizes[0]
        
        # Assert that search doesn't scale linearly with dataset size
        assert max_scaling_factor < dataset_scaling_factor, (
            f"Search time scaling factor ({max_scaling_factor:.2f}) should be less than "
            f"dataset scaling factor ({dataset_scaling_factor:.2f})"
        )
        
        # Assert reasonable search times even for largest dataset
        perf_assert.assert_duration(search_times[-1], 1.0, f"Search with {dataset_sizes[-1]} entries")

    async def test_memory_cleanup_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test performance of memory cleanup operations."""
        
        # Create test data with varied timestamps
        cleanup_entries = []
        base_time = datetime.now()
        
        for i in range(500):
            # Create entries with different ages
            age_days = i // 100  # Groups of 100 with different ages
            entry = MemoryEntry(
                id=f"cleanup_test_{i}",
                content=f"Cleanup test entry {i}",
                type=MemoryType.CONVERSATION,
                user_id="cleanup_user",
                timestamp=base_time - timedelta(days=age_days),
                importance=0.1 if age_days > 2 else 0.8  # Old entries low importance
            )
            cleanup_entries.append(entry)
            await memory_manager_performance.add_memory(entry)
        
        with performance_benchmark("memory_cleanup") as bench:
            # Test cleanup operations
            cleanup_result = await bench.measure_async_operation(
                memory_manager_performance.cleanup_old_memories, days=1
            )
            
            # Test statistics after cleanup
            stats_result = await bench.measure_async_operation(
                memory_manager_performance.get_memory_stats
            )
        
        # Performance assertions
        perf_assert.assert_duration(cleanup_result['duration'], 2.0, "Memory cleanup operation")
        perf_assert.assert_duration(stats_result['duration'], 0.5, "Statistics after cleanup")
        
        # Verify cleanup effectiveness
        final_stats = stats_result['result']
        assert final_stats.total_memories < 500, "Cleanup should have removed some entries"

    async def test_memory_system_resource_usage(
        self, memory_manager_performance, performance_monitor, perf_assert
    ):
        """Test memory system resource usage characteristics."""
        
        performance_monitor.start_monitoring()
        
        try:
            # Sustained operation test
            for batch in range(10):
                batch_entries = []
                for i in range(100):
                    entry = MemoryEntry(
                        id=f"resource_test_{batch}_{i}",
                        content=f"Resource test entry {i} in batch {batch}",
                        type=MemoryType.CONVERSATION,
                        user_id=f"resource_user_{batch % 3}",
                        timestamp=datetime.now()
                    )
                    batch_entries.append(entry)
                
                # Add batch
                for entry in batch_entries:
                    await memory_manager_performance.add_memory(entry)
                
                # Perform searches
                for user_id in [f"resource_user_{i}" for i in range(3)]:
                    context = MemoryContext(user_id=user_id, max_results=25)
                    await memory_manager_performance.search_memories(context)
                
                # Update peak monitoring
                performance_monitor.update_peaks()
                
                # Small delay to allow for cleanup
                await asyncio.sleep(0.1)
                
        finally:
            performance_monitor.stop_monitoring()
        
        # Resource usage assertions
        summary = performance_monitor.get_summary()
        
        # Memory should not grow excessively
        perf_assert.assert_memory_usage(
            summary['memory_delta_mb'], 200, "Memory system resource test"
        )
        
        # Should not leak file descriptors
        assert summary['file_descriptors_delta'] <= 5, (
            f"File descriptor leak detected: {summary['file_descriptors_delta']} descriptors"
        )
        
        # Should not create excessive threads
        assert summary['threads_delta'] <= 2, (
            f"Thread leak detected: {summary['threads_delta']} threads"
        )

    async def test_memory_system_stress_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Stress test memory system under high load."""
        
        with performance_benchmark("memory_stress_test") as bench:
            # High-frequency operations
            stress_start = time.perf_counter()
            
            # Concurrent high-frequency operations
            async def stress_operations():
                """Perform stress operations."""
                tasks = []
                
                # High-frequency memory creation
                for i in range(200):
                    entry = MemoryEntry(
                        id=f"stress_{i}_{int(time.time() * 1000000) % 1000000}",
                        content=f"Stress test entry {i} with high frequency creation",
                        type=MemoryType.CONVERSATION,
                        user_id=f"stress_user_{i % 10}",
                        timestamp=datetime.now()
                    )
                    tasks.append(memory_manager_performance.add_memory(entry))
                    
                    # Add search operations
                    if i % 20 == 0:
                        context = MemoryContext(
                            user_id=f"stress_user_{i % 10}",
                            max_results=10
                        )
                        tasks.append(memory_manager_performance.search_memories(context))
                
                # Execute all operations
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Run stress operations
            await stress_operations()
            
            stress_duration = time.perf_counter() - stress_start
        
        # Stress test assertions
        perf_assert.assert_duration(stress_duration, 30.0, "Memory stress test")
        
        # Verify system is still functional after stress
        final_context = MemoryContext(user_id="stress_user_0", max_results=5)
        final_search = await memory_manager_performance.search_memories(final_context)
        assert len(final_search.entries) > 0, "System should remain functional after stress test"


@pytest.mark.performance
@pytest.mark.integration_perf  
class TestMemoryIntegrationPerformance:
    """Test memory system integration performance."""

    async def test_cross_component_memory_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test memory system performance in cross-component scenarios."""
        
        with performance_benchmark("cross_component_memory") as bench:
            # Simulate agent-memory integration
            async def simulate_agent_memory_usage():
                """Simulate how agents would use memory system."""
                agent_id = "perf_test_agent"
                user_id = "agent_user"
                
                # Agent creates memories during execution
                for i in range(50):
                    # Task result memory
                    task_memory = MemoryEntry(
                        id=f"agent_task_{agent_id}_{i}",
                        content=f"Agent task result {i}: completed processing",
                        type=MemoryType.TASK_RESULT,
                        user_id=user_id,
                        agent_id=agent_id,
                        metadata={"task_type": "processing", "result": "success"},
                        importance=0.7,
                        timestamp=datetime.now()
                    )
                    await memory_manager_performance.add_memory(task_memory)
                    
                    # Agent searches for context every 10 operations
                    if i % 10 == 0:
                        context = MemoryContext(
                            user_id=user_id,
                            agent_id=agent_id,
                            memory_types=[MemoryType.TASK_RESULT],
                            max_results=20
                        )
                        await memory_manager_performance.search_memories(context)
            
            # Measure agent-memory integration performance
            integration_result = await bench.measure_async_operation(
                simulate_agent_memory_usage
            )
        
        # Integration performance assertions
        perf_assert.assert_duration(
            integration_result['duration'], 5.0, "Agent-memory integration"
        )
        
        assert integration_result['success'], "Integration simulation should succeed"

    async def test_memory_conversation_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test memory system performance for conversation scenarios."""
        
        with performance_benchmark("conversation_memory") as bench:
            # Simulate conversation flow
            user_id = "conversation_user"
            session_id = "conversation_session"
            
            conversation_start = time.perf_counter()
            
            # Simulate a long conversation
            for turn in range(100):
                # Add user message
                await memory_manager_performance.add_conversation(
                    user_message=f"User message {turn}: What can you tell me about topic {turn}?",
                    assistant_response=f"Assistant response {turn}: Here's information about topic {turn}...",
                    user_id=user_id,
                    session_id=session_id
                )
                
                # Search conversation history every 10 turns
                if turn % 10 == 0 and turn > 0:
                    history = await memory_manager_performance.get_conversation_history(
                        user_id=user_id,
                        session_id=session_id,
                        limit=20
                    )
                    assert len(history) > 0, f"Should have conversation history at turn {turn}"
            
            conversation_duration = time.perf_counter() - conversation_start
        
        # Conversation performance assertions
        perf_assert.assert_duration(
            conversation_duration, 10.0, "Conversation memory simulation"
        )
        
        perf_assert.assert_throughput(
            100, conversation_duration, 10, "Conversation turn throughput"
        )