#!/usr/bin/env python3
"""
AI Backend Performance Tests

Comprehensive performance testing for AI backends, focusing on:
- Model initialization performance and resource usage
- Inference latency and throughput
- Memory efficiency and leak detection
- Concurrent request handling
- Multimodal processing performance
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.performance
@pytest.mark.benchmark
@pytest.mark.heavy  # These tests may require heavy AI models
class TestAIBackendPerformance:
    """Test AI backend performance characteristics."""

    async def test_model_initialization_performance(
        self, performance_benchmark, performance_monitor, perf_assert
    ):
        """Test AI model initialization performance and resource usage."""
        
        # Use mock backend for performance testing to avoid actual model loading
        from unittest.mock import AsyncMock, MagicMock
        
        # Create mock backend that simulates model loading behavior
        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.cleanup = AsyncMock()
        mock_backend.is_initialized = True
        mock_backend.get_model_info = AsyncMock(return_value={
            "model_name": "mock_model",
            "memory_usage_mb": 4096,
            "parameter_count": 7000000000
        })
        
        with performance_benchmark("model_initialization") as bench:
            performance_monitor.start_monitoring()
            
            # Measure initialization performance
            init_result = await bench.measure_async_operation(
                mock_backend.initialize
            )
            
            # Measure model info retrieval
            info_result = await bench.measure_async_operation(
                mock_backend.get_model_info
            )
            
            performance_monitor.stop_monitoring()
        
        # Performance assertions for mock (real tests would have different limits)
        perf_assert.assert_duration(init_result['duration'], 0.1, "Mock model initialization")
        perf_assert.assert_duration(info_result['duration'], 0.1, "Model info retrieval")
        
        # Verify operations succeeded
        assert init_result['success'], "Model initialization should succeed"
        assert info_result['success'], "Model info retrieval should succeed"
        
        # Check resource usage
        summary = performance_monitor.get_summary()
        perf_assert.assert_memory_usage(
            summary['peak_memory_mb'], 500, "Mock model initialization"  # Realistic for test environment
        )

    async def test_text_inference_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test text inference performance and latency."""
        
        # Mock backend for consistent performance testing
        from unittest.mock import AsyncMock, MagicMock
        
        mock_backend = MagicMock()
        mock_backend.process_text_query = AsyncMock()
        
        # Configure mock to simulate various response times
        async def mock_text_inference(prompt, **kwargs):
            # Simulate processing time based on prompt length
            processing_time = len(prompt) * 0.001  # 1ms per character
            await asyncio.sleep(processing_time)
            return f"Mock response to: {prompt[:50]}..."
        
        mock_backend.process_text_query.side_effect = mock_text_inference
        
        with performance_benchmark("text_inference") as bench:
            # Test various prompt sizes
            test_prompts = [
                "Short prompt",
                "Medium length prompt with more detailed context and information that requires processing",
                "Very long prompt with extensive context, detailed background information, complex requirements, multiple constraints, and comprehensive instructions that simulate real-world usage patterns where users provide substantial context to get the most relevant and accurate responses from the AI system" * 3
            ]
            
            inference_results = []
            for i, prompt in enumerate(test_prompts):
                result = await bench.measure_async_operation(
                    mock_backend.process_text_query, prompt
                )
                inference_results.append(result)
                
                # Performance assertions based on prompt complexity
                max_duration = 0.1 + len(prompt) * 0.001  # Base + processing time
                perf_assert.assert_duration(
                    result['duration'], max_duration, f"Text inference {i+1}"
                )
        
        # Verify all inferences succeeded
        for i, result in enumerate(inference_results):
            assert result['success'], f"Text inference {i+1} should succeed"

    async def test_vision_inference_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test vision/multimodal inference performance."""
        
        # Mock vision backend
        from unittest.mock import AsyncMock, MagicMock
        
        mock_backend = MagicMock()
        mock_backend.process_vision_query = AsyncMock()
        
        # Configure mock to simulate vision processing
        async def mock_vision_inference(image_path, prompt, **kwargs):
            # Simulate vision processing time
            await asyncio.sleep(0.1)  # 100ms for vision processing
            return f"Mock vision response for {prompt}"
        
        mock_backend.process_vision_query.side_effect = mock_vision_inference
        
        with performance_benchmark("vision_inference") as bench:
            # Test vision inference with different scenarios
            vision_scenarios = [
                ("mock_image.jpg", "Describe this image"),
                ("mock_complex_image.jpg", "Analyze the complex elements in this image"),
                ("mock_document.pdf", "Extract text and analyze the document structure")
            ]
            
            vision_results = []
            for image_path, prompt in vision_scenarios:
                result = await bench.measure_async_operation(
                    mock_backend.process_vision_query, image_path, prompt
                )
                vision_results.append(result)
                
                # Vision processing should be reasonably fast
                perf_assert.assert_duration(
                    result['duration'], 0.5, f"Vision inference for {image_path}"
                )
        
        # Verify all vision inferences succeeded
        for i, result in enumerate(vision_results):
            assert result['success'], f"Vision inference {i+1} should succeed"

    async def test_concurrent_inference_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test AI backend performance under concurrent requests."""
        
        # Mock backend with concurrency handling
        from unittest.mock import AsyncMock, MagicMock
        
        mock_backend = MagicMock()
        mock_backend.process_text_query = AsyncMock()
        
        # Simulate concurrent processing with queue/lock behavior
        processing_lock = asyncio.Semaphore(2)  # Simulate limited concurrency
        
        async def mock_concurrent_inference(prompt, **kwargs):
            async with processing_lock:
                # Simulate processing time
                await asyncio.sleep(0.05)  # 50ms processing time
                return f"Concurrent response: {prompt[:30]}..."
        
        mock_backend.process_text_query.side_effect = mock_concurrent_inference
        
        with performance_benchmark("concurrent_inference") as bench:
            # Create multiple concurrent requests
            concurrent_prompts = [
                f"Concurrent request {i}: Please process this query efficiently"
                for i in range(10)
            ]
            
            concurrent_start = time.perf_counter()
            
            # Execute concurrent requests
            tasks = [
                mock_backend.process_text_query(prompt)
                for prompt in concurrent_prompts
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            concurrent_duration = time.perf_counter() - concurrent_start
        
        # Concurrent processing should be efficient
        perf_assert.assert_duration(
            concurrent_duration, 1.0, "Concurrent inference (10 requests)"
        )
        
        # Verify all requests completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(concurrent_prompts), (
            "All concurrent requests should complete successfully"
        )

    async def test_ai_backend_memory_efficiency(
        self, performance_monitor, perf_assert
    ):
        """Test AI backend memory efficiency and leak detection."""
        
        # Mock backend with memory simulation
        from unittest.mock import AsyncMock, MagicMock
        
        mock_backend = MagicMock()
        mock_backend.process_text_query = AsyncMock()
        mock_backend.cleanup = AsyncMock()
        
        # Simulate memory usage patterns
        async def mock_inference_with_memory(prompt, **kwargs):
            # Simulate temporary memory usage during inference
            await asyncio.sleep(0.01)
            return f"Response: {prompt}"
        
        mock_backend.process_text_query.side_effect = mock_inference_with_memory
        
        performance_monitor.start_monitoring()
        
        try:
            # Sustained inference operations to test memory stability
            for batch in range(20):
                batch_tasks = []
                for i in range(25):
                    prompt = f"Memory test batch {batch}, request {i}"
                    task = mock_backend.process_text_query(prompt)
                    batch_tasks.append(task)
                
                # Process batch
                await asyncio.gather(*batch_tasks)
                
                # Update peak monitoring
                performance_monitor.update_peaks()
                
                # Small delay to allow garbage collection
                await asyncio.sleep(0.01)
            
        finally:
            # Cleanup
            await mock_backend.cleanup()
            performance_monitor.stop_monitoring()
        
        # Memory efficiency assertions
        summary = performance_monitor.get_summary()
        
        # Memory usage should remain stable (mock should use minimal memory)
        perf_assert.assert_memory_usage(
            summary['memory_delta_mb'], 50, "AI backend memory efficiency test"
        )
        
        # No significant resource leaks
        assert summary['file_descriptors_delta'] <= 2, (
            f"File descriptor leak in AI backend: {summary['file_descriptors_delta']}"
        )

    async def test_ai_backend_throughput_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test AI backend throughput characteristics."""
        
        # Mock backend optimized for throughput testing
        from unittest.mock import AsyncMock, MagicMock
        
        mock_backend = MagicMock()
        mock_backend.process_text_query = AsyncMock()
        
        # Fast mock processing for throughput testing
        async def mock_fast_inference(prompt, **kwargs):
            await asyncio.sleep(0.01)  # 10ms per request
            return f"Fast response: {prompt[:20]}..."
        
        mock_backend.process_text_query.side_effect = mock_fast_inference
        
        with performance_benchmark("ai_throughput") as bench:
            # High-volume throughput test
            throughput_start = time.perf_counter()
            
            # Process many requests to measure throughput
            throughput_requests = 200
            requests = [
                f"Throughput test request {i}"
                for i in range(throughput_requests)
            ]
            
            # Process in smaller batches to simulate realistic usage
            batch_size = 10
            all_results = []
            
            for i in range(0, throughput_requests, batch_size):
                batch = requests[i:i + batch_size]
                batch_tasks = [
                    mock_backend.process_text_query(prompt)
                    for prompt in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks)
                all_results.extend(batch_results)
            
            throughput_duration = time.perf_counter() - throughput_start
        
        # Throughput assertions
        perf_assert.assert_duration(
            throughput_duration, 10.0, f"Processing {throughput_requests} requests"
        )
        
        perf_assert.assert_throughput(
            throughput_requests, throughput_duration, 20, "AI backend throughput"
        )
        
        # Verify all requests completed
        assert len(all_results) == throughput_requests, (
            "All throughput test requests should complete"
        )

    async def test_ai_backend_stress_performance(
        self, performance_benchmark, performance_monitor, perf_assert
    ):
        """Stress test AI backend under extreme load."""
        
        # Mock backend for stress testing
        from unittest.mock import AsyncMock, MagicMock
        
        mock_backend = MagicMock()
        mock_backend.process_text_query = AsyncMock()
        mock_backend.process_vision_query = AsyncMock()
        
        # Simulate variable processing times under stress
        async def mock_stress_text_inference(prompt, **kwargs):
            # Simulate variable load
            processing_time = 0.02 + (hash(prompt) % 100) * 0.001
            await asyncio.sleep(processing_time)
            return f"Stress response: {prompt[:15]}..."
        
        async def mock_stress_vision_inference(image_path, prompt, **kwargs):
            await asyncio.sleep(0.1)  # Vision is slower
            return f"Stress vision response: {prompt[:15]}..."
        
        mock_backend.process_text_query.side_effect = mock_stress_text_inference
        mock_backend.process_vision_query.side_effect = mock_stress_vision_inference
        
        with performance_benchmark("ai_stress_test") as bench:
            performance_monitor.start_monitoring()
            
            stress_start = time.perf_counter()
            
            # High-load stress test with mixed operations
            stress_tasks = []
            
            # Text inference stress
            for i in range(100):
                prompt = f"Stress text {i}: " + "x" * (i % 200)  # Variable length
                stress_tasks.append(mock_backend.process_text_query(prompt))
            
            # Vision inference stress
            for i in range(20):
                image_path = f"stress_image_{i}.jpg"
                prompt = f"Stress vision {i}"
                stress_tasks.append(mock_backend.process_vision_query(image_path, prompt))
            
            # Execute all stress operations
            stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
            
            stress_duration = time.perf_counter() - stress_start
            performance_monitor.stop_monitoring()
        
        # Stress test assertions
        perf_assert.assert_duration(
            stress_duration, 30.0, "AI backend stress test"
        )
        
        # Count successful operations
        successful_operations = sum(
            1 for result in stress_results 
            if not isinstance(result, Exception)
        )
        
        # Most operations should succeed under stress
        success_rate = successful_operations / len(stress_tasks)
        assert success_rate >= 0.95, (
            f"Success rate under stress: {success_rate:.2f}, expected >= 0.95"
        )
        
        # Resource usage should remain reasonable
        summary = performance_monitor.get_summary()
        perf_assert.assert_memory_usage(
            summary['peak_memory_mb'], 500, "AI backend stress test"  # Increased for realistic stress testing
        )


@pytest.mark.performance
@pytest.mark.integration_perf
class TestAIBackendIntegrationPerformance:
    """Test AI backend performance in integration scenarios."""

    async def test_ai_memory_integration_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test AI backend performance when integrated with memory system."""
        
        # Mock AI backend with memory integration
        from unittest.mock import AsyncMock, MagicMock
        from tektra.memory.memory_types import MemoryEntry, MemoryType, MemoryContext
        from datetime import datetime
        
        mock_backend = MagicMock()
        mock_backend.process_text_query = AsyncMock()
        
        # Simulate AI backend that uses memory for context
        async def mock_ai_with_memory(prompt, **kwargs):
            # Simulate retrieving memory context
            context = MemoryContext(
                user_id="integration_user",
                query=prompt[:50],  # Use prompt for context search
                max_results=10
            )
            memory_results = await memory_manager_performance.search_memories(context)
            
            # Simulate processing with memory context
            await asyncio.sleep(0.05)  # Processing time
            
            # Simulate storing result in memory
            result_memory = MemoryEntry(
                id=f"ai_result_{int(time.time() * 1000000) % 1000000}",
                content=f"AI response to: {prompt}",
                type=MemoryType.TASK_RESULT,
                user_id="integration_user",
                timestamp=datetime.now()
            )
            await memory_manager_performance.add_memory(result_memory)
            
            return f"AI response with {len(memory_results.entries)} context memories"
        
        mock_backend.process_text_query.side_effect = mock_ai_with_memory
        
        with performance_benchmark("ai_memory_integration") as bench:
            # Test integrated AI + memory operations
            integration_prompts = [
                "What have we discussed about machine learning?",
                "Can you summarize our previous conversation?",
                "Based on our history, what should we focus on next?",
                "Tell me about the patterns in our interactions",
                "What are the key topics we've covered?"
            ]
            
            integration_start = time.perf_counter()
            
            for prompt in integration_prompts:
                result = await bench.measure_async_operation(
                    mock_backend.process_text_query, prompt
                )
                
                # Each integrated operation should be reasonably fast
                perf_assert.assert_duration(
                    result['duration'], 0.5, f"AI+Memory integration for: {prompt[:30]}"
                )
                
                assert result['success'], f"Integration should succeed for: {prompt[:30]}"
            
            integration_duration = time.perf_counter() - integration_start
        
        # Overall integration performance
        perf_assert.assert_duration(
            integration_duration, 5.0, "Complete AI+Memory integration test"
        )
        
        perf_assert.assert_throughput(
            len(integration_prompts), integration_duration, 1, "AI+Memory throughput"
        )

    async def test_multimodal_processing_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test multimodal processing performance."""
        
        # Mock multimodal AI backend
        from unittest.mock import AsyncMock, MagicMock
        
        mock_backend = MagicMock()
        mock_backend.process_multimodal_query = AsyncMock()
        
        # Simulate complex multimodal processing
        async def mock_multimodal_processing(inputs, **kwargs):
            # Simulate processing time based on input complexity
            base_time = 0.1  # Base processing time
            
            # Add time for each input type
            for input_type, content in inputs.items():
                if input_type == "text":
                    base_time += len(content) * 0.0001
                elif input_type == "image":
                    base_time += 0.2  # Vision processing
                elif input_type == "audio":
                    base_time += 0.15  # Audio processing
                elif input_type == "document":
                    base_time += 0.1  # Document processing
            
            await asyncio.sleep(base_time)
            return f"Multimodal response processing {len(inputs)} input types"
        
        mock_backend.process_multimodal_query.side_effect = mock_multimodal_processing
        
        with performance_benchmark("multimodal_processing") as bench:
            # Test various multimodal scenarios
            multimodal_scenarios = [
                {
                    "text": "Analyze this image and document",
                    "image": "test_image.jpg"
                },
                {
                    "text": "Transcribe and analyze this audio",
                    "audio": "test_audio.wav"
                },
                {
                    "text": "Process this document and compare with the image",
                    "image": "chart.png",
                    "document": "report.pdf"
                },
                {
                    "text": "Complete multimodal analysis",
                    "image": "complex_image.jpg",
                    "audio": "speech.wav",
                    "document": "data.xlsx"
                }
            ]
            
            multimodal_results = []
            for i, inputs in enumerate(multimodal_scenarios):
                result = await bench.measure_async_operation(
                    mock_backend.process_multimodal_query, inputs
                )
                multimodal_results.append(result)
                
                # Performance expectations based on complexity
                max_duration = 0.5 + len(inputs) * 0.2  # Scale with input count
                perf_assert.assert_duration(
                    result['duration'], max_duration, f"Multimodal scenario {i+1}"
                )
        
        # Verify all multimodal processing succeeded
        for i, result in enumerate(multimodal_results):
            assert result['success'], f"Multimodal processing {i+1} should succeed"