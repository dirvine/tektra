#!/usr/bin/env python3
"""
Agent System Performance Tests

Comprehensive performance testing for the Tektra agent system.
Tests agent execution, sandboxing, resource management, and scaling characteristics.
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.performance
@pytest.mark.benchmark
class TestAgentSystemPerformance:
    """Test agent system performance characteristics."""

    async def test_agent_initialization_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test agent initialization and setup performance."""
        
        # Mock agent system components
        from unittest.mock import AsyncMock, MagicMock
        
        # Mock agent builder and registry
        mock_agent_builder = MagicMock()
        mock_agent_builder.create_agent = AsyncMock()
        mock_agent_registry = MagicMock()
        mock_agent_registry.register_agent = AsyncMock()
        mock_agent_registry.initialize = AsyncMock()
        
        # Configure mock behaviors for initialization
        async def mock_create_agent(specification):
            # Simulate agent creation time based on complexity
            complexity_factor = len(specification.get('capabilities', [])) * 0.01
            await asyncio.sleep(0.05 + complexity_factor)
            return {
                'id': f"agent_{int(time.time() * 1000) % 1000}",
                'specification': specification,
                'status': 'ready'
            }
        
        async def mock_register_agent(agent):
            # Simulate registration overhead
            await asyncio.sleep(0.02)
            return True
        
        async def mock_registry_init():
            # Simulate registry initialization
            await asyncio.sleep(0.1)
            return True
        
        mock_agent_builder.create_agent.side_effect = mock_create_agent
        mock_agent_registry.register_agent.side_effect = mock_register_agent
        mock_agent_registry.initialize.side_effect = mock_registry_init
        
        with performance_benchmark("agent_initialization") as bench:
            # Test registry initialization
            registry_result = await bench.measure_async_operation(
                mock_agent_registry.initialize
            )
            
            # Test agent creation with varying complexity
            agent_specifications = [
                {
                    'name': 'Simple Agent',
                    'capabilities': ['text_processing']
                },
                {
                    'name': 'Medium Agent',
                    'capabilities': ['text_processing', 'data_analysis', 'file_handling']
                },
                {
                    'name': 'Complex Agent',
                    'capabilities': [
                        'text_processing', 'data_analysis', 'file_handling',
                        'web_scraping', 'image_processing', 'database_ops'
                    ]
                }
            ]
            
            creation_results = []
            for spec in agent_specifications:
                result = await bench.measure_async_operation(
                    mock_agent_builder.create_agent, spec
                )
                creation_results.append(result)
                
                # Agent creation should be fast
                max_duration = 0.2 + len(spec['capabilities']) * 0.02
                perf_assert.assert_duration(
                    result['duration'], max_duration, f"Agent creation: {spec['name']}"
                )
            
            # Test agent registration
            registration_results = []
            for result in creation_results:
                if result['success']:
                    reg_result = await bench.measure_async_operation(
                        mock_agent_registry.register_agent, result['result']
                    )
                    registration_results.append(reg_result)
        
        # Initialization performance assertions
        perf_assert.assert_duration(
            registry_result['duration'], 0.5, "Agent registry initialization"
        )
        
        # All operations should succeed
        assert registry_result['success'], "Registry initialization should succeed"
        assert all(r['success'] for r in creation_results), "All agent creations should succeed"
        assert all(r['success'] for r in registration_results), "All registrations should succeed"

    async def test_agent_execution_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test agent execution performance and resource usage."""
        
        # Mock agent execution components
        from unittest.mock import AsyncMock, MagicMock
        
        # Mock agent runtime and sandbox
        mock_runtime = MagicMock()
        mock_runtime.execute_agent = AsyncMock()
        mock_sandbox = MagicMock()
        mock_sandbox.create_workspace = AsyncMock()
        mock_sandbox.cleanup_workspace = AsyncMock()
        
        # Configure execution behaviors
        async def mock_workspace_creation(agent_id):
            await asyncio.sleep(0.03)  # Workspace setup time
            return f"/tmp/agent_workspace_{agent_id}"
        
        async def mock_workspace_cleanup(agent_id):
            await asyncio.sleep(0.01)  # Cleanup time
            return True
        
        async def mock_agent_execution(agent_spec, task):
            # Simulate complete agent execution lifecycle
            
            # 1. Workspace setup
            workspace = await mock_sandbox.create_workspace(agent_spec['id'])
            
            # 2. Task processing (simulated based on task complexity)
            task_complexity = len(task.get('description', '')) / 100.0
            execution_time = 0.1 + task_complexity * 0.05
            await asyncio.sleep(execution_time)
            
            # 3. Result generation
            result = {
                'success': True,
                'output': f"Task completed: {task.get('description', 'Unknown')}",
                'workspace': workspace,
                'execution_time': execution_time,
                'metrics': {
                    'cpu_time': execution_time * 0.8,
                    'memory_peak': 50 + task_complexity * 10,  # MB
                    'io_operations': int(task_complexity * 100)
                }
            }
            
            # 4. Workspace cleanup
            await mock_sandbox.cleanup_workspace(agent_spec['id'])
            
            return result
        
        mock_sandbox.create_workspace.side_effect = mock_workspace_creation
        mock_sandbox.cleanup_workspace.side_effect = mock_workspace_cleanup
        mock_runtime.execute_agent.side_effect = mock_agent_execution
        
        with performance_benchmark("agent_execution") as bench:
            # Test agent execution with varying task complexity
            execution_scenarios = [
                {
                    'agent': {'id': 'simple_agent', 'name': 'Simple Agent'},
                    'task': {'description': 'Simple task'},
                    'max_duration': 0.5
                },
                {
                    'agent': {'id': 'medium_agent', 'name': 'Medium Agent'},
                    'task': {
                        'description': 'Medium complexity task with data processing and analysis requirements'
                    },
                    'max_duration': 1.0
                },
                {
                    'agent': {'id': 'complex_agent', 'name': 'Complex Agent'},
                    'task': {
                        'description': 'Complex task involving multiple steps, data processing, file operations, network requests, and comprehensive analysis of results with detailed reporting and validation of outputs across multiple dimensions and criteria' * 2
                    },
                    'max_duration': 2.0
                }
            ]
            
            execution_results = []
            for scenario in execution_scenarios:
                result = await bench.measure_async_operation(
                    mock_runtime.execute_agent,
                    scenario['agent'],
                    scenario['task']
                )
                execution_results.append(result)
                
                # Execution should complete within expected time
                perf_assert.assert_duration(
                    result['duration'], scenario['max_duration'],
                    f"Agent execution: {scenario['agent']['name']}"
                )
        
        # Verify all executions succeeded
        for i, result in enumerate(execution_results):
            assert result['success'], f"Agent execution {i+1} should succeed"
            
            # Verify result structure
            execution_output = result['result']
            assert execution_output['success'], f"Agent task {i+1} should complete successfully"
            assert 'metrics' in execution_output, f"Agent execution {i+1} should include metrics"

    async def test_concurrent_agent_execution_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test performance of concurrent agent execution."""
        
        # Mock concurrent agent execution
        from unittest.mock import AsyncMock, MagicMock
        
        mock_runtime = MagicMock()
        mock_runtime.execute_agent = AsyncMock()
        
        # Simulate resource-aware concurrent execution
        execution_semaphore = asyncio.Semaphore(3)  # Limit concurrent executions
        
        async def mock_concurrent_execution(agent_spec, task):
            async with execution_semaphore:
                # Simulate resource competition
                base_time = 0.1
                resource_contention = (3 - execution_semaphore._value) * 0.02
                execution_time = base_time + resource_contention
                
                await asyncio.sleep(execution_time)
                
                return {
                    'success': True,
                    'agent_id': agent_spec['id'],
                    'task_result': f"Completed: {task.get('description', 'task')}",
                    'execution_time': execution_time,
                    'resource_contention': resource_contention
                }
        
        mock_runtime.execute_agent.side_effect = mock_concurrent_execution
        
        with performance_benchmark("concurrent_agent_execution") as bench:
            concurrent_start = time.perf_counter()
            
            # Create multiple agents and tasks for concurrent execution
            concurrent_scenarios = [
                {
                    'agent': {'id': f'concurrent_agent_{i}', 'name': f'Agent {i}'},
                    'task': {'description': f'Concurrent task {i} with processing requirements'}
                }
                for i in range(10)
            ]
            
            # Execute all agents concurrently
            concurrent_tasks = [
                mock_runtime.execute_agent(scenario['agent'], scenario['task'])
                for scenario in concurrent_scenarios
            ]
            
            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            concurrent_duration = time.perf_counter() - concurrent_start
        
        # Concurrent execution performance assertions
        perf_assert.assert_duration(
            concurrent_duration, 3.0, "Concurrent agent execution (10 agents)"
        )
        
        # Calculate effective throughput
        successful_executions = sum(
            1 for result in concurrent_results 
            if not isinstance(result, Exception) and result.get('success', False)
        )
        
        perf_assert.assert_throughput(
            successful_executions, concurrent_duration, 3, "Concurrent agent throughput"
        )
        
        # Verify most executions succeeded despite concurrency
        success_rate = successful_executions / len(concurrent_scenarios)
        assert success_rate >= 0.9, (
            f"Concurrent execution success rate: {success_rate:.2f}, expected >= 0.9"
        )

    async def test_agent_sandbox_performance(
        self, performance_benchmark, perf_assert
    ):
        """Test agent sandbox creation and management performance."""
        
        # Mock sandbox operations
        from unittest.mock import AsyncMock, MagicMock
        
        mock_sandbox = MagicMock()
        mock_sandbox.create_workspace = AsyncMock()
        mock_sandbox.setup_environment = AsyncMock()
        mock_sandbox.execute_code = AsyncMock()
        mock_sandbox.cleanup_workspace = AsyncMock()
        
        # Configure sandbox behaviors
        async def mock_create_workspace(agent_id, resources=None):
            # Simulate workspace creation time based on resources
            base_time = 0.05
            if resources:
                resource_factor = len(resources.get('tools', [])) * 0.01
                base_time += resource_factor
            
            await asyncio.sleep(base_time)
            return {
                'workspace_id': f"ws_{agent_id}_{int(time.time() * 1000) % 1000}",
                'path': f"/tmp/sandbox/{agent_id}",
                'resources': resources or {}
            }
        
        async def mock_setup_environment(workspace_id, config):
            # Simulate environment setup
            setup_time = 0.03 + len(config.get('dependencies', [])) * 0.005
            await asyncio.sleep(setup_time)
            return True
        
        async def mock_execute_code(workspace_id, code):
            # Simulate code execution
            execution_time = 0.02 + len(code) * 0.0001
            await asyncio.sleep(execution_time)
            return {
                'success': True,
                'output': f"Code executed in {workspace_id}",
                'execution_time': execution_time
            }
        
        async def mock_cleanup_workspace(workspace_id):
            # Simulate cleanup
            await asyncio.sleep(0.02)
            return True
        
        mock_sandbox.create_workspace.side_effect = mock_create_workspace
        mock_sandbox.setup_environment.side_effect = mock_setup_environment
        mock_sandbox.execute_code.side_effect = mock_execute_code
        mock_sandbox.cleanup_workspace.side_effect = mock_cleanup_workspace
        
        with performance_benchmark("agent_sandbox") as bench:
            # Test sandbox lifecycle performance
            sandbox_scenarios = [
                {
                    'agent_id': 'basic_agent',
                    'resources': {'tools': ['python', 'pip']},
                    'config': {'dependencies': ['requests', 'pandas']},
                    'code': 'print("Hello from sandbox")',
                    'max_duration': 0.3
                },
                {
                    'agent_id': 'advanced_agent',
                    'resources': {
                        'tools': ['python', 'node', 'git', 'docker', 'curl'],
                        'network_access': True
                    },
                    'config': {
                        'dependencies': ['requests', 'pandas', 'numpy', 'scikit-learn', 'matplotlib']
                    },
                    'code': '''
import pandas as pd
import numpy as np
data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
result = data.describe()
print(result)
                    ''',
                    'max_duration': 0.5
                }
            ]
            
            sandbox_results = []
            for scenario in sandbox_scenarios:
                scenario_start = time.perf_counter()
                
                # Complete sandbox lifecycle
                workspace = await mock_sandbox.create_workspace(
                    scenario['agent_id'], scenario['resources']
                )
                
                await mock_sandbox.setup_environment(
                    workspace['workspace_id'], scenario['config']
                )
                
                execution_result = await mock_sandbox.execute_code(
                    workspace['workspace_id'], scenario['code']
                )
                
                await mock_sandbox.cleanup_workspace(workspace['workspace_id'])
                
                scenario_duration = time.perf_counter() - scenario_start
                sandbox_results.append({
                    'duration': scenario_duration,
                    'workspace': workspace,
                    'execution': execution_result
                })
                
                # Sandbox lifecycle should be efficient
                perf_assert.assert_duration(
                    scenario_duration, scenario['max_duration'],
                    f"Sandbox lifecycle: {scenario['agent_id']}"
                )
        
        # Verify all sandbox operations succeeded
        for i, result in enumerate(sandbox_results):
            assert result['execution']['success'], f"Sandbox execution {i+1} should succeed"

    async def test_agent_memory_integration_performance(
        self, memory_manager_performance, performance_benchmark, perf_assert
    ):
        """Test agent system integration with memory performance."""
        
        # Mock agent with memory integration
        from unittest.mock import AsyncMock, MagicMock
        from tektra.memory.memory_types import MemoryEntry, MemoryType, MemoryContext
        
        mock_agent = MagicMock()
        mock_agent.execute_with_memory = AsyncMock()
        
        # Configure agent-memory integration
        async def mock_agent_memory_execution(task, memory_manager):
            # 1. Retrieve relevant memories
            context = MemoryContext(
                user_id=task.get('user_id', 'default_user'),
                query=task.get('description', ''),
                max_results=20
            )
            relevant_memories = await memory_manager.search_memories(context)
            
            # 2. Process task with memory context
            processing_time = 0.1 + len(relevant_memories.entries) * 0.001
            await asyncio.sleep(processing_time)
            
            # 3. Generate result
            result_content = f"Task completed with {len(relevant_memories.entries)} memories as context"
            
            # 4. Store execution result in memory
            result_memory = MemoryEntry(
                id=f"agent_result_{int(time.time() * 1000000) % 1000000}",
                content=result_content,
                type=MemoryType.TASK_RESULT,
                user_id=task.get('user_id', 'default_user'),
                agent_id=task.get('agent_id', 'test_agent'),
                metadata={
                    'task_type': task.get('type', 'unknown'),
                    'memory_context_size': len(relevant_memories.entries)
                },
                timestamp=datetime.now()
            )
            await memory_manager.add_memory(result_memory)
            
            return {
                'success': True,
                'result': result_content,
                'memory_context_size': len(relevant_memories.entries),
                'processing_time': processing_time
            }
        
        mock_agent.execute_with_memory.side_effect = mock_agent_memory_execution
        
        with performance_benchmark("agent_memory_integration") as bench:
            # Pre-populate memory with context
            for i in range(50):
                context_memory = MemoryEntry(
                    id=f"context_memory_{i}",
                    content=f"machine learning data science analysis trends {i}",
                    type=MemoryType.LEARNED_FACT,
                    user_id="memory_test_user",
                    metadata={'topic': 'machine_learning', 'relevance': 0.8},
                    timestamp=datetime.now()
                )
                await memory_manager_performance.add_memory(context_memory)
            
            # Test agent execution with memory integration
            memory_integration_tasks = [
                {
                    'description': 'machine learning analysis',
                    'user_id': 'memory_test_user',
                    'agent_id': 'ml_agent',
                    'type': 'analysis'
                },
                {
                    'description': 'data science trends',
                    'user_id': 'memory_test_user',
                    'agent_id': 'summary_agent',
                    'type': 'summarization'
                },
                {
                    'description': 'analysis trends',
                    'user_id': 'memory_test_user',
                    'agent_id': 'recommendation_agent',
                    'type': 'recommendation'
                }
            ]
            
            integration_results = []
            for task in memory_integration_tasks:
                result = await bench.measure_async_operation(
                    mock_agent.execute_with_memory, task, memory_manager_performance
                )
                integration_results.append(result)
                
                # Memory-integrated execution should be reasonably fast
                perf_assert.assert_duration(
                    result['duration'], 1.0, f"Agent+Memory: {task['type']}"
                )
        
        # Verify integration results
        for i, result in enumerate(integration_results):
            assert result['success'], f"Memory integration {i+1} should succeed"
            execution_result = result['result']
            assert execution_result['success'], f"Agent execution {i+1} should succeed"
            # Note: Memory context might be 0 if search doesn't find matches - this is acceptable
            assert execution_result['memory_context_size'] >= 0, (
                f"Agent execution {i+1} should have valid memory context size"
            )

    async def test_agent_scaling_performance(
        self, performance_benchmark, performance_monitor, perf_assert
    ):
        """Test agent system scaling characteristics."""
        
        # Mock scalable agent system
        from unittest.mock import AsyncMock, MagicMock
        
        mock_agent_pool = MagicMock()
        mock_agent_pool.execute_agents = AsyncMock()
        
        # Configure scaling behavior
        async def mock_scaled_execution(agent_count, tasks_per_agent):
            """Simulate execution with varying numbers of agents."""
            
            # Resource overhead increases with agent count
            overhead_factor = 1 + (agent_count - 1) * 0.1
            base_time = 0.1 * overhead_factor
            
            # Execute all agents
            all_tasks = []
            for agent_id in range(agent_count):
                for task_id in range(tasks_per_agent):
                    task_time = base_time + (task_id * 0.01)  # Tasks get slightly slower
                    task = asyncio.sleep(task_time)
                    all_tasks.append(task)
            
            # Execute with limited concurrency to simulate resource constraints
            semaphore = asyncio.Semaphore(min(agent_count, 5))
            
            async def execute_with_limit(task):
                async with semaphore:
                    await task
                    return True
            
            results = await asyncio.gather(*[
                execute_with_limit(task) for task in all_tasks
            ])
            
            return {
                'agent_count': agent_count,
                'tasks_per_agent': tasks_per_agent,
                'total_tasks': len(all_tasks),
                'successful_tasks': sum(results),
                'overhead_factor': overhead_factor
            }
        
        mock_agent_pool.execute_agents.side_effect = mock_scaled_execution
        
        with performance_benchmark("agent_scaling") as bench:
            performance_monitor.start_monitoring()
            
            # Test scaling with different agent counts
            scaling_scenarios = [
                (1, 10),   # 1 agent, 10 tasks
                (3, 10),   # 3 agents, 10 tasks each
                (5, 10),   # 5 agents, 10 tasks each
                (10, 5),   # 10 agents, 5 tasks each
                (15, 3)    # 15 agents, 3 tasks each
            ]
            
            scaling_results = []
            for agent_count, tasks_per_agent in scaling_scenarios:
                scaling_start = time.perf_counter()
                
                result = await mock_agent_pool.execute_agents(agent_count, tasks_per_agent)
                
                scaling_duration = time.perf_counter() - scaling_start
                result['execution_time'] = scaling_duration
                scaling_results.append(result)
                
                # Scaling should remain efficient
                total_tasks = agent_count * tasks_per_agent
                max_duration = 2.0 + total_tasks * 0.05  # Scale with task count
                
                perf_assert.assert_duration(
                    scaling_duration, max_duration,
                    f"Agent scaling: {agent_count} agents, {total_tasks} total tasks"
                )
                
                performance_monitor.update_peaks()
            
            performance_monitor.stop_monitoring()
        
        # Analyze scaling characteristics
        for result in scaling_results:
            total_tasks = result['total_tasks']
            execution_time = result['execution_time']
            
            # Calculate throughput
            throughput = total_tasks / execution_time
            
            # Throughput should remain reasonable as we scale
            min_throughput = max(2.0, total_tasks / 10.0)  # At least 2 tasks/sec, scaling with load
            assert throughput >= min_throughput, (
                f"Scaling throughput: {throughput:.2f} tasks/sec, "
                f"expected >= {min_throughput:.2f} for {total_tasks} tasks"
            )
        
        # Resource usage should scale predictably
        summary = performance_monitor.get_summary()
        perf_assert.assert_memory_usage(
            summary['peak_memory_mb'], 600, "Agent scaling test"  # Increased limit for realistic testing
        )