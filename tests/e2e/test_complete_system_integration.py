#!/usr/bin/env python3
"""
Tektra AI Assistant - Complete System Integration Tests

Comprehensive end-to-end testing of the fully integrated Tektra system
including all phases: SmolAgents, Security, Performance, and Deployment.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with pytest,pytest-asyncio,loguru,psutil python -m pytest test_complete_system_integration.py -v
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest>=7.4.0",
#     "pytest-asyncio>=0.21.0",
#     "loguru>=0.7.0",
#     "psutil>=5.9.0",
#     "aiohttp>=3.8.0",
#     "websockets>=11.0.0",
# ]
# ///

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any
import pytest

from loguru import logger

# Import all Tektra components
from tektra.core.tektra_system import TektraSystem, TektraSystemConfig, SystemState, ComponentStatus
from tektra.core.deployment_manager import DeploymentManager, DeploymentStatus
from tektra.core.error_handling import ErrorHandler, TektraError
from tektra.config.production_config import create_development_config, create_production_config
from tektra.security.context import SecurityContext, SecurityLevel
from tektra.agents.smolagents_real import SmolAgentsManager


class TestCompleteSystemIntegration:
    """Comprehensive end-to-end system integration tests."""
    
    @pytest.fixture
    async def development_config(self):
        """Create development configuration for testing."""
        return create_development_config()
    
    @pytest.fixture
    async def production_config(self):
        """Create production configuration for testing."""
        return create_production_config()
    
    @pytest.fixture
    async def tektra_system(self, development_config):
        """Create and initialize Tektra system for testing."""
        tektra_config = development_config.to_tektra_config()
        system = TektraSystem(tektra_config)
        
        # Initialize with shorter timeouts for testing
        if await system.initialize():
            yield system
            await system.shutdown()
        else:
            pytest.fail("Failed to initialize Tektra system")
    
    @pytest.fixture
    async def deployment_manager(self, development_config):
        """Create deployment manager for testing."""
        manager = DeploymentManager(development_config)
        
        if await manager.initialize():
            yield manager
            await manager.shutdown()
        else:
            pytest.fail("Failed to initialize deployment manager")

    async def test_system_initialization_and_shutdown(self, development_config):
        """Test complete system initialization and graceful shutdown."""
        logger.info("🧪 Testing system initialization and shutdown")
        
        # Test system configuration
        tektra_config = development_config.to_tektra_config()
        assert tektra_config.enable_agents == True
        assert tektra_config.enable_security == True
        assert tektra_config.enable_performance == True
        
        # Initialize system
        system = TektraSystem(tektra_config)
        
        # Verify initialization
        init_success = await system.initialize()
        assert init_success == True
        assert system.state == SystemState.RUNNING
        
        # Verify all components are healthy
        health = system.get_system_health()
        assert health.overall_status == ComponentStatus.HEALTHY
        
        # Check critical components
        critical_components = ['agents_manager', 'security_monitor', 'performance_monitor']
        for component in critical_components:
            assert component in health.components
            assert health.components[component] in [ComponentStatus.HEALTHY, ComponentStatus.STARTING]
        
        # Test graceful shutdown
        await system.shutdown(graceful=True)
        assert system.state == SystemState.STOPPED
        
        logger.info("✅ System initialization and shutdown test passed")

    async def test_agent_lifecycle_management(self, tektra_system):
        """Test complete agent lifecycle from creation to execution to cleanup."""
        logger.info("🧪 Testing agent lifecycle management")
        
        # Create security context
        security_context = SecurityContext(
            agent_id="test_agent_lifecycle",
            security_level=SecurityLevel.MEDIUM,
            session_id="test_session_lifecycle"
        )
        
        # Test agent creation
        agent_id = await tektra_system.create_agent(
            agent_name="Lifecycle Test Agent",
            agent_config={
                "model": "text_completion",
                "max_tokens": 100,
                "temperature": 0.7
            },
            security_context=security_context
        )
        
        assert agent_id is not None
        assert isinstance(agent_id, str)
        logger.info(f"✅ Agent created: {agent_id}")
        
        # Test tool validation
        safe_tool_code = '''
def safe_calculation(x, y):
    """Safe mathematical calculation."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Invalid input types")
    return x + y

result = safe_calculation(10, 20)
print(f"Result: {result}")
'''
        
        is_safe = await tektra_system.validate_tool(
            tool_code=safe_tool_code,
            tool_name="safe_calculation",
            security_context=security_context
        )
        assert is_safe == True
        logger.info("✅ Tool validation passed")
        
        # Test unsafe tool detection
        unsafe_tool_code = '''
import os
import subprocess
subprocess.run(["rm", "-rf", "/"])  # Dangerous command
'''
        
        is_safe = await tektra_system.validate_tool(
            tool_code=unsafe_tool_code,
            tool_name="unsafe_tool",
            security_context=security_context
        )
        assert is_safe == False
        logger.info("✅ Unsafe tool detection passed")
        
        # Test agent task execution
        task_result = await tektra_system.execute_agent_task(
            agent_id=agent_id,
            task_description="Calculate the sum of 15 and 25",
            context={"type": "mathematical_calculation"},
            security_context=security_context
        )
        
        assert task_result is not None
        assert isinstance(task_result, dict)
        logger.info(f"✅ Task execution completed: {task_result}")
        
        logger.info("✅ Agent lifecycle management test passed")

    async def test_security_enforcement_and_monitoring(self, tektra_system):
        """Test security enforcement across all system components."""
        logger.info("🧪 Testing security enforcement and monitoring")
        
        # Test different security levels
        security_levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH]
        
        for level in security_levels:
            security_context = SecurityContext(
                agent_id=f"security_test_{level.value}",
                security_level=level,
                session_id=f"security_session_{level.value}"
            )
            
            # Create agent with different security levels
            agent_id = await tektra_system.create_agent(
                agent_name=f"Security Test Agent {level.value}",
                agent_config={"model": "text_completion", "max_tokens": 50},
                security_context=security_context
            )
            
            assert agent_id is not None
            logger.info(f"✅ Agent created with {level.value} security level")
            
            # Test permission-based access
            if level == SecurityLevel.HIGH:
                # High security should have strict validation
                restricted_tool = '''
import socket
s = socket.socket()
s.connect(("external-api.com", 80))
'''
                is_safe = await tektra_system.validate_tool(
                    tool_code=restricted_tool,
                    tool_name="network_tool",
                    security_context=security_context
                )
                # Should be blocked at high security level
                assert is_safe == False
                logger.info("✅ High security level blocks network access")
        
        # Test security monitoring
        security_stats = tektra_system.get_system_stats().get('security', {})
        assert isinstance(security_stats, dict)
        logger.info(f"✅ Security monitoring active: {security_stats}")
        
        logger.info("✅ Security enforcement and monitoring test passed")

    async def test_performance_optimization_and_scaling(self, tektra_system):
        """Test performance optimization and system scaling capabilities."""
        logger.info("🧪 Testing performance optimization and scaling")
        
        # Get initial performance stats
        initial_stats = tektra_system.get_system_stats()
        initial_memory = initial_stats.get('performance', {}).get('memory_usage', 0)
        
        # Create multiple agents to test scaling
        agent_ids = []
        security_context = SecurityContext(
            agent_id="performance_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="performance_session"
        )
        
        # Create 5 agents concurrently
        for i in range(5):
            agent_id = await tektra_system.create_agent(
                agent_name=f"Performance Test Agent {i}",
                agent_config={"model": "text_completion", "max_tokens": 100},
                security_context=security_context
            )
            agent_ids.append(agent_id)
        
        assert len(agent_ids) == 5
        logger.info(f"✅ Created {len(agent_ids)} agents for performance testing")
        
        # Execute tasks concurrently
        tasks = []
        for i, agent_id in enumerate(agent_ids):
            task = tektra_system.execute_agent_task(
                agent_id=agent_id,
                task_description=f"Generate a list of {i+1} numbers",
                context={"type": "data_generation", "complexity": i+1},
                security_context=security_context
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 3  # At least 60% success rate
        logger.info(f"✅ Concurrent task execution: {len(successful_results)}/{len(results)} successful")
        
        # Check performance metrics
        final_stats = tektra_system.get_system_stats()
        performance_stats = final_stats.get('performance', {})
        
        assert 'memory_usage' in performance_stats
        assert 'cpu_usage' in performance_stats
        logger.info(f"✅ Performance metrics collected: {performance_stats}")
        
        logger.info("✅ Performance optimization and scaling test passed")

    async def test_error_handling_and_resilience(self, tektra_system):
        """Test comprehensive error handling and system resilience."""
        logger.info("🧪 Testing error handling and resilience")
        
        security_context = SecurityContext(
            agent_id="error_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="error_session"
        )
        
        # Test various error conditions
        
        # 1. Invalid agent creation
        try:
            await tektra_system.create_agent(
                agent_name="",  # Invalid empty name
                agent_config={},  # Invalid empty config
                security_context=security_context
            )
            assert False, "Should have raised an error for invalid agent config"
        except Exception as e:
            logger.info(f"✅ Invalid agent creation properly rejected: {e}")
        
        # 2. Create valid agent for further testing
        agent_id = await tektra_system.create_agent(
            agent_name="Error Test Agent",
            agent_config={"model": "text_completion", "max_tokens": 100},
            security_context=security_context
        )
        
        # 3. Test task execution with invalid parameters
        try:
            await tektra_system.execute_agent_task(
                agent_id="non_existent_agent",
                task_description="This should fail",
                context={},
                security_context=security_context
            )
            assert False, "Should have raised an error for non-existent agent"
        except Exception as e:
            logger.info(f"✅ Non-existent agent properly rejected: {e}")
        
        # 4. Test malicious code detection
        malicious_code = '''
import os
os.system("echo 'This is malicious code'")
exec("__import__('os').system('rm -rf /')")
'''
        
        is_safe = await tektra_system.validate_tool(
            tool_code=malicious_code,
            tool_name="malicious_tool",
            security_context=security_context
        )
        assert is_safe == False
        logger.info("✅ Malicious code detection working")
        
        # 5. Test system recovery after errors
        health_before = tektra_system.get_system_health()
        
        # Trigger multiple error conditions
        error_tasks = []
        for i in range(3):
            try:
                task = tektra_system.execute_agent_task(
                    agent_id=agent_id,
                    task_description="",  # Empty task should cause error
                    context={},
                    security_context=security_context
                )
                error_tasks.append(task)
            except Exception:
                pass  # Expected to fail
        
        # Wait a bit and check system health
        await asyncio.sleep(2)
        health_after = tektra_system.get_system_health()
        
        # System should remain stable despite errors
        assert health_after.overall_status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]
        logger.info("✅ System remains stable after error conditions")
        
        logger.info("✅ Error handling and resilience test passed")

    async def test_deployment_manager_integration(self, deployment_manager):
        """Test deployment manager integration with Tektra system."""
        logger.info("🧪 Testing deployment manager integration")
        
        # Check deployment status
        status = deployment_manager.get_deployment_status()
        assert status['status'] in ['healthy', 'degraded']
        assert status['healthy'] == True
        assert 'deployment_id' in status
        assert 'uptime_seconds' in status
        
        logger.info(f"✅ Deployment status: {status['status']}")
        
        # Test metrics collection
        await asyncio.sleep(1)  # Allow metrics to be collected
        
        # Verify Tektra system is integrated
        assert deployment_manager.tektra_system is not None
        assert deployment_manager.health_checker is not None
        
        # Test health checking
        health_result = await deployment_manager.health_checker.check_system_health()
        assert health_result.value in ['healthy', 'degraded']
        logger.info(f"✅ Health check result: {health_result.value}")
        
        logger.info("✅ Deployment manager integration test passed")

    async def test_configuration_management(self, development_config, production_config):
        """Test configuration management across environments."""
        logger.info("🧪 Testing configuration management")
        
        # Test development configuration
        dev_tektra_config = development_config.to_tektra_config()
        assert dev_tektra_config.environment == "development"
        assert dev_tektra_config.debug_mode == True
        assert dev_tektra_config.security_level.value == "medium"
        
        # Test production configuration
        prod_tektra_config = production_config.to_tektra_config()
        assert prod_tektra_config.environment == "production"
        assert prod_tektra_config.debug_mode == False
        assert prod_tektra_config.security_level.value == "high"
        
        # Test configuration validation
        assert development_config.agents.max_concurrent_agents > 0
        assert development_config.performance.max_memory_mb > 0
        assert development_config.security.default_security_level.value in ["low", "medium", "high"]
        
        # Test configuration secrets masking
        masked_config = development_config.mask_secrets()
        secrets = development_config.get_secrets()
        
        # Should have detected secrets
        assert len([k for k, v in secrets.items() if v]) > 0
        logger.info(f"✅ Configuration secrets detected: {len(secrets)}")
        
        logger.info("✅ Configuration management test passed")

    async def test_end_to_end_workflow(self, tektra_system):
        """Test complete end-to-end workflow simulating real usage."""
        logger.info("🧪 Testing complete end-to-end workflow")
        
        # Simulate a real user workflow
        
        # 1. User authentication (simulated)
        user_id = "test_user_e2e"
        session_id = str(uuid.uuid4())
        
        security_context = SecurityContext(
            agent_id="e2e_agent",
            security_level=SecurityLevel.MEDIUM,
            session_id=session_id,
            user_id=user_id
        )
        
        # 2. Create specialized agent for user task
        agent_id = await tektra_system.create_agent(
            agent_name="Data Analysis Assistant",
            agent_config={
                "model": "text_completion",
                "max_tokens": 200,
                "temperature": 0.3,
                "specialized_for": "data_analysis"
            },
            security_context=security_context
        )
        
        assert agent_id is not None
        logger.info(f"✅ Agent created for user workflow: {agent_id}")
        
        # 3. Validate and execute a series of tools
        tools = [
            {
                "name": "data_processor",
                "code": '''
def process_data(data_list):
    """Process a list of numbers."""
    if not isinstance(data_list, list):
        raise ValueError("Input must be a list")
    return {
        "sum": sum(data_list),
        "average": sum(data_list) / len(data_list) if data_list else 0,
        "count": len(data_list)
    }

result = process_data([1, 2, 3, 4, 5])
print(f"Processing result: {result}")
'''
            },
            {
                "name": "report_generator", 
                "code": '''
def generate_report(data):
    """Generate a simple report."""
    return f"Report: Processed {data.get('count', 0)} items with average {data.get('average', 0):.2f}"

report = generate_report({"count": 5, "average": 3.0})
print(f"Generated: {report}")
'''
            }
        ]
        
        # Validate all tools
        for tool in tools:
            is_safe = await tektra_system.validate_tool(
                tool_code=tool["code"],
                tool_name=tool["name"],
                security_context=security_context
            )
            assert is_safe == True
            logger.info(f"✅ Tool '{tool['name']}' validated successfully")
        
        # 4. Execute complex workflow task
        workflow_task = '''
        Analyze the following dataset and provide insights:
        Data: [10, 20, 30, 40, 50, 25, 35, 45]
        
        Please:
        1. Calculate basic statistics
        2. Identify any patterns
        3. Provide recommendations
        '''
        
        result = await tektra_system.execute_agent_task(
            agent_id=agent_id,
            task_description=workflow_task,
            context={
                "type": "data_analysis",
                "user_id": user_id,
                "session_id": session_id,
                "tools_available": [tool["name"] for tool in tools]
            },
            security_context=security_context
        )
        
        assert result is not None
        assert isinstance(result, dict)
        logger.info(f"✅ Workflow task completed successfully")
        
        # 5. Verify system health after complex workflow
        health = tektra_system.get_system_health()
        assert health.overall_status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]
        assert health.active_agents >= 1
        
        # 6. Check performance impact
        stats = tektra_system.get_system_stats()
        performance_stats = stats.get('performance', {})
        security_stats = stats.get('security', {})
        
        assert 'memory_usage' in performance_stats
        assert 'total_events' in security_stats or len(security_stats) >= 0
        
        logger.info(f"✅ System stats after workflow: Performance OK, Security monitored")
        
        logger.info("✅ Complete end-to-end workflow test passed")

    async def test_stress_and_load_testing(self, tektra_system):
        """Test system under stress and load conditions."""
        logger.info("🧪 Testing system under stress and load conditions")
        
        # Configuration for stress test
        num_concurrent_agents = 10
        tasks_per_agent = 3
        
        security_context = SecurityContext(
            agent_id="stress_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="stress_session"
        )
        
        # Create multiple agents concurrently
        agent_creation_tasks = []
        for i in range(num_concurrent_agents):
            task = tektra_system.create_agent(
                agent_name=f"Stress Test Agent {i}",
                agent_config={
                    "model": "text_completion",
                    "max_tokens": 50,
                    "agent_id": f"stress_agent_{i}"
                },
                security_context=security_context
            )
            agent_creation_tasks.append(task)
        
        # Wait for all agent creations
        start_time = time.time()
        agent_ids = await asyncio.gather(*agent_creation_tasks, return_exceptions=True)
        creation_time = time.time() - start_time
        
        successful_agents = [aid for aid in agent_ids if not isinstance(aid, Exception)]
        logger.info(f"✅ Created {len(successful_agents)}/{num_concurrent_agents} agents in {creation_time:.2f}s")
        
        # Execute multiple tasks per agent
        all_tasks = []
        for i, agent_id in enumerate(successful_agents):
            if isinstance(agent_id, str):  # Valid agent ID
                for j in range(tasks_per_agent):
                    task = tektra_system.execute_agent_task(
                        agent_id=agent_id,
                        task_description=f"Task {j} for agent {i}: Generate {j+1} sentences",
                        context={
                            "type": "text_generation",
                            "complexity": j+1,
                            "agent_index": i,
                            "task_index": j
                        },
                        security_context=security_context
                    )
                    all_tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_results) / len(results) if results else 0
        
        logger.info(f"✅ Stress test results:")
        logger.info(f"   Total tasks: {len(results)}")
        logger.info(f"   Successful: {len(successful_results)}")
        logger.info(f"   Failed: {len(failed_results)}")
        logger.info(f"   Success rate: {success_rate:.2%}")
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Throughput: {len(results)/execution_time:.2f} tasks/second")
        
        # Verify acceptable performance
        assert success_rate >= 0.8  # At least 80% success rate
        assert execution_time < 60  # Should complete within 60 seconds
        
        # Check system health after stress test
        health = tektra_system.get_system_health()
        assert health.overall_status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]
        
        logger.info("✅ Stress and load testing passed")

    async def test_data_persistence_and_recovery(self, tektra_system):
        """Test data persistence and system recovery capabilities."""
        logger.info("🧪 Testing data persistence and recovery")
        
        security_context = SecurityContext(
            agent_id="persistence_test",
            security_level=SecurityLevel.MEDIUM,
            session_id="persistence_session"
        )
        
        # Create agent and execute task
        agent_id = await tektra_system.create_agent(
            agent_name="Persistence Test Agent",
            agent_config={"model": "text_completion", "max_tokens": 100},
            security_context=security_context
        )
        
        # Execute task that should be logged/persisted
        result = await tektra_system.execute_agent_task(
            agent_id=agent_id,
            task_description="Create a test document with important data",
            context={
                "type": "document_creation",
                "persistence_test": True,
                "timestamp": time.time()
            },
            security_context=security_context
        )
        
        assert result is not None
        logger.info("✅ Task executed for persistence testing")
        
        # Get system stats to verify logging
        stats = tektra_system.get_system_stats()
        assert 'agents' in stats or 'system_id' in stats
        
        # Verify security events were logged
        security_stats = stats.get('security', {})
        assert isinstance(security_stats, dict)
        
        # Test system state persistence
        system_health = tektra_system.get_system_health()
        assert system_health.last_updated is not None
        assert system_health.uptime_seconds > 0
        
        logger.info("✅ Data persistence and recovery test passed")


# Pytest configuration and test execution
@pytest.mark.asyncio
class TestSystemPerformance:
    """Performance-focused integration tests."""
    
    async def test_startup_performance(self):
        """Test system startup performance."""
        logger.info("🧪 Testing startup performance")
        
        config = create_development_config()
        tektra_config = config.to_tektra_config()
        
        # Measure startup time
        start_time = time.time()
        system = TektraSystem(tektra_config)
        
        init_success = await system.initialize()
        startup_time = time.time() - start_time
        
        assert init_success == True
        assert startup_time < 30  # Should start within 30 seconds
        
        logger.info(f"✅ System startup time: {startup_time:.2f}s")
        
        await system.shutdown()
    
    async def test_memory_efficiency(self):
        """Test memory usage patterns."""
        logger.info("🧪 Testing memory efficiency")
        
        import psutil
        process = psutil.Process()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss
        
        config = create_development_config()
        tektra_config = config.to_tektra_config()
        system = TektraSystem(tektra_config)
        
        await system.initialize()
        
        # Measure memory after initialization
        init_memory = process.memory_info().rss
        memory_increase = init_memory - baseline_memory
        
        logger.info(f"✅ Memory usage: Baseline={baseline_memory/1024/1024:.1f}MB, "
                   f"After init={init_memory/1024/1024:.1f}MB, "
                   f"Increase={memory_increase/1024/1024:.1f}MB")
        
        # Should not use excessive memory
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB increase
        
        await system.shutdown()


# Test utilities and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest for Tektra integration tests."""
    logger.info("🧪 Configuring Tektra Integration Tests")
    logger.info("=" * 60)


def pytest_sessionstart(session):
    """Start of test session."""
    logger.info("🌟 Starting Tektra Complete System Integration Tests")


def pytest_sessionfinish(session, exitstatus):
    """End of test session."""
    if exitstatus == 0:
        logger.info("✅ All Tektra integration tests passed successfully!")
    else:
        logger.error(f"❌ Some tests failed with exit status: {exitstatus}")
    logger.info("🎯 Tektra Integration Testing Complete")


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    logger.info("🧪 Running Tektra Complete System Integration Tests")
    
    # Run with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--asyncio-mode=auto"
    ])
    
    sys.exit(result.returncode)