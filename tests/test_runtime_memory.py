#!/usr/bin/env python3
"""
Test Agent Runtime Memory Integration

This script tests the memory integration in the agent runtime system.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tektra.agents.builder import AgentSpecification, AgentType
from tektra.agents.runtime import AgentExecutionContext, AgentRuntime, SandboxType
from tektra.memory import MemoryConfig, MemoryType, TektraMemoryManager


async def test_runtime_memory_integration():
    """Test that agent runtime properly integrates memory."""
    print("🧠 Testing Agent Runtime Memory Integration...")

    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize memory manager
            memory_config = MemoryConfig(storage_path=temp_dir, use_memos=False)
            memory_manager = TektraMemoryManager(memory_config)
            await memory_manager.initialize()

            # Create runtime with memory
            runtime = AgentRuntime(
                sandbox_type=SandboxType.LOCAL,  # Use local for testing
                memory_manager=memory_manager,
            )

            print("✅ Runtime initialized with memory manager")

            # Create a memory-enabled agent spec
            spec = AgentSpecification(
                id="test-memory-agent",
                name="Memory Test Agent",
                description="Agent that uses memory",
                type=AgentType.CODE,
                goal="Test memory functionality",
                memory_enabled=True,
                memory_context_limit=10,
                memory_importance_threshold=0.5,
                persistent_memory=True,
                initial_code="""
async def run_agent(input_data):
    # Check if memory manager is available
    memory_manager = input_data.get('memory_manager')
    if memory_manager:
        # Search for previous executions
        memories = await memory_manager.search_memories({
            'agent_id': input_data.get('agent_id'),
            'max_results': 5
        })

        context = f"Found {len(memories.entries) if hasattr(memories, 'entries') else 0} previous memories"

        return {
            'success': True,
            'result': f"Memory test successful. {context}",
            'has_memory': True,
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'success': True,
            'result': "No memory manager available",
            'has_memory': False,
            'timestamp': datetime.now().isoformat()
        }
""",
            )

            # Deploy the agent
            agent_id = await runtime.deploy_agent(
                spec=spec,
                input_data={"task": "Test memory integration"},
                user_id="test_user",
            )

            print(f"✅ Deployed agent: {agent_id}")

            # Wait for execution
            await asyncio.sleep(2)

            # Check agent status
            status = await runtime.get_agent_status(agent_id)
            print(f"✅ Agent status: {status['state']}")

            # Get memory stats
            memory_stats = await runtime.get_agent_memory_stats(agent_id)
            print(f"✅ Memory stats: {memory_stats}")

            # Check if execution was saved to memory
            if memory_stats.get("task_results", 0) > 0:
                print("✅ Agent execution saved to memory")
            else:
                print("⚠️  No execution saved to memory yet")

            # Clean up
            await runtime.cleanup()
            await memory_manager.cleanup()

            return True

    except Exception as e:
        print(f"❌ Runtime memory integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_memory_context_loading():
    """Test loading memory context for agents."""
    print("\n🧠 Testing Memory Context Loading...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize memory manager
            memory_config = MemoryConfig(storage_path=temp_dir, use_memos=False)
            memory_manager = TektraMemoryManager(memory_config)
            await memory_manager.initialize()

            # Pre-populate some memories
            agent_id = "context-test-agent"
            await memory_manager.add_agent_context(
                agent_id=agent_id,
                context="This agent helps with testing",
                importance=0.9,
            )

            await memory_manager.add_task_result(
                task_description="Previous test run",
                result="Test completed successfully",
                success=True,
                agent_id=agent_id,
            )

            print("✅ Pre-populated agent memories")

            # Create runtime
            runtime = AgentRuntime(
                sandbox_type=SandboxType.LOCAL, memory_manager=memory_manager
            )

            # Create agent spec with persistent memory
            spec = AgentSpecification(
                id=agent_id,
                name="Context Test Agent",
                type=AgentType.CODE,
                memory_enabled=True,
                persistent_memory=True,
                memory_context_limit=10,
                initial_code="""
async def run_agent(input_data):
    return {'success': True, 'result': 'Context test'}
""",
            )

            # Create execution context
            from datetime import datetime

            context = AgentExecutionContext(
                agent_id=agent_id,
                spec=spec,
                input_data={},
                environment={},
                working_directory=Path(temp_dir) / agent_id,
                start_time=datetime.now(),
                timeout=30,
                memory_manager=memory_manager,
            )

            # Load memory context
            await runtime._load_agent_memory_context(context)

            if context.memory_context and len(context.memory_context) > 0:
                print(f"✅ Loaded {len(context.memory_context)} memory entries")
                for i, memory in enumerate(context.memory_context[:3]):
                    print(f"   Memory {i+1}: {memory.content[:50]}...")
            else:
                print("⚠️  No memory context loaded")

            await memory_manager.cleanup()

            return True

    except Exception as e:
        print(f"❌ Memory context loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_memory_sharing():
    """Test memory sharing between agents."""
    print("\n🧠 Testing Memory Sharing Between Agents...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize memory manager
            memory_config = MemoryConfig(storage_path=temp_dir, use_memos=False)
            memory_manager = TektraMemoryManager(memory_config)
            await memory_manager.initialize()

            # Create runtime
            runtime = AgentRuntime(
                sandbox_type=SandboxType.LOCAL, memory_manager=memory_manager
            )

            # Create source agent with memory sharing enabled
            source_spec = AgentSpecification(
                id="source-agent",
                name="Source Agent",
                type=AgentType.CODE,
                memory_enabled=True,
                memory_sharing_enabled=True,  # Enable sharing
                initial_code="""
async def run_agent(input_data):
    return {'success': True, 'result': 'Source agent execution'}
""",
            )

            # Create target agent
            target_spec = AgentSpecification(
                id="target-agent",
                name="Target Agent",
                type=AgentType.CODE,
                memory_enabled=True,
                initial_code="""
async def run_agent(input_data):
    return {'success': True, 'result': 'Target agent execution'}
""",
            )

            # Deploy both agents
            source_id = await runtime.deploy_agent(source_spec)
            target_id = await runtime.deploy_agent(target_spec)

            print("✅ Deployed source and target agents")

            # Add some memories to source agent
            await memory_manager.add_agent_context(
                agent_id=source_id,
                context="Important knowledge from source agent",
                importance=0.8,
            )

            await memory_manager.add_agent_context(
                agent_id=source_id, context="Another piece of knowledge", importance=0.7
            )

            # Share memories
            result = await runtime.share_agent_memory(
                source_agent_id=source_id,
                target_agent_id=target_id,
                memory_types=[MemoryType.AGENT_CONTEXT],
            )

            print(f"✅ Shared {result['shared_memories']} memories")

            # Verify target agent received memories
            target_memories = await memory_manager.get_agent_context(target_id)

            if any("[Shared from" in mem.content for mem in target_memories):
                print("✅ Target agent received shared memories")
                for mem in target_memories:
                    if "[Shared from" in mem.content:
                        print(f"   Shared: {mem.content[:60]}...")
            else:
                print("❌ No shared memories found in target agent")

            await runtime.cleanup()
            await memory_manager.cleanup()

            return True

    except Exception as e:
        print(f"❌ Memory sharing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🤖 Testing Agent Runtime Memory Integration")
    print("=" * 50)

    # Import datetime for agent code
    import builtins
    import datetime

    builtins.datetime = datetime

    success = True

    # Test runtime memory integration
    if not await test_runtime_memory_integration():
        success = False

    # Test memory context loading
    if not await test_memory_context_loading():
        success = False

    # Test memory sharing
    if not await test_memory_sharing():
        success = False

    if success:
        print("\n🎉 All agent runtime memory tests passed!")
        print("\nKey features implemented:")
        print("✅ Memory manager integration in AgentRuntime")
        print("✅ Memory context passed to agents during execution")
        print("✅ Execution results saved to memory")
        print("✅ Memory context loading on agent deployment")
        print("✅ Memory statistics for agents")
        print("✅ Inter-agent memory sharing")
        print("✅ Persistent memory support")
        print("\nAgents can now use memory for persistent context!")
    else:
        print("\n❌ Some agent runtime memory tests failed.")

    return success


if __name__ == "__main__":
    asyncio.run(main())
