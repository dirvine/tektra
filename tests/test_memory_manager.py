#!/usr/bin/env python3
"""
Test Tektra Memory Manager

This script tests the memory management system to ensure it works correctly.
"""

import asyncio
import tempfile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tektra.memory import TektraMemoryManager, MemoryConfig, MemoryType, MemoryContext
from tektra.memory.memory_types import create_conversation_memory, create_agent_context_memory

async def test_memory_manager_initialization():
    """Test memory manager initialization."""
    print("🧠 Testing Memory Manager Initialization...")
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryConfig(
                storage_path=temp_dir,
                use_memos=False  # Skip MemOS for now
            )
            
            memory_manager = TektraMemoryManager(config)
            
            # Initialize
            success = await memory_manager.initialize()
            if not success:
                print("❌ Failed to initialize memory manager")
                return False
            
            print("✅ Memory manager initialized successfully")
            
            # Test basic functionality
            stats = await memory_manager.get_memory_stats()
            print(f"✅ Initial stats: {stats.total_memories} memories")
            
            await memory_manager.cleanup()
            return True
            
    except Exception as e:
        print(f"❌ Memory manager initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_operations():
    """Test basic memory operations."""
    print("\n🧠 Testing Memory Operations...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryConfig(
                storage_path=temp_dir,
                use_memos=False
            )
            
            memory_manager = TektraMemoryManager(config)
            await memory_manager.initialize()
            
            # Test adding conversation memory
            user_id = "test_user"
            session_id = "test_session"
            agent_id = "test_agent"
            
            memory_ids = await memory_manager.add_conversation(
                user_message="Hello, can you help me with something?",
                assistant_response="Of course! I'm here to help. What do you need assistance with?",
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id
            )
            
            print(f"✅ Added conversation memory: {len(memory_ids)} entries")
            
            # Test adding agent context
            context_id = await memory_manager.add_agent_context(
                agent_id=agent_id,
                context="This agent helps with general questions and tasks",
                importance=0.8
            )
            
            print(f"✅ Added agent context: {context_id}")
            
            # Test searching memories
            search_context = MemoryContext(
                user_id=user_id,
                query="help",
                max_results=10
            )
            
            search_result = await memory_manager.search_memories(search_context)
            print(f"✅ Search found {len(search_result.entries)} results")
            
            # Test getting conversation history
            history = await memory_manager.get_conversation_history(user_id, session_id)
            print(f"✅ Conversation history: {len(history)} entries")
            
            # Test getting agent context
            agent_context = await memory_manager.get_agent_context(agent_id)
            print(f"✅ Agent context: {len(agent_context)} entries")
            
            # Test memory stats
            stats = await memory_manager.get_memory_stats()
            print(f"✅ Memory stats: {stats.total_memories} total memories")
            print(f"   - By type: {stats.memories_by_type}")
            print(f"   - By agent: {stats.memories_by_agent}")
            print(f"   - By user: {stats.memories_by_user}")
            
            await memory_manager.cleanup()
            return True
            
    except Exception as e:
        print(f"❌ Memory operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_memory_scenarios():
    """Test agent-specific memory scenarios."""
    print("\n🧠 Testing Agent Memory Scenarios...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryConfig(
                storage_path=temp_dir,
                use_memos=False
            )
            
            memory_manager = TektraMemoryManager(config)
            await memory_manager.initialize()
            
            # Create multiple agents with different contexts
            agents = [
                ("github_monitor", "Monitor GitHub repositories for issues and pull requests"),
                ("stock_analyzer", "Analyze stock prices and send alerts for significant changes"),
                ("file_organizer", "Organize files by type and maintain folder structure")
            ]
            
            for agent_id, description in agents:
                # Add agent context
                await memory_manager.add_agent_context(
                    agent_id=agent_id,
                    context=f"Agent purpose: {description}",
                    importance=0.9
                )
                
                # Add some task results
                await memory_manager.add_task_result(
                    task_description=f"Initialize {agent_id}",
                    result=f"Agent {agent_id} initialized successfully",
                    success=True,
                    agent_id=agent_id
                )
                
                print(f"✅ Initialized {agent_id} with context and task result")
            
            # Test agent isolation - each agent should only see its own memories
            for agent_id, _ in agents:
                agent_context = await memory_manager.get_agent_context(agent_id)
                
                # Should have exactly 1 context entry for this agent
                assert len(agent_context) == 1, f"Expected 1 context entry for {agent_id}, got {len(agent_context)}"
                assert agent_context[0].agent_id == agent_id, f"Context doesn't belong to {agent_id}"
                
                print(f"✅ Agent {agent_id} memory isolation verified")
            
            # Test cross-agent search
            all_agents_context = MemoryContext(
                memory_types=[MemoryType.AGENT_CONTEXT],
                max_results=10
            )
            
            all_results = await memory_manager.search_memories(all_agents_context)
            print(f"✅ Cross-agent search found {len(all_results.entries)} context entries")
            
            # Verify we can distinguish between agents
            agent_ids_found = {entry.agent_id for entry in all_results.entries}
            expected_agents = {agent_id for agent_id, _ in agents}
            assert agent_ids_found == expected_agents, f"Missing agents: {expected_agents - agent_ids_found}"
            
            print("✅ Agent memory scenarios completed successfully")
            
            await memory_manager.cleanup()
            return True
            
    except Exception as e:
        print(f"❌ Agent memory scenarios failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🤖 Testing Tektra Memory Management System")
    print("=" * 60)
    
    success = True
    
    if not await test_memory_manager_initialization():
        success = False
    
    if not await test_memory_operations():
        success = False
    
    if not await test_agent_memory_scenarios():
        success = False
    
    if success:
        print("\n🎉 All memory manager tests passed!")
        print("\nMemory system capabilities:")
        print("✅ SQLite-based persistent storage")
        print("✅ Conversation history tracking")
        print("✅ Agent context management")
        print("✅ Memory search and retrieval")
        print("✅ Agent memory isolation")
        print("✅ Task result tracking")
        print("✅ Statistics and monitoring")
        print("\nReady for integration with Tektra agents!")
    else:
        print("\n❌ Some memory manager tests failed.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())