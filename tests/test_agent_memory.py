#!/usr/bin/env python3
"""
Test Agent Memory Integration

This script tests the enhanced agent builder with memory configuration.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_agent_memory_configuration(agent_builder, memory_manager):
    """Test that agent builder properly configures memory settings."""
    if agent_builder is None or memory_manager is None:
        pytest.skip("Agent builder or memory manager not available")

    print("🧠 Testing Agent Memory Configuration...")

    try:
        # Test 1: Memory-enabled agent
        description1 = "Create a coding assistant that remembers my preferred coding style and past projects"
        spec1 = await agent_builder.create_agent_from_description(description1)

        assert spec1 is not None, "Agent specification should be created"
        assert spec1.name, "Agent should have a name"

        print(f"✅ Created memory-enabled agent: {spec1.name}")
        if hasattr(spec1, "memory_enabled"):
            print(f"   Memory enabled: {spec1.memory_enabled}")

        # Test basic memory operations
        from tektra.memory.memory_types import MemoryEntry, MemoryType

        test_entry = MemoryEntry(
            id="agent_test_001",
            content="Test memory for agent",
            type=MemoryType.AGENT_CONTEXT,
            importance=0.7,
            agent_id="test_agent",
        )

        # Store and retrieve memory
        memory_id = await memory_manager.add_memory(test_entry)
        retrieved = await memory_manager.get_memory(memory_id)

        assert retrieved is not None, "Memory should be retrievable"
        assert retrieved.content == test_entry.content, "Content should match"

        print("✅ Agent memory integration test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        pytest.fail(f"Agent memory configuration failed: {e}")


@pytest.mark.asyncio
async def test_agent_memory_basic_functionality(memory_manager):
    """Test basic agent memory functionality without requiring heavy models."""
    if memory_manager is None:
        pytest.skip("Memory manager not available")

    print("🔍 Testing basic agent memory functionality...")

    from tektra.memory.memory_types import MemoryEntry, MemoryType

    # Test agent-specific memory storage
    agent_memory = MemoryEntry(
        id="agent_001",
        content="Agent learned user prefers Python for automation scripts",
        type=MemoryType.AGENT_CONTEXT,
        importance=0.8,
        agent_id="coding_assistant",
    )

    # Store agent memory
    memory_id = await memory_manager.add_memory(agent_memory)
    assert memory_id == agent_memory.id, "Memory ID should match"

    # Retrieve agent memory
    retrieved = await memory_manager.get_memory(memory_id)
    assert retrieved.type == MemoryType.AGENT_CONTEXT, "Should be agent context type"
    assert retrieved.agent_id == "coding_assistant", "Agent ID should match"

    print("✅ Basic agent memory functionality test completed")
