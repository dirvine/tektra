#!/usr/bin/env python3
"""
Test Qwen Memory Integration

This script tests the memory integration with Qwen backend.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_qwen_memory_integration(qwen_backend, memory_manager):
    """Test Qwen backend with memory integration."""
    if qwen_backend is None or memory_manager is None:
        pytest.skip("Qwen backend or memory manager not available")

    print("🧠 Testing Qwen Memory Integration...")

    try:
        # Both fixtures are already initialized, just test the integration
        print("✅ Memory manager and Qwen backend available from fixtures")

        # Test that we can store and retrieve a memory entry
        from tektra.memory.memory_types import MemoryEntry, MemoryType

        test_entry = MemoryEntry(
            id="test_qwen_001",
            content="Test memory for Qwen integration",
            type=MemoryType.CONVERSATION,
            importance=0.8,
            user_id="test_user",
        )

        # Store memory
        memory_id = await memory_manager.add_memory(test_entry)
        assert memory_id == test_entry.id, "Memory ID should match"
        print("✅ Memory stored successfully")

        # Retrieve memory
        retrieved = await memory_manager.get_memory(memory_id)
        assert retrieved is not None, "Memory should be retrievable"
        assert retrieved.content == test_entry.content, "Content should match"
        print("✅ Memory retrieved successfully")

        # Test basic Qwen functionality (if not mocked)
        if hasattr(qwen_backend, "process_text_query"):
            response = await qwen_backend.process_text_query("What is 2+2?")
            assert response is not None, "Qwen should return a response"
            print(f"✅ Qwen response: {str(response)[:50]}...")

        print("🎉 Qwen-Memory integration test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        pytest.fail(f"Qwen memory integration failed: {e}")


@pytest.mark.asyncio
async def test_qwen_backend_basic_functionality(qwen_backend):
    """Test basic Qwen backend functionality without memory."""
    if qwen_backend is None:
        pytest.skip("Qwen backend not available")

    print("🔍 Testing basic Qwen functionality...")

    # Test simple text query
    if hasattr(qwen_backend, "process_text_query"):
        response = await qwen_backend.process_text_query("Hello, how are you?")
        assert response is not None, "Should get a response"
        print(f"✅ Text query response: {str(response)[:50]}...")

    # Test if backend reports as initialized
    if hasattr(qwen_backend, "is_initialized"):
        assert qwen_backend.is_initialized, "Backend should be initialized"
        print("✅ Backend is properly initialized")

    print("✅ Basic Qwen functionality test completed")
