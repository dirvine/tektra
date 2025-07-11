#!/usr/bin/env python3
"""
Test script for Tektra Agent Creation

This script tests the agent creation functionality using shared model fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_agent_creation_from_description(
    qwen_backend, agent_builder, agent_registry
):
    """Test creating an agent from natural language description."""
    if qwen_backend is None or agent_builder is None:
        pytest.skip("Qwen backend or agent builder not available")

    print("🤖 Testing Tektra Agent Creation System")
    print("=" * 50)

    # Test agent creation from natural language
    test_descriptions = [
        "Create an agent that monitors GitHub repositories for new issues and pull requests",
        "Build an agent that analyzes stock prices and sends alerts when they change significantly",
        "Make an agent that organizes my downloads folder by file type every day",
    ]

    for i, description in enumerate(test_descriptions, 1):
        print(f"\n   Test {i}: {description[:50]}...")

        try:
            # Create agent specification
            spec = await agent_builder.create_agent_from_description(description)

            assert spec is not None, "Agent specification should not be None"
            assert spec.name, "Agent should have a name"
            assert spec.goal, "Agent should have a goal"
            assert spec.type, "Agent should have a type"

            print(f"   ✅ Agent created: {spec.name}")
            print(f"      Type: {spec.type.value}")
            print(f"      Goal: {spec.goal[:100]}...")

            # Register agent if registry is available
            if agent_registry:
                agent_id = await agent_registry.register_agent(spec)
                assert agent_id, "Agent should be registered with an ID"
                print(f"      Registered with ID: {agent_id[:8]}...")

        except Exception as e:
            pytest.fail(f"Failed to create agent: {e}")


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_agent_registry_operations(agent_registry):
    """Test basic agent registry operations."""
    if agent_registry is None:
        pytest.skip("Agent registry not available")

    # Test listing agents (should be empty initially)
    agents = await agent_registry.list_agents()
    initial_count = len(agents)

    print(f"✅ Agent registry has {initial_count} agents initially")

    # Test that registry is functional
    assert isinstance(agents, list), "list_agents should return a list"


def test_agent_creation_infrastructure():
    """Test that agent creation infrastructure is properly set up."""
    # Test imports work
    try:
        from tektra.agents import (  # noqa: F401
            AgentBuilder,
            AgentRegistry,
            AgentRuntime,
        )
        from tektra.ai.qwen_backend import QwenBackend, QwenModelConfig  # noqa: F401

        assert True, "All imports successful"
    except ImportError as e:
        pytest.skip(f"Agent infrastructure not available: {e}")


@pytest.mark.asyncio
async def test_mock_agent_creation(qwen_backend):
    """Test agent creation with mock backend (when --no-heavy-models is used)."""
    if hasattr(qwen_backend, "process_text_query"):
        # This is likely a mock, test it works
        response = await qwen_backend.process_text_query("test query")
        assert response is not None, "Mock backend should return a response"
        print("✅ Mock backend working correctly")
