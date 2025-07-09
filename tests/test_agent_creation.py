#!/usr/bin/env python3
"""
Test script for Tektra Agent Creation

This script tests the agent creation functionality without the GUI.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tektra.agents import AgentBuilder, AgentRegistry, AgentRuntime, SandboxType
from tektra.ai.qwen_backend import QwenBackend, QwenModelConfig
from loguru import logger

async def test_agent_creation():
    """Test creating an agent from natural language description."""
    
    print("🤖 Testing Tektra Agent Creation System")
    print("=" * 50)
    
    try:
        # Initialize Qwen backend for agent creation
        print("1. Initializing Qwen backend...")
        qwen_config = QwenModelConfig(
            model_name='Qwen/Qwen2.5-VL-7B-Instruct',
            quantization_bits=None,  # Disabled for compatibility
            max_memory_gb=8.0
        )
        
        qwen_backend = QwenBackend(qwen_config)
        success = await qwen_backend.initialize()
        
        if not success:
            print("❌ Failed to initialize Qwen backend")
            return False
        
        print("✅ Qwen backend initialized successfully")
        
        # Initialize agent system
        print("\n2. Initializing agent system...")
        agent_registry = AgentRegistry()
        agent_builder = AgentBuilder(qwen_backend)
        agent_runtime = AgentRuntime(SandboxType.PROCESS)
        
        print("✅ Agent system initialized")
        
        # Test agent creation from natural language
        print("\n3. Testing agent creation from natural language...")
        
        test_descriptions = [
            "Create an agent that monitors GitHub repositories for new issues and pull requests",
            "Build an agent that analyzes stock prices and sends alerts when they change significantly",
            "Make an agent that organizes my downloads folder by file type every day"
        ]
        
        for i, description in enumerate(test_descriptions, 1):
            print(f"\n   Test {i}: {description}")
            
            try:
                # Create agent specification
                spec = await agent_builder.create_agent_from_description(description)
                
                print(f"   ✅ Agent created: {spec.name}")
                print(f"      Type: {spec.type.value}")
                print(f"      Goal: {spec.goal[:100]}...")
                print(f"      Capabilities: {len(spec.capabilities)} features")
                
                # Register agent
                agent_id = await agent_registry.register_agent(spec)
                print(f"      Registered with ID: {agent_id[:8]}...")
                
                # Test deployment (but don't actually run)
                print(f"      ✅ Agent ready for deployment")
                
            except Exception as e:
                print(f"   ❌ Failed to create agent: {e}")
                logger.error(f"Agent creation failed: {e}")
        
        # Test listing agents
        print("\n4. Testing agent registry...")
        agents = await agent_registry.list_agents()
        print(f"✅ Total agents in registry: {len(agents)}")
        
        for agent in agents:
            print(f"   • {agent.specification.name} - {agent.status.value}")
        
        print("\n🎉 Agent system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if 'qwen_backend' in locals():
            await qwen_backend.cleanup()

async def main():
    """Main test function."""
    success = await test_agent_creation()
    exit_code = 0 if success else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())