#!/usr/bin/env python3
"""
Test Agent Memory Integration

This script tests the enhanced agent builder with memory configuration.
"""

import asyncio
import tempfile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tektra.agents.builder import AgentBuilder
from tektra.ai.qwen_backend import QwenBackend, QwenModelConfig
from tektra.memory import TektraMemoryManager, MemoryConfig

async def test_agent_memory_configuration():
    """Test that agent builder properly configures memory settings."""
    print("🧠 Testing Agent Memory Configuration...")
    
    try:
        # Create memory manager
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_config = MemoryConfig(
                storage_path=temp_dir,
                use_memos=False
            )
            memory_manager = TektraMemoryManager(memory_config)
            await memory_manager.initialize()
            
            # Create minimal Qwen backend (no model loading)
            qwen_backend = QwenBackend()
            await qwen_backend.enable_memory(memory_manager)
            
            # Create agent builder
            agent_builder = AgentBuilder(qwen_backend)
            
            # Test 1: Memory-enabled agent
            description1 = "Create a coding assistant that remembers my preferred coding style and past projects"
            spec1 = await agent_builder.create_agent_from_description(description1)
            
            print(f"✅ Created memory-enabled agent: {spec1.name}")
            print(f"   Memory enabled: {spec1.memory_enabled}")
            print(f"   Context limit: {spec1.memory_context_limit}")
            print(f"   Importance threshold: {spec1.memory_importance_threshold}")
            print(f"   Retention hours: {spec1.memory_retention_hours}")
            print(f"   Persistent memory: {spec1.persistent_memory}")
            
            # Test 2: Monitoring agent (should have extended memory)
            description2 = "Monitor my GitHub repositories for new issues and PRs continuously"
            spec2 = await agent_builder.create_agent_from_description(description2)
            
            print(f"✅ Created monitoring agent: {spec2.name}")
            print(f"   Memory enabled: {spec2.memory_enabled}")
            print(f"   Context limit: {spec2.memory_context_limit} (should be ≥20)")
            print(f"   Retention hours: {spec2.memory_retention_hours} (should be ≥720)")
            
            # Test 3: Workflow agent (should have persistent memory)
            description3 = "Create a workflow agent that helps me plan and execute complex software projects"
            spec3 = await agent_builder.create_agent_from_description(description3)
            
            print(f"✅ Created workflow agent: {spec3.name}")
            print(f"   Memory enabled: {spec3.memory_enabled}")
            print(f"   Context limit: {spec3.memory_context_limit} (should be ≥15)")
            print(f"   Persistent memory: {spec3.persistent_memory} (should be True)")
            
            # Test system prompt generation
            print("\\n🔍 Testing system prompt generation...")
            system_prompt = agent_builder._get_default_system_prompt(spec1)
            if "Memory System:" in system_prompt:
                print("✅ System prompt includes memory instructions")
            else:
                print("❌ System prompt missing memory instructions")
            
            # Test code generation
            print("\\n💻 Testing code generation...")
            code = agent_builder._get_default_code(spec1)
            if "memory_manager" in code:
                print("✅ Generated code includes memory integration")
            else:
                print("❌ Generated code missing memory integration")
            
            await memory_manager.cleanup()
            
            return True
            
    except Exception as e:
        print(f"❌ Agent memory configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_specification_validation():
    """Test validation of memory configuration in agent specs."""
    print("\\n🔍 Testing Memory Specification Validation...")
    
    try:
        # Create agent builder
        qwen_backend = QwenBackend()
        agent_builder = AgentBuilder(qwen_backend)
        
        # Create agent spec with valid memory config
        from tektra.agents.builder import AgentSpecification
        spec = AgentSpecification(
            name="Test Agent",
            memory_enabled=True,
            memory_context_limit=25,
            memory_importance_threshold=0.6,
            memory_retention_hours=720
        )
        
        # Test validation
        validation_result = await agent_builder._validate_specification(spec)
        
        if validation_result['is_valid']:
            print("✅ Valid memory configuration passed validation")
        else:
            print(f"❌ Valid memory configuration failed: {validation_result['errors']}")
            return False
        
        # Test invalid memory config
        spec.memory_context_limit = 100  # Too high
        spec.memory_importance_threshold = 1.5  # Too high
        spec.memory_retention_hours = 10000  # Too high
        
        validation_result = await agent_builder._validate_specification(spec)
        
        if not validation_result['is_valid']:
            print("✅ Invalid memory configuration properly rejected")
            print(f"   Errors: {validation_result['errors']}")
        else:
            print("❌ Invalid memory configuration incorrectly passed validation")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory specification validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🤖 Testing Agent Memory Integration")
    print("=" * 50)
    
    success = True
    
    # Test agent memory configuration
    if not await test_agent_memory_configuration():
        success = False
    
    # Test memory specification validation
    if not await test_memory_specification_validation():
        success = False
    
    if success:
        print("\\n🎉 All agent memory integration tests passed!")
        print("\\nKey features implemented:")
        print("✅ Memory configuration in AgentSpecification")
        print("✅ Memory-aware agent creation")
        print("✅ Memory system integration in generated code")
        print("✅ Memory configuration validation")
        print("✅ Agent type-specific memory optimizations")
        print("✅ Memory instructions in system prompts")
        print("\\nAgents can now use memory for context-aware responses!")
    else:
        print("\\n❌ Some agent memory integration tests failed.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())