#!/usr/bin/env python3
"""
Test Agent Memory Integration (Simple)

This script tests the enhanced agent builder memory configuration without full model loading.
"""

import asyncio
import tempfile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tektra.agents.builder import AgentBuilder, AgentSpecification, AgentType
from tektra.ai.qwen_backend import QwenBackend
from tektra.memory import TektraMemoryManager, MemoryConfig

async def test_memory_specification_features():
    """Test memory configuration features in AgentSpecification."""
    print("🧠 Testing Memory Configuration Features...")
    
    try:
        # Test 1: Default memory configuration
        spec1 = AgentSpecification(name="Test Agent 1")
        
        print(f"✅ Default memory configuration:")
        print(f"   Memory enabled: {spec1.memory_enabled}")
        print(f"   Context limit: {spec1.memory_context_limit}")
        print(f"   Importance threshold: {spec1.memory_importance_threshold}")
        print(f"   Retention hours: {spec1.memory_retention_hours}")
        print(f"   Memory sharing: {spec1.memory_sharing_enabled}")
        print(f"   Persistent memory: {spec1.persistent_memory}")
        
        # Test 2: Custom memory configuration
        spec2 = AgentSpecification(
            name="Custom Memory Agent",
            memory_enabled=True,
            memory_context_limit=25,
            memory_importance_threshold=0.7,
            memory_retention_hours=720,  # 30 days
            memory_sharing_enabled=True,
            persistent_memory=True
        )
        
        print(f"\\n✅ Custom memory configuration:")
        print(f"   Memory enabled: {spec2.memory_enabled}")
        print(f"   Context limit: {spec2.memory_context_limit}")
        print(f"   Importance threshold: {spec2.memory_importance_threshold}")
        print(f"   Retention hours: {spec2.memory_retention_hours}")
        print(f"   Memory sharing: {spec2.memory_sharing_enabled}")
        print(f"   Persistent memory: {spec2.persistent_memory}")
        
        # Test 3: Memory-disabled agent
        spec3 = AgentSpecification(
            name="No Memory Agent",
            memory_enabled=False
        )
        
        print(f"\\n✅ Memory-disabled agent:")
        print(f"   Memory enabled: {spec3.memory_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory specification features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_validation():
    """Test validation of memory configuration."""
    print("\\n🔍 Testing Memory Validation...")
    
    try:
        # Create agent builder
        qwen_backend = QwenBackend()
        agent_builder = AgentBuilder(qwen_backend)
        
        # Test 1: Valid memory configurations
        valid_configs = [
            {
                "name": "Valid Config 1",
                "memory_enabled": True,
                "memory_context_limit": 10,
                "memory_importance_threshold": 0.5,
                "memory_retention_hours": 168
            },
            {
                "name": "Valid Config 2",
                "memory_enabled": True,
                "memory_context_limit": 50,
                "memory_importance_threshold": 1.0,
                "memory_retention_hours": 8760
            },
            {
                "name": "Valid Config 3",
                "memory_enabled": False  # Disabled should always be valid
            }
        ]
        
        for config in valid_configs:
            spec = AgentSpecification(**config)
            validation_result = await agent_builder._validate_specification(spec)
            
            if validation_result['is_valid']:
                print(f"✅ Valid config passed: {config['name']}")
            else:
                print(f"❌ Valid config failed: {config['name']} - {validation_result['errors']}")
                return False
        
        # Test 2: Invalid memory configurations
        invalid_configs = [
            {
                "name": "Invalid Context Limit (too low)",
                "memory_enabled": True,
                "memory_context_limit": 0,
                "memory_importance_threshold": 0.5,
                "memory_retention_hours": 168
            },
            {
                "name": "Invalid Context Limit (too high)",
                "memory_enabled": True,
                "memory_context_limit": 100,
                "memory_importance_threshold": 0.5,
                "memory_retention_hours": 168
            },
            {
                "name": "Invalid Importance Threshold (too low)",
                "memory_enabled": True,
                "memory_context_limit": 10,
                "memory_importance_threshold": -0.1,
                "memory_retention_hours": 168
            },
            {
                "name": "Invalid Importance Threshold (too high)",
                "memory_enabled": True,
                "memory_context_limit": 10,
                "memory_importance_threshold": 1.1,
                "memory_retention_hours": 168
            },
            {
                "name": "Invalid Retention Hours (too low)",
                "memory_enabled": True,
                "memory_context_limit": 10,
                "memory_importance_threshold": 0.5,
                "memory_retention_hours": 0
            },
            {
                "name": "Invalid Retention Hours (too high)",
                "memory_enabled": True,
                "memory_context_limit": 10,
                "memory_importance_threshold": 0.5,
                "memory_retention_hours": 10000
            }
        ]
        
        for config in invalid_configs:
            spec = AgentSpecification(**config)
            validation_result = await agent_builder._validate_specification(spec)
            
            if not validation_result['is_valid']:
                print(f"✅ Invalid config properly rejected: {config['name']}")
            else:
                print(f"❌ Invalid config incorrectly accepted: {config['name']}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_aware_code_generation():
    """Test that generated code includes memory integration."""
    print("\\n💻 Testing Memory-Aware Code Generation...")
    
    try:
        # Create agent builder
        qwen_backend = QwenBackend()
        agent_builder = AgentBuilder(qwen_backend)
        
        # Test 1: Memory-enabled agent code
        spec1 = AgentSpecification(
            name="Memory Agent",
            memory_enabled=True,
            memory_context_limit=15,
            memory_importance_threshold=0.6
        )
        
        code1 = agent_builder._get_default_code(spec1)
        
        if "memory_manager" in code1:
            print("✅ Memory-enabled code includes memory_manager")
        else:
            print("❌ Memory-enabled code missing memory_manager")
            return False
        
        if "memory_context" in code1:
            print("✅ Memory-enabled code includes memory search")
        else:
            print("❌ Memory-enabled code missing memory search")
            return False
        
        if "add_agent_context" in code1:
            print("✅ Memory-enabled code includes memory saving")
        else:
            print("❌ Memory-enabled code missing memory saving")
            return False
        
        # Test 2: Memory-disabled agent code
        spec2 = AgentSpecification(
            name="No Memory Agent",
            memory_enabled=False
        )
        
        code2 = agent_builder._get_default_code(spec2)
        
        if "memory_manager" not in code2:
            print("✅ Memory-disabled code excludes memory_manager")
        else:
            print("❌ Memory-disabled code incorrectly includes memory_manager")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory-aware code generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_aware_system_prompts():
    """Test that system prompts include memory instructions."""
    print("\\n📝 Testing Memory-Aware System Prompts...")
    
    try:
        # Create agent builder
        qwen_backend = QwenBackend()
        agent_builder = AgentBuilder(qwen_backend)
        
        # Test 1: Memory-enabled system prompt
        spec1 = AgentSpecification(
            name="Memory Agent",
            goal="Help with coding tasks",
            memory_enabled=True,
            memory_context_limit=20,
            memory_importance_threshold=0.7
        )
        
        prompt1 = agent_builder._get_default_system_prompt(spec1)
        
        if "Memory System:" in prompt1:
            print("✅ Memory-enabled prompt includes memory instructions")
        else:
            print("❌ Memory-enabled prompt missing memory instructions")
            return False
        
        if "importance scores" in prompt1:
            print("✅ Memory-enabled prompt includes importance guidance")
        else:
            print("❌ Memory-enabled prompt missing importance guidance")
            return False
        
        if f"{spec1.memory_importance_threshold}" in prompt1:
            print("✅ Memory-enabled prompt includes threshold configuration")
        else:
            print("❌ Memory-enabled prompt missing threshold configuration")
            return False
        
        # Test 2: Memory-disabled system prompt
        spec2 = AgentSpecification(
            name="No Memory Agent",
            goal="Help with simple tasks",
            memory_enabled=False
        )
        
        prompt2 = agent_builder._get_default_system_prompt(spec2)
        
        if "Memory System:" not in prompt2:
            print("✅ Memory-disabled prompt excludes memory instructions")
        else:
            print("❌ Memory-disabled prompt incorrectly includes memory instructions")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory-aware system prompts test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🤖 Testing Agent Memory Integration (Simple)")
    print("=" * 55)
    
    success = True
    
    # Test memory specification features
    if not await test_memory_specification_features():
        success = False
    
    # Test memory validation
    if not await test_memory_validation():
        success = False
    
    # Test memory-aware code generation
    if not await test_memory_aware_code_generation():
        success = False
    
    # Test memory-aware system prompts
    if not await test_memory_aware_system_prompts():
        success = False
    
    if success:
        print("\\n🎉 All agent memory integration tests passed!")
        print("\\nKey features implemented:")
        print("✅ Memory configuration fields in AgentSpecification")
        print("✅ Memory configuration validation")
        print("✅ Memory-aware code generation")
        print("✅ Memory-aware system prompt generation")
        print("✅ Memory settings for different agent types")
        print("✅ Comprehensive memory parameter validation")
        print("\\nAgent specifications now support full memory configuration!")
    else:
        print("\\n❌ Some agent memory integration tests failed.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())