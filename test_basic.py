#!/usr/bin/env python3
"""
Basic test script to verify core components work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that basic imports work."""
    try:
        from tektra.ai.simple_llm import SimpleLLM
        print("✅ SimpleLLM import successful")
        
        from tektra.gui.chat_panel import ChatPanel, ChatManager
        print("✅ ChatPanel import successful")
        
        from tektra.agents.simple_agent import PythonAgent, SimpleAgentFactory
        print("✅ SimpleAgent import successful")
        
        from tektra.agents.simple_runtime import SimpleAgentRuntime
        print("✅ SimpleAgentRuntime import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_agent_creation():
    """Test agent creation without dependencies."""
    try:
        from tektra.agents.simple_agent import SimpleAgentFactory
        
        # Create a simple agent
        agent = SimpleAgentFactory.create_python_agent(
            "Test Agent",
            "A simple test agent for basic calculations"
        )
        
        print(f"✅ Agent created: {agent.spec.name}")
        print(f"   Description: {agent.spec.description}")
        print(f"   Status: {agent.status.value}")
        
        # Test agent info
        info = agent.get_info()
        print(f"   Agent info: {info}")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return False

def test_code_validation():
    """Test code validation."""
    try:
        from tektra.agents.simple_agent import PythonAgent, SimpleAgentSpec
        
        # Create a test agent
        spec = SimpleAgentSpec(
            id="test-123",
            name="Test Agent",
            description="Test agent",
            agent_type="python",
            code=""
        )
        
        agent = PythonAgent(spec)
        
        # Test safe code
        safe_code = "print('Hello, World!')\nresult = 2 + 2\nprint(f'Result: {result}')"
        is_safe = agent._validate_code_safety(safe_code)
        print(f"✅ Safe code validation: {is_safe}")
        
        # Test unsafe code
        unsafe_code = "import os\nos.system('rm -rf /')"
        is_unsafe = agent._validate_code_safety(unsafe_code)
        print(f"✅ Unsafe code validation: {not is_unsafe}")
        
        return True
    except Exception as e:
        print(f"❌ Code validation error: {e}")
        return False

def main():
    """Run basic tests."""
    print("🧪 Testing Tektra Basic Components")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_agent_creation,
        test_code_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"❌ {test.__name__} failed")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("💥 Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())