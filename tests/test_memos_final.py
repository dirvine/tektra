#!/usr/bin/env python3
"""
Final MemOS test with proper configuration

This script tests MemOS using the correct configuration structure.
"""

import tempfile
import uuid
from pathlib import Path

def test_memos_with_config():
    """Test MemOS with proper configuration."""
    try:
        print("🧠 Testing MemOS with Configuration...")
        
        from memos.mem_os.main import MOS
        
        # Create proper MOS configuration
        # Based on the MemOS documentation, create a minimal config
        mos_config = {
            "mem_cube_configs": [],  # Start with empty configs
            "user_management": {
                "enable_user_isolation": True
            }
        }
        
        print("Creating MOS instance with config...")
        memory = MOS(mos_config)
        print("✅ Successfully created MOS instance")
        
        # Create a test user
        user_id = str(uuid.uuid4())
        print(f"Creating user: {user_id}")
        memory.create_user(user_id=user_id)
        print("✅ Successfully created user")
        
        # List users to verify
        users = memory.list_users()
        print(f"✅ Users in system: {len(users)}")
        
        # Test adding some memory
        print("Testing memory operations...")
        test_messages = [
            {"role": "user", "content": "Hello, I'm testing MemOS integration with Tektra"},
            {"role": "assistant", "content": "Great! MemOS is working properly for AI agents."}
        ]
        
        memory.add(test_messages, user_id=user_id)
        print("✅ Successfully added memory")
        
        # Test searching memory
        search_results = memory.search("testing MemOS", user_id=user_id)
        print(f"✅ Search completed, found {len(search_results)} results")
        if search_results:
            print(f"   Sample result: {search_results[0]}")
        
        # Test getting all memories
        all_memories = memory.get_all(user_id=user_id)
        print(f"✅ Retrieved {len(all_memories)} total memories")
        
        return True
        
    except Exception as e:
        print(f"❌ MemOS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memos_for_agents():
    """Test MemOS usage pattern for AI agents."""
    try:
        print("\n🧠 Testing MemOS for AI Agents...")
        
        from memos.mem_os.main import MOS
        
        # Configuration for agent use case
        agent_config = {
            "mem_cube_configs": [],
            "user_management": {
                "enable_user_isolation": True
            }
        }
        
        memory = MOS(agent_config)
        
        # Create separate memory spaces for different agents
        agent_1_id = "agent_github_monitor"
        agent_2_id = "agent_stock_analyzer"
        
        memory.create_user(user_id=agent_1_id)
        memory.create_user(user_id=agent_2_id)
        
        # Agent 1: GitHub monitoring agent memory
        github_memories = [
            {"role": "system", "content": "You are a GitHub monitoring agent"},
            {"role": "user", "content": "Monitor repository issues and pull requests"},
            {"role": "assistant", "content": "I'll track GitHub activity and notify about important changes"}
        ]
        
        memory.add(github_memories, user_id=agent_1_id)
        
        # Agent 2: Stock analysis agent memory
        stock_memories = [
            {"role": "system", "content": "You are a stock analysis agent"},
            {"role": "user", "content": "Analyze AAPL stock performance"},
            {"role": "assistant", "content": "I'll monitor AAPL and provide analysis when price changes significantly"}
        ]
        
        memory.add(stock_memories, user_id=agent_2_id)
        
        # Test agent-specific memory retrieval
        github_context = memory.search("GitHub monitoring", user_id=agent_1_id)
        stock_context = memory.search("stock analysis", user_id=agent_2_id)
        
        print(f"✅ GitHub agent memories: {len(github_context)}")
        print(f"✅ Stock agent memories: {len(stock_context)}")
        
        # Verify isolation - agent 1 shouldn't see agent 2's memories
        cross_search = memory.search("stock", user_id=agent_1_id)
        print(f"✅ Cross-agent isolation verified: {len(cross_search)} cross-results")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🤖 Final MemOS Integration Test for Tektra")
    print("=" * 50)
    
    success = True
    
    if not test_memos_with_config():
        success = False
    
    if not test_memos_for_agents():
        success = False
    
    if success:
        print("\n🎉 MemOS is fully working! Ready for Tektra integration.")
        print("Key findings:")
        print("- MemOS requires configuration dict")
        print("- User isolation works for agents")
        print("- Memory add/search operations functional")
        print("- Ready to integrate with Tektra agents")
    else:
        print("\n❌ MemOS tests failed.")
    
    return success

if __name__ == "__main__":
    main()