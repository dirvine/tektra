#!/usr/bin/env python3
"""
Correct MemOS test using MOSConfig

This script tests MemOS using the proper configuration class.
"""

import uuid

def test_memos_with_proper_config():
    """Test MemOS with proper MOSConfig."""
    try:
        print("🧠 Testing MemOS with MOSConfig...")
        
        from memos.mem_os.main import MOS
        from memos.configs.mem_os import MOSConfig
        
        # Create proper MOS configuration
        print("Creating MOSConfig...")
        config = MOSConfig()
        print("✅ Successfully created MOSConfig")
        
        print("Creating MOS instance with MOSConfig...")
        memory = MOS(config)
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
            {"role": "user", "content": "Hello, I'm testing MemOS integration with Tektra AI agents"},
            {"role": "assistant", "content": "Great! MemOS is working properly for persistent agent memory."}
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

def test_agent_memory_patterns():
    """Test memory patterns specific to AI agents."""
    try:
        print("\n🧠 Testing Agent Memory Patterns...")
        
        from memos.mem_os.main import MOS
        from memos.configs.mem_os import MOSConfig
        
        config = MOSConfig()
        memory = MOS(config)
        
        # Test multiple agent scenarios
        agents = {
            "github_monitor": "Monitor GitHub repositories for issues and PRs",
            "stock_analyzer": "Analyze stock prices and send alerts",
            "file_organizer": "Organize downloads folder by file type"
        }
        
        for agent_id, description in agents.items():
            # Create agent user
            memory.create_user(user_id=agent_id)
            
            # Add agent context
            agent_context = [
                {"role": "system", "content": f"You are an AI agent: {description}"},
                {"role": "user", "content": f"Initialize {agent_id} with task: {description}"},
                {"role": "assistant", "content": f"I am {agent_id} and I understand my task: {description}"}
            ]
            
            memory.add(agent_context, user_id=agent_id)
            print(f"✅ Initialized memory for {agent_id}")
        
        # Test agent memory retrieval
        for agent_id in agents.keys():
            agent_memories = memory.get_all(user_id=agent_id)
            search_results = memory.search("task", user_id=agent_id)
            print(f"✅ {agent_id}: {len(agent_memories)} memories, {len(search_results)} task-related")
        
        print("✅ All agent memory patterns working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Agent memory pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🤖 Correct MemOS Integration Test for Tektra")
    print("=" * 55)
    
    success = True
    
    if not test_memos_with_proper_config():
        success = False
    
    if not test_agent_memory_patterns():
        success = False
    
    if success:
        print("\n🎉 MemOS is fully functional! Perfect for Tektra integration.")
        print("\nIntegration plan:")
        print("✅ Use MOSConfig for configuration")
        print("✅ Create user per agent for memory isolation") 
        print("✅ Use add() to store agent conversations")
        print("✅ Use search() for context-aware responses")
        print("✅ Use get_all() for complete agent history")
    else:
        print("\n❌ MemOS tests failed.")
    
    return success

if __name__ == "__main__":
    main()