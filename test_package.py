#!/usr/bin/env python3
"""
Test script to validate the Tektra AI Assistant package structure.
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🔍 Testing package imports...")
    
    try:
        # Test main package import
        import tektra
        print(f"✅ tektra version: {tektra.__version__}")
        
        # Test CLI import
        from tektra.cli import main as cli_main
        print("✅ CLI module imported successfully")
        
        # Test server import
        from tektra.server import start_server
        print("✅ Server module imported successfully")
        
        # Test app components
        from tektra.app.main import app
        from tektra.app.config import settings
        from tektra.app.database import init_database
        print("✅ App components imported successfully")
        
        # Test models
        from tektra.app.models.conversation import Conversation, Message
        from tektra.app.models.user import User
        print("✅ Database models imported successfully")
        
        # Test services
        from tektra.app.services.conversation_service import conversation_manager
        from tektra.app.services.ai_service import ai_manager
        print("✅ Services imported successfully")
        
        # Test routers
        from tektra.app.routers import ai, conversations, websocket
        print("✅ API routers imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_cli_commands():
    """Test that CLI commands are available."""
    print("\n🔍 Testing CLI command availability...")
    
    try:
        from tektra.cli import app as cli_app
        
        # Check that commands are registered
        commands = cli_app.commands
        expected_commands = ['start', 'setup', 'info', 'version']
        
        for cmd in expected_commands:
            if cmd in [c.name for c in commands.values()]:
                print(f"✅ Command '{cmd}' is available")
            else:
                print(f"❌ Command '{cmd}' is missing")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def test_database_models():
    """Test database model definitions."""
    print("\n🔍 Testing database models...")
    
    try:
        from tektra.app.models.conversation import Conversation, Message, MessageRole, MessageType
        from tektra.app.models.user import User
        from tektra.app.database import Base
        
        # Check that models inherit from Base
        assert issubclass(Conversation, Base), "Conversation should inherit from Base"
        assert issubclass(Message, Base), "Message should inherit from Base"
        assert issubclass(User, Base), "User should inherit from Base"
        
        print("✅ Database models are properly defined")
        
        # Check enums
        assert hasattr(MessageRole, 'USER'), "MessageRole should have USER"
        assert hasattr(MessageRole, 'ASSISTANT'), "MessageRole should have ASSISTANT"
        assert hasattr(MessageType, 'TEXT'), "MessageType should have TEXT"
        
        print("✅ Database enums are properly defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Database model test error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")
    
    try:
        from tektra.app.config import settings
        
        # Check that essential settings exist
        essential_settings = [
            'api_title', 'api_version', 'host', 'port', 
            'database_url', 'debug', 'model_cache_dir'
        ]
        
        for setting in essential_settings:
            if hasattr(settings, setting):
                print(f"✅ Setting '{setting}' is available")
            else:
                print(f"❌ Setting '{setting}' is missing")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Tektra AI Assistant Package\n")
    
    tests = [
        test_imports,
        test_cli_commands,
        test_database_models,
        test_configuration,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Package is ready for distribution.")
        return 0
    else:
        print("\n💥 Some tests failed. Please fix the issues before packaging.")
        return 1

if __name__ == "__main__":
    sys.exit(main())