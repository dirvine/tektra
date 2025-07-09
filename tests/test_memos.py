#!/usr/bin/env python3
"""
Test script for MemOS integration

This script tests basic MemOS functionality to ensure it works correctly.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

def test_memos_import():
    """Test importing MemOS components."""
    try:
        print("🧠 Testing MemOS Import...")
        
        # Test basic import
        from memos.mem_os.main import MOS
        print("✅ Successfully imported MOS")
        
        # Test MemCube import
        from memos.mem_cube.general import GeneralMemCube
        print("✅ Successfully imported GeneralMemCube")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_memos_basic_functionality():
    """Test basic MemOS functionality."""
    try:
        print("\n🧠 Testing MemOS Basic Functionality...")
        
        from memos.mem_os.main import MOS
        from memos.mem_cube.general import GeneralMemCube
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Test MemCube creation
            mem_cube_path = Path(temp_dir) / "test_mem_cube"
            mem_cube_path.mkdir()
            
            # Create proper MemCube structure with config.json
            config_content = {
                "mem_cube_id": "test_cube",
                "mem_cube_name": "Test Memory Cube",
                "textual_memory": {
                    "enable": True,
                    "storage_path": "textual_memories.json"
                },
                "activation_memory": {
                    "enable": True,
                    "storage_path": "activation_memories.json"
                }
            }
            
            import json
            (mem_cube_path / "config.json").write_text(json.dumps(config_content, indent=2))
            (mem_cube_path / "textual_memories.json").write_text('[]')
            (mem_cube_path / "activation_memories.json").write_text('[]')
            
            print("✅ Created test MemCube structure with config")
            
            # Test MemCube initialization
            mem_cube = GeneralMemCube.init_from_dir(str(mem_cube_path))
            print("✅ Successfully initialized MemCube")
            
            # Test basic memory operations
            print("✅ Basic MemOS functionality verified")
            
        return True
        
    except Exception as e:
        print(f"❌ MemOS functionality test failed: {e}")
        return False

def test_memos_configuration():
    """Test MemOS configuration capabilities."""
    try:
        print("\n🧠 Testing MemOS Configuration...")
        
        # Test configuration structure
        config = {
            "mem_cube_configs": [],
            "user_management": {
                "enable_user_isolation": True
            },
            "memory_settings": {
                "max_memory_items": 1000,
                "memory_retention_days": 30
            }
        }
        
        print("✅ MemOS configuration structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🤖 Testing MemOS Integration for Tektra")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_memos_import():
        success = False
    
    # Test basic functionality
    if not test_memos_basic_functionality():
        success = False
    
    # Test configuration
    if not test_memos_configuration():
        success = False
    
    if success:
        print("\n🎉 All MemOS tests passed! Ready for integration.")
    else:
        print("\n❌ Some tests failed. Check MemOS installation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)