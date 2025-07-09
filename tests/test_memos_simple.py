#!/usr/bin/env python3
"""
Simple MemOS test to understand the API

This script tests MemOS with minimal configuration to understand the correct structure.
"""

def test_memos_api_exploration():
    """Explore MemOS API to understand configuration."""
    try:
        print("🧠 Exploring MemOS API...")
        
        from memos.mem_os.main import MOS
        from memos.mem_cube.general import GeneralMemCube
        
        # Check if we can create a minimal config
        print("Available in MOS:", [attr for attr in dir(MOS) if not attr.startswith('_')])
        
        # Try to understand GeneralMemCube structure
        print("Available in GeneralMemCube:", [attr for attr in dir(GeneralMemCube) if not attr.startswith('_')])
        
        # Try to create empty config first
        empty_config = {}
        print("Testing with empty config...")
        
        return True
        
    except Exception as e:
        print(f"❌ API exploration failed: {e}")
        return False

def test_minimal_memos():
    """Test with absolutely minimal MemOS setup."""
    try:
        print("\n🧠 Testing Minimal MemOS Setup...")
        
        from memos.mem_os.main import MOS
        
        # Try creating MOS with minimal config
        minimal_config = {}
        
        print("Testing MOS creation...")
        # Just try to import and see what happens
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🤖 Simple MemOS API Exploration")
    print("=" * 40)
    
    success = True
    
    if not test_memos_api_exploration():
        success = False
    
    if not test_minimal_memos():
        success = False
    
    if success:
        print("\n🎉 MemOS API exploration completed!")
    else:
        print("\n❌ MemOS API exploration had issues.")
    
    return success

if __name__ == "__main__":
    main()