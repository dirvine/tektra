#!/usr/bin/env python3
"""
Simple test to check if CLI commands are available.
"""

import subprocess
import sys

def test_cli_commands():
    """Test that CLI commands are available."""
    print("🔍 Testing CLI commands...")
    
    commands_to_test = [
        ("tektra --help", "Help command"),
        ("tektra version", "Version command"),
        ("tektra info", "Info command"),
    ]
    
    for cmd, description in commands_to_test:
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✅ {description}: Working")
            else:
                print(f"❌ {description}: Failed with code {result.returncode}")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ {description}: Timed out")
        except Exception as e:
            print(f"❌ {description}: Exception - {e}")

def main():
    """Run CLI tests."""
    print("🧪 Testing Tektra CLI Commands\n")
    test_cli_commands()

if __name__ == "__main__":
    main()