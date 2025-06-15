#!/usr/bin/env python3
"""
Build and package Tektra AI Assistant for distribution.
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report success/failure."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"   Error: {e.stderr}")
        return False

def build_frontend():
    """Build the frontend if it exists."""
    frontend_dir = Path("frontend")
    if frontend_dir.exists():
        print("🔄 Building frontend...")
        if (frontend_dir / "package.json").exists():
            os.chdir(frontend_dir)
            if run_command("npm install", "Installing frontend dependencies"):
                if run_command("npm run build", "Building frontend"):
                    print("✅ Frontend built successfully")
                    os.chdir("..")
                    return True
            os.chdir("..")
    
    print("⚠️ Frontend build skipped (no package.json found)")
    return True

def build_package():
    """Build the Python package."""
    print("\n🏗️ Building Tektra AI Assistant Package")
    
    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    dirs_to_clean = ["build", "dist", "*.egg-info"]
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   Removed {path}")
    
    # Build frontend
    if not build_frontend():
        print("❌ Frontend build failed")
        return False
    
    # Build Python package
    if not run_command("python -m build", "Building Python package"):
        return False
    
    # Check package contents
    dist_dir = Path("dist")
    if dist_dir.exists():
        files = list(dist_dir.glob("*"))
        print(f"\n📦 Package files created:")
        for file in files:
            print(f"   {file}")
    
    print("\n🎉 Package build completed!")
    return True

def create_install_script():
    """Create a simple installation script."""
    script_content = """#!/bin/bash
# Tektra AI Assistant Installation Script

echo "🚀 Installing Tektra AI Assistant..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install the package
echo "📦 Installing tektra..."
pip install tektra

# Verify installation
if command -v tektra &> /dev/null; then
    echo "✅ Tektra AI Assistant installed successfully!"
    echo ""
    echo "🎯 Quick start:"
    echo "   tektra setup    # First time setup"
    echo "   tektra start    # Start the assistant"
    echo ""
    echo "📖 For help: tektra --help"
else
    echo "❌ Installation verification failed"
    exit 1
fi
"""
    
    with open("install.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("install.sh", 0o755)
    print("✅ Created install.sh script")

def main():
    """Main build function."""
    print("🔨 Tektra AI Assistant - Package Builder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run from the project root.")
        return 1
    
    # Install build tools
    if not run_command("pip install build twine", "Installing build tools"):
        return 1
    
    # Build the package
    if not build_package():
        return 1
    
    # Create installation script
    create_install_script()
    
    print("\n🎊 Build completed successfully!")
    print("\n📋 Next steps:")
    print("   • Test the package: pip install dist/tektra_ai-*.whl")
    print("   • Upload to PyPI: twine upload dist/*")
    print("   • Share install.sh for easy installation")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())