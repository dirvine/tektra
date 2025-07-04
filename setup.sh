#!/bin/bash

# Tektra AI Assistant Setup Script
# This script sets up the Tektra AI Assistant Desktop Extension

set -e

echo "🚀 Setting up Tektra AI Assistant..."

# Check system requirements
echo "📋 Checking system requirements..."

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Please install Rust first: https://rustup.rs/"
    exit 1
fi

# Check available memory
if command -v free &> /dev/null; then
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 4 ]; then
        echo "⚠️  Warning: Only ${MEMORY_GB}GB RAM detected. Minimum 4GB recommended."
    fi
elif command -v vm_stat &> /dev/null; then
    # macOS
    MEMORY_BYTES=$(sysctl -n hw.memsize)
    MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
    if [ "$MEMORY_GB" -lt 4 ]; then
        echo "⚠️  Warning: Only ${MEMORY_GB}GB RAM detected. Minimum 4GB recommended."
    fi
fi

echo "✅ System requirements check completed"

# Check for system Ollama
echo "🔍 Checking for system Ollama installation..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null | head -n1 || echo "unknown")
    echo "✅ Found system Ollama: $OLLAMA_VERSION"
    echo "   Tektra will use system Ollama if available, otherwise use bundled version"
else
    echo "ℹ️  System Ollama not found. Tektra will use bundled Ollama"
fi

# Build the project
echo "🔨 Building Tektra AI Assistant..."
echo "   This may take several minutes on first build..."

if ! cargo build --release; then
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi

echo "✅ Build completed successfully"

# Check if MCP server feature is available
echo "🧪 Building MCP server component..."
if cargo build --release --features mcp-server --bin tektra-mcp; then
    echo "✅ MCP server built successfully"
else
    echo "⚠️  MCP server build failed, continuing without MCP features"
fi

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p ~/.tektra/models
mkdir -p ~/.tektra/conversations
mkdir -p ~/.tektra/cache
mkdir -p ~/.tektra/logs

echo "✅ Application directories created"

# Set up default configuration
echo "⚙️  Setting up default configuration..."
cat > ~/.tektra/config.json << EOF
{
  "default_model": "qwen2.5-vl:7b",
  "auto_download_models": true,
  "enable_gpu_acceleration": true,
  "conversation": {
    "max_context_length": 32768,
    "memory_enabled": true,
    "auto_summarize": true
  },
  "vision": {
    "max_image_size_mb": 10,
    "auto_enhance": true,
    "enable_caching": true
  },
  "mcp": {
    "enable_server": true,
    "transport": "stdio",
    "max_concurrent_sessions": 10,
    "rate_limit_requests_per_minute": 60
  }
}
EOF

echo "✅ Default configuration created at ~/.tektra/config.json"

# Create desktop entry on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🖥️  Creating desktop entry..."
    TEKTRA_PATH="$(pwd)/src-tauri/target/release/tektra"
    mkdir -p ~/.local/share/applications
    cat > ~/.local/share/applications/tektra.desktop << EOF
[Desktop Entry]
Version=1.0
Name=Tektra AI Assistant
Comment=Advanced multimodal AI assistant
Exec=$TEKTRA_PATH
Icon=$TEKTRA_PATH
Terminal=false
Type=Application
Categories=Utility;Development;Science;
EOF
    echo "✅ Desktop entry created"
fi

# Print success message
echo ""
echo "🎉 Tektra AI Assistant setup completed successfully!"
echo ""
echo "📚 Quick Start:"
echo "   1. Run the main application:"
echo "      ./src-tauri/target/release/tektra"
echo ""
echo "   2. Run as MCP server:"
echo "      ./src-tauri/target/release/tektra-mcp"
echo ""
echo "   3. Configuration location:"
echo "      ~/.tektra/config.json"
echo ""
echo "🔧 First Run:"
echo "   - On first startup, Tektra will download the default AI model"
echo "   - This may take 5-15 minutes depending on your internet connection"
echo "   - The model will be cached locally for future use"
echo ""
echo "📖 Documentation:"
echo "   - View README.md for detailed usage instructions"
echo "   - Check manifest.json for available MCP tools"
echo "   - Visit the project homepage for updates and support"
echo ""
echo "🎯 MCP Integration:"
echo "   To use Tektra as an MCP server, configure your MCP client to run:"
echo "   $(pwd)/src-tauri/target/release/tektra-mcp"
echo ""
echo "Happy AI-ing! 🤖✨"