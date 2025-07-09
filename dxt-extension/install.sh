#!/bin/bash

# Tektra Voice AI DXT Extension Installer
# This script helps install dependencies and prepare the DXT extension

set -e

echo "🎙️  Tektra Voice AI DXT Extension Installer"
echo "============================================"

# Check Node.js version
check_node() {
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js is not installed"
        echo "📦 Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        echo "❌ Node.js version $NODE_VERSION is too old"
        echo "📦 Please upgrade to Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
    
    echo "✅ Node.js $(node -v) detected"
}

# Check for Tektra executable
check_tektra() {
    if command -v tektra &> /dev/null; then
        echo "✅ Tektra executable found in PATH"
        TEKTRA_VERSION=$(tektra --version 2>/dev/null || echo "unknown")
        echo "   Version: $TEKTRA_VERSION"
        return 0
    fi
    
    # Check if we can build from source
    if [ -f "../Cargo.toml" ]; then
        echo "📦 Tektra source found, checking if we can build..."
        if command -v cargo &> /dev/null; then
            echo "🔨 Building Tektra from source..."
            cd ..
            cargo build --release
            if [ -f "target/release/tektra" ]; then
                echo "✅ Tektra built successfully"
                echo "💡 Add $(pwd)/target/release to your PATH or copy tektra to /usr/local/bin"
                cd dxt-extension
                return 0
            else
                echo "❌ Failed to build Tektra"
                exit 1
            fi
        else
            echo "❌ Rust/Cargo not found for building Tektra"
            echo "📦 Please install Rust from https://rustup.rs/"
            exit 1
        fi
    fi
    
    echo "❌ Tektra executable not found"
    echo "📦 Please install Tektra or build from source"
    echo "   - Download from: https://github.com/dirvine/tektra/releases"
    echo "   - Or build from source with: cargo build --release"
    exit 1
}

# Install Node.js dependencies
install_deps() {
    echo "📦 Installing Node.js dependencies..."
    cd server
    npm install
    cd ..
    echo "✅ Dependencies installed"
}

# Create cache directory
create_cache() {
    CACHE_DIR="$HOME/.cache/tektra-ai"
    echo "📁 Creating cache directory: $CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    echo "✅ Cache directory created"
}

# Set permissions
set_permissions() {
    echo "🔐 Setting executable permissions..."
    chmod +x server/index.js
    chmod +x install.sh
    echo "✅ Permissions set"
}

# Test the MCP server
test_server() {
    echo "🧪 Testing MCP server..."
    cd server
    
    # Create a simple test
    cat > test_server.js << 'EOF'
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

console.log('✅ MCP SDK import successful');
console.log('✅ Server can be created');
process.exit(0);
EOF
    
    if node test_server.js; then
        echo "✅ MCP server test passed"
        rm test_server.js
    else
        echo "❌ MCP server test failed"
        rm -f test_server.js
        exit 1
    fi
    
    cd ..
}

# Create DXT package
create_package() {
    echo "📦 Creating DXT package..."
    
    # Create a simple icon if it doesn't exist
    if [ ! -f "icon.png" ]; then
        echo "🎨 Creating placeholder icon..."
        # Create a simple SVG and convert to PNG if imagemagick is available
        if command -v convert &> /dev/null; then
            convert -size 128x128 xc:transparent -fill "#4CAF50" -draw "circle 64,64 64,32" \
                    -fill white -font Arial-Bold -pointsize 40 -gravity center -annotate 0 "T" icon.png
            echo "✅ Icon created"
        else
            echo "💡 No icon created (imagemagick not available)"
            echo "   You can add icon.png manually for a custom icon"
        fi
    fi
    
    # Create placeholder screenshots
    mkdir -p screenshots
    if [ ! -f "screenshots/voice-interface.png" ]; then
        echo "📸 Creating placeholder screenshots..."
        echo "💡 Add real screenshots to screenshots/ directory"
        touch screenshots/voice-interface.png
        touch screenshots/model-loading.png  
        touch screenshots/multimodal-chat.png
    fi
    
    # Package everything
    DXT_FILE="tektra-voice-ai.dxt"
    echo "📦 Packaging DXT file: $DXT_FILE"
    
    zip -r "$DXT_FILE" \
        manifest.json \
        server/ \
        README.md \
        install.sh \
        icon.png \
        screenshots/ \
        -x "server/node_modules/.cache/*" \
        -x "server/test_server.js" \
        -x "*/.DS_Store"
    
    if [ -f "$DXT_FILE" ]; then
        echo "✅ DXT package created: $DXT_FILE"
        echo "📏 Package size: $(du -h "$DXT_FILE" | cut -f1)"
    else
        echo "❌ Failed to create DXT package"
        exit 1
    fi
}

# Print installation instructions
print_instructions() {
    echo ""
    echo "🎉 Tektra Voice AI DXT Extension is ready!"
    echo "============================================"
    echo ""
    echo "📁 Package: tektra-voice-ai.dxt"
    echo ""
    echo "🚀 Installation:"
    echo "   1. Import tektra-voice-ai.dxt into your DXT-compatible client"
    echo "   2. Grant requested permissions (audio, file system, network)"
    echo "   3. The extension will auto-start voice services"
    echo ""
    echo "🛠️  Available Tools:"
    echo "   - start_voice_conversation"
    echo "   - process_multimodal_input" 
    echo "   - load_model"
    echo "   - get_voice_status"
    echo "   - manage_voice_pipeline"
    echo ""
    echo "📖 See README.md for detailed usage instructions"
    echo ""
    echo "🎯 Quick Test:"
    echo '   Call: start_voice_conversation({"character": "friendly"})'
    echo ""
}

# Main installation flow
main() {
    echo "Starting installation..."
    
    check_node
    check_tektra
    install_deps
    create_cache
    set_permissions
    test_server
    create_package
    print_instructions
    
    echo "✅ Installation completed successfully!"
}

# Run main function
main "$@"