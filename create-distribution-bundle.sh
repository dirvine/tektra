#!/bin/bash

# Create Tektra distribution bundle for sharing

echo "📦 Creating Tektra Distribution Bundle"
echo "====================================="

# Variables
APP_NAME="Tektra"
VERSION="0.2.3"
DIST_DIR="./dist-bundle"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"

# Clean up and create distribution directory
echo "🧹 Preparing distribution directory..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Copy the working app bundle
echo "📱 Copying application bundle..."
cp -r Tektra.app "$APP_BUNDLE"

# Verify bundle structure
echo "🔍 Verifying bundle contents..."
if [ -f "$APP_BUNDLE/Contents/MacOS/tektra" ]; then
    echo "   ✅ Binary found"
else
    echo "   ❌ Binary missing"
    exit 1
fi

if [ -f "$APP_BUNDLE/Contents/Resources/index.html" ]; then
    echo "   ✅ Frontend assets found"
else
    echo "   ❌ Frontend assets missing"
    exit 1
fi

if [ -d "$APP_BUNDLE/Contents/Resources/assets" ]; then
    echo "   ✅ Asset directory found"
else
    echo "   ❌ Asset directory missing"
    exit 1
fi

# Set proper permissions
echo "🔐 Setting permissions..."
chmod -R 755 "$APP_BUNDLE"
chmod +x "$APP_BUNDLE/Contents/MacOS/tektra"

# Create README for distribution
echo "📝 Creating distribution README..."
cat > "$DIST_DIR/README.txt" << EOF
Tektra AI Voice Assistant v$VERSION
==================================

INSTALLATION:
1. Copy Tektra.app to your Applications folder
2. Double-click to launch
3. Grant camera and microphone permissions when prompted
4. Wait for initial AI model download (first launch only)

FEATURES:
- AI Chat with gemma3n:e4b model
- Voice interaction and speech recognition
- Vision AI for image analysis
- Bundled Ollama (no separate installation needed)
- Complete offline functionality

SYSTEM REQUIREMENTS:
- macOS 11.0 or later
- Apple Silicon (M1/M2/M3) or Intel Mac
- Camera and microphone (for full functionality)
- Internet connection (first launch for model download)

FIRST RUN:
The app will automatically download the AI model (~4GB) on first launch.
This may take several minutes depending on your internet connection.
After the initial download, the app works completely offline.

TROUBLESHOOTING:
- If the app doesn't open, try right-click → Open
- For permission issues, check System Preferences → Security & Privacy
- If AI features don't work, ensure the model download completed

TECHNICAL DETAILS:
- Built with Tauri 2.x
- Enhanced Ollama environment for reliability
- Self-contained with all dependencies
- No external installations required

For support or updates, visit: https://github.com/your-repo/tektra
EOF

# Create a simple installer script
echo "🛠️ Creating installer script..."
cat > "$DIST_DIR/Install Tektra.command" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"

echo "🚀 Installing Tektra AI Assistant..."
echo "==================================="

# Check if Applications directory exists
if [ ! -d "/Applications" ]; then
    echo "❌ Applications directory not found"
    exit 1
fi

# Copy app to Applications
if cp -r "Tektra.app" "/Applications/"; then
    echo "✅ Tektra installed successfully!"
    echo ""
    echo "📱 You can now find Tektra in your Applications folder"
    echo "🎯 Launch it from Launchpad or Applications folder"
    echo ""
    echo "💡 First launch will download AI models (this may take a few minutes)"
    
    # Try to open the app
    echo "🚀 Opening Tektra..."
    open "/Applications/Tektra.app"
else
    echo "❌ Installation failed. You may need administrator privileges."
    echo "💡 Try dragging Tektra.app to Applications folder manually"
fi

echo ""
echo "Press Enter to close this window..."
read
EOF

chmod +x "$DIST_DIR/Install Tektra.command"

# Create a ZIP archive for easy distribution
echo "🗜️ Creating ZIP archive..."
cd "$DIST_DIR"
zip -r "../Tektra-v$VERSION-Distribution.zip" . -x "*.DS_Store"
cd ..

# Create DMG for macOS-style distribution
echo "💿 Creating DMG installer..."
hdiutil create -volname "Tektra AI Assistant" \
    -srcfolder "$DIST_DIR" \
    -ov \
    -format UDZO \
    "Tektra-v$VERSION-Installer.dmg"

# Show results
echo ""
echo "✅ Distribution bundle created successfully!"
echo ""
echo "📦 Created files:"
echo "   📁 dist-bundle/              - Raw distribution folder"
echo "   📱 dist-bundle/Tektra.app    - Application bundle"
echo "   📝 dist-bundle/README.txt    - Installation instructions"
echo "   🛠️ dist-bundle/Install Tektra.command - Auto-installer"
echo "   🗜️ Tektra-v$VERSION-Distribution.zip - ZIP archive"
echo "   💿 Tektra-v$VERSION-Installer.dmg - DMG installer"
echo ""
echo "🚀 Ready for distribution!"
echo ""
echo "📋 Sharing options:"
echo "   • Share the ZIP file for easy download"
echo "   • Share the DMG for Mac-style installation"
echo "   • Users can drag Tektra.app directly to Applications"
echo ""
echo "💡 The app includes everything needed and works offline after initial setup!"
EOF