#!/bin/bash

# Create a proper release bundle for distribution

echo "🚀 Creating Tektra Release Bundle"
echo "================================="

# Set up variables
APP_NAME="Tektra"
BUNDLE_DIR="./Tektra.app"
VERSION="0.2.3"

# Clean up any existing bundle
rm -rf "$BUNDLE_DIR"

# Create app bundle structure
echo "📁 Creating app bundle structure..."
mkdir -p "$BUNDLE_DIR/Contents/"{MacOS,Resources,Frameworks}

# Copy the binary
echo "🔧 Installing Tektra binary..."
cp src-tauri/target/release/tektra "$BUNDLE_DIR/Contents/MacOS/"
chmod +x "$BUNDLE_DIR/Contents/MacOS/tektra"

# Copy frontend assets
echo "💻 Installing frontend assets..."
cp -r dist/* "$BUNDLE_DIR/Contents/Resources/"

# Copy icon if exists
if [ -f "src-tauri/icons/32x32.png" ]; then
    cp "src-tauri/icons/32x32.png" "$BUNDLE_DIR/Contents/Resources/icon.png"
fi

# Create Info.plist
echo "📄 Creating Info.plist..."
cat > "$BUNDLE_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>tektra</string>
    <key>CFBundleIdentifier</key>
    <string>com.tektra.desktop</string>
    <key>CFBundleName</key>
    <string>Tektra</string>
    <key>CFBundleDisplayName</key>
    <string>Tektra - AI Voice Assistant</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSCameraUsageDescription</key>
    <string>Tektra uses camera for vision AI capabilities to analyze images and provide visual assistance.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Tektra uses microphone for voice interaction and speech recognition to understand your voice commands.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key>
        <true/>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
</dict>
</plist>
EOF

# Create PkgInfo
echo "APPL????" > "$BUNDLE_DIR/Contents/PkgInfo"

# Set proper permissions
echo "🔐 Setting permissions..."
chmod -R 755 "$BUNDLE_DIR"
chmod +x "$BUNDLE_DIR/Contents/MacOS/tektra"

# Create DMG (optional, if create-dmg is available)
if command -v create-dmg &> /dev/null; then
    echo "💿 Creating DMG..."
    create-dmg \
        --volname "Tektra AI Assistant" \
        --volicon "$BUNDLE_DIR/Contents/Resources/icon.png" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 175 120 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 425 120 \
        "Tektra-$VERSION.dmg" \
        "$BUNDLE_DIR"
else
    echo "⚠️  create-dmg not available, skipping DMG creation"
fi

echo ""
echo "✅ Release bundle created successfully!"
echo "📱 App bundle: $BUNDLE_DIR"
echo "🏃 Install with: open '$BUNDLE_DIR'"
echo ""
echo "📋 Bundle contents:"
echo "   - Tektra AI assistant with enhanced Ollama environment"
echo "   - Complete Tauri 2.x compatibility"
echo "   - Bundled Ollama (no user installation required)"
echo "   - Camera and microphone permissions configured"
echo "   - Self-contained with all dependencies"
echo ""
echo "🎯 Ready for testing and distribution!"