#!/bin/bash

# Quick Tektra build script - bypasses slow bundling for development testing

echo "🚀 Quick Tektra Build Script"
echo "============================="

# Step 1: Build frontend (fast)
echo "1. Building frontend..."
npm run build --silent

# Step 2: Build Rust backend only (much faster than full bundle)
echo "2. Building Rust backend..."
cd src-tauri
cargo build --release --quiet

# Step 3: Create a simple app bundle structure manually
echo "3. Creating development app bundle..."
BUNDLE_DIR="../target/Tektra.app"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/Contents/"{MacOS,Resources}

# Copy the binary
cp target/release/tektra "$BUNDLE_DIR/Contents/MacOS/"

# Create Info.plist
cat > "$BUNDLE_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>tektra</string>
    <key>CFBundleIdentifier</key>
    <string>com.tektra.app</string>
    <key>CFBundleName</key>
    <string>Tektra</string>
    <key>CFBundleDisplayName</key>
    <string>Tektra</string>
    <key>CFBundleVersion</key>
    <string>0.2.3</string>
    <key>CFBundleShortVersionString</key>
    <string>0.2.3</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSCameraUsageDescription</key>
    <string>Tektra uses camera for vision AI capabilities</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Tektra uses microphone for voice interaction</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Copy frontend assets
cp -r ../dist "$BUNDLE_DIR/Contents/Resources/"

# Make executable
chmod +x "$BUNDLE_DIR/Contents/MacOS/tektra"

echo "✅ Quick build complete!"
echo "📱 App created at: $BUNDLE_DIR"
echo "🏃 Run with: open '$BUNDLE_DIR'"
echo ""
echo "💡 This development build skips:"
echo "   - Code signing"
echo "   - DMG creation"
echo "   - Notarization"
echo "   - Full Tauri bundling overhead"
echo ""
echo "For production builds, use: cargo tauri build"