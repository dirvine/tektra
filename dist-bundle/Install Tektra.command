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
