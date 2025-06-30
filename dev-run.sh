#!/bin/bash

# Ultra-fast development runner - no bundling, just build and run

echo "⚡ Ultra-Fast Dev Runner"
echo "======================="

# Kill any existing Tektra processes
pkill -f tektra 2>/dev/null || true
pkill -f ollama 2>/dev/null || true

# Quick build
echo "Building..."
cd src-tauri
cargo build --release --quiet

# Run directly
echo "🚀 Starting Tektra..."
echo "Press Ctrl+C to stop"
echo ""
./target/release/tektra