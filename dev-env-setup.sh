#!/bin/bash

# Development environment setup - run once to optimize build speeds

echo "🔧 Setting up optimized development environment..."

# Create cargo cache directory
mkdir -p ~/.cargo/registry

# Set up faster npm caching
npm config set fund false
npm config set audit false

# Pre-build dependencies (this takes time once, then speeds up future builds)
echo "Pre-building Rust dependencies..."
cd src-tauri
cargo fetch --locked
cargo check --release

# Create .env for development defaults
cat > ../.env << EOF
# Development configuration
RUST_LOG=info
OLLAMA_DEBUG=false
NODE_ENV=development
# Skip heavy operations in dev
SKIP_WHISPER_DOWNLOAD=true
SKIP_MODEL_VALIDATION=true
EOF

echo "✅ Development environment optimized!"
echo ""
echo "📋 Available commands:"
echo "  ./dev-run.sh        - Ultra-fast development run (no bundling)"
echo "  ./quick-build.sh    - Fast build with simple app bundle"
echo "  npm run dev         - Frontend development server"
echo "  cargo tauri dev     - Full Tauri development mode"
echo ""
echo "💡 First run may be slow, subsequent runs will be much faster!"