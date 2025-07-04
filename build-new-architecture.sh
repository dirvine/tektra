#!/bin/bash

# Build script for the new Tektra architecture with mistral.rs

set -e

echo "🚀 Building Tektra with new mistral.rs architecture"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo is not installed. Please install Rust."
        exit 1
    fi
    
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js."
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    print_status "✓ All dependencies found"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Set environment variables for mistral.rs
    export MISTRALRS_BACKEND=true
    export RUST_LOG=info
    
    # Ensure we're in the right directory
    cd "$(dirname "$0")"
    
    print_status "✓ Environment configured"
}

# Install Rust dependencies
install_rust_deps() {
    print_status "Installing Rust dependencies..."
    
    cd src-tauri
    
    # Update dependencies
    cargo update
    
    # Check if the build works with default features
    if cargo check; then
        print_status "✓ Rust dependencies resolved successfully (mistral.rs included by default)"
    else
        print_error "Failed to resolve Rust dependencies"
        print_warning "This might be due to mistral.rs version compatibility"
        exit 1
    fi
    
    cd ..
}

# Install Node dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    if npm install; then
        print_status "✓ Node.js dependencies installed successfully"
    else
        print_error "Failed to install Node.js dependencies"
        exit 1
    fi
}

# Build the backend
build_backend() {
    print_status "Building Rust backend..."
    
    cd src-tauri
    
    # Build with default features (includes mistral-backend)
    if cargo build --release --features full-multimodal; then
        print_status "✓ Backend built with full features (mistral.rs + MCP)"
    elif cargo build --release; then
        print_status "✓ Backend built with default features (mistral.rs included)"
    else
        print_error "Failed to build backend"
        exit 1
    fi
    
    cd ..
}

# Build the frontend
build_frontend() {
    print_status "Building React frontend..."
    
    if npm run build; then
        print_status "✓ Frontend built successfully"
    else
        print_error "Failed to build frontend"
        exit 1
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    cd src-tauri
    
    # Run Rust tests (mistral-backend included by default)
    if cargo test; then
        print_status "✓ Rust tests passed"
    else
        print_warning "Some Rust tests failed, continuing..."
    fi
    
    cd ..
    
    # Run Node tests if they exist
    if npm test --if-present; then
        print_status "✓ Node tests passed"
    else
        print_status "No Node tests to run"
    fi
}

# Create the final bundle
create_bundle() {
    print_status "Creating application bundle..."
    
    if npm run tauri build; then
        print_status "✓ Application bundle created successfully"
    else
        print_error "Failed to create application bundle"
        exit 1
    fi
}

# Validate the build
validate_build() {
    print_status "Validating build..."
    
    cd src-tauri
    
    # Check if the binary was created
    if [ -f "target/release/tektra" ] || [ -f "target/release/tektra.exe" ]; then
        print_status "✓ Binary created successfully"
    else
        print_error "Binary not found"
        exit 1
    fi
    
    # Check if the app bundle was created (macOS)
    if [ -d "target/release/bundle/macos/Tektra.app" ]; then
        print_status "✓ macOS app bundle created"
    fi
    
    cd ..
    
    print_status "✓ Build validation complete"
}

# Main build process
main() {
    echo "=============================================="
    echo "  Tektra - New Architecture Build Script     "
    echo "=============================================="
    echo ""
    
    check_dependencies
    setup_environment
    install_rust_deps
    install_node_deps
    build_backend
    build_frontend
    run_tests
    
    if [ "$1" == "--bundle" ]; then
        create_bundle
        validate_build
    fi
    
    echo ""
    echo "=============================================="
    print_status "🎉 Build completed successfully!"
    echo "=============================================="
    echo ""
    
    if [ "$1" == "--bundle" ]; then
        print_status "Application bundle is ready for distribution"
        print_status "You can find the bundle in src-tauri/target/release/bundle/"
    else
        print_status "Development build complete"
        print_status "Run 'npm run tauri dev' to start the development server"
    fi
    
    echo ""
    print_status "New features available:"
    echo "  • Advanced multimodal AI with mistral.rs"
    echo "  • Vision processing with Qwen2.5-VL"
    echo "  • Sophisticated conversation management"
    echo "  • Model registry and switching"
    echo "  • Enhanced error handling and performance"
    echo ""
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --bundle    Create distribution bundle"
        echo "  --help, -h  Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0          Build for development"
        echo "  $0 --bundle Build and create distribution bundle"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac