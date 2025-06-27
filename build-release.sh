#!/bin/bash

echo "Building Tektra for release..."

# Build frontend
echo "Building frontend..."
npm run build

# Build Tauri app in release mode
echo "Building Tauri app..."
cd src-tauri
cargo build --release

echo "Build complete! Binary is at: src-tauri/target/release/tektra"
echo ""
echo "To install locally:"
echo "  cargo install --path src-tauri"
echo ""
echo "To publish to crates.io:"
echo "  1. Make sure you're logged in: cargo login"
echo "  2. From src-tauri directory: cargo publish"