#!/bin/bash

# Install MLX dependencies for Tektra AI Assistant
echo "Installing MLX dependencies for Tektra..."

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: MLX is optimized for Apple Silicon (M-series chips)"
    echo "This may not work optimally on non-Apple Silicon systems"
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    echo "Please install Python 3 first"
    exit 1
fi

# Install MLX and related packages
echo "Installing MLX packages..."
pip3 install mlx mlx-lm

echo "MLX installation complete!"
echo "You can now use Tektra with MLX models on Apple Silicon"