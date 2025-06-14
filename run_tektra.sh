#!/bin/bash

# Tektra AI Assistant - Run Script
# This script helps run the Tektra AI assistant using UV

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Error: UV is not installed. Please install UV first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Set the Python version to 3.11
PYTHON_VERSION="3.11"
TEKTRA_SCRIPT="tektra.py"

# Set environment variables for better performance on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Display help if requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Tektra AI Assistant Runner"
    echo "Usage: ./run_tektra.sh [option]"
    echo "Options:"
    echo "  --menu          Launch interactive menu (default)"
    echo "  --chat          Start voice/text chat mode with camera"
    echo "  --text-only     Start text-only mode (no voice or camera)"
    echo "  --continuous    Start continuous chat mode"
    echo "  --info          Show system information"
    echo "  --fine-tune     Run model fine-tuning with collected episodes"
    echo "  --voice-input   Use voice input if available"
    echo "  --no-camera     Disable camera input"
    echo "  --model NAME    Specify a different model (default: Qwen/Qwen2.5-Omni-7B)"
    echo "  --help          Show this help message"
    exit 0
fi

# Check for command-line arguments
if [ $# -eq 0 ]; then
    # No arguments, use interactive menu
    echo "Launching Tektra AI Assistant..."
    echo "This will download and use a real AI model."
    echo "MPS acceleration is enabled for Apple Silicon."
    uv run --python=$PYTHON_VERSION $TEKTRA_SCRIPT --menu
else
    # Pass all arguments to the script
    echo "Launching Tektra with options: $@"
    echo "MPS acceleration is enabled for Apple Silicon."
    uv run --python=$PYTHON_VERSION $TEKTRA_SCRIPT "$@"
fi