# Tektra AI Assistant

A modern desktop AI assistant built with Rust/Tauri, featuring an animated avatar and MLX-powered AI on Apple Silicon. Tektra provides real-time conversation with visual feedback through a sophisticated 2D avatar system.

## Features

- **Modern Desktop App**: Built with Tauri (Rust + React) for optimal performance
- **Animated Avatar**: Real-time lip-sync and facial animations
- **MLX Integration**: Native Apple Silicon acceleration for AI models
- **Multimodal Capabilities**:
  - Text conversation with AI models
  - Animated avatar with lip-sync
  - Voice input/output (coming soon)
  - Camera integration (coming soon)
  - Robot control via FAST tokens (coming soon)

## Architecture

- **Frontend**: React TypeScript with Canvas-based avatar rendering
- **Backend**: Rust with Tauri for native performance
- **AI Engine**: Native Rust with Candle ML framework (Apple Silicon optimized)
- **Models**: HuggingFace Hub integration with automatic caching

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) - recommended for best performance
- Rust and Cargo (latest stable)
- Node.js and npm
- No Python dependencies required!

## Installation & Usage

### Quick Start

```bash
# Clone the repository
git clone https://github.com/dirvine/tektra.git
cd tektra

# Install frontend dependencies
npm install

# Run in development mode (downloads models automatically)
npm run tauri dev

# Build for production
npm run build
npm run tauri build
```

### Development Commands

```bash
# Frontend development
npm run dev          # Start Vite dev server
npm run build        # Build frontend for production

# Tauri development  
npm run tauri dev    # Run app in development mode
npm run tauri build  # Build app for production

# Rust backend
cd src-tauri && cargo check    # Check Rust code
cd src-tauri && cargo test     # Run tests
```

### Troubleshooting Model Loading Issues

If you experience issues with model loading, try the following:

1. Run with the `--force-download` flag to re-download model files:
   ```bash
   ./run_tektra.sh --force-download --model Qwen/Qwen2.5-4B-Instruct
   ```

2. Try a smaller model if memory is limited:
   ```bash
   ./run_tektra.sh --model Qwen/Qwen2.5-4B-Instruct
   ```

3. Check for version compatibility issues in the logs. The script includes multiple fallback methods for loading models.

## Command-Line Options

- `--chat`: Start chat mode
- `--continuous`: Start continuous chat mode
- `--fine-tune`: Fine-tune model using collected robot episodes
- `--voice-input`: Use voice input (microphone) if available
- `--voice-output`: Use voice output (text-to-speech) if available
- `--no-camera`: Disable camera input
- `--text-only`: Use text input and output only, no camera
- `--model MODEL`: Specify Hugging Face model to use (default: Qwen/Qwen2.5-Omni-7B)
- `--force-download`: Force download of model even if in cache
- `--menu`: Launch interactive menu
- `--info`: Show system information

## Apple Silicon Support

Tektra is optimized for Apple Silicon Macs (M1/M2/M3 series) and automatically uses MPS (Metal Performance Shaders) for hardware acceleration, making even large models run efficiently. The script includes:

- Automatic MPS detection and configuration
- Proper handling of half-precision (float16) on Apple Silicon
- Environment variables in run_tektra.sh to optimize memory usage
- Multiple fallback mechanisms to handle edge cases in the MPS backend
- Special handling for model loading and pipeline creation on Apple Silicon

For optimal performance on MacBook Pro with 96GB RAM:
- The script defaults to Qwen2.5-Omni-7B which performs well on Apple Silicon
- Half-precision (float16) is automatically used when supported
- Memory usage is optimized for large models

## Robot Control

When Tektra detects a physical action request, it generates FAST tokens which are decoded into robot control commands. These can be sent to:

- ROS 2 (for robot arms or simulators)
- UART/GPIO interfaces (for direct hardware control)
- MQTT or WebSockets (for networked robots)

## Directory Structure

```
tektra/
├── tektra.py            # Main script
├── tektra_log.txt       # Interaction log
├── data/
│   ├── robot_episodes.json  # Fine-tuning dataset
│   └── images/              # Captured camera images 
└── models/
    ├── tektra/              # Downloaded models
    └── fine_tuned/          # Fine-tuned models
```

## License

This project is available under the MIT License.