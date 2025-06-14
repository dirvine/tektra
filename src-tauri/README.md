# Tektra - AI Assistant with Avatar and Robotics Control

A modern AI assistant built with Rust and Tauri, featuring a React-based frontend with avatar visualization and robotics control capabilities.

## Features

- **Native Rust Performance**: Built with Rust backend using the Candle ML framework
- **Modern UI**: React-based frontend with Tauri integration
- **AI Chatbot**: Intelligent conversational AI with knowledge-based responses
- **Model Management**: Download and cache AI models locally
- **Avatar System**: Visual avatar with lip-sync and animation capabilities
- **Voice Input**: Speech recognition and text-to-speech
- **Vision Processing**: Camera capture and image analysis
- **Robotics Control**: Robot command and control integration
- **Cross-platform**: Runs on Windows, macOS, and Linux

## Installation

### From crates.io

```bash
# Install the binary
cargo install tektra

# The first time you run it, you may need to have Node.js installed
# as it needs to build the frontend assets during installation
```

### From source

```bash
git clone https://github.com/davidjohnirvine/tektra
cd tektra

# Build frontend assets
npm install
npm run build

# Build Rust backend
cd src-tauri
cargo build --release
```

### Prerequisites

- **For installation from crates.io**: Rust 1.70+ (Node.js not required after installation)
- **For building from source**: Rust 1.70+, Node.js 18+, npm

## Usage

After installation, simply run:

```bash
tektra
```

The application will open with a graphical interface where you can:

1. **Load AI Models**: Use the Settings panel to download and load AI models
2. **Chat**: Type messages in the chat interface to interact with the AI
3. **Voice Input**: Click the microphone button for voice interaction
4. **Capture Images**: Use the camera button for image capture
5. **Model Persistence**: Your last selected model is automatically remembered

## Technical Details

### Architecture

- **Backend**: Rust with Tauri for system integration
- **ML Framework**: Candle for native Rust ML inference
- **Frontend**: React with TypeScript
- **Models**: HuggingFace compatible models with local caching
- **Audio**: Native audio processing for TTS and STT
- **Vision**: Camera integration with image processing

### Supported Models

The application supports various AI models from HuggingFace:
- Qwen2.5-7B-Instruct-4bit
- Llama-3.2-3B-Instruct-4bit
- Phi-3.5-mini-instruct-4bit
- SmolLM2-1.7B-Instruct-4bit

## Configuration

Models are automatically cached in:
- macOS/Linux: `~/.cache/huggingface/`
- Windows: `%USERPROFILE%\.cache\huggingface\`

Application config is stored in:
- macOS/Linux: `~/.config/tektra/`
- Windows: `%APPDATA%\tektra\`

## Development

### Prerequisites

- Rust 1.70+
- Node.js 18+
- npm or yarn

### Building from source

```bash
# Clone repository
git clone https://github.com/davidjohnirvine/tektra
cd tektra

# Install frontend dependencies
npm install

# Build frontend
npm run build

# Build Rust backend
cd src-tauri
cargo build --release
```

### Running in development

```bash
# Terminal 1: Start frontend dev server
npm run dev

# Terminal 2: Start Tauri dev server
cd src-tauri
cargo tauri dev
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- Built with [Tauri](https://tauri.app/) for cross-platform desktop applications
- Powered by [Candle](https://github.com/huggingface/candle) for ML inference
- Models from [HuggingFace](https://huggingface.co/) model hub
- UI built with [React](https://react.dev/)