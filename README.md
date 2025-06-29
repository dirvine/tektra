# Tektra - AI Voice Assistant

🚀 **A voice-interactive AI assistant powered by Google's Gemma-3n model running locally on your machine**

## Features

✅ **Local AI** - Runs Gemma-3n model locally with Metal acceleration on Apple Silicon  
✅ **Voice Interaction** - Native audio recording with future speech-to-text support  
✅ **Smart Conversations** - Context-aware responses with chat history  
✅ **Beautiful UI** - Modern, gradient-based interface with real-time feedback  
✅ **Self-Contained** - Automatic model downloading and caching  
✅ **Privacy-First** - Everything runs locally on your machine  

## Architecture

- **Frontend**: React + TypeScript with Vite
- **Backend**: Rust with Tauri framework
- **AI Engine**: GGUF models for efficient inference (Gemma-3n E2B)
- **Audio**: Native audio recording with future STT/TTS support
- **Model**: Google Gemma-3n E2B (2.79GB) - automatically downloaded on first run

## Installation

### Pre-built Binaries (Recommended)

Download the latest pre-built binary for your platform from the [GitHub Releases](https://github.com/dirvine/tektra/releases) page.

### From Source

**Note**: Tektra is a desktop application that requires both Rust backend and React frontend. `cargo install` alone will not work properly.

```bash
# Clone the repository
git clone https://github.com/dirvine/tektra
cd tektra

# Install frontend dependencies
npm install

# Build the complete application
npm run tauri:build

# The built application will be in src-tauri/target/release/bundle/
```

### Cargo Install (Limited - Backend Only)

⚠️ **Not recommended for end users** - This only installs the Rust backend without the frontend UI:

```bash
cargo install tektra
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/dirvine/tektra
cd tektra

# Install frontend dependencies
npm install

# Run in development mode
tektra dev
# OR
npm run tauri dev
```

## Usage

### Running the Application

Simply run:
```bash
tektra
```

### CLI Commands

```bash
tektra --help     # Show help
tektra --version  # Show version information
tektra dev        # Run in development mode with hot reload
```

### In-App Features

1. **Text Chat**: Type messages and get intelligent responses from Gemma-3n
2. **Voice Input**: Click the microphone button to record audio (STT coming soon)
3. **Voice Output**: Enable auto-speech in settings for spoken responses
4. **Progress Tracking**: Visual progress bar shows model download status
5. **Settings**: Customize preferences and view model information

## Project Structure

```
tektra/
├── src/
│   └── main.rs          # Rust backend with built-in AI
├── icons/               # Application icons
├── index.html           # Main frontend interface
├── main.js              # Frontend JavaScript
├── package.json         # Node.js dependencies
├── Cargo.toml           # Rust dependencies
├── tauri.conf.json      # Tauri configuration
├── vite.config.js       # Build configuration
├── entitlements.plist   # macOS permissions
└── CLAUDE.md            # Development guidelines
```

## Building from Source

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- Node.js 16+ and npm
- macOS 11+ (for Metal acceleration, other platforms supported)

### Build Steps

1. Clone and enter the repository:
```bash
git clone https://github.com/dirvine/tektra
cd tektra
```

2. Install frontend dependencies:
```bash
npm install
```

3. Build for release:
```bash
./build-release.sh
```

4. Install locally:
```bash
cargo install --path src-tauri
```

## Models

Tektra uses the Gemma-3n E2B model (2.79GB) from Google, which is automatically downloaded on first run. The model is cached in `~/.cache/huggingface/hub/` for subsequent uses.

### Model Details
- **Name**: Gemma-3n E2B (2 billion parameters)
- **Size**: 2.79GB (4-bit quantized GGUF format)
- **Performance**: Optimized for Apple Silicon with Metal acceleration
- **Source**: Automatically downloaded from HuggingFace Hub

## Development

This project follows specification-driven development:

1. **Read CLAUDE.md** for detailed development guidelines
2. **Use UV** for Python dependencies (if needed)
3. **Test thoroughly** before committing changes
4. **Follow Rust best practices** - no unwrap() in production

## Migration from Previous Versions

This is a complete rewrite of Project Tektra as a native desktop application:

- **Previous**: Python FastAPI backend + Next.js frontend
- **Current**: Rust Tauri desktop application
- **Benefits**: Self-contained, offline, native performance, simplified deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the development guidelines in CLAUDE.md
4. Submit a pull request

## Publishing to Crates.io

To publish this package to crates.io:

1. Make sure you're logged in:
```bash
cargo login
```

2. From the src-tauri directory:
```bash
cd src-tauri
cargo publish
```

Note: The frontend assets are bundled with the binary during the build process.

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Credits

Project Tektra - Built with ❤️ using Rust, Tauri, and modern web technologies.

## Release History

For previous releases and development history, see the RELEASE_NOTES files in this repository.