# Project Tektra - Local AI Voice Assistant

🚀 **A completely self-contained desktop AI assistant that works entirely offline**

## Features

✅ **100% Offline** - No internet required after installation  
✅ **Self-Contained** - Built-in AI model, no external dependencies  
✅ **Voice Enabled** - Speech recognition and text-to-speech  
✅ **Native Performance** - Built with Rust and Tauri  
✅ **Modern UI** - Beautiful interface with real-time chat  
✅ **Cross-Platform** - Works on macOS, Windows, and Linux  

## Architecture

- **Frontend**: Modern web technologies (HTML5, JavaScript, CSS3)
- **Backend**: Rust with built-in AI assistant
- **Framework**: Tauri for native desktop performance
- **AI**: Local pattern-matching assistant with intelligent responses
- **Voice**: Web Speech API for recognition and synthesis

## Quick Start

### Prerequisites

- Node.js (v18 or later)
- Rust (latest stable)
- System microphone (for voice input)

### Installation

```bash
# Install frontend dependencies
npm install

# Run in development mode
cargo tauri dev

# Build for production
cargo tauri build
```

### Usage

1. **Text Chat**: Type messages and get intelligent responses
2. **Voice Input**: Click the microphone button to speak
3. **Voice Output**: Enable auto-speech in settings for spoken responses
4. **Settings**: Customize AI model and voice preferences

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

## Built-in AI Assistant

The local AI assistant includes:

- **Intelligent Responses**: Context-aware conversation
- **Multiple Personalities**: Varied response styles
- **Pattern Matching**: Recognizes greetings, questions, and requests
- **Memory**: Maintains conversation history
- **Fast Performance**: Instant responses without network calls

## Voice Features

- **Speech Recognition**: Uses Web Speech API
- **Text-to-Speech**: Natural voice synthesis
- **Microphone Controls**: Visual feedback and error handling
- **Platform Support**: Works on all major browsers/platforms

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

Project Tektra - Built with ❤️ using Rust, Tauri, and modern web technologies.

## Release History

For previous releases and development history, see the RELEASE_NOTES files in this repository.