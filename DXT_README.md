# Tektra AI Assistant - Desktop Extension (DXT)

Advanced multimodal AI assistant with voice, vision, and conversation capabilities using cutting-edge open models. Built as a Model Context Protocol (MCP) compatible Desktop Extension.

## 🌟 Features

### Core AI Capabilities
- **Multimodal Processing**: Text, vision, and document analysis
- **Advanced Conversation Management**: Context-aware conversations with memory
- **Local AI Inference**: Privacy-focused local processing with Ollama
- **Model Management**: Dynamic model switching and optimization
- **Vision Processing**: Image analysis, OCR, comparison, and composition analysis

### Technical Features
- **MCP Protocol Support**: Full Model Context Protocol compliance
- **Bundled Ollama**: Automatic Ollama management with fallback to system installation
- **Cross-Platform**: Supports macOS, Linux, and Windows
- **Performance Optimized**: GPU acceleration support with Metal, CUDA, and OpenCL
- **Extensible Architecture**: Plugin-ready tool system

### MCP Tools Available
1. **Text Generation**: Advanced text completion with sampling control
2. **Image Analysis**: Multi-type image analysis (general, detailed, OCR, scene, technical)
3. **Model Management**: List, switch, and manage AI models
4. **Conversation Sessions**: Start, continue, and manage conversation sessions
5. **Multimodal Processing**: Intelligent processing of combined text, image, and document content
6. **Image Comparison**: Compare multiple images with detailed analysis
7. **System Status**: Monitor system health and performance
8. **Performance Metrics**: Real-time usage statistics and analytics

## 🚀 Quick Start

### Installation

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/tektra/tektra.git
   cd tektra
   ./setup.sh
   ```

2. **Build from Source**:
   ```bash
   cargo build --release
   cargo build --release --features mcp-server --bin tektra-mcp
   ```

### Usage

#### As Standalone Application
```bash
./src-tauri/target/release/tektra
```

#### As MCP Server
```bash
./src-tauri/target/release/tektra-mcp
```

#### Configuration
Edit `~/.tektra/config.json` to customize settings:
```json
{
  "default_model": "qwen2.5-vl:7b",
  "auto_download_models": true,
  "enable_gpu_acceleration": true,
  "conversation": {
    "max_context_length": 32768,
    "memory_enabled": true
  },
  "mcp": {
    "enable_server": true,
    "max_concurrent_sessions": 10
  }
}
```

## 🛠️ MCP Integration

### Client Configuration

To use Tektra as an MCP server, configure your MCP client:

```json
{
  "servers": {
    "tektra": {
      "command": "/path/to/tektra/src-tauri/target/release/tektra-mcp",
      "args": [],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

### Available Tools

#### Text Generation
```json
{
  "name": "text_generation",
  "description": "Generate text completion using active AI model",
  "parameters": {
    "prompt": "Your text prompt here",
    "max_tokens": 2048,
    "temperature": 0.7
  }
}
```

#### Image Analysis
```json
{
  "name": "image_analysis", 
  "description": "Analyze images using multimodal AI",
  "parameters": {
    "image_data": "base64_encoded_image",
    "analysis_type": "general|detailed|ocr|scene|technical",
    "custom_prompt": "Optional custom analysis prompt"
  }
}
```

#### Conversation Management
```json
{
  "name": "conversation_session",
  "description": "Manage conversation sessions",
  "parameters": {
    "action": "start|continue|end",
    "session_id": "unique_session_id",
    "message": "User message for continue action",
    "persona": "AI persona for start action"
  }
}
```

## 🏗️ Architecture

### Core Components

1. **Inference Engine** (`src-tauri/src/inference/`)
   - Enhanced model registry with HuggingFace integration
   - MistralRS backend for high-performance inference
   - Token estimation and context management
   - Quantization support (Q4_K, Q5_K, Q6_K)

2. **Multimodal Processor** (`src-tauri/src/multimodal/`)
   - Advanced vision processing with multiple analysis types
   - Unified interface for multimodal content
   - Intelligent processing strategies (sequential, parallel, hierarchical)
   - Real-time performance metrics

3. **Conversation Manager** (`src-tauri/src/conversation/`)
   - Enhanced conversation flow management
   - Intelligent orchestrator for advanced dialogue
   - Context engine with compression and optimization
   - Memory systems (episodic, semantic, working)

4. **MCP Server** (`src-tauri/src/mcp/`)
   - Full MCP protocol implementation
   - Tool registry with extensible architecture
   - Transport layer with stdio/TCP/WebSocket support
   - Capability negotiation and security features

### Ollama Integration

Tektra includes sophisticated Ollama management:

- **Automatic Detection**: Checks for system Ollama installation
- **Bundled Fallback**: Downloads and manages local Ollama if needed
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Model Management**: Automatically pulls required models
- **Performance Optimization**: Optimizes for available hardware

## 🔧 Development

### Prerequisites
- Rust 1.70+ with Cargo
- 4GB+ RAM (8GB+ recommended)
- 2GB+ storage for models
- Optional: GPU with Metal/CUDA/OpenCL support

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# With MCP server features
cargo build --release --features mcp-server

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run
```

### Project Structure
```
tektra/
├── src-tauri/src/
│   ├── inference/          # AI model management
│   ├── multimodal/         # Multimodal processing
│   ├── conversation/       # Conversation management
│   ├── mcp/               # MCP server implementation
│   └── bin/               # Binary executables
├── src/                   # Frontend (React/TypeScript)
├── manifest.json          # DXT manifest
├── setup.sh              # Setup script
└── README.md             # This file
```

## 📊 Performance

### Supported Models
- **Qwen2.5-VL**: Primary multimodal model (7B, 72B)
- **Pixtral**: Vision-specialized models
- **Llama 3.2 Vision**: Meta's vision models
- **Custom Models**: Support for local GGUF models

### System Requirements
- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB+ RAM, 10GB+ storage, GPU acceleration
- **Platforms**: macOS (ARM64/Intel), Linux (x64/ARM64), Windows (x64)

### Performance Optimizations
- **Metal**: Native macOS GPU acceleration
- **CUDA**: NVIDIA GPU support on Linux/Windows
- **Quantization**: Reduced memory usage with minimal quality loss
- **Caching**: Intelligent image and context caching
- **Streaming**: Real-time response generation

## 🔒 Privacy & Security

### Local Processing
- All AI inference runs locally on your device
- No data sent to external servers
- Complete privacy and control over your data

### Security Features
- Rate limiting and request validation
- Optional authentication for MCP connections
- Secure file handling and sandboxing
- Memory-safe Rust implementation

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### Development Areas
- Audio processing implementation
- Document analysis expansion
- Additional model support
- Performance optimizations
- UI/UX improvements

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- **Homepage**: https://github.com/tektra/tektra
- **Documentation**: [Project Wiki](https://github.com/tektra/tektra/wiki)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/tektra/tektra/issues)
- **MCP Protocol**: https://github.com/anthropics/mcp
- **Desktop Extensions**: https://github.com/anthropics/dxt

## 🙏 Acknowledgments

- **Anthropic** for the MCP protocol and Desktop Extensions specification
- **Mistral AI** for the MistralRS inference engine
- **Alibaba** for the Qwen multimodal models
- **Ollama** for the local AI serving platform
- **Tauri** for the cross-platform desktop framework

---

Built with ❤️ for the open source AI community