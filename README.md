# Tektra AI Assistant

> **Open-Source Conversational AI Desktop App with Embedded Voice Intelligence**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-yellow.svg)](LICENSE)
[![Development](https://img.shields.io/badge/Status-In%20Development-orange.svg)](#current-status)

Tektra is an ambitious open-source AI assistant that aims to create a truly conversational desktop experience. Built with Python and Briefcase, it integrates Unmute voice AI models directly into a standalone desktop application.

## 🎯 Vision

Create a ChatGPT-like conversational experience but as a native desktop app with:
- **Natural voice interaction** - Talk to your AI like a friend
- **Embedded AI models** - No external services or API keys required
- **Conversation-first design** - Minimal UI, maximum conversation
- **Cross-platform native** - Built with Briefcase for native desktop experience

## 🚧 Current Status

### ✅ What's Working
- **Embedded Unmute Integration**: Direct model loading (STT, LLM, TTS)
- **Cross-platform Desktop App**: Toga GUI framework with Briefcase
- **Model Management**: Automatic model download and caching
- **Basic Voice Pipeline**: Audio input → transcription → response → audio output
- **Conversation Interface**: Basic chat UI with message history

### 🔧 In Development
- **Conversational UI Polish**: Smooth animations, better message rendering
- **Agent System**: Currently using mock implementations
- **Vision Features**: File upload and image analysis
- **Memory System**: Persistent conversation memory
- **Performance Optimization**: Better model loading and memory management

### 📋 Planned Features
- **Natural Language Agents**: Create AI agents through conversation
- **Multimodal Analysis**: Image and document understanding
- **Smart Model Routing**: Optimal AI selection for different tasks
- **Advanced Memory**: Context-aware conversations with learning

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 4GB+ RAM (for AI models)
- Microphone and speakers for voice features

### Installation
```bash
# Clone the repository
git clone https://github.com/dirvine/tektra.git
cd tektra

# Install with UV (recommended)
uv sync

# Or with pip
pip install -e .
```

### Running the App
```bash
# Start the desktop application
uv run python -m tektra

# Or if installed globally
tektra
```

On first run, the app will:
1. Download required AI models (~2GB)
2. Initialize voice system
3. Launch the desktop interface

## 💬 How to Use

### Basic Chat
- Type messages in the input field
- Press Enter to send
- AI responds with text and optionally voice

### Voice Conversation
- Click "Start Voice Mode" to begin listening
- Speak naturally - the AI will respond with voice
- Click "Stop Voice Mode" to return to text

### File Analysis
- Drag & drop files or use "Upload File" button
- Supported: Images, PDFs, text files
- Ask questions about the uploaded content

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tektra Desktop App                             │
├─────────────────────────────────────────────────────────────────────────┤
│  🖥️ Native Desktop Interface (Toga + Briefcase)                       │
│  ├─ Conversation UI          ├─ Voice Controls                        │
│  ├─ File Upload Interface    ├─ Settings & Status                     │
├─────────────────────────────────────────────────────────────────────────┤
│  🎤 Embedded Voice System                                             │
│  ├─ Speech-to-Text (STT)     ├─ Text-to-Speech (TTS)                 │
│  ├─ Language Model (LLM)     └─ Audio Processing                     │
├─────────────────────────────────────────────────────────────────────────┤
│  🧠 AI Processing Core                                                │
│  ├─ Model Manager            ├─ Memory System                        │
│  ├─ Agent Framework          └─ Multimodal Processor                 │
├─────────────────────────────────────────────────────────────────────────┤
│  🔧 System Integration                                                │
│  ├─ Model Downloads          ├─ Cross-platform Support               │
│  ├─ Resource Management      └─ Error Handling                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Development

### Project Structure
```
src/tektra/
├── app.py                 # Main Toga application
├── voice/                 # Voice processing components
│   ├── unmute_embedded.py # Direct Unmute integration
│   └── pipeline_*.py      # Voice processing pipelines
├── ai/                    # AI model backends
├── agents/                # Agent system (in development)
├── models/                # Model management
└── utils/                 # Configuration and utilities
```

### Key Technologies
- **Briefcase**: Cross-platform app packaging
- **Toga**: Native GUI framework
- **Unmute**: Voice AI models (STT, LLM, TTS)
- **PyTorch**: Model inference
- **asyncio**: Asynchronous processing

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📊 Performance Notes

### System Requirements
- **Minimum RAM**: 4GB (8GB recommended)
- **Storage**: 3GB for models + 1GB for app
- **CPU**: Multi-core recommended for real-time voice
- **GPU**: Optional but improves performance

### Model Sizes
- STT Model: ~500MB
- LLM Model: ~2GB (quantized)
- TTS Model: ~250MB
- Total: ~2.7GB

### Performance Tips
- Use SSD for faster model loading
- Enable GPU acceleration if available
- Close other heavy applications during use
- Consider quantized models for lower-end hardware

## 🔮 Future Roadmap

### Near-term (Next 4 weeks)
- [ ] Polish conversational UI
- [ ] Implement working agent system
- [ ] Add comprehensive memory system
- [ ] Optimize performance and resource usage

### Medium-term (Next 3 months)
- [ ] Advanced multimodal features
- [ ] Plugin system for extensions
- [ ] Mobile companion app
- [ ] Offline mode optimizations

### Long-term (6+ months)
- [ ] Distributed AI collaboration
- [ ] P2P agent networks
- [ ] Enterprise deployment features
- [ ] Advanced privacy controls

## 🤝 Community & Support

- **Issues**: [GitHub Issues](https://github.com/dirvine/tektra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dirvine/tektra/discussions)
- **Discord**: Coming soon
- **Documentation**: [docs/](docs/)

## 📄 License

This project is dual-licensed:
- **GNU Affero General Public License v3.0** - For open source projects
- **Commercial License** - For proprietary applications

See [LICENSING.md](LICENSING.md) for details.

## 🙏 Acknowledgments

Tektra builds upon excellent open-source projects:
- **Kyutai**: For the Unmute voice AI system
- **BeeWare**: For the Briefcase desktop framework
- **PyTorch**: For AI model inference
- **The Open Source Community**: For foundational technologies

---

**⚠️ Development Note**: This is an active development project. Features may change, and some functionality is still being implemented. We welcome contributions and feedback as we build the future of conversational AI.