# Tektra AI Assistant

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.9.0-orange.svg)](https://pypi.org/project/tektra/)
[![Phi-4 Integration](https://img.shields.io/badge/Phi--4-Multimodal-purple.svg)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
[![3D Avatar](https://img.shields.io/badge/3D%20Avatar-Lip--Sync-green.svg)](#-3d-avatar--lip-sync)
[![Biometric Security](https://img.shields.io/badge/Biometric-Security-red.svg)](#-biometric-security)

**Tektra AI Assistant** is an advanced AI assistant featuring **3D Avatar with Lip-Sync**, **Biometric Security**, **Microsoft Phi-4 Multimodal integration** with superior voice, vision, and robotics capabilities. It delivers state-of-the-art speech recognition, intelligent chat completion, multimodal understanding, and immersive avatar interactions with persistent conversation management and a beautiful web interface.

## ✨ Features

### 🎭 **3D Avatar & Lip-Sync** ⭐ *NEW in v0.9.0*
- **Realistic 3D Avatar**: Professional React Three Fiber-based 3D avatar rendering
- **Advanced Lip-Sync**: Real-time phoneme detection and viseme mapping for natural speech
- **14 Facial Expressions**: Happy, sad, angry, surprised, thinking, excited, and more
- **8+ Gesture Animations**: Wave, nod, point, thumbs up, shrug, and natural movements  
- **TTS Integration**: Seamless speech synthesis with synchronized lip-sync animation
- **Real-time Responsiveness**: 30fps smooth animation with professional lighting

### 🔐 **Biometric Security** ⭐ *NEW in v0.9.0*
- **Multi-Factor Authentication**: Face recognition + voice recognition + PIN
- **Encrypted User Vaults**: AES-256-CBC encryption for conversation storage
- **Query Anonymization**: PII detection and removal for external API protection
- **PBKDF2 Key Derivation**: 100,000 iterations with biometric fusion
- **Lab-Safe Privacy**: Context-aware anonymization for sensitive research environments
- **Session Management**: Secure user sessions with biometric validation

### 🧠 **Advanced AI Integration**
- **Microsoft Phi-4 Multimodal**: Primary processor with #1 OpenASR leaderboard performance
- **Intelligent Fallback**: Automatic fallback to OpenAI Whisper for reliability
- **Unified Architecture**: Single model handles speech recognition and chat completion
- **8-Language Support**: English, Chinese, German, French, Italian, Japanese, Spanish, Portuguese
- **Streaming Responses**: Real-time token-by-token generation with 128K context
- **Model Management**: Load, unload, and switch between models with automatic device detection

### 💬 **Rich Chat Interface**
- **Persistent Conversations**: Full conversation history with database storage
- **Real-time Chat**: WebSocket-based streaming chat
- **Conversation Management**: Create, search, organize, and delete conversations
- **Message Actions**: Copy, regenerate, and manage individual messages
- **Beautiful UI**: Modern, responsive web interface

### 🎤 **Superior Voice & Multimodal Capabilities**
- **Advanced Speech Recognition**: Phi-4 Multimodal with superior accuracy and 8-language support
- **Real-time Transcription**: WebSocket-based streaming with voice activity detection
- **Intelligent TTS**: Edge-TTS with 200+ neural voices and automatic language matching
- **Noise Cancellation**: WebRTC VAD with advanced audio preprocessing
- **Multimodal Understanding**: Vision, audio, and text processing in unified model
- **Camera Integration**: Real-time video processing with computer vision

### 🤖 **Robotics & Automation**
- **Robot Control**: Command and control robotic systems
- **Avatar System**: 3D avatar with expressions and gestures
- **Real-time Communication**: WebSocket-based robot communication
- **Safety Features**: Emergency stop and safety monitoring

## 🚀 Quick Start

### Installation

Install Tektra with UV (recommended):

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Basic installation
uv tool install tektra

# With all optional features
uv tool install tektra --with tektra[all]

# Alternative: Traditional pip installation also works
pip install tektra
```

### Setup and First Run

**Just run Tektra - setup happens automatically!**

```bash
tektra
```

That's it! 🎉

- **Auto-Setup**: Directories, database, and dependencies are created automatically
- **Model Loading**: Phi-4 Multimodal downloads and loads automatically on first run
- **Browser Opens**: Web interface opens automatically at `http://localhost:8000`
- **Ready to Use**: Start chatting, speaking, or using the camera immediately

### 🚀 Alternative Commands
```bash
tektra start       # Explicit start command
tektra enhance     # Enhanced setup with guided Phi-4 configuration
tektra enable-phi4 # Manually enable Phi-4 if needed
```

## 🚀 New in v0.8.0 - Streamlined Experience

### 🎯 **Zero-Setup Experience**
- **One Command Install**: `uv tool install tektra` → `tektra` → Done!
- **Automatic Setup**: No more `tektra setup` - directories, database, and models initialize automatically
- **Instant Start**: Just run `tektra` and everything works immediately
- **Smart Defaults**: Optimized configuration for immediate productivity

### 🚀 **UV Package Manager Integration**
- **Modern Installation**: UV-first approach with traditional pip fallback
- **Faster Dependencies**: UV's speed benefits for installation and development
- **Better Development**: `uv sync --all-extras` for seamless dev setup
- **Tool Integration**: Native UV tool installation support

### Microsoft Phi-4 Multimodal Integration
- **Superior Speech Recognition**: #1 performance on OpenASR leaderboard
- **Unified Processing**: Single model for both STT and chat completion
- **8-Language Audio Support**: Enhanced multilingual capabilities
- **128K Context Length**: Extended context for better understanding
- **Intelligent Fallback**: Automatic Whisper fallback for reliability

### Enhanced API Endpoints
- `/api/v1/audio/phi4/load` - Load Phi-4 Multimodal model
- `/api/v1/audio/phi4/unload` - Unload model to free memory
- `/api/v1/audio/phi4/info` - Get model status and capabilities
- Enhanced transcription and language detection with Phi-4 primary processing

### Performance Improvements
- **Faster Processing**: Unified model reduces latency
- **Better Accuracy**: Superior speech recognition and understanding
- **Memory Efficiency**: Optimized model loading and device detection
- **Real-time Streaming**: Enhanced WebSocket processing with Phi-4

## 📖 Usage

### Command Line Interface

```bash
# Start the server (default command)
tektra

# Start with custom options
tektra start --host 0.0.0.0 --port 8000 --no-browser

# Enhanced setup (server + Phi-4 guidance)
tektra enhance

# Enable Phi-4 Multimodal (superior AI)
tektra enable-phi4

# Check Phi-4 status
tektra phi4-status

# Disable Phi-4 (free memory)
tektra disable-phi4

# Show system information
tektra info

# Show version
tektra version

# Get help
tektra --help
```

### Web Interface

Once started, you can:

1. **Chat with AI**: Start conversations with intelligent responses
2. **Manage Conversations**: Browse, search, and organize your chat history
3. **Voice Interaction**: Use voice input and output
4. **Control Systems**: Manage robots, avatars, and connected devices

## 🔧 Configuration

### Environment Variables

Create a `.env` file in your working directory:

```env
# Database
DATABASE_URL=sqlite:///./tektra.db

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# AI Models
MODEL_CACHE_DIR=./models
MAX_MODEL_MEMORY_GB=8

# Audio
AUDIO_SAMPLE_RATE=16000
MAX_AUDIO_DURATION=30

# API Keys (optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## 🏗️ Architecture

Tektra AI Assistant is built with a modern, modular architecture:

```
tektra/
├── app/                    # Main application
│   ├── routers/           # API endpoints
│   ├── services/          # Business logic
│   ├── models/            # Database models
│   ├── database.py        # Database management
│   └── config.py          # Configuration
├── frontend/              # Web interface (React/Next.js)
├── cli.py                 # Command-line interface
└── server.py              # Server management
```

### Technology Stack

- **Backend**: FastAPI, SQLAlchemy, WebSockets, Python 3.9+
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Database**: SQLite (default), PostgreSQL (optional)
- **AI/ML**: Transformers, PyTorch, MLX (Apple Silicon)
- **Audio**: PyAudio, LibROSA, SoundFile
- **Vision**: OpenCV, MediaPipe

## 📦 Installation Options

### Basic Installation
```bash
pip install tektra
```

### With Machine Learning
```bash
pip install tektra[ml]
```

### With Audio Support
```bash
pip install tektra[audio]
```

### With Vision Support
```bash
pip install tektra[vision]
```

### With Robotics Support
```bash
pip install tektra[robotics]
```

### Everything Included
```bash
pip install tektra[all]
```

### Development Installation
```bash
git clone https://github.com/dirvine/tektra.git
cd tektra
uv sync --all-extras
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/dirvine/tektra.git
cd tektra

# Install in development mode with UV
uv sync --all-extras

# Set up pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Start development server
uv run tektra start --reload --debug
```

## 🐛 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# If you get permission errors
pip install --user tektra

# If you have dependency conflicts
pip install tektra --no-deps
pip install -r requirements.txt
```

**Audio Issues:**
```bash
# On macOS
brew install portaudio
pip install pyaudio

# On Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Database Issues:**
```bash
# Reset database
rm tektra.db
tektra setup
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by Saorsa Labs**