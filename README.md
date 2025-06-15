# Tektra AI Assistant

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.6.0-orange.svg)](https://pypi.org/project/tektra/)
[![Phi-4 Integration](https://img.shields.io/badge/Phi--4-Multimodal-purple.svg)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

**Tektra AI Assistant** is an advanced AI assistant featuring **Microsoft Phi-4 Multimodal integration** with superior voice, vision, and robotics capabilities. It delivers state-of-the-art speech recognition, intelligent chat completion, and multimodal understanding with persistent conversation management and a beautiful web interface.

## ✨ Features

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

Install Tektra with pip:

```bash
# Basic installation
pip install tektra

# With all optional features
pip install tektra[all]

# With specific features
pip install tektra[ml,audio,vision]
```

### Setup and First Run

1. **Initial Setup** (first time only):
   ```bash
   tektra setup
   ```

2. **Start the Assistant**:
   ```bash
   tektra start
   ```

3. **Open Your Browser**:
   The web interface will automatically open at `http://localhost:8000`

That's it! 🎉

## 🚀 New in v0.6.0 - Phase 3.1 Features

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
# Start the server (with options)
tektra start --host 0.0.0.0 --port 8000 --browser

# Setup for first use
tektra setup

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
git clone https://github.com/tektra/tektra.git
cd tektra
pip install -e .[dev,all]
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/tektra/tektra.git
cd tektra

# Install in development mode
pip install -e .[dev,all]

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Start development server
tektra start --reload --debug
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

**Made with ❤️ by the Tektra Team**