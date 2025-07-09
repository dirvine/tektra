# Tektra AI Assistant

> **A sophisticated voice-interactive AI assistant with multimodal capabilities, intelligent agents, and comprehensive memory system**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-yellow.svg)](LICENSE)
[![Built with Briefcase](https://img.shields.io/badge/Built%20with-Briefcase-green.svg)](https://briefcase.readthedocs.io/)

Tektra combines cutting-edge AI technologies to deliver a comprehensive, native desktop AI assistant with voice interaction, vision processing, intelligent agents, and advanced memory capabilities.

## 🌟 Key Features

### 🎯 **Dual AI Brain Architecture**
- **Conversational AI**: Kyutai Unmute for ultra-low latency voice conversations (STT-2.6B-EN)
- **Analytical AI**: Qwen 2.5-VL for complex reasoning, vision analysis, and analytical tasks
- **Smart Router**: Intelligent query routing between conversational and analytical systems
- **Hybrid Processing**: Seamless integration of both AI systems for complex queries

### 🤖 **Intelligent Agent System**
- **Natural Language Agent Creation**: Describe what you want - get a working agent
- **Multi-Type Agents**: CODE, TOOL_CALLING, HYBRID, MONITOR, WORKFLOW agents
- **Agent Builder**: Powered by SmolAgents framework for robust code execution
- **Agent Capabilities**: Web search, file access, APIs, databases, scheduling, and more
- **Sandboxed Execution**: Secure agent execution with Docker and process isolation

### 🧠 **Advanced Memory System**
- **Persistent Memory**: Long-term memory storage with SQLite backend
- **Context-Aware Conversations**: Remembers past interactions and learns from experience
- **Memory Sharing**: Inter-agent memory sharing for collaborative intelligence
- **Semantic Search**: Find relevant memories with intelligent search and relevance scoring
- **Memory Types**: Conversation, agent context, task results, and custom memory types

### 🎤 **Voice Interaction**
- **Real-time Voice Conversations**: Continuous voice interaction with Kyutai Unmute
- **Speech-to-Text**: Accurate transcription with STT-2.6B-EN
- **Text-to-Speech**: Natural speech synthesis with Kyutai TTS 2B
- **Push-to-Talk & Continuous Listening**: Flexible voice activation modes
- **Audio Enhancement**: Noise reduction and quality improvement

### 👁️ **Vision & Multimodal Processing**
- **Advanced Vision Analysis**: Powered by Qwen2.5-VL for image understanding
- **Multi-Format Support**: JPEG, PNG, GIF, WebP, BMP with automatic optimization
- **Camera Integration**: Real-time video feed processing
- **Vision-Text Integration**: Combined text and image analysis in single queries
- **Image Enhancement**: Automatic resizing, contrast enhancement, and quality improvement

### 📄 **Document Processing**
- **Multi-Format Support**: PDF, DOCX, TXT, Markdown, JSON, YAML, CSV, LOG files
- **Intelligent Text Extraction**: Advanced document understanding with metadata
- **Drag-and-Drop Upload**: Seamless file processing through GUI
- **Asynchronous Processing**: Concurrent processing of multiple documents

### 🖥️ **Native Desktop Experience**
- **Cross-Platform**: macOS, Linux, Windows with native look and feel
- **Modern UI**: Built with Briefcase and Toga for native desktop experience
- **Real-time Status**: Live indicators for models, services, and processing
- **Responsive Design**: Adaptive interface with progress tracking and status updates

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tektra AI Assistant                         │
├─────────────────────────────────────────────────────────────────┤
│  🖥️ Native Desktop UI (Briefcase + Toga)                      │
│  ├─ Chat Interface      ├─ Agent Dashboard    ├─ Controls      │
│  ├─ File Upload         ├─ Memory Viewer      ├─ Settings      │
│  └─ Status Indicators   └─ Performance Stats  └─ Voice Panel   │
├─────────────────────────────────────────────────────────────────┤
│  🤖 Agent System                                               │
│  ├─ Agent Builder (Natural Language → Working Agent)           │
│  ├─ Agent Runtime (Sandboxed Execution)                        │
│  ├─ Agent Registry (Management & Storage)                      │
│  └─ Agent Templates (Pre-built Common Agents)                  │
├─────────────────────────────────────────────────────────────────┤
│  🧠 Memory System                                              │
│  ├─ Memory Manager (SQLite Backend)                            │
│  ├─ Memory Types (Conversation, Agent Context, Task Results)   │
│  ├─ Memory Search (Semantic Search with Relevance Scoring)     │
│  └─ Memory Sharing (Inter-agent Memory Collaboration)          │
├─────────────────────────────────────────────────────────────────┤
│  🎯 AI Processing Core                                         │
│  ├─ Smart Router (Query Analysis & Routing)                    │
│  ├─ Qwen 2.5-VL Backend (Analytical & Vision AI)              │
│  ├─ Kyutai Unmute (Conversational AI)                         │
│  └─ Multimodal Processor (Text, Image, Audio Integration)      │
├─────────────────────────────────────────────────────────────────┤
│  🎤 Voice & Audio                                              │
│  ├─ Voice Pipeline (STT → LLM → TTS)                          │
│  ├─ Audio Capture (Microphone Integration)                     │
│  ├─ Audio Playback (Speaker Integration)                       │
│  └─ WebSocket Communication (Real-time Audio Streaming)        │
├─────────────────────────────────────────────────────────────────┤
│  👁️ Vision & Multimodal                                       │
│  ├─ Vision Pipeline (Image Analysis & Understanding)           │
│  ├─ Document Processor (Multi-format Document Processing)      │
│  ├─ Image Enhancement (Optimization & Quality Improvement)     │
│  └─ Camera Integration (Real-time Video Processing)            │
├─────────────────────────────────────────────────────────────────┤
│  🔧 Infrastructure                                             │
│  ├─ Docker Integration (Service Management)                    │
│  ├─ Model Registry (AI Model Management)                       │
│  ├─ Performance Monitor (Metrics & Analytics)                  │
│  └─ Configuration Manager (Settings & Environment)             │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Core Components

### 1. **AI Processing Core**
```python
# Smart Query Router
router = SmartRouter(qwen_backend, unmute_backend)
response = await router.process_query("Analyze this image and explain what you see")

# Qwen 2.5-VL for analytical tasks
qwen = QwenBackend()
result = await qwen.process_vision_query("What's in this image?", image_data)

# Kyutai Unmute for conversations
unmute = UnmuteBackend()
await unmute.start_conversation("Hello, how are you today?")
```

### 2. **Agent System**
```python
# Create agent from natural language
agent_builder = AgentBuilder(qwen_backend)
agent = await agent_builder.create_agent_from_description(
    "Create a GitHub monitor that alerts me about new issues and PRs"
)

# Deploy and run agent
runtime = AgentRuntime(memory_manager=memory_manager)
agent_id = await runtime.deploy_agent(agent)
```

### 3. **Memory System**
```python
# Memory-enhanced conversations
memory_manager = TektraMemoryManager()
await memory_manager.add_conversation(
    user_message="I like Python programming",
    assistant_response="Great! Python is versatile and beginner-friendly.",
    user_id="user123"
)

# Search memories
memories = await memory_manager.search_memories(
    query="Python programming",
    user_id="user123"
)
```

### 4. **Voice Integration**
```python
# Voice conversation pipeline
voice_pipeline = VoicePipeline(unmute_backend)
await voice_pipeline.start_conversation()

# Process voice input
audio_data = await voice_pipeline.capture_audio()
response = await voice_pipeline.process_voice_query(audio_data)
```

### 5. **Vision Processing**
```python
# Vision analysis
vision_processor = VisionProcessor(qwen_backend)
result = await vision_processor.analyze_image(
    image_path="photo.jpg",
    prompt="Describe what you see in detail"
)

# Document processing
doc_processor = DocumentProcessor()
content = await doc_processor.process_document("document.pdf")
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- macOS, Linux, or Windows
- Docker (for Kyutai Unmute services)
- 4GB+ RAM (8GB+ recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/dirvine/tektra.git
cd tektra
```

2. **Install dependencies:**
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

3. **Run the application:**
```bash
# Using UV
uv run python demo.py

# Or using Python
python demo.py
```

### Building Native Application

```bash
# Install Briefcase if not already installed
pip install briefcase

# Create native application
briefcase create
briefcase build
briefcase run
```

## 📱 Platform-Specific Features

### macOS
- **Native Audio**: AVFoundation and CoreAudio integration
- **Camera Access**: Native camera integration with permissions
- **App Store Ready**: Signed and notarized application bundles
- **System Integration**: Native notifications and system tray

### Linux
- **Audio Systems**: ALSA and PulseAudio support
- **Docker Integration**: Automatic service management
- **Desktop Integration**: .desktop files and system notifications
- **Hardware Access**: Camera and microphone permissions

### Windows
- **Audio Support**: Windows Audio Session API
- **System Integration**: Windows notifications and taskbar
- **Hardware Access**: Camera and microphone permissions
- **Windows Store**: Package-ready for Windows Store

## 🛠️ Agent Capabilities

Tektra's agent system supports a wide range of capabilities:

### **Built-in Agent Types**
- **CODE Agents**: Execute Python code in sandboxed environments
- **TOOL_CALLING Agents**: Use JSON-based tool calling for structured tasks
- **HYBRID Agents**: Combine code execution with tool calling
- **MONITOR Agents**: Continuous monitoring with scheduled execution
- **WORKFLOW Agents**: Multi-step processes with state management

### **Agent Capabilities**
- 🌐 **Web Search**: Internet research and data gathering
- 📁 **File Access**: File system operations and document processing
- 🗄️ **Database Operations**: Data storage and retrieval
- 🔗 **API Integration**: External service integration
- 💻 **Code Execution**: Python code execution in secure environments
- 📧 **Email Operations**: Email sending and management
- ⏰ **Scheduling**: Time-based and event-driven task execution
- 📊 **Data Analysis**: Statistical analysis and data processing
- 🖼️ **Image Processing**: Computer vision and image manipulation
- 🔔 **Notifications**: Alert and notification systems

### **Example Agents**
```python
# GitHub Monitor Agent
"Monitor my GitHub repositories for new issues and pull requests, 
send me notifications when activity occurs"

# Data Analysis Agent
"Analyze my CSV files and create visualizations showing trends 
and patterns in the data"

# Document Processor Agent
"Process incoming PDF documents, extract key information, and 
organize it into a structured database"

# System Monitor Agent
"Monitor system resources and alert me when CPU or memory usage 
exceeds specified thresholds"
```

## 🧠 Memory System Features

### **Memory Types**
- **Conversation Memory**: Chat history and context
- **Agent Context**: Agent-specific knowledge and state
- **Task Results**: Execution results and outcomes
- **User Preferences**: Personal settings and preferences
- **System Events**: Application events and logs
- **Learned Facts**: Extracted knowledge and insights

### **Memory Configuration**
```python
# Configure memory for agents
agent_spec = AgentSpecification(
    memory_enabled=True,
    memory_context_limit=20,           # Remember last 20 interactions
    memory_importance_threshold=0.7,   # Store important memories (0.7+)
    memory_retention_hours=720,        # Keep memories for 30 days
    persistent_memory=True,            # Persist across restarts
    memory_sharing_enabled=True        # Allow memory sharing with other agents
)
```

### **Memory Search**
```python
# Search memories with semantic understanding
memories = await memory_manager.search_memories(
    query="Python programming help",
    user_id="user123",
    max_results=10,
    min_relevance=0.5,
    time_window_hours=168  # Last week
)
```

## 🎤 Voice Interaction

### **Voice Features**
- **Real-time Conversations**: Continuous voice interaction
- **High-Quality STT**: Kyutai STT-2.6B-EN for accurate transcription
- **Natural TTS**: Kyutai TTS 2B for human-like speech synthesis
- **Voice Activity Detection**: Automatic speech detection
- **Noise Reduction**: Audio enhancement and quality improvement

### **Voice Modes**
```python
# Push-to-Talk Mode
voice_pipeline.set_mode("push_to_talk")
await voice_pipeline.start_listening()

# Continuous Listening Mode
voice_pipeline.set_mode("continuous")
await voice_pipeline.start_conversation()

# Voice-only Mode (no display)
voice_pipeline.set_mode("voice_only")
```

## 👁️ Vision & Multimodal

### **Vision Capabilities**
- **Image Analysis**: Detailed image understanding and description
- **Object Detection**: Identify and locate objects in images
- **Text Recognition**: Extract text from images (OCR)
- **Color Analysis**: Dominant colors and color palette extraction
- **Scene Understanding**: Context and scene interpretation

### **Supported Formats**
- **Images**: JPEG, PNG, GIF, WebP, BMP, TIFF
- **Documents**: PDF, DOCX, TXT, Markdown, JSON, YAML, CSV, LOG
- **Video**: MP4, AVI, MOV (processing capabilities)
- **Audio**: WAV, MP3, OGG (voice processing)

### **Vision Processing Example**
```python
# Advanced image analysis
result = await vision_processor.analyze_image(
    image_path="photo.jpg",
    analysis_types=["objects", "text", "colors", "scene"],
    detail_level="high"
)

# Multi-image comparison
comparison = await vision_processor.compare_images(
    image1="before.jpg",
    image2="after.jpg",
    comparison_type="changes"
)
```

## 📊 Performance & Monitoring

### **Performance Metrics**
- **Response Times**: Real-time inference timing
- **Token Counts**: Input/output token tracking
- **Memory Usage**: System and model memory monitoring
- **GPU Utilization**: Hardware acceleration tracking
- **Service Health**: Real-time service status monitoring

### **Analytics Dashboard**
```python
# Get performance statistics
stats = await performance_monitor.get_stats()
print(f"Average response time: {stats.avg_response_time}ms")
print(f"Total tokens processed: {stats.total_tokens}")
print(f"Memory usage: {stats.memory_usage}MB")
```

## 🔧 Configuration

### **Environment Configuration**
```bash
# .env file
TEKTRA_MODEL_PATH=/path/to/models
TEKTRA_MEMORY_PATH=/path/to/memory
TEKTRA_LOG_LEVEL=INFO
TEKTRA_ENABLE_GPU=true
TEKTRA_VOICE_ENABLED=true
TEKTRA_CAMERA_ENABLED=true
```

### **Model Configuration**
```python
# Qwen model configuration
qwen_config = QwenModelConfig(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_bits=None,  # Disable for compatibility
    max_memory_gb=8.0,
    device_map="auto",
    torch_dtype="float16"
)
```

## 🧪 Testing

Tektra includes comprehensive test coverage across all components with unit tests, integration tests, performance benchmarks, and property-based testing.

### **Run All Tests**

```bash
# Frontend Tests (JavaScript/TypeScript)
npm test                                    # Run all frontend tests
npm run test:coverage                       # Run with coverage report
npm run test:ui                            # Run with Vitest UI

# Python Backend Tests
uv run python -m pytest tests/ -v          # Run all Python tests
uv run python -m pytest tests/ -v --cov    # Run with coverage report

# Node.js/DXT Extension Tests
cd dxt-extension/server
npm test                                    # Run all Node.js tests
npm run test:coverage                      # Run with coverage report
```

### **Test Categories**

#### **Frontend Tests** (Vitest + React Testing Library)
```bash
# Unit tests for message formatting and security
npm test src/test/unit/

# Property-based tests with fast-check
npm test src/test/property/

# Integration tests with mocked APIs
npm test src/test/integration/
```

#### **Python Backend Tests** (pytest + hypothesis)
```bash
# Unit Tests - Basic functionality
uv run python -m pytest tests/test_unit_*.py -v

# Integration Tests - Memory system
uv run python -m pytest tests/test_integration_memory_core.py -v

# Performance Tests - Benchmarking and timing
uv run python -m pytest tests/test_performance_memory.py -v -s

# Property-Based Tests - Edge case discovery
uv run python -m pytest tests/test_property_based_simple.py -v
```

#### **Node.js Tests** (Jest)
```bash
# Basic functionality tests
cd dxt-extension/server
npm test test/basic-functionality.test.js

# MCP integration tests
npm test test/mcp-integration.test.js

# Server component tests
npm test test/TektraAIServer.test.js
```

### **Quick Test Commands**

```bash
# Using Makefile (recommended)
make test                    # Run all tests
make test-python            # Python tests only
make test-frontend          # Frontend tests only
make test-node             # Node.js tests only
make test-coverage         # All tests with coverage
make test-quick            # Fast tests only (skip slow)
make help                  # Show all test commands

# Or manually without Makefile:
npm test && \
uv run python -m pytest tests/ -v && \
(cd dxt-extension/server && npm test)

# Run only fast tests (skip performance/integration)
uv run python -m pytest tests/ -v -m "not slow"

# Run tests with coverage reports
npm run test:coverage && \
uv run python -m pytest tests/ -v --cov=src --cov-report=html && \
(cd dxt-extension/server && npm run test:coverage)

# Watch mode for development
npm run test:watch  # Frontend
uv run python -m pytest tests/ -v --watch  # Python (with pytest-watch)
(cd dxt-extension/server && npm run test:watch)  # Node.js
```

### **Test Coverage Targets**
- **Frontend**: 80% lines, 80% functions, 70% branches
- **Python**: 95%+ integration test coverage
- **Node.js**: 70% across all metrics

### **Performance Benchmarks**
The test suite validates performance requirements:
- Memory insertion: < 50ms per entry
- Memory retrieval: < 10ms
- Search operations: < 100ms
- Concurrent operations: < 50ms per entry

### **Property-Based Testing**
Automated edge case discovery using:
- **hypothesis** (Python): Random data generation for edge cases
- **fast-check** (JavaScript): Property-based testing for frontend

### **Test Reports**
Test results and coverage reports are generated in:
- `coverage/` - Frontend coverage reports
- `htmlcov/` - Python coverage reports  
- `dxt-extension/server/coverage/` - Node.js coverage reports

## 📦 Building & Distribution

### **Create Distribution**
```bash
# Build native application
briefcase create
briefcase build

# Package for distribution
briefcase package

# Platform-specific builds
briefcase build macOS
briefcase build linux
briefcase build windows
```

### **Distribution Formats**
- **macOS**: .app bundle, .dmg installer
- **Linux**: .AppImage, .deb package
- **Windows**: .exe installer, .msi package

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/dirvine/tektra.git
cd tektra

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest tests/

# Code formatting
uv run black src/
uv run isort src/
uv run ruff check src/
```

## 📝 License

This project is licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🙏 Acknowledgments

- **Kyutai**: For the Unmute voice conversation system
- **Qwen Team**: For the Qwen 2.5-VL multimodal model
- **HuggingFace**: For the Transformers library and model hosting
- **BeeWare**: For the Briefcase application framework
- **SmolAgents**: For the agent execution framework

## 📞 Support

- **Documentation**: [docs.tektra.ai](https://docs.tektra.ai)
- **Issues**: [GitHub Issues](https://github.com/dirvine/tektra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dirvine/tektra/discussions)
- **Community**: [Discord Server](https://discord.gg/tektra)

---

**Tektra AI Assistant** - Where voice meets intelligence, and agents meet memory. Built with ❤️ for the future of human-AI interaction.