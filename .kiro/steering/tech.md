# Tektra AI Assistant - Technology Stack

## Core Technologies

### Desktop Framework
- **Briefcase** - Cross-platform app packaging and distribution
- **Toga** - Native GUI framework for Python desktop apps
- **Python 3.11+** - Core language (3.11-3.12 supported)

### AI & ML Stack
- **Transformers** - Hugging Face transformers library
- **PyTorch** - Deep learning framework for model inference
- **Unmute** - Kyutai's voice AI models (STT, LLM, TTS)
- **Qwen 2.5-VL** - Multimodal vision-language model
- **Accelerate** - Hugging Face model acceleration

### Package Management
- **UV** - Fast Python package manager (preferred)
- **pip** - Fallback package manager

### Data & Storage
- **SQLite/PostgreSQL** - Database storage
- **Redis** - Caching and session storage
- **aiosqlite** - Async SQLite operations
- **Pydantic** - Data validation and serialization

### Voice Processing
- **sounddevice** - Audio input/output
- **librosa** - Audio processing
- **scipy** - Scientific computing for audio

### Development Tools
- **pytest** - Testing framework with async support
- **black** - Code formatting
- **ruff** - Fast Python linter
- **mypy** - Type checking

## Build System

### Development Setup
```bash
# Install UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/dirvine/tektra.git
cd tektra
uv sync

# Run the application
uv run python -m tektra
```

### Testing
```bash
# Run all tests (mocked models by default)
make test

# Run Python tests only
make test-python

# Run with real AI models (requires 4GB+ RAM)
make test-python-heavy

# Run quick tests (skip slow/heavy)
make test-quick

# Run specific test categories
make test-python-unit
make test-python-integration
make test-python-performance

# Direct testing with UV (preferred method)
uv run python -m pytest tests/ -v
uv run python -m pytest tests/test_animation_system.py -v
```

### Building
```bash
# Build desktop app with Briefcase
briefcase build

# Create installer
briefcase package

# Development mode
briefcase dev
```

## Architecture Patterns

### Async-First Design
- All I/O operations use `async/await`
- Event-driven architecture for UI updates
- Non-blocking model loading and inference

### Component-Based Architecture
```python
# Modular component structure
src/tektra/
├── app.py              # Main Toga application
├── ai/                 # AI backends and processing
├── agents/             # Agent system
├── gui/                # UI components
├── voice/              # Voice processing
├── models/             # Model management
├── memory/             # Conversation memory
├── security/           # Security framework
└── utils/              # Utilities and config
```

### Error Handling
- Comprehensive exception handling with custom exception types
- Circuit breaker patterns for resilient service calls
- Graceful degradation when components fail

### Security-First
- Sandbox execution for code tools
- Permission-based access control
- Input validation and sanitization

## Configuration Management

### Environment-Based Config
```python
# Configuration hierarchy
from tektra.utils.config import AppConfig

config = AppConfig()
model_name = config.get("llm_model_name", "microsoft/Phi-3-mini-4k-instruct")
```

### Model Configuration
```python
# Model settings in pyproject.toml or environment
TEKTRA_MODEL_CACHE_DIR = "~/.tektra/models"
TEKTRA_MAX_MEMORY_MB = 4096
TEKTRA_ENABLE_GPU = true
```

## Performance Considerations

### Model Management
- Automatic model downloading and caching
- Memory-efficient model loading
- GPU acceleration when available
- Model quantization support

### Resource Requirements
- **Minimum RAM**: 4GB (8GB recommended)
- **Storage**: 3GB for models + 1GB for app
- **CPU**: Multi-core recommended for real-time voice
- **GPU**: Optional but improves performance

### Optimization Strategies
- Progressive model loading with user feedback
- Efficient memory management
- Caching strategies for frequent operations
- Background processing for non-critical tasks

## Development Workflow

### Code Style
- **Black** for code formatting (88 character line length)
- **Ruff** for linting with modern Python practices
- **Type hints** required for all public APIs
- **Docstrings** for all modules, classes, and functions

### Testing Strategy
- **Unit tests** for individual components
- **Integration tests** for component interactions
- **Performance tests** for resource usage
- **Mock models** by default to prevent system overheating
- **Heavy model tests** available with explicit flag

### Git Workflow
- Feature branches for new development
- Pull requests for code review
- Automated testing on CI/CD
- Semantic versioning for releases

## Deployment Options

### Desktop Distribution
```bash
# Cross-platform installers via Briefcase
briefcase package

# Platform-specific builds
briefcase package macos
briefcase package windows
briefcase package linux
```

### Docker Support
```bash
# Development container
docker-compose up

# Production deployment
docker build -t tektra:latest .
```

### Kubernetes
```yaml
# Production Kubernetes deployment available
# See k8s/ directory for manifests
```

## Common Commands

**IMPORTANT: Always use `uv run` instead of plain `python` commands in this project.**

### Development
```bash
# Start development server
uv run python -m tektra

# Run tests (preferred method)
uv run python -m pytest tests/ -v
uv run python -m pytest tests/test_animation_system.py -v

# Alternative: Use make commands (which internally use uv)
make test

# Format code
uv run black src/ tests/
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Install dependencies
uv sync

# Add new dependencies
uv add package_name

# Run any Python script
uv run python script_name.py
```

### Model Management
```bash
# Models are automatically downloaded on first run
# Cache location: ~/.tektra/models/
# Clear cache: rm -rf ~/.tektra/models/
```

### Debugging
```bash
# Enable debug logging
export TEKTRA_LOG_LEVEL=DEBUG
uv run python -m tektra

# Run with profiling
uv run python -m cProfile -o profile.stats -m tektra
```

## Integration Points

### Voice Pipeline
- **STT**: Speech-to-text via Unmute models
- **LLM**: Language model inference
- **TTS**: Text-to-speech synthesis
- **Audio I/O**: Real-time audio processing

### Model Backends
- **Local models**: Embedded Transformers models
- **API backends**: Optional external API support
- **Smart routing**: Automatic model selection

### Memory System
- **MemOS integration**: Advanced conversation memory
- **SQLite storage**: Local conversation history
- **Vector search**: Semantic memory retrieval