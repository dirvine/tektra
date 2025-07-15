# Tektra AI Assistant - Project Structure

## Root Directory Layout

```
tektra/
├── src/tektra/           # Main application source code
├── tests/                # Test suite
├── docs/                 # Documentation
├── docker/               # Docker configuration
├── k8s/                  # Kubernetes manifests
├── scripts/              # Deployment and utility scripts
├── unmute/               # Embedded Unmute voice AI system
├── pyproject.toml        # Python project configuration
├── Makefile              # Build and test commands
├── docker-compose.yml    # Development environment
└── README.md             # Project overview
```

## Source Code Organization

### Main Application (`src/tektra/`)

```
src/tektra/
├── __init__.py           # Package initialization and exports
├── __main__.py           # CLI entry point
├── app.py                # Main Toga desktop application
├── ai/                   # AI processing backends
│   ├── __init__.py
│   ├── simple_llm.py     # Basic LLM backend
│   ├── qwen_backend.py   # Qwen multimodal model
│   ├── smart_router.py   # AI model routing
│   ├── multimodal.py     # Multimodal processing
│   └── model_validator.py
├── agents/               # Agent system
│   ├── __init__.py
│   ├── builder.py        # Agent creation
│   ├── registry.py       # Agent management
│   ├── runtime.py        # Agent execution
│   ├── simple_agent.py   # Basic agent implementation
│   ├── simple_runtime.py # Simple execution runtime
│   └── templates.py      # Agent templates
├── config/               # Configuration management
│   ├── __init__.py
│   └── production_config.py
├── core/                 # Core system components
│   ├── __init__.py
│   ├── tektra_system.py  # Main system orchestrator
│   ├── error_handling.py # Global error handling
│   └── deployment_manager.py
├── data/                 # Data storage and management
│   ├── __init__.py
│   ├── storage.py        # Data persistence
│   └── vector_db.py      # Vector database
├── gui/                  # User interface components
│   ├── __init__.py
│   ├── chat_panel.py     # Chat interface
│   ├── agent_panel.py    # Agent management UI
│   ├── progress_dialog.py # Progress indicators
│   ├── progress_overlay.py
│   ├── startup_dialog.py # App startup flow
│   ├── themes.py         # UI theming
│   ├── feature_discovery.py # Progressive feature discovery
│   └── markdown_renderer.py # Message rendering
├── memory/               # Conversation memory system
│   ├── __init__.py
│   ├── conversation_memory.py
│   ├── memory_manager.py
│   ├── memory_config.py
│   ├── memory_types.py
│   └── memos_integration.py # MemOS integration
├── models/               # AI model management
│   ├── __init__.py
│   ├── model_interface.py # Model abstraction
│   ├── model_manager.py  # Model lifecycle
│   └── model_updater.py  # Model updates
├── performance/          # Performance optimization
│   ├── __init__.py
│   ├── cache_manager.py  # Multi-level caching
│   ├── memory_manager.py # Memory optimization
│   ├── model_pool.py     # Model pooling
│   ├── optimizer.py      # Performance tuning
│   ├── performance_monitor.py
│   ├── resource_pool.py  # Resource management
│   └── task_scheduler.py # Task scheduling
├── security/             # Security framework
│   ├── __init__.py
│   ├── context.py        # Security contexts
│   ├── permissions.py    # Permission system
│   ├── sandbox.py        # Code execution sandbox
│   ├── advanced_sandbox.py
│   ├── validator.py      # Input validation
│   ├── tool_validator.py # Tool security
│   ├── monitor.py        # Security monitoring
│   ├── audit.py          # Audit logging
│   └── consent_framework.py
├── utils/                # Utilities and helpers
│   ├── __init__.py
│   ├── config.py         # Configuration utilities
│   ├── file_utils.py     # File operations
│   ├── download_progress.py # Download tracking
│   ├── docker_utils.py   # Docker integration
│   └── *_progress.py     # Various progress trackers
└── voice/                # Voice processing
    ├── __init__.py
    ├── pipeline_embedded.py # Embedded voice pipeline
    ├── pipeline_mock.py  # Mock for testing
    ├── unmute_embedded.py # Unmute integration
    ├── unmute_client.py  # Unmute client
    ├── services.py       # Voice services
    └── voice_patterns.py # Voice interaction patterns
```

## Testing Structure (`tests/`)

```
tests/
├── __init__.py
├── conftest.py           # Pytest configuration and fixtures
├── test_*.py             # Individual test files
├── e2e/                  # End-to-end tests
│   ├── test_complete_system_integration.py
│   ├── test_performance_benchmarks.py
│   ├── test_production_deployment.py
│   └── test_security_compliance.py
└── performance/          # Performance tests
    ├── conftest.py
    ├── test_agent_performance.py
    ├── test_ai_backend_performance.py
    ├── test_integration_performance.py
    └── test_memory_performance.py
```

## Documentation Structure (`docs/`)

```
docs/
├── README.md             # Documentation index
├── ARCHITECTURE.md       # System architecture
├── API_REFERENCE.md      # API documentation
├── PERFORMANCE_GUIDE.md  # Performance optimization
├── PRODUCTION_DEPLOYMENT.md # Deployment guide
├── SECURITY_GUIDE.md     # Security documentation
└── TROUBLESHOOTING.md    # Common issues
```

## Configuration Files

### Python Project (`pyproject.toml`)
- **Build system**: Briefcase configuration
- **Dependencies**: Core and optional dependencies
- **Development tools**: Black, Ruff, MyPy, Pytest configuration
- **Entry points**: CLI and GUI entry points
- **Platform-specific**: macOS, Windows, Linux requirements

### Development Tools
- **Makefile**: Test commands and build automation
- **docker-compose.yml**: Development environment setup
- **.gitignore**: Git ignore patterns
- **.env.example**: Environment variable template

## Deployment Structure

### Docker (`docker/`)
```
docker/
├── config/               # Configuration files
├── grafana/              # Grafana dashboards
├── nginx/                # Nginx configuration
│   ├── conf.d/
│   └── nginx.conf
├── postgres/             # Database initialization
│   └── init.sql
├── prometheus/           # Monitoring configuration
│   └── prometheus.yml
├── redis/                # Redis configuration
├── entrypoint.sh         # Container entry point
└── healthcheck.sh        # Health check script
```

### Kubernetes (`k8s/`)
```
k8s/
├── base/                 # Base Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml          # Horizontal Pod Autoscaler
│   └── namespace.yaml
└── overlays/             # Environment-specific overlays
    ├── development/
    ├── staging/
    └── production/
```

## Embedded Components

### Unmute Voice System (`unmute/`)
```
unmute/                   # Git submodule for voice AI
├── frontend/             # Voice UI components
├── services/             # Voice processing services
├── unmute/               # Core voice processing
│   ├── llm/              # Language model integration
│   ├── stt/              # Speech-to-text
│   ├── tts/              # Text-to-speech
│   └── scripts/          # Voice processing scripts
└── pyproject.toml        # Unmute configuration
```

## File Naming Conventions

### Python Files
- **Modules**: `snake_case.py`
- **Classes**: `PascalCase` in files
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

### Test Files
- **Unit tests**: `test_unit_*.py`
- **Integration tests**: `test_integration_*.py`
- **Performance tests**: `test_performance_*.py`
- **End-to-end tests**: `test_e2e_*.py`

### Configuration Files
- **Environment**: `.env`, `.env.example`
- **Docker**: `Dockerfile`, `docker-compose.yml`
- **Kubernetes**: `*.yaml` in `k8s/`
- **CI/CD**: `.github/workflows/*.yml`

## Import Patterns

### Internal Imports
```python
# Absolute imports from src/tektra/
from tektra.ai.simple_llm import SimpleLLM
from tektra.agents import AgentBuilder, AgentRegistry
from tektra.gui.chat_panel import ChatPanel
from tektra.utils.config import AppConfig
```

### Conditional Imports
```python
# Handle optional dependencies gracefully
try:
    from tektra.voice.unmute_embedded import EmbeddedUnmute
    UNMUTE_AVAILABLE = True
except ImportError:
    UNMUTE_AVAILABLE = False
    EmbeddedUnmute = None
```

### Lazy Imports
```python
# Import heavy dependencies only when needed
def get_qwen_backend():
    from tektra.ai.qwen_backend import QwenBackend
    return QwenBackend()
```

## Data Directory Structure

### User Data (`~/.tektra/`)
```
~/.tektra/
├── models/               # Downloaded AI models
│   ├── transformers/     # Hugging Face models
│   ├── unmute/           # Voice models
│   └── cache/            # Model cache
├── memory/               # Conversation memory
│   ├── conversations/    # Chat history
│   ├── memos/            # MemOS data
│   └── vectors/          # Vector embeddings
├── config/               # User configuration
│   ├── settings.json     # App settings
│   └── agents.json       # Agent configurations
└── logs/                 # Application logs
    ├── app.log
    ├── agents.log
    └── security.log
```

## Development Workflow

### Adding New Components
1. **Create module** in appropriate `src/tektra/` subdirectory
2. **Add tests** in corresponding `tests/` location
3. **Update `__init__.py`** to export public APIs
4. **Add documentation** in docstrings and `docs/`
5. **Update configuration** if needed

### Component Dependencies
- **Core components** (`core/`, `utils/`) have minimal dependencies
- **AI components** (`ai/`, `models/`) depend on ML libraries
- **GUI components** (`gui/`) depend on Toga framework
- **Voice components** (`voice/`) depend on audio libraries
- **Security components** (`security/`) are self-contained

### Testing Organization
- **Unit tests** test individual components in isolation
- **Integration tests** test component interactions
- **Performance tests** measure resource usage and speed
- **End-to-end tests** test complete user workflows

This structure supports the modular, scalable architecture needed for a production-ready AI assistant while maintaining clear separation of concerns and testability.