#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Tektra tests.

This module provides session-scoped fixtures for expensive resources
like AI models that should be loaded once and shared across tests.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
import tempfile
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
        mock_backend = AsyncMock()
        mock_backend.initialize = AsyncMock(return_value=True)
        mock_backend.cleanup = AsyncMock()
        mock_backend.process_text_query = AsyncMock(return_value="Mock response")
        mock_backend.process_vision_query = AsyncMock(return_value="Mock vision response")
        mock_backend.is_initialized = True
        _qwen_backend = mock_backend
        return _qwen_backend
    
    # Load the actual model
    try:
        from tektra.ai.qwen_backend import QwenBackend, QwenModelConfig
        
        print("\n🔄 Loading Qwen model (this will only happen once per test session)...")
        
        # Configure with reduced memory for testing
        config = QwenModelConfig(
            model_name='Qwen/Qwen2.5-VL-7B-Instruct',
            quantization_bits=None,
            max_memory_gb=4.0,  # Reduced for testing
            device_map="auto",
            torch_dtype="float16"  # Use float16 to reduce memory
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
    from tektra.memory import TektraMemoryManager, MemoryConfig
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MemoryConfig(
            storage_path=temp_dir,
            database_name="test.db",
            use_memos=False,  # Disable MemOS for tests
            enable_semantic_search=False  # Disable for speed
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
        await registry.initialize()
        yield registry
        await registry.cleanup()
    except ImportError:
        pytest.skip("Agent registry not available")
        yield None


@pytest.fixture(scope="function")
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "heavy: marks tests that use heavy AI models"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks slow-running tests"
    )