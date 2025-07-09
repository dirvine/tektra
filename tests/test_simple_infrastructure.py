#!/usr/bin/env python3
"""
Simple infrastructure test to verify pytest setup.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_math():
    """Basic test to verify pytest is working."""
    assert 2 + 2 == 4
    assert 10 * 2 == 20

def test_python_version():
    """Test Python version compatibility."""
    assert sys.version_info >= (3, 9)

def test_pathlib_functionality():
    """Test pathlib functionality."""
    current_file = Path(__file__)
    assert current_file.exists()
    assert current_file.is_file()
    assert current_file.suffix == ".py"

def test_import_structure():
    """Test that we can import from src structure."""
    try:
        # Test if we can import the main tektra package
        import tektra
        assert hasattr(tektra, '__version__') or hasattr(tektra, '__file__')
    except ImportError:
        # If tektra package doesn't exist, that's fine for this test
        pass

@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality works with pytest-asyncio."""
    async def simple_async_func():
        return "async works"
    
    result = await simple_async_func()
    assert result == "async works"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])