"""
Data Management Components

This package contains data storage and management functionality including:
- Chat history and conversation persistence
- Application settings and configuration
- Document processing and vector database integration
"""

from .storage import DataStorage
from .vector_db import VectorDatabase

__all__ = ["DataStorage", "VectorDatabase"]
