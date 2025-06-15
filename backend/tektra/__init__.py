"""
Tektra AI Assistant

An advanced AI assistant with voice, vision, and robotics capabilities.
Features persistent conversation management, real-time chat, and modular architecture.
"""

__version__ = "0.1.8"
__author__ = "Tektra Team"
__email__ = "contact@tektra.ai"

from .app.main import app
from .app.config import settings

__all__ = ["app", "settings", "__version__"]