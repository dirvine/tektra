"""
Tektra AI Assistant

An advanced AI assistant with voice, vision, and robotics capabilities.
Features persistent conversation management, real-time chat, and modular architecture.
"""

__version__ = "0.9.15"
__author__ = "Saorsa Labs"
__email__ = "saorsalabs@gmail.com"

from .app.config import settings
from .app.main import app

__all__ = ["app", "settings", "__version__"]
