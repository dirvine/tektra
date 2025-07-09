"""
Vision Processing Components

This package contains computer vision capabilities including:
- Camera integration and management
- Image processing and analysis with Qwen-VL
- Real-time video feed handling
"""

from .camera import CameraManager
from .processor import VisionProcessor

__all__ = ["CameraManager", "VisionProcessor"]