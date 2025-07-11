"""
AI Backend Components

This package contains the AI backend systems including:
- Qwen integration for complex reasoning and vision tasks
- Smart routing between different AI systems
- Multimodal processing capabilities
"""

from .multimodal import MultimodalProcessor
from .qwen_backend import QwenBackend
from .smart_router import QueryRoute, SmartRouter

__all__ = ["QwenBackend", "SmartRouter", "QueryRoute", "MultimodalProcessor"]
