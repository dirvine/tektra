"""
Voice Processing Components

This package contains voice conversation capabilities using Kyutai Unmute:
- Service management for Docker-based Unmute services
- WebSocket client for STT/TTS/LLM communication
- Voice conversation pipeline orchestration
"""

from .services import UnmuteServiceManager
from .unmute_client import UnmuteWebSocketClient

try:
    from .pipeline import VoiceConversationPipeline
except ImportError:
    # Fall back to mock implementation for clean builds
    from .pipeline_mock import VoiceConversationPipeline

__all__ = ["UnmuteServiceManager", "UnmuteWebSocketClient", "VoiceConversationPipeline"]