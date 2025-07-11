"""
Tektra AI Assistant

A voice-interactive AI assistant with multimodal capabilities using:
- Kyutai Unmute for ultra-low latency voice conversations
- Qwen 2.5-VL for complex reasoning and vision analysis
- Smart routing between conversational and analytical AI systems

Built with Python and Briefcase for native desktop experience.
"""

__version__ = "0.1.0"
__author__ = "David Irvine"
__email__ = "david.irvine@maidsafe.net"

from .app import TektraApp

__all__ = ["TektraApp"]
