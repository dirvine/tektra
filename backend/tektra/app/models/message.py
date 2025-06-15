"""Message model - compatibility import."""

# For backward compatibility, import Message from conversation module
from .conversation import Message, MessageRole, MessageType

__all__ = ["Message", "MessageRole", "MessageType"]