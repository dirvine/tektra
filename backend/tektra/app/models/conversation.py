"""Conversation and message models."""

from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import (
    Column, DateTime, Enum as SQLEnum, ForeignKey, Integer, String, Text, Boolean, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Message type enumeration."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    ACTION = "action"


class Conversation(Base):
    """Conversation model to group related messages."""
    
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=True)
    model_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class Message(Base):
    """Message model for conversation history."""
    
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    
    # Message content
    role = Column(SQLEnum(MessageRole), nullable=False)
    message_type = Column(SQLEnum(MessageType), default=MessageType.TEXT)
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON, nullable=True, default=dict)  # JSON for additional data
    
    # Audio/Video specific
    audio_url = Column(String(500), nullable=True)
    duration = Column(Integer, nullable=True)  # in seconds
    
    # Processing info
    model_name = Column(String(100), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    processing_time = Column(Integer, nullable=True)  # in milliseconds
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Message(id={self.id}, role='{self.role}', type='{self.message_type}')>"