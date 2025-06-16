"""Conversation and message models."""

from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import (
    Column, DateTime, Enum as SQLEnum, ForeignKey, Integer, String, Text, Boolean, JSON, Float, Table
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


class ConversationCategory(str, Enum):
    """Conversation category enumeration."""
    GENERAL = "general"
    WORK = "work"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    PERSONAL = "personal"
    RESEARCH = "research"
    BRAINSTORM = "brainstorm"
    SUPPORT = "support"


# Association table for conversation tags (many-to-many)
conversation_tags = Table(
    'conversation_tags',
    Base.metadata,
    Column('conversation_id', Integer, ForeignKey('conversations.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)


class Conversation(Base):
    """Conversation model to group related messages."""
    
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)  # Optional description for better organization
    model_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_pinned = Column(Boolean, default=False)  # Pin important conversations
    is_archived = Column(Boolean, default=False)  # Archive old conversations
    
    # Organization fields
    category = Column(SQLEnum(ConversationCategory), default=ConversationCategory.GENERAL)
    priority = Column(Integer, default=0)  # Higher numbers = higher priority
    color = Column(String(7), nullable=True)  # Hex color code for visual organization
    
    # Search and analytics
    message_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)  # Average response time in seconds
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    tags = relationship("Tag", secondary=conversation_tags, back_populates="conversations")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class Tag(Base):
    """Tag model for organizing conversations."""
    
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(50), nullable=False, index=True)
    color = Column(String(7), nullable=True)  # Hex color code
    description = Column(String(255), nullable=True)
    usage_count = Column(Integer, default=0)  # Track how often this tag is used
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="tags")
    conversations = relationship("Conversation", secondary=conversation_tags, back_populates="tags")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Tag(id={self.id}, name='{self.name}')>"


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
    
    # Search and organization
    is_important = Column(Boolean, default=False)  # Flag important messages
    is_favorite = Column(Boolean, default=False)  # User can favorite messages
    search_vector = Column(Text, nullable=True)  # Full-text search index
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Message(id={self.id}, role='{self.role}', type='{self.message_type}')>"