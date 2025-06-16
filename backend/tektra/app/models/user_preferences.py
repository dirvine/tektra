"""
User Preferences and Settings Models.

Models for storing user preferences, AI model settings, and personalization options.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class ThemeMode(str, Enum):
    """UI theme mode enumeration."""

    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class VoiceProvider(str, Enum):
    """Voice synthesis provider enumeration."""

    EDGE_TTS = "edge_tts"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    LOCAL = "local"


class NotificationLevel(str, Enum):
    """Notification level enumeration."""

    ALL = "all"
    IMPORTANT = "important"
    MENTIONS = "mentions"
    NONE = "none"


class UserPreferences(Base):
    """User preferences and settings model."""

    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    # UI/UX Preferences
    theme_mode = Column(SQLEnum(ThemeMode), default=ThemeMode.AUTO)
    language = Column(String(10), default="en")
    timezone = Column(String(50), default="UTC")
    date_format = Column(String(20), default="YYYY-MM-DD")
    time_format = Column(String(10), default="24h")  # 12h or 24h

    # Chat Interface Preferences
    chat_bubble_style = Column(String(20), default="modern")  # modern, classic, minimal
    message_grouping = Column(Boolean, default=True)
    show_timestamps = Column(Boolean, default=True)
    show_token_count = Column(Boolean, default=False)
    auto_scroll = Column(Boolean, default=True)
    typing_indicators = Column(Boolean, default=True)

    # AI Model Preferences
    default_model = Column(String(100), default="phi-3-mini")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2048)
    top_p = Column(Float, default=0.9)
    frequency_penalty = Column(Float, default=0.0)
    presence_penalty = Column(Float, default=0.0)

    # Voice and Audio Preferences
    voice_provider = Column(SQLEnum(VoiceProvider), default=VoiceProvider.EDGE_TTS)
    voice_name = Column(String(100), default="default")
    voice_speed = Column(Float, default=1.0)
    voice_pitch = Column(Float, default=0.0)
    auto_play_responses = Column(Boolean, default=False)
    voice_input_enabled = Column(Boolean, default=True)
    noise_suppression = Column(Boolean, default=True)

    # Avatar Preferences
    avatar_enabled = Column(Boolean, default=True)
    avatar_style = Column(String(50), default="realistic")
    avatar_gender = Column(String(20), default="neutral")
    avatar_expressions = Column(Boolean, default=True)
    avatar_lip_sync = Column(Boolean, default=True)

    # Privacy and Security
    data_retention_days = Column(Integer, default=90)  # 0 = forever
    analytics_enabled = Column(Boolean, default=True)
    crash_reporting = Column(Boolean, default=True)
    share_usage_data = Column(Boolean, default=False)

    # Notifications
    notification_level = Column(
        SQLEnum(NotificationLevel), default=NotificationLevel.IMPORTANT
    )
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=True)
    sound_notifications = Column(Boolean, default=True)

    # Advanced Settings
    developer_mode = Column(Boolean, default=False)
    debug_logging = Column(Boolean, default=False)
    experimental_features = Column(Boolean, default=False)
    beta_updates = Column(Boolean, default=False)

    # Custom settings (JSON for extensibility)
    custom_settings = Column(JSON, nullable=True, default=dict)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="preferences")
    model_settings = relationship(
        "ModelSettings", back_populates="user_preferences", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<UserPreferences(user_id={self.user_id}, theme={self.theme_mode})>"


class ModelSettings(Base):
    """Model-specific settings for individual AI models."""

    __tablename__ = "model_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_preferences_id = Column(
        Integer, ForeignKey("user_preferences.id"), nullable=False
    )
    model_name = Column(String(100), nullable=False, index=True)

    # Model Parameters
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    top_p = Column(Float, nullable=True)
    top_k = Column(Integer, nullable=True)
    frequency_penalty = Column(Float, nullable=True)
    presence_penalty = Column(Float, nullable=True)

    # Model-specific settings
    system_prompt = Column(Text, nullable=True)
    stop_sequences = Column(JSON, nullable=True)  # List of stop sequences

    # Performance settings
    streaming_enabled = Column(Boolean, default=True)
    cache_enabled = Column(Boolean, default=True)
    batch_size = Column(Integer, nullable=True)

    # Usage tracking
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)
    last_used = Column(DateTime(timezone=True), nullable=True)

    # Custom model parameters (JSON for extensibility)
    custom_parameters = Column(JSON, nullable=True, default=dict)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user_preferences = relationship("UserPreferences", back_populates="model_settings")

    def __repr__(self) -> str:
        """String representation."""
        return f"<ModelSettings(model={self.model_name}, temp={self.temperature})>"


class ConversationTemplate(Base):
    """Pre-defined conversation templates and prompts."""

    __tablename__ = "conversation_templates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), default="general")

    # Template content
    system_prompt = Column(Text, nullable=False)
    initial_message = Column(Text, nullable=True)

    # Template settings
    recommended_model = Column(String(100), nullable=True)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)

    # Organization
    is_public = Column(Boolean, default=False)  # Share with other users
    is_favorite = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)

    # Template metadata
    tags = Column(JSON, nullable=True)  # List of tags
    icon = Column(String(50), nullable=True)  # Icon name or emoji
    color = Column(String(7), nullable=True)  # Hex color

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="conversation_templates")

    def __repr__(self) -> str:
        """String representation."""
        return f"<ConversationTemplate(name='{self.name}', category='{self.category}')>"


class APIKey(Base):
    """Encrypted storage for user API keys."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    provider = Column(String(50), nullable=False)  # openai, anthropic, etc.
    key_name = Column(String(100), nullable=False)  # User-defined name
    encrypted_key = Column(Text, nullable=False)  # Encrypted API key

    # Key metadata
    is_active = Column(Boolean, default=True)
    usage_limit = Column(Integer, nullable=True)  # Monthly usage limit
    current_usage = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        """String representation."""
        return f"<APIKey(provider='{self.provider}', name='{self.key_name}')>"
