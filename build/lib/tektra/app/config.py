"""Configuration management for Tektra backend."""

from typing import Any, Dict, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_title: str = "Tektra AI Assistant API"
    api_version: str = "0.1.0"
    api_description: str = "Advanced AI assistant with voice, vision, and robotics"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./tektra.db",
        description="Database connection URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # Security Configuration
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT token generation"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # AI Model Configuration
    model_cache_dir: str = Field(
        default="./models",
        description="Directory to cache AI models"
    )
    max_model_memory_gb: int = Field(
        default=8,
        description="Maximum memory for model loading in GB"
    )
    
    # Data Storage Configuration
    data_dir: str = Field(
        default="./data",
        description="Directory for application data storage"
    )
    
    # Audio Configuration
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 1024
    max_audio_duration: int = 30  # seconds
    
    # Vision Configuration
    camera_fps: int = 30
    max_image_size: int = 1920  # pixels
    
    # WebSocket Configuration
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10
    
    # CORS Configuration
    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    allowed_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: list[str] = ["*"]
    
    @validator("database_url", pre=True)
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "postgresql+asyncpg://", "sqlite:///", "sqlite+aiosqlite:///")):
            raise ValueError("Database URL must be PostgreSQL or SQLite")
        return v
    
    @validator("redis_url", pre=True) 
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL must start with redis://")
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()