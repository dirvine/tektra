"""
Memory configuration for Tektra

This module handles configuration for the memory system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryConfig:
    """Configuration for Tektra memory system."""

    # Storage configuration
    storage_type: str = "sqlite"  # sqlite, json, memos
    storage_path: str = field(
        default_factory=lambda: os.path.expanduser("~/.tektra/memory")
    )
    database_name: str = "tektra_memory.db"

    # Memory behavior
    max_memories_per_user: int = 10000
    max_memories_per_agent: int = 5000
    default_importance_threshold: float = 0.3
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24

    # Search configuration
    default_search_limit: int = 10
    max_search_results: int = 100
    relevance_threshold: float = 0.3
    enable_semantic_search: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"

    # MemOS integration
    use_memos: bool = True
    memos_config: dict[str, Any] | None = None

    # Performance settings
    batch_size: int = 100
    cache_size: int = 1000
    enable_compression: bool = True

    # Privacy and security
    encrypt_storage: bool = False
    anonymize_content: bool = False
    retention_days: int = 365

    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure storage directory exists
        storage_dir = Path(self.storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Set up default MemOS configuration if not provided
        if self.use_memos and self.memos_config is None:
            self.memos_config = {
                "enable_textual_memory": True,
                "enable_activation_memory": False,
                "enable_parametric_memory": False,
                "PRO_MODE": False,
                "max_turns_window": 15,
                "top_k": 5,
            }

    @property
    def database_path(self) -> str:
        """Get full path to database file."""
        return str(Path(self.storage_path) / self.database_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "storage_type": self.storage_type,
            "storage_path": self.storage_path,
            "database_name": self.database_name,
            "max_memories_per_user": self.max_memories_per_user,
            "max_memories_per_agent": self.max_memories_per_agent,
            "default_importance_threshold": self.default_importance_threshold,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "cleanup_interval_hours": self.cleanup_interval_hours,
            "default_search_limit": self.default_search_limit,
            "max_search_results": self.max_search_results,
            "relevance_threshold": self.relevance_threshold,
            "enable_semantic_search": self.enable_semantic_search,
            "embedding_model": self.embedding_model,
            "use_memos": self.use_memos,
            "memos_config": self.memos_config,
            "batch_size": self.batch_size,
            "cache_size": self.cache_size,
            "enable_compression": self.enable_compression,
            "encrypt_storage": self.encrypt_storage,
            "anonymize_content": self.anonymize_content,
            "retention_days": self.retention_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def load_from_file(cls, config_path: str) -> "MemoryConfig":
        """Load configuration from file."""
        import json

        config_file = Path(config_path)
        if not config_file.exists():
            return cls()  # Return default config

        with open(config_file) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_to_file(self, config_path: str):
        """Save configuration to file."""
        import json

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> list[str]:
        """Validate configuration and return any errors."""
        errors = []

        # Validate storage path
        try:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Invalid storage path: {e}")

        # Validate limits
        if self.max_memories_per_user <= 0:
            errors.append("max_memories_per_user must be positive")

        if self.max_memories_per_agent <= 0:
            errors.append("max_memories_per_agent must be positive")

        if not 0 <= self.default_importance_threshold <= 1:
            errors.append("default_importance_threshold must be between 0 and 1")

        if not 0 <= self.relevance_threshold <= 1:
            errors.append("relevance_threshold must be between 0 and 1")

        if self.batch_size <= 0:
            errors.append("batch_size must be positive")

        if self.cache_size <= 0:
            errors.append("cache_size must be positive")

        if self.retention_days <= 0:
            errors.append("retention_days must be positive")

        return errors


# Default configurations for different use cases
DEFAULT_MEMORY_CONFIG = MemoryConfig()

LIGHTWEIGHT_MEMORY_CONFIG = MemoryConfig(
    max_memories_per_user=1000,
    max_memories_per_agent=500,
    enable_semantic_search=False,
    use_memos=False,
    cache_size=100,
)

PERFORMANCE_MEMORY_CONFIG = MemoryConfig(
    max_memories_per_user=50000,
    max_memories_per_agent=25000,
    batch_size=500,
    cache_size=5000,
    enable_compression=True,
    use_memos=True,
)

PRIVACY_MEMORY_CONFIG = MemoryConfig(
    encrypt_storage=True,
    anonymize_content=True,
    retention_days=30,
    auto_cleanup_enabled=True,
    cleanup_interval_hours=6,
)
