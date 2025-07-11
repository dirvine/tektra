"""
Application Configuration

This module provides configuration management for Tektra AI Assistant,
handling settings, preferences, and runtime configuration.
"""

import json
import platform
from pathlib import Path
from typing import Any

from loguru import logger


class AppConfig:
    """
    Application configuration manager.

    Handles loading, saving, and managing configuration settings
    for the Tektra AI Assistant.
    """

    def __init__(self, config_file: str | None = None):
        """
        Initialize application configuration.

        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config_data = {}
        self.default_config = self._get_default_config()

        # Load configuration
        self.load_config()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Platform-specific config directory
        if platform.system() == "Windows":
            config_dir = Path.home() / "AppData" / "Local" / "Tektra"
        elif platform.system() == "Darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "Tektra"
        else:  # Linux and others
            config_dir = Path.home() / ".config" / "tektra"

        # Create config directory if it doesn't exist
        config_dir.mkdir(parents=True, exist_ok=True)

        return str(config_dir / "config.json")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration values."""
        return {
            # AI Model Settings
            "qwen_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen_quantization": 8,
            "max_memory_gb": 8.0,
            "use_flash_attention": True,
            # Voice Settings
            "voice_enabled": True,
            "auto_start_voice": False,
            "voice_activity_threshold": 0.5,
            "audio_sample_rate": 16000,
            "audio_chunk_size": 1024,
            # UI Settings
            "window_width": 1200,
            "window_height": 800,
            "theme": "default",
            "show_performance_stats": True,
            "auto_scroll_chat": True,
            # Smart Router Settings
            "confidence_threshold": 0.6,
            "mixed_query_threshold": 0.4,
            "voice_bias": 0.1,
            "enable_hybrid_routing": True,
            # File Processing
            "max_file_size_mb": 100,
            "supported_image_formats": [
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".webp",
            ],
            "supported_document_formats": [".txt", ".md", ".pdf", ".docx"],
            # Performance Settings
            "max_conversation_history": 50,
            "enable_model_caching": True,
            "cleanup_interval_minutes": 30,
            # Logging
            "log_level": "INFO",
            "log_to_file": True,
            "max_log_files": 10,
            # Service URLs
            "unmute_base_url": "http://localhost",
            "unmute_websocket_url": "ws://localhost:8000",
            "unmute_backend_port": 8000,
            "unmute_stt_port": 8001,
            "unmute_tts_port": 8002,
            "unmute_llm_port": 8000,
            # Development
            "debug_mode": False,
            "development_mode": False,
            "mock_unmute_services": False,
        }

    def load_config(self) -> bool:
        """
        Load configuration from file.

        Returns:
            bool: True if loaded successfully
        """
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, encoding="utf-8") as f:
                    loaded_config = json.load(f)

                # Merge with defaults (defaults take precedence for missing keys)
                self.config_data = {**self.default_config, **loaded_config}
                logger.info(f"Configuration loaded from {self.config_file}")
                return True
            else:
                # Use defaults if config file doesn't exist
                self.config_data = self.default_config.copy()
                logger.info("Using default configuration")
                # Save defaults to create the config file
                self.save_config()
                return True

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Fall back to defaults
            self.config_data = self.default_config.copy()
            return False

    def save_config(self) -> bool:
        """
        Save current configuration to file.

        Returns:
            bool: True if saved successfully
        """
        try:
            # Ensure config directory exists
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config_data[key] = value
        logger.debug(f"Configuration updated: {key} = {value}")

    def update(self, updates: dict[str, Any]) -> None:
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        self.config_data.update(updates)
        logger.debug(f"Configuration updated with {len(updates)} values")

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config_data = self.default_config.copy()
        logger.info("Configuration reset to defaults")

    def get_section(self, prefix: str) -> dict[str, Any]:
        """
        Get all configuration values with a given prefix.

        Args:
            prefix: Key prefix to filter by

        Returns:
            Dictionary of matching configuration values
        """
        return {
            key: value
            for key, value in self.config_data.items()
            if key.startswith(prefix)
        }

    def validate_config(self) -> dict[str, str]:
        """
        Validate current configuration.

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        # Validate memory settings
        if self.get("max_memory_gb", 0) <= 0:
            errors["max_memory_gb"] = "Must be greater than 0"

        # Validate quantization settings
        quantization = self.get("qwen_quantization", 8)
        if quantization not in [4, 8, 16]:
            errors["qwen_quantization"] = "Must be 4, 8, or 16"

        # Validate thresholds
        confidence_threshold = self.get("confidence_threshold", 0.6)
        if not 0 <= confidence_threshold <= 1:
            errors["confidence_threshold"] = "Must be between 0 and 1"

        # Validate file size
        max_file_size = self.get("max_file_size_mb", 100)
        if max_file_size <= 0:
            errors["max_file_size_mb"] = "Must be greater than 0"

        # Validate window dimensions
        window_width = self.get("window_width", 1200)
        window_height = self.get("window_height", 800)
        if window_width < 800 or window_height < 600:
            errors["window_size"] = "Minimum window size is 800x600"

        return errors

    def get_model_config(self) -> dict[str, Any]:
        """Get AI model configuration section."""
        return {
            "model_name": self.get("qwen_model_name"),
            "quantization_bits": self.get("qwen_quantization"),
            "max_memory_gb": self.get("max_memory_gb"),
            "use_flash_attention": self.get("use_flash_attention"),
        }

    def get_voice_config(self) -> dict[str, Any]:
        """Get voice processing configuration section."""
        return {
            "enabled": self.get("voice_enabled"),
            "auto_start": self.get("auto_start_voice"),
            "activity_threshold": self.get("voice_activity_threshold"),
            "sample_rate": self.get("audio_sample_rate"),
            "chunk_size": self.get("audio_chunk_size"),
        }

    def get_ui_config(self) -> dict[str, Any]:
        """Get UI configuration section."""
        return {
            "window_width": self.get("window_width"),
            "window_height": self.get("window_height"),
            "theme": self.get("theme"),
            "show_performance_stats": self.get("show_performance_stats"),
            "auto_scroll_chat": self.get("auto_scroll_chat"),
        }

    def get_router_config(self) -> dict[str, Any]:
        """Get smart router configuration section."""
        return {
            "confidence_threshold": self.get("confidence_threshold"),
            "mixed_query_threshold": self.get("mixed_query_threshold"),
            "voice_bias": self.get("voice_bias"),
            "enable_hybrid_routing": self.get("enable_hybrid_routing"),
        }

    def get_service_config(self) -> dict[str, Any]:
        """Get service URL configuration section."""
        return {
            "base_url": self.get("unmute_base_url"),
            "websocket_url": self.get("unmute_websocket_url"),
            "backend_port": self.get("unmute_backend_port"),
            "stt_port": self.get("unmute_stt_port"),
            "tts_port": self.get("unmute_tts_port"),
            "llm_port": self.get("unmute_llm_port"),
        }

    def export_config(self, export_path: str) -> bool:
        """
        Export configuration to a file.

        Args:
            export_path: Path to export configuration

        Returns:
            bool: True if exported successfully
        """
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False

    def import_config(self, import_path: str) -> bool:
        """
        Import configuration from a file.

        Args:
            import_path: Path to import configuration from

        Returns:
            bool: True if imported successfully
        """
        try:
            with open(import_path, encoding="utf-8") as f:
                imported_config = json.load(f)

            # Validate imported config
            temp_config = AppConfig()
            temp_config.config_data = {**self.default_config, **imported_config}
            errors = temp_config.validate_config()

            if errors:
                logger.error(f"Invalid configuration: {errors}")
                return False

            # Import if valid
            self.config_data = temp_config.config_data
            logger.info(f"Configuration imported from {import_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False

    def get_config_info(self) -> dict[str, Any]:
        """Get configuration metadata and statistics."""
        return {
            "config_file": self.config_file,
            "config_exists": Path(self.config_file).exists(),
            "total_settings": len(self.config_data),
            "validation_errors": len(self.validate_config()),
            "default_settings": len(self.default_config),
            "custom_settings": len(
                [
                    k
                    for k in self.config_data
                    if k not in self.default_config
                    or self.config_data[k] != self.default_config[k]
                ]
            ),
        }
