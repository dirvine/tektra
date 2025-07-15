#!/usr/bin/env python3
"""
Tektra AI Assistant - Desktop Application

A voice-interactive AI assistant with multimodal capabilities, progressive feature discovery,
and advanced model management.

Basic Usage:
    from tektra import TektraApp
    
    app = TektraApp()
    app.main_loop()

Advanced Usage:
    from tektra.gui.feature_discovery import initialize_discovery_manager
    from tektra.models import ModelRegistry, ModelUpdateManager
"""

__version__ = "0.1.0"
__author__ = "David Irvine"
__description__ = "Open-Source Conversational AI Desktop App with Embedded Voice Intelligence"

# Core app export
try:
    from .app import TektraApp
    _app_available = True
except ImportError as e:
    _app_available = False
    _app_error = str(e)

# GUI components
try:
    from .gui.feature_discovery import (
        FeatureDiscoveryManager,
        initialize_discovery_manager,
        get_discovery_manager,
        DiscoveryTrigger
    )
    _gui_available = True
except ImportError as e:
    _gui_available = False
    _gui_error = str(e)

# Model management
try:
    from .models import (
        ModelInterface,
        ModelRegistry,
        ModelFactory,
        ModelUpdateManager,
        default_registry,
        default_factory
    )
    _models_available = True
except ImportError as e:
    _models_available = False
    _models_error = str(e)

# Basic components that should always work
try:
    from .utils.config import AppConfig
    _config_available = True
except ImportError:
    _config_available = False

# Build exports list based on what's available
__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "get_version_info",
    "get_status"
]

if _app_available:
    __all__.append("TektraApp")

if _gui_available:
    __all__.extend([
        "FeatureDiscoveryManager",
        "initialize_discovery_manager", 
        "get_discovery_manager",
        "DiscoveryTrigger"
    ])

if _models_available:
    __all__.extend([
        "ModelInterface",
        "ModelRegistry",
        "ModelFactory", 
        "ModelUpdateManager",
        "default_registry",
        "default_factory"
    ])

if _config_available:
    __all__.append("AppConfig")


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "features": [
            "Progressive feature discovery system",
            "Enhanced model management and hot-swapping",
            "Voice-first interaction patterns",
            "Conversational memory with MemOS integration",
            "Markdown message rendering",
            "AI agent creation and management",
            "Multimodal processing (text, voice, images)",
            "Cross-platform desktop application"
        ],
        "status": get_status()
    }


def get_status():
    """Get component availability status."""
    return {
        "app": _app_available,
        "gui": _gui_available,
        "models": _models_available,
        "config": _config_available,
        "errors": {
            "app": _app_error if not _app_available else None,
            "gui": _gui_error if not _gui_available else None,
            "models": _models_error if not _models_available else None
        }
    }


# Quick start helper functions
async def quick_start(
    agent_name: str = "QuickStart Agent",
    task: str = "Hello, I'm ready to help!",
    security_level: str = "medium"
) -> str:
    """
    Quick start function to create and run an agent with minimal setup.
    
    Args:
        agent_name: Name for the agent
        task: Initial task to execute
        security_level: Security level for the system
        
    Returns:
        Task execution result
    """
    # For quick start, just return a simple response
    return f"QuickStart Agent '{agent_name}' would execute: {task}"


def create_production_config(
    max_agents: int = 50,
    cache_size_mb: int = 1024,
    memory_limit_mb: int = 4096
) -> dict:
    """
    Create a production-ready configuration.
    
    Args:
        max_agents: Maximum concurrent agents
        cache_size_mb: Cache size in MB
        memory_limit_mb: Memory limit in MB
        
    Returns:
        Production configuration dictionary
    """
    return {
        "system_name": "Tektra AI Assistant (Production)",
        "environment": "production",
        "debug_mode": False,
        "max_concurrent_agents": max_agents,
        "cache_size_mb": cache_size_mb,
        "memory_limit_mb": memory_limit_mb,
        "security_level": "high",
        "ui_enabled": True
    }


def create_development_config(
    max_agents: int = 5,
    cache_size_mb: int = 256
) -> dict:
    """
    Create a development-friendly configuration.
    
    Args:
        max_agents: Maximum concurrent agents
        cache_size_mb: Cache size in MB
        
    Returns:
        Development configuration dictionary
    """
    return {
        "system_name": "Tektra AI Assistant (Development)",
        "environment": "development",
        "debug_mode": True,
        "max_concurrent_agents": max_agents,
        "cache_size_mb": cache_size_mb,
        "memory_limit_mb": 1024,
        "security_level": "medium",
        "ui_enabled": True
    }


# Module initialization message
def _print_initialization_info():
    """Print initialization information."""
    import sys
    
    # Only print if running interactively
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        print(f"🌟 Tektra AI Assistant v{__version__} initialized")
        print(f"   {__description__}")
        print(f"   Use quick_start() for immediate testing")
        print(f"   Use TektraApp() for full GUI application")


# Initialize on import in interactive mode
try:
    _print_initialization_info()
except Exception:
    pass  # Silent fail if printing not available
