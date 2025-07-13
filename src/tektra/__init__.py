#!/usr/bin/env python3
"""
Tektra AI Assistant - Complete System Package

Unified interface to the Tektra AI Assistant system integrating SmolAgents,
performance optimization, security framework, and production-ready architecture.

Usage:
    from tektra import TektraSystem, TektraSystemConfig
    
    # Simple usage
    async with TektraSystem() as system:
        agent_id = await system.create_agent("MyAgent")
        result = await system.execute_agent_task(agent_id, "Hello world")
    
    # Advanced configuration
    config = TektraSystemConfig(
        environment="production",
        security_level=SecurityLevel.HIGH,
        max_concurrent_agents=20
    )
    system = await create_tektra_system(config)
"""

__version__ = "1.0.0"
__author__ = "Tektra AI Team"
__description__ = "Production-ready AI Assistant with SmolAgents, Security, and Performance"

# Legacy app export (for backward compatibility)
try:
    from .app import TektraApp
    _legacy_available = True
except ImportError:
    _legacy_available = False

# Core system exports
from .core.tektra_system import (
    TektraSystem,
    TektraSystemConfig,
    SystemState,
    ComponentStatus,
    SystemHealth,
    create_tektra_system
)

# Security framework exports
from .security import (
    SecurityContext,
    SecurityLevel,
    PermissionManager,
    AdvancedSandbox,
    SandboxConfig,
    IsolationType,
    ToolValidator,
    ConsentFramework,
    ConsentMode,
    SecurityMonitor,
    EventType,
    EventSeverity,
    ThreatLevel,
    ValidationResult
)

# Performance framework exports
from .performance import (
    create_cache_manager,
    create_task_scheduler,
    create_memory_manager,
    create_performance_monitor,
    create_performance_optimizer,
    CacheLevel,
    Priority,
    OptimizationStrategy
)

# Agent system exports
from .agents import (
    SmolAgentsManager,
    AgentRole,
    AgentConfig
)

# Utility exports
from .utils.config import TektraConfig

# Configuration helpers
from .security.context import SecurityLevel
from .performance.cache_manager import CacheLevel
from .performance.task_scheduler import Priority

# Build exports list
__all__ = [
    # Core system
    "TektraSystem",
    "TektraSystemConfig", 
    "SystemState",
    "ComponentStatus",
    "SystemHealth",
    "create_tektra_system",
    
    # Security
    "SecurityContext",
    "SecurityLevel",
    "PermissionManager",
    "AdvancedSandbox",
    "SandboxConfig",
    "IsolationType",
    "ToolValidator",
    "ConsentFramework",
    "ConsentMode",
    "SecurityMonitor",
    "EventType",
    "EventSeverity",
    "ThreatLevel",
    "ValidationResult",
    
    # Performance
    "create_cache_manager",
    "create_task_scheduler", 
    "create_memory_manager",
    "create_performance_monitor",
    "create_performance_optimizer",
    "CacheLevel",
    "Priority",
    "OptimizationStrategy",
    
    # Agents
    "SmolAgentsManager",
    "AgentRole",
    "AgentConfig",
    
    # Utilities
    "TektraConfig",
    
    # Helper functions
    "quick_start",
    "create_production_config",
    "create_development_config",
    "get_version_info",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
]

# Add legacy app if available
if _legacy_available:
    __all__.append("TektraApp")


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "agents": "SmolAgents integration with tool ecosystem",
            "security": "Comprehensive security framework with sandboxing",
            "performance": "Resource management and optimization",
            "core": "Unified system architecture"
        },
        "features": [
            "Production-ready AI agent execution",
            "Advanced security sandboxing and validation", 
            "High-performance resource management",
            "Real-time monitoring and alerting",
            "Comprehensive permission and consent system",
            "Tool validation and malware detection",
            "Scalable task scheduling",
            "Multi-level caching",
            "Memory optimization",
            "Performance monitoring"
        ]
    }


# Quick start helper functions
async def quick_start(
    agent_name: str = "QuickStart Agent",
    task: str = "Hello, I'm ready to help!",
    security_level: SecurityLevel = SecurityLevel.MEDIUM
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
    config = TektraSystemConfig(
        environment="development",
        security_level=security_level,
        consent_mode="automatic",
        debug_mode=True
    )
    
    async with TektraSystem(config) as system:
        # Create security context
        security_context = SecurityContext(
            agent_id="quickstart_agent",
            security_level=security_level,
            session_id="quickstart_session"
        )
        
        # Create and run agent
        agent_id = await system.create_agent(
            agent_name=agent_name,
            security_context=security_context
        )
        
        result = await system.execute_agent_task(
            agent_id=agent_id,
            task_description=task,
            security_context=security_context
        )
        
        return result.get("response", "Task completed successfully")


def create_production_config(
    max_agents: int = 50,
    cache_size_mb: int = 1024,
    memory_limit_mb: int = 4096
) -> TektraSystemConfig:
    """
    Create a production-ready configuration.
    
    Args:
        max_agents: Maximum concurrent agents
        cache_size_mb: Cache size in MB
        memory_limit_mb: Memory limit in MB
        
    Returns:
        Production configuration
    """
    return TektraSystemConfig(
        system_name="Tektra AI Assistant (Production)",
        environment="production",
        debug_mode=False,
        
        # Agent configuration
        max_concurrent_agents=max_agents,
        agent_timeout_seconds=600.0,
        
        # Performance configuration
        cache_size_mb=cache_size_mb,
        memory_limit_mb=memory_limit_mb,
        task_queue_size=5000,
        
        # Security configuration
        security_level=SecurityLevel.HIGH,
        sandbox_enabled=True,
        tool_validation_enabled=True,
        consent_mode="interactive",
        
        # Monitoring configuration
        metrics_enabled=True,
        prometheus_port=8090,
        health_check_interval=15.0,
        
        # Integration configuration
        ui_enabled=True,
        api_enabled=True,
        websocket_enabled=True
    )


def create_development_config(
    max_agents: int = 5,
    cache_size_mb: int = 256
) -> TektraSystemConfig:
    """
    Create a development-friendly configuration.
    
    Args:
        max_agents: Maximum concurrent agents
        cache_size_mb: Cache size in MB
        
    Returns:
        Development configuration
    """
    return TektraSystemConfig(
        system_name="Tektra AI Assistant (Development)",
        environment="development", 
        debug_mode=True,
        
        # Agent configuration
        max_concurrent_agents=max_agents,
        agent_timeout_seconds=120.0,
        
        # Performance configuration
        cache_size_mb=cache_size_mb,
        memory_limit_mb=1024,
        task_queue_size=100,
        
        # Security configuration
        security_level=SecurityLevel.MEDIUM,
        sandbox_enabled=True,
        tool_validation_enabled=True,
        consent_mode="automatic",
        
        # Monitoring configuration
        metrics_enabled=True,
        prometheus_port=8091,
        health_check_interval=30.0,
        
        # Integration configuration
        ui_enabled=True,
        api_enabled=True,
        websocket_enabled=True
    )


# Module initialization message
def _print_initialization_info():
    """Print initialization information."""
    import sys
    
    # Only print if running interactively
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        print(f"🌟 Tektra AI Assistant v{__version__} initialized")
        print(f"   {__description__}")
        print(f"   Use quick_start() for immediate testing")
        print(f"   Use create_tektra_system() for full control")


# Initialize on import in interactive mode
try:
    _print_initialization_info()
except Exception:
    pass  # Silent fail if printing not available
