"""
Tektra Core System Module

Contains the unified system architecture that integrates all Tektra components
into a cohesive, production-ready AI assistant.
"""

from .tektra_system import (
    TektraSystem,
    TektraSystemConfig,
    SystemState,
    ComponentStatus,
    SystemHealth,
    create_tektra_system
)

from .error_handling import (
    ErrorHandler,
    TektraError,
    SecurityError,
    PerformanceError,
    AgentError,
    ConfigurationError,
    ResourceError,
    ErrorCategory,
    ErrorSeverity,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    error_boundary,
    retry_with_backoff,
    get_global_error_handler
)

__all__ = [
    # System components
    "TektraSystem",
    "TektraSystemConfig",
    "SystemState", 
    "ComponentStatus",
    "SystemHealth",
    "create_tektra_system",
    
    # Error handling
    "ErrorHandler",
    "TektraError",
    "SecurityError",
    "PerformanceError",
    "AgentError",
    "ConfigurationError",
    "ResourceError",
    "ErrorCategory",
    "ErrorSeverity",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "RetryConfig",
    "error_boundary",
    "retry_with_backoff",
    "get_global_error_handler"
]