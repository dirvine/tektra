#!/usr/bin/env python3
"""
Comprehensive Error Handling and Resilience Framework

Advanced error handling, circuit breakers, retry mechanisms, graceful degradation,
and system resilience features for production-ready AI assistant deployment.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru,asyncio,typing-extensions,dataclasses python error_handling.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
#     "asyncio-compat>=0.1.0",
#     "typing-extensions>=4.7.0",
# ]
# ///

import asyncio
import time
import traceback
import functools
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
import json
import uuid

from loguru import logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AGENT = "agent"
    NETWORK = "network"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    INTEGRATION = "integration"


class RecoveryAction(Enum):
    """Recovery actions that can be taken."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    RESTART_COMPONENT = "restart_component"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    ABORT = "abort"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


T = TypeVar('T')


@dataclass
class ErrorContext:
    """Context information about an error."""
    
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Error details
    error_type: str = ""
    error_message: str = ""
    error_category: ErrorCategory = ErrorCategory.SYSTEM
    severity: ErrorSeverity = ErrorSeverity.ERROR
    
    # Context information
    component: str = ""
    operation: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # System state
    system_load: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    
    # Error chain
    root_cause: Optional[str] = None
    error_chain: List[str] = field(default_factory=list)
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_action: Optional[RecoveryAction] = None
    recovery_successful: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_category": self.error_category.value,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "system_load": self.system_load,
            "memory_usage": self.memory_usage,
            "active_connections": self.active_connections,
            "root_cause": self.root_cause,
            "error_chain": self.error_chain,
            "recovery_attempted": self.recovery_attempted,
            "recovery_action": self.recovery_action.value if self.recovery_action else None,
            "recovery_successful": self.recovery_successful,
            "metadata": self.metadata
        }


class TektraError(Exception):
    """Base exception class for Tektra system errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        component: str = "",
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.component = component
        self.operation = operation
        self.metadata = metadata or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.error_id = str(uuid.uuid4())
    
    def to_context(self) -> ErrorContext:
        """Convert to ErrorContext."""
        return ErrorContext(
            error_id=self.error_id,
            timestamp=self.timestamp,
            error_type=type(self).__name__,
            error_message=self.message,
            error_category=self.category,
            severity=self.severity,
            component=self.component,
            operation=self.operation,
            metadata=self.metadata
        )


class SecurityError(TektraError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class PerformanceError(TektraError):
    """Performance-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )


class AgentError(TektraError):
    """Agent-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AGENT,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ConfigurationError(TektraError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class ResourceError(TektraError):
    """Resource-related errors (memory, CPU, etc.)."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_exceptions: Tuple[type, ...] = (Exception,)
    stop_on_exceptions: Tuple[type, ...] = (SecurityError, ConfigurationError)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5
    reset_timeout: float = 60.0
    test_request_timeout: float = 10.0
    minimum_requests: int = 10
    success_threshold: int = 3  # Successes needed to close from half-open


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply circuit breaker to function."""
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.call(func, *args, **kwargs)
        
        return wrapper
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering half-open state")
                else:
                    raise TektraError(
                        f"Circuit breaker {self.name} is OPEN",
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.WARNING,
                        component="circuit_breaker",
                        operation="call_blocked"
                    )
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.reset_timeout
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.success_count += 1
            self.last_success_time = datetime.now()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after successful recovery")
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success in closed state
                self.failure_count = 0
    
    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Failed in half-open state, go back to open
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker {self.name} opened after half-open failure")
            elif self.state == CircuitBreakerState.CLOSED:
                # Check if we should open
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.error(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None
            }


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.error_patterns: Dict[str, List[datetime]] = {}
        
        # System state
        self.degraded_components: Set[str] = set()
        self.maintenance_mode = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self) -> None:
        """Setup default recovery strategies."""
        self.recovery_strategies[ErrorCategory.PERFORMANCE] = self._handle_performance_error
        self.recovery_strategies[ErrorCategory.RESOURCE] = self._handle_resource_error
        self.recovery_strategies[ErrorCategory.NETWORK] = self._handle_network_error
        self.recovery_strategies[ErrorCategory.TIMEOUT] = self._handle_timeout_error
        self.recovery_strategies[ErrorCategory.AGENT] = self._handle_agent_error
    
    async def handle_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        component: str = "",
        operation: str = ""
    ) -> Tuple[bool, Optional[Any]]:
        """
        Handle an error with comprehensive recovery strategies.
        
        Args:
            exception: The exception that occurred
            context: Optional error context
            component: Component where error occurred
            operation: Operation that failed
            
        Returns:
            Tuple of (recovered_successfully, recovery_result)
        """
        # Create or update context
        if context is None:
            if isinstance(exception, TektraError):
                context = exception.to_context()
            else:
                context = ErrorContext(
                    error_type=type(exception).__name__,
                    error_message=str(exception),
                    component=component,
                    operation=operation
                )
        
        # Classify error if not already done
        if context.error_category == ErrorCategory.SYSTEM:
            context.error_category = self._classify_error(exception)
        
        # Log error
        await self._log_error(context, exception)
        
        # Store error
        with self._lock:
            self.error_history.append(context)
            self._update_error_tracking(context)
        
        # Attempt recovery
        recovered, result = await self._attempt_recovery(exception, context)
        
        # Update context with recovery info
        context.recovery_attempted = True
        context.recovery_successful = recovered
        
        # Trigger error handlers
        await self._trigger_error_handlers(context, exception)
        
        return recovered, result
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify error based on exception type and message."""
        error_message = str(exception).lower()
        
        # Security-related
        if any(keyword in error_message for keyword in ["permission", "unauthorized", "forbidden", "security"]):
            return ErrorCategory.SECURITY
        
        # Performance-related
        if any(keyword in error_message for keyword in ["timeout", "slow", "performance", "latency"]):
            return ErrorCategory.PERFORMANCE
        
        # Resource-related
        if any(keyword in error_message for keyword in ["memory", "disk", "cpu", "resource"]):
            return ErrorCategory.RESOURCE
        
        # Network-related
        if any(keyword in error_message for keyword in ["connection", "network", "socket", "dns"]):
            return ErrorCategory.NETWORK
        
        # Database-related
        if any(keyword in error_message for keyword in ["database", "sql", "query", "transaction"]):
            return ErrorCategory.DATABASE
        
        # Configuration-related
        if any(keyword in error_message for keyword in ["config", "setting", "parameter", "missing"]):
            return ErrorCategory.CONFIGURATION
        
        # Agent-related
        if any(keyword in error_message for keyword in ["agent", "model", "inference", "generation"]):
            return ErrorCategory.AGENT
        
        return ErrorCategory.SYSTEM
    
    async def _log_error(self, context: ErrorContext, exception: Exception) -> None:
        """Log error with appropriate level and detail."""
        log_data = {
            "error_id": context.error_id,
            "component": context.component,
            "operation": context.operation,
            "category": context.error_category.value,
            "severity": context.severity.value,
            "message": context.error_message,
            "traceback": traceback.format_exc() if context.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL] else None
        }
        
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error in {context.component}: {context.error_message}", **log_data)
        elif context.severity == ErrorSeverity.ERROR:
            logger.error(f"Error in {context.component}: {context.error_message}", **log_data)
        elif context.severity == ErrorSeverity.WARNING:
            logger.warning(f"Warning in {context.component}: {context.error_message}", **log_data)
        else:
            logger.info(f"Info in {context.component}: {context.error_message}", **log_data)
    
    def _update_error_tracking(self, context: ErrorContext) -> None:
        """Update error tracking statistics."""
        error_key = f"{context.component}:{context.error_type}"
        
        # Update counts
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Update patterns
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = []
        self.error_patterns[error_key].append(context.timestamp)
        
        # Keep only recent patterns (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.error_patterns[error_key] = [
            ts for ts in self.error_patterns[error_key] if ts > cutoff_time
        ]
    
    async def _attempt_recovery(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> Tuple[bool, Optional[Any]]:
        """Attempt to recover from the error."""
        category = context.error_category
        
        # Check if component is in maintenance mode
        if self.maintenance_mode:
            logger.info("System in maintenance mode, skipping recovery")
            return False, None
        
        # Use specific recovery strategy if available
        if category in self.recovery_strategies:
            try:
                return await self.recovery_strategies[category](exception, context)
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # Default recovery attempt
        return await self._default_recovery(exception, context)
    
    async def _default_recovery(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> Tuple[bool, Optional[Any]]:
        """Default recovery strategy."""
        # For recoverable errors, suggest retry
        if isinstance(exception, TektraError) and exception.recoverable:
            context.recovery_action = RecoveryAction.RETRY
            return True, "retry_suggested"
        
        # For critical errors, suggest escalation
        if context.severity == ErrorSeverity.CRITICAL:
            context.recovery_action = RecoveryAction.ESCALATE
            return False, "escalation_required"
        
        # For other errors, try graceful degradation
        context.recovery_action = RecoveryAction.GRACEFUL_DEGRADE
        return True, "graceful_degradation"
    
    async def _handle_performance_error(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> Tuple[bool, Optional[Any]]:
        """Handle performance-related errors."""
        # Check if this is a repeated performance issue
        error_key = f"{context.component}:performance"
        recent_errors = len(self.error_patterns.get(error_key, []))
        
        if recent_errors > 5:  # Too many performance errors
            # Degrade component temporarily
            with self._lock:
                self.degraded_components.add(context.component)
            
            context.recovery_action = RecoveryAction.GRACEFUL_DEGRADE
            logger.warning(f"Component {context.component} degraded due to performance issues")
            return True, "component_degraded"
        
        # For single performance issues, suggest retry with backoff
        context.recovery_action = RecoveryAction.RETRY
        return True, "retry_with_backoff"
    
    async def _handle_resource_error(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> Tuple[bool, Optional[Any]]:
        """Handle resource-related errors."""
        # Resource errors often require immediate attention
        if "memory" in context.error_message.lower():
            context.recovery_action = RecoveryAction.RESTART_COMPONENT
            return True, "restart_component"
        
        context.recovery_action = RecoveryAction.GRACEFUL_DEGRADE
        return True, "reduce_resource_usage"
    
    async def _handle_network_error(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> Tuple[bool, Optional[Any]]:
        """Handle network-related errors."""
        # Network errors are often transient
        context.recovery_action = RecoveryAction.RETRY
        return True, "retry_with_exponential_backoff"
    
    async def _handle_timeout_error(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> Tuple[bool, Optional[Any]]:
        """Handle timeout errors."""
        # Check if this is a pattern of timeouts
        error_key = f"{context.component}:timeout"
        recent_timeouts = len(self.error_patterns.get(error_key, []))
        
        if recent_timeouts > 3:
            # Multiple timeouts suggest a deeper issue
            context.recovery_action = RecoveryAction.CIRCUIT_BREAK
            return False, "circuit_breaker_triggered"
        
        context.recovery_action = RecoveryAction.RETRY
        return True, "retry_with_longer_timeout"
    
    async def _handle_agent_error(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> Tuple[bool, Optional[Any]]:
        """Handle agent-related errors."""
        # Agent errors might require model reload or fallback
        if "model" in context.error_message.lower():
            context.recovery_action = RecoveryAction.FALLBACK
            return True, "fallback_model"
        
        context.recovery_action = RecoveryAction.RETRY
        return True, "retry_agent_operation"
    
    async def _trigger_error_handlers(
        self,
        context: ErrorContext,
        exception: Exception
    ) -> None:
        """Trigger registered error handlers."""
        handlers = self.error_handlers.get(context.error_category, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context, exception)
                else:
                    handler(context, exception)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
    
    def register_error_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[ErrorContext, Exception], None]
    ) -> None:
        """Register an error handler for a specific category."""
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        self.error_handlers[category].append(handler)
    
    def register_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig
    ) -> CircuitBreaker:
        """Register a circuit breaker."""
        circuit_breaker = CircuitBreaker(config, name)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            return {
                "total_errors": len(self.error_history),
                "error_counts": self.error_counts.copy(),
                "recent_error_patterns": {
                    key: len(timestamps) for key, timestamps in self.error_patterns.items()
                },
                "degraded_components": list(self.degraded_components),
                "circuit_breaker_stats": {
                    name: cb.get_stats() for name, cb in self.circuit_breakers.items()
                },
                "maintenance_mode": self.maintenance_mode
            }
    
    def set_maintenance_mode(self, enabled: bool) -> None:
        """Enable or disable maintenance mode."""
        self.maintenance_mode = enabled
        logger.info(f"Maintenance mode {'enabled' if enabled else 'disabled'}")
    
    def clear_error_history(self, older_than_hours: int = 24) -> int:
        """Clear old error history."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self._lock:
            initial_count = len(self.error_history)
            self.error_history = [
                error for error in self.error_history
                if error.timestamp > cutoff_time
            ]
            cleared_count = initial_count - len(self.error_history)
        
        logger.info(f"Cleared {cleared_count} old error records")
        return cleared_count


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        config: Retry configuration
        circuit_breaker: Optional circuit breaker
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Apply circuit breaker if provided
                    if circuit_breaker:
                        return await circuit_breaker.call(func, *args, **kwargs)
                    else:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                
                except config.stop_on_exceptions as e:
                    # Don't retry on these exceptions
                    logger.error(f"Non-retryable exception: {e}")
                    raise
                
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    # Calculate delay
                    if config.exponential_backoff:
                        delay = min(config.base_delay * (2 ** attempt), config.max_delay)
                    else:
                        delay = config.base_delay
                    
                    # Add jitter if enabled
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
            
            # All attempts failed
            if last_exception:
                raise last_exception
            else:
                raise TektraError("All retry attempts failed")
        
        return wrapper
    return decorator


def error_boundary(
    component: str,
    operation: str = "",
    error_handler: Optional[ErrorHandler] = None,
    fallback_value: Any = None
):
    """
    Decorator that provides an error boundary for functions.
    
    Args:
        component: Component name for error context
        operation: Operation name for error context
        error_handler: Error handler instance
        fallback_value: Value to return on error
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            except Exception as e:
                # Handle error if handler provided
                if error_handler:
                    context = ErrorContext(
                        component=component,
                        operation=operation or func.__name__
                    )
                    
                    recovered, result = await error_handler.handle_error(e, context)
                    
                    if recovered and result is not None:
                        return result
                
                # Return fallback value if provided
                if fallback_value is not None:
                    logger.warning(f"Error in {component}.{operation}, returning fallback: {e}")
                    return fallback_value
                
                # Re-raise if no recovery
                raise
        
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_global_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


if __name__ == "__main__":
    async def demo_error_handling():
        """Demonstrate error handling capabilities."""
        print("🛡️ Error Handling and Resilience Demo")
        print("=" * 50)
        
        # Create error handler
        error_handler = ErrorHandler()
        
        # Register error handler for performance issues
        def performance_alert_handler(context: ErrorContext, exception: Exception):
            print(f"🚨 Performance Alert: {context.component} - {context.error_message}")
        
        error_handler.register_error_handler(ErrorCategory.PERFORMANCE, performance_alert_handler)
        
        # Create circuit breaker
        cb_config = CircuitBreakerConfig(failure_threshold=2, reset_timeout=5.0)
        circuit_breaker = error_handler.register_circuit_breaker("demo_service", cb_config)
        
        print(f"\n🔧 Testing Error Scenarios:")
        
        # Test 1: Performance error
        try:
            raise PerformanceError("High latency detected", component="agent_service", operation="inference")
        except Exception as e:
            recovered, result = await error_handler.handle_error(e)
            print(f"  Performance Error - Recovered: {recovered}, Result: {result}")
        
        # Test 2: Resource error
        try:
            raise ResourceError("Memory limit exceeded", component="model_loader", operation="load_model")
        except Exception as e:
            recovered, result = await error_handler.handle_error(e)
            print(f"  Resource Error - Recovered: {recovered}, Result: {result}")
        
        # Test 3: Circuit breaker demo
        @circuit_breaker
        async def unreliable_service():
            import random
            if random.random() < 0.7:  # 70% failure rate
                raise Exception("Service unavailable")
            return "Success"
        
        print(f"\n⚡ Testing Circuit Breaker:")
        
        for i in range(8):
            try:
                result = await unreliable_service()
                print(f"  Attempt {i+1}: {result}")
            except Exception as e:
                print(f"  Attempt {i+1}: Failed - {e}")
            
            # Show circuit breaker state
            stats = circuit_breaker.get_stats()
            print(f"    Circuit Breaker State: {stats['state']}")
            
            await asyncio.sleep(0.1)
        
        # Test 4: Retry decorator
        @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.1))
        async def flaky_function():
            import random
            if random.random() < 0.6:  # 60% failure rate
                raise Exception("Temporary failure")
            return "Success after retries"
        
        print(f"\n🔄 Testing Retry Logic:")
        try:
            result = await flaky_function()
            print(f"  Retry Success: {result}")
        except Exception as e:
            print(f"  Retry Failed: {e}")
        
        # Test 5: Error boundary
        @error_boundary("demo_component", "test_operation", error_handler, fallback_value="FALLBACK")
        async def boundary_protected_function():
            raise Exception("This will be caught by error boundary")
        
        result = await boundary_protected_function()
        print(f"\n🛡️ Error Boundary Result: {result}")
        
        # Show statistics
        print(f"\n📊 Error Statistics:")
        stats = error_handler.get_error_statistics()
        print(f"  Total Errors: {stats['total_errors']}")
        print(f"  Error Counts: {stats['error_counts']}")
        print(f"  Degraded Components: {stats['degraded_components']}")
        
        print(f"\n🛡️ Error Handling Demo Complete")
    
    # Run demo
    asyncio.run(demo_error_handling())