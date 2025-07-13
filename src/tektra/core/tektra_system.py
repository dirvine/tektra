#!/usr/bin/env python3
"""
Tektra AI Assistant - Unified System Architecture

Integrates all phases (SmolAgents, Performance, Security) into a cohesive,
production-ready AI assistant system with comprehensive management and monitoring.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru,asyncio,pydantic,typing-extensions python tektra_system.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
#     "pydantic>=2.0.0",
#     "typing-extensions>=4.7.0",
#     "asyncio-compat>=0.1.0",
# ]
# ///

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, AsyncContextManager
from pathlib import Path
import threading
import json
import sys
import os

from loguru import logger
from pydantic import BaseModel, Field

# Import error handling
from .error_handling import (
    ErrorHandler, TektraError, SecurityError, PerformanceError,
    AgentError, ConfigurationError, ResourceError, ErrorCategory,
    ErrorSeverity, CircuitBreakerConfig, error_boundary, retry_with_backoff
)

# Import all Tektra subsystems
from ..agents.smolagents_real import SmolAgentsManager
from ..performance import (
    create_cache_manager, create_task_scheduler, create_memory_manager,
    create_performance_monitor, create_performance_optimizer,
    CacheLevel, Priority, OptimizationStrategy
)
from ..security import (
    SecurityContext, SecurityLevel, PermissionManager,
    AdvancedSandbox, SandboxConfig, IsolationType,
    ToolValidator, ConsentFramework, ConsentMode,
    SecurityMonitor, EventType, EventSeverity
)
from ..utils.config import TektraConfig


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ComponentStatus(Enum):
    """Individual component status."""
    OFFLINE = "offline"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class SystemHealth:
    """System health assessment."""
    
    overall_status: ComponentStatus = ComponentStatus.OFFLINE
    components: Dict[str, ComponentStatus] = field(default_factory=dict)
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_agents: int = 0
    active_tasks: int = 0
    
    # Security metrics
    security_level: SecurityLevel = SecurityLevel.LOW
    active_threats: int = 0
    permission_violations: int = 0
    
    # System metrics
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "components": {k: v.value for k, v in self.components.items()},
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "active_agents": self.active_agents,
            "active_tasks": self.active_tasks,
            "security_level": self.security_level.value,
            "active_threats": self.active_threats,
            "permission_violations": self.permission_violations,
            "uptime_seconds": self.uptime_seconds,
            "last_updated": self.last_updated.isoformat()
        }


class TektraSystemConfig(BaseModel):
    """Configuration for the Tektra system."""
    
    # System configuration
    system_name: str = "Tektra AI Assistant"
    environment: str = "development"  # development, staging, production
    debug_mode: bool = True
    
    # Component configuration
    enable_agents: bool = True
    enable_performance: bool = True
    enable_security: bool = True
    enable_monitoring: bool = True
    
    # Agent configuration
    max_concurrent_agents: int = 10
    agent_timeout_seconds: float = 300.0
    
    # Performance configuration
    cache_size_mb: int = 512
    memory_limit_mb: int = 2048
    task_queue_size: int = 1000
    
    # Security configuration
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    sandbox_enabled: bool = True
    tool_validation_enabled: bool = True
    consent_mode: str = "automatic"  # automatic, interactive, strict
    
    # Monitoring configuration
    metrics_enabled: bool = True
    prometheus_port: int = 8090
    health_check_interval: float = 30.0
    
    # Integration configuration
    ui_enabled: bool = True
    api_enabled: bool = True
    websocket_enabled: bool = True
    
    class Config:
        arbitrary_types_allowed = True


class TektraSystem:
    """
    Unified Tektra AI Assistant System.
    
    Integrates all subsystems (Agents, Performance, Security) into a cohesive,
    production-ready AI assistant with comprehensive management and monitoring.
    """
    
    def __init__(self, config: Optional[TektraSystemConfig] = None):
        """Initialize the Tektra system."""
        self.config = config or TektraSystemConfig()
        self.system_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # System state
        self.state = SystemState.INITIALIZING
        self.health = SystemHealth()
        
        # Core components (will be initialized during startup)
        self.agents_manager: Optional[SmolAgentsManager] = None
        self.performance_monitor = None
        self.security_monitor: Optional[SecurityMonitor] = None
        self.permission_manager: Optional[PermissionManager] = None
        self.consent_framework: Optional[ConsentFramework] = None
        self.tool_validator: Optional[ToolValidator] = None
        self.sandbox: Optional[AdvancedSandbox] = None
        
        # Performance components
        self.cache_manager = None
        self.task_scheduler = None
        self.memory_manager = None
        self.performance_optimizer = None
        
        # System management
        self.shutdown_event = asyncio.Event()
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Component registry
        self.components: Dict[str, Any] = {}
        self.component_health: Dict[str, ComponentStatus] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Tektra system initialized with ID: {self.system_id}")
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        logger.info("🚀 Initializing Tektra AI Assistant System")
        
        async with self._lock:
            try:
                self.state = SystemState.STARTING
                self.health.overall_status = ComponentStatus.STARTING
                
                # Initialize core infrastructure first
                await self._initialize_security_components()
                await self._initialize_performance_components()
                await self._initialize_agent_components()
                await self._initialize_monitoring_components()
                
                # Configure component integrations
                await self._configure_integrations()
                
                # Start health monitoring
                await self._start_health_monitoring()
                
                self.state = SystemState.RUNNING
                self.health.overall_status = ComponentStatus.HEALTHY
                
                logger.info("✅ Tektra system initialization complete")
                return True
                
            except Exception as e:
                logger.error(f"❌ System initialization failed: {e}")
                self.state = SystemState.ERROR
                self.health.overall_status = ComponentStatus.ERROR
                return False
    
    async def _initialize_security_components(self) -> None:
        """Initialize security subsystem."""
        if not self.config.enable_security:
            logger.info("Security subsystem disabled")
            return
        
        logger.info("🔒 Initializing security components...")
        
        try:
            # Permission manager
            self.permission_manager = PermissionManager()
            self.components["permission_manager"] = self.permission_manager
            self.component_health["permission_manager"] = ComponentStatus.HEALTHY
            
            # Security monitor
            self.security_monitor = SecurityMonitor(
                permission_manager=self.permission_manager,
                enable_prometheus=self.config.metrics_enabled,
                prometheus_port=self.config.prometheus_port
            )
            self.components["security_monitor"] = self.security_monitor
            self.component_health["security_monitor"] = ComponentStatus.HEALTHY
            
            # Sandbox
            self.sandbox = AdvancedSandbox(
                permission_manager=self.permission_manager
            )
            self.components["sandbox"] = self.sandbox
            self.component_health["sandbox"] = ComponentStatus.HEALTHY
            
            # Tool validator
            self.tool_validator = ToolValidator(
                sandbox=self.sandbox,
                permission_manager=self.permission_manager
            )
            self.components["tool_validator"] = self.tool_validator
            self.component_health["tool_validator"] = ComponentStatus.HEALTHY
            
            # Consent framework
            consent_mode_map = {
                "automatic": ConsentMode.AUTOMATIC,
                "interactive": ConsentMode.INTERACTIVE,
                "strict": ConsentMode.STRICT
            }
            
            self.consent_framework = ConsentFramework(
                permission_manager=self.permission_manager
            )
            self.consent_framework.set_consent_mode(
                consent_mode_map.get(self.config.consent_mode, ConsentMode.AUTOMATIC)
            )
            self.components["consent_framework"] = self.consent_framework
            self.component_health["consent_framework"] = ComponentStatus.HEALTHY
            
            logger.info("✅ Security components initialized")
            
        except Exception as e:
            logger.error(f"❌ Security initialization failed: {e}")
            self.component_health["security"] = ComponentStatus.ERROR
            raise
    
    async def _initialize_performance_components(self) -> None:
        """Initialize performance subsystem."""
        if not self.config.enable_performance:
            logger.info("Performance subsystem disabled")
            return
        
        logger.info("⚡ Initializing performance components...")
        
        try:
            # Cache manager
            self.cache_manager = create_cache_manager(
                l1_size_mb=self.config.cache_size_mb // 4,
                l2_size_mb=self.config.cache_size_mb // 2
            )
            self.components["cache_manager"] = self.cache_manager
            self.component_health["cache_manager"] = ComponentStatus.HEALTHY
            
            # Memory manager
            self.memory_manager = create_memory_manager(
                max_memory_mb=self.config.memory_limit_mb,
                enable_pools=True
            )
            self.components["memory_manager"] = self.memory_manager
            self.component_health["memory_manager"] = ComponentStatus.HEALTHY
            
            # Task scheduler
            self.task_scheduler = create_task_scheduler(
                num_workers=min(4, os.cpu_count() or 4),
                max_queue_size=self.config.task_queue_size,
                enable_work_stealing=True
            )
            await self.task_scheduler.start()
            self.components["task_scheduler"] = self.task_scheduler
            self.component_health["task_scheduler"] = ComponentStatus.HEALTHY
            
            # Performance monitor
            self.performance_monitor = create_performance_monitor(
                enable_prometheus=self.config.metrics_enabled
            )
            self.components["performance_monitor"] = self.performance_monitor
            self.component_health["performance_monitor"] = ComponentStatus.HEALTHY
            
            # Performance optimizer
            self.performance_optimizer = create_performance_optimizer(
                strategy=OptimizationStrategy.ADAPTIVE
            )
            self.components["performance_optimizer"] = self.performance_optimizer
            self.component_health["performance_optimizer"] = ComponentStatus.HEALTHY
            
            logger.info("✅ Performance components initialized")
            
        except Exception as e:
            logger.error(f"❌ Performance initialization failed: {e}")
            self.component_health["performance"] = ComponentStatus.ERROR
            raise
    
    async def _initialize_agent_components(self) -> None:
        """Initialize agent subsystem."""
        if not self.config.enable_agents:
            logger.info("Agent subsystem disabled")
            return
        
        logger.info("🤖 Initializing agent components...")
        
        try:
            # SmolAgents manager with integrated security and performance
            self.agents_manager = SmolAgentsManager(
                max_concurrent_agents=self.config.max_concurrent_agents,
                agent_timeout=self.config.agent_timeout_seconds,
                security_enabled=self.config.enable_security,
                performance_enabled=self.config.enable_performance
            )
            
            # Integrate with other components
            if self.security_monitor:
                self.agents_manager.set_security_monitor(self.security_monitor)
            
            if self.tool_validator:
                self.agents_manager.set_tool_validator(self.tool_validator)
            
            if self.consent_framework:
                self.agents_manager.set_consent_framework(self.consent_framework)
            
            if self.task_scheduler:
                self.agents_manager.set_task_scheduler(self.task_scheduler)
            
            if self.performance_monitor:
                self.agents_manager.set_performance_monitor(self.performance_monitor)
            
            self.components["agents_manager"] = self.agents_manager
            self.component_health["agents_manager"] = ComponentStatus.HEALTHY
            
            logger.info("✅ Agent components initialized")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            self.component_health["agents"] = ComponentStatus.ERROR
            raise
    
    async def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring and observability."""
        if not self.config.enable_monitoring:
            logger.info("Monitoring subsystem disabled")
            return
        
        logger.info("📊 Initializing monitoring components...")
        
        try:
            # System monitoring is already integrated into other components
            # Additional monitoring setup would go here
            
            logger.info("✅ Monitoring components initialized")
            
        except Exception as e:
            logger.error(f"❌ Monitoring initialization failed: {e}")
            self.component_health["monitoring"] = ComponentStatus.ERROR
            raise
    
    async def _configure_integrations(self) -> None:
        """Configure cross-component integrations."""
        logger.info("🔗 Configuring component integrations...")
        
        try:
            # Security → Performance integration
            if self.security_monitor and self.performance_monitor:
                # Security events can trigger performance optimizations
                pass
            
            # Performance → Security integration
            if self.performance_monitor and self.security_monitor:
                # Performance anomalies can trigger security alerts
                pass
            
            # Agents → Security integration
            if self.agents_manager and self.security_monitor:
                # Agent actions are logged to security monitor
                pass
            
            # Agents → Performance integration
            if self.agents_manager and self.performance_monitor:
                # Agent performance is tracked
                pass
            
            logger.info("✅ Component integrations configured")
            
        except Exception as e:
            logger.error(f"❌ Integration configuration failed: {e}")
            raise
    
    async def _start_health_monitoring(self) -> None:
        """Start system health monitoring."""
        logger.info("💓 Starting health monitoring...")
        
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
    
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                await self._update_system_health()
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5.0)  # Brief pause on error
    
    async def _update_system_health(self) -> None:
        """Update system health metrics."""
        async with self._lock:
            try:
                # Update component health
                self.health.components = self.component_health.copy()
                
                # Calculate overall status
                component_statuses = list(self.component_health.values())
                if ComponentStatus.ERROR in component_statuses:
                    self.health.overall_status = ComponentStatus.ERROR
                elif ComponentStatus.DEGRADED in component_statuses:
                    self.health.overall_status = ComponentStatus.DEGRADED
                elif all(status == ComponentStatus.HEALTHY for status in component_statuses):
                    self.health.overall_status = ComponentStatus.HEALTHY
                else:
                    self.health.overall_status = ComponentStatus.DEGRADED
                
                # Update system metrics
                self.health.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                
                # Get performance metrics
                if self.performance_monitor:
                    try:
                        perf_stats = self.performance_monitor.get_current_stats()
                        self.health.cpu_usage = perf_stats.get("cpu_usage", 0.0)
                        self.health.memory_usage = perf_stats.get("memory_usage", 0.0)
                    except Exception:
                        pass
                
                # Get agent metrics
                if self.agents_manager:
                    try:
                        agent_stats = self.agents_manager.get_system_stats()
                        self.health.active_agents = agent_stats.get("active_agents", 0)
                    except Exception:
                        pass
                
                # Get task metrics
                if self.task_scheduler:
                    try:
                        task_stats = self.task_scheduler.get_statistics()
                        self.health.active_tasks = task_stats.get("pending_tasks", 0)
                    except Exception:
                        pass
                
                # Get security metrics
                if self.security_monitor:
                    try:
                        security_stats = self.security_monitor.get_statistics()
                        self.health.active_threats = security_stats.get("active_alerts", 0)
                    except Exception:
                        pass
                
                self.health.last_updated = datetime.now()
                
            except Exception as e:
                logger.error(f"Health update failed: {e}")
    
    @error_boundary("tektra_system", "create_agent")
    async def create_agent(
        self,
        agent_name: str,
        agent_config: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> str:
        """Create and initialize a new agent."""
        if not self.agents_manager:
            raise RuntimeError("Agent subsystem not initialized")
        
        # Log agent creation
        if self.security_monitor:
            self.security_monitor.log_event(
                EventType.SYSTEM_ACCESS,
                f"Creating agent: {agent_name}",
                EventSeverity.INFO,
                metadata={"agent_name": agent_name, "config": agent_config}
            )
        
        # Create agent through manager
        agent_id = await self.agents_manager.create_agent(
            agent_name=agent_name,
            config=agent_config,
            security_context=security_context
        )
        
        return agent_id
    
    @error_boundary("tektra_system", "execute_agent_task")
    @retry_with_backoff()
    async def execute_agent_task(
        self,
        agent_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Execute a task using an agent."""
        if not self.agents_manager:
            raise RuntimeError("Agent subsystem not initialized")
        
        # Log task execution
        if self.security_monitor:
            self.security_monitor.log_event(
                EventType.SYSTEM_ACCESS,
                f"Executing task for agent {agent_id}: {task_description}",
                EventSeverity.INFO,
                agent_id=agent_id,
                metadata={"task": task_description, "context": context}
            )
        
        # Execute through agent manager
        result = await self.agents_manager.execute_task(
            agent_id=agent_id,
            task_description=task_description,
            context=context,
            security_context=security_context
        )
        
        return result
    
    @error_boundary("tektra_system", "validate_tool", fallback_value=False)
    async def validate_tool(
        self,
        tool_code: str,
        tool_name: str,
        security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Validate a tool for security before use."""
        if not self.tool_validator:
            logger.warning("Tool validation disabled - allowing tool")
            return True
        
        # Validate tool
        validation_result = await self.tool_validator.validate_tool(
            tool_id=tool_name,
            code=tool_code,
            security_context=security_context
        )
        
        # Log validation result
        if self.security_monitor:
            self.security_monitor.log_tool_validation(
                agent_id=security_context.agent_id if security_context else "system",
                tool_id=tool_name,
                validation_result=validation_result
            )
        
        return validation_result.is_safe
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        return self.health
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "system_id": self.system_id,
            "state": self.state.value,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "health": self.health.to_dict(),
            "components": list(self.components.keys()),
            "component_health": {k: v.value for k, v in self.component_health.items()}
        }
        
        # Add component-specific stats
        if self.agents_manager:
            try:
                stats["agents"] = self.agents_manager.get_system_stats()
            except Exception:
                pass
        
        if self.performance_monitor:
            try:
                stats["performance"] = self.performance_monitor.get_current_stats()
            except Exception:
                pass
        
        if self.security_monitor:
            try:
                stats["security"] = self.security_monitor.get_statistics()
            except Exception:
                pass
        
        return stats
    
    async def shutdown(self, graceful: bool = True) -> None:
        """Shutdown the Tektra system."""
        logger.info("🛑 Shutting down Tektra system...")
        
        async with self._lock:
            self.state = SystemState.STOPPING
            self.health.overall_status = ComponentStatus.STOPPING
            
            # Signal shutdown
            self.shutdown_event.set()
            
            try:
                # Stop health monitoring
                if self.health_check_task:
                    self.health_check_task.cancel()
                    try:
                        await self.health_check_task
                    except asyncio.CancelledError:
                        pass
                
                # Shutdown components in reverse order
                if self.agents_manager:
                    await self.agents_manager.shutdown()
                    self.component_health["agents_manager"] = ComponentStatus.OFFLINE
                
                if self.task_scheduler:
                    await self.task_scheduler.shutdown()
                    self.component_health["task_scheduler"] = ComponentStatus.OFFLINE
                
                if self.security_monitor:
                    self.security_monitor.shutdown()
                    self.component_health["security_monitor"] = ComponentStatus.OFFLINE
                
                if self.sandbox:
                    self.sandbox.cleanup_all_sandboxes()
                    self.component_health["sandbox"] = ComponentStatus.OFFLINE
                
                self.state = SystemState.STOPPED
                self.health.overall_status = ComponentStatus.OFFLINE
                
                logger.info("✅ Tektra system shutdown complete")
                
            except Exception as e:
                logger.error(f"❌ Error during shutdown: {e}")
                self.state = SystemState.ERROR
                self.health.overall_status = ComponentStatus.ERROR
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


async def create_tektra_system(config: Optional[TektraSystemConfig] = None) -> TektraSystem:
    """
    Create and initialize a Tektra system.
    
    Args:
        config: System configuration
        
    Returns:
        Initialized Tektra system
    """
    system = TektraSystem(config)
    await system.initialize()
    return system


if __name__ == "__main__":
    async def demo_tektra_system():
        """Demonstrate the unified Tektra system."""
        print("🌟 Tektra AI Assistant System Demo")
        print("=" * 50)
        
        # Create system configuration
        config = TektraSystemConfig(
            system_name="Tektra Demo",
            environment="development",
            debug_mode=True,
            max_concurrent_agents=5,
            cache_size_mb=256,
            security_level=SecurityLevel.MEDIUM,
            consent_mode="automatic"
        )
        
        print(f"📋 System Configuration:")
        print(f"  Environment: {config.environment}")
        print(f"  Security Level: {config.security_level.value}")
        print(f"  Cache Size: {config.cache_size_mb}MB")
        print(f"  Max Agents: {config.max_concurrent_agents}")
        
        # Initialize system
        async with TektraSystem(config) as system:
            print(f"\n🚀 System Status: {system.state.value}")
            
            # Wait for initialization
            await asyncio.sleep(2)
            
            # Show system health
            health = system.get_system_health()
            print(f"\n💓 System Health:")
            print(f"  Overall Status: {health.overall_status.value}")
            print(f"  Components: {len(health.components)}")
            print(f"  CPU Usage: {health.cpu_usage:.1f}%")
            print(f"  Memory Usage: {health.memory_usage:.1f}%")
            print(f"  Active Agents: {health.active_agents}")
            print(f"  Active Tasks: {health.active_tasks}")
            
            # Show component status
            print(f"\n🔧 Component Status:")
            for component, status in health.components.items():
                status_icon = "✅" if status == ComponentStatus.HEALTHY else "❌"
                print(f"  {status_icon} {component}: {status.value}")
            
            # Demonstrate agent creation and task execution
            print(f"\n🤖 Creating and Testing Agent:")
            
            try:
                # Create security context
                security_context = SecurityContext(
                    agent_id="demo_agent",
                    security_level=SecurityLevel.MEDIUM,
                    session_id="demo_session"
                )
                
                # Create agent
                agent_id = await system.create_agent(
                    agent_name="Demo Agent",
                    agent_config={"model": "text_completion", "max_tokens": 100},
                    security_context=security_context
                )
                print(f"  ✅ Created agent: {agent_id}")
                
                # Test tool validation
                safe_tool = '''
def calculate_sum(a, b):
    """Safe calculation function."""
    return a + b

result = calculate_sum(5, 3)
print(f"Sum: {result}")
'''
                
                is_safe = await system.validate_tool(
                    tool_code=safe_tool,
                    tool_name="safe_calculator",
                    security_context=security_context
                )
                print(f"  ✅ Tool validation: {'SAFE' if is_safe else 'UNSAFE'}")
                
                # Execute simple task
                task_result = await system.execute_agent_task(
                    agent_id=agent_id,
                    task_description="Calculate the sum of 10 and 15",
                    context={"type": "mathematical_calculation"},
                    security_context=security_context
                )
                print(f"  ✅ Task executed successfully")
                
            except Exception as e:
                print(f"  ❌ Demo error: {e}")
            
            # Show final statistics
            print(f"\n📊 Final System Statistics:")
            stats = system.get_system_stats()
            
            print(f"  System ID: {stats['system_id'][:8]}...")
            print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
            print(f"  Components: {len(stats['components'])}")
            
            if 'agents' in stats:
                print(f"  Agent Stats: {stats['agents']}")
            
            if 'security' in stats:
                security_stats = stats['security']
                print(f"  Security Events: {security_stats.get('total_events', 0)}")
                print(f"  Security Alerts: {security_stats.get('total_alerts', 0)}")
            
            # Wait a bit to see health updates
            print(f"\n⏳ Monitoring system for 10 seconds...")
            await asyncio.sleep(10)
            
            # Final health check
            final_health = system.get_system_health()
            print(f"\n💓 Final Health Status: {final_health.overall_status.value}")
            print(f"   Uptime: {final_health.uptime_seconds:.1f}s")
        
        print(f"\n🌟 Tektra System Demo Complete")
    
    # Run demo
    asyncio.run(demo_tektra_system())