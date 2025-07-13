#!/usr/bin/env python3
"""
Tektra AI Assistant - Deployment Management

Comprehensive deployment management for production environments including
health monitoring, graceful shutdown, and deployment orchestration.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru,asyncio,psutil,prometheus_client python deployment_manager.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
#     "asyncio-compat>=0.1.0",
#     "psutil>=5.9.0",
#     "prometheus_client>=0.17.0",
#     "pydantic>=2.0.0",
# ]
# ///

import asyncio
import os
import signal
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
import json
import psutil

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pydantic import BaseModel

from .tektra_system import TektraSystem, TektraSystemConfig, SystemState, ComponentStatus
from .error_handling import ErrorHandler, TektraError, ErrorCategory, ErrorSeverity
from ..config.production_config import ProductionConfig


class DeploymentStatus(Enum):
    """Deployment status states."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class HealthCheckResult(Enum):
    """Health check results."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"


@dataclass
class DeploymentMetrics:
    """Deployment metrics for monitoring."""
    
    # System metrics
    uptime_seconds: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_usage_bytes: int = 0
    disk_usage_percent: float = 0.0
    
    # Application metrics
    active_agents: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    
    # Component health
    healthy_components: int = 0
    total_components: int = 0
    
    # Performance metrics
    cache_hit_rate: float = 0.0
    queue_length: int = 0
    concurrent_connections: int = 0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "memory_usage_bytes": self.memory_usage_bytes,
            "disk_usage_percent": self.disk_usage_percent,
            "active_agents": self.active_agents,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "average_response_time": self.average_response_time,
            "healthy_components": self.healthy_components,
            "total_components": self.total_components,
            "cache_hit_rate": self.cache_hit_rate,
            "queue_length": self.queue_length,
            "concurrent_connections": self.concurrent_connections,
            "timestamp": self.timestamp.isoformat()
        }


class PrometheusMetrics:
    """Prometheus metrics collection."""
    
    def __init__(self):
        # System metrics
        self.uptime = Gauge('tektra_uptime_seconds', 'Uptime in seconds')
        self.cpu_usage = Gauge('tektra_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('tektra_memory_usage_percent', 'Memory usage percentage')
        self.memory_bytes = Gauge('tektra_memory_usage_bytes', 'Memory usage in bytes')
        self.disk_usage = Gauge('tektra_disk_usage_percent', 'Disk usage percentage')
        
        # Application metrics
        self.active_agents = Gauge('tektra_active_agents', 'Number of active agents')
        self.total_requests = Counter('tektra_requests_total', 'Total requests', ['method', 'endpoint'])
        self.failed_requests = Counter('tektra_requests_failed', 'Failed requests', ['error_type'])
        self.response_time = Histogram('tektra_response_time_seconds', 'Response time in seconds')
        
        # Component health
        self.component_status = Gauge('tektra_component_status', 'Component status', ['component'])
        self.health_checks = Counter('tektra_health_checks_total', 'Health checks', ['status'])
        
        # Performance metrics
        self.cache_hit_rate = Gauge('tektra_cache_hit_rate', 'Cache hit rate')
        self.queue_length = Gauge('tektra_queue_length', 'Task queue length')
        self.connections = Gauge('tektra_concurrent_connections', 'Concurrent connections')
    
    def update_from_deployment_metrics(self, metrics: DeploymentMetrics):
        """Update Prometheus metrics from deployment metrics."""
        self.uptime.set(metrics.uptime_seconds)
        self.cpu_usage.set(metrics.cpu_usage_percent)
        self.memory_usage.set(metrics.memory_usage_percent)
        self.memory_bytes.set(metrics.memory_usage_bytes)
        self.disk_usage.set(metrics.disk_usage_percent)
        
        self.active_agents.set(metrics.active_agents)
        self.cache_hit_rate.set(metrics.cache_hit_rate)
        self.queue_length.set(metrics.queue_length)
        self.connections.set(metrics.concurrent_connections)


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, tektra_system: TektraSystem):
        self.tektra_system = tektra_system
        self.last_check_time: Optional[datetime] = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
    
    async def check_system_health(self, timeout: float = 10.0) -> HealthCheckResult:
        """Perform comprehensive system health check."""
        try:
            start_time = time.time()
            
            # Check system components
            system_health = self.tektra_system.get_system_health()
            
            # Check if any critical components are unhealthy
            critical_components = ['agents_manager', 'security_monitor', 'performance_monitor']
            unhealthy_critical = [
                comp for comp in critical_components 
                if system_health.components.get(comp) == ComponentStatus.ERROR
            ]
            
            if unhealthy_critical:
                self.consecutive_failures += 1
                return HealthCheckResult.UNHEALTHY
            
            # Check for degraded components
            degraded_components = [
                comp for comp, status in system_health.components.items()
                if status == ComponentStatus.DEGRADED
            ]
            
            if degraded_components:
                if len(degraded_components) > len(system_health.components) * 0.3:  # More than 30% degraded
                    self.consecutive_failures += 1
                    return HealthCheckResult.UNHEALTHY
                else:
                    return HealthCheckResult.DEGRADED
            
            # Check response time
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.consecutive_failures += 1
                return HealthCheckResult.TIMEOUT
            
            # Check system resources
            if system_health.memory_usage > 95 or system_health.cpu_usage > 95:
                return HealthCheckResult.DEGRADED
            
            # All checks passed
            self.consecutive_failures = 0
            self.last_check_time = datetime.now()
            return HealthCheckResult.HEALTHY
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.consecutive_failures += 1
            return HealthCheckResult.UNHEALTHY
    
    def is_system_failing(self) -> bool:
        """Check if system is consistently failing."""
        return self.consecutive_failures >= self.max_consecutive_failures


class DeploymentManager:
    """
    Comprehensive deployment management for production environments.
    
    Handles startup, shutdown, health monitoring, and deployment orchestration.
    """
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.deployment_status = DeploymentStatus.INITIALIZING
        self.start_time = datetime.now()
        
        # Core components
        self.tektra_system: Optional[TektraSystem] = None
        self.health_checker: Optional[HealthChecker] = None
        self.error_handler = ErrorHandler()
        
        # Monitoring
        self.prometheus_metrics = PrometheusMetrics()
        self.metrics_server_started = False
        
        # Runtime state
        self.shutdown_event = asyncio.Event()
        self.health_check_task: Optional[asyncio.Task] = None
        self.metrics_collection_task: Optional[asyncio.Task] = None
        
        # Signal handlers
        self.signal_handlers_registered = False
        
        # Deployment info
        self.deployment_id = f"tektra-{int(time.time())}"
        self.instance_id = os.environ.get('HOSTNAME', 'unknown')
        
        logger.info(f"Deployment Manager initialized: {self.deployment_id}")
    
    async def initialize(self) -> bool:
        """Initialize the deployment manager and all components."""
        try:
            logger.info("🚀 Initializing Tektra Deployment Manager")
            self.deployment_status = DeploymentStatus.STARTING
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start metrics server
            await self._start_metrics_server()
            
            # Initialize Tektra system
            tektra_config = self.config.to_tektra_config()
            self.tektra_system = TektraSystem(tektra_config)
            
            # Initialize system
            if not await self.tektra_system.initialize():
                raise TektraError("Failed to initialize Tektra system")
            
            # Setup health checker
            self.health_checker = HealthChecker(self.tektra_system)
            
            # Start monitoring tasks
            self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
            self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            
            self.deployment_status = DeploymentStatus.HEALTHY
            logger.info("✅ Deployment Manager initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment Manager initialization failed: {e}")
            self.deployment_status = DeploymentStatus.UNHEALTHY
            return False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        if self.signal_handlers_registered:
            return
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        self.signal_handlers_registered = True
        logger.info("Signal handlers registered")
    
    async def _start_metrics_server(self):
        """Start Prometheus metrics server."""
        if self.config.monitoring.prometheus_enabled and not self.metrics_server_started:
            try:
                start_http_server(self.config.monitoring.prometheus_port)
                self.metrics_server_started = True
                logger.info(f"📊 Prometheus metrics server started on port {self.config.monitoring.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                if self.health_checker:
                    health_result = await self.health_checker.check_system_health()
                    
                    # Update Prometheus metrics
                    self.prometheus_metrics.health_checks.labels(status=health_result.value).inc()
                    
                    # Update deployment status based on health
                    if health_result == HealthCheckResult.HEALTHY:
                        if self.deployment_status != DeploymentStatus.HEALTHY:
                            logger.info("✅ System health restored")
                            self.deployment_status = DeploymentStatus.HEALTHY
                    
                    elif health_result == HealthCheckResult.DEGRADED:
                        if self.deployment_status == DeploymentStatus.HEALTHY:
                            logger.warning("⚠️ System health degraded")
                            self.deployment_status = DeploymentStatus.DEGRADED
                    
                    elif health_result in [HealthCheckResult.UNHEALTHY, HealthCheckResult.TIMEOUT]:
                        if self.deployment_status not in [DeploymentStatus.UNHEALTHY, DeploymentStatus.SHUTTING_DOWN]:
                            logger.error("❌ System health critical")
                            self.deployment_status = DeploymentStatus.UNHEALTHY
                        
                        # Check if system is consistently failing
                        if self.health_checker.is_system_failing():
                            logger.critical("💀 System consistently failing, initiating emergency shutdown")
                            await self.emergency_shutdown()
                
                await asyncio.sleep(self.config.monitoring.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection loop."""
        while not self.shutdown_event.is_set():
            try:
                metrics = await self._collect_deployment_metrics()
                
                # Update Prometheus metrics
                self.prometheus_metrics.update_from_deployment_metrics(metrics)
                
                # Log metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info(f"📊 Metrics: CPU={metrics.cpu_usage_percent:.1f}%, "
                              f"Memory={metrics.memory_usage_percent:.1f}%, "
                              f"Agents={metrics.active_agents}")
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10.0)
    
    async def _collect_deployment_metrics(self) -> DeploymentMetrics:
        """Collect comprehensive deployment metrics."""
        metrics = DeploymentMetrics()
        
        try:
            # System metrics
            metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # CPU and Memory
            process = psutil.Process()
            metrics.cpu_usage_percent = process.cpu_percent()
            
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            metrics.memory_usage_bytes = memory_info.rss
            metrics.memory_usage_percent = (memory_info.rss / system_memory.total) * 100
            
            # Disk usage
            data_dir = Path(self.config.data_dir)
            if data_dir.exists():
                disk_usage = psutil.disk_usage(data_dir)
                metrics.disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Application metrics from Tektra system
            if self.tektra_system:
                system_health = self.tektra_system.get_system_health()
                metrics.active_agents = system_health.active_agents
                
                # Component health
                metrics.total_components = len(system_health.components)
                metrics.healthy_components = sum(
                    1 for status in system_health.components.values()
                    if status == ComponentStatus.HEALTHY
                )
                
                # Get additional stats if available
                system_stats = self.tektra_system.get_system_stats()
                if 'performance' in system_stats:
                    perf_stats = system_stats['performance']
                    metrics.cache_hit_rate = perf_stats.get('cache_hit_rate', 0.0)
                    metrics.queue_length = perf_stats.get('queue_length', 0)
        
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status information."""
        return {
            "deployment_id": self.deployment_id,
            "instance_id": self.instance_id,
            "status": self.deployment_status.value,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "start_time": self.start_time.isoformat(),
            "environment": self.config.environment.value,
            "version": self.config.app_version,
            "healthy": self.deployment_status in [DeploymentStatus.HEALTHY, DeploymentStatus.DEGRADED],
            "components": {
                "tektra_system": bool(self.tektra_system),
                "health_checker": bool(self.health_checker),
                "metrics_server": self.metrics_server_started
            }
        }
    
    async def emergency_shutdown(self):
        """Emergency shutdown for critical failures."""
        logger.critical("🚨 Initiating emergency shutdown")
        
        self.deployment_status = DeploymentStatus.SHUTTING_DOWN
        
        # Signal shutdown immediately
        self.shutdown_event.set()
        
        # Force shutdown after a short grace period
        await asyncio.sleep(5.0)
        logger.critical("💀 Force shutdown")
        os._exit(1)
    
    async def shutdown(self, graceful: bool = True):
        """Shutdown the deployment manager and all components."""
        if self.deployment_status == DeploymentStatus.SHUTTING_DOWN:
            return
        
        logger.info("🛑 Shutting down Tektra Deployment Manager")
        self.deployment_status = DeploymentStatus.SHUTTING_DOWN
        
        # Signal shutdown
        self.shutdown_event.set()
        
        try:
            # Cancel monitoring tasks
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self.metrics_collection_task:
                self.metrics_collection_task.cancel()
                try:
                    await self.metrics_collection_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown Tektra system
            if self.tektra_system:
                await self.tektra_system.shutdown(graceful=graceful)
            
            self.deployment_status = DeploymentStatus.STOPPED
            logger.info("✅ Deployment Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            self.deployment_status = DeploymentStatus.STOPPED
    
    async def run(self):
        """Run the deployment manager."""
        try:
            if not await self.initialize():
                logger.error("Failed to initialize deployment manager")
                return False
            
            logger.info("🌟 Tektra Deployment Manager is running")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment manager run error: {e}")
            await self.shutdown(graceful=False)
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


async def create_deployment_manager(config: Optional[ProductionConfig] = None) -> DeploymentManager:
    """Create and initialize a deployment manager."""
    if config is None:
        config = ProductionConfig()
    
    manager = DeploymentManager(config)
    await manager.initialize()
    return manager


if __name__ == "__main__":
    async def main():
        """Main deployment manager entry point."""
        print("🌟 Tektra AI Assistant - Deployment Manager")
        print("=" * 50)
        
        # Load configuration
        config = ProductionConfig()
        
        print(f"Environment: {config.environment.value}")
        print(f"Debug Mode: {config.debug}")
        print(f"Monitoring: {'Enabled' if config.monitoring.prometheus_enabled else 'Disabled'}")
        
        # Create and run deployment manager
        async with DeploymentManager(config) as manager:
            print(f"\n🚀 Deployment Manager Status:")
            status = manager.get_deployment_status()
            for key, value in status.items():
                print(f"  {key}: {value}")
            
            print(f"\n✅ Tektra AI Assistant is running")
            print(f"   Deployment ID: {manager.deployment_id}")
            print(f"   Health checks: http://localhost:8000/health")
            print(f"   Metrics: http://localhost:{config.monitoring.prometheus_port}/metrics")
            
            # Run until shutdown
            await manager.run()
    
    asyncio.run(main())