#!/usr/bin/env python3
"""
Security Context Management

Provides security context for agent execution with resource limits,
permission tracking, and threat monitoring.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,loguru python context.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
# ]
# ///

import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import threading
import resource

import psutil
from loguru import logger


class SecurityLevel(Enum):
    """Security levels for agent execution."""
    MINIMAL = "minimal"        # Basic sandbox, limited permissions
    STANDARD = "standard"      # Default level for most agents
    STRICT = "strict"         # High security, extensive monitoring
    PARANOID = "paranoid"     # Maximum security, minimal permissions


@dataclass
class ResourceLimits:
    """Resource limits for agent execution."""
    max_memory_mb: int = 512              # Maximum memory usage in MB
    max_cpu_percent: float = 50.0         # Maximum CPU usage percentage
    max_execution_time_seconds: int = 300  # Maximum execution time
    max_file_size_mb: int = 100           # Maximum file size for operations
    max_network_connections: int = 10      # Maximum simultaneous connections
    max_disk_usage_mb: int = 1024         # Maximum disk usage in MB
    max_processes: int = 5                # Maximum number of processes


@dataclass
class SecurityMetrics:
    """Security metrics and monitoring data."""
    threat_events: int = 0
    permission_denials: int = 0
    resource_violations: int = 0
    sandbox_escapes: int = 0
    suspicious_activities: int = 0
    last_threat_time: Optional[float] = None
    risk_score: float = 0.0  # 0.0 = low risk, 1.0 = high risk


class SecurityContext:
    """
    Security context for agent execution.
    
    Manages permissions, resource limits, monitoring, and audit trails
    for safe agent execution in sandboxed environments.
    """
    
    def __init__(
        self,
        agent_id: str,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        custom_limits: Optional[ResourceLimits] = None,
        allowed_domains: Optional[Set[str]] = None,
        blocked_domains: Optional[Set[str]] = None
    ):
        """
        Initialize security context.
        
        Args:
            agent_id: Unique identifier for the agent
            security_level: Security level for execution
            custom_limits: Custom resource limits
            allowed_domains: Allowed network domains
            blocked_domains: Blocked network domains
        """
        self.agent_id = agent_id
        self.context_id = str(uuid.uuid4())
        self.security_level = security_level
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Resource limits based on security level
        self.limits = custom_limits or self._get_default_limits(security_level)
        
        # Network access control
        self.allowed_domains = allowed_domains or set()
        self.blocked_domains = blocked_domains or set()
        
        # Permissions and capabilities
        self.granted_permissions: Set[str] = set()
        self.denied_permissions: Set[str] = set()
        self.pending_permissions: Set[str] = set()
        
        # Runtime tracking
        self.active = False
        self.start_time: Optional[float] = None
        self.process_ids: Set[int] = set()
        self.open_files: Set[str] = set()
        self.network_connections: List[Dict[str, Any]] = []
        
        # Security monitoring
        self.metrics = SecurityMetrics()
        self.audit_trail: List[Dict[str, Any]] = []
        self.threat_indicators: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Security context created: {self.context_id} for agent {agent_id}")
    
    def _get_default_limits(self, level: SecurityLevel) -> ResourceLimits:
        """Get default resource limits based on security level."""
        limits_by_level = {
            SecurityLevel.MINIMAL: ResourceLimits(
                max_memory_mb=256,
                max_cpu_percent=25.0,
                max_execution_time_seconds=60,
                max_file_size_mb=10,
                max_network_connections=3,
                max_disk_usage_mb=100,
                max_processes=2
            ),
            SecurityLevel.STANDARD: ResourceLimits(
                max_memory_mb=512,
                max_cpu_percent=50.0,
                max_execution_time_seconds=300,
                max_file_size_mb=100,
                max_network_connections=10,
                max_disk_usage_mb=1024,
                max_processes=5
            ),
            SecurityLevel.STRICT: ResourceLimits(
                max_memory_mb=256,
                max_cpu_percent=30.0,
                max_execution_time_seconds=120,
                max_file_size_mb=50,
                max_network_connections=5,
                max_disk_usage_mb=512,
                max_processes=3
            ),
            SecurityLevel.PARANOID: ResourceLimits(
                max_memory_mb=128,
                max_cpu_percent=20.0,
                max_execution_time_seconds=60,
                max_file_size_mb=10,
                max_network_connections=2,
                max_disk_usage_mb=256,
                max_processes=1
            )
        }
        return limits_by_level[level]
    
    def activate(self) -> bool:
        """
        Activate the security context for execution.
        
        Returns:
            True if activation successful, False otherwise
        """
        with self._lock:
            if self.active:
                logger.warning(f"Security context {self.context_id} already active")
                return False
            
            try:
                # Set resource limits
                self._apply_resource_limits()
                
                # Initialize monitoring
                self.start_time = time.time()
                self.active = True
                self.last_activity = time.time()
                
                # Log activation
                self._add_audit_event("context_activated", {
                    "security_level": self.security_level.value,
                    "limits": self.limits.__dict__
                })
                
                logger.info(f"Security context {self.context_id} activated")
                return True
                
            except Exception as e:
                logger.error(f"Failed to activate security context {self.context_id}: {e}")
                return False
    
    def deactivate(self) -> bool:
        """
        Deactivate the security context.
        
        Returns:
            True if deactivation successful, False otherwise
        """
        with self._lock:
            if not self.active:
                logger.warning(f"Security context {self.context_id} not active")
                return False
            
            try:
                # Clean up resources
                self._cleanup_resources()
                
                # Finalize metrics
                execution_time = time.time() - (self.start_time or 0)
                
                # Log deactivation
                self._add_audit_event("context_deactivated", {
                    "execution_time": execution_time,
                    "metrics": self.metrics.__dict__
                })
                
                self.active = False
                logger.info(f"Security context {self.context_id} deactivated")
                return True
                
            except Exception as e:
                logger.error(f"Failed to deactivate security context {self.context_id}: {e}")
                return False
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """
        Check current resource usage against limits.
        
        Returns:
            Dictionary with resource usage and violations
        """
        with self._lock:
            if not self.active:
                return {"error": "Context not active"}
            
            try:
                process = psutil.Process()
                
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # CPU usage
                cpu_percent = process.cpu_percent()
                
                # Execution time
                execution_time = time.time() - (self.start_time or 0)
                
                # File descriptors
                try:
                    open_files = len(process.open_files())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    open_files = 0
                
                # Network connections
                try:
                    connections = len(process.connections())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    connections = 0
                
                # Check violations
                violations = []
                
                if memory_mb > self.limits.max_memory_mb:
                    violations.append(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.limits.max_memory_mb}MB")
                    self.metrics.resource_violations += 1
                
                if cpu_percent > self.limits.max_cpu_percent:
                    violations.append(f"CPU limit exceeded: {cpu_percent:.1f}% > {self.limits.max_cpu_percent}%")
                    self.metrics.resource_violations += 1
                
                if execution_time > self.limits.max_execution_time_seconds:
                    violations.append(f"Time limit exceeded: {execution_time:.1f}s > {self.limits.max_execution_time_seconds}s")
                    self.metrics.resource_violations += 1
                
                if connections > self.limits.max_network_connections:
                    violations.append(f"Connection limit exceeded: {connections} > {self.limits.max_network_connections}")
                    self.metrics.resource_violations += 1
                
                # Update activity timestamp
                self.last_activity = time.time()
                
                usage = {
                    "memory_mb": memory_mb,
                    "memory_limit_mb": self.limits.max_memory_mb,
                    "cpu_percent": cpu_percent,
                    "cpu_limit_percent": self.limits.max_cpu_percent,
                    "execution_time": execution_time,
                    "time_limit": self.limits.max_execution_time_seconds,
                    "open_files": open_files,
                    "network_connections": connections,
                    "connection_limit": self.limits.max_network_connections,
                    "violations": violations,
                    "within_limits": len(violations) == 0
                }
                
                # Log violations
                if violations:
                    self._add_audit_event("resource_violation", {
                        "violations": violations,
                        "usage": usage
                    })
                
                return usage
                
            except Exception as e:
                logger.error(f"Error checking resource usage: {e}")
                return {"error": str(e)}
    
    def grant_permission(self, permission: str, reason: str = "") -> bool:
        """
        Grant a permission to the agent.
        
        Args:
            permission: Permission to grant
            reason: Reason for granting permission
            
        Returns:
            True if permission granted, False otherwise
        """
        with self._lock:
            if permission in self.granted_permissions:
                return True
            
            # Remove from pending and denied
            self.pending_permissions.discard(permission)
            self.denied_permissions.discard(permission)
            
            # Grant permission
            self.granted_permissions.add(permission)
            
            self._add_audit_event("permission_granted", {
                "permission": permission,
                "reason": reason
            })
            
            logger.info(f"Permission granted: {permission} for context {self.context_id}")
            return True
    
    def deny_permission(self, permission: str, reason: str = "") -> bool:
        """
        Deny a permission to the agent.
        
        Args:
            permission: Permission to deny
            reason: Reason for denying permission
            
        Returns:
            True if permission denied, False otherwise
        """
        with self._lock:
            if permission in self.denied_permissions:
                return True
            
            # Remove from pending and granted
            self.pending_permissions.discard(permission)
            self.granted_permissions.discard(permission)
            
            # Deny permission
            self.denied_permissions.add(permission)
            self.metrics.permission_denials += 1
            
            self._add_audit_event("permission_denied", {
                "permission": permission,
                "reason": reason
            })
            
            logger.warning(f"Permission denied: {permission} for context {self.context_id}")
            return True
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if agent has a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if permission granted, False otherwise
        """
        with self._lock:
            return permission in self.granted_permissions
    
    def add_threat_indicator(self, indicator_type: str, details: Dict[str, Any]) -> None:
        """
        Add a threat indicator to the context.
        
        Args:
            indicator_type: Type of threat indicator
            details: Details about the threat
        """
        with self._lock:
            indicator = {
                "type": indicator_type,
                "timestamp": time.time(),
                "details": details,
                "context_id": self.context_id,
                "agent_id": self.agent_id
            }
            
            self.threat_indicators.append(indicator)
            self.metrics.threat_events += 1
            self.metrics.last_threat_time = time.time()
            
            # Update risk score
            self._update_risk_score()
            
            self._add_audit_event("threat_detected", indicator)
            
            logger.warning(f"Threat indicator added: {indicator_type} for context {self.context_id}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get comprehensive security status.
        
        Returns:
            Dictionary with security status information
        """
        with self._lock:
            status = {
                "context_id": self.context_id,
                "agent_id": self.agent_id,
                "security_level": self.security_level.value,
                "active": self.active,
                "created_at": self.created_at,
                "last_activity": self.last_activity,
                "execution_time": (time.time() - self.start_time) if self.start_time else 0,
                "permissions": {
                    "granted": list(self.granted_permissions),
                    "denied": list(self.denied_permissions),
                    "pending": list(self.pending_permissions)
                },
                "limits": self.limits.__dict__,
                "metrics": self.metrics.__dict__,
                "threat_indicators": len(self.threat_indicators),
                "audit_events": len(self.audit_trail)
            }
            
            # Add current resource usage if active
            if self.active:
                status["resource_usage"] = self.check_resource_usage()
            
            return status
    
    def _apply_resource_limits(self) -> None:
        """Apply resource limits to the current process."""
        try:
            # Memory limit (virtual memory)
            memory_bytes = self.limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # CPU time limit
            cpu_time = self.limits.max_execution_time_seconds
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time, cpu_time))
            
            # File size limit
            file_size_bytes = self.limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
            
            # Number of open files
            max_files = min(1024, self.limits.max_network_connections * 10)
            resource.setrlimit(resource.RLIMIT_NOFILE, (max_files, max_files))
            
            logger.debug(f"Resource limits applied for context {self.context_id}")
            
        except Exception as e:
            logger.warning(f"Could not apply all resource limits: {e}")
    
    def _cleanup_resources(self) -> None:
        """Clean up resources used by the context."""
        try:
            # Close any tracked files
            for file_path in list(self.open_files):
                try:
                    # This would close files if we had handles
                    pass
                except Exception:
                    pass
            
            # Clear tracking
            self.open_files.clear()
            self.network_connections.clear()
            self.process_ids.clear()
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
    
    def _update_risk_score(self) -> None:
        """Update the risk score based on current metrics."""
        # Simple risk scoring algorithm
        base_score = 0.0
        
        # Factor in threat events
        if self.metrics.threat_events > 0:
            base_score += min(0.3, self.metrics.threat_events * 0.1)
        
        # Factor in permission denials
        if self.metrics.permission_denials > 0:
            base_score += min(0.2, self.metrics.permission_denials * 0.05)
        
        # Factor in resource violations
        if self.metrics.resource_violations > 0:
            base_score += min(0.3, self.metrics.resource_violations * 0.1)
        
        # Factor in suspicious activities
        if self.metrics.suspicious_activities > 0:
            base_score += min(0.2, self.metrics.suspicious_activities * 0.1)
        
        # Time decay factor
        if self.metrics.last_threat_time:
            time_since_threat = time.time() - self.metrics.last_threat_time
            decay_factor = max(0.5, 1.0 - (time_since_threat / 3600))  # Decay over 1 hour
            base_score *= decay_factor
        
        self.metrics.risk_score = min(1.0, base_score)
    
    def _add_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Add an event to the audit trail."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "context_id": self.context_id,
            "agent_id": self.agent_id,
            "details": details
        }
        
        self.audit_trail.append(event)
        
        # Limit audit trail size
        if len(self.audit_trail) > 1000:
            self.audit_trail = self.audit_trail[-500:]  # Keep last 500 events


def create_security_context(
    agent_id: str,
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    **kwargs
) -> SecurityContext:
    """
    Create a security context for agent execution.
    
    Args:
        agent_id: Unique identifier for the agent
        security_level: Security level for execution
        **kwargs: Additional configuration options
        
    Returns:
        SecurityContext instance
    """
    return SecurityContext(
        agent_id=agent_id,
        security_level=security_level,
        **kwargs
    )


if __name__ == "__main__":
    import asyncio
    
    async def demo_security_context():
        """Demonstrate security context functionality."""
        print("🔐 Security Context Demo")
        print("=" * 40)
        
        # Create security context
        context = create_security_context(
            agent_id="demo_agent",
            security_level=SecurityLevel.STANDARD
        )
        
        print(f"Created context: {context.context_id}")
        print(f"Security level: {context.security_level.value}")
        
        # Activate context
        if context.activate():
            print("✅ Context activated successfully")
            
            # Grant some permissions
            context.grant_permission("file.read", "Demo permission")
            context.grant_permission("network.request", "API access")
            context.deny_permission("file.write", "Security policy")
            
            # Check permissions
            print(f"Has file.read permission: {context.has_permission('file.read')}")
            print(f"Has file.write permission: {context.has_permission('file.write')}")
            
            # Check resource usage
            usage = context.check_resource_usage()
            if "error" not in usage:
                print(f"Memory usage: {usage['memory_mb']:.1f}MB (limit: {usage['memory_limit_mb']}MB)")
                print(f"CPU usage: {usage['cpu_percent']:.1f}% (limit: {usage['cpu_limit_percent']}%)")
                print(f"Within limits: {usage['within_limits']}")
            
            # Add threat indicator
            context.add_threat_indicator("suspicious_activity", {
                "description": "Demo threat indicator",
                "severity": "low"
            })
            
            # Get security status
            status = context.get_security_status()
            print(f"Risk score: {status['metrics']['risk_score']:.2f}")
            print(f"Threat indicators: {status['threat_indicators']}")
            print(f"Audit events: {status['audit_events']}")
            
            # Deactivate context
            if context.deactivate():
                print("✅ Context deactivated successfully")
        
        print("\n🔐 Security Context Demo Complete")
    
    # Run demo
    asyncio.run(demo_security_context())