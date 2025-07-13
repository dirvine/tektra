#!/usr/bin/env python3
"""
Permission Management System

Granular permission system for agent capabilities with user consent flows
and risk assessment.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru python permissions.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
# ]
# ///

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
import threading

from loguru import logger


class PermissionType(Enum):
    """Types of permissions agents can request."""
    
    # File system permissions
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    FILE_EXECUTE = "file.execute"
    FILE_DELETE = "file.delete"
    DIRECTORY_CREATE = "directory.create"
    DIRECTORY_LIST = "directory.list"
    
    # Network permissions
    NETWORK_REQUEST = "network.request"
    NETWORK_LISTEN = "network.listen"
    NETWORK_PROXY = "network.proxy"
    
    # System permissions
    SYSTEM_COMMAND = "system.command"
    SYSTEM_ENV = "system.environment"
    SYSTEM_PROCESS = "system.process"
    
    # Data permissions
    DATA_PERSONAL = "data.personal"
    DATA_SENSITIVE = "data.sensitive"
    DATA_EXPORT = "data.export"
    
    # External service permissions
    SERVICE_LLM = "service.llm"
    SERVICE_API = "service.api"
    SERVICE_DATABASE = "service.database"
    
    # Agent permissions
    AGENT_SPAWN = "agent.spawn"
    AGENT_COMMUNICATE = "agent.communicate"
    AGENT_MONITOR = "agent.monitor"


class RiskLevel(Enum):
    """Risk levels for permissions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Permission:
    """Represents a specific permission with metadata."""
    
    name: str                              # Permission name (e.g., "file.read")
    description: str                       # Human-readable description
    risk_level: RiskLevel                  # Risk assessment
    requires_user_consent: bool = True     # Whether user consent is required
    auto_grant_conditions: List[str] = field(default_factory=list)  # Conditions for auto-grant
    scope_restrictions: Dict[str, Any] = field(default_factory=dict)  # Scope limitations
    time_limited: bool = False             # Whether permission expires
    duration_seconds: Optional[int] = None # Duration if time-limited
    
    def __post_init__(self):
        """Validate permission configuration."""
        if self.time_limited and self.duration_seconds is None:
            raise ValueError("Time-limited permissions must specify duration")
        
        if self.risk_level == RiskLevel.CRITICAL and not self.requires_user_consent:
            logger.warning(f"Critical permission {self.name} should require user consent")


@dataclass
class PermissionRequest:
    """Represents a permission request from an agent."""
    
    request_id: str
    agent_id: str
    permission_name: str
    justification: str
    requested_scope: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # pending, approved, denied, expired
    expires_at: Optional[float] = None
    approved_by: Optional[str] = None
    denied_reason: Optional[str] = None


class PermissionManager:
    """
    Manages permissions for agent execution.
    
    Handles permission definitions, requests, approvals, and enforcement
    with user consent flows and security policies.
    """
    
    def __init__(self):
        """Initialize permission manager."""
        self._permissions: Dict[str, Permission] = {}
        self._active_grants: Dict[str, Dict[str, Any]] = {}  # agent_id -> {permission: grant_info}
        self._pending_requests: Dict[str, PermissionRequest] = {}
        self._request_history: List[PermissionRequest] = []
        self._consent_handlers: List[Callable] = []
        self._lock = threading.RLock()
        
        # Initialize default permissions
        self._setup_default_permissions()
        
        logger.info("Permission manager initialized")
    
    def _setup_default_permissions(self) -> None:
        """Set up default permission definitions."""
        default_permissions = [
            # File system permissions
            Permission(
                name=PermissionType.FILE_READ.value,
                description="Read files from the file system",
                risk_level=RiskLevel.LOW,
                requires_user_consent=False,
                auto_grant_conditions=["sandboxed_directory"],
                scope_restrictions={"allowed_extensions": [".txt", ".json", ".csv", ".md"]},
            ),
            Permission(
                name=PermissionType.FILE_WRITE.value,
                description="Write or modify files on the file system",
                risk_level=RiskLevel.MEDIUM,
                requires_user_consent=True,
                scope_restrictions={"max_file_size_mb": 100},
            ),
            Permission(
                name=PermissionType.FILE_EXECUTE.value,
                description="Execute files or scripts",
                risk_level=RiskLevel.HIGH,
                requires_user_consent=True,
                scope_restrictions={"allowed_extensions": [".py", ".sh", ".js"]},
            ),
            Permission(
                name=PermissionType.FILE_DELETE.value,
                description="Delete files from the file system",
                risk_level=RiskLevel.HIGH,
                requires_user_consent=True,
            ),
            
            # Network permissions
            Permission(
                name=PermissionType.NETWORK_REQUEST.value,
                description="Make outbound network requests",
                risk_level=RiskLevel.MEDIUM,
                requires_user_consent=True,
                scope_restrictions={"allowed_protocols": ["https"], "rate_limit_per_minute": 60},
            ),
            Permission(
                name=PermissionType.NETWORK_LISTEN.value,
                description="Listen for incoming network connections",
                risk_level=RiskLevel.HIGH,
                requires_user_consent=True,
            ),
            
            # System permissions
            Permission(
                name=PermissionType.SYSTEM_COMMAND.value,
                description="Execute system commands",
                risk_level=RiskLevel.CRITICAL,
                requires_user_consent=True,
                scope_restrictions={"allowed_commands": ["ls", "cat", "grep", "find"]},
            ),
            Permission(
                name=PermissionType.SYSTEM_ENV.value,
                description="Access environment variables",
                risk_level=RiskLevel.MEDIUM,
                requires_user_consent=True,
                scope_restrictions={"blocked_vars": ["PASSWORD", "SECRET", "TOKEN", "KEY"]},
            ),
            
            # Data permissions
            Permission(
                name=PermissionType.DATA_PERSONAL.value,
                description="Access personal or user data",
                risk_level=RiskLevel.HIGH,
                requires_user_consent=True,
            ),
            Permission(
                name=PermissionType.DATA_SENSITIVE.value,
                description="Access sensitive or confidential data",
                risk_level=RiskLevel.CRITICAL,
                requires_user_consent=True,
                time_limited=True,
                duration_seconds=3600,  # 1 hour
            ),
            
            # Service permissions
            Permission(
                name=PermissionType.SERVICE_LLM.value,
                description="Access large language model services",
                risk_level=RiskLevel.LOW,
                requires_user_consent=False,
                auto_grant_conditions=["trusted_agent"],
                scope_restrictions={"max_tokens_per_request": 4000, "rate_limit_per_hour": 100},
            ),
            Permission(
                name=PermissionType.SERVICE_API.value,
                description="Access external API services",
                risk_level=RiskLevel.MEDIUM,
                requires_user_consent=True,
                scope_restrictions={"rate_limit_per_hour": 1000},
            ),
            
            # Agent permissions
            Permission(
                name=PermissionType.AGENT_SPAWN.value,
                description="Create new agent instances",
                risk_level=RiskLevel.HIGH,
                requires_user_consent=True,
                scope_restrictions={"max_concurrent_agents": 5},
            ),
            Permission(
                name=PermissionType.AGENT_COMMUNICATE.value,
                description="Communicate with other agents",
                risk_level=RiskLevel.MEDIUM,
                requires_user_consent=False,
                auto_grant_conditions=["same_security_context"],
            ),
        ]
        
        for permission in default_permissions:
            self._permissions[permission.name] = permission
    
    def register_permission(self, permission: Permission) -> bool:
        """
        Register a new permission type.
        
        Args:
            permission: Permission to register
            
        Returns:
            True if registered successfully, False otherwise
        """
        with self._lock:
            if permission.name in self._permissions:
                logger.warning(f"Permission {permission.name} already registered")
                return False
            
            self._permissions[permission.name] = permission
            logger.info(f"Permission registered: {permission.name}")
            return True
    
    def request_permission(
        self,
        agent_id: str,
        permission_name: str,
        justification: str,
        requested_scope: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Request a permission for an agent.
        
        Args:
            agent_id: ID of the requesting agent
            permission_name: Name of the permission
            justification: Reason for requesting the permission
            requested_scope: Specific scope limitations requested
            
        Returns:
            Request ID for tracking the request
        """
        with self._lock:
            # Check if permission exists
            if permission_name not in self._permissions:
                raise ValueError(f"Unknown permission: {permission_name}")
            
            permission = self._permissions[permission_name]
            request_id = str(uuid.uuid4())
            
            # Create permission request
            request = PermissionRequest(
                request_id=request_id,
                agent_id=agent_id,
                permission_name=permission_name,
                justification=justification,
                requested_scope=requested_scope or {},
            )
            
            # Check if auto-grant is possible
            if self._can_auto_grant(agent_id, permission, request):
                self._grant_permission_internal(agent_id, permission, request)
                request.status = "approved"
                request.approved_by = "auto_grant"
                logger.info(f"Auto-granted permission {permission_name} to agent {agent_id}")
            else:
                # Store pending request
                self._pending_requests[request_id] = request
                
                # Trigger consent flow if needed
                if permission.requires_user_consent:
                    self._trigger_consent_flow(request)
                
                logger.info(f"Permission request created: {request_id} for {permission_name}")
            
            # Add to history
            self._request_history.append(request)
            
            return request_id
    
    def approve_request(
        self,
        request_id: str,
        approved_by: str,
        custom_scope: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Approve a pending permission request.
        
        Args:
            request_id: ID of the request to approve
            approved_by: Identifier of who approved the request
            custom_scope: Custom scope restrictions for this grant
            
        Returns:
            True if approved successfully, False otherwise
        """
        with self._lock:
            if request_id not in self._pending_requests:
                logger.warning(f"Request {request_id} not found or already processed")
                return False
            
            request = self._pending_requests[request_id]
            permission = self._permissions[request.permission_name]
            
            # Apply custom scope if provided
            if custom_scope:
                request.requested_scope.update(custom_scope)
            
            # Grant the permission
            if self._grant_permission_internal(request.agent_id, permission, request):
                request.status = "approved"
                request.approved_by = approved_by
                
                # Remove from pending
                del self._pending_requests[request_id]
                
                logger.info(f"Permission request approved: {request_id} by {approved_by}")
                return True
            else:
                logger.error(f"Failed to grant permission for request {request_id}")
                return False
    
    def deny_request(self, request_id: str, reason: str) -> bool:
        """
        Deny a pending permission request.
        
        Args:
            request_id: ID of the request to deny
            reason: Reason for denial
            
        Returns:
            True if denied successfully, False otherwise
        """
        with self._lock:
            if request_id not in self._pending_requests:
                logger.warning(f"Request {request_id} not found or already processed")
                return False
            
            request = self._pending_requests[request_id]
            request.status = "denied"
            request.denied_reason = reason
            
            # Remove from pending
            del self._pending_requests[request_id]
            
            logger.info(f"Permission request denied: {request_id} - {reason}")
            return True
    
    def has_permission(self, agent_id: str, permission_name: str) -> bool:
        """
        Check if an agent has a specific permission.
        
        Args:
            agent_id: ID of the agent
            permission_name: Name of the permission to check
            
        Returns:
            True if agent has the permission, False otherwise
        """
        with self._lock:
            agent_grants = self._active_grants.get(agent_id, {})
            
            if permission_name not in agent_grants:
                return False
            
            grant_info = agent_grants[permission_name]
            
            # Check if time-limited permission has expired
            if grant_info.get("expires_at") and time.time() > grant_info["expires_at"]:
                self._revoke_permission_internal(agent_id, permission_name)
                return False
            
            return True
    
    def check_permission_scope(
        self,
        agent_id: str,
        permission_name: str,
        requested_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if a specific action is allowed under the granted permission scope.
        
        Args:
            agent_id: ID of the agent
            permission_name: Name of the permission
            requested_action: Details of the action being requested
            
        Returns:
            Dictionary with 'allowed' boolean and optional 'reason' and 'modifications'
        """
        with self._lock:
            if not self.has_permission(agent_id, permission_name):
                return {"allowed": False, "reason": "Permission not granted"}
            
            agent_grants = self._active_grants.get(agent_id, {})
            grant_info = agent_grants[permission_name]
            scope = grant_info.get("scope", {})
            
            # Perform scope checking based on permission type
            result = self._check_scope_constraints(permission_name, scope, requested_action)
            
            return result
    
    def revoke_permission(self, agent_id: str, permission_name: str) -> bool:
        """
        Revoke a permission from an agent.
        
        Args:
            agent_id: ID of the agent
            permission_name: Name of the permission to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        with self._lock:
            return self._revoke_permission_internal(agent_id, permission_name)
    
    def get_agent_permissions(self, agent_id: str) -> Dict[str, Any]:
        """
        Get all permissions for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with permission information
        """
        with self._lock:
            agent_grants = self._active_grants.get(agent_id, {})
            
            permissions = {}
            for perm_name, grant_info in agent_grants.items():
                permissions[perm_name] = {
                    "granted_at": grant_info["granted_at"],
                    "expires_at": grant_info.get("expires_at"),
                    "scope": grant_info.get("scope", {}),
                    "granted_by": grant_info.get("granted_by", "system"),
                }
            
            return permissions
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """
        Get all pending permission requests.
        
        Returns:
            List of pending request information
        """
        with self._lock:
            pending = []
            for request in self._pending_requests.values():
                permission = self._permissions.get(request.permission_name)
                pending.append({
                    "request_id": request.request_id,
                    "agent_id": request.agent_id,
                    "permission_name": request.permission_name,
                    "justification": request.justification,
                    "timestamp": request.timestamp,
                    "risk_level": permission.risk_level.value if permission else "unknown",
                    "description": permission.description if permission else "Unknown permission",
                    "requested_scope": request.requested_scope,
                })
            
            return pending
    
    def add_consent_handler(self, handler: Callable[[PermissionRequest], None]) -> None:
        """
        Add a consent handler for user permission requests.
        
        Args:
            handler: Function to handle consent requests
        """
        self._consent_handlers.append(handler)
        logger.info("Consent handler added")
    
    def _can_auto_grant(
        self,
        agent_id: str,
        permission: Permission,
        request: PermissionRequest
    ) -> bool:
        """Check if a permission can be auto-granted."""
        if permission.requires_user_consent:
            return False
        
        # Check auto-grant conditions
        for condition in permission.auto_grant_conditions:
            if not self._check_auto_grant_condition(agent_id, condition, request):
                return False
        
        return True
    
    def _check_auto_grant_condition(
        self,
        agent_id: str,
        condition: str,
        request: PermissionRequest
    ) -> bool:
        """Check if an auto-grant condition is met."""
        # Simple condition checking - can be extended
        conditions = {
            "sandboxed_directory": True,  # Assume all access is sandboxed
            "trusted_agent": agent_id.startswith("trusted_"),
            "same_security_context": True,  # For now, allow same context communication
        }
        
        return conditions.get(condition, False)
    
    def _grant_permission_internal(
        self,
        agent_id: str,
        permission: Permission,
        request: PermissionRequest
    ) -> bool:
        """Internal method to grant a permission."""
        try:
            if agent_id not in self._active_grants:
                self._active_grants[agent_id] = {}
            
            # Merge scope restrictions with requested scope
            scope = permission.scope_restrictions.copy()
            scope.update(request.requested_scope)
            
            grant_info = {
                "granted_at": time.time(),
                "scope": scope,
                "granted_by": request.approved_by or "system",
                "request_id": request.request_id,
            }
            
            # Add expiration if time-limited
            if permission.time_limited and permission.duration_seconds:
                grant_info["expires_at"] = time.time() + permission.duration_seconds
            
            self._active_grants[agent_id][permission.name] = grant_info
            
            logger.info(f"Permission granted: {permission.name} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grant permission {permission.name} to agent {agent_id}: {e}")
            return False
    
    def _revoke_permission_internal(self, agent_id: str, permission_name: str) -> bool:
        """Internal method to revoke a permission."""
        try:
            agent_grants = self._active_grants.get(agent_id, {})
            
            if permission_name in agent_grants:
                del agent_grants[permission_name]
                logger.info(f"Permission revoked: {permission_name} from agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke permission {permission_name} from agent {agent_id}: {e}")
            return False
    
    def _trigger_consent_flow(self, request: PermissionRequest) -> None:
        """Trigger user consent flow for a permission request."""
        for handler in self._consent_handlers:
            try:
                handler(request)
            except Exception as e:
                logger.error(f"Error in consent handler: {e}")
    
    def _check_scope_constraints(
        self,
        permission_name: str,
        scope: Dict[str, Any],
        requested_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if an action complies with permission scope constraints."""
        # File permission scope checking
        if permission_name.startswith("file."):
            return self._check_file_scope(scope, requested_action)
        
        # Network permission scope checking
        elif permission_name.startswith("network."):
            return self._check_network_scope(scope, requested_action)
        
        # System permission scope checking
        elif permission_name.startswith("system."):
            return self._check_system_scope(scope, requested_action)
        
        # Default: allow if no specific constraints
        return {"allowed": True}
    
    def _check_file_scope(self, scope: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Check file operation scope constraints."""
        file_path = action.get("file_path", "")
        file_size = action.get("file_size", 0)
        
        # Check allowed extensions
        allowed_extensions = scope.get("allowed_extensions", [])
        if allowed_extensions:
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in allowed_extensions:
                return {
                    "allowed": False,
                    "reason": f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}"
                }
        
        # Check file size limits
        max_size_mb = scope.get("max_file_size_mb", float('inf'))
        if file_size > max_size_mb * 1024 * 1024:
            return {
                "allowed": False,
                "reason": f"File size {file_size} exceeds limit of {max_size_mb}MB"
            }
        
        return {"allowed": True}
    
    def _check_network_scope(self, scope: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Check network operation scope constraints."""
        url = action.get("url", "")
        protocol = action.get("protocol", "")
        
        # Check allowed protocols
        allowed_protocols = scope.get("allowed_protocols", [])
        if allowed_protocols and protocol not in allowed_protocols:
            return {
                "allowed": False,
                "reason": f"Protocol {protocol} not allowed. Allowed: {allowed_protocols}"
            }
        
        # Rate limiting would be checked here in a real implementation
        
        return {"allowed": True}
    
    def _check_system_scope(self, scope: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Check system operation scope constraints."""
        command = action.get("command", "")
        
        # Check allowed commands
        allowed_commands = scope.get("allowed_commands", [])
        if allowed_commands:
            base_command = command.split()[0] if command else ""
            if base_command not in allowed_commands:
                return {
                    "allowed": False,
                    "reason": f"Command {base_command} not allowed. Allowed: {allowed_commands}"
                }
        
        return {"allowed": True}


def create_permission_manager() -> PermissionManager:
    """
    Create a permission manager instance.
    
    Returns:
        PermissionManager instance
    """
    return PermissionManager()


if __name__ == "__main__":
    def demo_permission_system():
        """Demonstrate permission system functionality."""
        print("🔐 Permission System Demo")
        print("=" * 40)
        
        # Create permission manager
        perm_manager = create_permission_manager()
        
        # Example consent handler
        def mock_consent_handler(request: PermissionRequest):
            print(f"📝 Consent request: {request.permission_name} for agent {request.agent_id}")
            print(f"   Justification: {request.justification}")
        
        perm_manager.add_consent_handler(mock_consent_handler)
        
        # Test permission requests
        agent_id = "demo_agent"
        
        # Request file read permission (should auto-grant)
        request_id1 = perm_manager.request_permission(
            agent_id=agent_id,
            permission_name="file.read",
            justification="Need to read configuration files"
        )
        print(f"File read request: {request_id1}")
        print(f"Has file.read permission: {perm_manager.has_permission(agent_id, 'file.read')}")
        
        # Request file write permission (requires consent)
        request_id2 = perm_manager.request_permission(
            agent_id=agent_id,
            permission_name="file.write",
            justification="Need to save analysis results"
        )
        print(f"File write request: {request_id2}")
        
        # Check pending requests
        pending = perm_manager.get_pending_requests()
        print(f"Pending requests: {len(pending)}")
        for req in pending:
            print(f"  - {req['permission_name']} (Risk: {req['risk_level']})")
        
        # Approve the file write request
        if pending:
            perm_manager.approve_request(pending[0]["request_id"], "user")
            print(f"Approved file write permission")
        
        print(f"Has file.write permission: {perm_manager.has_permission(agent_id, 'file.write')}")
        
        # Test scope checking
        scope_result = perm_manager.check_permission_scope(
            agent_id=agent_id,
            permission_name="file.read",
            requested_action={
                "file_path": "/path/to/file.txt",
                "file_size": 1024
            }
        )
        print(f"File read scope check: {scope_result}")
        
        # Get all agent permissions
        agent_perms = perm_manager.get_agent_permissions(agent_id)
        print(f"Agent permissions: {list(agent_perms.keys())}")
        
        print("\n🔐 Permission System Demo Complete")
    
    # Run demo
    demo_permission_system()