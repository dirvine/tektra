#!/usr/bin/env python3
"""
Permission and Consent Framework

Comprehensive system for granular permissions, dynamic user consent,
and policy-based access control for agent operations.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru,pydantic,cryptography python consent_framework.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
#     "pydantic>=2.0.0",
#     "cryptography>=41.0.0",
#     "typing-extensions>=4.7.0",
# ]
# ///

import asyncio
import time
import uuid
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, AsyncContextManager
from pathlib import Path
import threading

from loguru import logger
from pydantic import BaseModel, Field, validator
from cryptography.fernet import Fernet

from .context import SecurityContext, SecurityLevel
from .permissions import PermissionManager


class PermissionScope(Enum):
    """Permission scope levels."""
    GLOBAL = "global"                # System-wide permissions
    AGENT = "agent"                  # Agent-specific permissions
    SESSION = "session"              # Session-specific permissions
    TASK = "task"                    # Single task permissions
    TEMPORARY = "temporary"          # Time-limited permissions


class ConsentAction(Enum):
    """Types of consent actions."""
    GRANT = "grant"                  # Grant permission
    DENY = "deny"                    # Deny permission
    DEFER = "defer"                  # Defer decision
    REVOKE = "revoke"                # Revoke existing permission
    DELEGATE = "delegate"            # Delegate to another entity


class ConsentMode(Enum):
    """Consent interaction modes."""
    INTERACTIVE = "interactive"      # Prompt user for consent
    AUTOMATIC = "automatic"          # Use predefined policies
    SILENT = "silent"                # Use defaults, no prompting
    STRICT = "strict"                # Require explicit consent for everything


class PermissionCategory(Enum):
    """High-level permission categories."""
    SYSTEM = "system"                # System-level operations
    NETWORK = "network"              # Network access
    FILESYSTEM = "filesystem"        # File system access
    PROCESS = "process"              # Process management
    DATA = "data"                    # Data access and manipulation
    COMPUTE = "compute"              # Computational resources
    IDENTITY = "identity"            # Identity and authentication
    COMMUNICATION = "communication"  # Inter-agent communication


@dataclass
class PermissionDetail:
    """Detailed permission specification."""
    
    category: PermissionCategory
    subcategory: str
    action: str
    resource: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert to string representation."""
        parts = [self.category.value, self.subcategory, self.action]
        if self.resource:
            parts.append(self.resource)
        return ".".join(parts)
    
    @classmethod
    def from_string(cls, permission_str: str) -> "PermissionDetail":
        """Parse from string representation."""
        parts = permission_str.split(".")
        if len(parts) < 3:
            raise ValueError(f"Invalid permission string: {permission_str}")
        
        category = PermissionCategory(parts[0])
        subcategory = parts[1]
        action = parts[2]
        resource = parts[3] if len(parts) > 3 else None
        
        return cls(
            category=category,
            subcategory=subcategory,
            action=action,
            resource=resource
        )


class ConsentRequest(BaseModel):
    """Request for user consent."""
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    permission: PermissionDetail
    scope: PermissionScope
    
    # Context information
    justification: str
    risk_level: str  # "low", "medium", "high", "critical"
    duration: Optional[timedelta] = None
    resource_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    
    # UI preferences
    ui_title: Optional[str] = None
    ui_description: Optional[str] = None
    ui_icon: Optional[str] = None
    ui_urgency: str = "normal"  # "low", "normal", "high", "critical"
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('duration')
    def validate_duration(cls, v):
        if v and v.total_seconds() <= 0:
            raise ValueError("Duration must be positive")
        return v


class ConsentResponse(BaseModel):
    """Response to consent request."""
    
    request_id: str
    action: ConsentAction
    granted_permissions: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing information
    expires_at: Optional[datetime] = None
    granted_at: datetime = Field(default_factory=datetime.now)
    
    # Delegation information
    delegated_to: Optional[str] = None
    delegation_constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # User information
    user_id: Optional[str] = None
    reason: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class PermissionPolicy(BaseModel):
    """Policy for automatic permission decisions."""
    
    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    
    # Matching criteria
    agent_patterns: List[str] = Field(default_factory=list)
    permission_patterns: List[str] = Field(default_factory=list)
    scope_filter: Optional[PermissionScope] = None
    risk_level_max: str = "medium"
    
    # Policy decision
    action: ConsentAction
    auto_grant_permissions: List[str] = Field(default_factory=list)
    auto_constraints: Dict[str, Any] = Field(default_factory=dict)
    auto_duration: Optional[timedelta] = None
    
    # Conditions
    conditions: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 100  # Higher numbers = higher priority
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    enabled: bool = True


@dataclass
class PermissionGrant:
    """Active permission grant."""
    
    grant_id: str
    agent_id: str
    permission: PermissionDetail
    scope: PermissionScope
    
    # Timing
    granted_at: datetime
    expires_at: Optional[datetime] = None
    
    # Constraints and limitations
    constraints: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    usage_limit: Optional[int] = None
    
    # Source information
    granted_by: Optional[str] = None
    source_request_id: Optional[str] = None
    policy_id: Optional[str] = None
    
    # Delegation chain
    delegated_from: Optional[str] = None
    can_delegate: bool = False
    
    def is_expired(self) -> bool:
        """Check if the grant has expired."""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        if self.usage_limit and self.usage_count >= self.usage_limit:
            return True
        return False
    
    def can_use(self) -> bool:
        """Check if the grant can be used."""
        return not self.is_expired()
    
    def record_usage(self) -> bool:
        """Record usage of the permission."""
        if not self.can_use():
            return False
        
        self.usage_count += 1
        return True


class ConsentUI:
    """User interface for consent management."""
    
    def __init__(self):
        self.pending_requests: Dict[str, ConsentRequest] = {}
        self.ui_handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()
    
    def register_ui_handler(
        self,
        handler_type: str,
        handler: Callable[[ConsentRequest], ConsentResponse]
    ) -> None:
        """Register a UI handler for consent requests."""
        with self._lock:
            self.ui_handlers[handler_type] = handler
            logger.debug(f"Registered UI handler: {handler_type}")
    
    async def request_consent(
        self,
        request: ConsentRequest,
        timeout: float = 300.0  # 5 minutes default
    ) -> ConsentResponse:
        """Request user consent through appropriate UI."""
        with self._lock:
            self.pending_requests[request.request_id] = request
        
        try:
            # Determine UI handler based on urgency and available handlers
            handler_type = self._select_ui_handler(request)
            
            if handler_type in self.ui_handlers:
                logger.info(f"Requesting consent via {handler_type} for {request.permission.to_string()}")
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.ui_handlers[handler_type], request),
                    timeout=timeout
                )
            else:
                # Default to automatic denial if no UI available
                logger.warning(f"No UI handler available for consent request {request.request_id}")
                response = ConsentResponse(
                    request_id=request.request_id,
                    action=ConsentAction.DENY,
                    reason="No UI handler available"
                )
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Consent request {request.request_id} timed out")
            return ConsentResponse(
                request_id=request.request_id,
                action=ConsentAction.DENY,
                reason="Request timed out"
            )
        
        finally:
            with self._lock:
                self.pending_requests.pop(request.request_id, None)
    
    def _select_ui_handler(self, request: ConsentRequest) -> str:
        """Select appropriate UI handler for request."""
        if request.ui_urgency == "critical":
            return "modal_dialog"
        elif request.ui_urgency == "high":
            return "notification_popup"
        else:
            return "background_notification"
    
    def get_pending_requests(self) -> List[ConsentRequest]:
        """Get all pending consent requests."""
        with self._lock:
            return list(self.pending_requests.values())
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending consent request."""
        with self._lock:
            return self.pending_requests.pop(request_id, None) is not None


class ConsentFramework:
    """
    Comprehensive permission and consent management framework.
    
    Provides granular permission control, dynamic user consent,
    policy-based automation, and delegation capabilities.
    """
    
    def __init__(
        self,
        permission_manager: Optional[PermissionManager] = None,
        encryption_key: Optional[bytes] = None
    ):
        """Initialize the consent framework."""
        self.permission_manager = permission_manager
        
        # Encryption for sensitive data
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # State management
        self.active_grants: Dict[str, PermissionGrant] = {}
        self.consent_history: List[ConsentResponse] = []
        self.policies: Dict[str, PermissionPolicy] = {}
        
        # Configuration
        self.consent_mode = ConsentMode.INTERACTIVE
        self.default_consent_timeout = 300.0  # 5 minutes
        self.max_grant_duration = timedelta(hours=24)
        
        # UI and handlers
        self.consent_ui = ConsentUI()
        self.consent_handlers: Dict[str, Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Default policies
        self._initialize_default_policies()
        
        logger.info("Consent framework initialized")
    
    def _initialize_default_policies(self) -> None:
        """Initialize default permission policies."""
        # Safe operations policy
        safe_policy = PermissionPolicy(
            name="Safe Operations",
            description="Auto-approve low-risk operations",
            permission_patterns=[
                "data.read.*",
                "compute.cpu.basic",
                "system.info.read"
            ],
            risk_level_max="low",
            action=ConsentAction.GRANT,
            auto_duration=timedelta(hours=1),
            priority=50
        )
        
        # Critical operations policy
        critical_policy = PermissionPolicy(
            name="Critical Operations Block",
            description="Block critical operations by default",
            permission_patterns=[
                "system.admin.*",
                "filesystem.delete.*",
                "network.server.*",
                "process.execute.system"
            ],
            risk_level_max="critical",
            action=ConsentAction.DENY,
            priority=200
        )
        
        self.policies[safe_policy.policy_id] = safe_policy
        self.policies[critical_policy.policy_id] = critical_policy
    
    async def request_permission(
        self,
        agent_id: str,
        permission: Union[str, PermissionDetail],
        scope: PermissionScope = PermissionScope.TASK,
        justification: str = "",
        context: Optional[SecurityContext] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Request permission with consent handling.
        
        Args:
            agent_id: ID of the requesting agent
            permission: Permission being requested
            scope: Scope of the permission
            justification: Reason for the request
            context: Security context
            **kwargs: Additional request parameters
            
        Returns:
            Tuple of (granted, grant_id)
        """
        # Parse permission if string
        if isinstance(permission, str):
            try:
                permission = PermissionDetail.from_string(permission)
            except ValueError as e:
                logger.error(f"Invalid permission string: {e}")
                return False, None
        
        # Check existing grants first
        grant_id = self._find_existing_grant(agent_id, permission, scope)
        if grant_id:
            grant = self.active_grants[grant_id]
            if grant.can_use():
                grant.record_usage()
                logger.debug(f"Using existing grant {grant_id} for {permission.to_string()}")
                return True, grant_id
            else:
                # Remove expired grant
                self._revoke_grant(grant_id)
        
        # Check policies for automatic decision
        policy_decision = await self._evaluate_policies(agent_id, permission, scope, context)
        if policy_decision:
            action, auto_grant_id = policy_decision
            if action == ConsentAction.GRANT and auto_grant_id:
                logger.info(f"Policy auto-granted permission {permission.to_string()} to {agent_id}")
                return True, auto_grant_id
            elif action == ConsentAction.DENY:
                logger.info(f"Policy auto-denied permission {permission.to_string()} to {agent_id}")
                return False, None
        
        # If not in interactive mode, deny by default
        if self.consent_mode != ConsentMode.INTERACTIVE:
            logger.warning(f"Non-interactive mode denied permission {permission.to_string()} to {agent_id}")
            return False, None
        
        # Create consent request
        request = ConsentRequest(
            agent_id=agent_id,
            permission=permission,
            scope=scope,
            justification=justification,
            risk_level=self._assess_risk_level(permission),
            duration=kwargs.get('duration'),
            resource_details=kwargs.get('resource_details', {}),
            session_id=context.session_id if context else None,
            ui_title=kwargs.get('ui_title'),
            ui_description=kwargs.get('ui_description'),
            ui_urgency=kwargs.get('ui_urgency', 'normal')
        )
        
        # Request user consent
        try:
            response = await self.consent_ui.request_consent(
                request, 
                timeout=kwargs.get('timeout', self.default_consent_timeout)
            )
            
            # Record consent decision
            self.consent_history.append(response)
            
            # Handle the response
            if response.action == ConsentAction.GRANT:
                grant_id = await self._create_grant(request, response)
                logger.info(f"User granted permission {permission.to_string()} to {agent_id}")
                return True, grant_id
            
            elif response.action == ConsentAction.DELEGATE:
                # Handle delegation
                if response.delegated_to:
                    logger.info(f"Permission {permission.to_string()} delegated to {response.delegated_to}")
                    # Create delegated grant
                    grant_id = await self._create_delegated_grant(request, response)
                    return True, grant_id
            
            logger.info(f"User denied permission {permission.to_string()} to {agent_id}: {response.reason}")
            return False, None
            
        except Exception as e:
            logger.error(f"Error processing consent request: {e}")
            return False, None
    
    def _find_existing_grant(
        self,
        agent_id: str,
        permission: PermissionDetail,
        scope: PermissionScope
    ) -> Optional[str]:
        """Find existing grant for the permission."""
        with self._lock:
            for grant_id, grant in self.active_grants.items():
                if (grant.agent_id == agent_id and
                    grant.permission.to_string() == permission.to_string() and
                    grant.scope == scope and
                    grant.can_use()):
                    return grant_id
        return None
    
    async def _evaluate_policies(
        self,
        agent_id: str,
        permission: PermissionDetail,
        scope: PermissionScope,
        context: Optional[SecurityContext]
    ) -> Optional[Tuple[ConsentAction, Optional[str]]]:
        """Evaluate policies for automatic consent decisions."""
        permission_str = permission.to_string()
        risk_level = self._assess_risk_level(permission)
        
        # Sort policies by priority (highest first)
        sorted_policies = sorted(
            [p for p in self.policies.values() if p.enabled],
            key=lambda p: p.priority,
            reverse=True
        )
        
        for policy in sorted_policies:
            if await self._policy_matches(policy, agent_id, permission_str, scope, risk_level, context):
                logger.debug(f"Policy '{policy.name}' matched for {permission_str}")
                
                if policy.action == ConsentAction.GRANT:
                    # Create grant based on policy
                    grant_id = await self._create_policy_grant(policy, agent_id, permission, scope)
                    return ConsentAction.GRANT, grant_id
                
                elif policy.action == ConsentAction.DENY:
                    return ConsentAction.DENY, None
        
        return None
    
    async def _policy_matches(
        self,
        policy: PermissionPolicy,
        agent_id: str,
        permission_str: str,
        scope: PermissionScope,
        risk_level: str,
        context: Optional[SecurityContext]
    ) -> bool:
        """Check if a policy matches the current request."""
        # Check agent patterns
        if policy.agent_patterns:
            if not any(self._pattern_matches(pattern, agent_id) for pattern in policy.agent_patterns):
                return False
        
        # Check permission patterns
        if policy.permission_patterns:
            if not any(self._pattern_matches(pattern, permission_str) for pattern in policy.permission_patterns):
                return False
        
        # Check scope filter
        if policy.scope_filter and policy.scope_filter != scope:
            return False
        
        # Check risk level
        risk_levels = ["low", "medium", "high", "critical"]
        if risk_level in risk_levels and policy.risk_level_max in risk_levels:
            if risk_levels.index(risk_level) > risk_levels.index(policy.risk_level_max):
                return False
        
        # Check additional conditions
        if policy.conditions:
            # Evaluate custom conditions (could be extended)
            pass
        
        return True
    
    def _pattern_matches(self, pattern: str, value: str) -> bool:
        """Check if a pattern matches a value (supports wildcards)."""
        import re
        # Convert glob-like pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        return bool(re.match(f"^{regex_pattern}$", value))
    
    def _assess_risk_level(self, permission: PermissionDetail) -> str:
        """Assess risk level of a permission."""
        high_risk_categories = [PermissionCategory.SYSTEM, PermissionCategory.PROCESS]
        high_risk_actions = ["execute", "delete", "admin", "root"]
        
        if permission.category in high_risk_categories:
            return "high"
        
        if any(action in permission.action.lower() for action in high_risk_actions):
            return "high"
        
        if permission.category == PermissionCategory.NETWORK:
            if "server" in permission.action.lower():
                return "medium"
            return "low"
        
        if permission.category == PermissionCategory.FILESYSTEM:
            if "write" in permission.action.lower() or "delete" in permission.action.lower():
                return "medium"
            return "low"
        
        return "low"
    
    async def _create_grant(
        self,
        request: ConsentRequest,
        response: ConsentResponse
    ) -> str:
        """Create a permission grant from consent response."""
        grant_id = str(uuid.uuid4())
        
        # Determine expiration
        expires_at = None
        if response.expires_at:
            expires_at = response.expires_at
        elif request.duration:
            expires_at = datetime.now() + request.duration
        else:
            # Default expiration based on scope
            if request.scope == PermissionScope.TASK:
                expires_at = datetime.now() + timedelta(hours=1)
            elif request.scope == PermissionScope.SESSION:
                expires_at = datetime.now() + timedelta(hours=8)
            elif request.scope == PermissionScope.AGENT:
                expires_at = datetime.now() + timedelta(days=1)
        
        grant = PermissionGrant(
            grant_id=grant_id,
            agent_id=request.agent_id,
            permission=request.permission,
            scope=request.scope,
            granted_at=datetime.now(),
            expires_at=expires_at,
            constraints=response.constraints,
            granted_by=response.user_id,
            source_request_id=request.request_id
        )
        
        with self._lock:
            self.active_grants[grant_id] = grant
        
        logger.debug(f"Created grant {grant_id} for {request.permission.to_string()}")
        return grant_id
    
    async def _create_policy_grant(
        self,
        policy: PermissionPolicy,
        agent_id: str,
        permission: PermissionDetail,
        scope: PermissionScope
    ) -> str:
        """Create a grant based on policy decision."""
        grant_id = str(uuid.uuid4())
        
        expires_at = None
        if policy.auto_duration:
            expires_at = datetime.now() + policy.auto_duration
        
        grant = PermissionGrant(
            grant_id=grant_id,
            agent_id=agent_id,
            permission=permission,
            scope=scope,
            granted_at=datetime.now(),
            expires_at=expires_at,
            constraints=policy.auto_constraints,
            policy_id=policy.policy_id
        )
        
        with self._lock:
            self.active_grants[grant_id] = grant
        
        logger.debug(f"Created policy grant {grant_id} from policy '{policy.name}'")
        return grant_id
    
    async def _create_delegated_grant(
        self,
        request: ConsentRequest,
        response: ConsentResponse
    ) -> str:
        """Create a delegated permission grant."""
        grant_id = str(uuid.uuid4())
        
        grant = PermissionGrant(
            grant_id=grant_id,
            agent_id=response.delegated_to,
            permission=request.permission,
            scope=request.scope,
            granted_at=datetime.now(),
            expires_at=response.expires_at,
            constraints=response.delegation_constraints,
            granted_by=response.user_id,
            source_request_id=request.request_id,
            delegated_from=request.agent_id,
            can_delegate=response.delegation_constraints.get('can_delegate', False)
        )
        
        with self._lock:
            self.active_grants[grant_id] = grant
        
        logger.debug(f"Created delegated grant {grant_id} for {response.delegated_to}")
        return grant_id
    
    def check_permission(
        self,
        agent_id: str,
        permission: Union[str, PermissionDetail],
        scope: PermissionScope = PermissionScope.TASK
    ) -> bool:
        """Check if an agent has a specific permission."""
        if isinstance(permission, str):
            try:
                permission = PermissionDetail.from_string(permission)
            except ValueError:
                return False
        
        grant_id = self._find_existing_grant(agent_id, permission, scope)
        if grant_id:
            grant = self.active_grants[grant_id]
            return grant.can_use()
        
        return False
    
    def revoke_permission(
        self,
        grant_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        permission: Optional[str] = None
    ) -> int:
        """Revoke permission grants."""
        revoked_count = 0
        
        with self._lock:
            grants_to_revoke = []
            
            for gid, grant in self.active_grants.items():
                if grant_id and gid != grant_id:
                    continue
                if agent_id and grant.agent_id != agent_id:
                    continue
                if permission and grant.permission.to_string() != permission:
                    continue
                
                grants_to_revoke.append(gid)
            
            for gid in grants_to_revoke:
                self._revoke_grant(gid)
                revoked_count += 1
        
        logger.info(f"Revoked {revoked_count} permission grants")
        return revoked_count
    
    def _revoke_grant(self, grant_id: str) -> bool:
        """Revoke a specific grant."""
        with self._lock:
            grant = self.active_grants.pop(grant_id, None)
            if grant:
                logger.debug(f"Revoked grant {grant_id} for {grant.permission.to_string()}")
                return True
        return False
    
    def add_policy(self, policy: PermissionPolicy) -> str:
        """Add a permission policy."""
        with self._lock:
            self.policies[policy.policy_id] = policy
        
        logger.info(f"Added policy '{policy.name}' with ID {policy.policy_id}")
        return policy.policy_id
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a permission policy."""
        with self._lock:
            policy = self.policies.pop(policy_id, None)
            if policy:
                logger.info(f"Removed policy '{policy.name}'")
                return True
        return False
    
    def get_active_grants(
        self,
        agent_id: Optional[str] = None
    ) -> List[PermissionGrant]:
        """Get active permission grants."""
        with self._lock:
            grants = list(self.active_grants.values())
            
            if agent_id:
                grants = [g for g in grants if g.agent_id == agent_id]
            
            # Filter out expired grants
            grants = [g for g in grants if g.can_use()]
            
        return grants
    
    def get_consent_history(
        self,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[ConsentResponse]:
        """Get consent decision history."""
        history = self.consent_history.copy()
        
        if since:
            history = [h for h in history if h.granted_at >= since]
        
        # Note: We'd need to track agent_id in ConsentResponse to filter by it
        return history
    
    def cleanup_expired_grants(self) -> int:
        """Clean up expired permission grants."""
        expired_count = 0
        
        with self._lock:
            expired_grants = [
                gid for gid, grant in self.active_grants.items()
                if grant.is_expired()
            ]
            
            for gid in expired_grants:
                self._revoke_grant(gid)
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired grants")
        
        return expired_count
    
    def set_consent_mode(self, mode: ConsentMode) -> None:
        """Set the consent interaction mode."""
        self.consent_mode = mode
        logger.info(f"Consent mode set to {mode.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics."""
        with self._lock:
            active_grants = len(self.active_grants)
            grants_by_scope = {}
            grants_by_category = {}
            
            for grant in self.active_grants.values():
                scope = grant.scope.value
                grants_by_scope[scope] = grants_by_scope.get(scope, 0) + 1
                
                category = grant.permission.category.value
                grants_by_category[category] = grants_by_category.get(category, 0) + 1
        
        return {
            "active_grants": active_grants,
            "grants_by_scope": grants_by_scope,
            "grants_by_category": grants_by_category,
            "consent_history_count": len(self.consent_history),
            "policies_count": len(self.policies),
            "consent_mode": self.consent_mode.value
        }


def create_consent_framework(**kwargs) -> ConsentFramework:
    """
    Create a consent framework with the given configuration.
    
    Args:
        **kwargs: Framework configuration
        
    Returns:
        Configured consent framework
    """
    return ConsentFramework(**kwargs)


# Default UI handlers for demonstration
def console_consent_handler(request: ConsentRequest) -> ConsentResponse:
    """Simple console-based consent handler."""
    print(f"\n🔐 Permission Request")
    print(f"Agent: {request.agent_id}")
    print(f"Permission: {request.permission.to_string()}")
    print(f"Justification: {request.justification}")
    print(f"Risk Level: {request.risk_level}")
    print(f"Scope: {request.scope.value}")
    
    while True:
        choice = input("\nAllow? (y/n/d for delegate): ").lower().strip()
        
        if choice == 'y':
            return ConsentResponse(
                request_id=request.request_id,
                action=ConsentAction.GRANT,
                granted_permissions=[request.permission.to_string()],
                reason="User approved"
            )
        elif choice == 'n':
            reason = input("Reason for denial (optional): ").strip()
            return ConsentResponse(
                request_id=request.request_id,
                action=ConsentAction.DENY,
                reason=reason or "User denied"
            )
        elif choice == 'd':
            delegate_to = input("Delegate to (agent ID): ").strip()
            if delegate_to:
                return ConsentResponse(
                    request_id=request.request_id,
                    action=ConsentAction.DELEGATE,
                    delegated_to=delegate_to,
                    reason="User delegated"
                )
        
        print("Please enter 'y', 'n', or 'd'")


if __name__ == "__main__":
    async def demo_consent_framework():
        """Demonstrate consent framework functionality."""
        print("🔐 Consent Framework Demo")
        print("=" * 40)
        
        # Create framework
        framework = create_consent_framework()
        
        # Register console UI handler
        framework.consent_ui.register_ui_handler("background_notification", console_consent_handler)
        framework.consent_ui.register_ui_handler("notification_popup", console_consent_handler)
        framework.consent_ui.register_ui_handler("modal_dialog", console_consent_handler)
        
        # Test permission requests
        test_permissions = [
            ("agent_001", "data.read.documents", "Need to read user documents for analysis"),
            ("agent_001", "network.http.request", "Need to fetch external data"),
            ("agent_002", "system.admin.users", "Need to manage user accounts"),
            ("agent_003", "filesystem.write.temp", "Need to write temporary files")
        ]
        
        print("\n🔍 Testing Permission Requests:")
        print("-" * 40)
        
        for agent_id, permission, justification in test_permissions:
            print(f"\nRequesting: {permission} for {agent_id}")
            
            granted, grant_id = await framework.request_permission(
                agent_id=agent_id,
                permission=permission,
                scope=PermissionScope.TASK,
                justification=justification,
                ui_urgency="normal"
            )
            
            if granted:
                print(f"✅ Permission granted with grant ID: {grant_id}")
            else:
                print(f"❌ Permission denied")
        
        # Show active grants
        print(f"\n📊 Active Grants:")
        print("-" * 20)
        
        grants = framework.get_active_grants()
        for grant in grants:
            print(f"  - {grant.agent_id}: {grant.permission.to_string()}")
            print(f"    Expires: {grant.expires_at}")
            print(f"    Usage: {grant.usage_count}/{grant.usage_limit or 'unlimited'}")
        
        # Show statistics
        print(f"\n📈 Framework Statistics:")
        print("-" * 25)
        
        stats = framework.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test policy-based decisions
        print(f"\n🤖 Testing Policy-Based Decisions:")
        print("-" * 35)
        
        # Switch to automatic mode
        framework.set_consent_mode(ConsentMode.AUTOMATIC)
        
        # Request safe operations (should be auto-approved)
        safe_permissions = [
            "data.read.public",
            "compute.cpu.basic",
            "system.info.read"
        ]
        
        for permission in safe_permissions:
            granted, grant_id = await framework.request_permission(
                agent_id="auto_agent",
                permission=permission,
                justification="Automatic request"
            )
            
            print(f"  {permission}: {'✅ Auto-granted' if granted else '❌ Auto-denied'}")
        
        # Cleanup
        expired = framework.cleanup_expired_grants()
        print(f"\n🧹 Cleaned up {expired} expired grants")
        
        print("\n🔐 Consent Framework Demo Complete")
    
    # Run demo
    asyncio.run(demo_consent_framework())