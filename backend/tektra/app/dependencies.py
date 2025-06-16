"""
FastAPI dependencies for authentication and authorization.

This module provides dependency functions that can be used across
different API endpoints for user authentication and session management.
"""

from typing import Optional

from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models.user import User

# Optional security scheme for future authentication
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Get the current authenticated user.

    For now, this returns a default user since authentication is not yet implemented.
    In the future, this will validate JWT tokens and return the authenticated user.

    Args:
        credentials: Optional authentication credentials
        db: Database session

    Returns:
        User: The current user (default user for now)

    Raises:
        HTTPException: If authentication fails (in future implementation)
    """
    # For now, return a default user since we don't have authentication implemented
    # In the future, this will:
    # 1. Validate the JWT token from credentials
    # 2. Extract user info from the token
    # 3. Query the database for the user
    # 4. Return the authenticated user or raise HTTPException

    # Create a default user for now
    default_user = User(
        id=1, username="default_user", email="user@tektra.ai", is_active=True
    )

    return default_user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get the current active user.

    Args:
        current_user: Current user from get_current_user

    Returns:
        User: Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Get current user if they have admin privileges.

    Args:
        current_user: Current active user

    Returns:
        User: Admin user

    Raises:
        HTTPException: If user is not an admin
    """
    # For now, assume all users are admins since we don't have role-based auth
    # In the future, check if user has admin role
    return current_user


async def get_current_user_websocket(
    websocket: WebSocket, token: Optional[str] = None
) -> User:
    """
    Get current user for WebSocket connections.

    Args:
        websocket: WebSocket connection
        token: Optional authentication token

    Returns:
        User: Default user for now (future: authenticate via token or headers)
    """
    # For now, return default user since WebSocket auth is not implemented
    # In the future, this will validate the token parameter
    # In the future, this could authenticate via:
    # - Query parameters: ?token=jwt_token
    # - Headers during handshake
    # - Cookies

    default_user = User(
        id=1, username="default_user", email="user@tektra.ai", is_active=True
    )

    return default_user


# Optional: Dependency for API key authentication
async def get_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    Extract API key from authorization header.

    Args:
        credentials: Authorization credentials

    Returns:
        Optional API key string
    """
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    return None
