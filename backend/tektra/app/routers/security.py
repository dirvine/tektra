"""
Security API endpoints.

Provides REST API for biometric authentication, vault management, and security operations.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import base64

from ..services.security_service import security_service
from ..security.anonymization import anonymization_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/security", tags=["security"])
security_scheme = HTTPBearer()


# Request Models
class RegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=50)
    face_image: str = Field(..., description="Base64 encoded face image")
    voice_audio: str = Field(..., description="Base64 encoded voice audio")
    pin: str = Field(..., min_length=4, max_length=8)


class AuthenticateRequest(BaseModel):
    face_image: str = Field(..., description="Base64 encoded face image")
    voice_audio: str = Field(..., description="Base64 encoded voice audio")
    pin: str = Field(..., min_length=4, max_length=8)


class AnonymizeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    preserve_technical: bool = Field(default=True)


class VaultMessageRequest(BaseModel):
    conversation_id: str
    message: Dict[str, Any]


# Dependency for session validation
async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> Dict[str, Any]:
    """Validate session token and return session data."""
    session_data = await security_service.validate_session(credentials.credentials)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token"
        )
    return session_data


@router.post("/register")
async def register_user(request: RegisterRequest):
    """
    Register a new user with biometric authentication.
    
    Args:
        request: Registration request with biometric data
        
    Returns:
        Registration result
    """
    try:
        # Decode base64 data
        face_image = base64.b64decode(request.face_image)
        voice_audio = base64.b64decode(request.voice_audio)
        
        result = await security_service.register_user(
            request.user_id,
            face_image,
            voice_audio,
            request.pin
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Registration failed")
            )
        
        logger.info(f"User registered successfully: {request.user_id}")
        return result
        
    except ValueError as e:
        logger.error(f"Invalid base64 data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image or audio data"
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/authenticate")
async def authenticate_user(request: AuthenticateRequest):
    """
    Authenticate user with biometric data and PIN.
    
    Args:
        request: Authentication request
        
    Returns:
        Authentication result with session token
    """
    try:
        # Decode base64 data
        face_image = base64.b64decode(request.face_image)
        voice_audio = base64.b64decode(request.voice_audio)
        
        result = await security_service.authenticate_user(
            face_image,
            voice_audio,
            request.pin
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.get("error", "Authentication failed")
            )
        
        logger.info(f"User authenticated successfully: {result['user_id']}")
        return result
        
    except ValueError as e:
        logger.error(f"Invalid base64 data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image or audio data"
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


@router.post("/logout")
async def logout_user(session: Dict = Depends(get_current_session)):
    """
    Logout current user and invalidate session.
    
    Args:
        session: Current session data
        
    Returns:
        Logout result
    """
    try:
        # Get session token from Authorization header
        # Note: In a real implementation, you'd extract this from the dependency
        session_token = "dummy_token"  # This would come from the dependency
        
        success = await security_service.logout_user(session_token)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed"
            )
        
        logger.info(f"User logged out: {session['user_id']}")
        return {"success": True, "message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/status")
async def get_security_status():
    """
    Get security system status and capabilities.
    
    Returns:
        Security system information
    """
    try:
        status_info = security_service.get_security_status()
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get security status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security status"
        )


@router.post("/anonymize")
async def anonymize_query(
    request: AnonymizeRequest,
    session: Dict = Depends(get_current_session)
):
    """
    Anonymize query for external API usage.
    
    Args:
        request: Query to anonymize
        session: Current session
        
    Returns:
        Anonymized query result
    """
    try:
        # Get session token (would be extracted from dependency in real implementation)
        session_token = "dummy_token"
        
        result = await security_service.anonymize_external_query(
            session_token,
            request.query,
            request.preserve_technical
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Anonymization failed")
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Query anonymization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Anonymization failed"
        )


# Vault Management Endpoints
@router.get("/vault/conversations")
async def get_conversations(session: Dict = Depends(get_current_session)):
    """
    Get all conversations for authenticated user.
    
    Args:
        session: Current session
        
    Returns:
        List of user conversations
    """
    try:
        session_token = "dummy_token"  # Would come from dependency
        conversations = await security_service.get_user_conversations(session_token)
        return conversations
        
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversations"
        )


@router.post("/vault/message")
async def save_message(
    request: VaultMessageRequest,
    session: Dict = Depends(get_current_session)
):
    """
    Save message to user's encrypted vault.
    
    Args:
        request: Message to save
        session: Current session
        
    Returns:
        Save result
    """
    try:
        session_token = "dummy_token"  # Would come from dependency
        
        success = await security_service.save_conversation_message(
            session_token,
            request.conversation_id,
            request.message
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save message"
            )
        
        return {"success": True, "message": "Message saved to vault"}
        
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save message"
        )


@router.get("/vault/stats")
async def get_vault_stats(session: Dict = Depends(get_current_session)):
    """
    Get vault statistics for authenticated user.
    
    Args:
        session: Current session
        
    Returns:
        Vault statistics
    """
    try:
        session_token = "dummy_token"  # Would come from dependency
        
        # Get user preferences which includes vault stats
        preferences = await security_service.get_user_preferences(session_token)
        
        # In a real implementation, you'd get specific vault stats
        # For now, return basic structure
        stats = {
            "total_conversations": 0,
            "total_messages": 0,
            "last_access": None
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get vault stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get vault stats"
        )


@router.get("/vault/preferences")
async def get_preferences(session: Dict = Depends(get_current_session)):
    """
    Get user preferences from vault.
    
    Args:
        session: Current session
        
    Returns:
        User preferences
    """
    try:
        session_token = "dummy_token"  # Would come from dependency
        preferences = await security_service.get_user_preferences(session_token)
        return preferences
        
    except Exception as e:
        logger.error(f"Failed to get preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get preferences"
        )


@router.put("/vault/preferences")
async def update_preferences(
    preferences: Dict[str, Any],
    session: Dict = Depends(get_current_session)
):
    """
    Update user preferences in vault.
    
    Args:
        preferences: New preferences
        session: Current session
        
    Returns:
        Update result
    """
    try:
        session_token = "dummy_token"  # Would come from dependency
        
        success = await security_service.update_user_preferences(
            session_token,
            preferences
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update preferences"
            )
        
        return {"success": True, "message": "Preferences updated"}
        
    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )


# Anonymization statistics
@router.get("/anonymization/stats")
async def get_anonymization_stats():
    """
    Get anonymization statistics.
    
    Returns:
        Anonymization statistics
    """
    try:
        stats = anonymization_service.get_anonymization_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get anonymization stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get anonymization stats"
        )