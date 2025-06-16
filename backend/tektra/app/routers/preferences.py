"""
User Preferences and Settings API Endpoints.

Endpoints for managing user preferences, model settings, and personalization.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user_preferences import NotificationLevel, ThemeMode, VoiceProvider
from ..services.preferences_service import (
    api_key_service,
    model_settings_service,
    preferences_service,
    template_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic Models


class PreferencesUpdateRequest(BaseModel):
    """Request model for updating user preferences."""

    # UI/UX Preferences
    theme_mode: Optional[ThemeMode] = None
    language: Optional[str] = Field(None, max_length=10)
    timezone: Optional[str] = Field(None, max_length=50)
    date_format: Optional[str] = Field(None, max_length=20)
    time_format: Optional[str] = Field(None, pattern="^(12h|24h)$")

    # Chat Interface Preferences
    chat_bubble_style: Optional[str] = Field(None, max_length=20)
    message_grouping: Optional[bool] = None
    show_timestamps: Optional[bool] = None
    show_token_count: Optional[bool] = None
    auto_scroll: Optional[bool] = None
    typing_indicators: Optional[bool] = None

    # AI Model Preferences
    default_model: Optional[str] = Field(None, max_length=100)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)

    # Voice and Audio Preferences
    voice_provider: Optional[VoiceProvider] = None
    voice_name: Optional[str] = Field(None, max_length=100)
    voice_speed: Optional[float] = Field(None, ge=0.1, le=3.0)
    voice_pitch: Optional[float] = Field(None, ge=-20.0, le=20.0)
    auto_play_responses: Optional[bool] = None
    voice_input_enabled: Optional[bool] = None
    noise_suppression: Optional[bool] = None

    # Avatar Preferences
    avatar_enabled: Optional[bool] = None
    avatar_style: Optional[str] = Field(None, max_length=50)
    avatar_gender: Optional[str] = Field(None, max_length=20)
    avatar_expressions: Optional[bool] = None
    avatar_lip_sync: Optional[bool] = None

    # Privacy and Security
    data_retention_days: Optional[int] = Field(None, ge=0, le=3650)
    analytics_enabled: Optional[bool] = None
    crash_reporting: Optional[bool] = None
    share_usage_data: Optional[bool] = None

    # Notifications
    notification_level: Optional[NotificationLevel] = None
    email_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    sound_notifications: Optional[bool] = None

    # Advanced Settings
    developer_mode: Optional[bool] = None
    debug_logging: Optional[bool] = None
    experimental_features: Optional[bool] = None
    beta_updates: Optional[bool] = None


class ModelSettingsRequest(BaseModel):
    """Request model for model-specific settings."""

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    streaming_enabled: Optional[bool] = None
    cache_enabled: Optional[bool] = None
    batch_size: Optional[int] = Field(None, ge=1, le=100)


class ConversationTemplateRequest(BaseModel):
    """Request model for conversation templates."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    category: str = Field(default="general", max_length=50)
    system_prompt: str = Field(..., min_length=1)
    initial_message: Optional[str] = None
    recommended_model: Optional[str] = Field(None, max_length=100)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    is_public: bool = False
    is_favorite: bool = False
    tags: Optional[List[str]] = None
    icon: Optional[str] = Field(None, max_length=50)
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")


class APIKeyRequest(BaseModel):
    """Request model for storing API keys."""

    provider: str = Field(..., min_length=1, max_length=50)
    key_name: str = Field(..., min_length=1, max_length=100)
    api_key: str = Field(..., min_length=1)
    usage_limit: Optional[int] = Field(None, ge=0)


class PreferencesResponse(BaseModel):
    """Response model for user preferences."""

    id: int
    user_id: int
    theme_mode: ThemeMode
    language: str
    timezone: str
    date_format: str
    time_format: str
    chat_bubble_style: str
    message_grouping: bool
    show_timestamps: bool
    show_token_count: bool
    auto_scroll: bool
    typing_indicators: bool
    default_model: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    voice_provider: VoiceProvider
    voice_name: str
    voice_speed: float
    voice_pitch: float
    auto_play_responses: bool
    voice_input_enabled: bool
    noise_suppression: bool
    avatar_enabled: bool
    avatar_style: str
    avatar_gender: str
    avatar_expressions: bool
    avatar_lip_sync: bool
    data_retention_days: int
    analytics_enabled: bool
    crash_reporting: bool
    share_usage_data: bool
    notification_level: NotificationLevel
    email_notifications: bool
    push_notifications: bool
    sound_notifications: bool
    developer_mode: bool
    debug_logging: bool
    experimental_features: bool
    beta_updates: bool


# Preferences Endpoints


@router.get("/preferences")
async def get_user_preferences(
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Get user preferences."""
    try:
        preferences = await preferences_service.get_user_preferences(db, user_id)

        if not preferences:
            raise HTTPException(status_code=404, detail="User preferences not found")

        return {
            "status": "success",
            "preferences": PreferencesResponse(
                id=preferences.id,
                user_id=preferences.user_id,
                theme_mode=preferences.theme_mode,
                language=preferences.language,
                timezone=preferences.timezone,
                date_format=preferences.date_format,
                time_format=preferences.time_format,
                chat_bubble_style=preferences.chat_bubble_style,
                message_grouping=preferences.message_grouping,
                show_timestamps=preferences.show_timestamps,
                show_token_count=preferences.show_token_count,
                auto_scroll=preferences.auto_scroll,
                typing_indicators=preferences.typing_indicators,
                default_model=preferences.default_model,
                temperature=preferences.temperature,
                max_tokens=preferences.max_tokens,
                top_p=preferences.top_p,
                frequency_penalty=preferences.frequency_penalty,
                presence_penalty=preferences.presence_penalty,
                voice_provider=preferences.voice_provider,
                voice_name=preferences.voice_name,
                voice_speed=preferences.voice_speed,
                voice_pitch=preferences.voice_pitch,
                auto_play_responses=preferences.auto_play_responses,
                voice_input_enabled=preferences.voice_input_enabled,
                noise_suppression=preferences.noise_suppression,
                avatar_enabled=preferences.avatar_enabled,
                avatar_style=preferences.avatar_style,
                avatar_gender=preferences.avatar_gender,
                avatar_expressions=preferences.avatar_expressions,
                avatar_lip_sync=preferences.avatar_lip_sync,
                data_retention_days=preferences.data_retention_days,
                analytics_enabled=preferences.analytics_enabled,
                crash_reporting=preferences.crash_reporting,
                share_usage_data=preferences.share_usage_data,
                notification_level=preferences.notification_level,
                email_notifications=preferences.email_notifications,
                push_notifications=preferences.push_notifications,
                sound_notifications=preferences.sound_notifications,
                developer_mode=preferences.developer_mode,
                debug_logging=preferences.debug_logging,
                experimental_features=preferences.experimental_features,
                beta_updates=preferences.beta_updates,
            ),
        }

    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/preferences")
async def update_user_preferences(
    request: PreferencesUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Update user preferences."""
    try:
        preferences_data = request.dict(exclude_unset=True)

        preferences = await preferences_service.update_preferences(
            db, user_id, preferences_data
        )

        return {
            "status": "success",
            "message": "Preferences updated successfully",
            "preferences_id": preferences.id,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preferences/reset")
async def reset_user_preferences(
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Reset user preferences to defaults."""
    try:
        preferences = await preferences_service.reset_preferences(db, user_id)

        return {
            "status": "success",
            "message": "Preferences reset to defaults",
            "preferences_id": preferences.id,
        }

    except Exception as e:
        logger.error(f"Error resetting preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Settings Endpoints


@router.get("/model-settings")
async def get_all_model_settings(
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Get all model-specific settings for the user."""
    try:
        settings_list = await model_settings_service.get_all_model_settings(db, user_id)

        return {
            "status": "success",
            "model_settings": [
                {
                    "id": settings.id,
                    "model_name": settings.model_name,
                    "temperature": settings.temperature,
                    "max_tokens": settings.max_tokens,
                    "top_p": settings.top_p,
                    "top_k": settings.top_k,
                    "frequency_penalty": settings.frequency_penalty,
                    "presence_penalty": settings.presence_penalty,
                    "system_prompt": settings.system_prompt,
                    "stop_sequences": settings.stop_sequences,
                    "streaming_enabled": settings.streaming_enabled,
                    "cache_enabled": settings.cache_enabled,
                    "batch_size": settings.batch_size,
                    "total_requests": settings.total_requests,
                    "total_tokens": settings.total_tokens,
                    "avg_response_time": settings.avg_response_time,
                    "last_used": settings.last_used,
                    "custom_parameters": settings.custom_parameters,
                }
                for settings in settings_list
            ],
            "total": len(settings_list),
        }

    except Exception as e:
        logger.error(f"Error getting model settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-settings/{model_name}")
async def get_model_settings(
    model_name: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Get settings for a specific model."""
    try:
        settings = await model_settings_service.get_model_settings(
            db, user_id, model_name
        )

        if not settings:
            return {
                "status": "success",
                "model_settings": None,
                "message": f"No custom settings found for model {model_name}",
            }

        return {
            "status": "success",
            "model_settings": {
                "id": settings.id,
                "model_name": settings.model_name,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "frequency_penalty": settings.frequency_penalty,
                "presence_penalty": settings.presence_penalty,
                "system_prompt": settings.system_prompt,
                "stop_sequences": settings.stop_sequences,
                "streaming_enabled": settings.streaming_enabled,
                "cache_enabled": settings.cache_enabled,
                "batch_size": settings.batch_size,
                "total_requests": settings.total_requests,
                "total_tokens": settings.total_tokens,
                "avg_response_time": settings.avg_response_time,
                "last_used": settings.last_used,
                "custom_parameters": settings.custom_parameters,
            },
        }

    except Exception as e:
        logger.error(f"Error getting model settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/model-settings/{model_name}")
async def update_model_settings(
    model_name: str,
    request: ModelSettingsRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Update settings for a specific model."""
    try:
        settings_data = request.dict(exclude_unset=True)

        settings = await model_settings_service.update_model_settings(
            db, user_id, model_name, settings_data
        )

        return {
            "status": "success",
            "message": f"Settings updated for model {model_name}",
            "model_name": model_name,
            "settings_id": settings.id,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating model settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/model-settings/{model_name}")
async def delete_model_settings(
    model_name: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Delete custom settings for a specific model."""
    try:
        success = await model_settings_service.delete_model_settings(
            db, user_id, model_name
        )

        if success:
            return {
                "status": "success",
                "message": f"Settings deleted for model {model_name}",
                "model_name": model_name,
            }
        else:
            raise HTTPException(
                status_code=404, detail=f"No settings found for model {model_name}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation Templates Endpoints


@router.get("/templates")
async def get_conversation_templates(
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
    category: Optional[str] = None,
    include_public: bool = True,
) -> Dict[str, Any]:
    """Get conversation templates."""
    try:
        templates = await template_service.get_user_templates(
            db, user_id, category, include_public
        )

        return {
            "status": "success",
            "templates": [
                {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "system_prompt": template.system_prompt,
                    "initial_message": template.initial_message,
                    "recommended_model": template.recommended_model,
                    "temperature": template.temperature,
                    "max_tokens": template.max_tokens,
                    "is_public": template.is_public,
                    "is_favorite": template.is_favorite,
                    "usage_count": template.usage_count,
                    "tags": template.tags,
                    "icon": template.icon,
                    "color": template.color,
                    "created_at": template.created_at,
                    "last_used": template.last_used,
                    "is_owner": template.user_id == user_id,
                }
                for template in templates
            ],
            "total": len(templates),
        }

    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates")
async def create_conversation_template(
    request: ConversationTemplateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Create a new conversation template."""
    try:
        template_data = request.dict()

        template = await template_service.create_template(db, user_id, template_data)

        return {
            "status": "success",
            "message": "Template created successfully",
            "template": {
                "id": template.id,
                "name": template.name,
                "category": template.category,
            },
        }

    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/templates/{template_id}")
async def update_conversation_template(
    template_id: int,
    request: ConversationTemplateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Update a conversation template."""
    try:
        template_data = request.dict(exclude_unset=True)

        template = await template_service.update_template(
            db, template_id, user_id, template_data
        )

        return {
            "status": "success",
            "message": "Template updated successfully",
            "template_id": template.id,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/templates/{template_id}")
async def delete_conversation_template(
    template_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Delete a conversation template."""
    try:
        success = await template_service.delete_template(db, template_id, user_id)

        if success:
            return {
                "status": "success",
                "message": "Template deleted successfully",
                "template_id": template_id,
            }
        else:
            raise HTTPException(status_code=404, detail="Template not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/{template_id}/use")
async def use_conversation_template(
    template_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Mark a template as used and get its content."""
    try:
        template = await template_service.use_template(db, template_id, user_id)

        return {
            "status": "success",
            "message": "Template marked as used",
            "template": {
                "id": template.id,
                "name": template.name,
                "system_prompt": template.system_prompt,
                "initial_message": template.initial_message,
                "recommended_model": template.recommended_model,
                "temperature": template.temperature,
                "max_tokens": template.max_tokens,
                "usage_count": template.usage_count,
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error using template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
