"""
User Preferences and Settings Service.

Service for managing user preferences, model settings, and personalization options.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import and_, or_

from ..models.user_preferences import (
    UserPreferences, ModelSettings, ConversationTemplate, APIKey,
    ThemeMode, VoiceProvider, NotificationLevel
)
from ..models.user import User

logger = logging.getLogger(__name__)


class PreferencesService:
    """Service for managing user preferences."""
    
    async def get_user_preferences(
        self,
        db: AsyncSession,
        user_id: int,
        create_if_missing: bool = True
    ) -> Optional[UserPreferences]:
        """Get user preferences, optionally creating defaults if missing."""
        try:
            preferences = await db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).options(
                selectinload(UserPreferences.model_settings)
            ).first()
            
            if not preferences and create_if_missing:
                preferences = await self.create_default_preferences(db, user_id)
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            raise
    
    async def create_default_preferences(
        self,
        db: AsyncSession,
        user_id: int
    ) -> UserPreferences:
        """Create default preferences for a new user."""
        try:
            preferences = UserPreferences(
                user_id=user_id,
                # Defaults are set in the model
            )
            
            db.add(preferences)
            await db.commit()
            await db.refresh(preferences)
            
            return preferences
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating default preferences: {e}")
            raise
    
    async def update_preferences(
        self,
        db: AsyncSession,
        user_id: int,
        preferences_data: Dict[str, Any]
    ) -> UserPreferences:
        """Update user preferences."""
        try:
            preferences = await self.get_user_preferences(db, user_id)
            
            if not preferences:
                raise ValueError("User preferences not found")
            
            # Update only provided fields
            for field, value in preferences_data.items():
                if hasattr(preferences, field):
                    setattr(preferences, field, value)
            
            preferences.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(preferences)
            
            return preferences
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating preferences: {e}")
            raise
    
    async def reset_preferences(
        self,
        db: AsyncSession,
        user_id: int
    ) -> UserPreferences:
        """Reset user preferences to defaults."""
        try:
            # Delete existing preferences
            await db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).delete()
            
            # Create new default preferences
            preferences = await self.create_default_preferences(db, user_id)
            
            return preferences
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error resetting preferences: {e}")
            raise


class ModelSettingsService:
    """Service for managing model-specific settings."""
    
    async def get_model_settings(
        self,
        db: AsyncSession,
        user_id: int,
        model_name: str
    ) -> Optional[ModelSettings]:
        """Get settings for a specific model."""
        try:
            preferences = await db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not preferences:
                return None
            
            settings = await db.query(ModelSettings).filter(
                and_(
                    ModelSettings.user_preferences_id == preferences.id,
                    ModelSettings.model_name == model_name
                )
            ).first()
            
            return settings
            
        except Exception as e:
            logger.error(f"Error getting model settings: {e}")
            raise
    
    async def update_model_settings(
        self,
        db: AsyncSession,
        user_id: int,
        model_name: str,
        settings_data: Dict[str, Any]
    ) -> ModelSettings:
        """Update or create settings for a specific model."""
        try:
            preferences = await db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not preferences:
                raise ValueError("User preferences not found")
            
            # Get existing settings or create new
            settings = await self.get_model_settings(db, user_id, model_name)
            
            if not settings:
                settings = ModelSettings(
                    user_preferences_id=preferences.id,
                    model_name=model_name
                )
                db.add(settings)
            
            # Update fields
            for field, value in settings_data.items():
                if hasattr(settings, field):
                    setattr(settings, field, value)
            
            settings.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(settings)
            
            return settings
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating model settings: {e}")
            raise
    
    async def get_all_model_settings(
        self,
        db: AsyncSession,
        user_id: int
    ) -> List[ModelSettings]:
        """Get all model settings for a user."""
        try:
            preferences = await db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not preferences:
                return []
            
            settings = await db.query(ModelSettings).filter(
                ModelSettings.user_preferences_id == preferences.id
            ).all()
            
            return settings
            
        except Exception as e:
            logger.error(f"Error getting all model settings: {e}")
            raise
    
    async def delete_model_settings(
        self,
        db: AsyncSession,
        user_id: int,
        model_name: str
    ) -> bool:
        """Delete settings for a specific model."""
        try:
            preferences = await db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not preferences:
                return False
            
            deleted_count = await db.query(ModelSettings).filter(
                and_(
                    ModelSettings.user_preferences_id == preferences.id,
                    ModelSettings.model_name == model_name
                )
            ).delete()
            
            await db.commit()
            return deleted_count > 0
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting model settings: {e}")
            raise


class ConversationTemplateService:
    """Service for managing conversation templates."""
    
    async def create_template(
        self,
        db: AsyncSession,
        user_id: int,
        template_data: Dict[str, Any]
    ) -> ConversationTemplate:
        """Create a new conversation template."""
        try:
            template = ConversationTemplate(
                user_id=user_id,
                **template_data
            )
            
            db.add(template)
            await db.commit()
            await db.refresh(template)
            
            return template
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating template: {e}")
            raise
    
    async def get_user_templates(
        self,
        db: AsyncSession,
        user_id: int,
        category: Optional[str] = None,
        include_public: bool = True
    ) -> List[ConversationTemplate]:
        """Get templates for a user."""
        try:
            query = db.query(ConversationTemplate)
            
            if include_public:
                query = query.filter(
                    or_(
                        ConversationTemplate.user_id == user_id,
                        ConversationTemplate.is_public == True
                    )
                )
            else:
                query = query.filter(ConversationTemplate.user_id == user_id)
            
            if category:
                query = query.filter(ConversationTemplate.category == category)
            
            templates = await query.order_by(
                ConversationTemplate.is_favorite.desc(),
                ConversationTemplate.usage_count.desc(),
                ConversationTemplate.name
            ).all()
            
            return templates
            
        except Exception as e:
            logger.error(f"Error getting user templates: {e}")
            raise
    
    async def update_template(
        self,
        db: AsyncSession,
        template_id: int,
        user_id: int,
        template_data: Dict[str, Any]
    ) -> ConversationTemplate:
        """Update a conversation template."""
        try:
            template = await db.query(ConversationTemplate).filter(
                and_(
                    ConversationTemplate.id == template_id,
                    ConversationTemplate.user_id == user_id
                )
            ).first()
            
            if not template:
                raise ValueError("Template not found")
            
            # Update fields
            for field, value in template_data.items():
                if hasattr(template, field):
                    setattr(template, field, value)
            
            template.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(template)
            
            return template
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating template: {e}")
            raise
    
    async def delete_template(
        self,
        db: AsyncSession,
        template_id: int,
        user_id: int
    ) -> bool:
        """Delete a conversation template."""
        try:
            deleted_count = await db.query(ConversationTemplate).filter(
                and_(
                    ConversationTemplate.id == template_id,
                    ConversationTemplate.user_id == user_id
                )
            ).delete()
            
            await db.commit()
            return deleted_count > 0
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting template: {e}")
            raise
    
    async def use_template(
        self,
        db: AsyncSession,
        template_id: int,
        user_id: int
    ) -> ConversationTemplate:
        """Mark a template as used (increment usage count)."""
        try:
            template = await db.query(ConversationTemplate).filter(
                ConversationTemplate.id == template_id
            ).first()
            
            if not template:
                raise ValueError("Template not found")
            
            # Check if user has access (owner or public)
            if template.user_id != user_id and not template.is_public:
                raise ValueError("Access denied")
            
            template.usage_count += 1
            template.last_used = datetime.utcnow()
            
            await db.commit()
            await db.refresh(template)
            
            return template
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error using template: {e}")
            raise


class APIKeyService:
    """Service for managing encrypted API keys."""
    
    def __init__(self):
        # In a real implementation, you'd use proper encryption
        # For now, we'll use a simple placeholder
        self.encryption_key = "your-secret-key"  # Use proper key management
    
    async def store_api_key(
        self,
        db: AsyncSession,
        user_id: int,
        provider: str,
        key_name: str,
        api_key: str,
        usage_limit: Optional[int] = None
    ) -> APIKey:
        """Store an encrypted API key."""
        try:
            # Encrypt the API key (placeholder implementation)
            encrypted_key = self._encrypt_key(api_key)
            
            api_key_record = APIKey(
                user_id=user_id,
                provider=provider,
                key_name=key_name,
                encrypted_key=encrypted_key,
                usage_limit=usage_limit
            )
            
            db.add(api_key_record)
            await db.commit()
            await db.refresh(api_key_record)
            
            return api_key_record
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error storing API key: {e}")
            raise
    
    async def get_api_key(
        self,
        db: AsyncSession,
        user_id: int,
        provider: str
    ) -> Optional[str]:
        """Get decrypted API key for a provider."""
        try:
            api_key_record = await db.query(APIKey).filter(
                and_(
                    APIKey.user_id == user_id,
                    APIKey.provider == provider,
                    APIKey.is_active == True
                )
            ).first()
            
            if not api_key_record:
                return None
            
            # Decrypt the API key (placeholder implementation)
            decrypted_key = self._decrypt_key(api_key_record.encrypted_key)
            
            # Update usage tracking
            api_key_record.last_used = datetime.utcnow()
            await db.commit()
            
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Error getting API key: {e}")
            raise
    
    async def list_api_keys(
        self,
        db: AsyncSession,
        user_id: int
    ) -> List[APIKey]:
        """List all API keys for a user (without decryption)."""
        try:
            api_keys = await db.query(APIKey).filter(
                APIKey.user_id == user_id
            ).all()
            
            return api_keys
            
        except Exception as e:
            logger.error(f"Error listing API keys: {e}")
            raise
    
    async def delete_api_key(
        self,
        db: AsyncSession,
        key_id: int,
        user_id: int
    ) -> bool:
        """Delete an API key."""
        try:
            deleted_count = await db.query(APIKey).filter(
                and_(
                    APIKey.id == key_id,
                    APIKey.user_id == user_id
                )
            ).delete()
            
            await db.commit()
            return deleted_count > 0
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting API key: {e}")
            raise
    
    def _encrypt_key(self, key: str) -> str:
        """Encrypt an API key (placeholder implementation)."""
        # In a real implementation, use proper encryption like Fernet
        import base64
        return base64.b64encode(key.encode()).decode()
    
    def _decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt an API key (placeholder implementation)."""
        # In a real implementation, use proper decryption
        import base64
        return base64.b64decode(encrypted_key.encode()).decode()


# Global service instances
preferences_service = PreferencesService()
model_settings_service = ModelSettingsService()
template_service = ConversationTemplateService()
api_key_service = APIKeyService()