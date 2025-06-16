"""
Security Coordination Service.

Coordinates all security-related operations and provides unified security interface.
"""

import asyncio
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
import uuid

from ..security.biometric_auth import biometric_auth_service
from ..security.vault_manager import vault_manager
from ..security.anonymization import anonymization_service

logger = logging.getLogger(__name__)


class SecurityService:
    """Coordinates all security operations for Tektra."""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.session_timeout = 3600  # 1 hour in seconds
        self.max_concurrent_sessions = 10
        
    async def initialize(self):
        """Initialize all security services."""
        try:
            # Initialize biometric authentication
            await biometric_auth_service.initialize()
            
            # Initialize vault manager
            await vault_manager.initialize()
            
            logger.info("Security service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize security service: {e}")
            raise
    
    async def register_user(
        self,
        user_id: str,
        face_image: bytes,
        voice_audio: bytes,
        pin: str
    ) -> Dict[str, Any]:
        """
        Register a new user with complete security setup.
        
        Args:
            user_id: Unique user identifier
            face_image: User's face image
            voice_audio: User's voice sample
            pin: User's chosen PIN
            
        Returns:
            Registration result
        """
        try:
            logger.info(f"Starting user registration for: {user_id}")
            
            # Step 1: Register biometric data
            biometric_result = await biometric_auth_service.register_user(
                user_id, face_image, voice_audio, pin
            )
            
            if not biometric_result["success"]:
                return biometric_result
            
            # Step 2: Create encrypted vault
            vault_key = vault_manager.key_derivation_service.decode_key_b64(
                biometric_result["vault_key"]
            )
            
            vault_created = await vault_manager.create_vault(
                user_id,
                biometric_result["biometric_hash"],
                vault_key
            )
            
            if not vault_created:
                # Cleanup biometric registration on vault failure
                biometric_auth_service.remove_user(user_id)
                return {
                    "success": False,
                    "error": "Failed to create secure vault"
                }
            
            logger.info(f"Successfully registered user: {user_id}")
            return {
                "success": True,
                "user_id": user_id,
                "biometric_hash": biometric_result["biometric_hash"],
                "vault_created": True,
                "registration_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def authenticate_user(
        self,
        face_image: bytes,
        voice_audio: bytes,
        pin: str
    ) -> Dict[str, Any]:
        """
        Authenticate user and create secure session.
        
        Args:
            face_image: Current face image
            voice_audio: Current voice sample
            pin: Provided PIN
            
        Returns:
            Authentication result with session token
        """
        try:
            logger.info("Starting user authentication")
            
            # Step 1: Biometric authentication
            auth_result = await biometric_auth_service.authenticate_user(
                face_image, voice_audio, pin
            )
            
            if not auth_result["success"]:
                return auth_result
            
            user_id = auth_result["user_id"]
            
            # Step 2: Open user vault
            vault_key = vault_manager.key_derivation_service.decode_key_b64(
                auth_result["vault_key"]
            )
            
            vault_data = await vault_manager.open_vault(user_id, vault_key)
            
            if vault_data is None:
                return {
                    "success": False,
                    "error": "Failed to open user vault"
                }
            
            # Step 3: Create secure session
            session_token = str(uuid.uuid4())
            session_data = {
                "user_id": user_id,
                "biometric_hash": auth_result["biometric_hash"],
                "vault_key": vault_key,
                "created_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "match_score": auth_result["match_score"]
            }
            
            # Clean up old sessions if at limit
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                await self._cleanup_old_sessions()
            
            self.active_sessions[session_token] = session_data
            
            logger.info(f"User authenticated successfully: {user_id}")
            return {
                "success": True,
                "user_id": user_id,
                "session_token": session_token,
                "vault_stats": vault_manager.get_vault_stats(user_id),
                "match_score": auth_result["match_score"],
                "session_created": session_data["created_at"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate_session(self, session_token: str) -> Optional[Dict]:
        """
        Validate session token and return session data.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            Session data if valid, None otherwise
        """
        try:
            if session_token not in self.active_sessions:
                return None
            
            session_data = self.active_sessions[session_token]
            
            # Check session timeout
            now = datetime.now(timezone.utc)
            last_activity = session_data["last_activity"]
            
            if (now - last_activity).total_seconds() > self.session_timeout:
                await self.logout_user(session_token)
                return None
            
            # Update last activity
            session_data["last_activity"] = now
            
            return session_data
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    async def logout_user(self, session_token: str) -> bool:
        """
        Logout user and cleanup session.
        
        Args:
            session_token: Session token to logout
            
        Returns:
            True if logout successful
        """
        try:
            if session_token not in self.active_sessions:
                return True  # Already logged out
            
            session_data = self.active_sessions[session_token]
            user_id = session_data["user_id"]
            
            # Close user vault
            await vault_manager.close_vault(user_id, save=True)
            
            # Remove session
            del self.active_sessions[session_token]
            
            logger.info(f"User logged out: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    async def save_conversation_message(
        self,
        session_token: str,
        conversation_id: str,
        message: Dict
    ) -> bool:
        """
        Save message to user's encrypted vault.
        
        Args:
            session_token: Active session token
            conversation_id: Conversation identifier
            message: Message data to save
            
        Returns:
            True if saved successfully
        """
        try:
            session_data = await self.validate_session(session_token)
            if not session_data:
                return False
            
            user_id = session_data["user_id"]
            
            # Add message to vault
            success = await vault_manager.add_message(
                user_id, conversation_id, message
            )
            
            if success:
                # Auto-save vault periodically
                await vault_manager.save_vault(user_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    async def create_conversation(
        self,
        session_token: str,
        conversation_id: str,
        title: str
    ) -> bool:
        """Create new conversation in user's vault."""
        try:
            session_data = await self.validate_session(session_token)
            if not session_data:
                return False
            
            user_id = session_data["user_id"]
            
            return await vault_manager.add_conversation(
                user_id, conversation_id, title
            )
            
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            return False
    
    async def get_user_conversations(self, session_token: str) -> List[Dict]:
        """Get all conversations for authenticated user."""
        try:
            session_data = await self.validate_session(session_token)
            if not session_data:
                return []
            
            user_id = session_data["user_id"]
            return await vault_manager.get_conversations(user_id)
            
        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            return []
    
    async def anonymize_external_query(
        self,
        session_token: str,
        query: str,
        preserve_technical: bool = True
    ) -> Dict[str, Any]:
        """
        Anonymize query for external API usage.
        
        Args:
            session_token: Active session token
            query: Original query
            preserve_technical: Whether to preserve technical terms
            
        Returns:
            Anonymization result
        """
        try:
            session_data = await self.validate_session(session_token)
            if not session_data:
                return {
                    "success": False,
                    "error": "Invalid session"
                }
            
            # Anonymize the query
            result = anonymization_service.anonymize_query(
                query, preserve_technical
            )
            
            # Log anonymization activity
            logger.info(f"Anonymized query for user {session_data['user_id']}: {result['pii_count']} PII items")
            
            return {
                "success": True,
                "anonymized_query": result["anonymized_query"],
                "anonymization_level": result["anonymization_level"],
                "pii_removed": result["pii_count"]
            }
            
        except Exception as e:
            logger.error(f"Query anonymization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_user_preferences(self, session_token: str) -> Dict:
        """Get user preferences from vault."""
        try:
            session_data = await self.validate_session(session_token)
            if not session_data:
                return {}
            
            user_id = session_data["user_id"]
            conversations = await vault_manager.get_conversations(user_id)
            
            # Get preferences from active vault
            if user_id in vault_manager.active_vaults:
                return vault_manager.active_vaults[user_id]["data"].get("preferences", {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return {}
    
    async def update_user_preferences(
        self,
        session_token: str,
        preferences: Dict
    ) -> bool:
        """Update user preferences in vault."""
        try:
            session_data = await self.validate_session(session_token)
            if not session_data:
                return False
            
            user_id = session_data["user_id"]
            return await vault_manager.update_preferences(user_id, preferences)
            
        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")
            return False
    
    async def _cleanup_old_sessions(self):
        """Clean up expired sessions."""
        try:
            now = datetime.now(timezone.utc)
            expired_tokens = []
            
            for token, session_data in self.active_sessions.items():
                last_activity = session_data["last_activity"]
                if (now - last_activity).total_seconds() > self.session_timeout:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                await self.logout_user(token)
            
            if expired_tokens:
                logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security system status."""
        capabilities = biometric_auth_service.get_capabilities()
        anonymization_stats = anonymization_service.get_anonymization_stats()
        
        return {
            "biometric_capabilities": capabilities,
            "active_sessions": len(self.active_sessions),
            "registered_users": len(biometric_auth_service.get_registered_users()),
            "anonymization_stats": anonymization_stats,
            "vault_directory": str(vault_manager.vault_directory),
            "session_timeout": self.session_timeout,
            "security_features": {
                "biometric_auth": capabilities["authentication"],
                "encrypted_vaults": True,
                "query_anonymization": True,
                "session_management": True
            }
        }


# Global service instance
security_service = SecurityService()