"""
Key Derivation Service.

Securely derives encryption keys from biometric data and PIN combinations.
"""

import hashlib
import secrets
from typing import Optional, Tuple
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)


class KeyDerivationService:
    """Service for secure key derivation from biometric data."""
    
    def __init__(self):
        self.iterations = 100000  # PBKDF2 iterations
        self.key_length = 32  # 256-bit keys
        self.salt_length = 32  # 256-bit salts
        
    def derive_biometric_hash(self, face_encoding: bytes, voice_print: bytes) -> str:
        """
        Create a stable hash from biometric data.
        
        Args:
            face_encoding: Face recognition encoding data
            voice_print: Voice recognition print data
            
        Returns:
            Stable biometric hash as hex string
        """
        try:
            # Combine biometric data
            combined_data = face_encoding + voice_print
            
            # Create stable hash using SHA-256
            biometric_hash = hashlib.sha256(combined_data).hexdigest()
            
            logger.debug("Generated biometric hash")
            return biometric_hash
            
        except Exception as e:
            logger.error(f"Failed to derive biometric hash: {e}")
            raise
    
    def generate_salt(self) -> bytes:
        """Generate a cryptographically secure random salt."""
        return secrets.token_bytes(self.salt_length)
    
    def derive_vault_key(
        self, 
        biometric_hash: str, 
        pin: str, 
        salt: bytes
    ) -> bytes:
        """
        Derive encryption key from biometric hash + PIN + salt.
        
        Args:
            biometric_hash: Stable biometric identifier
            pin: User-provided PIN
            salt: Random salt for key derivation
            
        Returns:
            Derived encryption key
        """
        try:
            # Combine biometric hash and PIN
            password = (biometric_hash + pin).encode('utf-8')
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.key_length,
                salt=salt,
                iterations=self.iterations,
                backend=default_backend()
            )
            
            key = kdf.derive(password)
            logger.debug("Successfully derived vault key")
            return key
            
        except Exception as e:
            logger.error(f"Failed to derive vault key: {e}")
            raise
    
    def verify_key_derivation(
        self,
        key: bytes,
        biometric_hash: str,
        pin: str,
        salt: bytes
    ) -> bool:
        """
        Verify that a key was derived from the given parameters.
        
        Args:
            key: Key to verify
            biometric_hash: Biometric identifier used
            pin: PIN used
            salt: Salt used
            
        Returns:
            True if key matches derived key
        """
        try:
            # Derive key with same parameters
            expected_key = self.derive_vault_key(biometric_hash, pin, salt)
            
            # Constant-time comparison
            return secrets.compare_digest(key, expected_key)
            
        except Exception as e:
            logger.error(f"Key verification failed: {e}")
            return False
    
    def create_user_credentials(
        self,
        face_encoding: bytes,
        voice_print: bytes,
        pin: str
    ) -> Tuple[str, bytes, bytes]:
        """
        Create complete user credentials for vault access.
        
        Args:
            face_encoding: User's face encoding
            voice_print: User's voice print
            pin: User's chosen PIN
            
        Returns:
            Tuple of (biometric_hash, salt, vault_key)
        """
        try:
            # Generate biometric hash
            biometric_hash = self.derive_biometric_hash(face_encoding, voice_print)
            
            # Generate random salt
            salt = self.generate_salt()
            
            # Derive vault key
            vault_key = self.derive_vault_key(biometric_hash, pin, salt)
            
            logger.info("Created user credentials successfully")
            return biometric_hash, salt, vault_key
            
        except Exception as e:
            logger.error(f"Failed to create user credentials: {e}")
            raise
    
    def authenticate_user(
        self,
        face_encoding: bytes,
        voice_print: bytes,
        pin: str,
        stored_salt: bytes
    ) -> Optional[bytes]:
        """
        Authenticate user and return vault key if successful.
        
        Args:
            face_encoding: Current face encoding
            voice_print: Current voice print
            pin: Provided PIN
            stored_salt: Salt from user registration
            
        Returns:
            Vault key if authentication successful, None otherwise
        """
        try:
            # Generate biometric hash from current data
            biometric_hash = self.derive_biometric_hash(face_encoding, voice_print)
            
            # Derive vault key
            vault_key = self.derive_vault_key(biometric_hash, pin, stored_salt)
            
            logger.debug("User authentication completed")
            return vault_key
            
        except Exception as e:
            logger.error(f"User authentication failed: {e}")
            return None
    
    def encode_key_b64(self, key: bytes) -> str:
        """Encode key as base64 string for storage."""
        return base64.b64encode(key).decode('utf-8')
    
    def decode_key_b64(self, key_str: str) -> bytes:
        """Decode base64 key string to bytes."""
        return base64.b64decode(key_str.encode('utf-8'))


# Global service instance
key_derivation_service = KeyDerivationService()