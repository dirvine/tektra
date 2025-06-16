"""
Basic security tests for core functionality.

Tests the essential security components without requiring heavy dependencies.
"""

import pytest
import secrets
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

def test_key_derivation_basic():
    """Test basic key derivation functionality."""
    from tektra.app.security.key_derivation import KeyDerivationService
    
    service = KeyDerivationService()
    
    # Test salt generation
    salt = service.generate_salt()
    assert len(salt) == 32
    assert isinstance(salt, bytes)
    
    # Test biometric hash generation
    face_data = b"test_face_data"
    voice_data = b"test_voice_data"
    hash1 = service.derive_biometric_hash(face_data, voice_data)
    hash2 = service.derive_biometric_hash(face_data, voice_data)
    
    assert hash1 == hash2  # Deterministic
    assert len(hash1) == 64  # SHA-256 hex
    
    # Test vault key derivation
    vault_key = service.derive_vault_key(hash1, "1234", salt)
    assert len(vault_key) == 32
    assert isinstance(vault_key, bytes)


def test_anonymization_basic():
    """Test basic anonymization functionality."""
    from tektra.app.security.anonymization import AnonymizationService
    
    service = AnonymizationService()
    
    # Test email anonymization
    text = "Contact me at john.doe@example.com for details."
    anonymized, detected = service._anonymize_text(text)
    
    assert "john.doe@example.com" not in anonymized
    assert len(detected) >= 1
    assert any(d["category"] == "email" for d in detected)
    
    # Test phone anonymization
    text = "Call me at +1-555-123-4567."
    anonymized, detected = service._anonymize_text(text)
    
    assert "+1-555-123-4567" not in anonymized
    assert any(d["category"] == "phone" for d in detected)
    
    # Test query anonymization
    query = "Send results to admin@company.com and call 555-123-4567"
    result = service.anonymize_query(query)
    
    assert result["pii_count"] >= 2
    assert "admin@company.com" not in result["anonymized_query"]
    assert "555-123-4567" not in result["anonymized_query"]


@pytest.mark.asyncio
async def test_vault_manager_basic():
    """Test basic vault management functionality."""
    from tektra.app.security.vault_manager import VaultManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_mgr = VaultManager()
        vault_mgr.vault_directory = Path(temp_dir) / "vaults"
        await vault_mgr.initialize()
        
        # Test encryption/decryption
        data = b"test sensitive data"
        key = secrets.token_bytes(32)
        
        encrypted = vault_mgr._encrypt_data(data, key)
        assert encrypted != data
        assert len(encrypted) > len(data)
        
        decrypted = vault_mgr._decrypt_data(encrypted, key)
        assert decrypted == data
        
        # Test vault creation
        user_id = "test_user"
        biometric_hash = "test_hash_123"
        vault_key = secrets.token_bytes(32)
        
        success = await vault_mgr.create_vault(user_id, biometric_hash, vault_key)
        assert success is True
        
        # Test vault opening
        vault_data = await vault_mgr.open_vault(user_id, vault_key)
        assert vault_data is not None
        assert vault_data["user_id"] == user_id
        assert vault_data["biometric_hash"] == biometric_hash


def test_biometric_auth_structure():
    """Test biometric auth service structure without dependencies."""
    from tektra.app.security.biometric_auth import BiometricAuthService
    
    service = BiometricAuthService()
    
    # Test initialization parameters
    assert service.face_tolerance == 0.6
    assert service.sample_rate == 16000
    assert service.min_voice_duration == 2.0
    assert service.voice_similarity_threshold == 0.25
    
    # Test capabilities without loaded models
    capabilities = service.get_capabilities()
    assert isinstance(capabilities, dict)
    assert "face_recognition" in capabilities
    assert "voice_recognition" in capabilities
    assert "registration" in capabilities
    assert "authentication" in capabilities
    
    # Test user management
    assert service.get_registered_users() == []
    
    # Add mock user
    service.registered_users["test_user"] = {"test": "data"}
    assert "test_user" in service.get_registered_users()
    
    # Remove user
    assert service.remove_user("test_user") is True
    assert service.remove_user("nonexistent") is False


@pytest.mark.asyncio
async def test_security_service_structure():
    """Test security service structure."""
    from tektra.app.services.security_service import SecurityService
    
    service = SecurityService()
    
    # Test basic configuration
    assert service.session_timeout == 3600
    assert service.max_concurrent_sessions == 10
    assert isinstance(service.active_sessions, dict)
    
    # Test status without initialization
    status = service.get_security_status()
    assert isinstance(status, dict)
    assert "security_features" in status


def test_security_patterns():
    """Test that security patterns are comprehensive."""
    from tektra.app.security.anonymization import AnonymizationService
    
    service = AnonymizationService()
    
    # Check essential PII patterns exist
    required_patterns = [
        "email", "phone", "ssn", "credit_card", 
        "ip_address", "url", "file_path", "api_key"
    ]
    
    for pattern in required_patterns:
        assert pattern in service.pii_patterns
        assert service.pii_patterns[pattern] is not None
    
    # Check context patterns exist
    context_patterns = ["lab_equipment", "project_names", "internal_domains"]
    
    for pattern in context_patterns:
        assert pattern in service.context_patterns
        assert service.context_patterns[pattern] is not None


def test_global_service_instances():
    """Test that global service instances are available."""
    # Test key derivation service
    from tektra.app.security.key_derivation import key_derivation_service
    assert key_derivation_service is not None
    
    # Test anonymization service  
    from tektra.app.security.anonymization import anonymization_service
    assert anonymization_service is not None
    
    # Test vault manager
    from tektra.app.security.vault_manager import vault_manager
    assert vault_manager is not None
    
    # Test biometric auth service
    from tektra.app.security.biometric_auth import biometric_auth_service
    assert biometric_auth_service is not None


def test_cryptographic_security():
    """Test cryptographic security parameters."""
    from tektra.app.security.key_derivation import KeyDerivationService
    from tektra.app.security.vault_manager import VaultManager
    
    # Key derivation security
    kd_service = KeyDerivationService()
    assert kd_service.iterations >= 100000  # PBKDF2 iterations
    assert kd_service.key_length >= 32       # 256-bit keys minimum
    assert kd_service.salt_length >= 16      # 128-bit salts minimum
    
    # Vault encryption security
    vault_mgr = VaultManager()
    assert vault_mgr.key_size == 32          # 256-bit encryption
    assert vault_mgr.iv_size == 16           # 128-bit IV
    assert vault_mgr.block_size == 128       # AES block size


@pytest.mark.asyncio 
async def test_security_integration():
    """Test basic security integration workflow."""
    from tektra.app.security.key_derivation import key_derivation_service
    from tektra.app.security.vault_manager import VaultManager
    from tektra.app.security.anonymization import anonymization_service
    
    # Simulate user registration workflow
    user_id = "integration_test_user"
    pin = "1234"
    face_data = b"mock_face_encoding_data"
    voice_data = b"mock_voice_print_data"
    
    # Step 1: Create user credentials
    biometric_hash, salt, vault_key = key_derivation_service.create_user_credentials(
        face_data, voice_data, pin
    )
    
    assert isinstance(biometric_hash, str)
    assert isinstance(salt, bytes)
    assert isinstance(vault_key, bytes)
    
    # Step 2: Create encrypted vault
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_mgr = VaultManager()
        vault_mgr.vault_directory = Path(temp_dir) / "vaults"
        await vault_mgr.initialize()
        
        vault_created = await vault_mgr.create_vault(user_id, biometric_hash, vault_key)
        assert vault_created is True
        
        # Step 3: Open vault and add data
        vault_data = await vault_mgr.open_vault(user_id, vault_key)
        assert vault_data is not None
        
        # Step 4: Test anonymization
        sensitive_query = "Send data to user@company.com at IP 192.168.1.100 and call 555-123-4567"
        result = anonymization_service.anonymize_query(sensitive_query)
        
        assert result["pii_count"] >= 3
        assert "user@company.com" not in result["anonymized_query"]
        assert "192.168.1.100" not in result["anonymized_query"]
        assert "555-123-4567" not in result["anonymized_query"]
        
        # Step 5: Add conversation to vault
        conv_success = await vault_mgr.add_conversation(
            user_id, "test_conv", "Integration Test"
        )
        assert conv_success is True
        
        # Step 6: Add message to conversation
        message = {
            "id": "msg_1",
            "role": "user", 
            "content": result["anonymized_query"],
            "type": "text"
        }
        
        msg_success = await vault_mgr.add_message(user_id, "test_conv", message)
        assert msg_success is True
        
        # Step 7: Verify data integrity
        conversations = await vault_mgr.get_conversations(user_id)
        assert len(conversations) == 1
        assert len(conversations[0]["messages"]) == 1
        
        stored_message = conversations[0]["messages"][0]
        assert stored_message["content"] == result["anonymized_query"]
        
        # Step 8: Close vault
        close_success = await vault_mgr.close_vault(user_id, save=True)
        assert close_success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])