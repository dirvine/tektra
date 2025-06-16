"""
Tests for Vault Manager Service.

Tests encrypted vault creation, management, conversation storage,
and security features with comprehensive scenarios.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
import secrets

from tektra.app.security.vault_manager import VaultManager, vault_manager


class TestVaultManager:
    """Test cases for VaultManager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, temp_dir):
        """Test vault manager initialization."""
        vault_mgr = VaultManager()
        vault_mgr.vault_directory = temp_dir / "vaults"
        
        await vault_mgr.initialize()
        
        assert vault_mgr.vault_directory.exists()
        assert vault_mgr.vault_extension == ".vault"
        assert vault_mgr.backup_extension == ".vault.bak"
    
    def test_vault_path_generation(self, vault_manager, temp_dir):
        """Test vault file path generation."""
        vault_manager.vault_directory = temp_dir
        
        # Test normal user ID
        path = vault_manager._get_vault_path("test_user")
        assert path == temp_dir / "test_user.vault"
        
        # Test user ID with special characters
        path = vault_manager._get_vault_path("user@domain.com")
        assert path == temp_dir / "userdomain.com.vault"
        
        # Test backup path
        backup_path = vault_manager._get_backup_path("test_user")
        assert backup_path == temp_dir / "test_user.vault.bak"
    
    def test_encryption_decryption(self, vault_manager):
        """Test data encryption and decryption."""
        data = b"This is test data for encryption"
        key = secrets.token_bytes(32)
        
        # Encrypt data
        encrypted = vault_manager._encrypt_data(data, key)
        assert encrypted != data
        assert len(encrypted) > len(data)  # IV + padding
        
        # Decrypt data
        decrypted = vault_manager._decrypt_data(encrypted, key)
        assert decrypted == data
    
    def test_encryption_with_different_keys(self, vault_manager):
        """Test that different keys produce different ciphertext."""
        data = b"Same data, different keys"
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        
        encrypted1 = vault_manager._encrypt_data(data, key1)
        encrypted2 = vault_manager._encrypt_data(data, key2)
        
        assert encrypted1 != encrypted2
        
        # Verify each can be decrypted with correct key
        assert vault_manager._decrypt_data(encrypted1, key1) == data
        assert vault_manager._decrypt_data(encrypted2, key2) == data
    
    def test_encryption_deterministic_iv(self, vault_manager):
        """Test that encryption uses different IVs each time."""
        data = b"Same data, different encryptions"
        key = secrets.token_bytes(32)
        
        encrypted1 = vault_manager._encrypt_data(data, key)
        encrypted2 = vault_manager._encrypt_data(data, key)
        
        # Should be different due to random IV
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same data
        assert vault_manager._decrypt_data(encrypted1, key) == data
        assert vault_manager._decrypt_data(encrypted2, key) == data
    
    @pytest.mark.asyncio
    async def test_create_vault_success(self, vault_manager, temp_dir):
        """Test successful vault creation."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        biometric_hash = "test_hash_123"
        vault_key = secrets.token_bytes(32)
        
        success = await vault_manager.create_vault(user_id, biometric_hash, vault_key)
        
        assert success is True
        
        # Check vault file exists
        vault_path = vault_manager._get_vault_path(user_id)
        assert vault_path.exists()
        
        # Verify we can open the vault
        vault_data = await vault_manager.open_vault(user_id, vault_key)
        assert vault_data is not None
        assert vault_data["user_id"] == user_id
        assert vault_data["biometric_hash"] == biometric_hash
        assert vault_data["version"] == "1.0"
    
    @pytest.mark.asyncio
    async def test_create_vault_already_exists(self, vault_manager, temp_dir):
        """Test vault creation when vault already exists."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault first time
        success1 = await vault_manager.create_vault(user_id, "hash1", vault_key)
        assert success1 is True
        
        # Try to create again
        success2 = await vault_manager.create_vault(user_id, "hash2", vault_key)
        assert success2 is False
    
    @pytest.mark.asyncio
    async def test_open_vault_success(self, vault_manager, temp_dir):
        """Test successful vault opening."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        
        # Open vault
        vault_data = await vault_manager.open_vault(user_id, vault_key)
        
        assert vault_data is not None
        assert vault_data["user_id"] == user_id
        assert "last_access" in vault_data["metadata"]
        
        # Check vault is cached
        assert user_id in vault_manager.active_vaults
    
    @pytest.mark.asyncio
    async def test_open_vault_nonexistent(self, vault_manager, temp_dir):
        """Test opening a vault that doesn't exist."""
        vault_manager.vault_directory = temp_dir
        
        vault_data = await vault_manager.open_vault("nonexistent_user", secrets.token_bytes(32))
        assert vault_data is None
    
    @pytest.mark.asyncio
    async def test_open_vault_wrong_key(self, vault_manager, temp_dir):
        """Test opening vault with wrong key."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        correct_key = secrets.token_bytes(32)
        wrong_key = secrets.token_bytes(32)
        
        # Create vault
        await vault_manager.create_vault(user_id, "test_hash", correct_key)
        
        # Try to open with wrong key
        with pytest.raises(Exception):  # Should raise decryption error
            await vault_manager.open_vault(user_id, wrong_key)
    
    @pytest.mark.asyncio
    async def test_save_vault(self, vault_manager, temp_dir):
        """Test saving vault data."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create and open vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        vault_data = await vault_manager.open_vault(user_id, vault_key)
        
        # Modify vault data
        original_updated = vault_data["updated_at"]
        vault_data["test_field"] = "test_value"
        vault_manager.active_vaults[user_id]["modified"] = True
        
        # Save vault
        success = await vault_manager.save_vault(user_id)
        assert success is True
        
        # Verify changes persisted
        vault_data_reloaded = await vault_manager.open_vault(user_id, vault_key)
        assert vault_data_reloaded["test_field"] == "test_value"
        assert vault_data_reloaded["updated_at"] != original_updated
    
    @pytest.mark.asyncio
    async def test_save_vault_with_backup(self, vault_manager, temp_dir):
        """Test that saving creates backup of existing vault."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        
        # Open and modify
        await vault_manager.open_vault(user_id, vault_key)
        vault_manager.active_vaults[user_id]["data"]["test"] = "original"
        vault_manager.active_vaults[user_id]["modified"] = True
        await vault_manager.save_vault(user_id)
        
        # Modify again
        vault_manager.active_vaults[user_id]["data"]["test"] = "modified"
        vault_manager.active_vaults[user_id]["modified"] = True
        await vault_manager.save_vault(user_id)
        
        # Check backup exists
        backup_path = vault_manager._get_backup_path(user_id)
        assert backup_path.exists()
    
    @pytest.mark.asyncio
    async def test_close_vault(self, vault_manager, temp_dir):
        """Test closing vault."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create and open vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Verify vault is active
        assert user_id in vault_manager.active_vaults
        
        # Close vault
        success = await vault_manager.close_vault(user_id)
        assert success is True
        
        # Verify vault is no longer active
        assert user_id not in vault_manager.active_vaults
    
    @pytest.mark.asyncio
    async def test_close_vault_with_save(self, vault_manager, temp_dir):
        """Test closing vault with automatic save."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create and open vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Mark as modified
        vault_manager.active_vaults[user_id]["modified"] = True
        vault_manager.active_vaults[user_id]["data"]["test"] = "modified"
        
        # Close with save
        success = await vault_manager.close_vault(user_id, save=True)
        assert success is True
        
        # Verify changes were saved
        vault_data = await vault_manager.open_vault(user_id, vault_key)
        assert vault_data["test"] == "modified"
    
    @pytest.mark.asyncio
    async def test_add_conversation(self, vault_manager, temp_dir):
        """Test adding conversation to vault."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create and open vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Add conversation
        conv_id = "conv_123"
        title = "Test Conversation"
        success = await vault_manager.add_conversation(user_id, conv_id, title)
        
        assert success is True
        
        # Verify conversation added
        vault_data = vault_manager.active_vaults[user_id]["data"]
        assert len(vault_data["conversations"]) == 1
        
        conversation = vault_data["conversations"][0]
        assert conversation["id"] == conv_id
        assert conversation["title"] == title
        assert "created_at" in conversation
        assert conversation["messages"] == []
        
        # Check metadata updated
        assert vault_data["metadata"]["total_conversations"] == 1
    
    @pytest.mark.asyncio
    async def test_add_message(self, vault_manager, temp_dir):
        """Test adding message to conversation."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        conv_id = "conv_123"
        
        # Create vault and conversation
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        await vault_manager.add_conversation(user_id, conv_id, "Test Conv")
        
        # Add message
        message = {
            "id": "msg_1",
            "role": "user",
            "content": "Hello, world!",
            "type": "text"
        }
        
        success = await vault_manager.add_message(user_id, conv_id, message)
        assert success is True
        
        # Verify message added
        vault_data = vault_manager.active_vaults[user_id]["data"]
        conversation = vault_data["conversations"][0]
        
        assert len(conversation["messages"]) == 1
        stored_message = conversation["messages"][0]
        
        assert stored_message["id"] == "msg_1"
        assert stored_message["content"] == "Hello, world!"
        assert "timestamp" in stored_message
        
        # Check metadata updated
        assert vault_data["metadata"]["total_messages"] == 1
    
    @pytest.mark.asyncio
    async def test_add_message_nonexistent_conversation(self, vault_manager, temp_dir):
        """Test adding message to nonexistent conversation."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault without conversation
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Try to add message
        message = {"id": "msg_1", "content": "test"}
        success = await vault_manager.add_message(user_id, "nonexistent_conv", message)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_conversations(self, vault_manager, temp_dir):
        """Test retrieving conversations."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Add multiple conversations
        conv_ids = ["conv_1", "conv_2", "conv_3"]
        for conv_id in conv_ids:
            await vault_manager.add_conversation(user_id, conv_id, f"Title {conv_id}")
        
        # Get conversations
        conversations = await vault_manager.get_conversations(user_id)
        
        assert len(conversations) == 3
        retrieved_ids = [conv["id"] for conv in conversations]
        assert set(retrieved_ids) == set(conv_ids)
    
    @pytest.mark.asyncio
    async def test_get_conversation(self, vault_manager, temp_dir):
        """Test retrieving specific conversation."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        conv_id = "conv_123"
        
        # Create vault and conversation
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        await vault_manager.add_conversation(user_id, conv_id, "Test Conv")
        
        # Get specific conversation
        conversation = await vault_manager.get_conversation(user_id, conv_id)
        
        assert conversation is not None
        assert conversation["id"] == conv_id
        assert conversation["title"] == "Test Conv"
        
        # Test nonexistent conversation
        nonexistent = await vault_manager.get_conversation(user_id, "nonexistent")
        assert nonexistent is None
    
    @pytest.mark.asyncio
    async def test_update_preferences(self, vault_manager, temp_dir):
        """Test updating user preferences."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Update preferences
        preferences = {
            "theme": "dark",
            "language": "en",
            "notifications": True
        }
        
        success = await vault_manager.update_preferences(user_id, preferences)
        assert success is True
        
        # Verify preferences updated
        vault_data = vault_manager.active_vaults[user_id]["data"]
        assert vault_data["preferences"]["theme"] == "dark"
        assert vault_data["preferences"]["language"] == "en"
        assert vault_data["preferences"]["notifications"] is True
    
    @pytest.mark.asyncio
    async def test_export_vault(self, vault_manager, temp_dir):
        """Test exporting vault data."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault with some data
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        await vault_manager.add_conversation(user_id, "conv_1", "Test Conv")
        
        # Export vault
        export_path = temp_dir / "vault_export.json"
        success = await vault_manager.export_vault(user_id, export_path, include_key=False)
        
        assert success is True
        assert export_path.exists()
        
        # Verify export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        assert "vault_data" in export_data
        assert "exported_at" in export_data
        assert export_data["include_key"] is False
        assert "vault_key" not in export_data
        
        # Test export with key
        export_path_with_key = temp_dir / "vault_export_with_key.json"
        success = await vault_manager.export_vault(user_id, export_path_with_key, include_key=True)
        
        assert success is True
        
        with open(export_path_with_key, 'r') as f:
            export_data_with_key = json.load(f)
        
        assert export_data_with_key["include_key"] is True
        assert "vault_key" in export_data_with_key
    
    def test_get_vault_stats(self, vault_manager):
        """Test getting vault statistics."""
        user_id = "test_user"
        
        # No active vault
        stats = vault_manager.get_vault_stats(user_id)
        assert stats is None
        
        # Create mock active vault
        vault_manager.active_vaults[user_id] = {
            "data": {
                "metadata": {
                    "total_conversations": 5,
                    "total_messages": 42,
                    "last_access": "2024-01-01T12:00:00Z"
                }
            }
        }
        
        stats = vault_manager.get_vault_stats(user_id)
        assert stats["total_conversations"] == 5
        assert stats["total_messages"] == 42
        assert stats["last_access"] == "2024-01-01T12:00:00Z"
    
    @pytest.mark.asyncio
    async def test_vault_error_handling(self, vault_manager, temp_dir):
        """Test error handling in vault operations."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        
        # Test operations on inactive vault
        success = await vault_manager.save_vault(user_id)
        assert success is False
        
        success = await vault_manager.add_conversation(user_id, "conv", "title")
        assert success is False
        
        success = await vault_manager.add_message(user_id, "conv", {"msg": "test"})
        assert success is False
        
        success = await vault_manager.update_preferences(user_id, {"pref": "value"})
        assert success is False
    
    @pytest.mark.asyncio
    async def test_concurrent_vault_operations(self, vault_manager, temp_dir):
        """Test concurrent operations on same vault."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Perform concurrent operations
        tasks = []
        for i in range(10):
            task = vault_manager.add_conversation(user_id, f"conv_{i}", f"Title {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert all(result is True for result in results)
        
        # Verify all conversations added
        conversations = await vault_manager.get_conversations(user_id)
        assert len(conversations) == 10
    
    @pytest.mark.asyncio
    async def test_vault_data_integrity(self, vault_manager, temp_dir):
        """Test vault data integrity across operations."""
        vault_manager.vault_directory = temp_dir
        
        user_id = "test_user"
        vault_key = secrets.token_bytes(32)
        
        # Create vault and add data
        await vault_manager.create_vault(user_id, "test_hash", vault_key)
        await vault_manager.open_vault(user_id, vault_key)
        
        # Add conversation and messages
        conv_id = "conv_test"
        await vault_manager.add_conversation(user_id, conv_id, "Test Conversation")
        
        messages = [
            {"id": "msg_1", "role": "user", "content": "Hello"},
            {"id": "msg_2", "role": "assistant", "content": "Hi there"},
            {"id": "msg_3", "role": "user", "content": "How are you?"}
        ]
        
        for message in messages:
            await vault_manager.add_message(user_id, conv_id, message)
        
        # Save and reload vault
        await vault_manager.save_vault(user_id)
        await vault_manager.close_vault(user_id)
        
        # Reopen and verify data integrity
        vault_data = await vault_manager.open_vault(user_id, vault_key)
        
        assert vault_data["metadata"]["total_conversations"] == 1
        assert vault_data["metadata"]["total_messages"] == 3
        
        conversation = vault_data["conversations"][0]
        assert len(conversation["messages"]) == 3
        
        # Verify message order and content
        for i, original_msg in enumerate(messages):
            stored_msg = conversation["messages"][i]
            assert stored_msg["id"] == original_msg["id"]
            assert stored_msg["content"] == original_msg["content"]


class TestGlobalService:
    """Test the global vault manager instance."""
    
    def test_global_service_exists(self):
        """Test that global service instance exists."""
        assert vault_manager is not None
        assert isinstance(vault_manager, VaultManager)
    
    def test_global_service_configuration(self):
        """Test that global service is properly configured."""
        assert vault_manager.vault_extension == ".vault"
        assert vault_manager.backup_extension == ".vault.bak"
        assert vault_manager.algorithm is not None
        assert vault_manager.key_size == 32
        assert vault_manager.iv_size == 16


class TestVaultSecurity:
    """Test security aspects of vault management."""
    
    def test_encryption_key_requirements(self, vault_manager):
        """Test that encryption requires proper key size."""
        data = b"test data"
        
        # Test with wrong key size
        with pytest.raises(Exception):
            vault_manager._encrypt_data(data, b"short_key")
    
    def test_decryption_wrong_key_fails(self, vault_manager):
        """Test that decryption fails with wrong key."""
        data = b"sensitive data"
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        
        encrypted = vault_manager._encrypt_data(data, key1)
        
        # Try to decrypt with wrong key
        with pytest.raises(Exception):
            vault_manager._decrypt_data(encrypted, key2)
    
    def test_encrypted_data_format(self, vault_manager):
        """Test encrypted data format includes IV."""
        data = b"test data"
        key = secrets.token_bytes(32)
        
        encrypted = vault_manager._encrypt_data(data, key)
        
        # Should be IV (16 bytes) + ciphertext + padding
        assert len(encrypted) >= 16 + len(data)
        assert len(encrypted) % 16 == 0  # AES block alignment
    
    @pytest.mark.asyncio
    async def test_vault_isolation(self, vault_manager, temp_dir):
        """Test that user vaults are isolated from each other."""
        vault_manager.vault_directory = temp_dir
        
        # Create two users with different keys
        user1_key = secrets.token_bytes(32)
        user2_key = secrets.token_bytes(32)
        
        await vault_manager.create_vault("user1", "hash1", user1_key)
        await vault_manager.create_vault("user2", "hash2", user2_key)
        
        # Add data to each vault
        await vault_manager.open_vault("user1", user1_key)
        await vault_manager.add_conversation("user1", "conv1", "User 1 Conversation")
        
        await vault_manager.open_vault("user2", user2_key)
        await vault_manager.add_conversation("user2", "conv2", "User 2 Conversation")
        
        # Verify isolation
        user1_convs = await vault_manager.get_conversations("user1")
        user2_convs = await vault_manager.get_conversations("user2")
        
        assert len(user1_convs) == 1
        assert len(user2_convs) == 1
        assert user1_convs[0]["title"] == "User 1 Conversation"
        assert user2_convs[0]["title"] == "User 2 Conversation"
        
        # User 1 cannot access User 2's vault with User 1's key
        with pytest.raises(Exception):
            await vault_manager.open_vault("user2", user1_key)