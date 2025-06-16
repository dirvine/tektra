"""
Vault Manager Service.

Manages encrypted user vaults for secure conversation and data storage.
"""

import asyncio
import base64
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import aiofiles.os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from ..config import settings
from .key_derivation import key_derivation_service

logger = logging.getLogger(__name__)


class VaultManager:
    """Manages encrypted user vaults for secure data storage."""

    def __init__(self):
        self.vault_directory = Path(settings.data_dir) / "vaults"
        self.vault_extension = ".vault"
        self.backup_extension = ".vault.bak"

        # Encryption settings
        self.algorithm = algorithms.AES
        self.key_size = 32  # 256-bit
        self.iv_size = 16  # 128-bit
        self.block_size = 128  # AES block size

        # Vault structure version for future upgrades
        self.vault_version = "1.0"

        # Active vaults cache
        self.active_vaults: Dict[str, Dict] = {}

    async def initialize(self):
        """Initialize vault storage directory."""
        try:
            await aiofiles.os.makedirs(self.vault_directory, exist_ok=True)
            logger.info(f"Vault directory initialized: {self.vault_directory}")

        except Exception as e:
            logger.error(f"Failed to initialize vault directory: {e}")
            raise

    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """
        Encrypt data using AES-256-CBC.

        Args:
            data: Data to encrypt
            key: Encryption key

        Returns:
            Encrypted data with IV prepended
        """
        try:
            # Generate random IV
            iv = secrets.token_bytes(self.iv_size)

            # Pad data to block size
            padder = padding.PKCS7(self.block_size).padder()
            padded_data = padder.update(data) + padder.finalize()

            # Encrypt
            cipher = Cipher(
                self.algorithm(key), modes.CBC(iv), backend=default_backend()
            )
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

            # Prepend IV to encrypted data
            return iv + encrypted_data

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def _decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        Decrypt data using AES-256-CBC.

        Args:
            encrypted_data: Data to decrypt (with IV prepended)
            key: Decryption key

        Returns:
            Decrypted data
        """
        try:
            # Extract IV and encrypted data
            iv = encrypted_data[: self.iv_size]
            ciphertext = encrypted_data[self.iv_size :]

            # Decrypt
            cipher = Cipher(
                self.algorithm(key), modes.CBC(iv), backend=default_backend()
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()

            # Remove padding
            unpadder = padding.PKCS7(self.block_size).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()

            return data

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def _get_vault_path(self, user_id: str) -> Path:
        """Get vault file path for user."""
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_.")
        return self.vault_directory / f"{safe_user_id}{self.vault_extension}"

    def _get_backup_path(self, user_id: str) -> Path:
        """Get backup vault file path for user."""
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_.")
        return self.vault_directory / f"{safe_user_id}{self.backup_extension}"

    async def create_vault(
        self, user_id: str, biometric_hash: str, vault_key: bytes
    ) -> bool:
        """
        Create a new encrypted vault for user.

        Args:
            user_id: User identifier
            biometric_hash: User's biometric hash
            vault_key: Encryption key for vault

        Returns:
            True if vault created successfully
        """
        try:
            vault_path = self._get_vault_path(user_id)

            # Check if vault already exists
            if await aiofiles.os.path.exists(vault_path):
                logger.warning(f"Vault already exists for user: {user_id}")
                return False

            # Create initial vault data
            vault_data = {
                "version": self.vault_version,
                "user_id": user_id,
                "biometric_hash": biometric_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "conversations": [],
                "preferences": {},
                "knowledge_base": {},
                "metadata": {
                    "total_conversations": 0,
                    "total_messages": 0,
                    "last_access": None,
                },
            }

            # Serialize and encrypt
            vault_json = json.dumps(vault_data, indent=2).encode("utf-8")
            encrypted_vault = self._encrypt_data(vault_json, vault_key)

            # Write to file
            async with aiofiles.open(vault_path, "wb") as f:
                await f.write(encrypted_vault)

            logger.info(f"Created vault for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create vault for {user_id}: {e}")
            return False

    async def open_vault(self, user_id: str, vault_key: bytes) -> Optional[Dict]:
        """
        Open and decrypt user vault.

        Args:
            user_id: User identifier
            vault_key: Decryption key

        Returns:
            Vault data if successful, None otherwise
        """
        try:
            vault_path = self._get_vault_path(user_id)

            if not await aiofiles.os.path.exists(vault_path):
                logger.warning(f"Vault does not exist for user: {user_id}")
                return None

            # Read encrypted vault
            async with aiofiles.open(vault_path, "rb") as f:
                encrypted_vault = await f.read()

            # Decrypt and deserialize
            vault_json = self._decrypt_data(encrypted_vault, vault_key)
            vault_data = json.loads(vault_json.decode("utf-8"))

            # Update last access time
            vault_data["metadata"]["last_access"] = datetime.now(
                timezone.utc
            ).isoformat()
            vault_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Cache active vault
            self.active_vaults[user_id] = {
                "data": vault_data,
                "key": vault_key,
                "modified": False,
            }

            logger.info(f"Opened vault for user: {user_id}")
            return vault_data

        except Exception as e:
            logger.error(f"Failed to open vault for {user_id}: {e}")
            return None

    async def save_vault(self, user_id: str) -> bool:
        """
        Save vault data to disk.

        Args:
            user_id: User identifier

        Returns:
            True if saved successfully
        """
        try:
            if user_id not in self.active_vaults:
                logger.error(f"No active vault for user: {user_id}")
                return False

            vault_info = self.active_vaults[user_id]
            vault_data = vault_info["data"]
            vault_key = vault_info["key"]

            # Update timestamp
            vault_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Create backup of existing vault
            vault_path = self._get_vault_path(user_id)
            backup_path = self._get_backup_path(user_id)

            if await aiofiles.os.path.exists(vault_path):
                async with (
                    aiofiles.open(vault_path, "rb") as src,
                    aiofiles.open(backup_path, "wb") as dst,
                ):
                    content = await src.read()
                    await dst.write(content)

            # Serialize and encrypt
            vault_json = json.dumps(vault_data, indent=2).encode("utf-8")
            encrypted_vault = self._encrypt_data(vault_json, vault_key)

            # Write to file
            async with aiofiles.open(vault_path, "wb") as f:
                await f.write(encrypted_vault)

            # Mark as not modified
            vault_info["modified"] = False

            logger.debug(f"Saved vault for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save vault for {user_id}: {e}")
            return False

    async def close_vault(self, user_id: str, save: bool = True) -> bool:
        """
        Close and optionally save vault.

        Args:
            user_id: User identifier
            save: Whether to save before closing

        Returns:
            True if closed successfully
        """
        try:
            if user_id not in self.active_vaults:
                return True  # Already closed

            if save and self.active_vaults[user_id]["modified"]:
                await self.save_vault(user_id)

            # Clear from cache (secure memory clearing would be better)
            del self.active_vaults[user_id]

            logger.info(f"Closed vault for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to close vault for {user_id}: {e}")
            return False

    async def add_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str,
        messages: List[Dict] = None,
    ) -> bool:
        """
        Add a new conversation to user's vault.

        Args:
            user_id: User identifier
            conversation_id: Unique conversation identifier
            title: Conversation title
            messages: Initial messages

        Returns:
            True if added successfully
        """
        try:
            if user_id not in self.active_vaults:
                logger.error(f"No active vault for user: {user_id}")
                return False

            vault_data = self.active_vaults[user_id]["data"]

            conversation = {
                "id": conversation_id,
                "title": title,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "messages": messages or [],
            }

            vault_data["conversations"].append(conversation)
            vault_data["metadata"]["total_conversations"] += 1

            self.active_vaults[user_id]["modified"] = True

            logger.debug(f"Added conversation {conversation_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add conversation: {e}")
            return False

    async def add_message(
        self, user_id: str, conversation_id: str, message: Dict
    ) -> bool:
        """
        Add a message to a conversation.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            message: Message data

        Returns:
            True if added successfully
        """
        try:
            if user_id not in self.active_vaults:
                logger.error(f"No active vault for user: {user_id}")
                return False

            vault_data = self.active_vaults[user_id]["data"]

            # Find conversation
            conversation = None
            for conv in vault_data["conversations"]:
                if conv["id"] == conversation_id:
                    conversation = conv
                    break

            if not conversation:
                logger.error(f"Conversation {conversation_id} not found")
                return False

            # Add message
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
            conversation["messages"].append(message)
            conversation["updated_at"] = datetime.now(timezone.utc).isoformat()

            vault_data["metadata"]["total_messages"] += 1

            self.active_vaults[user_id]["modified"] = True

            return True

        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return False

    async def get_conversations(self, user_id: str) -> List[Dict]:
        """Get all conversations for user."""
        if user_id not in self.active_vaults:
            return []

        return self.active_vaults[user_id]["data"]["conversations"]

    async def get_conversation(
        self, user_id: str, conversation_id: str
    ) -> Optional[Dict]:
        """Get specific conversation."""
        conversations = await self.get_conversations(user_id)

        for conv in conversations:
            if conv["id"] == conversation_id:
                return conv

        return None

    async def update_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences."""
        try:
            if user_id not in self.active_vaults:
                logger.error(f"No active vault for user: {user_id}")
                return False

            vault_data = self.active_vaults[user_id]["data"]
            vault_data["preferences"].update(preferences)

            self.active_vaults[user_id]["modified"] = True

            return True

        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")
            return False

    async def export_vault(
        self, user_id: str, export_path: Path, include_key: bool = False
    ) -> bool:
        """
        Export vault data to file.

        Args:
            user_id: User identifier
            export_path: Export file path
            include_key: Whether to include decryption key

        Returns:
            True if exported successfully
        """
        try:
            if user_id not in self.active_vaults:
                logger.error(f"No active vault for user: {user_id}")
                return False

            vault_info = self.active_vaults[user_id]
            export_data = {
                "vault_data": vault_info["data"],
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "include_key": include_key,
            }

            if include_key:
                export_data["vault_key"] = key_derivation_service.encode_key_b64(
                    vault_info["key"]
                )

            async with aiofiles.open(export_path, "w") as f:
                await f.write(json.dumps(export_data, indent=2))

            logger.info(f"Exported vault for user {user_id} to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export vault: {e}")
            return False

    def get_vault_stats(self, user_id: str) -> Optional[Dict]:
        """Get vault statistics."""
        if user_id not in self.active_vaults:
            return None

        vault_data = self.active_vaults[user_id]["data"]
        return vault_data.get("metadata", {})


# Global service instance
vault_manager = VaultManager()
