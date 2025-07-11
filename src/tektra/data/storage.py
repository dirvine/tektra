"""
Data Storage Management

This module provides data storage and persistence capabilities for Tektra AI Assistant,
including conversation history, user preferences, and session data.
"""

import json
import platform
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


class ConversationStorage:
    """Manage conversation history storage."""

    def __init__(self, db_path: str):
        """Initialize conversation storage with SQLite database."""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        routing_info TEXT
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                        message_count INTEGER DEFAULT 0,
                        session_metadata TEXT
                    )
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversations_session
                    ON conversations(session_id)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                    ON conversations(timestamp)
                """
                )

                conn.commit()
                logger.debug("Conversation database initialized")

        except Exception as e:
            logger.error(f"Error initializing conversation database: {e}")

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        routing_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a conversation message.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional message metadata
            routing_info: Optional routing information

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations
                    (id, session_id, role, content, metadata, routing_info)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        message_id,
                        session_id,
                        role,
                        content,
                        json.dumps(metadata) if metadata else None,
                        json.dumps(routing_info) if routing_info else None,
                    ),
                )

                # Update session activity
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sessions
                    (id, last_activity, message_count)
                    VALUES (
                        ?,
                        CURRENT_TIMESTAMP,
                        COALESCE((SELECT message_count FROM sessions WHERE id = ?), 0) + 1
                    )
                """,
                    (session_id, session_id),
                )

                conn.commit()
                logger.debug(f"Saved message {message_id} to session {session_id}")

        except Exception as e:
            logger.error(f"Error saving message: {e}")

        return message_id

    async def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of conversation messages
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT id, role, content, timestamp, metadata, routing_info
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """,
                    (session_id, limit),
                )

                messages = []
                for row in cursor:
                    message = {
                        "id": row["id"],
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                    }

                    if row["metadata"]:
                        message["metadata"] = json.loads(row["metadata"])

                    if row["routing_info"]:
                        message["routing_info"] = json.loads(row["routing_info"])

                    messages.append(message)

                return messages

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def clear_old_conversations(self, days_old: int = 30) -> int:
        """
        Clear conversations older than specified days.

        Args:
            days_old: Age threshold in days

        Returns:
            Number of messages deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM conversations
                    WHERE timestamp < ?
                """,
                    (cutoff_date.isoformat(),),
                )

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Deleted {deleted_count} old conversation messages")
                return deleted_count

        except Exception as e:
            logger.error(f"Error clearing old conversations: {e}")
            return 0


class SessionStorage:
    """Manage session data and temporary storage."""

    def __init__(self):
        """Initialize session storage."""
        self.sessions = {}
        self.current_session_id = None

    def create_session(self, session_id: str | None = None) -> str:
        """
        Create a new session.

        Args:
            session_id: Optional session ID (generated if not provided)

        Returns:
            Session ID
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "data": {},
            "message_count": 0,
        }

        self.current_session_id = session_id
        logger.debug(f"Created session: {session_id}")
        return session_id

    def get_current_session(self) -> str | None:
        """Get current session ID."""
        return self.current_session_id

    def set_session_data(self, session_id: str, key: str, value: Any) -> None:
        """Set data for a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["data"][key] = value
            self.sessions[session_id]["last_activity"] = datetime.now()

    def get_session_data(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get data from a session."""
        if session_id in self.sessions:
            return self.sessions[session_id]["data"].get(key, default)
        return default

    def increment_message_count(self, session_id: str) -> None:
        """Increment message count for a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] += 1
            self.sessions[session_id]["last_activity"] = datetime.now()

    def cleanup_inactive_sessions(self, hours_inactive: int = 24) -> int:
        """
        Clean up inactive sessions.

        Args:
            hours_inactive: Hours of inactivity threshold

        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_inactive)
        inactive_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if session["last_activity"] < cutoff_time
        ]

        for session_id in inactive_sessions:
            del self.sessions[session_id]

        logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
        return len(inactive_sessions)


class DataStorage:
    """
    Main data storage manager for Tektra AI Assistant.

    Provides centralized data management including conversations,
    sessions, user preferences, and temporary data.
    """

    def __init__(self, data_dir: str | None = None):
        """
        Initialize data storage.

        Args:
            data_dir: Optional data directory path
        """
        self.data_dir = Path(data_dir) if data_dir else self._get_default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage components
        db_path = self.data_dir / "conversations.db"
        self.conversation_storage = ConversationStorage(str(db_path))
        self.session_storage = SessionStorage()

        # Cache and temporary storage
        self.cache = {}
        self.temp_files = {}

        logger.info(f"Data storage initialized at {self.data_dir}")

    def _get_default_data_dir(self) -> Path:
        """Get default data directory path."""
        if platform.system() == "Windows":
            data_dir = Path.home() / "AppData" / "Local" / "Tektra" / "Data"
        elif platform.system() == "Darwin":  # macOS
            data_dir = (
                Path.home() / "Library" / "Application Support" / "Tektra" / "Data"
            )
        else:  # Linux and others
            data_dir = Path.home() / ".local" / "share" / "tektra"

        return data_dir

    async def start_new_session(self) -> str:
        """
        Start a new conversation session.

        Returns:
            Session ID
        """
        session_id = self.session_storage.create_session()
        logger.info(f"Started new session: {session_id}")
        return session_id

    async def save_message(
        self,
        role: str,
        content: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        routing_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a conversation message.

        Args:
            role: Message role
            content: Message content
            session_id: Optional session ID (uses current if not provided)
            metadata: Optional message metadata
            routing_info: Optional routing information

        Returns:
            Message ID
        """
        if not session_id:
            session_id = self.session_storage.get_current_session()
            if not session_id:
                session_id = await self.start_new_session()

        message_id = await self.conversation_storage.save_message(
            session_id, role, content, metadata, routing_info
        )

        self.session_storage.increment_message_count(session_id)
        return message_id

    async def get_conversation_history(
        self, session_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Get conversation history.

        Args:
            session_id: Optional session ID (uses current if not provided)
            limit: Maximum number of messages

        Returns:
            List of conversation messages
        """
        if not session_id:
            session_id = self.session_storage.get_current_session()
            if not session_id:
                return []

        return await self.conversation_storage.get_conversation_history(
            session_id, limit
        )

    def set_cache(
        self, key: str, value: Any, ttl_seconds: int | None = None
    ) -> None:
        """
        Set cache value with optional TTL.

        Args:
            key: Cache key
            value: Cache value
            ttl_seconds: Time to live in seconds
        """
        cache_entry = {"value": value, "timestamp": datetime.now()}

        if ttl_seconds:
            cache_entry["expires_at"] = datetime.now() + timedelta(seconds=ttl_seconds)

        self.cache[key] = cache_entry

    def get_cache(self, key: str, default: Any = None) -> Any:
        """
        Get cache value.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        if key not in self.cache:
            return default

        entry = self.cache[key]

        # Check expiration
        if "expires_at" in entry and datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return default

        return entry["value"]

    def clear_cache(self, pattern: str | None = None) -> int:
        """
        Clear cache entries.

        Args:
            pattern: Optional pattern to match keys (clears all if None)

        Returns:
            Number of entries cleared
        """
        if pattern:
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
        else:
            keys_to_remove = list(self.cache.keys())

        for key in keys_to_remove:
            del self.cache[key]

        return len(keys_to_remove)

    async def store_temp_file(self, filename: str, content: bytes) -> str:
        """
        Store temporary file.

        Args:
            filename: Original filename
            content: File content

        Returns:
            Temporary file path
        """
        temp_dir = self.data_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = Path(filename).suffix
        temp_filename = f"{file_id}{file_ext}"
        temp_path = temp_dir / temp_filename

        # Write file
        with open(temp_path, "wb") as f:
            f.write(content)

        # Track temporary file
        self.temp_files[file_id] = {
            "original_name": filename,
            "temp_path": str(temp_path),
            "created_at": datetime.now(),
            "size": len(content),
        }

        logger.debug(f"Stored temporary file: {filename} -> {temp_path}")
        return str(temp_path)

    def cleanup_temp_files(self, hours_old: int = 24) -> int:
        """
        Clean up old temporary files.

        Args:
            hours_old: Age threshold in hours

        Returns:
            Number of files cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        files_removed = 0

        files_to_remove = []
        for file_id, file_info in self.temp_files.items():
            if file_info["created_at"] < cutoff_time:
                try:
                    temp_path = Path(file_info["temp_path"])
                    if temp_path.exists():
                        temp_path.unlink()
                    files_to_remove.append(file_id)
                    files_removed += 1
                except Exception as e:
                    logger.warning(f"Error removing temp file {file_id}: {e}")

        # Remove from tracking
        for file_id in files_to_remove:
            del self.temp_files[file_id]

        logger.info(f"Cleaned up {files_removed} temporary files")
        return files_removed

    async def perform_maintenance(self) -> dict[str, int]:
        """
        Perform routine maintenance tasks.

        Returns:
            Dictionary with maintenance results
        """
        results = {}

        # Clean up old conversations
        results["conversations_deleted"] = (
            await self.conversation_storage.clear_old_conversations()
        )

        # Clean up inactive sessions
        results["sessions_cleaned"] = self.session_storage.cleanup_inactive_sessions()

        # Clean up temporary files
        results["temp_files_removed"] = self.cleanup_temp_files()

        # Clear expired cache entries
        results["cache_entries_cleared"] = self.clear_cache()

        logger.info(f"Maintenance completed: {results}")
        return results

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "data_directory": str(self.data_dir),
            "cache_entries": len(self.cache),
            "temp_files": len(self.temp_files),
            "active_sessions": len(self.session_storage.sessions),
            "current_session": self.session_storage.get_current_session(),
        }

        # Database stats
        try:
            with sqlite3.connect(self.conversation_storage.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                stats["total_messages"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                stats["total_sessions"] = cursor.fetchone()[0]
        except Exception as e:
            logger.debug(f"Error getting database stats: {e}")
            stats["total_messages"] = 0
            stats["total_sessions"] = 0

        return stats

    async def export_data(self, export_path: str) -> bool:
        """
        Export all conversation data.

        Args:
            export_path: Path to export data

        Returns:
            bool: True if exported successfully
        """
        try:
            # Get all conversations
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "conversations": {},
                "sessions": {},
            }

            # Export all sessions
            for session_id in self.session_storage.sessions:
                conversations = await self.get_conversation_history(
                    session_id, limit=1000
                )
                export_data["conversations"][session_id] = conversations
                export_data["sessions"][session_id] = self.session_storage.sessions[
                    session_id
                ]

            # Write to file
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"Data exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
