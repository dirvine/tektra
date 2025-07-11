"""
Tektra Memory Manager

This module provides the main interface for memory operations in Tektra.
It integrates with MemOS when available and provides fallback implementations.
"""

import json
import time
from datetime import datetime, timedelta

import aiosqlite
from loguru import logger

from .memory_config import MemoryConfig
from .memory_types import (
    MemoryContext,
    MemoryEntry,
    MemorySearchResult,
    MemoryStats,
    MemoryType,
    create_agent_context_memory,
    create_conversation_memory,
    create_task_result_memory,
)


class TektraMemoryManager:
    """
    Main memory management class for Tektra.

    Provides persistent memory for agents and conversations with search capabilities.
    Uses MemOS when available, falls back to SQLite-based storage.
    """

    def __init__(self, config: MemoryConfig | None = None):
        """Initialize the memory manager."""
        self.config = config or MemoryConfig()
        self.db_path = self.config.database_path
        self.is_initialized = False
        self.memos_instance = None
        self.cache: dict[str, MemoryEntry] = {}
        self.stats = MemoryStats()

        logger.info(
            f"Memory manager initialized with config: {self.config.storage_type}"
        )

    async def initialize(self) -> bool:
        """Initialize the memory system."""
        try:
            # Initialize database
            await self._initialize_database()

            # Try to initialize MemOS if enabled
            if self.config.use_memos:
                await self._initialize_memos()

            # Load initial statistics
            await self._update_stats()

            self.is_initialized = True
            logger.success("Memory manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            return False

    async def _initialize_database(self):
        """Initialize SQLite database for memory storage."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    agent_id TEXT,
                    session_id TEXT,
                    importance REAL DEFAULT 0.5,
                    embedding TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)
            """
            )

            await db.commit()

        logger.debug("Database initialized successfully")

    async def _initialize_memos(self):
        """Initialize MemOS integration."""
        try:
            # For now, we'll skip MemOS due to configuration complexity
            # and focus on the SQLite implementation
            logger.info("MemOS integration skipped for now - using SQLite backend")

        except Exception as e:
            logger.warning(f"Failed to initialize MemOS: {e}")
            self.config.use_memos = False

    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a memory entry to the system."""
        if not self.is_initialized:
            raise RuntimeError("Memory manager not initialized")

        # Add to cache
        self.cache[entry.id] = entry

        # Add to database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, type, content, metadata, timestamp, user_id, agent_id, session_id, importance, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.id,
                    entry.type.value,
                    entry.content,
                    json.dumps(entry.metadata),
                    entry.timestamp.isoformat(),
                    entry.user_id,
                    entry.agent_id,
                    entry.session_id,
                    entry.importance,
                    json.dumps(entry.embedding) if entry.embedding else None,
                ),
            )
            await db.commit()

        # Update stats
        await self._update_stats()

        logger.debug(f"Added memory: {entry.id} ({entry.type.value})")
        return entry.id

    async def add_memories(self, entries: list[MemoryEntry]) -> list[str]:
        """Add multiple memory entries."""
        memory_ids = []

        for entry in entries:
            memory_id = await self.add_memory(entry)
            memory_ids.append(memory_id)

        return memory_ids

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a specific memory entry by ID."""
        if not self.is_initialized:
            raise RuntimeError("Memory manager not initialized")

        # Check cache first
        if memory_id in self.cache:
            return self.cache[memory_id]

        # Query database
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id, type, content, metadata, timestamp, user_id, agent_id, session_id, importance, embedding
                FROM memories WHERE id = ?
            """,
                (memory_id,),
            ) as cursor:
                row = await cursor.fetchone()

                if row:
                    entry = self._row_to_memory_entry(row)
                    self.cache[memory_id] = entry
                    return entry

        return None

    async def search_memories(self, context: MemoryContext) -> MemorySearchResult:
        """Search for memories based on context."""
        if not self.is_initialized:
            raise RuntimeError("Memory manager not initialized")

        start_time = time.time()

        # Build query
        query_parts = ["SELECT * FROM memories WHERE 1=1"]
        params = []

        # Filter by user
        if context.user_id:
            query_parts.append("AND user_id = ?")
            params.append(context.user_id)

        # Filter by agent
        if context.agent_id:
            query_parts.append("AND agent_id = ?")
            params.append(context.agent_id)

        # Filter by session
        if context.session_id:
            query_parts.append("AND session_id = ?")
            params.append(context.session_id)

        # Filter by memory types
        if context.memory_types:
            type_placeholders = ",".join(["?" for _ in context.memory_types])
            query_parts.append(f"AND type IN ({type_placeholders})")
            params.extend([mt.value for mt in context.memory_types])

        # Filter by time window
        if context.time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=context.time_window_hours)
            query_parts.append("AND timestamp >= ?")
            params.append(cutoff_time.isoformat())

        # Text search in content
        if context.query:
            query_parts.append("AND content LIKE ?")
            params.append(f"%{context.query}%")

        # Importance threshold
        query_parts.append("AND importance >= ?")
        params.append(context.min_relevance)

        # Order by relevance (importance and recency)
        query_parts.append("ORDER BY importance DESC, timestamp DESC")

        # Limit results
        query_parts.append("LIMIT ?")
        params.append(min(context.max_results, self.config.max_search_results))

        query = " ".join(query_parts)

        # Execute search
        entries = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    entry = self._row_to_memory_entry(row)
                    entries.append(entry)

        # Calculate search time
        search_time = (time.time() - start_time) * 1000

        # Create result
        result = MemorySearchResult(
            entries=entries,
            total_found=len(entries),
            search_time_ms=search_time,
            query=context.query or "",
            relevance_scores=[entry.importance for entry in entries],
        )

        logger.debug(
            f"Memory search completed: {len(entries)} results in {search_time:.2f}ms"
        )
        return result

    async def get_conversation_history(
        self, user_id: str, session_id: str | None = None, limit: int = 50
    ) -> list[MemoryEntry]:
        """Get conversation history for a user."""
        context = MemoryContext(
            user_id=user_id,
            session_id=session_id,
            memory_types=[MemoryType.CONVERSATION],
            max_results=limit,
        )

        result = await self.search_memories(context)
        return result.entries

    async def get_agent_context(self, agent_id: str) -> list[MemoryEntry]:
        """Get context memories for an agent."""
        context = MemoryContext(
            agent_id=agent_id, memory_types=[MemoryType.AGENT_CONTEXT], max_results=100
        )

        result = await self.search_memories(context)
        return result.entries

    async def add_conversation(
        self,
        user_message: str,
        assistant_response: str,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[str]:
        """Add a conversation exchange to memory."""
        memories = create_conversation_memory(
            user_message=user_message,
            assistant_response=assistant_response,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
        )

        return await self.add_memories(memories)

    async def add_agent_context(
        self, agent_id: str, context: str, importance: float = 0.7
    ) -> str:
        """Add context memory for an agent."""
        memory = create_agent_context_memory(
            agent_id=agent_id, context=context, importance=importance
        )

        return await self.add_memory(memory)

    async def add_task_result(
        self,
        task_description: str,
        result: str,
        success: bool,
        agent_id: str,
        user_id: str | None = None,
    ) -> str:
        """Add task result to memory."""
        memory = create_task_result_memory(
            task_description=task_description,
            result=result,
            success=success,
            agent_id=agent_id,
            user_id=user_id,
        )

        return await self.add_memory(memory)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        if not self.is_initialized:
            raise RuntimeError("Memory manager not initialized")

        # Remove from cache
        if memory_id in self.cache:
            del self.cache[memory_id]

        # Remove from database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await db.commit()

        # Update stats
        await self._update_stats()

        logger.debug(f"Deleted memory: {memory_id}")
        return True

    async def cleanup_old_memories(self, days: int = None) -> int:
        """Clean up old memories based on retention policy."""
        if not self.is_initialized:
            raise RuntimeError("Memory manager not initialized")

        if days is None:
            days = self.config.retention_days

        cutoff_date = datetime.now() - timedelta(days=days)

        # Get memories to delete
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id FROM memories WHERE timestamp < ?
            """,
                (cutoff_date.isoformat(),),
            ) as cursor:
                to_delete = [row[0] async for row in cursor]

            # Delete old memories
            if to_delete:
                await db.execute(
                    """
                    DELETE FROM memories WHERE timestamp < ?
                """,
                    (cutoff_date.isoformat(),),
                )
                await db.commit()

        # Clear cache of deleted items
        for memory_id in to_delete:
            if memory_id in self.cache:
                del self.cache[memory_id]

        # Update stats
        await self._update_stats()

        logger.info(f"Cleaned up {len(to_delete)} old memories")
        return len(to_delete)

    async def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        await self._update_stats()
        return self.stats

    async def _update_stats(self):
        """Update memory statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            # Total memories
            async with db.execute("SELECT COUNT(*) FROM memories") as cursor:
                self.stats.total_memories = (await cursor.fetchone())[0]

            # Memories by type
            async with db.execute(
                """
                SELECT type, COUNT(*) FROM memories GROUP BY type
            """
            ) as cursor:
                self.stats.memories_by_type = {row[0]: row[1] async for row in cursor}

            # Memories by agent
            async with db.execute(
                """
                SELECT agent_id, COUNT(*) FROM memories
                WHERE agent_id IS NOT NULL GROUP BY agent_id
            """
            ) as cursor:
                self.stats.memories_by_agent = {row[0]: row[1] async for row in cursor}

            # Memories by user
            async with db.execute(
                """
                SELECT user_id, COUNT(*) FROM memories
                WHERE user_id IS NOT NULL GROUP BY user_id
            """
            ) as cursor:
                self.stats.memories_by_user = {row[0]: row[1] async for row in cursor}

            # Oldest and newest memories
            async with db.execute(
                """
                SELECT MIN(timestamp), MAX(timestamp) FROM memories
            """
            ) as cursor:
                row = await cursor.fetchone()
                if row[0]:
                    self.stats.oldest_memory = datetime.fromisoformat(row[0])
                if row[1]:
                    self.stats.newest_memory = datetime.fromisoformat(row[1])

            # Average importance
            async with db.execute(
                """
                SELECT AVG(importance) FROM memories
            """
            ) as cursor:
                avg_importance = (await cursor.fetchone())[0]
                self.stats.average_importance = avg_importance or 0.0

    def _row_to_memory_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            type=MemoryType(row[1]),
            content=row[2],
            metadata=json.loads(row[3]) if row[3] else {},
            timestamp=datetime.fromisoformat(row[4]),
            user_id=row[5],
            agent_id=row[6],
            session_id=row[7],
            importance=row[8] or 0.5,
            embedding=json.loads(row[9]) if row[9] else None,
        )

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "memos_instance") and self.memos_instance:
            # Cleanup MemOS if initialized
            try:
                # Close any active connections in MemOS
                await self.memos_instance.close()
                logger.debug("MemOS instance closed successfully")
            except Exception as e:
                logger.warning(f"Error closing MemOS instance: {e}")
            finally:
                self.memos_instance = None

        # Clear cache
        self.cache.clear()

        logger.info("Memory manager cleaned up")
