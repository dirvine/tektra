"""
Conversational Memory System

This module provides intelligent memory capabilities for Tektra conversations,
including context awareness, learning from interactions, and semantic memory.
"""

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    memory_type: str  # 'fact', 'preference', 'context', 'conversation'
    importance: float  # 0.0 to 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int
    tags: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None  # For semantic search (future)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class ConversationMemory:
    """
    Manages conversational memory for more intelligent interactions.
    
    Features:
    - Stores important facts and preferences
    - Learns from conversation patterns
    - Provides context-aware responses
    - Maintains conversation history
    - Simple semantic search (keyword-based)
    """

    def __init__(self, memory_dir: Path):
        """
        Initialize conversation memory.
        
        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.memory_dir / "conversation_memory.db"
        self.memories: Dict[str, MemoryEntry] = {}
        
        # Memory configuration
        self.max_memories = 10000
        self.importance_threshold = 0.1
        self.context_window_hours = 24
        
        # Initialize database
        self._init_database()
        self._load_memories()
        
        logger.info(f"Conversation memory initialized with {len(self.memories)} memories")

    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
            
            conn.commit()

    def _load_memories(self):
        """Load memories from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memories ORDER BY importance DESC")
                
                for row in cursor.fetchall():
                    memory_data = {
                        'id': row[0],
                        'content': row[1],
                        'memory_type': row[2],
                        'importance': row[3],
                        'created_at': row[4],
                        'last_accessed': row[5],
                        'access_count': row[6],
                        'tags': json.loads(row[7]),
                        'metadata': json.loads(row[8]),
                        'embedding': json.loads(row[9]) if row[9] else None
                    }
                    
                    memory = MemoryEntry.from_dict(memory_data)
                    self.memories[memory.id] = memory
                    
        except Exception as e:
            logger.warning(f"Failed to load memories: {e}")

    def _save_memory(self, memory: MemoryEntry):
        """Save a single memory to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, content, memory_type, importance, created_at, last_accessed, 
                     access_count, tags, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.content,
                    memory.memory_type,
                    memory.importance,
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    json.dumps(memory.tags),
                    json.dumps(memory.metadata),
                    json.dumps(memory.embedding) if memory.embedding else None
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    async def add_memory(
        self,
        content: str,
        memory_type: str = "conversation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory (fact, preference, context, conversation)
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization
            metadata: Optional metadata
            
        Returns:
            str: Memory ID
        """
        memory_id = str(uuid.uuid4())
        now = datetime.now()
        
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=now,
            last_accessed=now,
            access_count=0,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.memories[memory_id] = memory
        self._save_memory(memory)
        
        # Clean up old memories if needed
        await self._cleanup_memories()
        
        logger.debug(f"Added memory: {content[:50]}... (importance: {importance})")
        return memory_id

    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Search memories by content.
        
        Args:
            query: Search query
            memory_types: Filter by memory types
            limit: Maximum number of results
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memories
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for memory in self.memories.values():
            # Skip if below importance threshold
            if memory.importance < min_importance:
                continue
            
            # Filter by memory type
            if memory_types and memory.memory_type not in memory_types:
                continue
            
            # Simple keyword matching
            content_lower = memory.content.lower()
            content_words = set(content_lower.split())
            
            # Calculate relevance score
            relevance = 0.0
            
            # Exact phrase match (high score)
            if query_lower in content_lower:
                relevance += 0.8
            
            # Word overlap (medium score)
            word_overlap = len(query_words.intersection(content_words))
            if word_overlap > 0:
                relevance += 0.4 * (word_overlap / len(query_words))
            
            # Tag matches (medium score)
            for tag in memory.tags:
                if any(word in tag.lower() for word in query_words):
                    relevance += 0.3
            
            # Apply importance weighting
            relevance *= memory.importance
            
            if relevance > 0.1:  # Minimum relevance threshold
                memory.last_accessed = datetime.now()
                memory.access_count += 1
                results.append((memory, relevance))
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        memories = [memory for memory, _ in results[:limit]]
        
        # Update access information
        for memory in memories:
            self._save_memory(memory)
        
        logger.debug(f"Memory search '{query}': {len(memories)} results")
        return memories

    async def get_recent_context(self, hours: int = 2) -> List[MemoryEntry]:
        """
        Get recent conversation context.
        
        Args:
            hours: How many hours back to look
            
        Returns:
            List of recent memories
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_memories = [
            memory for memory in self.memories.values()
            if (memory.created_at >= cutoff_time and 
                memory.memory_type in ['conversation', 'context'])
        ]
        
        # Sort by creation time
        recent_memories.sort(key=lambda m: m.created_at, reverse=True)
        
        return recent_memories[:20]  # Limit to last 20 entries

    async def learn_from_conversation(
        self,
        user_message: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Learn from a conversation turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            context: Optional context information
        """
        # Extract facts and preferences
        await self._extract_facts(user_message, context)
        
        # Store conversation turn
        conversation_entry = f"User: {user_message}\nAssistant: {assistant_response}"
        
        await self.add_memory(
            content=conversation_entry,
            memory_type="conversation",
            importance=0.3,
            tags=["conversation_turn"],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response
            }
        )

    async def _extract_facts(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Extract facts and preferences from user message."""
        message_lower = message.lower()
        
        # Simple fact extraction patterns
        fact_patterns = [
            ("I am", "user_identity"),
            ("my name is", "user_name"),
            ("I like", "preference_positive"),
            ("I don't like", "preference_negative"),
            ("I prefer", "preference"),
            ("I work", "user_occupation"),
            ("I live", "user_location"),
        ]
        
        for pattern, fact_type in fact_patterns:
            if pattern in message_lower:
                # Extract the fact
                start_idx = message_lower.find(pattern)
                fact_content = message[start_idx:].split('.')[0]  # Get first sentence
                
                await self.add_memory(
                    content=fact_content,
                    memory_type="fact",
                    importance=0.7,
                    tags=[fact_type, "extracted_fact"],
                    metadata={
                        "extraction_pattern": pattern,
                        "source_message": message
                    }
                )

    async def get_contextual_memories(self, query: str) -> str:
        """
        Get contextual memories as a formatted string for LLM context.
        
        Args:
            query: Current user query
            
        Returns:
            str: Formatted memory context
        """
        # Search for relevant memories
        relevant_memories = await self.search_memories(query, limit=5, min_importance=0.3)
        
        # Get recent context
        recent_memories = await self.get_recent_context(hours=2)
        
        context_parts = []
        
        # Add relevant memories
        if relevant_memories:
            context_parts.append("## Relevant Context:")
            for memory in relevant_memories:
                context_parts.append(f"- {memory.content}")
        
        # Add recent conversation context
        if recent_memories and len(recent_memories) > 1:
            context_parts.append("\n## Recent Conversation:")
            for memory in recent_memories[-3:]:  # Last 3 entries
                if memory.memory_type == "conversation":
                    # Extract just the user message part
                    if "User:" in memory.content:
                        user_part = memory.content.split("User:")[1].split("Assistant:")[0].strip()
                        context_parts.append(f"- User mentioned: {user_part}")
        
        if context_parts:
            return "\n".join(context_parts) + "\n"
        
        return ""

    async def _cleanup_memories(self):
        """Clean up old or low-importance memories."""
        if len(self.memories) <= self.max_memories:
            return
        
        # Sort memories by importance and age
        memory_list = list(self.memories.values())
        memory_list.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
        
        # Keep only the most important memories
        to_keep = memory_list[:self.max_memories]
        to_remove = memory_list[self.max_memories:]
        
        # Remove old memories
        for memory in to_remove:
            if memory.importance < self.importance_threshold:
                del self.memories[memory.id]
                
                # Remove from database
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("DELETE FROM memories WHERE id = ?", (memory.id,))
                        conn.commit()
                except Exception as e:
                    logger.warning(f"Failed to remove memory from database: {e}")
        
        logger.debug(f"Cleaned up {len(to_remove)} old memories")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self.memories:
            return {"total_memories": 0}
        
        # Count by type
        type_counts = {}
        importance_sum = 0
        
        for memory in self.memories.values():
            type_counts[memory.memory_type] = type_counts.get(memory.memory_type, 0) + 1
            importance_sum += memory.importance
        
        return {
            "total_memories": len(self.memories),
            "memory_types": type_counts,
            "average_importance": importance_sum / len(self.memories),
            "database_path": str(self.db_path)
        }

    async def forget_memories(self, memory_ids: List[str]) -> int:
        """
        Remove specific memories.
        
        Args:
            memory_ids: List of memory IDs to remove
            
        Returns:
            int: Number of memories removed
        """
        removed_count = 0
        
        for memory_id in memory_ids:
            if memory_id in self.memories:
                del self.memories[memory_id]
                
                # Remove from database
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                        conn.commit()
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove memory {memory_id}: {e}")
        
        logger.info(f"Removed {removed_count} memories")
        return removed_count