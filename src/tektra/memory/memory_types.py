"""
Memory types and data structures for Tektra

This module defines the core data structures used by the memory system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(Enum):
    """Types of memory entries."""

    CONVERSATION = "conversation"
    AGENT_CONTEXT = "agent_context"
    USER_PREFERENCE = "user_preference"
    SYSTEM_EVENT = "system_event"
    TASK_RESULT = "task_result"
    LEARNED_FACT = "learned_fact"


@dataclass
class MemoryEntry:
    """A single memory entry in the system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType = MemoryType.CONVERSATION
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    importance: float = 0.5  # 0.0 to 1.0
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert memory entry to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "importance": self.importance,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """Create memory entry from dictionary."""
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            importance=data.get("importance", 0.5),
            embedding=data.get("embedding"),
        )


@dataclass
class MemoryContext:
    """Context for memory operations."""

    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    query: str | None = None
    max_results: int = 10
    min_relevance: float = 0.3
    time_window_hours: int | None = None
    memory_types: list[MemoryType] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "query": self.query,
            "max_results": self.max_results,
            "min_relevance": self.min_relevance,
            "time_window_hours": self.time_window_hours,
            "memory_types": [mt.value for mt in self.memory_types],
        }


@dataclass
class MemorySearchResult:
    """Result of a memory search operation."""

    entries: list[MemoryEntry] = field(default_factory=list)
    total_found: int = 0
    search_time_ms: float = 0.0
    query: str = ""
    relevance_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "total_found": self.total_found,
            "search_time_ms": self.search_time_ms,
            "query": self.query,
            "relevance_scores": self.relevance_scores,
        }


@dataclass
class MemoryStats:
    """Statistics about memory usage."""

    total_memories: int = 0
    memories_by_type: dict[str, int] = field(default_factory=dict)
    memories_by_agent: dict[str, int] = field(default_factory=dict)
    memories_by_user: dict[str, int] = field(default_factory=dict)
    oldest_memory: datetime | None = None
    newest_memory: datetime | None = None
    average_importance: float = 0.0
    storage_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_memories": self.total_memories,
            "memories_by_type": self.memories_by_type,
            "memories_by_agent": self.memories_by_agent,
            "memories_by_user": self.memories_by_user,
            "oldest_memory": (
                self.oldest_memory.isoformat() if self.oldest_memory else None
            ),
            "newest_memory": (
                self.newest_memory.isoformat() if self.newest_memory else None
            ),
            "average_importance": self.average_importance,
            "storage_size_bytes": self.storage_size_bytes,
        }


# Utility functions for memory operations
def create_conversation_memory(
    user_message: str,
    assistant_response: str,
    user_id: str,
    session_id: str,
    agent_id: str | None = None,
) -> list[MemoryEntry]:
    """Create memory entries for a conversation exchange."""
    memories = []

    # User message
    user_memory = MemoryEntry(
        type=MemoryType.CONVERSATION,
        content=user_message,
        user_id=user_id,
        session_id=session_id,
        agent_id=agent_id,
        metadata={"role": "user", "exchange_type": "conversation"},
    )
    memories.append(user_memory)

    # Assistant response
    assistant_memory = MemoryEntry(
        type=MemoryType.CONVERSATION,
        content=assistant_response,
        user_id=user_id,
        session_id=session_id,
        agent_id=agent_id,
        metadata={"role": "assistant", "exchange_type": "conversation"},
    )
    memories.append(assistant_memory)

    return memories


def create_agent_context_memory(
    agent_id: str, context: str, importance: float = 0.7
) -> MemoryEntry:
    """Create memory entry for agent context."""
    return MemoryEntry(
        type=MemoryType.AGENT_CONTEXT,
        content=context,
        agent_id=agent_id,
        importance=importance,
        metadata={"context_type": "agent_initialization"},
    )


def create_task_result_memory(
    task_description: str,
    result: str,
    success: bool,
    agent_id: str,
    user_id: str | None = None,
) -> MemoryEntry:
    """Create memory entry for task results."""
    return MemoryEntry(
        type=MemoryType.TASK_RESULT,
        content=f"Task: {task_description}\nResult: {result}",
        agent_id=agent_id,
        user_id=user_id,
        importance=0.8 if success else 0.6,
        metadata={
            "task_description": task_description,
            "result": result,
            "success": success,
            "result_type": "task_execution",
        },
    )
