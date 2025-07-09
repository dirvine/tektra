#!/usr/bin/env python3
"""
Core memory system integration tests.
"""

import pytest
import sys
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tektra.memory.memory_manager import TektraMemoryManager
from tektra.memory.memory_types import MemoryType, MemoryEntry, MemoryContext
from tektra.memory.memory_config import MemoryConfig


class TestMemoryCore:
    """Test core memory functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def memory_config(self, temp_dir):
        """Create a test memory configuration."""
        return MemoryConfig(
            storage_path=str(temp_dir),
            database_name="test_memory.db",
            max_memories_per_user=100,
            use_memos=False,  # Disable for testing
            enable_semantic_search=False  # Disable for testing
        )
    
    @pytest.fixture
    def memory_manager(self, memory_config):
        """Create a TektraMemoryManager instance for testing."""
        return TektraMemoryManager(memory_config)
    
    def test_memory_config_creation(self, memory_config):
        """Test memory configuration creation."""
        assert memory_config is not None
        assert memory_config.storage_path is not None
        assert memory_config.database_name == "test_memory.db"
        assert memory_config.max_memories_per_user == 100
        assert memory_config.use_memos is False
    
    def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager is not None
        assert memory_manager.config is not None
        assert memory_manager.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_and_cleanup(self, memory_manager):
        """Test memory manager initialization and cleanup."""
        # Initialize
        result = await memory_manager.initialize()
        assert result is True
        assert memory_manager.is_initialized is True
        
        # Cleanup
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_basic_memory_operations(self, memory_manager):
        """Test basic memory operations."""
        # Initialize
        await memory_manager.initialize()
        
        # Create a test memory entry
        test_entry = MemoryEntry(
            id="test_001",
            content="This is a test memory entry",
            type=MemoryType.CONVERSATION,
            metadata={"source": "test"},
            importance=0.8,
            timestamp=datetime.now(),
            user_id="test_user",
            session_id="test_session"
        )
        
        # Store the memory entry
        stored_id = await memory_manager.add_memory(test_entry)
        assert stored_id == "test_001"
        
        # Retrieve the memory entry
        retrieved_entry = await memory_manager.get_memory(stored_id)
        assert retrieved_entry is not None
        assert retrieved_entry.content == "This is a test memory entry"
        assert retrieved_entry.type == MemoryType.CONVERSATION
        assert retrieved_entry.importance == 0.8
        assert retrieved_entry.user_id == "test_user"
        assert retrieved_entry.session_id == "test_session"
        
        # Cleanup
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_search(self, memory_manager):
        """Test memory search functionality."""
        await memory_manager.initialize()
        
        # Store multiple memory entries
        entries = [
            MemoryEntry(
                id="search_001",
                content="Python programming tutorial",
                type=MemoryType.LEARNED_FACT,
                metadata={"topic": "programming"},
                importance=0.9,
                timestamp=datetime.now(),
                user_id="user1"
            ),
            MemoryEntry(
                id="search_002",
                content="Machine learning algorithms",
                type=MemoryType.LEARNED_FACT,
                metadata={"topic": "AI"},
                importance=0.8,
                timestamp=datetime.now(),
                user_id="user1"
            ),
            MemoryEntry(
                id="search_003",
                content="User said hello",
                type=MemoryType.CONVERSATION,
                metadata={"speaker": "user"},
                importance=0.3,
                timestamp=datetime.now(),
                user_id="user1",
                session_id="session1"
            )
        ]
        
        for entry in entries:
            await memory_manager.add_memory(entry)
        
        # Test search by content
        search_context = MemoryContext(
            query="programming",
            max_results=5,
            user_id="user1"
        )
        result = await memory_manager.search_memories(search_context)
        assert result is not None
        assert len(result.entries) >= 1
        assert any("programming" in entry.content.lower() for entry in result.entries)
        
        # Test search by memory type
        knowledge_context = MemoryContext(
            memory_types=[MemoryType.LEARNED_FACT],
            max_results=10,
            user_id="user1"
        )
        knowledge_result = await memory_manager.search_memories(knowledge_context)
        assert len(knowledge_result.entries) >= 2
        assert all(entry.type == MemoryType.LEARNED_FACT for entry in knowledge_result.entries)
        
        # Test conversation search
        conversation_context = MemoryContext(
            memory_types=[MemoryType.CONVERSATION],
            max_results=10,
            user_id="user1",
            session_id="session1"
        )
        conversation_result = await memory_manager.search_memories(conversation_context)
        assert len(conversation_result.entries) >= 1
        assert all(entry.type == MemoryType.CONVERSATION for entry in conversation_result.entries)
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, memory_manager):
        """Test conversation history functionality."""
        await memory_manager.initialize()
        
        # Add conversation entries
        await memory_manager.add_conversation(
            user_message="Hello, how are you?",
            assistant_response="I'm doing well, thank you!",
            user_id="user1",
            session_id="session1"
        )
        
        await memory_manager.add_conversation(
            user_message="What can you help me with?",
            assistant_response="I can help with many things!",
            user_id="user1",
            session_id="session1"
        )
        
        # Get conversation history
        history = await memory_manager.get_conversation_history(
            user_id="user1",
            session_id="session1",
            limit=10
        )
        
        assert len(history) >= 2
        assert all(entry.type == MemoryType.CONVERSATION for entry in history)
        assert all(entry.user_id == "user1" for entry in history)
        assert all(entry.session_id == "session1" for entry in history)
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_deletion(self, memory_manager):
        """Test memory deletion."""
        await memory_manager.initialize()
        
        # Create and store a memory entry
        test_entry = MemoryEntry(
            id="delete_test",
            content="This will be deleted",
            type=MemoryType.CONVERSATION,
            timestamp=datetime.now(),
            user_id="user1"
        )
        
        stored_id = await memory_manager.add_memory(test_entry)
        assert stored_id == "delete_test"
        
        # Verify it exists
        retrieved = await memory_manager.get_memory(stored_id)
        assert retrieved is not None
        
        # Delete it
        deleted = await memory_manager.delete_memory(stored_id)
        assert deleted is True
        
        # Verify it's gone
        retrieved_after_delete = await memory_manager.get_memory(stored_id)
        assert retrieved_after_delete is None
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_manager):
        """Test memory cleanup functionality."""
        await memory_manager.initialize()
        
        # Create old and new memory entries
        old_date = datetime.now() - timedelta(days=30)
        recent_date = datetime.now()
        
        old_entry = MemoryEntry(
            id="old_memory",
            content="Old memory entry",
            type=MemoryType.CONVERSATION,
            timestamp=old_date,
            importance=0.1,
            user_id="user1"
        )
        
        recent_entry = MemoryEntry(
            id="recent_memory",
            content="Recent memory entry",
            type=MemoryType.CONVERSATION,
            timestamp=recent_date,
            importance=0.9,
            user_id="user1"
        )
        
        # Store both entries
        await memory_manager.add_memory(old_entry)
        await memory_manager.add_memory(recent_entry)
        
        # Perform cleanup (delete entries older than 7 days)
        cleaned_count = await memory_manager.cleanup_old_memories(days=7)
        assert cleaned_count >= 0
        
        # Recent entry should still exist
        recent_retrieved = await memory_manager.get_memory("recent_memory")
        assert recent_retrieved is not None
        
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_statistics(self, memory_manager):
        """Test memory statistics."""
        await memory_manager.initialize()
        
        # Store some test entries
        entries = [
            MemoryEntry(
                id="stats_1",
                content="Conversation memory",
                type=MemoryType.CONVERSATION,
                importance=0.7,
                timestamp=datetime.now(),
                user_id="user1"
            ),
            MemoryEntry(
                id="stats_2",
                content="Learned fact",
                type=MemoryType.LEARNED_FACT,
                importance=0.9,
                timestamp=datetime.now(),
                user_id="user1"
            ),
            MemoryEntry(
                id="stats_3",
                content="Task result",
                type=MemoryType.TASK_RESULT,
                importance=0.8,
                timestamp=datetime.now(),
                user_id="user1"
            )
        ]
        
        for entry in entries:
            await memory_manager.add_memory(entry)
        
        # Get statistics
        stats = await memory_manager.get_memory_stats()
        
        assert stats is not None
        assert stats.total_memories >= 3
        assert len(stats.memories_by_type) >= 3
        assert stats.average_importance > 0
        
        await memory_manager.cleanup()


class TestMemoryTypes:
    """Test memory types and data structures."""
    
    def test_memory_type_enum(self):
        """Test MemoryType enum."""
        assert MemoryType.CONVERSATION.value == "conversation"
        assert MemoryType.LEARNED_FACT.value == "learned_fact"
        assert MemoryType.TASK_RESULT.value == "task_result"
    
    def test_memory_entry_creation(self):
        """Test MemoryEntry creation."""
        entry = MemoryEntry(
            id="test_entry",
            content="Test content",
            type=MemoryType.CONVERSATION,
            metadata={"test": "value"},
            importance=0.5,
            timestamp=datetime.now(),
            user_id="user1"
        )
        
        assert entry.id == "test_entry"
        assert entry.content == "Test content"
        assert entry.type == MemoryType.CONVERSATION
        assert entry.metadata["test"] == "value"
        assert entry.importance == 0.5
        assert entry.user_id == "user1"
    
    def test_memory_entry_serialization(self):
        """Test MemoryEntry serialization."""
        entry = MemoryEntry(
            id="serialize_test",
            content="Serialization test",
            type=MemoryType.LEARNED_FACT,
            metadata={"key": "value"},
            importance=0.8,
            timestamp=datetime.now(),
            user_id="user1"
        )
        
        # Test to_dict
        entry_dict = entry.to_dict()
        assert entry_dict["id"] == "serialize_test"
        assert entry_dict["content"] == "Serialization test"
        assert entry_dict["type"] == MemoryType.LEARNED_FACT.value
        assert entry_dict["metadata"]["key"] == "value"
        assert entry_dict["importance"] == 0.8
        assert entry_dict["user_id"] == "user1"
        
        # Test from_dict
        reconstructed = MemoryEntry.from_dict(entry_dict)
        assert reconstructed.id == entry.id
        assert reconstructed.content == entry.content
        assert reconstructed.type == entry.type
        assert reconstructed.metadata == entry.metadata
        assert reconstructed.importance == entry.importance
        assert reconstructed.user_id == entry.user_id
    
    def test_memory_context_creation(self):
        """Test MemoryContext creation."""
        context = MemoryContext(
            user_id="user1",
            session_id="session1",
            query="test query",
            max_results=10,
            min_relevance=0.5,
            memory_types=[MemoryType.CONVERSATION]
        )
        
        assert context.user_id == "user1"
        assert context.session_id == "session1"
        assert context.query == "test query"
        assert context.max_results == 10
        assert context.min_relevance == 0.5
        assert MemoryType.CONVERSATION in context.memory_types
    
    def test_memory_context_serialization(self):
        """Test MemoryContext serialization."""
        context = MemoryContext(
            user_id="user1",
            query="test",
            max_results=5,
            memory_types=[MemoryType.CONVERSATION, MemoryType.LEARNED_FACT]
        )
        
        context_dict = context.to_dict()
        assert context_dict["user_id"] == "user1"
        assert context_dict["query"] == "test"
        assert context_dict["max_results"] == 5
        assert "conversation" in context_dict["memory_types"]
        assert "learned_fact" in context_dict["memory_types"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])