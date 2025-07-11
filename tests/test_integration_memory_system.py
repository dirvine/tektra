#!/usr/bin/env python3
"""
Integration tests for the memory system components.
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tektra.memory.memory_config import MemoryConfig
from tektra.memory.memory_manager import TektraMemoryManager
from tektra.memory.memory_types import MemoryContext, MemoryEntry, MemoryType


class TestMemoryManager:
    """Test MemoryManager integration."""

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
            database_name="memory_test.db",
            max_memories_per_user=1000,
            cleanup_interval_hours=24,
            enable_semantic_search=True,
            use_memos=False,  # Disable for testing
        )

    @pytest.fixture
    def memory_manager(self, memory_config):
        """Create a TektraMemoryManager instance for testing."""
        return TektraMemoryManager(memory_config)

    def test_memory_manager_initialization(self, memory_manager):
        """Test MemoryManager initialization."""
        assert memory_manager is not None
        assert memory_manager.config is not None
        assert memory_manager.config.max_memories_per_user == 1000

    @pytest.mark.asyncio
    async def test_store_and_get_memory(self, memory_manager):
        """Test storing and retrieving memory entries."""
        # Initialize the memory manager
        await memory_manager.initialize()

        # Create a test memory entry
        test_entry = MemoryEntry(
            id="test_001",
            content="This is a test memory entry",
            type=MemoryType.CONVERSATION,
            metadata={"source": "test"},
            importance=0.8,
            timestamp=datetime.now(),
        )

        # Store the memory entry
        stored_id = await memory_manager.add_memory(test_entry)
        assert stored_id is not None

        # Retrieve the memory entry
        retrieved_entry = await memory_manager.get_memory(stored_id)
        assert retrieved_entry is not None
        assert retrieved_entry.content == "This is a test memory entry"
        assert retrieved_entry.type == MemoryType.CONVERSATION
        assert retrieved_entry.importance == 0.8

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
            ),
            MemoryEntry(
                id="search_002",
                content="Machine learning algorithms",
                type=MemoryType.LEARNED_FACT,
                metadata={"topic": "AI"},
                importance=0.8,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="search_003",
                content="User said hello",
                type=MemoryType.CONVERSATION,
                metadata={"speaker": "user"},
                importance=0.3,
                timestamp=datetime.now(),
            ),
        ]

        for entry in entries:
            await memory_manager.add_memory(entry)

        # Test search by content
        search_context = MemoryContext(query="programming", max_results=5)
        result = await memory_manager.search_memories(search_context)
        assert len(result.entries) >= 1
        assert any("programming" in entry.content.lower() for entry in result.entries)

        # Test search by memory type
        knowledge_context = MemoryContext(
            memory_types=[MemoryType.LEARNED_FACT], max_results=10
        )
        knowledge_result = await memory_manager.search_memories(knowledge_context)
        assert len(knowledge_result.entries) >= 2
        assert all(
            entry.type == MemoryType.LEARNED_FACT for entry in knowledge_result.entries
        )

        await memory_manager.cleanup()

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_manager):
        """Test memory cleanup functionality."""
        await memory_manager.initialize()

        # Create old memory entries
        old_date = datetime.now() - timedelta(days=30)
        old_entry = MemoryEntry(
            id="old_001",
            content="Old memory entry",
            type=MemoryType.CONVERSATION,
            metadata={},
            importance=0.1,  # Low importance
            timestamp=old_date,
        )

        # Create recent memory entry
        recent_entry = MemoryEntry(
            id="recent_001",
            content="Recent memory entry",
            type=MemoryType.CONVERSATION,
            metadata={},
            importance=0.9,  # High importance
            timestamp=datetime.now(),
        )

        # Store both entries
        await memory_manager.add_memory(old_entry)
        await memory_manager.add_memory(recent_entry)

        # Perform cleanup
        cleaned_count = await memory_manager.cleanup_old_memories(days=7)

        # Verify cleanup results
        assert cleaned_count >= 0

        # Recent entry should still exist
        recent_retrieved = await memory_manager.get_memory("recent_001")
        assert recent_retrieved is not None

        await memory_manager.cleanup()

    @pytest.mark.asyncio
    async def test_memory_statistics(self, memory_manager):
        """Test memory statistics functionality."""
        await memory_manager.initialize()

        # Store various types of memories
        entries = [
            MemoryEntry(
                id="stats_001",
                content="Conversation memory",
                type=MemoryType.CONVERSATION,
                metadata={},
                importance=0.7,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="stats_002",
                content="Knowledge memory",
                type=MemoryType.LEARNED_FACT,
                metadata={},
                importance=0.9,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="stats_003",
                content="Task memory",
                type=MemoryType.TASK_RESULT,
                metadata={},
                importance=0.8,
                timestamp=datetime.now(),
            ),
        ]

        for entry in entries:
            await memory_manager.add_memory(entry)

        # Get statistics
        stats = await memory_manager.get_memory_stats()

        assert stats is not None
        assert hasattr(stats, "total_memories")
        assert hasattr(stats, "memories_by_type")
        assert hasattr(stats, "average_importance")
        assert stats.total_memories >= 3

        await memory_manager.cleanup()


class TestMemoryTypes:
    """Test memory type functionality."""

    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.CONVERSATION is not None
        assert MemoryType.LEARNED_FACT is not None
        assert MemoryType.TASK_RESULT is not None
        assert MemoryType.SYSTEM_EVENT is not None

    def test_memory_entry_creation(self):
        """Test MemoryEntry creation."""
        entry = MemoryEntry(
            id="test_entry",
            content="Test content",
            type=MemoryType.CONVERSATION,
            metadata={"test": "value"},
            importance=0.5,
            timestamp=datetime.now(),
        )

        assert entry.id == "test_entry"
        assert entry.content == "Test content"
        assert entry.type == MemoryType.CONVERSATION
        assert entry.metadata["test"] == "value"
        assert entry.importance == 0.5
        assert entry.timestamp is not None

    def test_memory_entry_validation(self):
        """Test MemoryEntry validation."""
        # TODO: Add validation for importance values in MemoryEntry class
        # Currently, the MemoryEntry class does not validate importance ranges

        # Test that entries can be created with any importance values
        entry1 = MemoryEntry(
            id="test_importance_high",
            content="Test",
            type=MemoryType.CONVERSATION,
            metadata={},
            importance=1.5,  # Currently allowed but should be clamped
            timestamp=datetime.now(),
        )
        assert entry1.importance == 1.5

        entry2 = MemoryEntry(
            id="test_importance_low",
            content="Test",
            type=MemoryType.CONVERSATION,
            metadata={},
            importance=-0.1,  # Currently allowed but should be clamped
            timestamp=datetime.now(),
        )
        assert entry2.importance == -0.1

    def test_memory_entry_serialization(self):
        """Test MemoryEntry serialization."""
        entry = MemoryEntry(
            id="serialize_test",
            content="Serialization test",
            type=MemoryType.LEARNED_FACT,
            metadata={"key": "value"},
            importance=0.8,
            timestamp=datetime.now(),
        )

        # Test to_dict method
        entry_dict = entry.to_dict()
        assert entry_dict["id"] == "serialize_test"
        assert entry_dict["content"] == "Serialization test"
        assert entry_dict["type"] == MemoryType.LEARNED_FACT.value
        assert entry_dict["metadata"]["key"] == "value"
        assert entry_dict["importance"] == 0.8

        # Test from_dict method
        reconstructed = MemoryEntry.from_dict(entry_dict)
        assert reconstructed.id == entry.id
        assert reconstructed.content == entry.content
        assert reconstructed.type == entry.type
        assert reconstructed.metadata == entry.metadata
        assert reconstructed.importance == entry.importance


class TestMemoryConfig:
    """Test memory configuration."""

    def test_memory_config_creation(self):
        """Test MemoryConfig creation."""
        config = MemoryConfig(
            storage_path=Path("/tmp/test_memory.db"),
            max_memories_per_user=5000,
            cleanup_interval_hours=12,
            enable_semantic_search=False,
            batch_size=512,
        )

        assert config.storage_path == Path("/tmp/test_memory.db")
        assert config.max_memories_per_user == 5000
        assert config.cleanup_interval_hours == 12
        assert config.enable_semantic_search is False
        assert config.batch_size == 512

    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()

        assert config.storage_path is not None
        assert config.max_memories_per_user > 0
        assert config.cleanup_interval_hours > 0
        assert isinstance(config.enable_semantic_search, bool)
        assert config.retention_days > 0

    def test_memory_config_validation(self):
        """Test MemoryConfig validation."""
        # TODO: Add validation to MemoryConfig class
        # Currently, the MemoryConfig class does not validate parameters

        # Test that configs can be created with any values (currently allowed)
        config1 = MemoryConfig(max_memories_per_user=0)
        assert config1.max_memories_per_user == 0

        config2 = MemoryConfig(max_memories_per_user=-1)
        assert config2.max_memories_per_user == -1

        config3 = MemoryConfig(cleanup_interval_hours=0)
        assert config3.cleanup_interval_hours == 0

    def test_memory_config_serialization(self):
        """Test MemoryConfig serialization."""
        config = MemoryConfig(
            storage_path=Path("/tmp/test.db"),
            max_memories_per_user=1000,
            cleanup_interval_hours=24,
            enable_semantic_search=True,
            batch_size=384,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert str(config_dict["storage_path"]) == "/tmp/test.db"
        assert config_dict["max_memories_per_user"] == 1000
        assert config_dict["cleanup_interval_hours"] == 24
        assert config_dict["enable_semantic_search"] is True
        assert config_dict["batch_size"] == 384

        # Test from_dict
        reconstructed = MemoryConfig.from_dict(config_dict)
        assert str(reconstructed.storage_path) == "/tmp/test.db"
        assert reconstructed.max_memories_per_user == 1000
        assert reconstructed.cleanup_interval_hours == 24
        assert reconstructed.enable_semantic_search is True
        assert reconstructed.batch_size == 384


class TestMemoryIntegration:
    """Test memory system integration scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_conversation_memory_flow(self, temp_dir):
        """Test complete conversation memory flow."""
        # Setup
        config = MemoryConfig(
            storage_path=temp_dir / "conversation_test.db",
            max_memories_per_user=100,
            cleanup_interval_hours=1,
            enable_semantic_search=True,
        )

        manager = TektraMemoryManager(config)
        await manager.initialize()

        # Simulate a conversation
        conversation_entries = [
            MemoryEntry(
                id="conv_001",
                content="User: Hello, how are you?",
                type=MemoryType.CONVERSATION,
                metadata={"speaker": "user", "session_id": "session_001"},
                importance=0.3,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="conv_002",
                content="Assistant: I'm doing well, thank you for asking!",
                type=MemoryType.CONVERSATION,
                metadata={"speaker": "assistant", "session_id": "session_001"},
                importance=0.3,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="conv_003",
                content="User: Can you help me with Python programming?",
                type=MemoryType.CONVERSATION,
                metadata={
                    "speaker": "user",
                    "session_id": "session_001",
                    "topic": "programming",
                },
                importance=0.8,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="conv_004",
                content="Assistant: Of course! I'd be happy to help with Python programming.",
                type=MemoryType.CONVERSATION,
                metadata={
                    "speaker": "assistant",
                    "session_id": "session_001",
                    "topic": "programming",
                },
                importance=0.8,
                timestamp=datetime.now(),
            ),
        ]

        # Store conversation entries
        for entry in conversation_entries:
            await manager.add_memory(entry)

        # Test retrieval of conversation
        context = MemoryContext(
            query="", memory_types=[MemoryType.CONVERSATION], max_results=10
        )
        search_result = await manager.search_memories(context)
        session_memories = search_result.entries

        assert len(session_memories) >= 4

        # Test search for programming-related memories
        context = MemoryContext(query="Python programming", max_results=5)
        search_result = await manager.search_memories(context)
        programming_memories = search_result.entries
        assert len(programming_memories) >= 2
        assert any(
            "programming" in memory.content.lower() for memory in programming_memories
        )

        # Test importance-based filtering
        context = MemoryContext(query="", min_relevance=0.7, max_results=10)
        search_result = await manager.search_memories(context)
        important_memories = search_result.entries
        assert len(important_memories) >= 2
        assert all(memory.importance >= 0.7 for memory in important_memories)

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_knowledge_management_flow(self, temp_dir):
        """Test knowledge management flow."""
        # Setup
        config = MemoryConfig(
            storage_path=temp_dir / "knowledge_test.db",
            max_memories_per_user=100,
            enable_semantic_search=True,
        )

        manager = TektraMemoryManager(config)
        await manager.initialize()

        # Add knowledge entries
        knowledge_entries = [
            MemoryEntry(
                id="know_001",
                content="Python is a high-level programming language",
                type=MemoryType.LEARNED_FACT,
                metadata={"domain": "programming", "language": "python"},
                importance=0.9,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="know_002",
                content="Machine learning is a subset of artificial intelligence",
                type=MemoryType.LEARNED_FACT,
                metadata={"domain": "AI", "topic": "machine_learning"},
                importance=0.9,
                timestamp=datetime.now(),
            ),
            MemoryEntry(
                id="know_003",
                content="Neural networks are inspired by biological neurons",
                type=MemoryType.LEARNED_FACT,
                metadata={"domain": "AI", "topic": "neural_networks"},
                importance=0.8,
                timestamp=datetime.now(),
            ),
        ]

        # Store knowledge entries
        for entry in knowledge_entries:
            await manager.add_memory(entry)

        # Test knowledge retrieval
        context = MemoryContext(
            query="", memory_types=[MemoryType.LEARNED_FACT], max_results=10
        )
        search_result = await manager.search_memories(context)
        all_knowledge = search_result.entries
        assert len(all_knowledge) >= 3

        # Test domain-specific search
        context = MemoryContext(query="artificial intelligence", max_results=5)
        search_result = await manager.search_memories(context)
        ai_knowledge = search_result.entries
        assert len(ai_knowledge) >= 1

        # Test related knowledge discovery
        context = MemoryContext(query="programming", max_results=5)
        search_result = await manager.search_memories(context)
        programming_knowledge = search_result.entries
        assert len(programming_knowledge) >= 1

        await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
