#!/usr/bin/env python3
"""
Integration tests for the memory system components.
"""

import pytest
import sys
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tektra.memory.memory_manager import TektraMemoryManager
from tektra.memory.memory_types import MemoryType, MemoryEntry, MemoryContext
from tektra.memory.memory_config import MemoryConfig


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
            use_memos=False  # Disable for testing
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
    async def test_store_and_retrieve_memory(self, memory_manager):
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
            timestamp=datetime.now()
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
                timestamp=datetime.now()
            ),
            MemoryEntry(
                id="search_002",
                content="Machine learning algorithms",
                type=MemoryType.LEARNED_FACT,
                metadata={"topic": "AI"},
                importance=0.8,
                timestamp=datetime.now()
            ),
            MemoryEntry(
                id="search_003",
                content="User said hello",
                type=MemoryType.CONVERSATION,
                metadata={"speaker": "user"},
                importance=0.3,
                timestamp=datetime.now()
            )
        ]
        
        for entry in entries:
            await memory_manager.add_memory(entry)
        
        # Test search by content
        search_context = MemoryContext(
            query="programming",
            max_results=5
        )
        result = await memory_manager.search_memories(search_context)
        assert len(result.entries) >= 1
        assert any("programming" in entry.content.lower() for entry in result.entries)
        
        # Test search by memory type
        knowledge_context = MemoryContext(
            memory_types=[MemoryType.LEARNED_FACT],
            max_results=10
        )
        knowledge_result = await memory_manager.search_memories(knowledge_context)
        assert len(knowledge_result.entries) >= 2
        assert all(entry.type == MemoryType.LEARNED_FACT for entry in knowledge_result.entries)
        
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
            memory_type=MemoryType.CONVERSATION,
            metadata={},
            importance=0.1,  # Low importance
            created_at=old_date
        )
        
        # Create recent memory entry
        recent_entry = MemoryEntry(
            id="recent_001",
            content="Recent memory entry",
            memory_type=MemoryType.CONVERSATION,
            metadata={},
            importance=0.9,  # High importance
            created_at=datetime.now()
        )
        
        # Store both entries
        await memory_manager.store_memory(old_entry)
        await memory_manager.store_memory(recent_entry)
        
        # Perform cleanup
        cleaned_count = await memory_manager.cleanup_old_memories(max_age_days=7)
        
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
                memory_type=MemoryType.CONVERSATION,
                metadata={},
                importance=0.7,
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="stats_002",
                content="Knowledge memory",
                memory_type=MemoryType.KNOWLEDGE,
                metadata={},
                importance=0.9,
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="stats_003",
                content="Task memory",
                memory_type=MemoryType.TASK,
                metadata={},
                importance=0.8,
                created_at=datetime.now()
            )
        ]
        
        for entry in entries:
            await memory_manager.store_memory(entry)
        
        # Get statistics
        stats = await memory_manager.get_memory_statistics()
        
        assert stats is not None
        assert "total_memories" in stats
        assert "memory_types" in stats
        assert "average_importance" in stats
        assert stats["total_memories"] >= 3
        
        await memory_manager.cleanup()


class TestMemoryTypes:
    """Test memory type functionality."""
    
    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.CONVERSATION is not None
        assert MemoryType.KNOWLEDGE is not None
        assert MemoryType.TASK is not None
        assert MemoryType.SYSTEM is not None
    
    def test_memory_entry_creation(self):
        """Test MemoryEntry creation."""
        entry = MemoryEntry(
            id="test_entry",
            content="Test content",
            memory_type=MemoryType.CONVERSATION,
            metadata={"test": "value"},
            importance=0.5,
            created_at=datetime.now()
        )
        
        assert entry.id == "test_entry"
        assert entry.content == "Test content"
        assert entry.memory_type == MemoryType.CONVERSATION
        assert entry.metadata["test"] == "value"
        assert entry.importance == 0.5
        assert entry.created_at is not None
    
    def test_memory_entry_validation(self):
        """Test MemoryEntry validation."""
        # Test invalid importance values
        with pytest.raises(ValueError):
            MemoryEntry(
                id="invalid_importance",
                content="Test",
                memory_type=MemoryType.CONVERSATION,
                metadata={},
                importance=1.5,  # Invalid: > 1.0
                created_at=datetime.now()
            )
        
        with pytest.raises(ValueError):
            MemoryEntry(
                id="invalid_importance2",
                content="Test",
                memory_type=MemoryType.CONVERSATION,
                metadata={},
                importance=-0.1,  # Invalid: < 0.0
                created_at=datetime.now()
            )
    
    def test_memory_entry_serialization(self):
        """Test MemoryEntry serialization."""
        entry = MemoryEntry(
            id="serialize_test",
            content="Serialization test",
            memory_type=MemoryType.KNOWLEDGE,
            metadata={"key": "value"},
            importance=0.8,
            created_at=datetime.now()
        )
        
        # Test to_dict method
        entry_dict = entry.to_dict()
        assert entry_dict["id"] == "serialize_test"
        assert entry_dict["content"] == "Serialization test"
        assert entry_dict["memory_type"] == MemoryType.KNOWLEDGE.value
        assert entry_dict["metadata"]["key"] == "value"
        assert entry_dict["importance"] == 0.8
        
        # Test from_dict method
        reconstructed = MemoryEntry.from_dict(entry_dict)
        assert reconstructed.id == entry.id
        assert reconstructed.content == entry.content
        assert reconstructed.memory_type == entry.memory_type
        assert reconstructed.metadata == entry.metadata
        assert reconstructed.importance == entry.importance


class TestMemoryConfig:
    """Test memory configuration."""
    
    def test_memory_config_creation(self):
        """Test MemoryConfig creation."""
        config = MemoryConfig(
            storage_path=Path("/tmp/test_memory.db"),
            max_memory_entries=5000,
            cleanup_interval_hours=12,
            enable_vector_search=False,
            vector_dimensions=512
        )
        
        assert config.storage_path == Path("/tmp/test_memory.db")
        assert config.max_memory_entries == 5000
        assert config.cleanup_interval_hours == 12
        assert config.enable_vector_search is False
        assert config.vector_dimensions == 512
    
    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()
        
        assert config.storage_path is not None
        assert config.max_memory_entries > 0
        assert config.cleanup_interval_hours > 0
        assert isinstance(config.enable_vector_search, bool)
        assert config.vector_dimensions > 0
    
    def test_memory_config_validation(self):
        """Test MemoryConfig validation."""
        # Test invalid max_memory_entries
        with pytest.raises(ValueError):
            MemoryConfig(max_memory_entries=0)
        
        with pytest.raises(ValueError):
            MemoryConfig(max_memory_entries=-1)
        
        # Test invalid cleanup_interval_hours
        with pytest.raises(ValueError):
            MemoryConfig(cleanup_interval_hours=0)
        
        with pytest.raises(ValueError):
            MemoryConfig(cleanup_interval_hours=-1)
        
        # Test invalid vector_dimensions
        with pytest.raises(ValueError):
            MemoryConfig(vector_dimensions=0)
        
        with pytest.raises(ValueError):
            MemoryConfig(vector_dimensions=-1)
    
    def test_memory_config_serialization(self):
        """Test MemoryConfig serialization."""
        config = MemoryConfig(
            storage_path=Path("/tmp/test.db"),
            max_memory_entries=1000,
            cleanup_interval_hours=24,
            enable_vector_search=True,
            vector_dimensions=384
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["storage_path"] == "/tmp/test.db"
        assert config_dict["max_memory_entries"] == 1000
        assert config_dict["cleanup_interval_hours"] == 24
        assert config_dict["enable_vector_search"] is True
        assert config_dict["vector_dimensions"] == 384
        
        # Test from_dict
        reconstructed = MemoryConfig.from_dict(config_dict)
        assert reconstructed.storage_path == Path("/tmp/test.db")
        assert reconstructed.max_memory_entries == 1000
        assert reconstructed.cleanup_interval_hours == 24
        assert reconstructed.enable_vector_search is True
        assert reconstructed.vector_dimensions == 384


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
            max_memory_entries=100,
            cleanup_interval_hours=1,
            enable_vector_search=True
        )
        
        manager = TektraMemoryManager(config)
        await manager.initialize()
        
        # Simulate a conversation
        conversation_entries = [
            MemoryEntry(
                id="conv_001",
                content="User: Hello, how are you?",
                memory_type=MemoryType.CONVERSATION,
                metadata={"speaker": "user", "session_id": "session_001"},
                importance=0.3,
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="conv_002",
                content="Assistant: I'm doing well, thank you for asking!",
                memory_type=MemoryType.CONVERSATION,
                metadata={"speaker": "assistant", "session_id": "session_001"},
                importance=0.3,
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="conv_003",
                content="User: Can you help me with Python programming?",
                memory_type=MemoryType.CONVERSATION,
                metadata={"speaker": "user", "session_id": "session_001", "topic": "programming"},
                importance=0.8,
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="conv_004",
                content="Assistant: Of course! I'd be happy to help with Python programming.",
                memory_type=MemoryType.CONVERSATION,
                metadata={"speaker": "assistant", "session_id": "session_001", "topic": "programming"},
                importance=0.8,
                created_at=datetime.now()
            )
        ]
        
        # Store conversation entries
        for entry in conversation_entries:
            await manager.store_memory(entry)
        
        # Test retrieval of conversation
        session_memories = await manager.search_memories(
            "",
            memory_type=MemoryType.CONVERSATION,
            limit=10
        )
        
        assert len(session_memories) >= 4
        
        # Test search for programming-related memories
        programming_memories = await manager.search_memories("Python programming", limit=5)
        assert len(programming_memories) >= 2
        assert any("programming" in memory.content.lower() for memory in programming_memories)
        
        # Test importance-based filtering
        important_memories = await manager.search_memories(
            "",
            min_importance=0.7,
            limit=10
        )
        assert len(important_memories) >= 2
        assert all(memory.importance >= 0.7 for memory in important_memories)
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_knowledge_management_flow(self, temp_dir):
        """Test knowledge management flow."""
        # Setup
        config = MemoryConfig(
            storage_path=temp_dir / "knowledge_test.db",
            max_memory_entries=100,
            enable_vector_search=True
        )
        
        manager = TektraMemoryManager(config)
        await manager.initialize()
        
        # Add knowledge entries
        knowledge_entries = [
            MemoryEntry(
                id="know_001",
                content="Python is a high-level programming language",
                memory_type=MemoryType.KNOWLEDGE,
                metadata={"domain": "programming", "language": "python"},
                importance=0.9,
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="know_002",
                content="Machine learning is a subset of artificial intelligence",
                memory_type=MemoryType.KNOWLEDGE,
                metadata={"domain": "AI", "topic": "machine_learning"},
                importance=0.9,
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="know_003",
                content="Neural networks are inspired by biological neurons",
                memory_type=MemoryType.KNOWLEDGE,
                metadata={"domain": "AI", "topic": "neural_networks"},
                importance=0.8,
                created_at=datetime.now()
            )
        ]
        
        # Store knowledge entries
        for entry in knowledge_entries:
            await manager.store_memory(entry)
        
        # Test knowledge retrieval
        all_knowledge = await manager.search_memories(
            "",
            memory_type=MemoryType.KNOWLEDGE,
            limit=10
        )
        assert len(all_knowledge) >= 3
        
        # Test domain-specific search
        ai_knowledge = await manager.search_memories("artificial intelligence", limit=5)
        assert len(ai_knowledge) >= 1
        
        # Test related knowledge discovery
        programming_knowledge = await manager.search_memories("programming", limit=5)
        assert len(programming_knowledge) >= 1
        
        await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])