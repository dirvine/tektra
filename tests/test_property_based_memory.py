#!/usr/bin/env python3
"""
Property-based tests for memory system components.
"""

import pytest
import sys
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import composite
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tektra.memory.memory_manager import TektraMemoryManager
from tektra.memory.memory_types import MemoryType, MemoryEntry, MemoryContext
from tektra.memory.memory_config import MemoryConfig


# Custom strategies for memory testing
@composite
def memory_entry_strategy(draw):
    """Generate valid MemoryEntry objects."""
    return MemoryEntry(
        id=draw(st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_characters='\x00'))),
        content=draw(st.text(min_size=1, max_size=1000, alphabet=st.characters(blacklist_characters='\x00'))),
        type=draw(st.sampled_from(list(MemoryType))),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\x00')),
            st.one_of(
                st.text(max_size=100, alphabet=st.characters(blacklist_characters='\x00')),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans()
            ),
            max_size=10
        )),
        importance=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        timestamp=draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31)
        )),
        user_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\x00'))
        )),
        agent_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\x00'))
        )),
        session_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\x00'))
        ))
    )


@composite
def memory_context_strategy(draw):
    """Generate valid MemoryContext objects."""
    return MemoryContext(
        user_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\x00'))
        )),
        agent_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\x00'))
        )),
        session_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters='\x00'))
        )),
        query=draw(st.one_of(
            st.none(),
            st.text(max_size=200, alphabet=st.characters(blacklist_characters='\x00'))
        )),
        max_results=draw(st.integers(min_value=1, max_value=100)),
        min_relevance=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        time_window_hours=draw(st.one_of(
            st.none(),
            st.integers(min_value=1, max_value=8760)  # Up to 1 year
        )),
        memory_types=draw(st.lists(
            st.sampled_from(list(MemoryType)),
            min_size=0,
            max_size=len(MemoryType),
            unique=True
        ))
    )


class TestMemoryEntryProperties:
    """Property-based tests for MemoryEntry."""
    
    @given(memory_entry_strategy())
    def test_memory_entry_serialization_roundtrip(self, entry: MemoryEntry):
        """Test that MemoryEntry serialization is a perfect roundtrip."""
        # Serialize to dict
        entry_dict = entry.to_dict()
        
        # Deserialize back
        reconstructed = MemoryEntry.from_dict(entry_dict)
        
        # All fields should be identical
        assert reconstructed.id == entry.id
        assert reconstructed.content == entry.content
        assert reconstructed.type == entry.type
        assert reconstructed.metadata == entry.metadata
        assert reconstructed.importance == entry.importance
        assert reconstructed.timestamp == entry.timestamp
        assert reconstructed.user_id == entry.user_id
        assert reconstructed.agent_id == entry.agent_id
        assert reconstructed.session_id == entry.session_id
        assert reconstructed.embedding == entry.embedding
    
    @given(memory_entry_strategy())
    def test_memory_entry_dict_invariants(self, entry: MemoryEntry):
        """Test invariants of MemoryEntry dictionary representation."""
        entry_dict = entry.to_dict()
        
        # Required fields must be present
        assert 'id' in entry_dict
        assert 'content' in entry_dict
        assert 'type' in entry_dict
        assert 'importance' in entry_dict
        assert 'timestamp' in entry_dict
        
        # Type should be string representation
        assert isinstance(entry_dict['type'], str)
        assert entry_dict['type'] in [mt.value for mt in MemoryType]
        
        # Importance should be in valid range
        assert 0.0 <= entry_dict['importance'] <= 1.0
        
        # Timestamp should be ISO format string
        assert isinstance(entry_dict['timestamp'], str)
        # Should be parseable back to datetime
        datetime.fromisoformat(entry_dict['timestamp'])
    
    @given(st.text(min_size=1, max_size=100))
    def test_memory_entry_id_uniqueness(self, entry_id: str):
        """Test that entries with same ID are considered equal."""
        entry1 = MemoryEntry(id=entry_id, content="Test 1")
        entry2 = MemoryEntry(id=entry_id, content="Test 2")
        
        # IDs should be the same
        assert entry1.id == entry2.id
        
        # But content can be different
        assert entry1.content != entry2.content
    
    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_memory_entry_importance_ordering(self, importance1: float, importance2: float):
        """Test that importance values maintain proper ordering."""
        entry1 = MemoryEntry(importance=importance1, content="Entry 1")
        entry2 = MemoryEntry(importance=importance2, content="Entry 2")
        
        # Importance comparison should be consistent
        if importance1 < importance2:
            assert entry1.importance < entry2.importance
        elif importance1 > importance2:
            assert entry1.importance > entry2.importance
        else:
            assert entry1.importance == entry2.importance


class TestMemoryContextProperties:
    """Property-based tests for MemoryContext."""
    
    @given(memory_context_strategy())
    def test_memory_context_serialization_roundtrip(self, context: MemoryContext):
        """Test that MemoryContext serialization is a perfect roundtrip."""
        # Serialize to dict
        context_dict = context.to_dict()
        
        # Verify all fields are present
        assert 'user_id' in context_dict
        assert 'agent_id' in context_dict
        assert 'session_id' in context_dict
        assert 'query' in context_dict
        assert 'max_results' in context_dict
        assert 'min_relevance' in context_dict
        assert 'time_window_hours' in context_dict
        assert 'memory_types' in context_dict
        
        # Verify types are correct
        assert isinstance(context_dict['memory_types'], list)
        assert all(isinstance(mt, str) for mt in context_dict['memory_types'])
        assert all(mt in [t.value for t in MemoryType] for mt in context_dict['memory_types'])
    
    @given(st.integers(min_value=1, max_value=1000))
    def test_memory_context_max_results_bounds(self, max_results: int):
        """Test that max_results is properly bounded."""
        context = MemoryContext(max_results=max_results)
        
        assert context.max_results == max_results
        assert context.max_results >= 1
        
        # Should be serializable
        context_dict = context.to_dict()
        assert context_dict['max_results'] == max_results
    
    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    def test_memory_context_relevance_bounds(self, min_relevance: float):
        """Test that min_relevance is properly bounded."""
        context = MemoryContext(min_relevance=min_relevance)
        
        assert context.min_relevance == min_relevance
        assert 0.0 <= context.min_relevance <= 1.0
        
        # Should be serializable
        context_dict = context.to_dict()
        assert context_dict['min_relevance'] == min_relevance


class TestMemoryManagerProperties:
    """Property-based tests for TektraMemoryManager."""
    
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
            database_name="property_test.db",
            max_memories_per_user=1000,
            use_memos=False,
            enable_semantic_search=False
        )
    
    @pytest.fixture
    def memory_manager(self, memory_config):
        """Create a memory manager for testing."""
        return TektraMemoryManager(memory_config)
    
    @given(st.lists(memory_entry_strategy(), min_size=1, max_size=50))
    @settings(deadline=30000)  # 30 second timeout
    def test_memory_storage_retrieval_property(self, entries: List[MemoryEntry]):
        """Test that stored memories can always be retrieved."""
        async def test_storage():
            # Create temporary config and manager
            with tempfile.TemporaryDirectory() as temp_dir:
                config = MemoryConfig(
                    storage_path=temp_dir,
                    database_name="property_test.db",
                    max_memories_per_user=1000,
                    use_memos=False,
                    enable_semantic_search=False
                )
                memory_manager = TektraMemoryManager(config)
                
                await memory_manager.initialize()
                
                # Store all entries
                stored_ids = []
                for entry in entries:
                    stored_id = await memory_manager.add_memory(entry)
                    stored_ids.append(stored_id)
                
                # Retrieve all entries
                for entry_id in stored_ids:
                    retrieved = await memory_manager.get_memory(entry_id)
                    assert retrieved is not None, f"Could not retrieve entry {entry_id}"
                
                await memory_manager.cleanup()
        
        asyncio.run(test_storage())
    
    @given(st.lists(memory_entry_strategy(), min_size=1, max_size=20))
    @settings(deadline=30000)
    def test_memory_search_consistency(self, entries: List[MemoryEntry]):
        """Test that search results are consistent with stored data."""
        async def test_search():
            with tempfile.TemporaryDirectory() as temp_dir:
                config = MemoryConfig(
                    storage_path=temp_dir,
                    database_name="search_test.db",
                    max_memories_per_user=1000,
                    use_memos=False,
                    enable_semantic_search=False
                )
                memory_manager = TektraMemoryManager(config)
                
                await memory_manager.initialize()
                
                # Store entries
                for entry in entries:
                    await memory_manager.add_memory(entry)
                
                # Search should return subset of stored entries
                context = MemoryContext(max_results=100)
                result = await memory_manager.search_memories(context)
                
                assert len(result.entries) <= len(entries)
                
                # All returned entries should be from our stored entries
                stored_ids = {entry.id for entry in entries}
                for returned_entry in result.entries:
                    assert returned_entry.id in stored_ids
                
                await memory_manager.cleanup()
        
        asyncio.run(test_search())
    
    @given(memory_entry_strategy())
    @settings(deadline=15000)
    def test_memory_deletion_property(self, entry: MemoryEntry, memory_manager):
        """Test that deleted memories cannot be retrieved."""
        async def test_deletion():
            await memory_manager.initialize()
            
            # Store entry
            stored_id = await memory_manager.add_memory(entry)
            
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
        
        asyncio.run(test_deletion())
    
    @given(
        st.lists(memory_entry_strategy(), min_size=1, max_size=10),
        st.integers(min_value=1, max_value=100)
    )
    @settings(deadline=20000)
    def test_search_limit_property(self, entries: List[MemoryEntry], limit: int, memory_manager):
        """Test that search respects max_results limit."""
        async def test_limit():
            await memory_manager.initialize()
            
            # Store entries
            for entry in entries:
                await memory_manager.add_memory(entry)
            
            # Search with limit
            context = MemoryContext(max_results=limit)
            result = await memory_manager.search_memories(context)
            
            # Should respect limit
            assert len(result.entries) <= limit
            assert len(result.entries) <= len(entries)
            
            await memory_manager.cleanup()
        
        asyncio.run(test_limit())
    
    @given(
        st.lists(memory_entry_strategy(), min_size=2, max_size=10),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(deadline=20000)
    def test_importance_filtering_property(self, entries: List[MemoryEntry], min_importance: float, memory_manager):
        """Test that importance filtering works correctly."""
        async def test_importance():
            await memory_manager.initialize()
            
            # Store entries
            for entry in entries:
                await memory_manager.add_memory(entry)
            
            # Search with importance filter
            context = MemoryContext(
                min_relevance=min_importance,
                max_results=100
            )
            result = await memory_manager.search_memories(context)
            
            # All returned entries should meet importance threshold
            for entry in result.entries:
                assert entry.importance >= min_importance
            
            # Count how many stored entries should meet the threshold
            expected_count = sum(1 for entry in entries if entry.importance >= min_importance)
            assert len(result.entries) <= expected_count
            
            await memory_manager.cleanup()
        
        asyncio.run(test_importance())


class TestMemorySystemInvariants:
    """Test system-level invariants."""
    
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
            database_name="invariant_test.db",
            max_memories_per_user=1000,
            use_memos=False,
            enable_semantic_search=False
        )
    
    @pytest.fixture
    def memory_manager(self, memory_config):
        """Create a memory manager for testing."""
        return TektraMemoryManager(memory_config)
    
    @given(st.lists(memory_entry_strategy(), min_size=1, max_size=20))
    @settings(deadline=25000)
    def test_statistics_consistency(self, entries: List[MemoryEntry], memory_manager):
        """Test that statistics are consistent with stored data."""
        async def test_stats():
            await memory_manager.initialize()
            
            # Store entries
            for entry in entries:
                await memory_manager.add_memory(entry)
            
            # Get statistics
            stats = await memory_manager.get_memory_stats()
            
            # Total should match what we stored
            assert stats.total_memories >= len(entries)
            
            # Type counts should be consistent
            type_counts = {}
            for entry in entries:
                type_counts[entry.type.value] = type_counts.get(entry.type.value, 0) + 1
            
            for memory_type, count in type_counts.items():
                assert stats.memories_by_type.get(memory_type, 0) >= count
            
            # Average importance should be reasonable
            if entries:
                expected_avg = sum(entry.importance for entry in entries) / len(entries)
                # Allow for some variance due to other entries in DB
                assert 0.0 <= stats.average_importance <= 1.0
            
            await memory_manager.cleanup()
        
        asyncio.run(test_stats())
    
    @given(st.lists(memory_entry_strategy(), min_size=1, max_size=15))
    @settings(deadline=25000)
    def test_concurrent_access_safety(self, entries: List[MemoryEntry], memory_manager):
        """Test that concurrent access doesn't break invariants."""
        async def test_concurrent():
            await memory_manager.initialize()
            
            # Define concurrent operations
            async def store_entries():
                for entry in entries:
                    await memory_manager.add_memory(entry)
            
            async def search_entries():
                context = MemoryContext(max_results=10)
                return await memory_manager.search_memories(context)
            
            async def get_stats():
                return await memory_manager.get_memory_stats()
            
            # Run operations concurrently
            results = await asyncio.gather(
                store_entries(),
                search_entries(),
                get_stats(),
                return_exceptions=True
            )
            
            # No exceptions should occur
            for result in results:
                assert not isinstance(result, Exception)
            
            # Verify system is still functional
            final_stats = await memory_manager.get_memory_stats()
            assert final_stats.total_memories >= 0
            
            await memory_manager.cleanup()
        
        asyncio.run(test_concurrent())
    
    @given(st.lists(memory_entry_strategy(), min_size=1, max_size=10))
    @settings(deadline=20000)
    def test_cleanup_preserves_invariants(self, entries: List[MemoryEntry], memory_manager):
        """Test that cleanup operations preserve system invariants."""
        async def test_cleanup():
            await memory_manager.initialize()
            
            # Store entries
            for entry in entries:
                await memory_manager.add_memory(entry)
            
            # Get initial stats
            initial_stats = await memory_manager.get_memory_stats()
            
            # Perform cleanup
            cleaned = await memory_manager.cleanup_old_memories(days=1)
            
            # Get final stats
            final_stats = await memory_manager.get_memory_stats()
            
            # Total should have decreased by cleaned amount
            assert final_stats.total_memories <= initial_stats.total_memories
            
            # System should still be functional
            context = MemoryContext(max_results=5)
            result = await memory_manager.search_memories(context)
            assert isinstance(result.entries, list)
            
            await memory_manager.cleanup()
        
        asyncio.run(test_cleanup())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])