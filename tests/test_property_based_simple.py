#!/usr/bin/env python3
"""
Simple property-based tests for memory system components.
"""

import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tektra.memory.memory_config import MemoryConfig
from tektra.memory.memory_manager import TektraMemoryManager
from tektra.memory.memory_types import MemoryContext, MemoryEntry, MemoryType


# Custom strategies for memory testing
@composite
def memory_entry_strategy(draw):
    """Generate valid MemoryEntry objects."""
    return MemoryEntry(
        id=draw(
            st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(
                    blacklist_characters="\x00", blacklist_categories=("Cs",)
                ),
            )
        ),
        content=draw(
            st.text(
                min_size=1,
                max_size=1000,
                alphabet=st.characters(
                    blacklist_characters="\x00", blacklist_categories=("Cs",)
                ),
            )
        ),
        type=draw(st.sampled_from(list(MemoryType))),
        metadata=draw(
            st.dictionaries(
                st.text(
                    min_size=1,
                    max_size=50,
                    alphabet=st.characters(
                        blacklist_characters="\x00", blacklist_categories=("Cs",)
                    ),
                ),
                st.one_of(
                    st.text(
                        max_size=100,
                        alphabet=st.characters(
                            blacklist_characters="\x00", blacklist_categories=("Cs",)
                        ),
                    ),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans(),
                ),
                max_size=10,
            )
        ),
        importance=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        timestamp=draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31)
            )
        ),
        user_id=draw(
            st.one_of(
                st.none(),
                st.text(
                    min_size=1,
                    max_size=50,
                    alphabet=st.characters(
                        blacklist_characters="\x00", blacklist_categories=("Cs",)
                    ),
                ),
            )
        ),
        agent_id=draw(
            st.one_of(
                st.none(),
                st.text(
                    min_size=1,
                    max_size=50,
                    alphabet=st.characters(
                        blacklist_characters="\x00", blacklist_categories=("Cs",)
                    ),
                ),
            )
        ),
        session_id=draw(
            st.one_of(
                st.none(),
                st.text(
                    min_size=1,
                    max_size=50,
                    alphabet=st.characters(
                        blacklist_characters="\x00", blacklist_categories=("Cs",)
                    ),
                ),
            )
        ),
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
        assert "id" in entry_dict
        assert "content" in entry_dict
        assert "type" in entry_dict
        assert "importance" in entry_dict
        assert "timestamp" in entry_dict

        # Type should be string representation
        assert isinstance(entry_dict["type"], str)
        assert entry_dict["type"] in [mt.value for mt in MemoryType]

        # Importance should be in valid range
        assert 0.0 <= entry_dict["importance"] <= 1.0

        # Timestamp should be ISO format string
        assert isinstance(entry_dict["timestamp"], str)
        # Should be parseable back to datetime
        datetime.fromisoformat(entry_dict["timestamp"])

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_memory_entry_importance_ordering(
        self, importance1: float, importance2: float
    ):
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

    @given(st.text(min_size=1, max_size=100))
    def test_memory_entry_id_uniqueness(self, entry_id: str):
        """Test that entries with same ID are considered equal."""
        entry1 = MemoryEntry(id=entry_id, content="Test 1")
        entry2 = MemoryEntry(id=entry_id, content="Test 2")

        # IDs should be the same
        assert entry1.id == entry2.id

        # But content can be different
        assert entry1.content != entry2.content


class TestMemoryContextProperties:
    """Property-based tests for MemoryContext."""

    @given(st.integers(min_value=1, max_value=1000))
    def test_memory_context_max_results_bounds(self, max_results: int):
        """Test that max_results is properly bounded."""
        context = MemoryContext(max_results=max_results)

        assert context.max_results == max_results
        assert context.max_results >= 1

        # Should be serializable
        context_dict = context.to_dict()
        assert context_dict["max_results"] == max_results

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_memory_context_relevance_bounds(self, min_relevance: float):
        """Test that min_relevance is properly bounded."""
        context = MemoryContext(min_relevance=min_relevance)

        assert context.min_relevance == min_relevance
        assert 0.0 <= context.min_relevance <= 1.0

        # Should be serializable
        context_dict = context.to_dict()
        assert context_dict["min_relevance"] == min_relevance

    @given(
        st.lists(
            st.sampled_from(list(MemoryType)),
            min_size=0,
            max_size=len(MemoryType),
            unique=True,
        )
    )
    def test_memory_context_type_filtering(self, memory_types):
        """Test that memory type filtering works correctly."""
        context = MemoryContext(memory_types=memory_types)

        assert context.memory_types == memory_types
        assert all(isinstance(mt, MemoryType) for mt in context.memory_types)

        # Should be serializable
        context_dict = context.to_dict()
        expected_types = [mt.value for mt in memory_types]
        assert context_dict["memory_types"] == expected_types


class TestMemorySystemProperties:
    """Property-based tests for the memory system."""

    @given(memory_entry_strategy())
    @settings(deadline=10000)
    def test_single_memory_storage_retrieval(self, entry: MemoryEntry):
        """Test that a single memory can be stored and retrieved."""

        async def test_storage():
            with tempfile.TemporaryDirectory() as temp_dir:
                config = MemoryConfig(
                    storage_path=temp_dir,
                    database_name="single_test.db",
                    max_memories_per_user=1000,
                    use_memos=False,
                    enable_semantic_search=False,
                )
                memory_manager = TektraMemoryManager(config)

                await memory_manager.initialize()

                # Store entry
                stored_id = await memory_manager.add_memory(entry)
                assert stored_id == entry.id

                # Retrieve entry
                retrieved = await memory_manager.get_memory(stored_id)
                assert retrieved is not None
                assert retrieved.id == entry.id
                assert retrieved.content == entry.content
                assert retrieved.type == entry.type
                assert retrieved.importance == entry.importance

                await memory_manager.cleanup()

        asyncio.run(test_storage())

    @given(memory_entry_strategy())
    @settings(deadline=10000)
    def test_memory_deletion_property(self, entry: MemoryEntry):
        """Test that deleted memories cannot be retrieved."""

        async def test_deletion():
            with tempfile.TemporaryDirectory() as temp_dir:
                config = MemoryConfig(
                    storage_path=temp_dir,
                    database_name="deletion_test.db",
                    max_memories_per_user=1000,
                    use_memos=False,
                    enable_semantic_search=False,
                )
                memory_manager = TektraMemoryManager(config)

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

    @given(st.lists(memory_entry_strategy(), min_size=1, max_size=10))
    @settings(deadline=15000)
    def test_batch_memory_operations(self, entries):
        """Test that multiple memories can be stored and retrieved."""

        async def test_batch():
            with tempfile.TemporaryDirectory() as temp_dir:
                config = MemoryConfig(
                    storage_path=temp_dir,
                    database_name="batch_test.db",
                    max_memories_per_user=1000,
                    use_memos=False,
                    enable_semantic_search=False,
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
                    assert retrieved is not None

                # Test search
                context = MemoryContext(max_results=100)
                result = await memory_manager.search_memories(context)
                assert len(result.entries) <= len(entries)

                # All returned entries should be from our stored entries
                stored_ids_set = set(stored_ids)
                for returned_entry in result.entries:
                    assert returned_entry.id in stored_ids_set

                await memory_manager.cleanup()

        asyncio.run(test_batch())

    @given(
        st.lists(memory_entry_strategy(), min_size=2, max_size=10),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=15000)
    def test_importance_filtering(self, entries, min_importance):
        """Test that importance filtering works correctly."""

        async def test_filtering():
            with tempfile.TemporaryDirectory() as temp_dir:
                config = MemoryConfig(
                    storage_path=temp_dir,
                    database_name="importance_test.db",
                    max_memories_per_user=1000,
                    use_memos=False,
                    enable_semantic_search=False,
                )
                memory_manager = TektraMemoryManager(config)

                await memory_manager.initialize()

                # Store entries
                for entry in entries:
                    await memory_manager.add_memory(entry)

                # Search with importance filter
                context = MemoryContext(min_relevance=min_importance, max_results=100)
                result = await memory_manager.search_memories(context)

                # All returned entries should meet importance threshold
                for entry in result.entries:
                    assert entry.importance >= min_importance

                # Count how many stored entries should meet the threshold
                expected_count = sum(
                    1 for entry in entries if entry.importance >= min_importance
                )
                assert len(result.entries) <= expected_count

                await memory_manager.cleanup()

        asyncio.run(test_filtering())

    @given(st.lists(memory_entry_strategy(), min_size=1, max_size=10))
    @settings(deadline=15000)
    def test_memory_statistics_consistency(self, entries):
        """Test that statistics are consistent with stored data."""

        async def test_stats():
            with tempfile.TemporaryDirectory() as temp_dir:
                config = MemoryConfig(
                    storage_path=temp_dir,
                    database_name="stats_test.db",
                    max_memories_per_user=1000,
                    use_memos=False,
                    enable_semantic_search=False,
                )
                memory_manager = TektraMemoryManager(config)

                await memory_manager.initialize()

                # Store entries
                for entry in entries:
                    await memory_manager.add_memory(entry)

                # Get statistics
                stats = await memory_manager.get_memory_stats()

                # Total should match what we stored (accounting for duplicate IDs)
                unique_ids = len({entry.id for entry in entries})
                assert stats.total_memories >= unique_ids

                # Type counts should be consistent (accounting for duplicate IDs)
                # Count unique entries by type
                unique_entries = {}
                for entry in entries:
                    unique_entries[entry.id] = entry

                type_counts = {}
                for entry in unique_entries.values():
                    type_counts[entry.type.value] = (
                        type_counts.get(entry.type.value, 0) + 1
                    )

                for memory_type, count in type_counts.items():
                    assert stats.memories_by_type.get(memory_type, 0) >= count

                # Average importance should be reasonable
                assert 0.0 <= stats.average_importance <= 1.0

                await memory_manager.cleanup()

        asyncio.run(test_stats())


class TestMemoryConfigProperties:
    """Property-based tests for MemoryConfig."""

    @given(st.integers(min_value=1, max_value=100000))
    def test_memory_config_limits(self, max_memories: int):
        """Test that memory config limits are properly handled."""
        config = MemoryConfig(
            max_memories_per_user=max_memories, max_memories_per_agent=max_memories // 2
        )

        assert config.max_memories_per_user == max_memories
        assert config.max_memories_per_agent == max_memories // 2

        # Should be serializable
        config_dict = config.to_dict()
        assert config_dict["max_memories_per_user"] == max_memories
        assert config_dict["max_memories_per_agent"] == max_memories // 2

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_memory_config_thresholds(self, threshold: float):
        """Test that memory config thresholds are properly handled."""
        config = MemoryConfig(
            default_importance_threshold=threshold, relevance_threshold=threshold
        )

        assert config.default_importance_threshold == threshold
        assert config.relevance_threshold == threshold

        # Should be serializable
        config_dict = config.to_dict()
        assert config_dict["default_importance_threshold"] == threshold
        assert config_dict["relevance_threshold"] == threshold

    @given(st.integers(min_value=1, max_value=365))
    def test_memory_config_retention(self, retention_days: int):
        """Test that memory config retention is properly handled."""
        config = MemoryConfig(retention_days=retention_days)

        assert config.retention_days == retention_days

        # Should be serializable
        config_dict = config.to_dict()
        assert config_dict["retention_days"] == retention_days

        # Should validate correctly
        errors = config.validate()
        # Should have no errors related to retention_days
        assert not any("retention_days" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
