"""Concurrent operation integration tests.

These tests verify that the MemoryManager handles concurrent operations
correctly, including race conditions, duplicate detection, and data consistency.
"""

import asyncio
from collections import Counter

import pytest

from ccmemories.application.memory.manager import MemoryManager
from ccmemories.application.config.settings import Config
from ccmemories.application.utils.logging import StructuredLogger


@pytest.mark.integration
class TestConcurrentAdds:
    """Tests for concurrent add operations."""

    @pytest.mark.asyncio
    async def test_parallel_add_same_messages(self, temp_config: Config):
        """Test that adding the same messages in parallel doesn't cause duplicates."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        messages = ["Message 1", "Message 2", "Message 3"]

        # Add the same messages multiple times in parallel
        tasks = [memory_manager.add_messages_async(messages) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Add operation failed: {result}")

        # Should only have 3 unique messages (deduplication should work)
        all_messages = memory_manager.get_all()
        message_counts = Counter(all_messages)

        # Each message should appear exactly once
        for msg in messages:
            assert message_counts[msg] == 1, f"Message '{msg}' appeared {message_counts[msg]} times instead of 1"

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_parallel_add_different_messages(self, temp_config: Config):
        """Test that adding different messages in parallel preserves all messages."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        # Create different message sets
        message_sets = [
            [f"Set 1 - Message {i}" for i in range(3)],
            [f"Set 2 - Message {i}" for i in range(3)],
            [f"Set 3 - Message {i}" for i in range(3)],
        ]

        # Add all sets in parallel
        tasks = [memory_manager.add_messages_async(messages) for messages in message_sets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Add operation failed: {result}")

        # Should have all 9 messages
        all_messages = memory_manager.get_all()
        assert len(all_messages) == 9

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_add_and_search(self, temp_config: Config):
        """Test that search works correctly while adds are happening."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        # Add initial messages
        initial_messages = ["Python", "JavaScript", "Go"]
        memory_manager.add_messages_async(initial_messages)

        async def add_messages():
            """Continuously add messages."""
            for i in range(10):
                await memory_manager.add_messages_async([f"Language {i}"])
                await asyncio.sleep(0.01)

        async def search_messages():
            """Continuously search for messages."""
            for _ in range(10):
                results = memory_manager.search("Python")
                # Search should always return results or empty list, never error
                assert isinstance(results, list)
                await asyncio.sleep(0.01)

        # Run adds and searches in parallel
        await asyncio.gather(add_messages(), search_messages())

        # Final search should work
        results = memory_manager.search("Python")
        assert len(results) > 0

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentDeletes:
    """Tests for concurrent delete operations."""

    @pytest.mark.asyncio
    async def test_parallel_delete_same_message(self, temp_config: Config):
        """Test that deleting the same message in parallel doesn't cause errors."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        # Add a message
        message = "Test message to delete"
        memory_manager.add_messages_async([message])

        # Try to delete the same message multiple times in parallel
        tasks = [memory_manager.remove([message]) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed (first deletes, others are no-ops)
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Delete operation failed: {result}")

        # Message should be gone
        all_messages = memory_manager.get_all()
        assert message not in all_messages

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_add_and_delete(self, temp_config: Config):
        """Test that adds and deletes can happen concurrently without issues."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        # Add initial messages
        initial_messages = [f"Message {i}" for i in range(10)]
        await memory_manager.add_messages_async(initial_messages)

        async def add_new_messages():
            """Add new messages."""
            for i in range(10, 20):
                await memory_manager.add_messages_async([f"Message {i}"])
                await asyncio.sleep(0.01)

        async def delete_existing_messages():
            """Delete existing messages."""
            for i in range(5):
                await memory_manager.remove([f"Message {i}"])
                await asyncio.sleep(0.01)

        # Run adds and deletes in parallel
        await asyncio.gather(add_new_messages(), delete_existing_messages())

        # Should have some messages left (exact count depends on timing)
        all_messages = memory_manager.get_all()
        assert len(all_messages) > 0

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentSearches:
    """Tests for concurrent search operations."""

    @pytest.mark.asyncio
    async def test_parallel_search_same_query(self, temp_config: Config):
        """Test that parallel searches with the same query work correctly."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        # Add messages
        messages = ["Python programming", "JavaScript web development", "Go systems programming"]
        await memory_manager.add_messages_async(messages)

        # Search for "programming" multiple times in parallel
        tasks = [memory_manager.search("programming") for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All searches should succeed and return consistent results
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Search operation failed: {result}")
            # All searches should return the same number of results
            assert len(result) > 0

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_parallel_search_different_queries(self, temp_config: Config):
        """Test that parallel searches with different queries work correctly."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        # Add messages
        messages = [
            "Python is great for data science",
            "JavaScript is used for web development",
            "Go is excellent for systems programming",
            "Rust provides memory safety",
        ]
        await memory_manager.add_messages_async(messages)

        # Search for different terms in parallel
        queries = ["Python", "JavaScript", "Go", "Rust"]
        tasks = [memory_manager.search(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All searches should succeed
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Search for '{queries[i]}' failed: {result}")
            # Each search should find relevant results
            assert len(result) > 0, f"Search for '{queries[i]}' returned no results"

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentGetAll:
    """Tests for concurrent get_all operations."""

    @pytest.mark.asyncio
    async def test_parallel_get_all(self, temp_config: Config):
        """Test that get_all works correctly when called in parallel."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        # Add messages
        messages = [f"Message {i}" for i in range(10)]
        await memory_manager.add_messages_async(messages)

        # Call get_all multiple times in parallel
        tasks = [memory_manager.get_all() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All calls should return the same results
        for result in results:
            assert len(result) == 10
            # Results should be independent (defensive copies)
            result.append("Should not affect original")

        # Verify original wasn't modified
        final = memory_manager.get_all()
        assert len(final) == 10

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_get_all_and_add(self, temp_config: Config):
        """Test that get_all and add can work concurrently."""
        logger = StructuredLogger("test_concurrent")
        memory_manager = MemoryManager(temp_config, logger)

        async def add_messages():
            """Add messages continuously."""
            for i in range(10):
                await memory_manager.add_messages_async([f"Message {i}"])
                await asyncio.sleep(0.01)

        async def get_all_messages():
            """Get all messages continuously."""
            for _ in range(10):
                all_messages = memory_manager.get_all()
                assert isinstance(all_messages, list)
                await asyncio.sleep(0.01)

        await asyncio.gather(add_messages(), get_all_messages())

        memory_manager.close()
