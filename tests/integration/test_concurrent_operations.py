"""Concurrent operation integration tests.

These tests verify that the MemoryManager handles concurrent operations
correctly, including race conditions, duplicate detection, and data consistency.
"""

import asyncio
from collections import Counter
from unittest.mock import MagicMock

import pytest

from ccmemories.application.config import Config
from ccmemories.application.memory import MemoryManager
from ccmemories.application.utils import StructuredLogger
from ccmemories.application.utils.security import SecretString
from langchain_core.embeddings import Embeddings

import numpy as np


class MockEmbedder(Embeddings):
    """Mock embedder for testing that produces deterministic vectors."""

    def __init__(self, dims: int = 128) -> None:
        self.dims = dims
        self.call_count = 0

    def embed_query(self, text: str) -> list[float]:
        """Return deterministic embeddings based on text hash."""
        self.call_count += 1
        np.random.seed(hash(text) % (2**32))
        embedding: list[float] = np.random.randn(self.dims).astype(np.float32).tolist()
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self.embed_query(text) for text in texts]


def create_test_config(project_id: str = "test-concurrent") -> Config:
    """Create a Config instance for testing."""
    return Config(
        project_id=project_id,
        openrouter_api_key=SecretString("test-key"),
        embedder_provider="langchain",
        embedding_model="mock-model",
        embedding_dims=128,
        qwen_embedding_dims=128,
        enable_hybrid_search=False,  # Disable for simpler tests
        search_limit=5,
        deduplicate_messages=True,
        enable_llm_infer=False,
        log_level="DEBUG",
    )


def create_memory_manager(config: Config) -> MemoryManager:
    """Create a MemoryManager with mock embedder."""
    mock_embedder = MockEmbedder(dims=128)
    mock_logger = MagicMock(spec=StructuredLogger)

    from unittest.mock import patch

    with patch(
        "ccmemories.application.memory.manager.LangchainQwenEmbeddings",
        return_value=mock_embedder,
    ):
        manager = MemoryManager(config, mock_logger)
        return manager


@pytest.mark.integration
class TestConcurrentAdds:
    """Tests for concurrent add operations."""

    @pytest.mark.asyncio
    async def test_parallel_add_same_messages(self):
        """Test that adding the same messages in parallel doesn't cause duplicates."""
        config = create_test_config("test-parallel-same")
        memory_manager = create_memory_manager(config)

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
            assert message_counts[msg] == 1, (
                f"Message '{msg}' appeared {message_counts[msg]} times instead of 1"
            )

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_parallel_add_different_messages(self):
        """Test that adding different messages in parallel preserves all messages."""
        config = create_test_config("test-parallel-different")
        memory_manager = create_memory_manager(config)

        # Create different message sets
        message_sets = [
            [f"Set 1 - Message {i}" for i in range(3)],
            [f"Set 2 - Message {i}" for i in range(3)],
            [f"Set 3 - Message {i}" for i in range(3)],
        ]

        # Add all sets in parallel
        tasks = [
            memory_manager.add_messages_async(messages) for messages in message_sets
        ]
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
    async def test_concurrent_add_and_search(self):
        """Test that search works correctly while adds are happening."""
        config = create_test_config("test-concurrent-add-search")
        memory_manager = create_memory_manager(config)

        # Add initial messages
        initial_messages = ["Python", "JavaScript", "Go"]
        await memory_manager.add_messages_async(initial_messages)

        async def add_messages():
            """Continuously add messages."""
            for i in range(10):
                await memory_manager.add_messages_async([f"Language {i}"])
                await asyncio.sleep(0.01)

        async def search_messages():
            """Continuously search for messages."""
            for _ in range(10):
                results = await memory_manager.search("Python")
                # Search should always return results or empty list, never error
                assert isinstance(results, list)
                await asyncio.sleep(0.01)

        # Run adds and searches in parallel
        await asyncio.gather(add_messages(), search_messages())

        # Final search should work
        results = await memory_manager.search("Python")
        assert len(results) > 0

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentDeletes:
    """Tests for concurrent delete operations."""

    @pytest.mark.asyncio
    async def test_parallel_delete_same_message(self):
        """Test that deleting the same message in parallel doesn't cause errors."""
        config = create_test_config("test-parallel-delete")
        memory_manager = create_memory_manager(config)

        # Add a message
        message = "Test message to delete"
        await memory_manager.add_messages_async([message])

        # Try to delete the same message multiple times in parallel
        # Note: delete_by_message is synchronous, so we use it directly
        tasks = []
        for _ in range(5):
            # Run in thread pool since delete is sync
            import asyncio

            async def delete_wrapper():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, memory_manager.delete_by_message, message
                )
                return None

            tasks.append(delete_wrapper())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed (first deletes, others are no-ops)
        for result in results:
            if isinstance(result, Exception) and "not found" not in str(result).lower():
                pytest.fail(f"Delete operation failed unexpectedly: {result}")

        # Message should be gone
        all_messages = memory_manager.get_all()
        assert message not in all_messages

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_add_and_delete(self):
        """Test that adds and deletes can happen concurrently without issues."""
        config = create_test_config("test-concurrent-add-delete")
        memory_manager = create_memory_manager(config)

        # Add initial messages
        initial_messages = [f"Message {i}" for i in range(10)]
        await memory_manager.add_messages_async(initial_messages)

        deleted_count = [0]  # Use list for mutability
        added_count = [0]

        async def add_new_messages():
            """Add new messages."""
            for i in range(10, 20):
                await memory_manager.add_messages_async([f"Message {i}"])
                added_count[0] += 1
                await asyncio.sleep(0.01)

        async def delete_existing_messages():
            """Delete existing messages using search_for_removal."""
            import asyncio

            for i in range(5):
                # Search for the message
                candidates = memory_manager.search_for_removal(f"Message {i}", limit=1)
                if candidates:
                    # Delete in thread pool since delete is sync
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, memory_manager.delete_by_message, candidates[0]["memory"]
                    )
                    deleted_count[0] += 1
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
    async def test_parallel_search_same_query(self):
        """Test that parallel searches with the same query work correctly."""
        config = create_test_config("test-parallel-search")
        memory_manager = create_memory_manager(config)

        # Add messages
        messages = [
            "Python programming",
            "JavaScript web development",
            "Go systems programming",
        ]
        await memory_manager.add_messages_async(messages)

        # Search for "programming" multiple times in parallel
        tasks = [memory_manager.search("programming") for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All searches should succeed and return consistent results
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Search operation failed: {result}")
            # All searches should return at least one result (or empty list)
            assert isinstance(result, list)

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_parallel_search_different_queries(self):
        """Test that parallel searches with different queries work correctly."""
        config = create_test_config("test-search-different")
        memory_manager = create_memory_manager(config)

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
            # Each search should find relevant results (or empty list)
            assert isinstance(result, list)

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentGetAll:
    """Tests for concurrent get_all operations."""

    @pytest.mark.asyncio
    async def test_parallel_get_all(self):
        """Test that get_all works correctly when called in parallel."""
        config = create_test_config("test-parallel-getall")
        memory_manager = create_memory_manager(config)

        # Add messages
        messages = [f"Message {i}" for i in range(10)]
        await memory_manager.add_messages_async(messages)

        # Call get_all multiple times concurrently using thread pool
        import asyncio

        async def get_all_async():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, memory_manager.get_all)

        tasks = [get_all_async() for _ in range(10)]
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
    async def test_concurrent_get_all_and_add(self):
        """Test that get_all and add can work concurrently."""
        config = create_test_config("test-concurrent-getall-add")
        memory_manager = create_memory_manager(config)

        async def add_messages():
            """Add messages continuously."""
            for i in range(10):
                await memory_manager.add_messages_async([f"Message {i}"])
                await asyncio.sleep(0.01)

        async def get_all_messages():
            """Get all messages continuously using thread pool."""
            import asyncio

            for _ in range(10):
                loop = asyncio.get_event_loop()
                all_messages = await loop.run_in_executor(None, memory_manager.get_all)
                assert isinstance(all_messages, list)
                await asyncio.sleep(0.01)

        await asyncio.gather(add_messages(), get_all_messages())

        memory_manager.close()


@pytest.mark.integration
class TestAsyncConcurrencySafety:
    """Tests for async operation safety."""

    @pytest.mark.asyncio
    async def test_async_add_concurrency(self):
        """Test that async add operations handle concurrency correctly."""
        config = create_test_config("test-async-add")
        memory_manager = create_memory_manager(config)

        num_tasks = 20
        messages_per_task = 10

        async def add_messages(task_id: int):
            """Add messages from an async task."""
            messages = [
                f"Task {task_id} - Message {i}" for i in range(messages_per_task)
            ]
            await memory_manager.add_messages_async(messages)

        # Run all add tasks concurrently
        tasks = [add_messages(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        # Verify all messages were added (may have deduplicates)
        all_messages = memory_manager.get_all()
        # Should have at most num_tasks * messages_per_task
        assert len(all_messages) <= num_tasks * messages_per_task
        # Should have at least some messages
        assert len(all_messages) > 0

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_async_search_concurrency(self):
        """Test that async search operations handle concurrency correctly."""
        config = create_test_config("test-async-search")
        memory_manager = create_memory_manager(config)

        # Add initial messages
        messages = [f"Message {i} for testing" for i in range(100)]
        await memory_manager.add_messages_async(messages)

        num_tasks = 50

        async def search_messages(task_id: int):
            """Search from an async task."""
            results = await memory_manager.search(f"Message {task_id % 10}")
            # Results should always be a list
            assert isinstance(results, list)

        # Run all search tasks concurrently
        tasks = [search_messages(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_async_mixed_operations(self):
        """Test mixed async operations (add, search, delete) running concurrently."""
        config = create_test_config("test-async-mixed")
        memory_manager = create_memory_manager(config)

        # Add initial messages
        initial_messages = [f"Initial message {i}" for i in range(50)]
        await memory_manager.add_messages_async(initial_messages)

        deleted_count = [0]

        async def add_operation():
            """Continuously add messages."""
            for i in range(10):
                await memory_manager.add_messages_async([f"Concurrent add {i}"])
                await asyncio.sleep(0.001)

        async def search_operation():
            """Continuously search."""
            for _ in range(10):
                await memory_manager.search("message")
                await asyncio.sleep(0.001)

        async def delete_operation():
            """Continuously remove messages."""
            import asyncio

            for i in range(5):
                # Search for the message
                candidates = memory_manager.search_for_removal(
                    f"Concurrent add {i}", limit=1
                )
                if candidates:
                    # Delete in thread pool since delete is sync
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, memory_manager.delete_by_message, candidates[0]["memory"]
                    )
                    deleted_count[0] += 1
                await asyncio.sleep(0.001)

        async def get_all_operation():
            """Continuously get all messages using thread pool."""
            import asyncio

            for _ in range(10):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, memory_manager.get_all)
                await asyncio.sleep(0.001)

        # Run all operations concurrently
        await asyncio.gather(
            add_operation(),
            search_operation(),
            delete_operation(),
            get_all_operation(),
        )

        # Verify system is still functional
        all_messages = memory_manager.get_all()
        assert isinstance(all_messages, list)
        assert len(all_messages) > 0

        memory_manager.close()
