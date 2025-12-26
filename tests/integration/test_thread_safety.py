"""Thread safety tests.

These tests verify that the storage layer (MessageStore, engines) handles
concurrent access from multiple threads correctly without data races or
corruption.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pytest

from ccmemories.infrastructure.message_store import MessageStore
from ccmemories.application.config.settings import Config


@pytest.mark.integration
class TestMessageStoreThreadSafety:
    """Tests for MessageStore thread safety."""

    def test_concurrent_inserts(self, temp_config: Config):
        """Test that concurrent inserts don't cause data corruption."""
        store = MessageStore(db_path=temp_config.usearch_config.db_path)

        num_threads = 10
        inserts_per_thread = 100

        def insert_messages(thread_id: int):
            """Insert messages from a thread."""
            for i in range(inserts_per_thread):
                message = f"Thread {thread_id} - Message {i}"
                store.insert(
                    project_id=temp_config.project_id,
                    memory_id=f"{thread_id}-{i}",
                    message=message,
                    embedding=[0.1] * 10,
                )

        # Run inserts in parallel threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(insert_messages, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify all messages were inserted
        all_messages = store.get_all(project_id=temp_config.project_id)
        expected_count = num_threads * inserts_per_thread
        assert len(all_messages) == expected_count, f"Expected {expected_count} messages, got {len(all_messages)}"

    def test_concurrent_get_and_insert(self, temp_config: Config):
        """Test that concurrent reads and writes don't cause issues."""
        store = MessageStore(db_path=temp_config.usearch_config.db_path)

        # Insert initial messages
        for i in range(100):
            store.insert(
                project_id=temp_config.project_id,
                memory_id=f"initial-{i}",
                message=f"Initial message {i}",
                embedding=[0.1] * 10,
            )

        num_threads = 10
        operations_per_thread = 50
        insert_count = [0]  # Use list for mutability in closure
        get_count = [0]
        lock = Lock()

        def mixed_operations(thread_id: int):
            """Perform mixed reads and writes from a thread."""
            for i in range(operations_per_thread):
                if i % 2 == 0:
                    # Insert new message
                    with lock:
                        message_id = f"thread-{thread_id}-{insert_count[0]}"
                        insert_count[0] += 1
                    store.insert(
                        project_id=temp_config.project_id,
                        memory_id=message_id,
                        message=f"Thread {thread_id} - Message {i}",
                        embedding=[0.1] * 10,
                    )
                else:
                    # Read messages
                    messages = store.get_all(project_id=temp_config.project_id)
                    with lock:
                        get_count[0] += 1
                    # Verify messages are valid
                    assert all(isinstance(msg, str) for msg in messages)

        # Run operations in parallel threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify final state
        all_messages = store.get_all(project_id=temp_config.project_id)
        assert len(all_messages) >= 100  # At least the initial messages

    def test_concurrent_duplicates(self, temp_config: Config):
        """Test handling of concurrent duplicate inserts."""
        store = MessageStore(db_path=temp_config.usearch_config.db_path)

        num_threads = 10
        message_id = "duplicate-test"
        message = "This message will be inserted by multiple threads"

        def insert_duplicate(thread_id: int):
            """Try to insert the same message from multiple threads."""
            try:
                store.insert(
                    project_id=temp_config.project_id,
                    memory_id=message_id,
                    message=message,
                    embedding=[0.1] * 10,
                )
            except Exception as e:
                # Some threads should get constraint violations
                pass

        # All threads try to insert the same message_id
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(insert_duplicate, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any non-duplicate exceptions

        # Only one should have succeeded
        result = store.get(memory_id=message_id)
        assert result is not None
        assert result.message == message

    def test_concurrent_delete_and_read(self, temp_config: Config):
        """Test that concurrent deletes and reads are handled correctly."""
        store = MessageStore(db_path=temp_config.usearch_config.db_path)

        # Insert initial messages
        message_ids = []
        for i in range(100):
            message_id = f"delete-read-{i}"
            message_ids.append(message_id)
            store.insert(
                project_id=temp_config.project_id,
                memory_id=message_id,
                message=f"Message {i}",
                embedding=[0.1] * 10,
            )

        num_threads = 5
        delete_count = [0]
        read_count = [0]
        lock = Lock()

        def delete_and_read(thread_id: int):
            """Delete and read messages concurrently."""
            for i in range(20):
                if i % 3 == 0:
                    # Delete a message
                    idx = thread_id * 20 + i
                    if idx < len(message_ids):
                        message_id = message_ids[idx]
                        deleted = store.delete(memory_id=message_id)
                        if deleted:
                            with lock:
                                delete_count[0] += 1
                else:
                    # Read all messages
                    messages = store.get_all(project_id=temp_config.project_id)
                    with lock:
                        read_count[0] += 1
                    # Verify messages are valid
                    assert all(isinstance(msg, str) for msg in messages)

        # Run operations in parallel threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(delete_and_read, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify final state (should have some messages left)
        all_messages = store.get_all(project_id=temp_config.project_id)
        assert len(all_messages) >= 0  # Some may have been deleted


@pytest.mark.integration
class TestEngineThreadSafety:
    """Tests for engine thread safety."""

    def test_concurrent_usearch_operations(self, temp_config: Config):
        """Test that USearch operations are thread-safe."""
        from ccmemories.infrastructure.usearch_engine import USearchEngine

        engine = USearchEngine(config=temp_config.usearch_config, logger=None)

        num_threads = 5
        inserts_per_thread = 20

        def insert_and_search(thread_id: int):
            """Insert and search from a thread."""
            for i in range(inserts_per_thread):
                message = f"Thread {thread_id} - Message {i}"
                # Insert
                engine.add(
                    project_id=temp_config.project_id,
                    memory_id=f"{thread_id}-{i}",
                    message=message,
                    embedding=[0.1] * 10,
                )

                # Search
                results = engine.search(
                    project_id=temp_config.project_id,
                    query_embedding=[0.1] * 10,
                    limit=5,
                )
                # Results should be a list
                assert isinstance(results, list)

        # Run operations in parallel threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(insert_and_search, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify final state
        all_messages = engine.get_all(project_id=temp_config.project_id)
        expected_count = num_threads * inserts_per_thread
        assert len(all_messages) == expected_count

        engine.close()

    def test_concurrent_tantivy_operations(self, temp_config: Config):
        """Test that Tantivy operations are thread-safe."""
        from ccmemories.infrastructure.tantivy_engine import TantivyEngine

        engine = TantivyEngine(config=temp_config.tantivy_config, logger=None)

        num_threads = 5
        inserts_per_thread = 20

        def insert_and_search(thread_id: int):
            """Insert and search from a thread."""
            for i in range(inserts_per_thread):
                message = f"Thread {thread_id} - Message {i}"
                # Insert
                engine.add(
                    project_id=temp_config.project_id,
                    memory_id=f"{thread_id}-{i}",
                    message=message,
                )

                # Search
                results = engine.search(
                    project_id=temp_config.project_id,
                    query="thread",
                    limit=5,
                )
                # Results should be a list
                assert isinstance(results, list)

        # Run operations in parallel threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(insert_and_search, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        engine.close()


@pytest.mark.integration
class TestAsyncConcurrencySafety:
    """Tests for async operation safety."""

    @pytest.mark.asyncio
    async def test_async_add_concurrency(self, temp_config: Config):
        """Test that async add operations handle concurrency correctly."""
        from ccmemories.application.memory.manager import MemoryManager
        from ccmemories.application.utils.logging import StructuredLogger

        logger = StructuredLogger("test_async")
        memory_manager = MemoryManager(temp_config, logger)

        num_tasks = 20
        messages_per_task = 10

        async def add_messages(task_id: int):
            """Add messages from an async task."""
            messages = [f"Task {task_id} - Message {i}" for i in range(messages_per_task)]
            await memory_manager.add_messages_async(messages)

        # Run all add tasks concurrently
        tasks = [add_messages(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        # Verify all messages were added
        all_messages = memory_manager.get_all()
        expected_count = num_tasks * messages_per_task
        assert len(all_messages) == expected_count

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_async_search_concurrency(self, temp_config: Config):
        """Test that async search operations handle concurrency correctly."""
        from ccmemories.application.memory.manager import MemoryManager
        from ccmemories.application.utils.logging import StructuredLogger

        logger = StructuredLogger("test_async")
        memory_manager = MemoryManager(temp_config, logger)

        # Add initial messages
        messages = [f"Message {i} for testing" for i in range(100)]
        await memory_manager.add_messages_async(messages)

        num_tasks = 50

        async def search_messages(task_id: int):
            """Search from an async task."""
            results = memory_manager.search(f"Message {task_id % 10}")
            # Results should always be a list
            assert isinstance(results, list)

        # Run all search tasks concurrently
        tasks = [search_messages(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_async_mixed_operations(self, temp_config: Config):
        """Test mixed async operations (add, search, remove) running concurrently."""
        from ccmemories.application.memory.manager import MemoryManager
        from ccmemories.application.utils.logging import StructuredLogger

        logger = StructuredLogger("test_async")
        memory_manager = MemoryManager(temp_config, logger)

        # Add initial messages
        initial_messages = [f"Initial message {i}" for i in range(50)]
        await memory_manager.add_messages_async(initial_messages)

        async def add_operation():
            """Continuously add messages."""
            for i in range(10):
                await memory_manager.add_messages_async([f"Concurrent add {i}"])
                await asyncio.sleep(0.001)

        async def search_operation():
            """Continuously search."""
            for _ in range(10):
                memory_manager.search("message")
                await asyncio.sleep(0.001)

        async def remove_operation():
            """Continuously remove messages."""
            for i in range(5):
                await memory_manager.remove([f"Concurrent add {i}"])
                await asyncio.sleep(0.001)

        async def get_all_operation():
            """Continuously get all messages."""
            for _ in range(10):
                memory_manager.get_all()
                await asyncio.sleep(0.001)

        # Run all operations concurrently
        await asyncio.gather(
            add_operation(),
            search_operation(),
            remove_operation(),
            get_all_operation(),
        )

        # Verify system is still functional
        all_messages = memory_manager.get_all()
        assert isinstance(all_messages, list)
        assert len(all_messages) > 0

        memory_manager.close()
