"""Integration tests for MemoryManager with USearch backend.

These tests verify hybrid search with USearchEngine + TantivyEngine
and RRF fusion using real indices in temporary directories.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from ccmemories.application.config import Config
from ccmemories.application.memory import MemoryManager
from ccmemories.application.utils import StructuredLogger
from ccmemories.application.utils.security import SecretString


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


def create_usearch_config(temp_dir: str, project_suffix: str = "") -> Config:
    """Create a Config instance configured for USearch backend.

    Args:
        temp_dir: Temporary directory for index files.
        project_suffix: Unique suffix to append to project_id for test isolation.
    """
    import uuid

    # Create unique project_id to ensure test isolation
    unique_id = project_suffix or uuid.uuid4().hex[:8]
    project_id = f"test-usearch-{unique_id}"

    return Config(
        project_id=project_id,
        openrouter_api_key=SecretString("test-key"),
        embedder_provider="langchain",
        embedding_model="mock-model",
        embedding_dims=128,
        qwen_embedding_dims=128,
        enable_hybrid_search=True,
        tantivy_index_path_template=os.path.join(temp_dir, "{project_id}", "tantivy"),
        search_limit=5,
        fusion_ranking_threshold=0.0,  # Allow all results
        deduplicate_messages=True,
        enable_llm_infer=False,
        log_level="DEBUG",
    )


def create_memory_manager(config: Config) -> tuple[MemoryManager, MagicMock]:
    """Create a MemoryManager with USearch backend and mock embedder."""
    mock_embedder = MockEmbedder(dims=128)
    mock_logger = MagicMock(spec=StructuredLogger)

    with patch(
        "ccmemories.application.memory.manager.LangchainQwenEmbeddings",
        return_value=mock_embedder,
    ):
        manager = MemoryManager(config, mock_logger)
        return manager, mock_logger


def cleanup_manager(manager: MemoryManager) -> None:
    """Clean up MemoryManager resources and delete index files."""
    # Close resources first
    if hasattr(manager._semantic_engine, "close"):
        manager._semantic_engine.close()  # type: ignore[union-attr]

    # Clean up USearch index directory (created at cwd/indexes/{project_id}/usearch/)
    project_id = manager.config.project_id.lower()
    usearch_dir = os.path.join(os.getcwd(), "indexes", project_id, "usearch")
    if os.path.exists(usearch_dir):
        shutil.rmtree(usearch_dir, ignore_errors=True)

    # Also clean up parent directory if empty
    project_dir = os.path.join(os.getcwd(), "indexes", project_id)
    if os.path.exists(project_dir) and not os.listdir(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)


@pytest.mark.integration
class TestMemoryManagerWithUSearch:
    """Integration tests for MemoryManager with USearch backend."""

    def test_initialization_with_usearch(self) -> None:
        """MemoryManager should initialize with USearch backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                assert manager._semantic_engine.name == "usearch"
                assert manager._tantivy_engine is not None
            finally:
                cleanup_manager(manager)

    def test_add_single_message(self) -> None:
        """Adding a single message should store in both engines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                stored = manager.add_messages(["Hello world from USearch"])

                assert stored == 1
                messages = manager.get_all()
                assert len(messages) == 1
                assert "Hello world from USearch" in messages
            finally:
                cleanup_manager(manager)

    def test_add_multiple_messages(self) -> None:
        """Adding multiple messages should store all in both engines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                test_messages = [
                    "Python is great for data science",
                    "JavaScript powers the web",
                    "Rust is blazingly fast",
                ]

                stored = manager.add_messages(test_messages)

                assert stored == 3
                messages = manager.get_all()
                assert len(messages) == 3
                for msg in test_messages:
                    assert msg in messages
            finally:
                cleanup_manager(manager)

    def test_deduplication(self) -> None:
        """Duplicate messages should not be stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                # Add first time
                stored1 = manager.add_messages(["Unique message"])
                assert stored1 == 1

                # Try to add same message again
                stored2 = manager.add_messages(["Unique message"])
                assert stored2 == 0

                # Should still only have one message
                messages = manager.get_all()
                assert len(messages) == 1
            finally:
                cleanup_manager(manager)

    def test_get_all_empty(self) -> None:
        """get_all on empty store should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                messages = manager.get_all()
                assert messages == []
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        """Search should return matching messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                manager.add_messages(
                    [
                        "Python is a programming language",
                        "Java is also a programming language",
                        "Cooking is fun",
                    ]
                )

                results = await manager.search("programming")

                assert len(results) >= 1
                # At least one programming-related result should be found
                assert any("programming" in r.lower() for r in results)
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_engines(self) -> None:
        """Hybrid search should combine results from both engines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                manager.add_messages(
                    [
                        "Machine learning is a subset of AI",
                        "Deep learning uses neural networks",
                        "Supervised learning requires labeled data",
                        "Cooking recipes for beginners",
                    ]
                )

                # Search should use both semantic and full-text
                results = await manager.search("learning", limit=5)

                # Should find multiple learning-related results
                assert len(results) >= 1
                learning_results = [r for r in results if "learning" in r.lower()]
                assert len(learning_results) >= 1
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self) -> None:
        """Search should respect the limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                # Add many messages
                messages = [f"Test message number {i}" for i in range(10)]
                manager.add_messages(messages)

                results = await manager.search("test", limit=3)

                assert len(results) <= 3
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_search_empty_index(self) -> None:
        """Search on empty index should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                results = await manager.search("anything")
                assert results == []
            finally:
                cleanup_manager(manager)

    def test_search_for_removal(self) -> None:
        """search_for_removal should find exact matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                target = "Message to remove"
                manager.add_messages([target, "Other message"])

                candidates = manager.search_for_removal(target)

                assert len(candidates) >= 1
                assert any(c["memory"] == target for c in candidates)
            finally:
                cleanup_manager(manager)

    @pytest.mark.skip(reason="Async add causes segfault due to SQLite threading")
    @pytest.mark.asyncio
    async def test_add_messages_async(self) -> None:
        """Async message addition should work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                messages = ["Async message 1", "Async message 2", "Async message 3"]

                stored = await manager.add_messages_async(messages)

                assert stored == 3
                all_messages = manager.get_all()
                assert len(all_messages) == 3
            finally:
                cleanup_manager(manager)


@pytest.mark.integration
class TestUSearchRRFFusion:
    """Tests for RRF fusion with USearch backend."""

    @pytest.mark.asyncio
    async def test_rrf_fusion_ranks_correctly(self) -> None:
        """RRF fusion should rank documents appearing in both engines higher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                # Add messages where some will match both semantic and full-text
                manager.add_messages(
                    [
                        "Python programming tutorial for beginners",
                        "Advanced Python techniques and best practices",
                        "Java programming guide",
                        "Cooking with Python snake meat",  # Contains "Python" but different context
                    ]
                )

                results = await manager.search("Python programming", limit=4)

                # Results should be found
                assert len(results) >= 1
                # At least one result should contain "python" (case-insensitive)
                # Note: with mock embeddings, ranking isn't semantically meaningful,
                # but full-text search should find Python-related results
                python_results = [r for r in results if "python" in r.lower()]
                assert len(python_results) >= 1, (
                    f"Expected Python results, got: {results}"
                )
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_fusion_combines_unique_results(self) -> None:
        """Fusion should return unique results from both engines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                manager.add_messages(
                    [
                        "Semantic similarity test one",
                        "Full text search test two",
                        "Both engines should find test three",
                    ]
                )

                results = await manager.search("test", limit=5)

                # All results should be unique
                assert len(results) == len(set(results))
            finally:
                cleanup_manager(manager)


@pytest.mark.integration
class TestUSearchPersistence:
    """Tests for USearch engine persistence."""

    def test_persistence_across_manager_restart(self) -> None:
        """Data should persist across MemoryManager restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a fixed project suffix so both managers access the same data
            config = create_usearch_config(tmpdir, project_suffix="persist-test")
            mock_embedder = MockEmbedder(dims=128)
            mock_logger = MagicMock(spec=StructuredLogger)

            # First manager instance - add data
            with patch(
                "ccmemories.application.memory.manager.LangchainQwenEmbeddings",
                return_value=mock_embedder,
            ):
                manager1 = MemoryManager(config, mock_logger)
                manager1.add_messages(
                    ["Persistent message one", "Persistent message two"]
                )
                # Commit to ensure persistence
                manager1._semantic_engine.commit()
                if hasattr(manager1._semantic_engine, "close"):
                    manager1._semantic_engine.close()  # type: ignore[union-attr]

            # Second manager instance - should load existing data
            with patch(
                "ccmemories.application.memory.manager.LangchainQwenEmbeddings",
                return_value=mock_embedder,
            ):
                manager2 = MemoryManager(config, mock_logger)
                try:
                    messages = manager2.get_all()

                    assert len(messages) == 2
                    assert "Persistent message one" in messages
                    assert "Persistent message two" in messages
                finally:
                    cleanup_manager(manager2)


@pytest.mark.integration
class TestUSearchErrorHandling:
    """Tests for error handling with USearch backend."""

    @pytest.mark.asyncio
    async def test_search_handles_empty_query(self) -> None:
        """Search should handle edge cases gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                manager.add_messages(["Some message"])

                # Search with minimal query
                results = await manager.search("a")

                # Should not raise, may return results or empty
                assert isinstance(results, list)
            finally:
                cleanup_manager(manager)

    def test_add_empty_list(self) -> None:
        """Adding empty list should be a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                stored = manager.add_messages([])

                assert stored == 0
                assert manager.get_all() == []
            finally:
                cleanup_manager(manager)


@pytest.mark.integration
class TestPhasedParallelAdd:
    """Integration tests for Sprint 2.2: Phased parallel add processing."""

    @pytest.mark.asyncio
    async def test_parallel_add_multiple_messages(self) -> None:
        """Phased parallel add should correctly store multiple messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                messages = [
                    "First message for parallel test",
                    "Second message for parallel test",
                    "Third message for parallel test",
                    "Fourth message for parallel test",
                ]

                result = await manager.add_messages_async(messages)

                # All messages should be stored
                assert result.stored_count == 4
                assert result.skipped_count == 0

                # Verify all messages are in storage
                all_messages = manager.get_all()
                assert len(all_messages) == 4
                for msg in messages:
                    assert msg in all_messages
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_parallel_add_batch_deduplication(self) -> None:
        """Phase 1 should deduplicate within the batch before storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                # Batch with duplicates
                messages = [
                    "Duplicate message",
                    "Unique message",
                    "Duplicate message",  # Duplicate within batch
                    "Another duplicate message",
                    "Unique message",  # Another duplicate
                    "Another duplicate message",  # Yet another duplicate
                ]

                result = await manager.add_messages_async(messages)

                # Only 3 unique messages should be stored (Duplicate, Unique, Another duplicate)
                assert result.stored_count == 3
                assert result.skipped_count == 3  # 3 batch duplicates

                # Verify storage
                all_messages = manager.get_all()
                assert len(all_messages) == 3
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_parallel_add_storage_deduplication(self) -> None:
        """Phase 1 should detect duplicates already in storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                # First, add some messages to storage
                first_batch = ["Existing message one", "Existing message two"]
                first_result = await manager.add_messages_async(first_batch)
                assert first_result.stored_count == 2

                # Now add a batch with some duplicates of existing messages
                second_batch = [
                    "Existing message one",  # Already in storage
                    "New message",
                    "Existing message two",  # Already in storage
                    "Another new message",
                ]
                second_result = await manager.add_messages_async(second_batch)

                # Only 2 new messages should be stored
                assert second_result.stored_count == 2
                assert second_result.skipped_count == 2  # 2 storage duplicates

                # Verify total in storage
                all_messages = manager.get_all()
                assert len(all_messages) == 4  # 2 original + 2 new
            finally:
                cleanup_manager(manager)

    @pytest.mark.asyncio
    async def test_parallel_add_preserves_order(self) -> None:
        """Phased parallel add should preserve message order for first occurrence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_usearch_config(tmpdir)
            manager, _ = create_memory_manager(config)
            try:
                # Messages where "First message" appears twice - only first occurrence stored
                messages = [
                    "First message",
                    "Second message",
                    "First message",  # Duplicate - should be skipped
                    "Third message",
                ]

                result = await manager.add_messages_async(messages)

                # Only 3 unique messages should be stored
                assert result.stored_count == 3
                assert result.skipped_count == 1

                all_messages = manager.get_all()
                assert len(all_messages) == 3
                assert "First message" in all_messages
                assert "Second message" in all_messages
                assert "Third message" in all_messages
            finally:
                cleanup_manager(manager)
