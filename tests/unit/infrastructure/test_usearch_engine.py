"""Unit tests for USearchEngine."""

import os
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reflectlog.application.types import Embeddings
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine


class MockEmbedder(Embeddings):
    """Mock embedder for testing."""

    def __init__(self, dims: int = 128) -> None:
        self.dims = dims
        self.call_count = 0

    def embed_query(self, text: str) -> list[float]:
        """Return deterministic embeddings based on text hash."""
        self.call_count += 1
        # Create deterministic embedding from text hash
        np.random.seed(hash(text) % (2**32))
        embedding: list[float] = np.random.randn(self.dims).astype(np.float32).tolist()
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def temp_engine() -> Generator[tuple[USearchConfig, MockEmbedder, str], None, None]:
    """Create a temporary engine configuration and embedder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = USearchConfig(
            project_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
        )
        embedder = MockEmbedder(dims=128)
        yield config, embedder, tmpdir


class TestUSearchConfigFromAppConfig:
    """Tests for USearchConfig.from_app_config factory method."""

    def test_creates_config_from_app_config(self) -> None:
        """Factory should create config from application Config."""
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.embedder_provider = "openai"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_app_config(mock_config)

        assert config.project_id == "test-project"
        assert config.embedding_dims == 3072
        assert "test-project" in config.index_path
        assert "test-project" in config.db_path

    def test_uses_qwen_dims_for_langchain_provider(self) -> None:
        """Factory should use qwen_embedding_dims for langchain provider."""
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.embedder_provider = "langchain"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_app_config(mock_config)

        assert config.embedding_dims == 4096


class TestUSearchEngineInitialization:
    """Tests for USearchEngine initialization."""

    def test_engine_name_is_usearch(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Engine name should be 'usearch'."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            assert engine.name == "usearch"
        finally:
            engine.close()

    def test_lazy_initialization(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Index and message store should be lazily initialized."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Before accessing, files should not exist
            assert not os.path.exists(config.db_path)

            # After ensure_initialized, files should exist
            engine.ensure_initialized()
            assert os.path.exists(config.db_path)
        finally:
            engine.close()


class TestUSearchEngineAdd:
    """Tests for USearchEngine.add method."""

    def test_add_message(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Add should store message in both index and SQLite."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            # Message should be in SQLite
            messages = engine.get_all("user1")
            assert "Hello world" in messages

            # Embedder should have been called
            assert embedder.call_count == 1
        finally:
            engine.close()

    def test_add_skips_duplicates(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Add should skip duplicate messages."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Hello world", infer=False)  # Duplicate

            # Only one message should be stored
            messages = engine.get_all("user1")
            assert len(messages) == 1

            # Embedder should only be called once
            assert embedder.call_count == 1
        finally:
            engine.close()

    def test_add_with_infer_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Add with infer=True should log a warning."""
        config, embedder, _ = temp_engine
        logger = MagicMock()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "Hello world", infer=True)

            # Should log warning about infer not being supported
            logger.warning.assert_called()
        finally:
            engine.close()


class TestUSearchEngineSearch:
    """Tests for USearchEngine.search method."""

    def test_search_empty_index(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search on empty index should return empty list."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()

            results = engine.search("hello", "user1", limit=5)

            assert results == []
        finally:
            engine.close()

    def test_search_returns_matches(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should return matching messages with scores."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Goodbye world", infer=False)

            results = engine.search("Hello", "user1", limit=5)

            assert len(results) > 0
            # Results should be (message, score, created_at) tuples
            assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        finally:
            engine.close()

    def test_search_filters_by_project_id(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should only return messages for the specified project."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "User1 message", infer=False)
            engine.add("user2", "User2 message", infer=False)

            results = engine.search("message", "user1", limit=5)

            # Should only return user1's message
            messages = [r[0] for r in results]
            assert "User1 message" in messages
            assert "User2 message" not in messages
        finally:
            engine.close()

    def test_search_respects_limit(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should respect the limit parameter."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Add many messages
            for i in range(10):
                engine.add("user1", f"Message {i}", infer=False)

            results = engine.search("message", "user1", limit=3)

            assert len(results) <= 3
        finally:
            engine.close()


class TestUSearchEngineGetAll:
    """Tests for USearchEngine.get_all method."""

    def test_get_all_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Get all on empty store should return empty list."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            messages = engine.get_all("user1")
            assert messages == []
        finally:
            engine.close()

    def test_get_all_returns_messages(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Get all should return all messages for user."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Message 1", infer=False)
            engine.add("user1", "Message 2", infer=False)

            messages = engine.get_all("user1")

            assert len(messages) == 2
            assert "Message 1" in messages
            assert "Message 2" in messages
        finally:
            engine.close()


class TestUSearchEngineDelete:
    """Tests for USearchEngine.delete method."""

    def test_delete_existing_message(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Delete should remove message from both index and SQLite."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            # Get the message ID from SQLite
            msg_id = engine.message_store.get_id_by_message("user1", "Hello world")
            assert msg_id is not None

            engine.delete(str(msg_id))

            # Message should be removed
            messages = engine.get_all("user1")
            assert "Hello world" not in messages
        finally:
            engine.close()

    def test_delete_nonexistent_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Delete of non-existent message should log warning."""
        config, embedder, _ = temp_engine
        logger = MagicMock()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            engine.delete("999")

            # Should log warning about not found
            logger.warning.assert_called()
        finally:
            engine.close()

    def test_delete_invalid_id_raises_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Delete with invalid ID format should raise error."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()

            with pytest.raises(RuntimeError, match="Invalid memory_id format"):
                engine.delete("not-an-integer")
        finally:
            engine.close()


class TestUSearchEngineCommit:
    """Tests for USearchEngine.commit method."""

    def test_commit_saves_index(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Commit should save the USearch index to disk."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.commit()

            # Index file should exist
            assert os.path.exists(config.index_path)
        finally:
            engine.close()


class TestUSearchEnginePersistence:
    """Tests for USearchEngine persistence across restarts."""

    def test_persistence_across_restart(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Messages should persist across engine restart."""
        config, embedder, _ = temp_engine

        # First engine instance
        engine1 = USearchEngine(config=config, embedder=embedder)
        try:
            engine1.add("user1", "Persistent message", infer=False)
            engine1.commit()
        finally:
            engine1.close()

        # Second engine instance (simulating restart)
        engine2 = USearchEngine(config=config, embedder=embedder)
        try:
            # Message should still be accessible
            messages = engine2.get_all("user1")
            assert "Persistent message" in messages
        finally:
            engine2.close()


class TestUSearchEngineExactSearch:
    """Tests for USearchEngine exact vs approximate search modes."""

    def test_default_uses_exact_search(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Default config should use exact (brute-force) search for small databases."""
        config, embedder, _ = temp_engine
        # Note: temp_engine fixture uses explicit config, so we need to test with default
        # The default USearchConfig has exact_search=True
        default_config = USearchConfig(
            project_id=config.project_id,
            index_path=config.index_path,
            db_path=config.db_path,
            embedding_dims=config.embedding_dims,
            # Using defaults: exact_search=True, exact_search_threshold=0
        )
        engine = USearchEngine(config=default_config, embedder=embedder)

        try:
            # Default config: exact_search=True, exact_search_threshold=0
            assert default_config.exact_search is True
            assert default_config.exact_search_threshold == 0
            assert engine._should_use_exact_search() is True
        finally:
            engine.close()

    def test_exact_search_when_forced(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Should use exact search when exact_search=True."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            project_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            exact_search=True,  # Force exact search
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            assert engine._should_use_exact_search() is True
        finally:
            engine.close()

    def test_exact_search_auto_switch_below_threshold(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Should auto-switch to exact search when index size < threshold."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            project_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            exact_search=False,
            exact_search_threshold=1000,  # Auto-switch when < 1000 vectors
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Add a few messages (below threshold)
            for i in range(5):
                engine.add("user1", f"Message {i}", infer=False)

            # Index size is 5, below threshold of 1000
            assert len(engine.index) == 5
            assert engine._should_use_exact_search() is True
        finally:
            engine.close()

    def test_approximate_search_above_threshold(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Should use approximate search when index size >= threshold."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            project_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            exact_search=False,
            exact_search_threshold=3,  # Auto-switch when < 3 vectors
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Add messages to exceed threshold
            for i in range(5):
                engine.add("user1", f"Message {i}", infer=False)

            # Index size is 5, above threshold of 3
            assert len(engine.index) == 5
            assert engine._should_use_exact_search() is False
        finally:
            engine.close()

    def test_exact_search_returns_valid_results(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Exact search should return valid search results."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            project_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            exact_search=True,  # Force exact search
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Goodbye world", infer=False)

            results = engine.search("Hello", "user1", limit=5)

            # Should return valid results (3-tuples: message, score, created_at)
            assert len(results) > 0
            assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        finally:
            engine.close()

    def test_search_logs_search_mode(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should log the search mode being used."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            project_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            exact_search=True,
        )
        logger = MagicMock()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.search("Hello", "user1", limit=5)

            # Should have logged the search mode
            debug_calls = logger.debug.call_args_list
            search_mode_logged = any(
                "search mode" in str(call).lower() and "exact" in str(call).lower()
                for call in debug_calls
            )
            assert search_mode_logged, "Should log search mode as 'exact'"
        finally:
            engine.close()

    def test_config_from_app_config_includes_exact_search(self) -> None:
        """USearchConfig.from_app_config should include exact search settings."""
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.embedder_provider = "openai"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096
        mock_config.usearch_exact_search = True
        mock_config.usearch_exact_search_threshold = 5000

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_app_config(mock_config)

        assert config.exact_search is True
        assert config.exact_search_threshold == 5000
