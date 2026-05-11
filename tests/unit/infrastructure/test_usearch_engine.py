'''Unit tests for USearchEngine.'''

import os
import tempfile
from typing import Generator, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reflectlog.core.exceptions import StorageError
from reflectlog.core.types import Embeddings
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine


def create_mock_logger() -> IStructuredLogger:
    '''Create a properly typed mock logger for testing.'''
    return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))


class MockEmbedder(Embeddings):
    '''Mock embedder for testing.'''

    def __init__(self, dims: int = 128) -> None:
        super().__init__()
        self.dims = dims
        self.call_count = 0

    def embed_query(self, text: str) -> list[float]:
        '''Return deterministic embeddings based on text hash.'''
        self.call_count += 1
        np.random.seed(hash(text) % (2**32))
        embedding: list[float] = np.random.randn(self.dims).astype(np.float32).tolist()
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        '''Embed a list of documents.'''
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class FailingQueryEmbedder(MockEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("Embedding API down")


class FailingBatchEmbedder(MockEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding batch failed")


class MismatchedSizeEmbedder(MockEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dims]


class ToggleableFailingQueryEmbedder(MockEmbedder):
    def __init__(self, dims: int = 128) -> None:
        super().__init__(dims)
        self.should_fail = False

    def embed_query(self, text: str) -> list[float]:
        if self.should_fail:
            raise RuntimeError("API failure")
        return super().embed_query(text)


@pytest.fixture
def temp_engine() -> Generator[tuple[USearchConfig, MockEmbedder, str], None, None]:
    '''Create a temporary engine configuration and embedder.'''
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
    '''Tests for USearchConfig.from_app_config factory method.'''

    def test_creates_config_from_app_config(self) -> None:
        '''Factory should create config from application Config.'''
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.embedder_provider = "openai"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_config(mock_config)

        assert config.project_id == "test-project"
        assert config.embedding_dims == 3072
        assert "test-project" in config.index_path
        assert "test-project" in config.db_path

    def test_uses_qwen_dims_for_langchain_provider(self) -> None:
        '''Factory should use qwen_embedding_dims for langchain provider.'''
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.embedder_provider = "langchain"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_config(mock_config)

        assert config.embedding_dims == 4096


class TestUSearchEngineInitialization:
    '''Tests for USearchEngine initialization.'''

    def test_engine_name_is_usearch(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Engine name should be 'usearch'.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            assert engine.name == "usearch"
        finally:
            engine.close()

    def test_lazy_initialization(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Index and memory store should be lazily initialized.'''
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
    '''Tests for USearchEngine.add method.'''

    def test_add_content(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Add should store content in both index and SQLite.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            # Content should be in SQLite
            contents = engine.get_all("user1")
            assert "Hello world" in contents

            # Embedder should have been called
            assert embedder.call_count == 1
        finally:
            engine.close()

    def test_add_skips_duplicates(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Add should skip duplicate contents.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Hello world", infer=False)  # Duplicate

            # Only one content should be stored
            contents = engine.get_all("user1")
            assert len(contents) == 1

            # Embedder should only be called once
            assert embedder.call_count == 1
        finally:
            engine.close()

    def test_add_with_infer_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Add with infer=True should log a warning.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "Hello world", infer=True)

            # Should log warning about infer not being supported
            logger.warning.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineSearch:
    '''Tests for USearchEngine.search method.'''

    def test_search_empty_index(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Search on empty index should return empty list.'''
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
        '''Search should return matching contents with scores.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Goodbye world", infer=False)

            results = engine.search("Hello", "user1", limit=5)

            assert len(results) > 0
            # Results should be (content, score, created_at) tuples
            assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        finally:
            engine.close()

    def test_search_filters_by_project_id(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Search should only return contents for the specified project.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "User1 content", infer=False)
            engine.add("user2", "User2 content", infer=False)

            results = engine.search("content", "user1", limit=5)

            # Should only return user1's content
            contents = [r[0] for r in results]
            assert "User1 content" in contents
            assert "User2 content" not in contents
        finally:
            engine.close()

    def test_search_respects_limit(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Search should respect the limit parameter.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Add many contents
            for i in range(10):
                engine.add("user1", f"Content {i}", infer=False)

            results = engine.search("content", "user1", limit=3)

            assert len(results) <= 3
        finally:
            engine.close()


class TestUSearchEngineGetAll:
    '''Tests for USearchEngine.get_all method.'''

    def test_get_all_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Get all on empty store should return empty list.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            contents = engine.get_all("user1")
            assert contents == []
        finally:
            engine.close()

    def test_get_all_returns_contents(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Get all should return all contents for user.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Content 1", infer=False)
            engine.add("user1", "Content 2", infer=False)

            contents = engine.get_all("user1")

            assert len(contents) == 2
            assert "Content 1" in contents
            assert "Content 2" in contents
        finally:
            engine.close()


class TestUSearchEngineDelete:
    '''Tests for USearchEngine.delete method.'''

    def test_delete_existing_content(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Delete should remove content from both index and SQLite.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            # Get the content ID from SQLite
            content_id = engine.memory_store.get_id_by_content("user1", "Hello world")
            assert content_id is not None

            engine.delete(str(content_id))

            # Content should be removed
            contents = engine.get_all("user1")
            assert "Hello world" not in contents
        finally:
            engine.close()

    def test_delete_nonexistent_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Delete of non-existent content should log warning.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            engine.delete("999")

            # Should log warning about not found
            logger.warning.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_delete_invalid_id_raises_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Delete with invalid ID format should raise error.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()

            with pytest.raises(RuntimeError, match="Invalid memory_id format"):
                engine.delete("not-an-integer")
        finally:
            engine.close()


class TestUSearchEngineCommit:
    '''Tests for USearchEngine.commit method.'''

    def test_commit_saves_index(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Commit should save the USearch index to disk.'''
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
    '''Tests for USearchEngine persistence across restarts.'''

    def test_persistence_across_restart(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Contents should persist across engine restart.'''
        config, embedder, _ = temp_engine

        # First engine instance
        engine1 = USearchEngine(config=config, embedder=embedder)
        try:
            engine1.add("user1", "Persistent content", infer=False)
            engine1.commit()
        finally:
            engine1.close()

        # Second engine instance (simulating restart)
        engine2 = USearchEngine(config=config, embedder=embedder)
        try:
            # Content should still be accessible
            contents = engine2.get_all("user1")
            assert "Persistent content" in contents
        finally:
            engine2.close()


class TestUSearchEngineExactSearch:
    '''Tests for USearchEngine exact vs approximate search modes.'''

    def test_default_uses_exact_search(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Default config should use exact (brute-force) search for small databases.'''
        config, embedder, _ = temp_engine
        # Note: temp_engine uses explicit config, so we test with default config
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
        '''Should use exact search when exact_search=True.'''
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
        '''Should auto-switch to exact search when index size < threshold.'''
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
            # Add a few contents (below threshold)
            for i in range(5):
                engine.add("user1", f"Content {i}", infer=False)

            # Index size is 5, below threshold of 1000
            assert len(engine.index) == 5
            assert engine._should_use_exact_search() is True
        finally:
            engine.close()

    def test_approximate_search_above_threshold(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Should use approximate search when index size >= threshold.'''
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
            # Add contents to exceed threshold
            for i in range(5):
                engine.add("user1", f"Content {i}", infer=False)

            # Index size is 5, above threshold of 3
            assert len(engine.index) == 5
            assert engine._should_use_exact_search() is False
        finally:
            engine.close()

    def test_exact_search_returns_valid_results(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Exact search should return valid search results.'''
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

            # Should return valid results (3-tuples: content, score, created_at)
            assert len(results) > 0
            assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        finally:
            engine.close()

    def test_search_logs_search_mode(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Search should log the search mode being used.'''
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            project_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            exact_search=True,
        )
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.search("Hello", "user1", limit=5)

            # Should have logged the search mode
            debug_calls = logger.debug.call_args_list  # type: ignore
            search_mode_logged = any(
                "search mode" in str(call).lower() and "exact" in str(call).lower()
                for call in debug_calls
            )
            assert search_mode_logged, "Should log search mode as 'exact'"
        finally:
            engine.close()

    def test_config_from_app_config_includes_exact_search(self) -> None:
        '''USearchConfig.from_app_config should include exact search settings.'''
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.embedder_provider = "openai"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096
        mock_config.usearch_exact_search = True
        mock_config.usearch_exact_search_threshold = 5000

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_config(mock_config)

        assert config.exact_search is True
        assert config.exact_search_threshold == 5000


class TestUSearchConfigFromDict:
    '''Tests for USearchConfig.from_dict factory method.'''

    def test_from_dict_with_full_data(self) -> None:
        '''from_dict should create config from a complete dictionary.'''
        data = {
            "project_id": "my-proj",
            "index_path": "/tmp/index.usearch",
            "db_path": "/tmp/messages.db",
            "embedding_dims": 256,
            "metric": "l2",
            "connectivity": 32,
            "expansion_add": 256,
            "expansion_search": 128,
            "exact_search": False,
            "exact_search_threshold": 500,
        }
        config = USearchConfig.from_dict(data)

        assert config.project_id == "my-proj"
        assert config.index_path == "/tmp/index.usearch"
        assert config.db_path == "/tmp/messages.db"
        assert config.embedding_dims == 256
        assert config.metric == "l2"
        assert config.connectivity == 32
        assert config.expansion_add == 256
        assert config.expansion_search == 128
        assert config.exact_search is False
        assert config.exact_search_threshold == 500

    def test_from_dict_with_defaults(self) -> None:
        '''from_dict should use defaults for missing keys.'''
        config = USearchConfig.from_dict({})

        assert config.project_id == ""
        assert config.index_path == ""
        assert config.db_path == ""
        assert config.embedding_dims == 3072
        assert config.metric == "cos"
        assert config.connectivity == 16
        assert config.expansion_add == 128
        assert config.expansion_search == 64
        assert config.exact_search is True
        assert config.exact_search_threshold == 0


class TestUSearchEngineInitWithDict:
    '''Tests for USearchEngine initialization with dict config.'''

    def test_init_with_dict_config(self) -> None:
        '''Engine should accept a dict and convert to USearchConfig.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = {
                "project_id": "test",
                "index_path": os.path.join(tmpdir, "index.usearch"),
                "db_path": os.path.join(tmpdir, "messages.db"),
                "embedding_dims": 128,
            }
            embedder = MockEmbedder(dims=128)
            engine = USearchEngine(config=config_dict, embedder=embedder)

            try:
                assert isinstance(engine.config, USearchConfig)
                assert engine.config.project_id == "test"
            finally:
                engine.close()


class TestUSearchEngineIndexInit:
    '''Tests for USearch index lazy initialization error paths.'''

    def test_index_restore_returns_none_creates_new(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''When Index.restore returns None, should create a new index.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            idx = engine.index
            assert idx is not None
            logger.info.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_index_init_failure_raises_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''If both restore and create fail, should raise RuntimeError.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        with patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls:
            mock_index_cls.restore.side_effect = FileNotFoundError("no file")
            mock_index_cls.side_effect = MemoryError("out of memory")

            with pytest.raises(
                RuntimeError, match="Failed to initialize USearch index"
            ):
                _ = engine.index

            logger.error.assert_called()  # type: ignore

    def test_index_init_logs_debug_on_create(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Creating new index should log debug and info messages.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            _ = engine.index
            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            info_calls = [str(c) for c in logger.info.call_args_list]  # type: ignore
            assert any("not found" in c.lower() for c in debug_calls)
            assert any("Created new" in c for c in info_calls)
        finally:
            engine.close()

    def test_index_loaded_from_existing_file(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Restoring an existing index should log loading info.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()

        engine1 = USearchEngine(config=config, embedder=embedder)
        try:
            engine1.add("test", "Hello", infer=False)
            engine1.commit()
        finally:
            engine1.close()

        engine2 = USearchEngine(config=config, embedder=embedder, logger=logger)
        try:
            _ = engine2.index
            info_calls = [str(c) for c in logger.info.call_args_list]  # type: ignore
            assert any("Loaded existing" in c for c in info_calls)
        finally:
            engine2.close()


class TestUSearchEngineAddErrorPaths:
    '''Tests for add() error handling paths.'''

    def test_add_embedding_failure_rolls_back(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        failing_embedder = FailingQueryEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=failing_embedder, logger=logger)

        try:
            with pytest.raises(RuntimeError, match="Failed to generate embedding"):
                engine.add("user1", "rollback test", infer=False)

            contents = engine.get_all("user1")
            assert "rollback test" not in contents
            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_duplicate_via_storage_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''StorageError with "Duplicate message" should be silently skipped.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            mock_store = MagicMock()
            mock_store.insert.side_effect = StorageError("Duplicate message detected")
            object.__setattr__(engine, "_memory_store", mock_store)

            engine.add("user1", "dup msg", infer=False)

            logger.debug.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_unexpected_exception_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Unexpected exceptions in add should be wrapped in RuntimeError.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            mock_store = MagicMock()
            mock_store.insert.side_effect = TypeError("unexpected type error")
            object.__setattr__(engine, "_memory_store", mock_store)

            with pytest.raises(RuntimeError, match="Failed to add content"):
                engine.add("user1", "error msg", infer=False)

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_non_duplicate_runtime_error_reraises(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''RuntimeError without "Duplicate message" should be re-raised.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            mock_store = MagicMock()
            mock_store.insert.side_effect = RuntimeError("connection lost")
            object.__setattr__(engine, "_memory_store", mock_store)

            with pytest.raises(RuntimeError, match="connection lost"):
                engine.add("user1", "error msg", infer=False)

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineAddBatch:
    '''Tests for add_batch() method.'''

    def test_add_batch_empty_list(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''add_batch with empty list should return empty list.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            result = engine.add_batch("user1", [], infer=False)
            assert result == []
        finally:
            engine.close()

    def test_add_batch_success(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''add_batch should add multiple contents and return them.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            contents = ["Batch msg 1", "Batch msg 2", "Batch msg 3"]
            result = engine.add_batch("user1", contents, infer=False)

            assert len(result) == 3
            assert set(result) == set(contents)

            all_msgs = engine.get_all("user1")
            assert len(all_msgs) == 3

            logger.debug.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_batch_with_infer_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''add_batch with infer=True should log a warning.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add_batch("user1", ["msg"], infer=True)
            warning_calls = [str(c) for c in logger.warning.call_args_list]  # type: ignore
            assert any("infer=True" in c for c in warning_calls)
        finally:
            engine.close()

    def test_add_batch_all_duplicates_returns_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''add_batch where insert_many returns empty should return empty list.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()
            mock_store = MagicMock()
            mock_store.insert_many.return_value = []
            object.__setattr__(engine, "_memory_store", mock_store)

            result = engine.add_batch("user1", ["dup1", "dup2"], infer=False)
            assert result == []
        finally:
            engine.close()

    def test_add_batch_embedding_failure_rolls_back(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        failing_embedder = FailingBatchEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=failing_embedder, logger=logger)

        try:
            engine.ensure_initialized()

            with pytest.raises(RuntimeError, match="Failed to add content batch"):
                engine.add_batch("user1", ["rb1", "rb2"], infer=False)

            contents = engine.get_all("user1")
            assert "rb1" not in contents
            assert "rb2" not in contents
        finally:
            engine.close()

    def test_add_batch_embedding_size_mismatch(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        bad_embedder = MismatchedSizeEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=bad_embedder, logger=logger)

        try:
            with pytest.raises(RuntimeError, match="Failed to add content batch"):
                engine.add_batch("user1", ["m1", "m2"], infer=False)
        finally:
            engine.close()


class TestUSearchEngineDistanceScoring:
    '''Tests for _rank_scores and _distances_to_scores methods.'''

    def test_rank_scores_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''_rank_scores with count=0 should return empty array.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            scores = engine._rank_scores(0)
            assert len(scores) == 0
        finally:
            engine.close()

    def test_rank_scores_single(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''_rank_scores with count=1 should return [1.0].'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            scores = engine._rank_scores(1)
            assert len(scores) == 1
            assert float(scores[0]) == pytest.approx(1.0)
        finally:
            engine.close()

    def test_rank_scores_multiple(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''_rank_scores with count>1 should return descending scores.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            scores = engine._rank_scores(5)
            assert len(scores) == 5
            assert float(scores[0]) == pytest.approx(1.0)
            assert float(scores[-1]) == pytest.approx(0.0)
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]
        finally:
            engine.close()

    def test_distances_to_scores_l2_metric(self) -> None:
        '''L2 metric should use 1/(1+d) conversion.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = USearchConfig(
                project_id="test",
                index_path=os.path.join(tmpdir, "index.usearch"),
                db_path=os.path.join(tmpdir, "messages.db"),
                embedding_dims=128,
                metric="l2",
            )
            embedder = MockEmbedder(dims=128)
            engine = USearchEngine(config=config, embedder=embedder)

            try:
                distances = np.array([0.0, 1.0, 4.0], dtype=np.float32)
                scores = engine._distances_to_scores(distances)

                assert float(scores[0]) == pytest.approx(1.0)
                assert float(scores[1]) == pytest.approx(0.5)
                assert float(scores[2]) == pytest.approx(0.2)
            finally:
                engine.close()

    def test_distances_to_scores_unknown_metric_uses_rank(self) -> None:
        '''Unknown metric should fall back to rank-based scores.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = USearchConfig(
                project_id="test",
                index_path=os.path.join(tmpdir, "index.usearch"),
                db_path=os.path.join(tmpdir, "messages.db"),
                embedding_dims=128,
                metric="ip",
            )
            embedder = MockEmbedder(dims=128)
            logger = create_mock_logger()
            engine = USearchEngine(config=config, embedder=embedder, logger=logger)

            try:
                distances = np.array([0.1, 0.5, 0.9], dtype=np.float32)
                scores = engine._distances_to_scores(distances)

                assert len(scores) == 3
                assert float(scores[0]) == pytest.approx(1.0)
                assert float(scores[-1]) == pytest.approx(0.0)
                logger.warning.assert_called()  # type: ignore
            finally:
                engine.close()


class TestUSearchEngineSearchErrorPaths:
    '''Tests for search() error handling and edge cases.'''

    def test_search_empty_index_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Search on empty index with logger should log debug message.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            results = engine.search("hello", "user1", limit=5)

            assert results == []
            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("empty" in c.lower() for c in debug_calls)
        finally:
            engine.close()

    def test_search_no_matching_project_returns_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Search where no results match project_id should return empty.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            results = engine.search("Hello", "nonexistent-project", limit=5)
            assert results == []
        finally:
            engine.close()

    def test_search_exception_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        failing_embedder = ToggleableFailingQueryEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=failing_embedder, logger=logger)

        try:
            engine.add("user1", "Test msg", infer=False)

            failing_embedder.should_fail = True

            with pytest.raises(RuntimeError, match="USearch search failed"):
                engine.search("broken query", "user1", limit=5)

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineGetAllErrorPaths:
    '''Tests for get_all() error handling paths.'''

    def test_get_all_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''get_all with logger should log content count.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "msg1", infer=False)
            engine.get_all("user1")
            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("Retrieved all" in c for c in debug_calls)
        finally:
            engine.close()

    def test_get_all_error_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Exceptions in get_all should be wrapped in RuntimeError.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            original_store = engine.memory_store
            original_store.close()
            mock_store = MagicMock()
            mock_store.get_all.side_effect = OSError("disk read error")
            object.__setattr__(engine, "_memory_store", mock_store)

            with pytest.raises(RuntimeError, match="Failed to retrieve contents"):
                engine.get_all("user1")

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineDeleteErrorPaths:
    '''Tests for delete() additional error handling.'''

    def test_delete_existing_with_logger_logs_debug(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Deleting an existing content with logger should log debug.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "to delete", infer=False)
            content_id = engine.memory_store.get_id_by_content("user1", "to delete")
            assert content_id is not None

            engine.delete(str(content_id))

            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("deleted" in c.lower() for c in debug_calls)
        finally:
            engine.close()

    def test_delete_invalid_id_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Invalid memory_id with logger should log error.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            with pytest.raises(RuntimeError, match="Invalid memory_id format"):
                engine.delete("not-a-number")
            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_delete_general_exception_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Unexpected exceptions in delete should be wrapped in RuntimeError.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            engine._index = MagicMock()
            engine._index.__contains__ = MagicMock(
                side_effect=OSError("corrupted index")
            )

            with pytest.raises(RuntimeError, match="Failed to delete content"):
                engine.delete("42")

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineCommitErrorPaths:
    '''Tests for commit() error handling and edge cases.'''

    def test_commit_no_index_is_noop(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Commit when _index is None should be a no-op.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.commit()
        finally:
            engine.close()

    def test_commit_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Commit with logger should log save info.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "commit test", infer=False)
            engine.commit()

            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("saved" in c.lower() for c in debug_calls)
        finally:
            engine.close()

    def test_commit_error_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Exceptions in commit should be wrapped in RuntimeError.'''
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            engine._index = MagicMock()
            engine._index.save = MagicMock(side_effect=OSError("disk full"))

            with pytest.raises(RuntimeError, match="Failed to save USearch index"):
                engine.commit()

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineGetIdByContent:
    '''Tests for get_id_by_content() method.'''

    def test_get_id_by_content_found(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''get_id_by_content should return ID for existing content.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "findable msg", infer=False)
            content_id = engine.get_id_by_content("user1", "findable msg")
            assert content_id is not None
            assert isinstance(content_id, int)
        finally:
            engine.close()

    def test_get_id_by_content_not_found(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''get_id_by_content should return None for non-existent content.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()
            content_id = engine.get_id_by_content("user1", "nonexistent")
            assert content_id is None
        finally:
            engine.close()


class TestUSearchEngineContextManager:
    '''Tests for context manager protocol (__enter__/__exit__).'''

    def test_context_manager_initializes_and_closes(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''Context manager should initialize on enter and close on exit.'''
        config, embedder, _ = temp_engine

        with USearchEngine(config=config, embedder=embedder) as engine:
            assert engine is not None
            assert engine.name == "usearch"
            engine.add("user1", "ctx msg", infer=False)

    def test_context_manager_does_not_suppress_exceptions(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''__exit__ should return False to not suppress exceptions.'''
        config, embedder, _ = temp_engine

        with pytest.raises(ValueError, match="test error"):
            with USearchEngine(config=config, embedder=embedder) as engine:
                raise ValueError("test error")

    def test_close_with_no_memory_store(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        '''close() when _memory_store is None should be a no-op.'''
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        engine.close()
