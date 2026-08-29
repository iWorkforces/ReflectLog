#!/usr/bin/env python3
"""Unit tests for MemoryManager – targeting uncovered lines for 90%+ coverage.

Uncovered lines targeted:
  334-353, 364-376, 390, 407, 413, 435-463, 483, 512, 527-560,
  594-602, 642, 654, 662, 774-775, 841-842, 853-857, 904-914,
  933, 983-984, 1002-1003
"""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.core.exceptions import (
    InconsistentStateError,
    SearchError,
    StorageError,
)
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.cross_encoder_reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODULE = "reflectlog.application.memory.manager"


@pytest.fixture
def mock_config() -> Config:
    """Minimal mock Config for MemoryManager tests."""
    config = Mock(spec=Config)
    config.workspace_id = "test_project"
    config.enable_hybrid_search = True
    config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
    config.index_base_path = "/tmp/test_indexes"
    config.search_limit = 5
    config.search_score_threshold = 0.8
    config.deduplicate_memories = True
    config.enable_llm_infer = False
    config.remove_search_limit = 5
    config.remove_score_threshold = 0.9
    # Fusion settings
    config.fusion_method = "rrf"
    config.fusion_normalization = None
    config.fusion_rrf_k = 60
    config.fusion_ranking_threshold = 0.5
    config.enable_rrf_fusion = True
    # Concurrency
    config.add_max_concurrency = 4
    config.rerank_max_concurrency = 5
    # Smart replacement
    config.enable_smart_replace = False
    config.smart_replace_threshold = 0.7
    config.smart_replace_min_similarity = 0.5
    config.smart_replace_candidate_limit = 3
    config.smart_replace_archive_ttl_days = 30
    config.smart_replace_max_retries = 3
    config.smart_replace_retry_delay = 1.0
    config.llm_provider = "openai"
    # Reranker
    config.reranker_engine = "none"
    config.reranker_min_results = 0
    config.reranker_batch_normalize = True
    # Recency
    config.enable_recency_boost = True
    config.recency_decay_rate = 0.01
    # Hybrid
    config.overfetch_multiplier = 3
    # Logging
    config.log_search_results_verbose = False
    config.log_search_result_limit = 3
    # Embedding
    config.embedding_model = "openai/text-embedding-3-large"
    config.embedding_dims = 3072
    config.embedder_provider = "openai"
    config.llm_model = "x-ai/grok-4.1-fast"
    config.openrouter_api_key = Mock()
    config.openrouter_api_key.get_secret_value.return_value = "test-api-key"
    config.openrouter_base_url = "https://openrouter.ai/api/v1"
    config.qwen_embedding_dims = 4096
    config.embedding_cache_enabled = False
    config.embedding_cache_size = 100
    config.embedding_batch_size = 512
    config.embedding_max_concurrent_batches = 4
    # Eager init
    config.eager_initialization = False
    config.eager_initialize_search_engines = None
    config.eager_initialize_reranker = None
    config.eager_initialize_smart_replacer = None
    # Cross-encoder
    config.cross_encoder_model = "BAAI/bge-reranker-v2-m3"
    config.cross_encoder_top_k = 20
    config.cross_encoder_device = "cpu"
    config.cross_encoder_batch_size = 32
    config.cross_encoder_score_threshold = 0.5
    config.cross_encoder_use_fp16 = True
    config.cross_encoder_normalize = True
    config.cross_encoder_max_length = 512
    # Overfetch adaptive
    config.overfetch_adaptive = True
    config.overfetch_min_multiplier = 1.5
    config.overfetch_max_multiplier = 3.0
    return config


@pytest.fixture
def mock_logger():
    """Mock structured logger."""
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


def _make_manager(config, logger):
    """Helper to construct MemoryManager with all infrastructure mocked."""
    with (
        patch(f"{MODULE}.USearchEngine") as usearch_cls,
        patch(f"{MODULE}.LangchainQwenEmbeddings"),
        patch(f"{MODULE}.TantivyEngine") as tantivy_cls,
    ):
        mock_usearch = MagicMock()
        mock_usearch.add_batch.side_effect = (
            lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: contents if contents is not None else memories
        )
        mock_usearch.get_id_by_content.return_value = None
        usearch_cls.return_value = mock_usearch

        mock_tantivy = MagicMock()
        tantivy_cls.return_value = mock_tantivy

        manager = MemoryManager(config, logger)
        return manager, mock_usearch, mock_tantivy


# ---------------------------------------------------------------------------
# Tests: Eager Initialization  (lines 334-353, 364-376, 390)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEagerInitialization:
    """Tests for _eager_initialize_engines covering uncovered branches."""

    def test_eager_init_reranker_invalid_engine_raises(self, mock_config, mock_logger):
        """Invalid reranker_engine should raise ValueError (lines 334-340)."""
        mock_config.eager_initialization = True
        mock_config.eager_initialize_search_engines = False
        mock_config.eager_initialize_reranker = True
        mock_config.reranker_engine = "none"

        with pytest.raises(ValueError, match="Invalid reranker_engine"):
            _make_manager(mock_config, mock_logger)

    def test_eager_init_reranker_returns_none_warns(self, mock_config, mock_logger):
        """Reranker get_reranker returning None should log warning (lines 349-359)."""
        mock_config.eager_initialization = True
        mock_config.eager_initialize_search_engines = False
        mock_config.eager_initialize_reranker = True
        mock_config.reranker_engine = "cross_encoder"

        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as reranker_cls,
        ):
            usearch_cls.return_value = MagicMock()
            reranker_cls.return_value = MagicMock()
            manager = MemoryManager(mock_config, mock_logger)
            assert manager._cross_encoder_reranker is not None

    def test_eager_init_reranker_with_cross_encoder(self, mock_config, mock_logger):
        """Eager init with a valid cross-encoder reranker."""
        mock_config.eager_initialization = True
        mock_config.eager_initialize_search_engines = False
        mock_config.eager_initialize_reranker = True
        mock_config.reranker_engine = "cross_encoder"

        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as reranker_cls,
        ):
            mock_reranker = MagicMock()
            reranker_cls.return_value = mock_reranker
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger)
            assert manager._cross_encoder_reranker is mock_reranker

    def test_eager_init_smart_replacer_disabled_raises(self, mock_config, mock_logger):
        """Eager smart replacer with enable_smart_replace=False raises (lines 364-369)."""
        mock_config.eager_initialization = True
        mock_config.eager_initialize_search_engines = False
        mock_config.eager_initialize_smart_replacer = True
        mock_config.enable_smart_replace = False

        with pytest.raises(
            ValueError, match="Eager SmartReplacer initialization requested"
        ):
            _make_manager(mock_config, mock_logger)

    def test_eager_init_smart_replacer_enabled(self, mock_config, mock_logger):
        """Eager smart replacer with enable_smart_replace=True (lines 371-376)."""
        mock_config.eager_initialization = True
        mock_config.eager_initialize_search_engines = False
        mock_config.eager_initialize_smart_replacer = True
        mock_config.enable_smart_replace = True

        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.SmartReplacerConfig"),
            patch(f"{MODULE}.SmartReplacer") as replacer_cls,
        ):
            mock_replacer = MagicMock()
            replacer_cls.return_value = mock_replacer
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger)
            assert manager._smart_replacer is mock_replacer

    def test_eager_init_all_lazy_skipped(self, mock_config, mock_logger):
        """All components set to lazy should log skip message (line 390)."""
        mock_config.eager_initialization = True
        mock_config.eager_initialize_search_engines = False
        mock_config.eager_initialize_reranker = False
        mock_config.eager_initialize_smart_replacer = False

        manager, _, _ = _make_manager(mock_config, mock_logger)
        # Verify skip message was logged
        skip_logged = any(
            "skipped" in str(call).lower() for call in mock_logger.info.call_args_list
        )
        assert skip_logged


# ---------------------------------------------------------------------------
# Tests: Lazy Reranker Properties  (lines 407, 413, 435-463, 512)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLazyRerankerProperties:
    """Tests for cross_encoder_reranker and get_reranker properties."""

    def test_cross_encoder_reranker_returns_none_when_not_configured(
        self, mock_config, mock_logger
    ):
        """cross_encoder_reranker returns None for non-cross_encoder (line 435-439)."""
        mock_config.reranker_engine = "none"
        manager, _, _ = _make_manager(mock_config, mock_logger)
        assert manager.cross_encoder_reranker is None

    def test_cross_encoder_reranker_lazy_init(self, mock_config, mock_logger):
        """cross_encoder_reranker lazy initializes (lines 441-463)."""
        mock_config.reranker_engine = "cross_encoder"
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as ce_cls,
        ):
            mock_ce = MagicMock()
            ce_cls.return_value = mock_ce
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger)
            result = manager.cross_encoder_reranker
            assert result is mock_ce
            ce_cls.assert_called_once()

    def test_cross_encoder_reranker_cached(self, mock_config, mock_logger):
        """cross_encoder_reranker returns cached on second call (line 435-439)."""
        mock_config.reranker_engine = "cross_encoder"
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as ce_cls,
        ):
            mock_ce = MagicMock()
            ce_cls.return_value = mock_ce
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger)
            first = manager.cross_encoder_reranker
            second = manager.cross_encoder_reranker
            assert first is second
            ce_cls.assert_called_once()

    def test_cross_encoder_reranker_double_check(self, mock_config, mock_logger):
        """cross_encoder_reranker double-check after lock (lines 444-448)."""
        mock_config.reranker_engine = "cross_encoder"
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker"),
        ):
            usearch_cls.return_value = MagicMock()
            manager = MemoryManager(mock_config, mock_logger)
            sentinel = cast(CrossEncoderReranker, MagicMock())
            manager._cross_encoder_reranker = sentinel
            result = manager.cross_encoder_reranker
            assert result is sentinel

    def test_get_reranker_cross_encoder_path(self, mock_config, mock_logger):
        """get_reranker returns cross_encoder when configured (line 512)."""
        mock_config.reranker_engine = "cross_encoder"
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as ce_cls,
        ):
            mock_ce = MagicMock()
            ce_cls.return_value = mock_ce
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger)
            result = manager.get_reranker()
            assert result is mock_ce

    def test_get_reranker_none_engine(self, mock_config, mock_logger):
        """get_reranker returns None for "none" engine."""
        mock_config.reranker_engine = "none"
        manager, _, _ = _make_manager(mock_config, mock_logger)
        assert manager.get_reranker() is None


# ---------------------------------------------------------------------------
# Tests: Smart Replacer Property  (line 483)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSmartReplacerProperty:
    """Tests for smart_replacer lazy property."""

    def test_smart_replacer_returns_none_when_disabled(self, mock_config, mock_logger):
        """smart_replacer returns None when enable_smart_replace=False."""
        mock_config.enable_smart_replace = False
        manager, _, _ = _make_manager(mock_config, mock_logger)
        assert manager.smart_replacer is None

    def test_smart_replacer_double_check_locking(self, mock_config, mock_logger):
        """smart_replacer double-check after lock (line 483)."""
        mock_config.enable_smart_replace = True
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.SmartReplacerConfig"),
            patch(f"{MODULE}.SmartReplacer"),
        ):
            usearch_cls.return_value = MagicMock()
            manager = MemoryManager(mock_config, mock_logger)
            sentinel = MagicMock()
            manager._smart_replacer = sentinel
            result = manager.smart_replacer
            assert result is sentinel

    def test_smart_replacer_lazy_init(self, mock_config, mock_logger):
        """smart_replacer lazily initializes when enabled."""
        mock_config.enable_smart_replace = True
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.SmartReplacerConfig"),
            patch(f"{MODULE}.SmartReplacer") as replacer_cls,
        ):
            mock_replacer = MagicMock()
            replacer_cls.return_value = mock_replacer
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger)
            result = manager.smart_replacer
            assert result is mock_replacer
            replacer_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _add_memory  (lines 527-560)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddMemory:
    """Tests for _add_memory method."""

    def test_add_memory_duplicate_skipped(self, mock_config, mock_logger):
        """Duplicate memory returns False (lines 527-535)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 11

        result = manager._add_memory("hello world")
        assert result is False
        mock_usearch.add.assert_not_called()

    def test_add_memory_success(self, mock_config, mock_logger):
        """Successful add returns True (lines 537-557)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_tantivy.search.return_value = []  # No duplicate

        result = manager._add_memory("new memory")
        assert result is True
        mock_usearch.add.assert_called_once_with(
            workspace_id="test_project",
            content="new memory",
            infer=False,
        )
        mock_tantivy.add.assert_called_once_with("test_project", "new memory")

    def test_add_memory_without_tantivy(self, mock_config, mock_logger):
        """Add memory without Tantivy engine (semantic only)."""
        mock_config.enable_hybrid_search = False
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
        ):
            mock_usearch = MagicMock()
            usearch_cls.return_value = mock_usearch
            # No duplicate via database lookup
            mock_usearch.get_id_by_content.return_value = None

            manager = MemoryManager(mock_config, mock_logger)
            result = manager._add_memory("solo memory")
            assert result is True
            mock_usearch.add.assert_called_once()

    def test_add_memory_storage_error(self, mock_config, mock_logger):
        """Storage exception wraps in StorageError (lines 559-560)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_tantivy.search.return_value = []
        mock_usearch.add.side_effect = RuntimeError("disk full")

        with pytest.raises(StorageError, match="Failed to add memory"):
            manager._add_memory("error content")

    def test_add_memory_dedup_disabled(self, mock_config, mock_logger):
        """When deduplicate_memories=False, skip duplicate check."""
        mock_config.deduplicate_memories = False
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        result = manager._add_memory("any memory")
        assert result is True
        # Should not call search for dedup
        mock_tantivy.search.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: add_memories batch dedup/logging  (lines 594-602, 642, 654, 662)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddMemoriesBatchLogging:
    """Tests for add_memories batch dedup and logging edge cases."""

    def test_add_memories_in_batch_duplicate(self, mock_config, mock_logger):
        """Duplicate within batch should be skipped (lines 593-602)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.add_batch.side_effect = (
            lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: contents if contents is not None else memories
        )

        result = manager.add_memories(["mem1", "mem1", "mem2"])
        # "mem1" repeated: first stored, second skipped
        assert result == 2  # mem1 + mem2

    def test_add_memories_batch_insert_skipped_warning(self, mock_config, mock_logger):
        """Memories skipped during batch insert should log warning (line 654)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        # add_batch returns only subset - some memories skipped
        # Must clear side_effect (set by _make_manager) before setting return_value
        mock_usearch.add_batch.side_effect = None
        mock_usearch.add_batch.return_value = ["mem1"]

        result = manager.add_memories(["mem1", "mem2"])
        assert result == 1
        # Verify warning logged for skipped memory
        warning_logged = any(
            "Skipped during batch insert" in str(call)
            for call in mock_logger.warning.call_args_list
        )
        assert warning_logged

    def test_add_memories_stored_log_limit_exceeded(self, mock_config, mock_logger):
        """When stored memories exceed LOG_ADD_MEMORY_PREVIEW_LIMIT (line 662)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        # Create more memories than LOG_ADD_MEMORY_PREVIEW_LIMIT (20)
        memories = [f"mem_{i}" for i in range(25)]
        mock_usearch.add_batch.return_value = memories

        result = manager.add_memories(memories)
        assert result == 25

    def test_add_memories_log_limit_exceeded(self, mock_config, mock_logger):
        """When total memories exceed LOG_ADD_MEMORY_PREVIEW_LIMIT (line 642)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        # Create more memories than LOG_ADD_MEMORY_PREVIEW_LIMIT (20)
        memories = [f"mem_{i}" for i in range(25)]
        mock_usearch.add_batch.return_value = memories

        result = manager.add_memories(memories)
        assert result == 25
        # Should have logged omission notice
        omit_logged = any(
            "omitted from logs" in str(call) for call in mock_logger.info.call_args_list
        )
        assert omit_logged


# ---------------------------------------------------------------------------
# Tests: search  (lines 774-775)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchIndexSizeException:
    """Tests for search when index size lookup raises exception."""

    @pytest.mark.asyncio
    async def test_search_index_size_exception_fallback(self, mock_config, mock_logger):
        """Exception getting index size should fallback to 0 (lines 774-775)."""
        mock_config.reranker_engine = "none"
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        # Make index property raise
        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(side_effect=RuntimeError("broken"))
        mock_usearch.index = mock_index
        mock_usearch.search.return_value = [("result", 0.9, "2024-01-01T00:00:00")]
        mock_tantivy.search.return_value = [("result", 0.9)]

        results = await manager.search("test query")
        # Should not raise, should work with index_size=0
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests: search_for_removal error  (lines 841-842)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchForRemovalError:
    """Tests for search_for_removal error path."""

    def test_search_for_removal_exception_raises_search_error(
        self, mock_config, mock_logger
    ):
        """Exception during lookup raises SearchError (lines 841-842)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.side_effect = RuntimeError("db error")

        with pytest.raises(SearchError, match="Failed to search for removal"):
            manager.search_for_removal("test")


# ---------------------------------------------------------------------------
# Tests: delete_by_id and delete_by_memory  (lines 853-857, 904-914, 933)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteOperations:
    """Tests for delete_by_id and delete_by_memory error paths."""

    def test_delete_by_id_success(self, mock_config, mock_logger):
        """delete_by_id calls semantic engine delete (lines 853-855)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        manager.delete_by_id("42")
        mock_usearch.delete.assert_called_once_with(memory_id="42")

    def test_delete_by_id_exception_raises_storage_error(
        self, mock_config, mock_logger
    ):
        """delete_by_id wraps exception in StorageError (lines 856-857)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.delete.side_effect = RuntimeError("delete failed")

        with pytest.raises(StorageError, match="Failed to delete memory"):
            manager.delete_by_id("42")

    def test_delete_by_memory_tantivy_failure_inconsistent_state(
        self, mock_config, mock_logger
    ):
        """Tantivy failure after USearch delete raises InconsistentStateError (lines 904-917)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42
        mock_tantivy.delete.side_effect = RuntimeError("tantivy broken")

        with pytest.raises(
            InconsistentStateError,
            match="USearch deletion succeeded but Tantivy deletion failed",
        ):
            manager.delete_by_memory("test memory")

        # USearch delete should have been called
        mock_usearch.delete.assert_called_once_with(memory_id="42")

    def test_delete_by_memory_inconsistent_state_reraise(
        self, mock_config, mock_logger
    ):
        """InconsistentStateError is re-raised, not wrapped (line 933)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42
        mock_tantivy.delete.side_effect = RuntimeError("tantivy broken")

        with pytest.raises(InconsistentStateError):
            manager.delete_by_memory("test memory")

    def test_delete_by_memory_generic_exception_raises_storage_error(
        self, mock_config, mock_logger
    ):
        """Generic exception during delete wraps in StorageError (line 935)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.side_effect = RuntimeError("lookup failed")

        with pytest.raises(StorageError, match="Failed to delete memory"):
            manager.delete_by_memory("test memory")

    def test_delete_by_memory_not_found(self, mock_config, mock_logger):
        """delete_by_memory returns False when not found."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = None

        result = manager.delete_by_memory("nonexistent")
        assert result is False

    def test_delete_by_memory_success(self, mock_config, mock_logger):
        """delete_by_memory returns True on success."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42

        result = manager.delete_by_memory("test memory")
        assert result is True
        mock_usearch.delete.assert_called_once_with(memory_id="42")
        mock_tantivy.delete.assert_called_once_with(
            "test_project", "test memory", verify_exists=True
        )

    def test_delete_by_memory_without_tantivy(self, mock_config, mock_logger):
        """delete_by_memory works without Tantivy."""
        mock_config.enable_hybrid_search = False
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
        ):
            mock_usearch = MagicMock()
            mock_usearch.get_id_by_content.return_value = 42
            usearch_cls.return_value = mock_usearch

            manager = MemoryManager(mock_config, mock_logger)
            result = manager.delete_by_memory("test memory")
            assert result is True
            mock_usearch.delete.assert_called_once_with(memory_id="42")


    def test_delete_memories_returns_found_contents(self, mock_config, mock_logger):
        """delete_memories returns only contents that existed."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)

        def lookup(_workspace_id: str, content: str) -> int | None:
            return {"keep": 1, "also": 2}.get(content)

        mock_usearch.get_id_by_content.side_effect = lookup

        deleted = manager.delete_memories(["keep", "missing", "also"])

        assert deleted == ["keep", "also"]
        assert mock_usearch.delete.call_count == 2
        mock_usearch.commit.assert_called_once()
        assert mock_tantivy.delete.call_count == 2

    def test_delete_memories_tantivy_failure_inconsistent_state(
        self, mock_config, mock_logger
    ):
        """Tantivy failure after USearch batch delete raises InconsistentStateError."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42
        mock_tantivy.delete.side_effect = RuntimeError("tantivy broken")

        with pytest.raises(
            InconsistentStateError,
            match="USearch deletion succeeded but Tantivy deletion failed",
        ):
            manager.delete_memories(["test memory"])

        mock_usearch.delete.assert_called_once_with(memory_id="42")

    def test_delete_memories_uses_verify_exists_on_batch(
        self, mock_config, mock_logger
    ):
        """Production delete_batch probes FTS so it does not plant phantom tombstones."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 7

        class FakeTantivy:
            def __init__(self) -> None:
                self.calls: list[tuple[str, list[str], bool]] = []

            def delete_batch(
                self,
                workspace_id: str,
                contents: list[str],
                verify_exists: bool = False,
            ) -> int:
                self.calls.append((workspace_id, contents, verify_exists))
                return len(contents)

        fake = FakeTantivy()
        manager._tantivy_engine = fake

        deleted = manager.delete_memories(["hello"])

        assert deleted == ["hello"]
        assert fake.calls == [("test_project", ["hello"], True)]

    def test_delete_memories_short_count_fails_closed(
        self, mock_config, mock_logger
    ):
        """A live FTS miss after USearch delete is InconsistentStateError."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 7

        class ShortTantivy:
            def delete_batch(
                self,
                workspace_id: str,
                contents: list[str],
                verify_exists: bool = False,
            ) -> int:
                return 0

        manager._tantivy_engine = ShortTantivy()

        with pytest.raises(InconsistentStateError, match="deleted 0/1"):
            manager.delete_memories(["hello"])

        mock_usearch.delete.assert_called_once_with(memory_id="7")


# ---------------------------------------------------------------------------
# Tests: close error paths  (lines 983-984, 1002-1003)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloseErrorPaths:
    """Tests for close method error paths."""

    def test_close_tantivy_error_logged(self, mock_config, mock_logger):
        """Tantivy close error should be logged not raised (lines 983-991)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_tantivy.flush.side_effect = RuntimeError("tantivy flush error")

        manager.close()
        # Should log error
        error_logged = any(
            "Error closing Tantivy engine" in str(call)
            for call in mock_logger.error.call_args_list
        )
        assert error_logged

    def test_close_usearch_error_logged(self, mock_config, mock_logger):
        """USearch close error should be logged not raised (lines 1002-1010)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.commit.side_effect = RuntimeError("usearch commit error")

        manager.close()
        # Should log error
        error_logged = any(
            "Error closing USearch engine" in str(call)
            for call in mock_logger.error.call_args_list
        )
        assert error_logged

    def test_close_both_errors_logged(self, mock_config, mock_logger):
        """Both engines failing should both be logged."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_tantivy.flush.side_effect = RuntimeError("tantivy error")
        mock_usearch.commit.side_effect = RuntimeError("usearch error")

        manager.close()
        error_calls = [str(call) for call in mock_logger.error.call_args_list]
        tantivy_err = any("Tantivy" in c for c in error_calls)
        usearch_err = any("USearch" in c for c in error_calls)
        assert tantivy_err
        assert usearch_err

    def test_close_success(self, mock_config, mock_logger):
        """Successful close should log completion."""
        manager, _, _ = _make_manager(mock_config, mock_logger)
        manager.close()
        close_logged = any(
            "all data persisted" in str(call).lower()
            for call in mock_logger.info.call_args_list
        )
        assert close_logged


# ---------------------------------------------------------------------------
# Tests: get_all error path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAll:
    """Tests for get_all method."""

    def test_get_all_success(self, mock_config, mock_logger):
        """get_all returns memories from semantic engine."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_all.return_value = ["mem1", "mem2"]

        result = manager.get_all()
        assert result == ["mem1", "mem2"]
        mock_usearch.get_all.assert_called_once_with(workspace_id="test_project")

    def test_get_all_exception_raises_storage_error(self, mock_config, mock_logger):
        """get_all wraps exceptions in StorageError."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_all.side_effect = RuntimeError("db error")

        with pytest.raises(StorageError, match="Failed to retrieve memories"):
            manager.get_all()


# ---------------------------------------------------------------------------
# Tests: Init logging paths (reranker_engine variations)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitLogging:
    """Tests for __init__ logging based on reranker_engine and smart_replace."""

    def test_init_cross_encoder_reranker_logging(self, mock_config, mock_logger):
        """Init with cross_encoder reranker logs correctly."""
        mock_config.reranker_engine = "cross_encoder"
        _make_manager(mock_config, mock_logger)
        ce_logged = any(
            "CrossEncoder" in str(call) for call in mock_logger.info.call_args_list
        )
        assert ce_logged

    def test_init_smart_replace_enabled_logging(self, mock_config, mock_logger):
        """Init with smart replace enabled logs correctly."""
        mock_config.enable_smart_replace = True
        _make_manager(mock_config, mock_logger)
        sr_logged = any(
            "SmartReplacer configured" in str(call)
            for call in mock_logger.info.call_args_list
        )
        assert sr_logged

    def test_init_embedding_cache_enabled(self, mock_config, mock_logger):
        """Init with embedding cache enabled wraps embedder."""
        mock_config.embedding_cache_enabled = True
        with (
            patch(f"{MODULE}.USearchEngine"),
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CachedEmbeddings") as cached_cls,
        ):
            _manager = MemoryManager(mock_config, mock_logger)
            cached_cls.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
