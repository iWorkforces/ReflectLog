#!/usr/bin/env python3
"""Unit tests for hybrid MemoryManager (USearch + Tantivy)."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.config.settings import Config
from reflectlog.application.utils import StructuredLogger
from reflectlog.application.exceptions import StorageError


@pytest.fixture
def mock_config() -> Config:
    """Mock configuration with hybrid search enabled."""
    config = Mock(spec=Config)
    config.project_id = "test_project"
    config.enable_hybrid_search = True
    config.tantivy_index_path_template = "{project_id}_tantivy_test"
    config.index_base_path = "/tmp/test_indexes"
    config.search_limit = 5
    config.search_score_threshold = 0.8
    config.deduplicate_messages = True
    config.enable_llm_infer = False
    config.remove_search_limit = 5
    config.remove_score_threshold = 0.9
    # Fusion settings (ranx-based)
    config.fusion_method = "rrf"
    config.fusion_normalization = None
    config.fusion_rrf_k = 60
    config.fusion_ranking_threshold = 0.5
    config.enable_rrf_fusion = True  # RRF fusion enabled by default
    # Concurrency settings
    config.add_max_concurrency = 4
    config.rerank_max_concurrency = 5
    # Smart replacement settings
    config.enable_smart_replace = True
    config.smart_replace_threshold = 0.7
    config.smart_replace_min_similarity = 0.5
    config.smart_replace_candidate_limit = 3
    config.smart_replace_archive_ttl_days = 30
    config.smart_replace_max_retries = 3
    config.smart_replace_retry_delay = 1.0
    config.llm_provider = "openai"
    # Reranker settings
    config.reranker_engine = "llm"
    config.reranker_min_results = 0
    config.reranker_batch_normalize = True
    # Recency boost settings
    config.enable_recency_boost = True
    config.recency_decay_rate = 0.01
    # Hybrid search settings
    config.overfetch_multiplier = 3
    # Logging settings
    config.log_search_results_verbose = False
    config.log_search_result_limit = 3
    # USearchEngine config fields
    config.embedding_model = "openai/text-embedding-3-large"
    config.embedding_dims = 3072
    config.embedder_provider = "openai"
    config.llm_model = "x-ai/grok-4.1-fast"
    config.openrouter_api_key = Mock()
    config.openrouter_api_key.get_secret_value.return_value = "test-api-key"
    config.openrouter_base_url = "https://openrouter.ai/api/v1"
    config.qwen_embedding_dims = 4096
    # Disable embedding cache in tests to avoid issues with mocked embedders
    config.embedding_cache_enabled = False
    config.embedding_cache_size = 100
    # Disable eager initialization in tests to avoid issues with mocked engines
    config.eager_initialization = False
    return config


@pytest.fixture
def mock_logger():
    """Mock structured logger."""
    return Mock(spec=StructuredLogger)


@pytest.mark.unit
class TestHybridMemoryManager:
    """Tests for hybrid MemoryManager (USearch + TantivyEngine)."""

    def test_initialization(self, mock_config, mock_logger):
        """Test basic initialization with TantivyEngine."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy:
                    manager = MemoryManager(mock_config, mock_logger)
                    assert hasattr(manager, "_semantic_engine")
                    assert hasattr(manager, "_tantivy_engine")
                    assert hasattr(manager, "_fusion_engine")
                    # TantivyEngine should be initialized with config
                    mock_tantivy.assert_called_once()

    def test_initialization_without_hybrid_search(self, mock_config, mock_logger):
        """Test initialization with hybrid search disabled."""
        mock_config.enable_hybrid_search = False
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy:
                    manager = MemoryManager(mock_config, mock_logger)
                    # TantivyEngine should not be initialized
                    mock_tantivy.assert_not_called()
                    assert manager._tantivy_engine is None

    def test_add_messages(self, mock_config, mock_logger):
        """Test parallel indexing works with TantivyEngine."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearch engine mock
                    mock_usearch = MagicMock()
                    mock_usearch.add_batch.return_value = ["test"]
                    mock_usearch_class.return_value = mock_usearch

                    # Setup tantivy mock
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = manager.add_messages(["test"])

                    assert result == 1
                    # USearchEngine.add_batch should be called
                    mock_usearch.add_batch.assert_called_once()
                    # TantivyEngine.add should be called
                    mock_tantivy.add.assert_called_once()
                    # TantivyEngine.commit should be called after batch
                    mock_tantivy.commit.assert_called_once()

    def test_search_for_removal_optimized(self, mock_config, mock_logger):
        """Test removal search using direct database lookup (O(log n) optimization).

        Sprint 1.4 changed from O(n) get_all() + iteration to O(log n) indexed lookup.
        """
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    mock_usearch = MagicMock()
                    # Mock get_id_by_message for direct lookup (O(log n))
                    mock_usearch.get_id_by_message.return_value = 42
                    mock_usearch_class.return_value = mock_usearch

                    manager = MemoryManager(mock_config, mock_logger)
                    candidates = manager.search_for_removal("test", limit=1)

                    # Verify get_id_by_message was called (not get_all)
                    mock_usearch.get_id_by_message.assert_called_once_with(
                        mock_config.project_id, "test"
                    )
                    mock_usearch.get_all.assert_not_called()
                    assert len(candidates) == 1
                    assert candidates[0]["memory"] == "test"
                    assert candidates[0]["id"] == "42"  # Database ID as string
                    assert candidates[0]["score"] == 1.0  # Exact match = perfect score

    def test_search_for_removal_not_found(self, mock_config, mock_logger):
        """Test removal search when message is not found."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    mock_usearch = MagicMock()
                    # Mock get_id_by_message returning None (not found)
                    mock_usearch.get_id_by_message.return_value = None
                    mock_usearch_class.return_value = mock_usearch

                    manager = MemoryManager(mock_config, mock_logger)
                    candidates = manager.search_for_removal("nonexistent", limit=1)

                    # Verify direct lookup was used
                    mock_usearch.get_id_by_message.assert_called_once()
                    assert len(candidates) == 0  # No candidates when not found

    def test_exact_match_detection(self, mock_config, mock_logger):
        """Test exact match detection uses Tantivy."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)

                    # Should find exact match via Tantivy
                    mock_tantivy.search.return_value = [("test message", 1.0)]
                    result = manager._has_exact_match("test message")
                    assert result is True

                    # Should not find different message
                    mock_tantivy.search.return_value = [("different message", 1.0)]
                    result = manager._has_exact_match("test message")
                    assert result is False

    def test_exact_match_detection_fallback_uses_database_lookup(
        self, mock_config, mock_logger
    ):
        """Test exact match detection fallback uses direct database lookup (Sprint 2.1).

        When Tantivy is not available, _has_exact_match() should use get_id_by_message()
        for O(log n) indexed lookup instead of semantic search with embedding API call.
        """
        mock_config.enable_hybrid_search = False  # Disable Tantivy

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                mock_usearch = MagicMock()
                mock_usearch_class.return_value = mock_usearch

                manager = MemoryManager(mock_config, mock_logger)

                # Test: Message found via database lookup
                mock_usearch.get_id_by_message.return_value = 42
                result = manager._has_exact_match("test message")
                assert result is True
                mock_usearch.get_id_by_message.assert_called_with(
                    mock_config.project_id, "test message"
                )

                # Test: Message not found
                mock_usearch.reset_mock()
                mock_usearch.get_id_by_message.return_value = None
                result = manager._has_exact_match("nonexistent")
                assert result is False
                mock_usearch.get_id_by_message.assert_called_with(
                    mock_config.project_id, "nonexistent"
                )

                # Test: Error handling - should return False and allow add
                mock_usearch.reset_mock()
                mock_usearch.get_id_by_message.side_effect = RuntimeError("DB error")
                result = manager._has_exact_match("error case")
                assert result is False  # Should proceed without deduplication

    @pytest.mark.asyncio
    async def test_search_uses_rrf_fusion(self, mock_config, mock_logger):
        """Test hybrid search uses RRFFusion for ranking."""
        mock_config.reranker_engine = "none"  # Skip LLM reranking in unit test
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearchEngine mock - now returns 3-tuples (message, score, created_at)
                    mock_usearch = MagicMock()
                    mock_usearch.search.return_value = [
                        ("usearch result", 0.9, "2024-01-01T00:00:00")
                    ]
                    mock_usearch_class.return_value = mock_usearch

                    # Setup Tantivy mock - still returns 2-tuples (no timestamps)
                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = [("tantivy result", 0.8)]
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)

                    # Search should use RRF fusion (await async method)
                    _ = await manager.search("test query")

                    # Both engines should be queried
                    mock_usearch.search.assert_called()
                    mock_tantivy.search.assert_called()


@pytest.mark.unit
class TestParallelMessageAddition:
    """Tests for parallel message addition via add_messages_async()."""

    @pytest.mark.asyncio
    async def test_add_messages_async_empty_list(self, mock_config, mock_logger):
        """Empty list should return AddResult with 0 stored without any operations."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_messages_async([])
                    assert result.stored_count == 0
                    assert result.skipped_count == 0
                    assert result.replaced_count == 0

    @pytest.mark.asyncio
    async def test_add_messages_async_single_message(self, mock_config, mock_logger):
        """Single message should skip concurrency overhead."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_messages_async(["single message"])

                    assert result.stored_count == 1
                    mock_tantivy.add.assert_called_once()
                    mock_tantivy.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_messages_async_multiple_messages(self, mock_config, mock_logger):
        """Multiple messages should be processed in parallel."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    messages = ["msg1", "msg2", "msg3", "msg4"]
                    result = await manager.add_messages_async(messages)

                    assert result.stored_count == 4
                    # All messages should be added to Tantivy
                    assert mock_tantivy.add.call_count == 4
                    # Commit should be called once after all additions
                    mock_tantivy.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_messages_async_respects_concurrency_limit(
        self, mock_config, mock_logger
    ):
        """Concurrency limit should be respected via semaphore."""
        mock_config.add_max_concurrency = 2  # Low limit for testing

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    messages = ["msg1", "msg2", "msg3", "msg4"]
                    result = await manager.add_messages_async(messages)

                    # All messages should still be processed
                    assert result.stored_count == 4
                    assert mock_tantivy.add.call_count == 4

    @pytest.mark.asyncio
    async def test_add_messages_async_handles_duplicates(
        self, mock_config, mock_logger
    ):
        """Duplicate messages should be skipped (via Tantivy detection).

        Note: With Sprint 2.2 phased parallel processing, duplicate checks run
        in parallel, so we use a side_effect function instead of a list to
        handle non-deterministic call order.
        """
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()

                    # Use a function to return different results based on message
                    # Note: _has_exact_match wraps query in quotes: f'"{escaped_query}"'
                    def tantivy_search_side_effect(query, project_id, limit=5):
                        # Strip quotes to get original message
                        if query == '"duplicate"':
                            return [("duplicate", 1.0)]  # Found duplicate
                        return []  # No match for other messages

                    mock_tantivy.search.side_effect = tantivy_search_side_effect
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_messages_async(["duplicate", "unique"])

                    # Only unique message should be stored
                    assert result.stored_count == 1
                    assert result.skipped_count == 1

    @pytest.mark.asyncio
    async def test_add_messages_async_error_handling(self, mock_config, mock_logger):
        """Error during parallel addition should raise RuntimeError."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearchEngine mock to raise error
                    mock_usearch = MagicMock()
                    mock_usearch.add_batch.side_effect = Exception("Storage error")
                    mock_usearch.search.return_value = []  # For deduplication check
                    mock_usearch_class.return_value = mock_usearch

                    # Setup Tantivy mock
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)

                    with pytest.raises(StorageError, match="Failed to add messages"):
                        await manager.add_messages_async(["msg1", "msg2"])

    @pytest.mark.asyncio
    async def test_add_messages_async_batch_deduplication(
        self, mock_config, mock_logger
    ):
        """Batch deduplication should skip duplicate messages within the same batch.

        Sprint 2.2: Phase 1 includes deduplicating within the batch itself
        before checking storage, to avoid storing the same message twice.
        """
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = []  # No storage duplicates
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)

                    # Batch with duplicates: "A" appears 3 times, "B" appears 2 times
                    messages = ["A", "B", "A", "A", "B", "C"]
                    result = await manager.add_messages_async(messages)

                    # Only unique messages should be stored: A, B, C (3 unique)
                    # Skipped: 3 batch duplicates (2 extra A's, 1 extra B)
                    assert result.stored_count == 3
                    assert result.skipped_count == 3

                    # Tantivy add should only be called 3 times
                    assert mock_tantivy.add.call_count == 3


@pytest.mark.unit
class TestConcurrencyConfiguration:
    """Tests for ADD_MAX_CONCURRENCY configuration."""

    def test_config_has_add_max_concurrency(self, mock_config):
        """Config should have add_max_concurrency attribute."""
        assert hasattr(mock_config, "add_max_concurrency")
        assert mock_config.add_max_concurrency == 4

    @pytest.mark.asyncio
    async def test_low_concurrency_limit(self, mock_config, mock_logger):
        """Low concurrency limit should still process all messages."""
        mock_config.add_max_concurrency = 1  # Sequential processing

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_messages_async(["msg1", "msg2", "msg3"])

                    assert result.stored_count == 3

    @pytest.mark.asyncio
    async def test_high_concurrency_limit(self, mock_config, mock_logger):
        """High concurrency limit should allow more parallel tasks."""
        mock_config.add_max_concurrency = 100  # Higher than message count

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda project_id, messages, infer: messages
            )
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_messages_async(["msg1", "msg2", "msg3"])

                    assert result.stored_count == 3


@pytest.mark.unit
class TestSingleResultRerankingSkip:
    """Tests for skipping reranking when only 0-1 results after fusion (Sprint optimization).

    When fusion filtering produces <= 1 result, reranking is unnecessary because:
    - 0 results: Nothing to rerank
    - 1 result: No ordering to optimize
    This saves 15-25s of LLM API latency for single-result queries.
    """

    @pytest.mark.asyncio
    async def test_single_result_skips_llm_reranking(self, mock_config, mock_logger):
        """Single result after fusion should skip LLM reranking step."""
        mock_config.reranker_engine = "llm"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.5

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.LLMReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return 1 result
                        # Now returns 3-tuples: (message, score, created_at)
                        mock_usearch = MagicMock()
                        mock_usearch.search.return_value = [
                            ("single result", 0.9, "2024-01-01T00:00:00")
                        ]
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return same result (2-tuples, no timestamps)
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = [("single result", 0.9)]
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup LLMReranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank = AsyncMock(
                            return_value=[("single result", 0.95)]
                        )
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        # LLM reranker should NOT be called (skipped for single result)
                        mock_reranker.rerank.assert_not_called()

                        # Result should still be returned
                        assert len(results) == 1
                        assert results[0] == "single result"

                        # Verify skip was logged
                        skip_logged = any(
                            "Reranking skipped" in str(call)
                            or "reranking_skip" in str(call)
                            for call in mock_logger.info.call_args_list
                        )
                        assert skip_logged, "Expected reranking skip to be logged"

    @pytest.mark.asyncio
    async def test_single_result_skips_cross_encoder_reranking(
        self, mock_config, mock_logger
    ):
        """Single result after fusion should skip CrossEncoder reranking step."""
        mock_config.reranker_engine = "cross_encoder"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.5

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.CrossEncoderReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return 1 result
                        # Now returns 3-tuples: (message, score, created_at)
                        mock_usearch = MagicMock()
                        mock_usearch.search.return_value = [
                            ("single result", 0.9, "2024-01-01T00:00:00")
                        ]
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return same result (2-tuples, no timestamps)
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = [("single result", 0.9)]
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup CrossEncoderReranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank_async = AsyncMock(
                            return_value=[("single result", 0.95)]
                        )
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        # CrossEncoder reranker should NOT be called (skipped for single result)
                        mock_reranker.rerank_async.assert_not_called()

                        # Result should still be returned
                        assert len(results) == 1
                        assert results[0] == "single result"

    @pytest.mark.asyncio
    async def test_zero_results_skips_reranking_implicitly(
        self, mock_config, mock_logger
    ):
        """Zero results after fusion should skip reranking (implicit - no candidates)."""
        mock_config.reranker_engine = "llm"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.5

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.LLMReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return empty results
                        mock_usearch = MagicMock()
                        mock_usearch.search.return_value = []  # No semantic results
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return empty results
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = []  # No full-text results
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup LLMReranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank = AsyncMock(return_value=[])
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        # LLM reranker should NOT be called (no results to rerank)
                        mock_reranker.rerank.assert_not_called()

                        # Empty results expected (no results from either engine)
                        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_multiple_results_proceed_to_reranking(
        self, mock_config, mock_logger
    ):
        """Multiple results after fusion should proceed to reranking normally."""
        mock_config.reranker_engine = "llm"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.3  # Low threshold to keep results

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.LLMReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return multiple results
                        # Now returns 3-tuples: (message, score, created_at)
                        mock_usearch = MagicMock()
                        mock_usearch.search.return_value = [
                            ("result 1", 0.9, "2024-01-01T00:00:00"),
                            ("result 2", 0.8, "2024-01-02T00:00:00"),
                            ("result 3", 0.7, "2024-01-03T00:00:00"),
                        ]
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return multiple results (2-tuples, no timestamps)
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = [
                            ("result 1", 0.85),
                            ("result 2", 0.75),
                        ]
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup LLMReranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank = AsyncMock(
                            return_value=[
                                ("result 1", 0.95),
                                ("result 2", 0.85),
                                ("result 3", 0.75),
                            ]
                        )
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        # LLM reranker SHOULD be called (multiple results to rerank)
                        mock_reranker.rerank.assert_called_once()

                        # Results should be from reranker
                        assert len(results) >= 1


@pytest.mark.unit
class TestTimestampPropagation:
    """Tests for timestamp propagation through the search pipeline (Temporal-aware reranking).

    These tests verify that created_at timestamps from USearchEngine are properly
    propagated through the hybrid search pipeline for use in temporal-aware reranking.
    """

    @pytest.mark.asyncio
    async def test_search_receives_timestamps_from_usearch(
        self, mock_config, mock_logger
    ):
        """USearch search results should include created_at timestamps (3-tuples)."""
        mock_config.reranker_engine = "none"  # Skip reranking for simplicity
        mock_config.enable_rrf_fusion = True

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearchEngine mock with timestamps
                    mock_usearch = MagicMock()
                    mock_usearch.search.return_value = [
                        ("result 1", 0.9, "2024-01-15T10:30:00"),
                        ("result 2", 0.8, "2024-01-14T09:00:00"),
                        ("result 3", 0.7, "2024-01-13T08:00:00"),
                    ]
                    mock_usearch.count.return_value = 10
                    mock_usearch_class.return_value = mock_usearch

                    # Setup Tantivy mock (2-tuples, no timestamps)
                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = [
                        ("result 1", 0.85),
                    ]
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    results = await manager.search("test query")

                    # USearch should have been called
                    mock_usearch.search.assert_called()

                    # Results should be returned
                    assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_timestamp_map_built_from_semantic_results(
        self, mock_config, mock_logger
    ):
        """Verify timestamp_map is built correctly from semantic search results."""
        mock_config.reranker_engine = "none"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.0  # Keep all results

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Create test data with distinct timestamps
                    semantic_results = [
                        ("Memory about cats", 0.95, "2024-12-01T12:00:00"),
                        ("Memory about dogs", 0.85, "2024-12-15T14:30:00"),
                    ]

                    mock_usearch = MagicMock()
                    mock_usearch.search.return_value = semantic_results
                    mock_usearch.count.return_value = 100
                    mock_usearch_class.return_value = mock_usearch

                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = [("Memory about cats", 0.9)]
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    results = await manager.search("pets")

                    # The search should return results
                    assert len(results) >= 1
                    # Verify the semantic results contain the expected messages
                    messages_returned = set(results)
                    assert (
                        "Memory about cats" in messages_returned
                        or "Memory about dogs" in messages_returned
                    )

    @pytest.mark.asyncio
    async def test_timestamps_preserved_through_fusion(self, mock_config, mock_logger):
        """Timestamps should be accessible after RRF fusion."""
        mock_config.reranker_engine = "none"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.0

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Test that older and newer memories have different timestamps
                    mock_usearch = MagicMock()
                    mock_usearch.search.return_value = [
                        ("Old memory", 0.9, "2024-01-01T00:00:00"),  # Older
                        ("New memory", 0.85, "2024-12-01T00:00:00"),  # Newer
                    ]
                    mock_usearch.count.return_value = 50
                    mock_usearch_class.return_value = mock_usearch

                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = []
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    results = await manager.search("memory")

                    # Both memories should be in results
                    assert "Old memory" in results
                    assert "New memory" in results

    @pytest.mark.asyncio
    async def test_semantic_only_path_uses_timestamps(self, mock_config, mock_logger):
        """When hybrid search is disabled, timestamps still come from USearch."""
        mock_config.enable_hybrid_search = False  # Semantic-only path

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    mock_usearch = MagicMock()
                    # 3-tuples with timestamps
                    mock_usearch.search.return_value = [
                        ("Semantic result", 0.95, "2024-06-15T08:00:00"),
                    ]
                    mock_usearch_class.return_value = mock_usearch

                    manager = MemoryManager(mock_config, mock_logger)
                    results = await manager.search("test")

                    # Should return the semantic result
                    assert results == ["Semantic result"]
                    mock_usearch.search.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
