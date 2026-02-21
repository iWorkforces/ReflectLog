"""Unit tests for search_strategies.py - 4-step search pipeline.

Covers uncovered lines: 128-137, 295, 365, 374-378, 451, 457, 475,
493-495, 538-573, 671-683.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.exceptions import SearchError
from reflectlog.application.memory.search_strategies import (
    MIN_OVERFETCH_LIMIT,
    SearchContext,
    SearchPipeline,
    SearchResult,
    calculate_adaptive_overfetch,
)
from reflectlog.application.utils import StructuredLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> Mock:
    """Minimal mock Config for search_strategies tests."""
    config = Mock(spec=Config)
    config.project_id = "test_project"
    config.fusion_ranking_threshold = 0.1
    config.reranker_engine = "none"
    config.search_score_threshold = 0.5
    config.cross_encoder_model = "BAAI/bge-reranker-v2-m3"
    config.overfetch_multiplier = 3
    config.overfetch_adaptive = True
    config.overfetch_min_multiplier = 1.5
    config.overfetch_max_multiplier = 3.0
    return config


@pytest.fixture
def mock_logger() -> Mock:
    """Mock structured logger."""
    return Mock(spec=StructuredLogger)


@pytest.fixture
def mock_semantic_engine() -> MagicMock:
    """Mock ISemanticSearchEngine."""
    engine = MagicMock()
    engine.search = MagicMock(return_value=[])
    engine.ensure_initialized = MagicMock()
    return engine


@pytest.fixture
def mock_tantivy_engine() -> MagicMock:
    """Mock TantivyEngine."""
    engine = MagicMock()
    engine.search = MagicMock(return_value=[])
    engine.ensure_initialized = MagicMock()
    return engine


@pytest.fixture
def mock_fusion_engine() -> MagicMock:
    """Mock FusionEngine."""
    engine = MagicMock()
    engine.method = "rrf"
    engine.fuse = MagicMock(return_value=[])
    return engine


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Mock MemoryManager for lazy reranker fetching."""
    manager = MagicMock()
    manager.get_reranker = MagicMock(return_value=None)
    return manager


@pytest.fixture
def pipeline(
    mock_semantic_engine,
    mock_tantivy_engine,
    mock_fusion_engine,
    mock_config,
    mock_logger,
    mock_memory_manager,
) -> SearchPipeline:
    """Construct a SearchPipeline with mocked dependencies."""
    return SearchPipeline(
        semantic_engine=mock_semantic_engine,
        tantivy_engine=mock_tantivy_engine,
        fusion_engine=mock_fusion_engine,
        config=mock_config,
        logger=mock_logger,
        memory_manager=mock_memory_manager,
    )


def _make_context(
    *,
    query: str = "test query",
    limit: int = 5,
    overfetch_limit: int = 15,
    enable_hybrid_search: bool = True,
    enable_rrf_fusion: bool = True,
    reranker_engine: str = "none",
    project_id: str = "test_project",
) -> SearchContext:
    """Build a SearchContext with sensible defaults."""
    return SearchContext(
        query=query,
        limit=limit,
        overfetch_limit=overfetch_limit,
        enable_hybrid_search=enable_hybrid_search,
        enable_rrf_fusion=enable_rrf_fusion,
        reranker_engine=reranker_engine,
        project_id=project_id,
    )


# ---------------------------------------------------------------------------
# TestSearchContext & TestSearchResult
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchContext:
    """Tests for SearchContext dataclass."""

    def test_construction(self) -> None:
        """Fields should be stored correctly."""
        ctx = _make_context(query="hello", limit=10, project_id="proj")
        assert ctx.query == "hello"
        assert ctx.limit == 10
        assert ctx.project_id == "proj"


@pytest.mark.unit
class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_construction(self) -> None:
        """Fields should be stored correctly."""
        result = SearchResult(
            memories=["a", "b"],
            timestamp_map={"a": "2025-01-01T00:00:00Z"},
            semantic_results=[("a", 0.9, "2025-01-01T00:00:00Z")],
            tantivy_results=[("b", 0.8)],
        )
        assert result.memories == ["a", "b"]
        assert len(result.semantic_results) == 1
        assert len(result.tantivy_results) == 1


# ---------------------------------------------------------------------------
# TestSearchPipelineExecute – lines 128-137 (error path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchPipelineExecute:
    """Tests for SearchPipeline.execute()."""

    @pytest.mark.asyncio
    async def test_execute_raises_search_error_on_failure(
        self, pipeline, mock_semantic_engine, mock_logger
    ) -> None:
        """execute() wraps unexpected exceptions in SearchError (lines 128-137)."""
        mock_semantic_engine.search.side_effect = RuntimeError("engine boom")

        ctx = _make_context(enable_hybrid_search=False)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(ctx)

        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_semantic_only(self, pipeline, mock_semantic_engine) -> None:
        """Semantic-only search when hybrid is disabled."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        ctx = _make_context(enable_hybrid_search=False)

        result = await pipeline.execute(ctx)

        assert result.memories == ["msg1"]
        assert result.tantivy_results == []

    @pytest.mark.asyncio
    async def test_execute_hybrid_search(
        self,
        pipeline,
        mock_semantic_engine,
        mock_tantivy_engine,
        mock_fusion_engine,
    ) -> None:
        """Hybrid search invokes full pipeline."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = [("msg2", 0.8)]
        mock_fusion_engine.fuse.return_value = [("msg1", 0.5), ("msg2", 0.3)]

        ctx = _make_context(enable_hybrid_search=True, enable_rrf_fusion=True)

        result = await pipeline.execute(ctx)

        assert len(result.memories) >= 1


# ---------------------------------------------------------------------------
# TestSearchTantivy – line 295 (tantivy_engine is None)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchTantivy:
    """Tests for _search_tantivy()."""

    def test_returns_empty_when_tantivy_is_none(
        self,
        mock_config,
        mock_logger,
        mock_fusion_engine,
        mock_semantic_engine,
        mock_memory_manager,
    ) -> None:
        """_search_tantivy() returns [] when tantivy_engine is None (line 295)."""
        pipeline = SearchPipeline(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            fusion_engine=mock_fusion_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_memory_manager,
        )

        result = pipeline._search_tantivy("query", 10, "proj")
        assert result == []


# ---------------------------------------------------------------------------
# TestConcatenateResults – lines 365, 374-378
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConcatenateResults:
    """Tests for _concatenate_results()."""

    def test_limit_reached_during_semantic(self, pipeline) -> None:
        """Stops adding semantic results once limit is reached (line 365)."""
        semantic = [("s1", 0.9), ("s2", 0.8), ("s3", 0.7)]
        tantivy = [("t1", 0.6)]

        combined = pipeline._concatenate_results(semantic, tantivy, limit=2)

        assert len(combined) == 2
        assert combined[0][0] == "s1"
        assert combined[1][0] == "s2"

    def test_limit_reached_during_tantivy(self, pipeline) -> None:
        """Stops adding tantivy results once limit is reached (lines 374-378)."""
        semantic = [("s1", 0.9)]
        tantivy = [("t1", 0.6), ("t2", 0.5), ("t3", 0.4)]

        combined = pipeline._concatenate_results(semantic, tantivy, limit=2)

        assert len(combined) == 2
        assert combined[0][0] == "s1"
        assert combined[1][0] == "t1"

    def test_deduplication_across_engines(self, pipeline) -> None:
        """Duplicate memories across engines are skipped."""
        semantic = [("shared", 0.9)]
        tantivy = [("shared", 0.6), ("unique", 0.5)]

        combined = pipeline._concatenate_results(semantic, tantivy, limit=10)

        assert len(combined) == 2
        msgs = [m for m, _ in combined]
        assert msgs == ["shared", "unique"]


# ---------------------------------------------------------------------------
# TestGetReranker – lines 451, 457
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetReranker:
    """Tests for _get_reranker()."""

    def test_returns_none_when_memory_manager_is_none(
        self, mock_config, mock_logger, mock_fusion_engine, mock_semantic_engine
    ) -> None:
        """_get_reranker() returns (None, None) when memory_manager is None (line 451)."""
        pipeline = SearchPipeline(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            fusion_engine=mock_fusion_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=None,
        )

        rtype, rinstance = pipeline._get_reranker()
        assert rtype is None
        assert rinstance is None

    def test_returns_llm_reranker(
        self, pipeline, mock_config, mock_memory_manager
    ) -> None:
        """Returns ("llm", reranker) when config says llm and reranker exists."""
        mock_config.reranker_engine = "llm"
        mock_reranker = MagicMock()
        mock_memory_manager.get_reranker.return_value = mock_reranker

        rtype, rinstance = pipeline._get_reranker()
        assert rtype == "llm"
        assert rinstance is mock_reranker

    def test_returns_cross_encoder_reranker(
        self, pipeline, mock_config, mock_memory_manager
    ) -> None:
        """Returns ("cross_encoder", reranker) when configured (line 457)."""
        mock_config.reranker_engine = "cross_encoder"
        mock_reranker = MagicMock()
        mock_memory_manager.get_reranker.return_value = mock_reranker

        rtype, rinstance = pipeline._get_reranker()
        assert rtype == "cross_encoder"
        assert rinstance is mock_reranker

    def test_returns_none_when_reranker_is_none(
        self, pipeline, mock_config, mock_memory_manager
    ) -> None:
        """Returns (None, None) when get_reranker() returns None."""
        mock_config.reranker_engine = "llm"
        mock_memory_manager.get_reranker.return_value = None

        rtype, rinstance = pipeline._get_reranker()
        assert rtype is None
        assert rinstance is None


# ---------------------------------------------------------------------------
# TestStep4Reranking – line 475 (cross_encoder path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStep4Reranking:
    """Tests for _step4_reranking()."""

    @pytest.mark.asyncio
    async def test_cross_encoder_path(
        self, pipeline, mock_config, mock_memory_manager
    ) -> None:
        """_step4_reranking() delegates to _rerank_cross_encoder (line 475)."""
        mock_config.reranker_engine = "cross_encoder"
        mock_reranker = AsyncMock()
        mock_reranker.rerank_async = AsyncMock(return_value=[("msg1", 0.95)])
        mock_memory_manager.get_reranker.return_value = mock_reranker

        ctx = _make_context(reranker_engine="cross_encoder")
        results = [("msg1", 0.5), ("msg2", 0.3)]
        timestamp_map = {"msg1": "2025-01-01T00:00:00Z"}

        reranked = await pipeline._step4_reranking(ctx, results, timestamp_map, 4)

        assert reranked == [("msg1", 0.95)]

    @pytest.mark.asyncio
    async def test_no_reranker_returns_original(
        self, pipeline, mock_config, mock_memory_manager
    ) -> None:
        """When no reranker is configured, returns results unchanged."""
        mock_config.reranker_engine = "none"
        mock_memory_manager.get_reranker.return_value = None

        ctx = _make_context(reranker_engine="none")
        results = [("msg1", 0.5), ("msg2", 0.3)]

        reranked = await pipeline._step4_reranking(ctx, results, {}, 4)
        assert reranked == results


# ---------------------------------------------------------------------------
# TestRerankLLM – lines 493-495 (fallback when llm_reranker is None)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankLLM:
    """Tests for _rerank_llm()."""

    @pytest.mark.asyncio
    async def test_fallback_when_no_reranker_provided(
        self, pipeline, mock_config, mock_memory_manager
    ) -> None:
        """_rerank_llm() returns results unmodified when no reranker (lines 493-495)."""
        mock_config.reranker_engine = "none"
        mock_memory_manager.get_reranker.return_value = None

        ctx = _make_context()
        results = [("msg1", 0.5), ("msg2", 0.3)]

        reranked = await pipeline._rerank_llm(
            ctx, results, {}, step_num=4, llm_reranker=None
        )
        assert reranked == results

    @pytest.mark.asyncio
    async def test_llm_reranking_with_provided_reranker(
        self, pipeline, mock_config
    ) -> None:
        """_rerank_llm() calls reranker.rerank() and logs timing."""
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(return_value=[("msg1", 0.95)])

        ctx = _make_context()
        results = [("msg1", 0.5), ("msg2", 0.3)]
        ts_map = {"msg1": "2025-01-01T00:00:00Z"}

        reranked = await pipeline._rerank_llm(
            ctx, results, ts_map, step_num=4, llm_reranker=mock_reranker
        )
        assert reranked == [("msg1", 0.95)]
        mock_reranker.rerank.assert_awaited_once_with(ctx.query, results, ts_map)


# ---------------------------------------------------------------------------
# TestRerankCrossEncoder – lines 538-573
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankCrossEncoder:
    """Tests for _rerank_cross_encoder()."""

    @pytest.mark.asyncio
    async def test_fallback_when_no_reranker_provided(
        self, pipeline, mock_config, mock_memory_manager
    ) -> None:
        """Returns results unmodified when cross_encoder_reranker is None (lines 538-541)."""
        mock_config.reranker_engine = "none"
        mock_memory_manager.get_reranker.return_value = None

        ctx = _make_context()
        results = [("msg1", 0.5), ("msg2", 0.3)]

        reranked = await pipeline._rerank_cross_encoder(
            ctx, results, step_num=4, cross_encoder_reranker=None
        )
        assert reranked == results

    @pytest.mark.asyncio
    async def test_cross_encoder_reranking_with_provided_reranker(
        self, pipeline, mock_config, mock_logger
    ) -> None:
        """Full cross-encoder reranking path (lines 543-573)."""
        mock_reranker = AsyncMock()
        mock_reranker.rerank_async = AsyncMock(return_value=[("msg1", 0.95)])

        ctx = _make_context()
        results = [("msg1", 0.5), ("msg2", 0.3)]

        reranked = await pipeline._rerank_cross_encoder(
            ctx, results, step_num=4, cross_encoder_reranker=mock_reranker
        )

        assert reranked == [("msg1", 0.95)]
        mock_reranker.rerank_async.assert_awaited_once_with(ctx.query, results)

        # Verify logging calls happened
        assert mock_logger.info.call_count >= 2


# ---------------------------------------------------------------------------
# TestCalculateAdaptiveOverfetch – lines 671-683
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateAdaptiveOverfetch:
    """Tests for calculate_adaptive_overfetch()."""

    def test_static_multiplier_when_adaptive_disabled(self, mock_config) -> None:
        """Uses static multiplier when adaptive is disabled."""
        mock_config.overfetch_adaptive = False
        mock_config.overfetch_multiplier = 3

        result = calculate_adaptive_overfetch(5, 500, mock_config)
        assert result == max(5 * 3, MIN_OVERFETCH_LIMIT)

    def test_static_multiplier_when_index_empty(self, mock_config) -> None:
        """Uses static multiplier when index_size is 0."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_multiplier = 3

        result = calculate_adaptive_overfetch(5, 0, mock_config)
        assert result == max(5 * 3, MIN_OVERFETCH_LIMIT)

    def test_max_multiplier_for_small_index(self, mock_config) -> None:
        """Uses max multiplier for small indexes (<= 100)."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 50, mock_config)
        assert result == int(10 * 3.0)

    def test_min_multiplier_for_large_index(self, mock_config) -> None:
        """Uses min multiplier for large indexes (>= 10000)."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 10000, mock_config)
        assert result == max(int(10 * 1.5), MIN_OVERFETCH_LIMIT)

    def test_logarithmic_interpolation_mid_index(self, mock_config) -> None:
        """Logarithmic interpolation for mid-range index (lines 671-683)."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 1000, mock_config)

        # Should be between min and max multiplier
        assert int(10 * 1.5) <= result <= int(10 * 3.0)
        # And above the minimum overfetch limit
        assert result >= MIN_OVERFETCH_LIMIT

    def test_minimum_overfetch_limit_enforced(self, mock_config) -> None:
        """Result is never below MIN_OVERFETCH_LIMIT."""
        mock_config.overfetch_adaptive = False
        mock_config.overfetch_multiplier = 1

        result = calculate_adaptive_overfetch(1, 0, mock_config)
        assert result >= MIN_OVERFETCH_LIMIT

    def test_interpolation_at_boundary_100(self, mock_config) -> None:
        """Exact boundary at index_size=100 yields max multiplier."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 100, mock_config)
        assert result == int(10 * 3.0)

    def test_interpolation_at_boundary_10000(self, mock_config) -> None:
        """Exact boundary at index_size=10000 yields min multiplier."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 10000, mock_config)
        assert result == max(int(10 * 1.5), MIN_OVERFETCH_LIMIT)


# ---------------------------------------------------------------------------
# TestHybridSearchIntegration – end-to-end pipeline paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridSearchIntegration:
    """Integration-style tests covering full pipeline flows."""

    @pytest.mark.asyncio
    async def test_rrf_fusion_filters_all_results(
        self,
        pipeline,
        mock_semantic_engine,
        mock_tantivy_engine,
        mock_fusion_engine,
        mock_config,
    ) -> None:
        """When all fused results are below threshold, returns empty."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = [("msg2", 0.8)]
        mock_fusion_engine.fuse.return_value = [("msg1", 0.05), ("msg2", 0.01)]
        mock_config.fusion_ranking_threshold = 0.5

        ctx = _make_context(enable_hybrid_search=True, enable_rrf_fusion=True)
        result = await pipeline.execute(ctx)

        assert result.memories == []

    @pytest.mark.asyncio
    async def test_single_result_skips_reranking(
        self,
        pipeline,
        mock_semantic_engine,
        mock_tantivy_engine,
        mock_fusion_engine,
        mock_config,
        mock_logger,
    ) -> None:
        """A single result after fusion skips reranking."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = []
        mock_fusion_engine.fuse.return_value = [("msg1", 0.5)]
        mock_config.fusion_ranking_threshold = 0.1

        ctx = _make_context(enable_hybrid_search=True, enable_rrf_fusion=True)
        result = await pipeline.execute(ctx)

        assert result.memories == ["msg1"]
        # Check skip-reranking log was called
        skip_logged = any(
            "skipped" in str(call).lower() for call in mock_logger.info.call_args_list
        )
        assert skip_logged

    @pytest.mark.asyncio
    async def test_concatenation_mode_skips_fusion_threshold(
        self,
        pipeline,
        mock_semantic_engine,
        mock_tantivy_engine,
        mock_config,
        mock_logger,
    ) -> None:
        """When RRF fusion disabled, uses concatenation (no threshold step)."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = [("msg2", 0.8)]

        ctx = _make_context(
            enable_hybrid_search=True,
            enable_rrf_fusion=False,
        )
        result = await pipeline.execute(ctx)

        assert "msg1" in result.memories
        assert "msg2" in result.memories


# ---------------------------------------------------------------------------
# TestSearchSemantic – error fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchSemantic:
    """Tests for _search_semantic()."""

    def test_fallback_on_error(
        self, pipeline, mock_semantic_engine, mock_logger
    ) -> None:
        """Returns empty list and logs warning on exception."""
        mock_semantic_engine.search.side_effect = RuntimeError("embed fail")

        result = pipeline._search_semantic("query", 10, "proj")

        assert result == []
        mock_logger.warning.assert_called_once()
