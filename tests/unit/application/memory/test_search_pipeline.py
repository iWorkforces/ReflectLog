"""Unit tests for search_pipeline module."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.memory.search_pipeline import (
    ConcatenationFusionStage,
    DefaultBackendExecutor,
    NoopRerankerStage,
    RRFFusionStage,
    SearchPipeline,
    SearchPipelineConfig,
    ThresholdFilterStage,
    create_default_pipeline,
)
from reflectlog.core.search import ISearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(content: str, score: float, memory_id: str = "") -> ISearchResult:
    """Create an ISearchResult with minimal boilerplate."""
    return ISearchResult(content=content, score=score, memory_id=memory_id)


def _make_pipeline_config(
    *,
    enable_hybrid_search: bool = True,
    enable_rrf_fusion: bool = True,
    reranker_engine: str = "none",
    fusion_ranking_threshold: float = 0.01,
    search_limit: int = 5,
) -> SearchPipelineConfig:
    """Build a SearchPipelineConfig with sensible defaults."""
    return SearchPipelineConfig(
        enable_hybrid_search=enable_hybrid_search,
        enable_rrf_fusion=enable_rrf_fusion,
        reranker_engine=reranker_engine,
        fusion_ranking_threshold=fusion_ranking_threshold,
        search_limit=search_limit,
    )


# ---------------------------------------------------------------------------
# SearchPipelineConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchPipelineConfig:
    """Tests for SearchPipelineConfig dataclass."""

    def test_default_construction(self) -> None:
        """Config fields should be stored correctly."""
        cfg = SearchPipelineConfig(
            enable_hybrid_search=True,
            enable_rrf_fusion=False,
            reranker_engine="llm",
            fusion_ranking_threshold=0.5,
            search_limit=10,
        )
        assert cfg.enable_hybrid_search is True
        assert cfg.enable_rrf_fusion is False
        assert cfg.reranker_engine == "llm"
        assert cfg.fusion_ranking_threshold == 0.5
        assert cfg.search_limit == 10


# ---------------------------------------------------------------------------
# DefaultBackendExecutor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultBackendExecutor:
    """Tests for DefaultBackendExecutor."""

    @pytest.fixture
    def semantic_backend(self) -> AsyncMock:
        """Mock semantic search backend."""
        backend = AsyncMock()
        backend.search = AsyncMock(
            return_value=[
                _result("semantic hit 1", 0.95),
                _result("semantic hit 2", 0.80),
            ]
        )
        return backend

    @pytest.fixture
    def fulltext_backend(self) -> AsyncMock:
        """Mock full-text search backend."""
        backend = AsyncMock()
        backend.search = AsyncMock(
            return_value=[
                _result("fulltext hit 1", 0.90),
            ]
        )
        return backend

    @pytest.mark.asyncio
    async def test_semantic_only_when_no_fulltext(
        self,
        semantic_backend: AsyncMock,
    ) -> None:
        """Only semantic results when fulltext_backend is None."""
        executor = DefaultBackendExecutor(semantic_backend, None)
        from reflectlog.core.search import SearchContext

        ctx = SearchContext(
            query="test",
            limit=5,
            overfetch_limit=15,
            enable_hybrid_search=True,
            enable_rrf_fusion=True,
            reranker_engine="none",
            project_id="proj",
        )
        results = await executor.execute(ctx)

        assert "semantic" in results
        assert "fulltext" not in results
        assert len(results["semantic"]) == 2
        semantic_backend.search.assert_awaited_once_with(
            query="test",
            project_id="proj",
            limit=15,
        )

    @pytest.mark.asyncio
    async def test_hybrid_search_both_backends(
        self,
        semantic_backend: AsyncMock,
        fulltext_backend: AsyncMock,
    ) -> None:
        """Both backends executed when hybrid search enabled."""
        executor = DefaultBackendExecutor(semantic_backend, fulltext_backend)
        from reflectlog.core.search import SearchContext

        ctx = SearchContext(
            query="test",
            limit=5,
            overfetch_limit=15,
            enable_hybrid_search=True,
            enable_rrf_fusion=True,
            reranker_engine="none",
            project_id="proj",
        )
        results = await executor.execute(ctx)

        assert "semantic" in results
        assert "fulltext" in results
        assert len(results["semantic"]) == 2
        assert len(results["fulltext"]) == 1

    @pytest.mark.asyncio
    async def test_fulltext_skipped_when_hybrid_disabled(
        self,
        semantic_backend: AsyncMock,
        fulltext_backend: AsyncMock,
    ) -> None:
        """Fulltext backend skipped when enable_hybrid_search=False."""
        executor = DefaultBackendExecutor(semantic_backend, fulltext_backend)
        from reflectlog.core.search import SearchContext

        ctx = SearchContext(
            query="test",
            limit=5,
            overfetch_limit=15,
            enable_hybrid_search=False,
            enable_rrf_fusion=True,
            reranker_engine="none",
            project_id="proj",
        )
        results = await executor.execute(ctx)

        assert "semantic" in results
        assert "fulltext" not in results
        fulltext_backend.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# RRFFusionStage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRFFusionStage:
    """Tests for RRFFusionStage."""

    @pytest.fixture
    def fusion(self) -> RRFFusionStage:
        """RRFFusionStage with default k=60."""
        return RRFFusionStage(k=60.0)

    def test_empty_results(self, fusion: RRFFusionStage) -> None:
        """Empty input should return empty list."""
        assert fusion.fuse({}, limit=10) == []

    def test_single_backend(self, fusion: RRFFusionStage) -> None:
        """Single backend results returned with RRF scores."""
        results = {
            "semantic": [
                _result("A", 0.9),
                _result("B", 0.8),
            ],
        }
        fused = fusion.fuse(results, limit=10)

        assert len(fused) == 2
        # A was rank 1 → RRF = 1/(60+1), B was rank 2 → RRF = 1/(60+2)
        assert fused[0].content == "A"
        assert fused[1].content == "B"
        assert fused[0].score > fused[1].score

    def test_multiple_backends_aggregation(self, fusion: RRFFusionStage) -> None:
        """Results from multiple backends get aggregated RRF scores."""
        results = {
            "semantic": [_result("A", 0.9), _result("B", 0.8)],
            "fulltext": [_result("B", 0.95), _result("C", 0.7)],
        }
        fused = fusion.fuse(results, limit=10)

        contents = [r.content for r in fused]
        assert "A" in contents
        assert "B" in contents
        assert "C" in contents
        # B appears in both backends → highest aggregated score
        b_item = next(r for r in fused if r.content == "B")
        a_item = next(r for r in fused if r.content == "A")
        assert b_item.score > a_item.score

    def test_limit_respected(self, fusion: RRFFusionStage) -> None:
        """Only limit number of results returned."""
        results = {
            "semantic": [_result(f"item{i}", 0.9 - i * 0.1) for i in range(5)],
        }
        fused = fusion.fuse(results, limit=2)
        assert len(fused) == 2

    def test_custom_k_value(self) -> None:
        """Custom k changes RRF scores."""
        fusion_k10 = RRFFusionStage(k=10.0)
        fusion_k100 = RRFFusionStage(k=100.0)

        results = {"semantic": [_result("X", 0.9)]}

        fused_k10 = fusion_k10.fuse(results, limit=1)
        fused_k100 = fusion_k100.fuse(results, limit=1)

        # Lower k → higher score for top rank: 1/(10+1) > 1/(100+1)
        assert fused_k10[0].score > fused_k100[0].score

    def test_rrf_score_correctness(self) -> None:
        """Verify RRF formula: score = sum(1/(k+rank))."""
        fusion = RRFFusionStage(k=60.0)
        results = {
            "semantic": [_result("A", 0.9)],
            "fulltext": [_result("A", 0.85)],
        }
        fused = fusion.fuse(results, limit=10)

        expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)  # rank 1 in both
        assert abs(fused[0].score - expected_score) < 1e-9

    def test_original_score_in_metadata(self, fusion: RRFFusionStage) -> None:
        """Original score from first matching backend stored in metadata."""
        results = {
            "semantic": [_result("A", 0.92)],
        }
        fused = fusion.fuse(results, limit=10)
        assert fused[0].metadata["original_score"] == 0.92

    def test_memory_id_empty_string(self, fusion: RRFFusionStage) -> None:
        """Fused results get empty memory_id."""
        results = {
            "semantic": [_result("A", 0.9, memory_id="id1")],
        }
        fused = fusion.fuse(results, limit=10)
        assert fused[0].memory_id == ""

    def test_deduplication_by_content(self, fusion: RRFFusionStage) -> None:
        """Duplicate content from same backend is deduplicated."""
        results = {
            "semantic": [_result("dup", 0.9), _result("dup", 0.8)],
        }
        fused = fusion.fuse(results, limit=10)
        # Rankings dict uses content as key so second "dup" overwrites first
        # But aggregated dict still only has one entry for "dup"
        dup_items = [r for r in fused if r.content == "dup"]
        assert len(dup_items) == 1


# ---------------------------------------------------------------------------
# ConcatenationFusionStage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConcatenationFusionStage:
    """Tests for ConcatenationFusionStage."""

    @pytest.fixture
    def fusion(self) -> ConcatenationFusionStage:
        """ConcatenationFusionStage instance."""
        return ConcatenationFusionStage()

    def test_empty_results(self, fusion: ConcatenationFusionStage) -> None:
        """Empty input should return empty list."""
        assert fusion.fuse({}, limit=10) == []

    def test_single_backend(self, fusion: ConcatenationFusionStage) -> None:
        """Results from one backend returned sorted by score."""
        results = {
            "semantic": [_result("B", 0.7), _result("A", 0.9)],
        }
        fused = fusion.fuse(results, limit=10)
        assert fused[0].content == "A"
        assert fused[1].content == "B"

    def test_multiple_backends_combined(
        self,
        fusion: ConcatenationFusionStage,
    ) -> None:
        """Results from multiple backends combined and sorted."""
        results = {
            "semantic": [_result("A", 0.9)],
            "fulltext": [_result("B", 0.95)],
        }
        fused = fusion.fuse(results, limit=10)
        assert len(fused) == 2
        assert fused[0].content == "B"  # higher score
        assert fused[1].content == "A"

    def test_deduplication(self, fusion: ConcatenationFusionStage) -> None:
        """Duplicate content deduplicated (first occurrence kept)."""
        results = {
            "semantic": [_result("same", 0.8)],
            "fulltext": [_result("same", 0.95)],
        }
        fused = fusion.fuse(results, limit=10)
        assert len(fused) == 1
        # First occurrence wins → score from semantic
        assert fused[0].score == 0.8

    def test_limit_respected(self, fusion: ConcatenationFusionStage) -> None:
        """Only limit number of results returned."""
        results = {
            "semantic": [_result(f"item{i}", 0.9 - i * 0.1) for i in range(5)],
        }
        fused = fusion.fuse(results, limit=2)
        assert len(fused) == 2


# ---------------------------------------------------------------------------
# ThresholdFilterStage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThresholdFilterStage:
    """Tests for ThresholdFilterStage."""

    @pytest.fixture
    def filter_stage(self) -> ThresholdFilterStage:
        """ThresholdFilterStage instance."""
        return ThresholdFilterStage()

    def test_all_above_threshold(
        self,
        filter_stage: ThresholdFilterStage,
    ) -> None:
        """All results kept when above threshold."""
        results = [_result("A", 0.9), _result("B", 0.8)]
        filtered = filter_stage.filter(results, threshold=0.5)
        assert len(filtered) == 2

    def test_all_below_threshold(
        self,
        filter_stage: ThresholdFilterStage,
    ) -> None:
        """All results removed when below threshold."""
        results = [_result("A", 0.3), _result("B", 0.2)]
        filtered = filter_stage.filter(results, threshold=0.5)
        assert len(filtered) == 0

    def test_partial_filtering(
        self,
        filter_stage: ThresholdFilterStage,
    ) -> None:
        """Only results above threshold kept."""
        results = [_result("A", 0.9), _result("B", 0.3)]
        filtered = filter_stage.filter(results, threshold=0.5)
        assert len(filtered) == 1
        assert filtered[0].content == "A"

    def test_exact_threshold_included(
        self,
        filter_stage: ThresholdFilterStage,
    ) -> None:
        """Score exactly at threshold is included (>=)."""
        results = [_result("A", 0.5)]
        filtered = filter_stage.filter(results, threshold=0.5)
        assert len(filtered) == 1

    def test_empty_input(self, filter_stage: ThresholdFilterStage) -> None:
        """Empty list returns empty list."""
        filtered = filter_stage.filter([], threshold=0.5)
        assert filtered == []

    def test_zero_threshold_keeps_all(
        self,
        filter_stage: ThresholdFilterStage,
    ) -> None:
        """Zero threshold keeps everything."""
        results = [_result("A", 0.01), _result("B", 0.0)]
        filtered = filter_stage.filter(results, threshold=0.0)
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# NoopRerankerStage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoopRerankerStage:
    """Tests for NoopRerankerStage."""

    @pytest.mark.asyncio
    async def test_passthrough(self) -> None:
        """Results returned unchanged."""
        stage = NoopRerankerStage()
        results = [_result("A", 0.9), _result("B", 0.8)]
        reranked = await stage.rerank("query", results)
        assert reranked == results

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        """Empty list returns empty list."""
        stage = NoopRerankerStage()
        reranked = await stage.rerank("query", [])
        assert reranked == []


# ---------------------------------------------------------------------------
# SearchPipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchPipeline:
    """Tests for SearchPipeline orchestration."""

    @pytest.fixture
    def mock_backend_executor(self) -> AsyncMock:
        """Mock backend executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value={
                "semantic": [
                    _result("hit1", 0.9),
                    _result("hit2", 0.8),
                ],
            }
        )
        return executor

    @pytest.fixture
    def mock_fusion(self) -> MagicMock:
        """Mock fusion stage."""
        fusion = MagicMock()
        fusion.fuse.return_value = [
            _result("hit1", 0.032),
            _result("hit2", 0.016),
        ]
        return fusion

    @pytest.fixture
    def mock_filter(self) -> MagicMock:
        """Mock filter stage."""
        filter_stage = MagicMock()
        filter_stage.filter.side_effect = lambda results, threshold: [
            r for r in results if r.score >= threshold
        ]
        return filter_stage

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Mock structured logger."""
        return MagicMock()

    def _make_pipeline(
        self,
        executor: AsyncMock,
        fusion: MagicMock,
        filter_stage: MagicMock,
        logger: MagicMock,
        *,
        reranker: Any = None,
        config: SearchPipelineConfig | None = None,
    ) -> SearchPipeline:
        """Build a SearchPipeline with provided or default config."""
        if config is None:
            config = _make_pipeline_config()
        return SearchPipeline(
            backend_executor=executor,
            fusion_stage=fusion,
            filter_stage=filter_stage,
            reranker_stage=reranker,
            config=config,
            logger=logger,
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_no_reranker(
        self,
        mock_backend_executor: AsyncMock,
        mock_fusion: MagicMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Pipeline returns content strings without reranker."""
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
        )
        results = await pipeline.execute("test query", "proj-1")

        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)
        mock_backend_executor.execute.assert_awaited_once()
        mock_fusion.fuse.assert_called_once()

    @pytest.mark.asyncio
    async def test_limit_defaults_from_config(
        self,
        mock_backend_executor: AsyncMock,
        mock_fusion: MagicMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """When limit=None, uses config.search_limit."""
        cfg = _make_pipeline_config(search_limit=3)
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            config=cfg,
        )
        results = await pipeline.execute("query", "proj")

        # Fusion receives config limit
        _, call_kwargs = mock_fusion.fuse.call_args
        assert call_kwargs.get("limit", mock_fusion.fuse.call_args[0][1]) == 3

    @pytest.mark.asyncio
    async def test_explicit_limit_overrides_config(
        self,
        mock_backend_executor: AsyncMock,
        mock_fusion: MagicMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Explicit limit overrides config default."""
        cfg = _make_pipeline_config(search_limit=10)
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            config=cfg,
        )
        await pipeline.execute("query", "proj", limit=2)

        call_args = mock_fusion.fuse.call_args
        assert call_args[0][1] == 2  # limit positional arg

    @pytest.mark.asyncio
    async def test_filter_applied_when_rrf_enabled(
        self,
        mock_backend_executor: AsyncMock,
        mock_fusion: MagicMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Filter stage called when enable_rrf_fusion=True."""
        cfg = _make_pipeline_config(
            enable_rrf_fusion=True,
            fusion_ranking_threshold=0.02,
        )
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            config=cfg,
        )
        await pipeline.execute("query", "proj")

        mock_filter.filter.assert_called_once()
        filter_args = mock_filter.filter.call_args
        assert filter_args[0][1] == 0.02  # threshold

    @pytest.mark.asyncio
    async def test_filter_skipped_when_rrf_disabled(
        self,
        mock_backend_executor: AsyncMock,
        mock_fusion: MagicMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Filter stage NOT called when enable_rrf_fusion=False."""
        cfg = _make_pipeline_config(enable_rrf_fusion=False)
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            config=cfg,
        )
        await pipeline.execute("query", "proj")

        mock_filter.filter.assert_not_called()

    @pytest.mark.asyncio
    async def test_reranker_applied_when_present(
        self,
        mock_backend_executor: AsyncMock,
        mock_fusion: MagicMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Reranker transforms results when provided."""
        mock_reranker = AsyncMock()

        @dataclass(frozen=True)
        class FakeRankingResult:
            content: str
            score: float
            metadata: dict[str, Any]

        mock_reranker.rerank = AsyncMock(
            return_value=[
                FakeRankingResult(content="hit1", score=0.99, metadata={}),
            ]
        )

        cfg = _make_pipeline_config(
            enable_rrf_fusion=True, fusion_ranking_threshold=0.0
        )
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            reranker=mock_reranker,
            config=cfg,
        )
        results = await pipeline.execute("test query", "proj")

        mock_reranker.rerank.assert_awaited_once()
        assert "hit1" in results

    @pytest.mark.asyncio
    async def test_reranker_skipped_for_empty_memories(
        self,
        mock_backend_executor: AsyncMock,
        mock_logger: MagicMock,
    ) -> None:
        """Reranker not called when filtered results are empty."""
        mock_fusion = MagicMock()
        mock_fusion.fuse.return_value = []

        mock_filter = MagicMock()
        mock_filter.filter.return_value = []

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock()

        cfg = _make_pipeline_config(
            enable_rrf_fusion=True, fusion_ranking_threshold=0.0
        )
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            reranker=mock_reranker,
            config=cfg,
        )
        results = await pipeline.execute("query", "proj")

        mock_reranker.rerank.assert_not_awaited()
        assert results == []

    @pytest.mark.asyncio
    async def test_results_truncated_to_limit(
        self,
        mock_backend_executor: AsyncMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Final results truncated to limit."""
        mock_fusion = MagicMock()
        mock_fusion.fuse.return_value = [
            _result(f"hit{i}", 0.9 - i * 0.01) for i in range(10)
        ]

        cfg = _make_pipeline_config(
            enable_rrf_fusion=False,
            search_limit=3,
        )
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            config=cfg,
        )
        results = await pipeline.execute("query", "proj")

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_context_constructed_correctly(
        self,
        mock_backend_executor: AsyncMock,
        mock_fusion: MagicMock,
        mock_filter: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """SearchContext passed to executor has correct fields."""
        cfg = _make_pipeline_config(
            enable_hybrid_search=True,
            enable_rrf_fusion=False,
            reranker_engine="llm",
            search_limit=7,
        )
        pipeline = self._make_pipeline(
            mock_backend_executor,
            mock_fusion,
            mock_filter,
            mock_logger,
            config=cfg,
        )
        await pipeline.execute("my query", "my-proj", limit=7)

        ctx = mock_backend_executor.execute.call_args[0][0]
        assert ctx.query == "my query"
        assert ctx.project_id == "my-proj"
        assert ctx.limit == 7
        assert ctx.enable_hybrid_search is True
        assert ctx.enable_rrf_fusion is False
        assert ctx.reranker_engine == "llm"


# ---------------------------------------------------------------------------
# SearchPipeline._calculate_overfetch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateOverfetch:
    """Tests for SearchPipeline._calculate_overfetch."""

    @pytest.fixture
    def pipeline(self) -> SearchPipeline:
        """Minimal pipeline for overfetch tests."""
        return SearchPipeline(
            backend_executor=AsyncMock(),
            fusion_stage=MagicMock(),
            filter_stage=MagicMock(),
            reranker_stage=None,
            config=_make_pipeline_config(),
            logger=MagicMock(),
        )

    @pytest.mark.parametrize(
        "limit,expected",
        [
            (1, 3),  # 1*3 = 3, min(3, 50) = 3
            (5, 15),  # 5*3 = 15, min(15, 50) = 15
            (10, 30),  # 10*3 = 30, min(30, 50) = 30
            (20, 40),  # limit>10: 20*2 = 40, min(40, 100) = 40
            (50, 100),  # 50*2 = 100, min(100, 100) = 100
            (51, 76),  # limit>50: int(51*1.5) = 76
            (100, 150),  # int(100*1.5) = 150
        ],
    )
    def test_overfetch_ranges(
        self,
        pipeline: SearchPipeline,
        limit: int,
        expected: int,
    ) -> None:
        """Overfetch multiplier adapts to limit size."""
        assert pipeline._calculate_overfetch(limit) == expected

    def test_overfetch_small_limit_capped(self, pipeline: SearchPipeline) -> None:
        """Small limits capped at 50."""
        result = pipeline._calculate_overfetch(10)
        assert result <= 50

    def test_overfetch_medium_limit_capped(self, pipeline: SearchPipeline) -> None:
        """Medium limits capped at 100."""
        result = pipeline._calculate_overfetch(50)
        assert result <= 100


# ---------------------------------------------------------------------------
# create_default_pipeline factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDefaultPipeline:
    """Tests for create_default_pipeline factory function."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Mock Config with required attributes."""
        cfg = MagicMock()
        cfg.enable_hybrid_search = True
        cfg.enable_rrf_fusion = True
        cfg.fusion_rrf_k = 60
        cfg.fusion_ranking_threshold = 0.8
        cfg.search_limit = 5
        cfg.reranker_engine = "llm"
        return cfg

    @pytest.fixture
    def mock_semantic(self) -> AsyncMock:
        """Mock semantic backend."""
        return AsyncMock()

    @pytest.fixture
    def mock_fulltext(self) -> AsyncMock:
        """Mock fulltext backend."""
        return AsyncMock()

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Mock structured logger."""
        return MagicMock()

    def test_creates_pipeline_with_rrf_fusion(
        self,
        mock_semantic: AsyncMock,
        mock_fulltext: AsyncMock,
        mock_config: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """With enable_rrf_fusion=True, uses RRFFusionStage."""
        mock_config.enable_rrf_fusion = True
        pipeline = create_default_pipeline(
            mock_semantic,
            mock_fulltext,
            None,
            mock_config,
            mock_logger,
        )
        assert isinstance(pipeline, SearchPipeline)
        assert isinstance(pipeline._fusion_stage, RRFFusionStage)

    def test_creates_pipeline_with_concatenation_fusion(
        self,
        mock_semantic: AsyncMock,
        mock_fulltext: AsyncMock,
        mock_config: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """With enable_rrf_fusion=False, uses ConcatenationFusionStage."""
        mock_config.enable_rrf_fusion = False
        pipeline = create_default_pipeline(
            mock_semantic,
            mock_fulltext,
            None,
            mock_config,
            mock_logger,
        )
        assert isinstance(pipeline._fusion_stage, ConcatenationFusionStage)

    def test_reranker_passed_through(
        self,
        mock_semantic: AsyncMock,
        mock_config: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Reranker instance passed to pipeline."""
        mock_reranker = AsyncMock()
        pipeline = create_default_pipeline(
            mock_semantic,
            None,
            mock_reranker,
            mock_config,
            mock_logger,
        )
        assert pipeline._reranker is mock_reranker

    def test_no_reranker_is_none(
        self,
        mock_semantic: AsyncMock,
        mock_config: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """None reranker stored as None."""
        pipeline = create_default_pipeline(
            mock_semantic,
            None,
            None,
            mock_config,
            mock_logger,
        )
        assert pipeline._reranker is None

    def test_pipeline_config_populated(
        self,
        mock_semantic: AsyncMock,
        mock_config: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Pipeline config fields match source config."""
        mock_config.enable_hybrid_search = True
        mock_config.enable_rrf_fusion = True
        mock_config.reranker_engine = "cross_encoder"
        mock_config.fusion_ranking_threshold = 0.75
        mock_config.search_limit = 8
        pipeline = create_default_pipeline(
            mock_semantic,
            None,
            None,
            mock_config,
            mock_logger,
        )
        assert pipeline._config.enable_hybrid_search is True
        assert pipeline._config.enable_rrf_fusion is True
        assert pipeline._config.reranker_engine == "cross_encoder"
        assert pipeline._config.fusion_ranking_threshold == 0.75
        assert pipeline._config.search_limit == 8

    def test_rrf_k_passed_to_fusion_stage(
        self,
        mock_semantic: AsyncMock,
        mock_config: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Custom fusion_rrf_k propagated to RRFFusionStage."""
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_rrf_k = 42
        pipeline = create_default_pipeline(
            mock_semantic,
            None,
            None,
            mock_config,
            mock_logger,
        )
        assert isinstance(pipeline._fusion_stage, RRFFusionStage)
        assert pipeline._fusion_stage._k == 42

    def test_fulltext_none_allowed(
        self,
        mock_semantic: AsyncMock,
        mock_config: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Pipeline can be created with fulltext_backend=None."""
        pipeline = create_default_pipeline(
            mock_semantic,
            None,
            None,
            mock_config,
            mock_logger,
        )
        assert isinstance(pipeline, SearchPipeline)


# ---------------------------------------------------------------------------
# Integration-style tests (all real stages, no mocks for stages)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchPipelineIntegration:
    """Integration tests using real stage implementations."""

    @pytest.mark.asyncio
    async def test_end_to_end_with_rrf(self) -> None:
        """Full pipeline with real RRF fusion and threshold filter."""
        semantic = AsyncMock()
        semantic.search = AsyncMock(
            return_value=[
                _result("Python is great", 0.95),
                _result("Java is fine", 0.80),
            ]
        )
        fulltext = AsyncMock()
        fulltext.search = AsyncMock(
            return_value=[
                _result("Python is great", 0.90),
                _result("Go is fast", 0.85),
            ]
        )

        executor = DefaultBackendExecutor(semantic, fulltext)
        fusion = RRFFusionStage(k=60)
        filter_stage = ThresholdFilterStage()
        cfg = _make_pipeline_config(
            enable_hybrid_search=True,
            enable_rrf_fusion=True,
            fusion_ranking_threshold=0.0,
            search_limit=10,
        )

        pipeline = SearchPipeline(
            backend_executor=executor,
            fusion_stage=fusion,
            filter_stage=filter_stage,
            reranker_stage=None,
            config=cfg,
            logger=MagicMock(),
        )

        results = await pipeline.execute("Python", "proj")

        assert isinstance(results, list)
        assert len(results) > 0
        # "Python is great" appears in both → should be top
        assert results[0] == "Python is great"

    @pytest.mark.asyncio
    async def test_end_to_end_with_concatenation(self) -> None:
        """Full pipeline with concatenation fusion (no RRF)."""
        semantic = AsyncMock()
        semantic.search = AsyncMock(
            return_value=[
                _result("alpha", 0.9),
                _result("beta", 0.7),
            ]
        )

        executor = DefaultBackendExecutor(semantic, None)
        fusion = ConcatenationFusionStage()
        filter_stage = ThresholdFilterStage()
        cfg = _make_pipeline_config(
            enable_hybrid_search=False,
            enable_rrf_fusion=False,
            search_limit=5,
        )

        pipeline = SearchPipeline(
            backend_executor=executor,
            fusion_stage=fusion,
            filter_stage=filter_stage,
            reranker_stage=None,
            config=cfg,
            logger=MagicMock(),
        )

        results = await pipeline.execute("test", "proj")

        assert results == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_end_to_end_with_mock_reranker(self) -> None:
        """Full pipeline with a reranker returning IRankingResult objects."""
        semantic = AsyncMock()
        semantic.search = AsyncMock(
            return_value=[
                _result("item1", 0.9),
            ]
        )

        @dataclass(frozen=True)
        class FakeRanking:
            content: str
            score: float
            metadata: dict[str, Any]

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(
            return_value=[
                FakeRanking(content="item1", score=0.95, metadata={}),
            ]
        )

        executor = DefaultBackendExecutor(semantic, None)
        fusion = ConcatenationFusionStage()
        filter_stage = ThresholdFilterStage()
        cfg = _make_pipeline_config(
            enable_hybrid_search=False,
            enable_rrf_fusion=False,
            search_limit=5,
        )

        pipeline = SearchPipeline(
            backend_executor=executor,
            fusion_stage=fusion,
            filter_stage=filter_stage,
            reranker_stage=mock_reranker,
            config=cfg,
            logger=MagicMock(),
        )

        results = await pipeline.execute("q", "proj")

        assert results == ["item1"]
        mock_reranker.rerank.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_threshold_filters_low_rrf_scores(self) -> None:
        """High threshold filters out low RRF scores."""
        semantic = AsyncMock()
        semantic.search = AsyncMock(
            return_value=[
                _result("good", 0.95),
                _result("bad", 0.1),
            ]
        )

        executor = DefaultBackendExecutor(semantic, None)
        fusion = RRFFusionStage(k=60)
        filter_stage = ThresholdFilterStage()
        # RRF scores are very small (1/(60+rank)), so 0.5 threshold filters all
        cfg = _make_pipeline_config(
            enable_hybrid_search=False,
            enable_rrf_fusion=True,
            fusion_ranking_threshold=0.5,
            search_limit=10,
        )

        pipeline = SearchPipeline(
            backend_executor=executor,
            fusion_stage=fusion,
            filter_stage=filter_stage,
            reranker_stage=None,
            config=cfg,
            logger=MagicMock(),
        )

        results = await pipeline.execute("test", "proj")

        # RRF score for rank 1 = 1/61 ≈ 0.016 < 0.5, all filtered
        assert results == []
