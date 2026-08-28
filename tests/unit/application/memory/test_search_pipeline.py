"""Canonical SearchPipeline tests: behavior plus event-loop responsiveness.

These tests target ``search_strategies.SearchPipeline`` — the single
production search path used by MemoryManager.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.search_strategies import (
    SearchContext,
    SearchPipeline,
    calculate_adaptive_overfetch,
)
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.exceptions import SearchError
from reflectlog.core.logging import IStructuredLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BACKEND_WAIT_TIMEOUT = 1.0
_TS = "2025-01-01T00:00:00Z"


def _make_config(
    *,
    fusion_ranking_threshold: float = 0.1,
    reranker_engine: str = "none",
) -> Mock:
    config = Mock(spec=Config)
    config.project_id = "test_project"
    config.fusion_ranking_threshold = fusion_ranking_threshold
    config.reranker_engine = reranker_engine
    config.search_score_threshold = 0.5
    config.cross_encoder_model = "BAAI/bge-reranker-v2-m3"
    config.overfetch_multiplier = 3
    config.overfetch_adaptive = True
    config.overfetch_min_multiplier = 1.5
    config.overfetch_max_multiplier = 3.0
    return config


def _make_logger() -> IStructuredLogger:
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


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
    return SearchContext(
        query=query,
        limit=limit,
        overfetch_limit=overfetch_limit,
        enable_hybrid_search=enable_hybrid_search,
        enable_rrf_fusion=enable_rrf_fusion,
        reranker_engine=reranker_engine,
        project_id=project_id,
    )


def _make_pipeline(
    *,
    semantic: Any,
    tantivy: Any = None,
    fusion: Any = None,
    config: Mock | None = None,
    logger: IStructuredLogger | None = None,
    memory_manager: Any = None,
) -> SearchPipeline:
    if fusion is None:
        fusion_engine = MagicMock()
        fusion_engine.method = "rrf"
        fusion_engine.fuse = MagicMock(return_value=[])
    else:
        fusion_engine = fusion
    if memory_manager is None:
        manager = MagicMock()
        manager.llm_reranker = None
        manager.cross_encoder_reranker = None
    else:
        manager = memory_manager
    return SearchPipeline(
        semantic_engine=semantic,
        tantivy_engine=tantivy,
        fusion_engine=fusion_engine,
        config=config or _make_config(),
        logger=logger or _make_logger(),
        memory_manager=manager,
    )


class ControllableBackend:
    """Synchronous backend that can block until the event loop progresses."""

    def __init__(
        self,
        results: list[object],
        *,
        entered: threading.Event,
        loop_progressed: threading.Event,
        barrier: threading.Barrier | None = None,
        thread_ids: list[int] | None = None,
        fail: Exception | None = None,
    ) -> None:
        self.results = results
        self.entered = entered
        self.loop_progressed = loop_progressed
        self.barrier = barrier
        self.thread_ids = thread_ids
        self.fail = fail
        self.search_calls: list[tuple[str, str, int]] = []

    def ensure_initialized(self) -> None:
        return None

    def search(self, query: str, project_id: str, limit: int) -> list[object]:
        self.search_calls.append((query, project_id, limit))
        if self.thread_ids is not None:
            self.thread_ids.append(threading.get_ident())
        self.entered.set()
        if self.barrier is not None:
            _ = self.barrier.wait(timeout=_BACKEND_WAIT_TIMEOUT)
        if not self.loop_progressed.wait(timeout=_BACKEND_WAIT_TIMEOUT):
            raise TimeoutError("event loop did not progress while backend blocked")
        if self.fail is not None:
            raise self.fail
        return list(self.results)


# ---------------------------------------------------------------------------
# Single canonical pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanonicalPipelineIdentity:
    """Guard against a second production search implementation."""

    def test_search_pipeline_module_removed(self) -> None:
        """Compatibility re-export module no longer exists."""
        import importlib

        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module(
                "reflectlog.application.memory.search_pipeline"
            )

    def test_manager_wires_strategies_pipeline(self) -> None:
        """MemoryManager._init_pipelines() uses search_strategies.SearchPipeline."""
        config = MagicMock()
        config.project_id = "test_project"
        config.enable_hybrid_search = True
        config.tantivy_index_path_template = "{project_id}_tantivy_test"
        config.enable_smart_replace = False
        config.reranker_engine = "none"
        config.embedding_cache_enabled = False
        config.eager_initialization = False
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.openrouter_api_key.get_secret_value.return_value = "test-key"

        with (
            patch("reflectlog.application.memory.manager.USearchEngine"),
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            manager = MemoryManager(config, _make_logger())

        assert isinstance(manager._search_pipeline, SearchPipeline)
        assert (
            type(manager._search_pipeline).__module__
            == "reflectlog.application.memory.search_strategies"
        )

    def test_logger_required(self) -> None:
        """Construction still requires a logger."""
        with pytest.raises(ValueError, match="logger is required"):
            _ = SearchPipeline(
                semantic_engine=MagicMock(),
                tantivy_engine=None,
                fusion_engine=MagicMock(),
                config=_make_config(),
                logger=None,
                memory_manager=None,
            )


# ---------------------------------------------------------------------------
# Semantic-only and hybrid behavior (migrated from old pipeline tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticOnlyBehavior:
    """Semantic-only path: empty, one result, exception, ordering."""

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = []
        pipeline = _make_pipeline(semantic=semantic)

        result = await pipeline.execute(_make_context(enable_hybrid_search=False))

        assert result.memories == []
        assert result.timestamp_map == {}
        assert result.semantic_results == []
        assert result.tantivy_results == []

    @pytest.mark.asyncio
    async def test_single_result_preserves_order_and_timestamp(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("only", 0.9, _TS)]
        pipeline = _make_pipeline(semantic=semantic)

        result = await pipeline.execute(_make_context(enable_hybrid_search=False))

        assert result.memories == ["only"]
        assert result.timestamp_map == {"only": _TS}
        semantic.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_results_keep_backend_order(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [
            ("alpha", 0.9, _TS),
            ("beta", 0.7, "2025-02-01T00:00:00Z"),
        ]
        pipeline = _make_pipeline(semantic=semantic)

        result = await pipeline.execute(
            _make_context(enable_hybrid_search=False, limit=5)
        )

        assert result.memories == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_engine_failure_raises_search_error(self) -> None:
        semantic = MagicMock()
        semantic.search.side_effect = RuntimeError("engine boom")
        pipeline = _make_pipeline(semantic=semantic)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(_make_context(enable_hybrid_search=False))

    @pytest.mark.asyncio
    async def test_does_not_query_tantivy(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("hit", 0.8, _TS)]
        tantivy = MagicMock()
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        await pipeline.execute(_make_context(enable_hybrid_search=False))

        tantivy.search.assert_not_called()


@pytest.mark.unit
class TestHybridFusionAndFilter:
    """RRF vs concatenation, thresholds, limits — migrated orchestration tests."""

    @pytest.mark.asyncio
    async def test_rrf_calls_fusion_engine_and_respects_limit(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [
            ("hit1", 0.9, _TS),
            ("hit2", 0.8, _TS),
            ("hit3", 0.7, _TS),
        ]
        tantivy = MagicMock()
        tantivy.search.return_value = [("hit2", 0.95), ("hit4", 0.6)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [
            ("hit2", 0.5),
            ("hit1", 0.4),
            ("hit4", 0.3),
            ("hit3", 0.2),
        ]
        config = _make_config(fusion_ranking_threshold=0.0)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, limit=2, overfetch_limit=10)
        )

        fusion.fuse.assert_called_once_with(
            [("hit1", 0.9), ("hit2", 0.8), ("hit3", 0.7)],
            [("hit2", 0.95), ("hit4", 0.6)],
        )
        assert result.memories == ["hit2", "hit1"]

    @pytest.mark.asyncio
    async def test_concatenation_semantic_first_not_score_sort(self) -> None:
        """Canonical concat keeps semantic hits first, even if Tantivy scores higher."""
        semantic = MagicMock()
        semantic.search.return_value = [("semantic-low", 0.2, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("tantivy-high", 0.99)]
        fusion = MagicMock()
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False, limit=5, overfetch_limit=10)
        )

        fusion.fuse.assert_not_called()
        assert result.memories == ["semantic-low", "tantivy-high"]

    @pytest.mark.asyncio
    async def test_concatenation_deduplicates_and_respects_limit(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("shared", 0.8, _TS), ("s2", 0.7, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("shared", 0.99), ("t2", 0.5)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False, limit=2, overfetch_limit=2)
        )

        assert result.memories == ["shared", "s2"]

    @pytest.mark.asyncio
    async def test_empty_backends_return_empty(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = []
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = []
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == []

    @pytest.mark.asyncio
    async def test_fusion_threshold_keeps_scores_at_or_above(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("keep", 0.9, _TS), ("drop", 0.8, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("keep", 0.5), ("drop", 0.3)]
        config = _make_config(fusion_ranking_threshold=0.5)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == ["keep"]

    @pytest.mark.asyncio
    async def test_fusion_threshold_filters_all_returns_empty(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("low", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("low", 0.01)]
        config = _make_config(fusion_ranking_threshold=0.5)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == []
        assert result.semantic_results == [("low", 0.9, _TS)]

    @pytest.mark.asyncio
    async def test_concatenation_skips_threshold_filter(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("low", 0.01, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        config = _make_config(fusion_ranking_threshold=0.9)
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, config=config)

        result = await pipeline.execute(_make_context(enable_rrf_fusion=False))

        assert result.memories == ["low"]

    def test_filter_stage_empty_and_zero_threshold(self) -> None:
        pipeline = _make_pipeline(semantic=MagicMock())
        assert pipeline._filter_by_fusion_threshold([], 0.5, "q") == []
        kept = pipeline._filter_by_fusion_threshold(
            [("a", 0.0), ("b", 0.01)], 0.0, "q"
        )
        assert kept == [("a", 0.0), ("b", 0.01)]


@pytest.mark.unit
class TestBackendFailureContracts:
    """Hybrid fallback and exception contracts stay unchanged."""

    @pytest.mark.asyncio
    async def test_semantic_search_failure_falls_back_to_tantivy(self) -> None:
        semantic = MagicMock()
        semantic.search.side_effect = RuntimeError("embed fail")
        tantivy = MagicMock()
        tantivy.search.return_value = [("from-tantivy", 0.8)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("from-tantivy", 0.4)]
        config = _make_config(fusion_ranking_threshold=0.0)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == ["from-tantivy"]
        assert result.semantic_results == []
        fusion.fuse.assert_called_once_with([], [("from-tantivy", 0.8)])

    @pytest.mark.asyncio
    async def test_semantic_init_failure_raises_search_error(self) -> None:
        semantic = MagicMock()
        semantic.ensure_initialized.side_effect = RuntimeError("init fail")
        tantivy = MagicMock()
        tantivy.search.return_value = [("from-tantivy", 0.8)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(_make_context(enable_hybrid_search=True))

    @pytest.mark.asyncio
    async def test_tantivy_search_failure_raises_search_error(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("ok", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.side_effect = RuntimeError("tantivy boom")
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(_make_context(enable_hybrid_search=True))

    @pytest.mark.asyncio
    async def test_tantivy_none_returns_empty(self) -> None:
        pipeline = _make_pipeline(semantic=MagicMock(), tantivy=None)
        assert await pipeline._search_tantivy("q", 10, "proj") == []


@pytest.mark.unit
class TestRerankerSettings:
    """Lazy LLM, cross-encoder, none, and skip-on-single-result contracts."""

    @pytest.mark.asyncio
    async def test_none_passthrough(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("a", 0.9, _TS), ("b", 0.8, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("a", 0.4), ("b", 0.3)]
        config = _make_config(fusion_ranking_threshold=0.0, reranker_engine="none")
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="none")
        )

        assert result.memories == ["a", "b"]

    @pytest.mark.asyncio
    async def test_llm_reranker_receives_timestamp_map(self) -> None:
        semantic = MagicMock()
        older = "2024-01-01T00:00:00Z"
        semantic.search.return_value = [("a", 0.9, _TS), ("b", 0.8, older)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("a", 0.4), ("b", 0.3)]
        config = _make_config(fusion_ranking_threshold=0.0, reranker_engine="llm")
        llm = AsyncMock()
        llm.rerank = AsyncMock(return_value=[("b", 0.99), ("a", 0.1)])
        manager = MagicMock()
        manager.llm_reranker = llm
        manager.cross_encoder_reranker = None
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=config,
            memory_manager=manager,
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="llm")
        )

        llm.rerank.assert_awaited_once()
        args = llm.rerank.await_args
        assert args is not None
        assert args.args[0] == "test query"
        assert args.args[1] == [("a", 0.4), ("b", 0.3)]
        assert args.args[2] == {"a": _TS, "b": older}
        assert result.memories == ["b", "a"]

    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_applied(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("a", 0.9, _TS), ("b", 0.8, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("a", 0.4), ("b", 0.3)]
        config = _make_config(
            fusion_ranking_threshold=0.0, reranker_engine="cross_encoder"
        )
        encoder = AsyncMock()
        encoder.rerank_async = AsyncMock(return_value=[("b", 0.95), ("a", 0.2)])
        manager = MagicMock()
        manager.llm_reranker = None
        manager.cross_encoder_reranker = encoder
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=config,
            memory_manager=manager,
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="cross_encoder")
        )

        encoder.rerank_async.assert_awaited_once_with(
            "test query", [("a", 0.4), ("b", 0.3)]
        )
        assert result.memories == ["b", "a"]

    @pytest.mark.asyncio
    async def test_single_result_skips_reranker(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("only", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("only", 0.4)]
        config = _make_config(fusion_ranking_threshold=0.0, reranker_engine="llm")
        llm = AsyncMock()
        llm.rerank = AsyncMock(return_value=[("only", 0.99)])
        manager = MagicMock()
        manager.llm_reranker = llm
        manager.cross_encoder_reranker = None
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=config,
            memory_manager=manager,
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="llm")
        )

        llm.rerank.assert_not_awaited()
        assert result.memories == ["only"]

    @pytest.mark.asyncio
    async def test_empty_after_filter_skips_reranker(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("low", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("low", 0.01)]
        config = _make_config(fusion_ranking_threshold=0.5, reranker_engine="llm")
        llm = AsyncMock()
        manager = MagicMock()
        manager.llm_reranker = llm
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=config,
            memory_manager=manager,
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="llm")
        )

        llm.rerank.assert_not_awaited()
        assert result.memories == []


# ---------------------------------------------------------------------------
# Responsiveness: event-loop ticker + overlapping hybrid backends
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchResponsiveness:
    """Slow sync fakes must not stall the AnyIO/asyncio event loop."""

    @pytest.mark.asyncio
    async def test_semantic_only_ticker_advances(self) -> None:
        entered = threading.Event()
        loop_progressed = threading.Event()
        loop_thread = threading.get_ident()
        worker_threads: list[int] = []
        ticks_after_enter = 0

        backend = ControllableBackend(
            [("mem", 0.9, _TS)],
            entered=entered,
            loop_progressed=loop_progressed,
            thread_ids=worker_threads,
        )
        pipeline = _make_pipeline(semantic=backend)

        async def ticker() -> None:
            nonlocal ticks_after_enter
            while True:
                if entered.is_set():
                    ticks_after_enter += 1
                    loop_progressed.set()
                await asyncio.sleep(0)
                if loop_progressed.is_set() and ticks_after_enter >= 3:
                    break

        ticker_task = asyncio.create_task(ticker())
        search_task = asyncio.create_task(
            pipeline.execute(_make_context(enable_hybrid_search=False))
        )
        result = await search_task
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        assert result.memories == ["mem"]
        assert entered.is_set()
        assert ticks_after_enter >= 1
        assert worker_threads
        assert worker_threads[0] != loop_thread

    @pytest.mark.asyncio
    async def test_hybrid_backends_overlap_and_ticker_advances(self) -> None:
        semantic_entered = threading.Event()
        tantivy_entered = threading.Event()
        loop_progressed = threading.Event()
        barrier = threading.Barrier(2, timeout=_BACKEND_WAIT_TIMEOUT)
        loop_thread = threading.get_ident()
        semantic_threads: list[int] = []
        tantivy_threads: list[int] = []
        ticks_after_both = 0

        semantic = ControllableBackend(
            [("sem", 0.9, _TS)],
            entered=semantic_entered,
            loop_progressed=loop_progressed,
            barrier=barrier,
            thread_ids=semantic_threads,
        )
        tantivy = ControllableBackend(
            [("tan", 0.8)],
            entered=tantivy_entered,
            loop_progressed=loop_progressed,
            barrier=barrier,
            thread_ids=tantivy_threads,
        )
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("sem", 0.4), ("tan", 0.3)]
        config = _make_config(fusion_ranking_threshold=0.0)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        async def ticker() -> None:
            nonlocal ticks_after_both
            while True:
                if semantic_entered.is_set() and tantivy_entered.is_set():
                    ticks_after_both += 1
                    loop_progressed.set()
                await asyncio.sleep(0)
                if loop_progressed.is_set() and ticks_after_both >= 3:
                    break

        ticker_task = asyncio.create_task(ticker())
        result = await pipeline.execute(_make_context(enable_hybrid_search=True))
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        assert result.memories == ["sem", "tan"]
        assert semantic_entered.is_set()
        assert tantivy_entered.is_set()
        assert ticks_after_both >= 1
        assert semantic_threads and tantivy_threads
        assert semantic_threads[0] != loop_thread
        assert tantivy_threads[0] != loop_thread
        assert semantic_threads[0] != tantivy_threads[0]
        assert semantic.search_calls[0][2] == 15
        assert tantivy.search_calls[0][2] == 15


# ---------------------------------------------------------------------------
# Adaptive overfetch remains the canonical policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanonicalOverfetch:
    """Old limit-tier overfetch is gone; adaptive policy is the only one."""

    def test_adaptive_policy_used_not_limit_tiers(self) -> None:
        config = _make_config()
        config.overfetch_adaptive = True
        config.overfetch_max_multiplier = 3.0
        # Competing pipeline used min(limit*3, 50) for limit<=10 → 30 for limit=10.
        # Canonical adaptive uses max multiplier for small indexes: 10*3=30, then
        # the same number happens to match — check a large-index case instead.
        large = calculate_adaptive_overfetch(10, 10000, config)
        assert large == 20  # 10 * min_multiplier 1.5, above MIN_OVERFETCH_LIMIT

    def test_minimum_overfetch_floor(self) -> None:
        config = _make_config()
        config.overfetch_adaptive = False
        config.overfetch_multiplier = 1
        assert calculate_adaptive_overfetch(1, 0, config) >= 20
