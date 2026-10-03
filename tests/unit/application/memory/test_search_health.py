"""Failure accounting contracts and real search-pipeline instrumentation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
import threading
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import anyio
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.fusion.base import FusionEngine
from reflectlog.application.memory.search_health import SearchFailureRecorder
from reflectlog.application.memory.search_strategies import (
    RerankerProvider,
    SearchContext,
    SearchPipeline,
)
from reflectlog.core.enums import RerankerEngine, SearchComponent
from reflectlog.core.exceptions import InitializationError, SearchError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.search_health import (
    ComponentFailureSnapshot,
    SearchFailureSnapshot,
)
from reflectlog.core.types import ISemanticSearchEngine
from reflectlog.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from reflectlog.infrastructure.openrouter_reranker import OpenRouterReranker
from reflectlog.infrastructure.tantivy_engine import TantivyEngine


def test_recorder_zero_state() -> None:
    recorder = SearchFailureRecorder()

    snapshot = recorder.snapshot()

    assert snapshot == SearchFailureSnapshot()
    assert snapshot.to_dict() == {
        component.value: {"count": 0, "last_failure_at": None, "exception_type": None}
        for component in SearchComponent
    }


def test_record_retains_only_type_and_utc_timestamp() -> None:
    recorder = SearchFailureRecorder()
    secret = "secret-query-memory-exception-args"
    error = RuntimeError(secret, {"memory": secret})

    recorder.record(SearchComponent.SEMANTIC, error)

    snapshot = recorder.snapshot()
    assert snapshot.semantic.count == 1
    assert snapshot.semantic.exception_type == "RuntimeError"
    assert snapshot.semantic.last_failure_at is not None
    assert datetime.fromisoformat(snapshot.semantic.last_failure_at).tzinfo == UTC
    assert secret not in repr(snapshot)
    assert secret not in repr(snapshot.to_dict())
    assert error not in vars(recorder).values()


def test_snapshot_is_detached_and_immutable() -> None:
    recorder = SearchFailureRecorder()
    before = recorder.snapshot()

    recorder.record(SearchComponent.TANTIVY, ValueError("private"))

    assert before.tantivy == ComponentFailureSnapshot()
    assert recorder.snapshot().tantivy.count == 1
    with pytest.raises(FrozenInstanceError):
        before.__setattr__("tantivy", ComponentFailureSnapshot(count=42))
    with pytest.raises(FrozenInstanceError):
        before.semantic.__setattr__("count", 42)
    payload = recorder.snapshot().to_dict()
    payload["tantivy"]["count"] = 99
    assert recorder.snapshot().tantivy.count == 1


@pytest.mark.parametrize("component", list(SearchComponent))
def test_components_are_independent(component: SearchComponent) -> None:
    recorder = SearchFailureRecorder()

    recorder.record(component, KeyboardInterrupt("private"))
    recorder.record(component, LookupError("private"))

    for name, state in recorder.snapshot().to_dict().items():
        assert state["count"] == (2 if name == component.value else 0)
        assert state["exception_type"] == (
            "LookupError" if name == component.value else None
        )


def test_concurrent_records_and_snapshots_have_exact_counts() -> None:
    recorder = SearchFailureRecorder()
    writers, repetitions = 8, 1000
    barrier = threading.Barrier(writers + 1)

    def write() -> None:
        barrier.wait(timeout=10)
        for _ in range(repetitions):
            recorder.record(SearchComponent.SEMANTIC, RuntimeError("private"))
            barrier.wait(timeout=10)

    def read() -> None:
        barrier.wait(timeout=10)
        previous = 0
        for _ in range(repetitions):
            state = recorder.snapshot().semantic
            assert previous <= state.count <= writers * repetitions
            assert (state.last_failure_at is None) == (state.count == 0)
            assert (state.exception_type is None) == (state.count == 0)
            previous = state.count
            barrier.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=writers + 1) as executor:
        futures = [executor.submit(write) for _ in range(writers)]
        futures.append(executor.submit(read))
        for future in futures:
            future.result(timeout=10)

    assert recorder.snapshot().semantic.count == writers * repetitions


def test_recorder_uses_own_lock_without_external_lock_dependency() -> None:
    recorder = SearchFailureRecorder()
    write_lock, manager_lock, lease_lock = (threading.Lock() for _ in range(3))

    with (
        write_lock,
        manager_lock,
        lease_lock,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        executor.submit(
            recorder.record, SearchComponent.TANTIVY, RuntimeError()
        ).result(timeout=2)
        snapshot = executor.submit(recorder.snapshot).result(timeout=2)

    assert isinstance(recorder._lock, type(threading.Lock()))
    assert snapshot.tantivy.count == 1


def _pipeline(
    recorder: SearchFailureRecorder | None,
    reranker: RerankerEngine = RerankerEngine.NONE,
) -> tuple[SearchPipeline, MagicMock, MagicMock, MagicMock, SearchContext]:
    semantic = MagicMock(spec=ISemanticSearchEngine)
    semantic.is_ready.return_value = True
    semantic.search.return_value = [("memory-private", 0.9, "2026-01-01T00:00:00Z")]
    semantic.memory_store.exists_many.side_effect = lambda _workspace, contents: set(
        contents
    )
    tantivy = MagicMock(spec=TantivyEngine)
    tantivy.is_ready.return_value = True
    tantivy.search.return_value = [("memory-private", 0.8)]
    config = MagicMock(spec=Config)
    config.search_score_threshold = 0.0
    config.log_search_results_verbose = False
    config.reranker_engine = reranker
    manager = MagicMock(spec=RerankerProvider)
    manager.cross_encoder_reranker = None
    manager.openrouter_reranker = None
    args = (
        semantic,
        tantivy,
        MagicMock(spec=FusionEngine),
        config,
        MagicMock(spec=IStructuredLogger),
        manager,
    )
    pipeline = (
        SearchPipeline(*args)
        if recorder is None
        else SearchPipeline(*args, failure_recorder=recorder)
    )
    context = SearchContext("query-private", 5, 15, False, reranker, "workspace")
    return pipeline, semantic, tantivy, manager, context


@pytest.mark.parametrize("component", list(SearchComponent))
async def test_pipeline_records_each_swallowed_failure_once(
    component: SearchComponent,
) -> None:
    recorder = SearchFailureRecorder()
    reranker = RerankerEngine.NONE
    if component in (
        SearchComponent.CROSS_ENCODER,
        SearchComponent.OPENROUTER_RERANKER,
    ):
        reranker = RerankerEngine(component.value.replace("_reranker", ""))
    pipeline, semantic, tantivy, manager, context = _pipeline(recorder, reranker)
    error = RuntimeError("exception-private")
    match component:
        case SearchComponent.SEMANTIC:
            semantic.search.side_effect = error
        case SearchComponent.TANTIVY:
            tantivy.search.side_effect = error
        case SearchComponent.CROSS_ENCODER:
            semantic.search.return_value.append(
                ("second-private", 0.8, "2026-01-01T00:00:00Z")
            )
            manager.cross_encoder_reranker = MagicMock(
                spec=CrossEncoderReranker, rerank_async=AsyncMock(side_effect=error)
            )
        case SearchComponent.OPENROUTER_RERANKER:
            semantic.search.return_value.append(
                ("second-private", 0.8, "2026-01-01T00:00:00Z")
            )
            manager.openrouter_reranker = MagicMock(
                spec=OpenRouterReranker, rerank_async=AsyncMock(side_effect=error)
            )

    result = await pipeline.execute(context)

    expected = ["memory-private"]
    if component in (
        SearchComponent.CROSS_ENCODER,
        SearchComponent.OPENROUTER_RERANKER,
    ):
        expected.append("second-private")
    assert result.memories == expected
    for name, state in recorder.snapshot().to_dict().items():
        assert state["count"] == (1 if name == component.value else 0)
        assert state["exception_type"] == (
            "RuntimeError" if name == component.value else None
        )
        assert (state["last_failure_at"] is not None) == (name == component.value)
    for secret in (
        context.query,
        "memory-private",
        "second-private",
        "exception-private",
    ):
        assert secret not in repr(recorder.snapshot())


@pytest.mark.parametrize(
    "reranker", [RerankerEngine.CROSS_ENCODER, RerankerEngine.OPENROUTER]
)
@pytest.mark.parametrize("hits", [0, 1, 2])
async def test_absent_and_skipped_rerankers_do_not_count(
    reranker: RerankerEngine, hits: int
) -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, tantivy, _manager, context = _pipeline(recorder, reranker)
    semantic.search.return_value = [
        (f"memory-{i}", 0.9, "2026-01-01T00:00:00Z") for i in range(hits)
    ]
    tantivy.search.return_value = []

    result = await pipeline.execute(context)

    assert len(result.memories) == hits
    assert recorder.snapshot() == SearchFailureSnapshot()


async def test_disabled_tantivy_does_not_count() -> None:
    recorder = SearchFailureRecorder()
    pipeline, _semantic, _tantivy, _manager, context = _pipeline(recorder)
    pipeline._tantivy_engine = None

    result = await pipeline.execute(context)

    assert result.memories == ["memory-private"]
    assert recorder.snapshot() == SearchFailureSnapshot()


@pytest.mark.parametrize(
    "reranker", [RerankerEngine.CROSS_ENCODER, RerankerEngine.OPENROUTER]
)
@pytest.mark.parametrize("hits", [0, 1])
async def test_configured_failing_rerankers_are_skipped_for_small_results(
    reranker: RerankerEngine, hits: int
) -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, tantivy, manager, context = _pipeline(recorder, reranker)
    semantic.search.return_value = [
        (f"memory-{i}", 0.9, "2026-01-01T00:00:00Z") for i in range(hits)
    ]
    tantivy.search.return_value = []
    fail = AsyncMock(side_effect=RuntimeError("private"))
    manager.cross_encoder_reranker = MagicMock(
        spec=CrossEncoderReranker, rerank_async=fail
    )
    manager.openrouter_reranker = MagicMock(spec=OpenRouterReranker, rerank_async=fail)

    result = await pipeline.execute(context)

    assert len(result.memories) == hits
    fail.assert_not_awaited()
    assert recorder.snapshot() == SearchFailureSnapshot()


async def test_openrouter_without_manager_does_not_count() -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, _tantivy, _manager, context = _pipeline(
        recorder, RerankerEngine.OPENROUTER
    )
    semantic.search.return_value.append(("second-private", 0.8, "2026-01-01T00:00:00Z"))
    pipeline._memory_manager = None

    result = await pipeline.execute(context)

    assert result.memories == ["memory-private", "second-private"]
    assert recorder.snapshot() == SearchFailureSnapshot()


async def test_dual_backend_failure_records_both_and_preserves_error() -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, tantivy, _manager, context = _pipeline(recorder)
    semantic_error = RuntimeError("semantic-private")
    semantic.search.side_effect = semantic_error
    tantivy.search.side_effect = ValueError("tantivy-private")

    with pytest.raises(
        SearchError, match="Failed to execute search: semantic-private"
    ) as raised:
        await pipeline.execute(context)

    assert raised.value.__cause__ is semantic_error
    assert recorder.snapshot().semantic.count == 1
    assert recorder.snapshot().tantivy.count == 1
    assert recorder.snapshot().cross_encoder.count == 0
    assert recorder.snapshot().openrouter_reranker.count == 0


async def test_concurrent_pipeline_searches_record_exact_total() -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, _tantivy, _manager, context = _pipeline(recorder)
    thread_ids: set[int] = set()
    thread_ids_lock = threading.Lock()

    def fail_search(**_kwargs: str | int) -> None:
        with thread_ids_lock:
            thread_ids.add(threading.get_ident())
        raise RuntimeError("private")

    semantic.search.side_effect = fail_search

    async def search() -> None:
        assert (await pipeline.execute(context)).memories == ["memory-private"]

    async with anyio.create_task_group() as group:
        for _ in range(100):
            group.start_soon(search)

    assert recorder.snapshot().semantic.count == 100
    assert len(thread_ids) > 1


@pytest.mark.parametrize(
    "failing_backend", [None, SearchComponent.SEMANTIC, SearchComponent.TANTIVY]
)
async def test_old_six_argument_constructor_preserves_results(
    failing_backend: SearchComponent | None,
) -> None:
    old, old_semantic, old_tantivy, _old_manager, context = _pipeline(None)
    recorder = SearchFailureRecorder()
    instrumented, semantic, tantivy, _new_manager, _context = _pipeline(recorder)
    match failing_backend:
        case SearchComponent.SEMANTIC:
            old_semantic.search.side_effect = semantic.search.side_effect = (
                RuntimeError("private")
            )
        case SearchComponent.TANTIVY:
            old_tantivy.search.side_effect = tantivy.search.side_effect = RuntimeError(
                "private"
            )
        case None:
            pass
        case _:
            pytest.fail("unsupported backend fixture")

    assert await old.execute(context) == await instrumented.execute(context)
    snapshot = recorder.snapshot()
    assert snapshot.semantic.count == (
        1 if failing_backend == SearchComponent.SEMANTIC else 0
    )
    assert snapshot.tantivy.count == (
        1 if failing_backend == SearchComponent.TANTIVY else 0
    )
    assert snapshot.cross_encoder.count == snapshot.openrouter_reranker.count == 0


async def test_cross_encoder_getter_failure_is_not_swallowed_or_counted() -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, _tantivy, manager, context = _pipeline(
        recorder, RerankerEngine.CROSS_ENCODER
    )
    semantic.search.return_value.append(("second-private", 0.8, "2026-01-01T00:00:00Z"))
    type(manager).cross_encoder_reranker = PropertyMock(
        side_effect=RuntimeError("getter-private")
    )

    with pytest.raises(SearchError, match="getter-private"):
        await pipeline.execute(context)

    assert recorder.snapshot() == SearchFailureSnapshot()


REFUSAL = "disagree. Restore a consistent backup or rebuild this workspace offline."


async def test_refused_semantic_index_propagates_even_with_live_full_text_hits() -> (
    None
):
    recorder = SearchFailureRecorder()
    pipeline, semantic, tantivy, _manager, context = _pipeline(recorder)
    refusal = InitializationError(REFUSAL)
    semantic.search.side_effect = refusal
    assert tantivy.search.return_value == [("memory-private", 0.8)]

    with pytest.raises(InitializationError) as raised:
        await pipeline.execute(context)

    assert raised.value is refusal
    assert str(raised.value) == REFUSAL
    assert recorder.snapshot().semantic.count == 1
    assert recorder.snapshot().semantic.exception_type == "InitializationError"
    assert recorder.snapshot().tantivy.count == 0


async def test_refused_semantic_index_is_raised_when_it_fails_to_open_lazily() -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, _tantivy, _manager, context = _pipeline(recorder)
    semantic.is_ready.return_value = False
    semantic.ensure_initialized.side_effect = InitializationError(REFUSAL)

    with pytest.raises(InitializationError, match="Restore a consistent backup"):
        await pipeline.execute(context)

    semantic.search.assert_not_called()
    assert recorder.snapshot().semantic.count == 1


async def test_other_semantic_errors_still_fall_back_to_full_text() -> None:
    recorder = SearchFailureRecorder()
    pipeline, semantic, _tantivy, _manager, context = _pipeline(recorder)
    semantic.search.side_effect = RuntimeError("provider down")

    result = await pipeline.execute(context)

    assert result.memories == ["memory-private"]
    assert recorder.snapshot().semantic.count == 1
    assert recorder.snapshot().semantic.exception_type == "RuntimeError"


async def test_refused_index_with_a_full_text_outage_still_reports_the_refusal() -> (
    None
):
    pipeline, semantic, tantivy, _manager, context = _pipeline(None)
    semantic.search.side_effect = InitializationError(REFUSAL)
    tantivy.search.side_effect = RuntimeError("tantivy down")

    with pytest.raises(InitializationError, match="Restore a consistent backup"):
        await pipeline.execute(context)
