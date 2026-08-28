"""Integration tests for restart-safe smart replacement.

Each crash-injection case records a durable SQLite transition, stops at a
persistence boundary, then constructs a fresh MemoryManager over the same
files to prove convergence to one active replacement plus an audit row.
"""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
import tempfile
from unittest.mock import patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.add_phases import ReplacementInfo
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.core.exceptions import StorageError
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.tantivy_engine import TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchEngine
from tests.integration.test_memory_manager_usearch import (
    cleanup_manager,
    create_memory_manager,
    create_usearch_config,
)

OLD = "Prefer tabs for indentation in this repository"
NEW = "Prefer spaces for indentation in this repository"

Injector = Callable[[MemoryManager], AbstractContextManager[None]]


def _replacement() -> ReplacementInfo:
    return ReplacementInfo(
        old_memory=OLD,
        new_memory=NEW,
        confidence=0.93,
        reason="updated convention",
        similarity_score=0.88,
    )


def _config(tmpdir: str, hybrid: bool) -> Config:
    return replace(create_usearch_config(tmpdir), enable_hybrid_search=hybrid)


@contextmanager
def _manager(tmpdir: str, *, hybrid: bool = True) -> Generator[MemoryManager]:
    manager, _logger = create_memory_manager(_config(tmpdir, hybrid))
    try:
        yield manager
    finally:
        cleanup_manager(manager)


def _assert_converged(manager: MemoryManager) -> None:
    memories = manager.get_all()
    assert memories == [NEW]
    store = manager._semantic_engine.memory_store
    assert isinstance(store, MemoryStore)
    archives = store.get_archived(manager.project_id)
    assert len(archives) == 1
    assert archives[0].content == OLD
    assert archives[0].replaced_by == NEW
    assert archives[0].reason == "updated convention"
    assert archives[0].confidence == 0.93
    assert store.list_pending_transitions() == []

    tantivy = manager._tantivy_engine
    if tantivy is not None:
        assert tantivy.find_by_exact_match(manager.project_id, NEW)
        assert not tantivy.find_by_exact_match(manager.project_id, OLD)


async def _replace(manager: MemoryManager) -> None:
    result = await manager._storage_phase.execute(
        [NEW],
        {NEW: [_replacement()]},
    )
    assert result.stored_count == 1
    assert result.replaced_count == 1
    assert result.replacements[0].old_memory == OLD


@contextmanager
def _crash_after_transition(_manager: MemoryManager) -> Generator[None]:
    def boom(self: USearchEngine, memory_id: str) -> None:
        raise RuntimeError("crash after transition")

    with patch.object(USearchEngine, "delete", boom):
        yield


@contextmanager
def _crash_after_semantic_delete(_manager: MemoryManager) -> Generator[None]:
    original = USearchEngine.delete

    def boom(self: USearchEngine, memory_id: str) -> None:
        original(self, memory_id)
        raise RuntimeError("crash after semantic delete")

    with patch.object(USearchEngine, "delete", boom):
        yield


@contextmanager
def _crash_after_tantivy_delete(_manager: MemoryManager) -> Generator[None]:
    original = TantivyEngine.delete

    def boom(self: TantivyEngine, project_id: str, content: str) -> bool:
        deleted = original(self, project_id, content)
        raise RuntimeError("crash after tantivy delete")
        return deleted

    with patch.object(TantivyEngine, "delete", boom):
        yield


@contextmanager
def _crash_after_insert(_manager: MemoryManager) -> Generator[None]:
    original = USearchEngine.add_batch

    def boom(
        self: USearchEngine, project_id: str, contents: list[str], infer: bool
    ) -> list[str]:
        stored = original(self, project_id, contents, infer)
        raise RuntimeError("crash after insert")
        return stored

    with patch.object(USearchEngine, "add_batch", boom):
        yield


@contextmanager
def _crash_after_commit(manager: MemoryManager) -> Generator[None]:
    original_semantic = USearchEngine.commit
    original_tantivy = TantivyEngine.commit
    hybrid = manager._tantivy_engine is not None
    committed = {"semantic": False, "tantivy": not hybrid}

    def semantic_boom(self: USearchEngine) -> None:
        original_semantic(self)
        committed["semantic"] = True
        if committed["tantivy"]:
            raise RuntimeError("crash after commit")

    def tantivy_boom(self: TantivyEngine) -> None:
        original_tantivy(self)
        committed["tantivy"] = True
        if committed["semantic"]:
            raise RuntimeError("crash after commit")

    with patch.object(USearchEngine, "commit", semantic_boom):
        if not hybrid:
            yield
        else:
            with patch.object(TantivyEngine, "commit", tantivy_boom):
                yield


@contextmanager
def _crash_after_complete(_manager: MemoryManager) -> Generator[None]:
    original = MemoryStore.complete_replacement_transition

    def boom(self: MemoryStore, transition_id: int) -> None:
        original(self, transition_id)
        raise RuntimeError("crash after complete")

    with patch.object(MemoryStore, "complete_replacement_transition", boom):
        yield


async def _crash_and_reopen(
    tmpdir: str,
    inject: Injector,
    *,
    hybrid: bool = True,
) -> None:
    config = _config(tmpdir, hybrid)
    first, _ = create_memory_manager(config)
    try:
        assert first.add_memories([OLD]) == 1
        with inject(first):
            with pytest.raises((StorageError, RuntimeError)):
                _ = await first._storage_phase.execute(
                    [NEW],
                    {NEW: [_replacement()]},
                )
    finally:
        first.close()

    second, _ = create_memory_manager(config)
    try:
        _assert_converged(second)
        _assert_converged(second)
    finally:
        cleanup_manager(second)


@pytest.mark.integration
class TestReplacementRecoveryIntegration:
    """Crash-injection + reopen proves restart-safe replacement."""

    async def test_normal_replacement_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _manager(tmpdir) as manager:
                assert manager.add_memories([OLD]) == 1
                dry = await manager._storage_phase.execute(
                    [NEW],
                    {NEW: [_replacement()]},
                    dry_run=True,
                )
                assert dry.stored_count == 1
                assert dry.replaced_count == 1
                assert manager.get_all() == [OLD]
                store = manager._semantic_engine.memory_store
                assert isinstance(store, MemoryStore)
                assert store.get_archived(manager.project_id) == []

                await _replace(manager)
                _assert_converged(manager)

    async def test_crash_after_transition_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_transition)

    async def test_crash_after_semantic_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_semantic_delete)

    async def test_crash_after_tantivy_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_tantivy_delete)

    async def test_crash_after_new_memory_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_insert)

    async def test_crash_after_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_commit)

    async def test_crash_after_complete_then_repeat_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_complete)

    async def test_disabled_hybrid_crash_after_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_transition, hybrid=False)

    async def test_disabled_hybrid_normal_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _manager(tmpdir, hybrid=False) as manager:
                assert manager._tantivy_engine is None
                assert manager.add_memories([OLD]) == 1
                await _replace(manager)
                _assert_converged(manager)
