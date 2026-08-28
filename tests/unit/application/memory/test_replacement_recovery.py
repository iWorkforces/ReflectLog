"""Unit tests for restart reconciliation of unfinished replacements."""

import os
import tempfile
import threading
from unittest.mock import MagicMock

import pytest

from reflectlog.application.memory.replacement_recovery import (
    apply_pending_transition,
    reconcile_pending_replacements,
    replacement_converged,
)
from reflectlog.core.types import ReplacementTransition
from reflectlog.infrastructure.memory_store import MemoryStore


def _transition() -> ReplacementTransition:
    return ReplacementTransition(
        id=3,
        workspace_id="proj",
        old_memory_id=11,
        old_content="old convention",
        new_content="new convention",
        archive_id=8,
        reason="updated",
        confidence=0.9,
        status="pending",
    )


@pytest.mark.unit
class TestApplyPendingTransition:
    """apply_pending_transition is idempotent per backend."""

    def test_deletes_old_and_inserts_missing_new(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.side_effect = [None, None, 99, None]
        semantic.index = {99}
        tantivy = MagicMock()
        seen_new = {"yes": False}

        def find(_pid: str, content: str) -> list[str]:
            if content == "new convention" and seen_new["yes"]:
                return [content]
            return []

        def add(_pid: str, content: str) -> None:
            if content == "new convention":
                seen_new["yes"] = True

        tantivy.find_by_exact_match.side_effect = find
        tantivy.add.side_effect = add
        logger = MagicMock()

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=logger,
        )

        semantic.delete.assert_called_once_with(memory_id="11")
        tantivy.delete.assert_called_once_with("proj", "old convention")
        semantic.add.assert_called_once_with(
            workspace_id="proj",
            content="new convention",
            infer=False,
        )
        tantivy.add.assert_called_once_with("proj", "new convention")
        tantivy.commit.assert_called_once()
        semantic.commit.assert_called_once()
        semantic.memory_store.complete_replacement_transition.assert_called_once_with(3)

    def test_skips_insert_when_replacement_already_present(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.side_effect = [None, 99, 99, None]
        semantic.index = {99}
        tantivy = MagicMock()
        tantivy.find_by_exact_match.side_effect = (
            lambda _pid, content: [content] if content == "new convention" else []
        )

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=MagicMock(),
        )

        semantic.add.assert_not_called()
        tantivy.add.assert_not_called()
        semantic.memory_store.complete_replacement_transition.assert_called_once_with(3)

    def test_reindexes_missing_vector(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.side_effect = lambda _ws, content: (
            7 if content == "new convention" else None
        )
        semantic.index = set()

        def add_and_index(**_kwargs: object) -> None:
            semantic.index.add(7)

        semantic.add.side_effect = add_and_index

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=None,
            logger=MagicMock(),
        )

        semantic.delete.assert_any_call(memory_id="11")
        semantic.delete.assert_any_call(memory_id="7")
        semantic.add.assert_called_once_with(
            workspace_id="proj",
            content="new convention",
            infer=False,
        )
        semantic.memory_store.complete_replacement_transition.assert_called_once()

    def test_skips_tantivy_delete_when_old_text_was_readded(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.side_effect = [22, 99, 99, 22]
        semantic.index = {99}
        tantivy = MagicMock()
        tantivy.find_by_exact_match.side_effect = (
            lambda _pid, content: [content] if content == "new convention" else []
        )

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=MagicMock(),
        )

        tantivy.delete.assert_not_called()
        semantic.memory_store.complete_replacement_transition.assert_called_once()

    def test_leaves_pending_when_tantivy_still_has_old(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.side_effect = [None, 99, 99, None]
        semantic.index = {99}
        tantivy = MagicMock()
        tantivy.find_by_exact_match.side_effect = lambda _pid, content: [content]

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=MagicMock(),
        )

        semantic.memory_store.complete_replacement_transition.assert_not_called()

    def test_does_not_complete_when_indexes_disagree(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.return_value = None
        semantic.index = set()
        tantivy = MagicMock()
        tantivy.find_by_exact_match.return_value = []
        semantic.add.side_effect = RuntimeError("embedder down")

        with pytest.raises(RuntimeError, match="embedder down"):
            _ = apply_pending_transition(
                _transition(),
                semantic_engine=semantic,
                tantivy_engine=tantivy,
                logger=MagicMock(),
            )
        semantic.memory_store.complete_replacement_transition.assert_not_called()

    def test_works_without_tantivy(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.side_effect = [None, 7, None]
        semantic.index = {7}

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=None,
            logger=MagicMock(),
        )

        semantic.add.assert_called_once()
        semantic.commit.assert_called_once()
        semantic.memory_store.complete_replacement_transition.assert_called_once()

    def test_converged_requires_old_id_gone(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.side_effect = lambda _ws, content: (
            11 if content == "old convention" else 99
        )
        semantic.index = {99}
        assert (
            replacement_converged(
                _transition(), semantic_engine=semantic, tantivy_engine=None
            )
            is False
        )

    def test_converged_false_when_index_missing(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.return_value = 99
        semantic.index = None
        assert (
            replacement_converged(
                _transition(), semantic_engine=semantic, tantivy_engine=None
            )
            is False
        )


@pytest.mark.unit
class TestReconcilePendingReplacements:
    """Startup reconciliation respects lock order and skips empty stores."""

    def test_noops_for_mock_store(self) -> None:
        semantic = MagicMock()
        semantic.memory_store = MagicMock()
        count = reconcile_pending_replacements(
            semantic_engine=semantic,
            tantivy_engine=None,
            write_lock=threading.Lock(),
            lock=threading.RLock(),
            logger=MagicMock(),
        )
        assert count == 0

    def test_noops_when_nothing_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.connection
            semantic = MagicMock()
            semantic.memory_store = store
            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )
            assert count == 0
            store.close()

    def test_acquires_write_lock_before_lock(self) -> None:
        order: list[str] = []

        class OrderLock:
            def __init__(self, name: str, inner: threading.Lock | threading.RLock):
                self.name = name
                self.inner = inner

            def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
                order.append(self.name)
                return self.inner.acquire(blocking, timeout)

            def release(self) -> None:
                self.inner.release()

            def __enter__(self) -> OrderLock:
                _ = self.acquire()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: object,
            ) -> None:
                self.release()

        write_lock = OrderLock("write", threading.Lock())
        inner_lock = OrderLock("inner", threading.RLock())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            planted = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="old convention",
                new_content="new convention",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.get_id_by_content.side_effect = [None, 99, None]
            semantic.index = {99}

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=write_lock,
                lock=inner_lock,
                logger=MagicMock(),
            )

            assert planted.id == 1
            assert count == 1
            assert order == ["write", "inner"]
            semantic.delete.assert_called_once_with(memory_id="11")
            assert store.list_pending_transitions() == []
            store.close()

    def test_repeated_reconcile_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="old convention",
                new_content="new convention",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.get_id_by_content.return_value = 99
            semantic.index = {99: object()}

            kwargs = {
                "semantic_engine": semantic,
                "tantivy_engine": None,
                "write_lock": threading.Lock(),
                "lock": threading.RLock(),
                "logger": MagicMock(),
            }
            first = reconcile_pending_replacements(**kwargs)
            second = reconcile_pending_replacements(**kwargs)

            assert first == 1
            assert second == 0
            assert store.list_pending_transitions() == []
            assert len(store.get_archived("proj")) == 1
            store.close()

    def test_count_skips_transitions_that_stay_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="old convention",
                new_content="new convention",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.get_id_by_content.side_effect = lambda _ws, content: (
                99 if content == "new convention" else None
            )
            semantic.index = {99}
            tantivy = MagicMock()
            tantivy.find_by_exact_match.side_effect = (
                lambda _ws, content: [content]
            )

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=tantivy,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )

            assert count == 0
            assert len(store.list_pending_transitions()) == 1
            store.close()
