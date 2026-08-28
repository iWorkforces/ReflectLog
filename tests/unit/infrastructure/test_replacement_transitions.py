"""Unit tests for durable replacement transitions on MemoryStore."""

import os
import tempfile

import pytest

from reflectlog.core.exceptions import StorageError
from reflectlog.core.types import ReplacementTransition
from reflectlog.infrastructure.memory_store import (
    TRANSITION_COMPLETED,
    TRANSITION_PENDING,
    MemoryStore,
)


def _begin(store: MemoryStore) -> ReplacementTransition:
    return store.begin_replacement_transition(
        old_memory_id=11,
        workspace_id="proj1",
        old_content="Use tabs",
        new_content="Use spaces",
        reason="updated convention",
        confidence=0.95,
    )


@pytest.mark.unit
class TestBeginReplacementTransition:
    """begin_replacement_transition is one SQLite transaction."""

    def test_records_archive_and_pending_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)

            assert transition.status == TRANSITION_PENDING
            assert transition.old_memory_id == 11
            assert transition.old_content == "Use tabs"
            assert transition.new_content == "Use spaces"
            assert transition.reason == "updated convention"
            assert transition.confidence == 0.95
            assert transition.workspace_id == "proj1"

            archives = store.get_archived("proj1")
            assert len(archives) == 1
            assert archives[0].id == transition.archive_id
            assert archives[0].original_id == 11
            assert archives[0].content == "Use tabs"
            assert archives[0].replaced_by == "Use spaces"
            store.close()

    def test_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            first = _begin(store)
            second = _begin(store)

            assert first.id == second.id
            assert first.archive_id == second.archive_id
            assert len(store.get_archived("proj1")) == 1
            assert len(store.list_pending_transitions()) == 1
            store.close()

    def test_dedupes_archive_pairs_before_unique_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP INDEX IF EXISTS idx_archived_original_replaced")
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (1, 'proj1', 'old', 'new', 'r', 0.9), "
                "(1, 'proj1', 'old', 'new', 'r', 0.8)"
            )
            store.connection.commit()
            cursor.close()
            store.close()

            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            archives = store.get_archived("proj1")
            assert len(archives) == 1
            store.close()

    def test_rejects_second_successor_for_same_old_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = _begin(store)
            with pytest.raises(StorageError, match="already has a replacement"):
                _ = store.begin_replacement_transition(
                    old_memory_id=11,
                    workspace_id="proj1",
                    old_content="Use tabs",
                    new_content="Use two spaces",
                    reason="conflict",
                    confidence=0.5,
                )
            assert len(store.list_pending_transitions()) == 1
            assert store.list_pending_transitions()[0].new_content == "Use spaces"
            store.close()

    def test_batch_records_all_or_none(self) -> None:
        from reflectlog.core.types import ReplacementTransitionRequest

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.begin_replacement_transitions(
                [
                    ReplacementTransitionRequest(
                        old_memory_id=1,
                        workspace_id="proj1",
                        old_content="old-a",
                        new_content="new",
                        reason="updated",
                        confidence=0.9,
                    ),
                    ReplacementTransitionRequest(
                        old_memory_id=2,
                        workspace_id="proj1",
                        old_content="old-b",
                        new_content="new",
                        reason="updated",
                        confidence=0.9,
                    ),
                ]
            )
            assert len(store.list_pending_transitions()) == 2
            store.close()

    def test_rolls_back_archive_when_transition_insert_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.connection
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE replacement_transitions")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="replacement transition"):
                _ = _begin(store)

            assert store.get_archived("proj1") == []
            store.close()


@pytest.mark.unit
class TestPendingTransitionLifecycle:
    """list/complete are idempotent and leave an audit row."""

    def test_list_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)

            pending = store.list_pending_transitions()
            assert len(pending) == 1
            assert pending[0].id == transition.id

            store.complete_replacement_transition(transition.id)
            store.complete_replacement_transition(transition.id)

            assert store.list_pending_transitions() == []
            archives = store.get_archived("proj1")
            assert len(archives) == 1
            assert archives[0].replaced_by == "Use spaces"
            store.close()

    def test_complete_keeps_row_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)
            store.complete_replacement_transition(transition.id)

            cursor = store.connection.cursor()
            _ = cursor.execute(
                "SELECT status FROM replacement_transitions WHERE id = ?",
                (transition.id,),
            )
            row = cursor.fetchone()
            cursor.close()
            assert row is not None
            assert row[0] == TRANSITION_COMPLETED
            store.close()

    def test_lookup_by_old_memory_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)
            found = store.get_transition_for_old_memory("proj1", 11)
            assert found is not None
            assert found.id == transition.id
            assert store.get_transition_for_old_memory("proj1", 99) is None
            store.close()

    def test_complete_missing_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.connection
            with pytest.raises(StorageError, match="was not pending"):
                store.complete_replacement_transition(999)
            store.close()
