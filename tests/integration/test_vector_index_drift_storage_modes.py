"""A refused drifted workspace is left byte-identical in every SQLite storage mode.

``test_vector_index_drift.py`` uses WAL-mode databases created by the current
schema, where ``PRAGMA journal_mode=WAL`` and ``CREATE TABLE IF NOT EXISTS``
are no-ops. An older or converted backup differs: its database is in
rollback-journal (``DELETE``) mode and/or predates the journal table (or its
``kind`` column). Opening such a file through the writable ``MemoryStore``
switches it to WAL and migrates the schema, so any step that opens that store
before the vector check refuses would change the bytes of a workspace the
contract says is not modified.

Every test here uses a real ``USearchEngine``, ``MemoryStore`` and Tantivy
index through real ``MemoryManager`` objects. After each step it compares the
sha256 and mtime of the database and the vector file, the database journal
mode (read through a read-only connection), and the absence of ``-wal``,
``-shm`` and ``-journal`` files for rollback-journal databases.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import sqlite3

import pytest

from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.core.exceptions import InitializationError
from reflectlog.infrastructure.usearch_engine import USearchEngine
from tests.integration.test_memory_manager_usearch import create_memory_manager
from tests.integration.test_vector_index_drift import (
    CONTENTS,
    Drift,
    FileState,
    Operation,
    Workspace,
    _assert_no_memory_text_logged,
    _assert_refusal_text,
    _build,
    _checkpoint,
    _close,
    _construct_expecting_refusal,
    _drifted,
    _drop_vector,
    _file_state,
    _invoke,
    _journal,
    _lazy,
    _new_logger,
    _open,
    _sqlite_ids,
    read_memory_store_snapshot,
)

pytestmark = pytest.mark.integration

LEGACY_TRANSITIONS_WITHOUT_KIND = """
    CREATE TABLE replacement_transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        old_memory_id INTEGER NOT NULL,
        old_content TEXT NOT NULL,
        new_content TEXT NOT NULL,
        archive_id INTEGER NOT NULL,
        reason TEXT NOT NULL,
        confidence REAL NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

AUXILIARY_FILES = ("-wal", "-shm", "-journal")


class Storage(StrEnum):
    """How the SQLite file of a drifted workspace was written."""

    DELETE_MODE = "delete_mode"
    NO_TRANSITIONS_TABLE = "no_transitions_table"
    TRANSITIONS_WITHOUT_KIND = "transitions_without_kind"
    TRANSITIONS_WITHOUT_KIND_WAL = "transitions_without_kind_wal"

    @property
    def journal_mode(self) -> str:
        return "wal" if self is Storage.TRANSITIONS_WITHOUT_KIND_WAL else "delete"


def _rewrite_database(workspace: Workspace, storage: Storage) -> None:
    """Rewrite the database the way an older or converted backup would look."""
    connection = sqlite3.connect(workspace.db_path)
    try:
        if storage is not Storage.DELETE_MODE:
            _ = connection.execute("DROP TABLE IF EXISTS replacement_transitions")
        if storage is Storage.NO_TRANSITIONS_TABLE:
            _ = connection.execute("DROP TABLE IF EXISTS archived_memories")
        if storage in {
            Storage.TRANSITIONS_WITHOUT_KIND,
            Storage.TRANSITIONS_WITHOUT_KIND_WAL,
        }:
            _ = connection.execute(LEGACY_TRANSITIONS_WITHOUT_KIND)
        connection.commit()
    finally:
        connection.close()
    if storage.journal_mode == "delete":
        _convert_to_rollback_journal(workspace)
    else:
        _checkpoint(workspace.db_path)


def _convert_to_rollback_journal(workspace: Workspace) -> None:
    """Switch a closed database to ``DELETE`` mode and drop the WAL leftovers.

    SQLite on some platforms leaves an empty ``-shm`` behind after the
    conversion; with no connection open it carries no data, so the fixture
    removes it and the tests can then assert that nothing recreates it.
    """
    _checkpoint(workspace.db_path)
    connection = sqlite3.connect(workspace.db_path)
    try:
        row = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
    finally:
        connection.close()
    assert row is not None
    assert str(row[0]).lower() == "delete"
    Path(f"{workspace.db_path}-shm").unlink(missing_ok=True)
    assert _auxiliary_files(workspace) == []


def _stored_journal_mode(workspace: Workspace) -> str:
    uri = f"{workspace.db_path.absolute().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        row = connection.execute("PRAGMA journal_mode").fetchone()
    finally:
        connection.close()
    assert row is not None
    return str(row[0]).lower()


def _auxiliary_files(workspace: Workspace) -> list[str]:
    return [
        suffix
        for suffix in AUXILIARY_FILES
        if Path(f"{workspace.db_path}{suffix}").exists()
    ]


@dataclass(frozen=True)
class Fixture:
    workspace: Workspace
    storage: Storage
    before: FileState

    def assert_untouched(self, step: str) -> None:
        """Bytes, mtimes, journal mode and sidecar files are exactly as found."""
        assert _file_state(self.workspace) == self.before, step
        assert _stored_journal_mode(self.workspace) == self.storage.journal_mode, step
        if self.storage.journal_mode == "delete":
            assert _auxiliary_files(self.workspace) == [], step


def _fixture(tmp_path: Path, drift: Drift, storage: Storage) -> Fixture:
    workspace = _drifted(tmp_path, drift)
    _rewrite_database(workspace, storage)
    assert _stored_journal_mode(workspace) == storage.journal_mode
    if storage.journal_mode == "delete":
        assert _auxiliary_files(workspace) == []
    # The reader must still see drift, with no pending rows in the journal.
    snapshot = read_memory_store_snapshot(
        str(workspace.db_path), workspace.workspace_id
    )
    assert not snapshot.pending_transitions
    assert snapshot.unrecognized_pending_count == 0
    fixture = Fixture(workspace, storage, _file_state(workspace))
    fixture.assert_untouched("fixture")
    return fixture


STORAGES = list(Storage)
DRIFTS = list(Drift)


class TestEagerConstructionRefusesWithoutWriting:
    @pytest.mark.parametrize("storage", STORAGES)
    @pytest.mark.parametrize("drift", DRIFTS)
    def test_construction_refuses_and_leaves_the_files_alone(
        self, tmp_path: Path, drift: Drift, storage: Storage
    ) -> None:
        fixture = _fixture(tmp_path, drift, storage)
        logger = _new_logger()

        error = _construct_expecting_refusal(fixture.workspace, logger)

        _assert_refusal_text(str(error), fixture.workspace)
        fixture.assert_untouched("eager construction")
        _assert_no_memory_text_logged(logger)

    @pytest.mark.parametrize("storage", STORAGES)
    @pytest.mark.parametrize("drift", [Drift.MISSING_VECTOR, Drift.SWAPPED_IDS])
    async def test_registry_acquire_refuses_and_leaves_the_files_alone(
        self, tmp_path: Path, drift: Drift, storage: Storage
    ) -> None:
        fixture = _fixture(tmp_path, drift, storage)
        registry = WorkspaceRegistry(
            fixture.workspace.config,
            manager_factory=lambda config: create_memory_manager(config)[0],
        )
        try:
            with pytest.raises(InitializationError) as caught:
                async with registry.acquire(fixture.workspace.workspace_id):
                    pass
        finally:
            await registry.close()

        _assert_refusal_text(str(caught.value), fixture.workspace)
        fixture.assert_untouched("registry acquire")


class TestLazyWorkspaceRefusesEveryStepWithoutWriting:
    @pytest.mark.parametrize("storage", STORAGES)
    @pytest.mark.parametrize("drift", DRIFTS)
    async def test_every_operation_refuses_and_nothing_changes(
        self, tmp_path: Path, drift: Drift, storage: Storage
    ) -> None:
        fixture = _fixture(tmp_path, drift, storage)
        logger = _new_logger()
        manager = _open(_lazy(fixture.workspace.config), logger)
        try:
            fixture.assert_untouched("lazy construction")
            for operation in (
                Operation.SEARCH,
                Operation.ADD,
                Operation.ADD_ASYNC,
                Operation.REMOVE,
                # A retry refuses again: the failed open leaves nothing behind.
                Operation.SEARCH,
            ):
                with pytest.raises(InitializationError) as caught:
                    _ = await _invoke(manager, fixture.workspace, operation)
                _assert_refusal_text(str(caught.value), fixture.workspace)
                fixture.assert_untouched(operation.value)
        finally:
            _close(manager)
        fixture.assert_untouched("close")
        _assert_no_memory_text_logged(logger)

    @pytest.mark.parametrize("storage", STORAGES)
    def test_sqlite_only_reads_still_work_on_the_refused_workspace(
        self, tmp_path: Path, storage: Storage
    ) -> None:
        """``get_all`` and ``count`` read SQLite and are not refusals."""
        fixture = _fixture(tmp_path, Drift.MISSING_VECTOR, storage)
        stored = list(
            read_memory_store_snapshot(
                str(fixture.workspace.db_path), fixture.workspace.workspace_id
            ).contents_by_id.values()
        )
        manager = _open(_lazy(fixture.workspace.config), _new_logger())
        try:
            assert manager.get_all() == stored
            assert manager.count() == len(stored)
        finally:
            _close(manager)
        assert sorted(stored) == sorted(CONTENTS.values())


class TestPendingIntentsStillConvergeFromAnOlderDatabase:
    """A pending row is still recovered, now after the index is verified."""

    def test_a_pending_add_in_a_rollback_journal_database_converges(
        self, tmp_path: Path
    ) -> None:
        workspace = _build(tmp_path)
        _drop_vector(workspace, workspace.ids["delta"])
        _journal(
            workspace,
            lambda store: store.begin_add_intents(
                workspace.workspace_id, [CONTENTS["delta"]]
            ),
        )
        _convert_to_rollback_journal(workspace)
        assert _stored_journal_mode(workspace) == "delete"

        manager, _logger = create_memory_manager(workspace.config)
        try:
            assert sorted(manager.get_all()) == sorted(CONTENTS.values())
            engine = manager._semantic_engine
            assert isinstance(engine, USearchEngine)
            assert {int(key) for key in engine.index.keys} == set(
                _sqlite_ids(workspace)
            )
            assert manager.pending_intent_count() == 0
        finally:
            _close(manager)

    def test_a_pending_add_does_not_excuse_drift_in_a_rollback_journal_database(
        self, tmp_path: Path
    ) -> None:
        """Recovery verifies first: unrelated drift is refused without writes."""
        workspace = _build(tmp_path)
        _drop_vector(workspace, workspace.ids["delta"])
        _drop_vector(workspace, workspace.ids["beta"])
        _journal(
            workspace,
            lambda store: store.begin_add_intents(
                workspace.workspace_id, [CONTENTS["delta"]]
            ),
        )
        _convert_to_rollback_journal(workspace)
        before = _file_state(workspace)

        _ = _construct_expecting_refusal(workspace, _new_logger())

        assert _file_state(workspace) == before
        assert _stored_journal_mode(workspace) == "delete"
        assert _auxiliary_files(workspace) == []
