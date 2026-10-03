"""A refused workspace is not changed even by SQLite's own crash recovery.

A rollback-journal database whose last transaction crashed carries a hot
journal. Opening such a file read-write makes SQLite roll that journal back,
which rewrites the database before the vector check can refuse the
workspace. The check therefore reads SQLite read-only: a database that needs
recovery reads as unreadable, the existing fail-closed rule applies, and the
operator recovers it deliberately.

Everything here uses real engines. The hot journal is produced by a child
process that spills an uncommitted transaction to the database file and then
exits without rolling it back.
"""

import hashlib
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.core.exceptions import InitializationError
from reflectlog.infrastructure.usearch_engine import _sqlite_memory_count
from tests.integration.test_memory_manager_usearch import create_memory_manager
from tests.integration.test_vector_index_drift import (
    Drift,
    Operation,
    Workspace,
    _close,
    _construct_expecting_refusal,
    _file_state,
    _invoke,
    _lazy,
    _new_logger,
    _open,
)
from tests.integration.test_vector_index_drift_storage_modes import (
    Fixture,
    Storage,
    _fixture,
)

pytestmark = pytest.mark.integration

UNREADABLE = "SQLite is unreadable"

CRASH_IN_TRANSACTION = """
import os
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("PRAGMA cache_size=1")
connection.execute("PRAGMA cache_spill=ON")
connection.execute("BEGIN")
for index in range(40):
    connection.execute(
        "INSERT INTO memories(workspace_id, content) VALUES (?, ?)",
        (sys.argv[2], "x" * 3000 + str(index)),
    )
os._exit(0)
"""


def _journal(db_path: Path) -> Path:
    return Path(f"{db_path}-journal")


def _leave_hot_journal(db_path: Path, workspace_id: str) -> None:
    """Crash a writer mid-transaction so the journal stays behind, hot."""
    _ = subprocess.run(
        [sys.executable, "-c", CRASH_IN_TRANSACTION, str(db_path), workspace_id],
        check=True,
        timeout=60,
    )
    assert _journal(db_path).exists()
    uri = f"{db_path.absolute().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            _ = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
    finally:
        connection.close()


def _journal_digest(workspace: Workspace) -> str:
    return hashlib.sha256(_journal(workspace.db_path).read_bytes()).hexdigest()


def _assert_crash_state_untouched(fixture: Fixture, journal: str, step: str) -> None:
    workspace = fixture.workspace
    assert _file_state(workspace) == fixture.before, step
    assert _journal(workspace.db_path).exists(), step
    assert _journal_digest(workspace) == journal, step
    assert not Path(f"{workspace.db_path}-wal").exists(), step


def _crashed_fixture(tmp_path: Path) -> tuple[Fixture, str]:
    fixture = _fixture(tmp_path, Drift.MISSING_VECTOR, Storage.DELETE_MODE)
    _leave_hot_journal(fixture.workspace.db_path, fixture.workspace.workspace_id)
    crashed = Fixture(
        fixture.workspace, fixture.storage, _file_state(fixture.workspace)
    )
    return crashed, _journal_digest(fixture.workspace)


class TestHotJournalWorkspaceIsRefusedWithoutRecovery:
    def test_eager_construction_refuses_and_leaves_the_files_alone(
        self, tmp_path: Path
    ) -> None:
        fixture, journal = _crashed_fixture(tmp_path)

        error = _construct_expecting_refusal(fixture.workspace, _new_logger())

        assert UNREADABLE in str(error)
        _assert_crash_state_untouched(fixture, journal, "eager construction")

    def test_lazy_construction_refuses_before_the_writable_store_opens(
        self, tmp_path: Path
    ) -> None:
        fixture, journal = _crashed_fixture(tmp_path)

        with pytest.raises(InitializationError) as caught:
            _close(_open(_lazy(fixture.workspace.config), _new_logger()))

        assert UNREADABLE in str(caught.value)
        _assert_crash_state_untouched(fixture, journal, "lazy construction")

    async def test_registry_acquire_refuses_and_leaves_the_files_alone(
        self, tmp_path: Path
    ) -> None:
        fixture, journal = _crashed_fixture(tmp_path)
        registry = WorkspaceRegistry(
            fixture.workspace.config,
            manager_factory=lambda config: create_memory_manager(config)[0],
        )
        try:
            with pytest.raises(InitializationError):
                async with registry.acquire(fixture.workspace.workspace_id):
                    pass
        finally:
            await registry.close()

        _assert_crash_state_untouched(fixture, journal, "registry acquire")

    @pytest.mark.parametrize(
        "operation",
        [Operation.SEARCH, Operation.ADD, Operation.ADD_ASYNC, Operation.REMOVE],
    )
    async def test_a_crash_after_a_lazy_construction_refuses_every_operation(
        self, tmp_path: Path, operation: Operation
    ) -> None:
        """The drift check itself must not recover the journal either."""
        fixture = _fixture(tmp_path, Drift.MISSING_VECTOR, Storage.DELETE_MODE)
        manager = _open(_lazy(fixture.workspace.config), _new_logger())
        try:
            _leave_hot_journal(
                fixture.workspace.db_path, fixture.workspace.workspace_id
            )
            crashed = Fixture(
                fixture.workspace, fixture.storage, _file_state(fixture.workspace)
            )
            journal = _journal_digest(fixture.workspace)

            with pytest.raises(InitializationError) as caught:
                _ = await _invoke(manager, fixture.workspace, operation)

            assert UNREADABLE in str(caught.value)
            _assert_crash_state_untouched(crashed, journal, operation.value)
        finally:
            _close(manager)


class TestSqliteMemoryCountNeverRecovers:
    def test_a_healthy_database_is_counted_and_left_alone(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memories.db"
        connection = sqlite3.connect(db_path)
        try:
            _ = connection.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY)")
            _ = connection.executemany(
                "INSERT INTO memories(id) VALUES (?)", [(1,), (2,), (3,)]
            )
            connection.commit()
        finally:
            connection.close()
        before = (db_path.read_bytes(), db_path.stat().st_mtime_ns)

        assert _sqlite_memory_count(str(db_path)) == 3
        assert (db_path.read_bytes(), db_path.stat().st_mtime_ns) == before

    def test_an_absent_database_counts_zero_and_is_not_created(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "missing.db"

        assert _sqlite_memory_count(str(db_path)) == 0
        assert not db_path.exists()

    def test_a_database_without_the_table_is_unreadable(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        sqlite3.connect(db_path).close()

        assert _sqlite_memory_count(str(db_path)) is None

    def test_a_hot_journal_is_unreadable_and_not_rolled_back(
        self, tmp_path: Path
    ) -> None:
        fixture = _fixture(tmp_path, Drift.MISSING_VECTOR, Storage.DELETE_MODE)
        db_path = fixture.workspace.db_path
        _leave_hot_journal(db_path, fixture.workspace.workspace_id)
        before = (db_path.read_bytes(), db_path.stat().st_mtime_ns)
        journal = _journal(db_path).read_bytes()

        assert _sqlite_memory_count(str(db_path)) is None

        assert (db_path.read_bytes(), db_path.stat().st_mtime_ns) == before
        assert _journal(db_path).read_bytes() == journal
