"""Unit tests for the strictly read-only SQLite snapshot of a workspace."""

import hashlib
import os
from pathlib import Path
import shutil
import sqlite3

import pytest

from reflectlog.core.enums import TransitionKind, TransitionStatus
from reflectlog.core.exceptions import StorageError
from reflectlog.core.types import (
    ReplacementTransition,
    ReplacementTransitionRequest,
)
from reflectlog.infrastructure import memory_store as memory_store_module
from reflectlog.infrastructure.memory_store import (
    MemoryStore,
    MemoryStoreSnapshotError,
    read_memory_store_snapshot,
)

WS = "snap-ws"
OTHER = "other-ws"
SECRET = "SECRET-MEMORY-TEXT"


def test_foreign_pending_count_excludes_own_and_completed(db_path: Path) -> None:
    store = MemoryStore(db_path=str(db_path))
    try:
        store.begin_add_intents(WS, ["own"])
        store.begin_add_intents(OTHER, ["foreign"])
        completed = store.begin_add_intents(OTHER, ["completed"])
        store.complete_replacement_transition(completed[0].id)
    finally:
        store.close()
    snapshot = read_memory_store_snapshot(str(db_path), WS)
    assert snapshot.foreign_pending_count == 1
    assert len(snapshot.pending_transitions) == 1


def test_foreign_pending_count_without_journal(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE memories (id INTEGER, workspace_id TEXT, content TEXT)"
        )
        connection.commit()
    finally:
        connection.close()
    assert read_memory_store_snapshot(str(db_path), WS).foreign_pending_count == 0


def test_foreign_pending_count_without_kind_column(db_path: Path) -> None:
    store = MemoryStore(db_path=str(db_path))
    store.begin_add_intents(OTHER, ["foreign"])
    store.close()
    connection = sqlite3.connect(db_path)
    try:
        for name in (
            "idx_transition_old_replace",
            "idx_transition_old_delete",
            "idx_pending_add",
        ):
            connection.execute(f"DROP INDEX {name}")
        connection.execute("ALTER TABLE replacement_transitions DROP COLUMN kind")
        connection.commit()
    finally:
        connection.close()
    assert read_memory_store_snapshot(str(db_path), WS).foreign_pending_count == 1


def _fingerprint(path: Path) -> tuple[str, int] | None:
    """Return (sha256, mtime_ns) of a file, or None when it does not exist."""
    if not path.exists():
        return None
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_mtime_ns,
    )


def _wal_digest(path: Path) -> str:
    """Return the sha256 of the WAL's bytes, counting an absent WAL as empty."""
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def _seed(db_path: Path) -> dict[str, int]:
    """Create rows and one pending journal row of each kind, then checkpoint."""
    store = MemoryStore(db_path=str(db_path))
    try:
        ids = {
            "alpha": store.insert(WS, f"{SECRET} alpha"),
            "beta": store.insert(WS, f"{SECRET} beta"),
            "gamma": store.insert(WS, f"{SECRET} gamma"),
            "foreign": store.insert(OTHER, f"{SECRET} foreign"),
        }
        _ = store.begin_add_intents(WS, [f"{SECRET} added"])
        _ = store.begin_delete_intents(WS, [(ids["beta"], f"{SECRET} beta")])
        _ = store.begin_replacement_transitions(
            [
                ReplacementTransitionRequest(
                    old_memory_id=ids["gamma"],
                    workspace_id=WS,
                    old_content=f"{SECRET} gamma",
                    new_content=f"{SECRET} gamma v2",
                    reason="update",
                    confidence=0.9,
                )
            ]
        )
        _ = store.begin_add_intents(OTHER, [f"{SECRET} foreign added"])
    finally:
        store.close()
    _checkpoint(db_path)
    return ids


def _checkpoint(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memories.db"


class TestSnapshotContent:
    def test_returns_only_this_workspaces_rows(self, db_path: Path) -> None:
        ids = _seed(db_path)

        snapshot = read_memory_store_snapshot(str(db_path), WS)

        assert snapshot.workspace_id == WS
        assert dict(snapshot.contents_by_id) == {
            ids["alpha"]: f"{SECRET} alpha",
            ids["beta"]: f"{SECRET} beta",
            ids["gamma"]: f"{SECRET} gamma",
        }

    def test_returns_only_pending_rows_of_this_workspace_in_journal_order(
        self, db_path: Path
    ) -> None:
        ids = _seed(db_path)

        snapshot = read_memory_store_snapshot(str(db_path), WS)

        kinds = [row.kind for row in snapshot.pending_transitions]
        assert kinds == [
            TransitionKind.ADD,
            TransitionKind.DELETE,
            TransitionKind.REPLACE,
        ]
        assert {row.workspace_id for row in snapshot.pending_transitions} == {WS}
        assert {row.status for row in snapshot.pending_transitions} == {
            TransitionStatus.PENDING
        }
        ordered = [row.id for row in snapshot.pending_transitions]
        assert ordered == sorted(ordered)
        add, delete, replace = snapshot.pending_transitions
        assert add.old_memory_id == 0
        assert add.new_content == f"{SECRET} added"
        assert (delete.old_memory_id, delete.old_content) == (
            ids["beta"],
            f"{SECRET} beta",
        )
        assert replace.new_content == f"{SECRET} gamma v2"
        assert snapshot.unrecognized_pending_count == 0

    def test_completed_rows_are_not_returned(self, db_path: Path) -> None:
        _ = _seed(db_path)
        store = MemoryStore(db_path=str(db_path))
        try:
            first = store.list_pending_transitions()[0]
            store.complete_replacement_transition(first.id)
        finally:
            store.close()

        snapshot = read_memory_store_snapshot(str(db_path), WS)

        assert first.id not in {row.id for row in snapshot.pending_transitions}
        assert len(snapshot.pending_transitions) == 2

    def test_other_workspace_snapshot_is_independent(self, db_path: Path) -> None:
        _ = _seed(db_path)

        snapshot = read_memory_store_snapshot(str(db_path), OTHER)

        assert list(snapshot.contents_by_id.values()) == [f"{SECRET} foreign"]
        assert [row.new_content for row in snapshot.pending_transitions] == [
            f"{SECRET} foreign added"
        ]

    def test_empty_workspace_is_an_empty_snapshot(self, db_path: Path) -> None:
        _ = _seed(db_path)

        snapshot = read_memory_store_snapshot(str(db_path), "nobody")

        assert dict(snapshot.contents_by_id) == {}
        assert snapshot.pending_transitions == ()

    def test_unknown_pending_kind_authorises_nothing_and_is_counted(
        self, db_path: Path
    ) -> None:
        _ = _seed(db_path)
        connection = sqlite3.connect(db_path)
        try:
            _ = connection.execute(
                "UPDATE replacement_transitions SET kind = 'mystery' "
                "WHERE workspace_id = ? AND kind = 'delete'",
                (WS,),
            )
            connection.commit()
        finally:
            connection.close()
        _checkpoint(db_path)

        snapshot = read_memory_store_snapshot(str(db_path), WS)

        assert [row.kind for row in snapshot.pending_transitions] == [
            TransitionKind.ADD,
            TransitionKind.REPLACE,
        ]
        assert snapshot.unrecognized_pending_count == 1

    def test_sees_committed_rows_while_a_writable_store_is_open(
        self, db_path: Path
    ) -> None:
        store = MemoryStore(db_path=str(db_path))
        try:
            first = store.insert(WS, "first")
            snapshot = read_memory_store_snapshot(str(db_path), WS)
            second = store.insert(WS, "second")
            later = read_memory_store_snapshot(str(db_path), WS)
        finally:
            store.close()

        assert dict(snapshot.contents_by_id) == {first: "first"}
        assert dict(later.contents_by_id) == {first: "first", second: "second"}

    def test_reads_uncheckpointed_wal_frames_of_a_crashed_writer(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        crashed = tmp_path / "crashed"
        crashed.mkdir()
        store = MemoryStore(db_path=str(db_path))
        try:
            row_id = store.insert(WS, "committed only in the wal")
            for suffix in ("", "-wal", "-shm"):
                source = Path(f"{db_path}{suffix}")
                if source.exists():
                    _ = shutil.copy2(source, crashed / f"memories.db{suffix}")
        finally:
            store.close()

        snapshot = read_memory_store_snapshot(str(crashed / "memories.db"), WS)

        assert dict(snapshot.contents_by_id) == {row_id: "committed only in the wal"}


class TestSnapshotIsReadOnly:
    def test_database_and_wal_are_byte_and_mtime_identical_afterwards(
        self, db_path: Path
    ) -> None:
        _ = _seed(db_path)
        wal = Path(f"{db_path}-wal")
        before = (_fingerprint(db_path), _wal_digest(wal))

        _ = read_memory_store_snapshot(str(db_path), WS)
        _ = read_memory_store_snapshot(str(db_path), OTHER)

        # Sidecar existence and -shm are not compared: stock SQLite deletes
        # -wal and -shm when the last connection closes and a read-only
        # connection must recreate them (sqlite.org/wal.html, section 5),
        # while Apple's build keeps them. The contract is therefore the
        # database by sha256 and mtime and the WAL by content, absent == empty.
        assert (_fingerprint(db_path), _wal_digest(wal)) == before

    def test_rollback_journal_database_gains_no_sidecar_files(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        _ = _seed(db_path)
        connection = sqlite3.connect(db_path)
        try:
            mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            assert mode is not None
            assert str(mode[0]).lower() == "delete"
        finally:
            connection.close()
        plain_dir = tmp_path / "rollback"
        plain_dir.mkdir()
        plain = plain_dir / "memories.db"
        _ = shutil.copy2(db_path, plain)
        before = _fingerprint(plain)

        snapshot = read_memory_store_snapshot(str(plain), WS)

        assert len(snapshot.contents_by_id) == 3
        assert sorted(os.listdir(plain_dir)) == ["memories.db"]
        assert _fingerprint(plain) == before

    def test_does_not_create_schema_in_an_empty_database_file(
        self, db_path: Path
    ) -> None:
        db_path.write_bytes(b"")

        with pytest.raises(MemoryStoreSnapshotError, match="no memories table"):
            _ = read_memory_store_snapshot(str(db_path), WS)

        assert db_path.read_bytes() == b""
        assert sorted(os.listdir(db_path.parent)) == ["memories.db"]

    def test_database_without_journal_table_has_no_pending_and_stays_so(
        self, db_path: Path
    ) -> None:
        connection = sqlite3.connect(db_path)
        try:
            _ = connection.execute(
                "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "workspace_id TEXT NOT NULL, content TEXT NOT NULL)"
            )
            _ = connection.execute(
                "INSERT INTO memories (workspace_id, content) VALUES (?, 'kept')",
                (WS,),
            )
            connection.commit()
        finally:
            connection.close()
        before = _fingerprint(db_path)

        snapshot = read_memory_store_snapshot(str(db_path), WS)

        assert dict(snapshot.contents_by_id) == {1: "kept"}
        assert snapshot.pending_transitions == ()
        assert _fingerprint(db_path) == before
        check = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in check.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            check.close()
        assert "replacement_transitions" not in tables

    def test_journal_table_without_kind_column_reads_as_replace_without_migrating(
        self, db_path: Path
    ) -> None:
        connection = sqlite3.connect(db_path)
        try:
            _ = connection.execute(
                "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "workspace_id TEXT NOT NULL, content TEXT NOT NULL)"
            )
            _ = connection.execute(
                "CREATE TABLE replacement_transitions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL, "
                "old_memory_id INTEGER NOT NULL, old_content TEXT NOT NULL, "
                "new_content TEXT NOT NULL, archive_id INTEGER NOT NULL, "
                "reason TEXT NOT NULL, confidence REAL NOT NULL, "
                "status TEXT NOT NULL)"
            )
            _ = connection.execute(
                "INSERT INTO replacement_transitions (workspace_id, old_memory_id, "
                "old_content, new_content, archive_id, reason, confidence, status) "
                "VALUES (?, 4, 'old', 'new', 1, 'r', 0.5, 'pending')",
                (WS,),
            )
            connection.commit()
        finally:
            connection.close()
        before = _fingerprint(db_path)

        snapshot = read_memory_store_snapshot(str(db_path), WS)

        assert [row.kind for row in snapshot.pending_transitions] == [
            TransitionKind.REPLACE
        ]
        assert _fingerprint(db_path) == before
        check = sqlite3.connect(db_path)
        try:
            columns = {
                row[1]
                for row in check.execute("PRAGMA table_info(replacement_transitions)")
            }
        finally:
            check.close()
        assert "kind" not in columns


class TestSnapshotFailures:
    def test_missing_database_is_reported_explicitly_and_not_created(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "nested" / "memories.db"

        with pytest.raises(MemoryStoreSnapshotError, match="missing") as caught:
            _ = read_memory_store_snapshot(str(db_path), WS)

        assert caught.value.db_missing is True
        assert isinstance(caught.value, StorageError)
        assert not db_path.exists()
        assert not db_path.parent.exists()

    def test_corrupt_database_raises_a_clear_error_and_is_left_alone(
        self, db_path: Path
    ) -> None:
        payload = b"this is not a sqlite database" * 64
        db_path.write_bytes(payload)
        before = _fingerprint(db_path)

        with pytest.raises(MemoryStoreSnapshotError, match="unreadable") as caught:
            _ = read_memory_store_snapshot(str(db_path), WS)

        assert caught.value.db_missing is False
        assert db_path.read_bytes() == payload
        assert _fingerprint(db_path) == before

    def test_errors_never_contain_memory_text(self, db_path: Path) -> None:
        _ = _seed(db_path)
        connection = sqlite3.connect(db_path)
        try:
            _ = connection.execute("DROP TABLE memories")
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(MemoryStoreSnapshotError) as caught:
            _ = read_memory_store_snapshot(str(db_path), WS)

        assert SECRET not in str(caught.value)

    def test_malformed_row_id_is_refused(self, db_path: Path) -> None:
        connection = sqlite3.connect(db_path)
        try:
            _ = connection.execute(
                "CREATE TABLE memories (id, workspace_id TEXT, content TEXT)"
            )
            _ = connection.execute(
                "INSERT INTO memories VALUES ('not-an-int', ?, 'x')", (WS,)
            )
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(MemoryStoreSnapshotError, match="malformed"):
            _ = read_memory_store_snapshot(str(db_path), WS)


class TestSnapshotConsistency:
    def test_rows_and_journal_come_from_one_transaction(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A commit between the two reads must be invisible to both."""
        _ = _seed(db_path)
        original = memory_store_module._read_pending_snapshot

        def commit_then_read(
            connection: sqlite3.Connection, tables: set[str], workspace_id: str
        ) -> tuple[tuple[ReplacementTransition, ...], int]:
            writer = MemoryStore(db_path=str(db_path))
            try:
                _ = writer.insert(WS, "inserted mid-snapshot")
                _ = writer.begin_add_intents(WS, ["journaled mid-snapshot"])
            finally:
                writer.close()
            return original(connection, tables, workspace_id)

        monkeypatch.setattr(
            memory_store_module, "_read_pending_snapshot", commit_then_read
        )

        snapshot = read_memory_store_snapshot(str(db_path), WS)

        assert "inserted mid-snapshot" not in snapshot.contents_by_id.values()
        assert "journaled mid-snapshot" not in {
            row.new_content for row in snapshot.pending_transitions
        }
        monkeypatch.undo()
        fresh = read_memory_store_snapshot(str(db_path), WS)
        assert "inserted mid-snapshot" in fresh.contents_by_id.values()
        assert "journaled mid-snapshot" in {
            row.new_content for row in fresh.pending_transitions
        }
