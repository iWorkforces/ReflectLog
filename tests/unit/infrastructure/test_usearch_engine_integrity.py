"""Engine-level vector-index integrity: real USearchEngine and MemoryStore.

The engine compares SQLite memory ids with the live keys of the vector index
whenever it accepts a candidate index (first open and external reload), and
on demand through ``verify_index_integrity``. Pending journal rows account
for bounded transitional differences only. A refusal writes nothing and
leaves the engine without an index so every retry refuses again.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pytest
from usearch.index import Index

from reflectlog.core.enums import EmbedderProvider, TransitionKind
from reflectlog.core.exceptions import InitializationError, StorageError
from reflectlog.core.storage_coordination import LeaseMode
from reflectlog.core.types import (
    IIndexIntegrityVerifier,
    IMemorySnapshotReader,
    ReplacementTransitionRequest,
)
from reflectlog.infrastructure.index_integrity import INDEX_DRIFT_OPERATOR_ACTION
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine
from tests.unit.infrastructure.test_usearch_engine import MockEmbedder, journal_add

WS = "test"
CONTENTS = ("alpha memory", "beta memory", "gamma memory", "delta memory")
DIMS = 128


@dataclass(frozen=True)
class Store:
    config: USearchConfig
    ids: dict[str, int]


@pytest.fixture
def config(tmp_path: Path) -> USearchConfig:
    return USearchConfig(
        workspace_id=WS,
        index_path=str(tmp_path / "index.usearch"),
        db_path=str(tmp_path / "messages.db"),
        embedding_dims=DIMS,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/mock-128",
    )


def _engine(config: USearchConfig) -> USearchEngine:
    return USearchEngine(config=config, embedder=MockEmbedder(dims=DIMS))


def _checkpoint(config: USearchConfig) -> None:
    connection = sqlite3.connect(config.db_path)
    try:
        _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


def _seed(config: USearchConfig, contents: tuple[str, ...] = CONTENTS) -> Store:
    engine = _engine(config)
    try:
        for content in contents:
            engine.add(WS, content, infer=False)
        engine.commit()
        ids: dict[str, int] = {}
        for content in contents:
            row_id = engine.get_id_by_content(WS, content)
            assert row_id is not None
            ids[content] = row_id
    finally:
        engine.close()
    _checkpoint(config)
    return Store(config, ids)


def _load(path: str) -> Index:
    index = Index.restore(path)
    assert index is not None
    return index


def _keys(path: str) -> set[int]:
    keys: list[int] = list(_load(path).keys)
    return set(keys)


def _drop_vector(config: USearchConfig, row_id: int) -> None:
    index = _load(config.index_path)
    index.remove(row_id)
    index.save(config.index_path)


def _add_vector(config: USearchConfig, key: int) -> None:
    index = _load(config.index_path)
    index.add(key, np.random.default_rng(key).random(DIMS).astype(np.float32))
    index.save(config.index_path)


def _sql(config: USearchConfig, statement: str, *params: object) -> None:
    connection = sqlite3.connect(config.db_path)
    try:
        _ = connection.execute(statement, params)
        connection.commit()
    finally:
        connection.close()
    _checkpoint(config)


def _delete_row(config: USearchConfig, row_id: int) -> None:
    _sql(config, "DELETE FROM memories WHERE id = ?", row_id)


def _fingerprint(config: USearchConfig) -> tuple[object, ...]:
    """Vector file and db by sha256+mtime; the WAL by content (absent == empty)."""

    def digest(path: str) -> tuple[str, int]:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest(), os.stat(path).st_mtime_ns

    wal = f"{config.db_path}-wal"
    wal_bytes = b""
    if os.path.exists(wal):
        with open(wal, "rb") as handle:
            wal_bytes = handle.read()
    return (
        digest(config.index_path),
        digest(config.db_path),
        hashlib.sha256(wal_bytes).hexdigest(),
    )


def _drifted(config: USearchConfig, kind: str) -> Store:
    """Build one of the drift shapes and return the seeded store."""
    if kind == "older_index":
        store = _seed(config, CONTENTS[:2])
        _ = shutil.copy2(config.index_path, f"{config.index_path}.older")
        engine = _engine(config)
        try:
            for content in CONTENTS[2:]:
                engine.add(WS, content, infer=False)
            engine.commit()
            for content in CONTENTS[2:]:
                row_id = engine.get_id_by_content(WS, content)
                assert row_id is not None
                store.ids[content] = row_id
        finally:
            engine.close()
        _ = shutil.copy2(f"{config.index_path}.older", config.index_path)
        os.remove(f"{config.index_path}.older")
        _checkpoint(config)
        return store
    store = _seed(config)
    beta = store.ids[CONTENTS[1]]
    match kind:
        case "missing_vector":
            _drop_vector(config, beta)
        case "orphan_vector":
            _delete_row(config, beta)
        case "swapped_ids":
            _drop_vector(config, beta)
            _add_vector(config, max(store.ids.values()) + 10)
        case _:
            raise AssertionError(kind)
    return store


DRIFTS = ["missing_vector", "orphan_vector", "swapped_ids", "older_index"]


def _difference(config: USearchConfig) -> tuple[set[int], set[int]]:
    connection = sqlite3.connect(f"file:{config.db_path}?mode=ro", uri=True)
    try:
        rows = {int(row[0]) for row in connection.execute("SELECT id FROM memories")}
    finally:
        connection.close()
    keys = _keys(config.index_path)
    return rows - keys, keys - rows


def _assert_refusal(text: str, config: USearchConfig) -> None:
    missing, orphans = _difference(config)
    assert INDEX_DRIFT_OPERATOR_ACTION in text
    if missing:
        ids = ", ".join(str(item) for item in sorted(missing))
        assert f"{len(missing)} rows without a vector (ids: {ids})" in text
    if orphans:
        ids = ", ".join(str(item) for item in sorted(orphans))
        assert f"{len(orphans)} vectors without a row (ids: {ids})" in text
    for content in CONTENTS:
        assert content not in text


class TestFirstOpen:
    @pytest.mark.parametrize("kind", DRIFTS)
    def test_drift_is_refused_without_writes_and_without_a_half_open_index(
        self, config: USearchConfig, kind: str
    ) -> None:
        _ = _drifted(config, kind)
        before = _fingerprint(config)
        engine = _engine(config)
        try:
            for _attempt in range(2):
                with pytest.raises(InitializationError) as caught:
                    _ = engine.index
                _assert_refusal(str(caught.value), config)
                assert engine._index is None
                assert engine._memory_store is None
        finally:
            engine.close()

        assert _fingerprint(config) == before

    def test_equal_counts_do_not_hide_swapped_ids(self, config: USearchConfig) -> None:
        store = _drifted(config, "swapped_ids")
        assert len(_keys(config.index_path)) == len(store.ids)

        engine = _engine(config)
        try:
            with pytest.raises(InitializationError, match="rows without a vector"):
                _ = engine.index
        finally:
            engine.close()

    def test_agreeing_ids_open_including_after_removal_and_reload(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        engine = _engine(config)
        try:
            engine.delete(str(store.ids[CONTENTS[1]]))
            engine.delete(str(store.ids[CONTENTS[3]]))
            engine.commit()
        finally:
            engine.close()
        remaining = {store.ids[CONTENTS[0]], store.ids[CONTENTS[2]]}
        assert _keys(config.index_path) == remaining

        reopened = _engine(config)
        try:
            assert {int(key) for key in reopened.index.keys} == remaining
            assert set(reopened.get_all(WS)) == {CONTENTS[0], CONTENTS[2]}
        finally:
            reopened.close()

    def test_refusal_keeps_temp_files_untouched(self, tmp_path: Path) -> None:
        nested = tmp_path / "ws" / "usearch"
        nested.mkdir(parents=True)
        config = USearchConfig(
            workspace_id=WS,
            index_path=str(nested / "vectors.usearch"),
            db_path=str(nested / "memories.db"),
            embedding_dims=DIMS,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
        )
        _ = _drifted(config, "missing_vector")
        own_temp = f"{config.index_path}.{os.getpid()}.1.tmp"
        with open(own_temp, "wb") as handle:
            _ = handle.write(b"stale temp")

        engine = _engine(config)
        try:
            with pytest.raises(InitializationError):
                _ = engine.index
        finally:
            engine.close()

        assert os.path.exists(own_temp)

    def test_first_create_never_saves_the_live_file(
        self, config: USearchConfig
    ) -> None:
        engine = _engine(config)
        try:
            assert len(engine.index) == 0
            assert not os.path.exists(config.index_path)
        finally:
            engine.close()


class TestPendingJournalAllowances:
    def _open(self, config: USearchConfig) -> USearchEngine:
        engine = _engine(config)
        try:
            _ = engine.index
        except BaseException:
            engine.close()
            raise
        return engine

    def test_pending_add_allows_its_row_without_a_vector(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        _drop_vector(config, store.ids[CONTENTS[3]])
        journal_add(config, CONTENTS[3])

        engine = self._open(config)
        engine.close()

    def test_pending_add_never_allows_an_orphan_vector(
        self, config: USearchConfig
    ) -> None:
        _ = _seed(config)
        _add_vector(config, 99)
        journal_add(config, CONTENTS[0])

        with pytest.raises(InitializationError, match=r"vectors without a row"):
            self._open(config).close()

    def test_pending_delete_allows_the_recorded_orphan(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        beta = store.ids[CONTENTS[1]]
        _journal_delete(config, beta, CONTENTS[1])
        _delete_row(config, beta)

        engine = self._open(config)
        engine.close()

    def test_pending_delete_allows_a_row_still_live_without_its_vector(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        beta = store.ids[CONTENTS[1]]
        _journal_delete(config, beta, CONTENTS[1])
        _drop_vector(config, beta)

        engine = self._open(config)
        engine.close()

    def test_pending_replace_allows_old_orphan_and_new_row_without_vector(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        old = store.ids[CONTENTS[0]]
        _journal_replace(config, old, CONTENTS[0], "alpha memory v2")
        _delete_row(config, old)
        with_store = MemoryStore(db_path=config.db_path)
        try:
            _ = with_store.insert(WS, "alpha memory v2")
        finally:
            with_store.close()
        _checkpoint(config)

        engine = self._open(config)
        engine.close()

    def test_pending_replace_does_not_allow_an_arbitrary_orphan(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        old = store.ids[CONTENTS[0]]
        _journal_replace(config, old, CONTENTS[0], "alpha memory v2")
        _delete_row(config, old)
        _add_vector(config, 99)

        with pytest.raises(InitializationError) as caught:
            self._open(config).close()

        assert "1 vectors without a row (ids: 99)" in str(caught.value)

    def test_pending_row_cannot_mask_an_unrelated_missing_vector(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        excused = store.ids[CONTENTS[3]]
        unrelated = store.ids[CONTENTS[1]]
        _drop_vector(config, excused)
        _drop_vector(config, unrelated)
        journal_add(config, CONTENTS[3])

        with pytest.raises(InitializationError) as caught:
            self._open(config).close()

        assert f"1 rows without a vector (ids: {unrelated})" in str(caught.value)

    def test_completed_row_allows_nothing(self, config: USearchConfig) -> None:
        store = _seed(config)
        _drop_vector(config, store.ids[CONTENTS[3]])
        journal_add(config, CONTENTS[3])
        journal = MemoryStore(db_path=config.db_path)
        try:
            for row in journal.list_pending_transitions():
                journal.complete_replacement_transition(row.id)
        finally:
            journal.close()
        _checkpoint(config)

        with pytest.raises(InitializationError, match="rows without a vector"):
            self._open(config).close()


def _journal_delete(config: USearchConfig, row_id: int, content: str) -> None:
    store = MemoryStore(db_path=config.db_path)
    try:
        _ = store.begin_delete_intents(WS, [(row_id, content)])
    finally:
        store.close()
    _checkpoint(config)


def _journal_replace(config: USearchConfig, old_id: int, old: str, new: str) -> None:
    store = MemoryStore(db_path=config.db_path)
    try:
        _ = store.begin_replacement_transitions(
            [
                ReplacementTransitionRequest(
                    old_memory_id=old_id,
                    workspace_id=WS,
                    old_content=old,
                    new_content=new,
                    reason="update",
                    confidence=0.9,
                )
            ]
        )
        pending = store.list_pending_transitions()
        assert [row.kind for row in pending] == [TransitionKind.REPLACE]
    finally:
        store.close()
    _checkpoint(config)


class TestExternalReload:
    def _publish_drift(self, config: USearchConfig, store: Store) -> None:
        """Another writer publishes a vector file that disagrees with SQLite."""
        _drop_vector(config, store.ids[CONTENTS[1]])

    def test_invalid_candidate_is_refused_and_accepted_index_is_kept(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        engine = _engine(config)
        try:
            accepted = engine.index
            identity = engine._seen_identity
            self._publish_drift(config, store)

            for _attempt in range(2):
                with pytest.raises(InitializationError) as caught:
                    engine.refresh()
                _assert_refusal(str(caught.value), config)
                assert engine._index is accepted
                assert engine._seen_identity == identity
            assert {int(key) for key in accepted.keys} == set(store.ids.values())
        finally:
            engine.close()

    def test_valid_newer_candidate_replaces_the_accepted_index(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        engine = _engine(config)
        writer = _engine(config)
        try:
            accepted = engine.index
            writer.delete(str(store.ids[CONTENTS[1]]))
            writer.commit()

            engine.refresh()

            assert engine._index is not accepted
            assert store.ids[CONTENTS[1]] not in engine.index
        finally:
            writer.close()
            engine.close()

    def test_refresh_recovers_once_the_candidate_is_fixed(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        engine = _engine(config)
        try:
            _ = engine.index
            self._publish_drift(config, store)
            with pytest.raises(InitializationError):
                engine.refresh()
            _delete_row(config, store.ids[CONTENTS[1]])

            engine.refresh()

            assert {int(key) for key in engine.index.keys} == _keys(config.index_path)
        finally:
            engine.close()


class TestRefresh:
    def test_refresh_does_not_open_an_unloaded_index(
        self, config: USearchConfig
    ) -> None:
        _ = _seed(config)
        engine = _engine(config)
        try:
            engine.refresh()

            assert engine._index is None
            assert engine._memory_store is None
        finally:
            engine.close()

    @pytest.mark.parametrize("kind", DRIFTS)
    def test_sqlite_only_reads_work_on_a_drifted_unloaded_workspace(
        self, config: USearchConfig, kind: str
    ) -> None:
        _ = _drifted(config, kind)
        engine = _engine(config)
        try:
            engine.refresh()

            rows = engine.get_all(WS)
            assert engine.count(WS) == len(rows)
            assert engine._index is None
        finally:
            engine.close()

    def test_refresh_on_a_closed_engine_still_raises(
        self, config: USearchConfig
    ) -> None:
        engine = _engine(config)
        engine.close()

        with pytest.raises(StorageError, match="closed"):
            engine.refresh()

    def test_refresh_keeps_reloading_an_open_stale_index(
        self, config: USearchConfig
    ) -> None:
        stale = _engine(config)
        writer = _engine(config)
        try:
            _ = stale.index
            writer.add(WS, CONTENTS[0], infer=False)
            writer.commit()
            assert len(stale.index) == 0

            stale.refresh()

            assert len(stale.index) == 1
        finally:
            stale.close()
            writer.close()


class TestVerifyIndexIntegrity:
    def test_engine_implements_the_capability_protocols(
        self, config: USearchConfig
    ) -> None:
        engine = _engine(config)
        try:
            assert isinstance(engine, IIndexIntegrityVerifier)
            assert isinstance(engine, IMemorySnapshotReader)
        finally:
            engine.close()

    def test_consistent_open_index_passes(self, config: USearchConfig) -> None:
        _ = _seed(config)
        engine = _engine(config)
        try:
            _ = engine.index
            engine.verify_index_integrity()
        finally:
            engine.close()

    def test_unloaded_engine_opens_and_verifies(self, config: USearchConfig) -> None:
        _ = _drifted(config, "missing_vector")
        engine = _engine(config)
        try:
            with pytest.raises(InitializationError, match="rows without a vector"):
                engine.verify_index_integrity()
            assert engine._index is None
        finally:
            engine.close()

    def test_detects_drift_that_appears_after_the_index_was_accepted(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        engine = _engine(config)
        try:
            _ = engine.index
            engine.verify_index_integrity()
            _delete_row(config, store.ids[CONTENTS[1]])
            before = _fingerprint(config)

            with pytest.raises(InitializationError) as caught:
                engine.verify_index_integrity()

            assert "vectors without a row" in str(caught.value)
            assert INDEX_DRIFT_OPERATOR_ACTION in str(caught.value)
            assert _fingerprint(config) == before
        finally:
            engine.close()

    def test_pending_journal_is_honoured_on_reverification(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        beta = store.ids[CONTENTS[1]]
        engine = _engine(config)
        try:
            _ = engine.index
            _journal_delete(config, beta, CONTENTS[1])
            _delete_row(config, beta)

            engine.verify_index_integrity()
        finally:
            engine.close()

    def test_takes_no_lease_while_the_exclusive_lease_is_held(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        directory = root / WS / "usearch"
        directory.mkdir(parents=True)
        config = USearchConfig(
            workspace_id=WS,
            index_path=str(directory / "vectors.usearch"),
            db_path=str(directory / "memories.db"),
            embedding_dims=DIMS,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
        )
        _ = _seed(config)
        coordinator = PortalockerStorageCoordinator(str(root), timeout=1.0)
        engine = USearchEngine(
            config=config, embedder=MockEmbedder(dims=DIMS), coordinator=coordinator
        )
        try:
            _ = engine.index
            with coordinator.acquire(WS, LeaseMode.EXCLUSIVE):
                engine.verify_index_integrity()
        finally:
            engine.close()

    def test_closed_engine_refuses(self, config: USearchConfig) -> None:
        engine = _engine(config)
        engine.close()

        with pytest.raises(StorageError, match="closed"):
            engine.verify_index_integrity()


class TestReadMemorySnapshot:
    def test_reads_rows_and_journal_without_creating_the_writable_store(
        self, config: USearchConfig
    ) -> None:
        store = _seed(config)
        journal_add(config, "pending text")
        before = _fingerprint(config)
        engine = _engine(config)
        try:
            snapshot = engine.read_memory_snapshot()

            assert set(snapshot.contents_by_id) == set(store.ids.values())
            assert [row.new_content for row in snapshot.pending_transitions] == [
                "pending text"
            ]
            assert engine._memory_store is None
        finally:
            engine.close()

        assert _fingerprint(config) == before

    def test_missing_database_is_reported(self, config: USearchConfig) -> None:
        engine = _engine(config)
        try:
            with pytest.raises(StorageError, match="missing"):
                _ = engine.read_memory_snapshot()
            assert not os.path.exists(config.db_path)
        finally:
            engine.close()
