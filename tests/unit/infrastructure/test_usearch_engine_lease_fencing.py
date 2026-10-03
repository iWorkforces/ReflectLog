"""Lease fencing of the vector index's lazy open and external reload.

Restoring the file, comparing its keys with SQLite, installing it and capturing
its identity must not interleave with another writer's publish. The engine
therefore takes a SHARED workspace lease around those steps (before any engine
lock), and skips it when the calling thread already holds a lease. These tests
use a real ``PortalockerStorageCoordinator`` with real ``USearchEngine`` and
``MemoryStore`` objects. Threads coordinate through Events and bounded joins;
the only timed wait is a short join that proves a thread is *still blocked*,
which cannot fail spuriously when the fence works.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading

import pytest
from usearch.index import Index

from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.storage_coordination import IStorageLease, LeaseMode
from reflectlog.infrastructure import usearch_engine
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine
from tests.unit.infrastructure.test_usearch_engine import MockEmbedder

WS = "test"
DIMS = 128
SEED = ("alpha memory", "beta memory")
FRESH = "gamma memory"
READER = "reader"
WRITER = "writer"
FAILURE_BOUND = 10.0
STILL_BLOCKED_PROBE = 0.3


@pytest.fixture
def config(tmp_path: Path) -> USearchConfig:
    directory = tmp_path / "root" / WS / "usearch"
    directory.mkdir(parents=True)
    return USearchConfig(
        workspace_id=WS,
        index_path=str(directory / "vectors.usearch"),
        db_path=str(directory / "memories.db"),
        embedding_dims=DIMS,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/mock-128",
    )


@pytest.fixture
def coordinator(tmp_path: Path) -> PortalockerStorageCoordinator:
    return PortalockerStorageCoordinator(str(tmp_path / "root"), timeout=FAILURE_BOUND)


def _engine(
    config: USearchConfig, coordinator: PortalockerStorageCoordinator
) -> USearchEngine:
    return USearchEngine(
        config=config, embedder=MockEmbedder(dims=DIMS), coordinator=coordinator
    )


def _seed(config: USearchConfig, coordinator: PortalockerStorageCoordinator) -> None:
    engine = _engine(config, coordinator)
    try:
        for content in SEED:
            engine.add(WS, content, infer=False)
        engine.commit()
    finally:
        engine.close()


def _file_identity(config: USearchConfig) -> tuple[int, int] | None:
    return usearch_engine._index_file_identity(config.index_path)


class Task:
    """A named thread that captures the exception it ends with."""

    def __init__(self, name: str, target: Callable[[], object]) -> None:
        self.error: BaseException | None = None
        self._target = target
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def _run(self) -> None:
        try:
            _ = self._target()
        except Exception as error:
            self.error = error

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float = FAILURE_BOUND) -> None:
        self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


class Writer:
    """Holds the exclusive lease, then publishes one memory on command."""

    def __init__(
        self,
        coordinator: PortalockerStorageCoordinator,
        engine: USearchEngine,
        content: str = FRESH,
    ) -> None:
        self.holding = threading.Event()
        self.go = threading.Event()
        self.published = threading.Event()
        self.release = threading.Event()
        self._coordinator = coordinator
        self._engine = engine
        self._content = content
        self.task = Task(WRITER, self._body)

    def _body(self) -> None:
        with self._coordinator.acquire(WS, LeaseMode.EXCLUSIVE):
            self.holding.set()
            assert self.go.wait(FAILURE_BOUND)
            self._engine.add(WS, self._content, infer=False)
            self._engine.commit()
            self.published.set()
            assert self.release.wait(FAILURE_BOUND)


@dataclass
class Probe:
    """What the spies saw about the reader's lease and engine locks."""

    acquires: list[tuple[str, LeaseMode, bool]] = field(default_factory=list)
    restore_held: list[bool] = field(default_factory=list)
    verify_held: list[bool] = field(default_factory=list)
    reader_waiting: threading.Event = field(default_factory=threading.Event)
    writer_waiting: threading.Event = field(default_factory=threading.Event)
    restore_seen: threading.Event = field(default_factory=threading.Event)

    def reader_modes(self) -> list[LeaseMode]:
        return [mode for name, mode, _ in self.acquires if name == READER]


def _install_probe(
    monkeypatch: pytest.MonkeyPatch,
    config: USearchConfig,
    coordinator: PortalockerStorageCoordinator,
    reader: USearchEngine,
    *,
    after_reader_restore: Callable[[], None] | None = None,
) -> Probe:
    probe = Probe()
    real_acquire = coordinator.acquire
    original_restore = Index.restore
    original_verify = USearchEngine._verify_candidate
    restored_once = threading.Event()

    def acquire(
        workspace_id: str,
        mode: LeaseMode = LeaseMode.EXCLUSIVE,
        *,
        timeout: float | None = None,
    ) -> IStorageLease:
        name = threading.current_thread().name
        probe.acquires.append((name, mode, reader._init_lock.locked()))
        if name == READER:
            probe.reader_waiting.set()
        if name == WRITER:
            probe.writer_waiting.set()
        return real_acquire(workspace_id, mode, timeout=timeout)

    def restore(path: str) -> Index | None:
        is_reader_open = (
            threading.current_thread().name == READER and path == config.index_path
        )
        if is_reader_open:
            probe.restore_seen.set()
            probe.restore_held.append(coordinator.is_held(WS))
        loaded = original_restore(path)
        if is_reader_open and after_reader_restore and not restored_once.is_set():
            restored_once.set()
            after_reader_restore()
        return loaded

    def verify(self: USearchEngine, candidate: Index) -> None:
        if self is reader and threading.current_thread().name == READER:
            probe.verify_held.append(coordinator.is_held(WS))
        original_verify(self, candidate)

    monkeypatch.setattr(coordinator, "acquire", acquire)
    monkeypatch.setattr(Index, "restore", restore)
    monkeypatch.setattr(USearchEngine, "_verify_candidate", verify)
    return probe


class TestOpenIsFenced:
    def test_open_waits_for_a_publishing_writer_and_sees_the_published_pair(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        writer_engine = _engine(config, coordinator)
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        writer = Writer(coordinator, writer_engine)
        opening = Task(READER, reader.ensure_initialized)
        try:
            writer.task.start()
            assert writer.holding.wait(FAILURE_BOUND)

            opening.start()
            assert probe.reader_waiting.wait(FAILURE_BOUND)
            opening.join(STILL_BLOCKED_PROBE)
            assert opening.alive
            assert not probe.restore_seen.is_set()

            writer.go.set()
            assert writer.published.wait(FAILURE_BOUND)
            writer.release.set()
            writer.task.join()
            opening.join()

            assert writer.task.error is None
            assert not opening.alive
            assert opening.error is None
            assert len(reader.index) == len(SEED) + 1
            assert reader.count(WS) == len(SEED) + 1
            assert reader._seen_identity == _file_identity(config)
            assert probe.reader_modes() == [LeaseMode.SHARED]
        finally:
            writer.go.set()
            writer.release.set()
            reader.close()
            writer_engine.close()

    def test_lease_is_taken_before_the_init_lock(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        opening = Task(READER, reader.ensure_initialized)
        try:
            opening.start()
            opening.join()

            assert opening.error is None
            assert probe.acquires == [(READER, LeaseMode.SHARED, False)]
        finally:
            reader.close()

    def test_writer_cannot_publish_between_restore_and_identity_capture(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        writer_engine = _engine(config, coordinator)
        writer = Writer(coordinator, writer_engine)
        writer.go.set()
        writer.release.set()
        blocked: list[bool] = []
        holder: dict[str, Probe] = {}

        def start_a_competing_writer() -> None:
            writer.task.start()
            assert holder["probe"].writer_waiting.wait(FAILURE_BOUND)
            writer.task.join(STILL_BLOCKED_PROBE)
            blocked.append(writer.task.alive and not writer.published.is_set())

        holder["probe"] = _install_probe(
            monkeypatch,
            config,
            coordinator,
            reader,
            after_reader_restore=start_a_competing_writer,
        )
        old_identity = _file_identity(config)
        opening = Task(READER, reader.ensure_initialized)
        try:
            opening.start()
            opening.join()
            writer.task.join()

            assert opening.error is None
            assert writer.task.error is None
            assert blocked == [True]
            assert reader._seen_identity == old_identity
            assert len(reader.index) == len(SEED)
            assert _file_identity(config) != old_identity

            reader.refresh()

            assert len(reader.index) == len(SEED) + 1
            assert reader._seen_identity == _file_identity(config)
        finally:
            writer.release.set()
            reader.close()
            writer_engine.close()


class TestReloadIsFenced:
    def test_refresh_waits_for_a_writer_that_already_published(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        writer_engine = _engine(config, coordinator)
        accepted = reader.index
        old_identity = reader._seen_identity
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        writer = Writer(coordinator, writer_engine)
        refreshing = Task(READER, reader.refresh)
        try:
            writer.task.start()
            assert writer.holding.wait(FAILURE_BOUND)
            writer.go.set()
            assert writer.published.wait(FAILURE_BOUND)
            assert _file_identity(config) != old_identity

            refreshing.start()
            assert probe.reader_waiting.wait(FAILURE_BOUND)
            refreshing.join(STILL_BLOCKED_PROBE)
            assert refreshing.alive
            assert reader._index is accepted
            assert reader._seen_identity == old_identity

            writer.release.set()
            writer.task.join()
            refreshing.join()

            assert writer.task.error is None
            assert refreshing.error is None
            assert reader._index is not accepted
            assert len(reader.index) == len(SEED) + 1
            assert reader._seen_identity == _file_identity(config)
            assert reader._seen_identity != old_identity
            assert probe.reader_modes() == [LeaseMode.SHARED]
        finally:
            writer.go.set()
            writer.release.set()
            reader.close()
            writer_engine.close()

    def test_refresh_of_an_unchanged_file_takes_no_lease(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        _ = reader.index
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        try:
            refreshing = Task(READER, reader.refresh)
            refreshing.start()
            refreshing.join()

            assert refreshing.error is None
            assert probe.acquires == []
        finally:
            reader.close()


class TestRestoreAndVerifyRunUnderTheLease:
    def test_open_restores_and_verifies_while_holding_the_lease(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        try:
            opening = Task(READER, reader.ensure_initialized)
            opening.start()
            opening.join()

            assert opening.error is None
            assert probe.restore_held == [True]
            assert probe.verify_held == [True]
            assert not coordinator.is_held(WS)
        finally:
            reader.close()

    def test_reload_restores_and_verifies_while_holding_the_lease(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        writer_engine = _engine(config, coordinator)
        _ = reader.index
        writer_engine.add(WS, FRESH, infer=False)
        writer_engine.commit()
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        try:
            refreshing = Task(READER, reader.refresh)
            refreshing.start()
            refreshing.join()

            assert refreshing.error is None
            assert probe.restore_held == [True]
            assert probe.verify_held == [True]
            assert probe.acquires == [(READER, LeaseMode.SHARED, False)]
            assert len(reader.index) == len(SEED) + 1
        finally:
            reader.close()
            writer_engine.close()

    def test_verify_index_integrity_compares_under_the_lease(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        _ = reader.index
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        try:
            verifying = Task(READER, reader.verify_index_integrity)
            verifying.start()
            verifying.join()

            assert verifying.error is None
            assert probe.verify_held == [True]
            assert probe.acquires == [(READER, LeaseMode.SHARED, False)]
        finally:
            reader.close()


class TestHolderOfTheLeaseNeverReacquires:
    def test_exclusive_holder_opens_and_reloads_without_a_nested_acquire(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        writer_engine = _engine(config, coordinator)
        probe = _install_probe(monkeypatch, config, coordinator, reader)
        sizes: list[int] = []

        def hold_open_and_reload() -> None:
            with coordinator.acquire(WS, LeaseMode.EXCLUSIVE):
                reader.ensure_initialized()
                sizes.append(len(reader.index))
                writer_engine.add(WS, FRESH, infer=False)
                writer_engine.commit()
                reader.refresh()
                reader.verify_index_integrity()
                sizes.append(len(reader.index))

        task = Task(READER, hold_open_and_reload)
        try:
            task.start()
            task.join()

            assert not task.alive
            assert task.error is None
            assert sizes == [len(SEED), len(SEED) + 1]
            assert probe.reader_modes() == [LeaseMode.EXCLUSIVE]
            # restores: reader open, the writer's own open, reader reload
            assert probe.restore_held == [True, True, True]
            # verifies: open, reload, then the explicit re-verification
            assert probe.verify_held == [True, True, True]
            assert reader._seen_identity == _file_identity(config)
        finally:
            reader.close()
            writer_engine.close()

    def test_shared_holder_opens_without_a_nested_acquire(
        self,
        config: USearchConfig,
        coordinator: PortalockerStorageCoordinator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(config, coordinator)
        reader = _engine(config, coordinator)
        probe = _install_probe(monkeypatch, config, coordinator, reader)

        def hold_and_open() -> None:
            with coordinator.acquire(WS, LeaseMode.SHARED):
                reader.ensure_initialized()

        task = Task(READER, hold_and_open)
        try:
            task.start()
            task.join()

            assert task.error is None
            assert probe.reader_modes() == [LeaseMode.SHARED]
            assert probe.restore_held == [True]
        finally:
            reader.close()
