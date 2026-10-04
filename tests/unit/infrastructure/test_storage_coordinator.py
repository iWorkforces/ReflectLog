"""Unit tests for the Portalocker workspace storage coordinator."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from reflectlog.core.exceptions import (
    GenerationError,
    LeaseTimeoutError,
    LeaseUpgradeError,
    StorageCoordinationError,
)
from reflectlog.core.storage_coordination import LeaseMode
from reflectlog.infrastructure.storage_coordinator import (
    PortalockerStorageCoordinator,
)


@pytest.fixture
def coordinator(tmp_path: Path) -> PortalockerStorageCoordinator:
    return PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=0.3)


def test_missing_generation_is_zero(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    assert coordinator.read_generation("alpha") == 0


def test_publish_and_read_generation(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    coordinator.publish_generation("alpha", 3)
    assert coordinator.read_generation("alpha") == 3
    paths = coordinator.paths_for("alpha")
    assert Path(paths.generation_path).exists()


def test_corrupt_generation_fails_closed(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    paths = coordinator.paths_for("alpha")
    Path(paths.root).mkdir(parents=True, exist_ok=True)
    Path(paths.generation_path).write_text("not-an-int\n", encoding="utf-8")
    with pytest.raises(GenerationError, match="corrupt"):
        _ = coordinator.read_generation("alpha")


def test_empty_generation_fails_closed(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    paths = coordinator.paths_for("alpha")
    Path(paths.root).mkdir(parents=True, exist_ok=True)
    Path(paths.generation_path).write_text("   \n", encoding="utf-8")
    with pytest.raises(GenerationError, match="empty"):
        _ = coordinator.read_generation("alpha")


def test_exclusive_reentrancy(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE) as outer:
        with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE) as inner:
            assert inner.workspace_id == outer.workspace_id
        assert Path(coordinator.paths_for("alpha").lock_path).exists()
    assert Path(coordinator.paths_for("alpha").lock_path).exists()


def test_separate_workspaces_do_not_contend(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    with coordinator.acquire("alpha"):
        with coordinator.acquire("beta"):
            coordinator.publish_generation("beta", 1)
    assert coordinator.read_generation("beta") == 1


def test_same_workspace_contention_times_out(
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "indexes")
    owner = PortalockerStorageCoordinator(root, timeout=0.2)
    waiter = PortalockerStorageCoordinator(root, timeout=0.2)
    with owner.acquire("alpha"):
        with pytest.raises(LeaseTimeoutError, match="alpha"):
            with waiter.acquire("alpha"):
                raise AssertionError("waiter must not enter")


def test_exception_releases_lease(
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "indexes")
    first = PortalockerStorageCoordinator(root, timeout=0.2)
    second = PortalockerStorageCoordinator(root, timeout=0.2)
    with pytest.raises(RuntimeError, match="boom"):
        with first.acquire("alpha"):
            raise RuntimeError("boom")
    with second.acquire("alpha"):
        second.publish_generation("alpha", 2)
    assert second.read_generation("alpha") == 2


def test_lock_file_remains_after_release(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    with coordinator.acquire("alpha"):
        lock_path = Path(coordinator.paths_for("alpha").lock_path)
        assert lock_path.exists()
    assert lock_path.exists()


def test_concurrent_shared_leases_do_not_drop_os_lock(
    tmp_path: Path,
) -> None:
    import threading

    coordinator = PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=1.0)
    started = threading.Event()
    release = threading.Event()
    errors: list[str] = []

    def _hold_shared() -> None:
        try:
            with coordinator.acquire("alpha", LeaseMode.SHARED):
                started.set()
                _ = release.wait(timeout=2.0)
        except Exception as exc:
            errors.append(str(exc))

    holder = threading.Thread(target=_hold_shared)
    holder.start()
    assert started.wait(timeout=2.0)
    with coordinator.acquire("alpha", LeaseMode.SHARED):
        assert coordinator.is_held("alpha", LeaseMode.SHARED)
    release.set()
    holder.join(timeout=2.0)
    assert errors == []
    waiter = PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=0.2)
    with waiter.acquire("alpha"):
        waiter.publish_generation("alpha", 1)


def test_exclusive_waits_for_same_process_shared(
    tmp_path: Path,
) -> None:
    import threading

    coordinator = PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=1.0)
    started = threading.Event()
    release = threading.Event()

    def _hold_shared() -> None:
        with coordinator.acquire("alpha", LeaseMode.SHARED):
            started.set()
            _ = release.wait(timeout=2.0)

    holder = threading.Thread(target=_hold_shared)
    holder.start()
    assert started.wait(timeout=2.0)
    acquired = threading.Event()

    def _take_exclusive() -> None:
        with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
            acquired.set()

    waiter = threading.Thread(target=_take_exclusive)
    waiter.start()
    waiter.join(timeout=0.2)
    assert not acquired.is_set()
    release.set()
    waiter.join(timeout=2.0)
    holder.join(timeout=2.0)
    assert acquired.is_set()


def test_exclusive_waits_for_other_thread_exclusive(
    tmp_path: Path,
) -> None:
    import threading

    coordinator = PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=1.0)
    started = threading.Event()
    release = threading.Event()

    def _hold_exclusive() -> None:
        with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
            started.set()
            _ = release.wait(timeout=2.0)

    holder = threading.Thread(target=_hold_exclusive)
    holder.start()
    assert started.wait(timeout=2.0)
    acquired = threading.Event()

    def _take_exclusive() -> None:
        with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
            acquired.set()

    waiter = threading.Thread(target=_take_exclusive)
    waiter.start()
    waiter.join(timeout=0.2)
    assert not acquired.is_set()
    release.set()
    waiter.join(timeout=2.0)
    holder.join(timeout=2.0)
    assert acquired.is_set()


def test_shared_waits_for_other_thread_exclusive(
    tmp_path: Path,
) -> None:
    import threading

    coordinator = PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=1.0)
    started = threading.Event()
    release = threading.Event()

    def _hold_exclusive() -> None:
        with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
            started.set()
            _ = release.wait(timeout=2.0)

    holder = threading.Thread(target=_hold_exclusive)
    holder.start()
    assert started.wait(timeout=2.0)
    acquired = threading.Event()

    def _take_shared() -> None:
        with coordinator.acquire("alpha", LeaseMode.SHARED):
            acquired.set()

    waiter = threading.Thread(target=_take_shared)
    waiter.start()
    waiter.join(timeout=0.2)
    assert not acquired.is_set()
    assert not coordinator.is_held("alpha", LeaseMode.EXCLUSIVE)
    release.set()
    waiter.join(timeout=2.0)
    holder.join(timeout=2.0)
    assert acquired.is_set()


def test_same_thread_shared_reuses_exclusive(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
        assert coordinator.is_held("alpha", LeaseMode.EXCLUSIVE)
        with coordinator.acquire("alpha", LeaseMode.SHARED):
            assert coordinator.is_held("alpha", LeaseMode.SHARED)


def test_sidecar_paths_are_stable(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    paths = coordinator.paths_for("My.Workspace")
    assert paths.lock_path.endswith(
        str(Path("my.workspace") / ".reflectlog.writer.lock")
    )
    assert paths.generation_path.endswith(
        str(Path("my.workspace") / ".reflectlog.storage-generation")
    )


def test_shared_upgrade_rejected_and_shared_retained(tmp_path: Path) -> None:
    root = str(tmp_path / "indexes")
    coordinator = PortalockerStorageCoordinator(root, timeout=5.0)
    with coordinator.acquire("alpha", LeaseMode.SHARED):
        state = coordinator._states["alpha"]
        before = (
            state.exclusive_depth,
            state.exclusive_owner,
            dict(state.shared_holders),
            state.os_lock,
        )
        started = time.monotonic()
        with pytest.raises(LeaseUpgradeError) as error:
            with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
                pytest.fail("upgrade granted")
        assert type(error.value).__name__ == "LeaseUpgradeError"
        assert time.monotonic() - started < 2.0
        assert coordinator.is_held("alpha", LeaseMode.SHARED)
        assert not coordinator.is_held("alpha", LeaseMode.EXCLUSIVE)
        assert before == (
            state.exclusive_depth,
            state.exclusive_owner,
            dict(state.shared_holders),
            state.os_lock,
        )
        with coordinator.acquire("alpha", LeaseMode.SHARED):
            assert coordinator.is_held("alpha", LeaseMode.SHARED)
    with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE, timeout=0.2):
        assert coordinator.is_held("alpha", LeaseMode.EXCLUSIVE)
    other = PortalockerStorageCoordinator(root, timeout=0.2)
    with other.acquire("alpha", LeaseMode.EXCLUSIVE):
        assert other.is_held("alpha", LeaseMode.EXCLUSIVE)


@pytest.mark.parametrize(
    "modes",
    [
        (LeaseMode.EXCLUSIVE, LeaseMode.EXCLUSIVE),
        (LeaseMode.EXCLUSIVE, LeaseMode.SHARED),
        (LeaseMode.SHARED, LeaseMode.SHARED),
        (LeaseMode.EXCLUSIVE, LeaseMode.SHARED, LeaseMode.EXCLUSIVE),
    ],
)
def test_legal_lease_nesting(
    coordinator: PortalockerStorageCoordinator, modes: tuple[LeaseMode, ...]
) -> None:
    from contextlib import ExitStack

    with ExitStack() as stack:
        for mode in modes:
            stack.enter_context(coordinator.acquire("alpha", mode))
            assert coordinator.is_held("alpha", mode)
    assert not coordinator.is_held("alpha")


def test_non_lifo_shared_holder_cannot_regain_exclusive(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    exclusive = coordinator.acquire("alpha", LeaseMode.EXCLUSIVE)
    shared = coordinator.acquire("alpha", LeaseMode.SHARED)
    exclusive.release()
    try:
        with pytest.raises(LeaseUpgradeError) as error:
            with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
                pytest.fail("upgrade granted")
        assert type(error.value).__name__ == "LeaseUpgradeError"
        assert coordinator.is_held("alpha", LeaseMode.SHARED)
        assert not coordinator.is_held("alpha", LeaseMode.EXCLUSIVE)
    finally:
        shared.release()
    with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
        assert coordinator.is_held("alpha", LeaseMode.EXCLUSIVE)


def test_other_thread_exclusive_still_times_out_under_shared(
    coordinator: PortalockerStorageCoordinator,
) -> None:
    errors: list[StorageCoordinationError] = []

    def request() -> None:
        try:
            with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
                pass
        except StorageCoordinationError as error:
            errors.append(error)

    with coordinator.acquire("alpha", LeaseMode.SHARED):
        waiter = threading.Thread(target=request)
        waiter.start()
        waiter.join(timeout=2.0)
        assert not waiter.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], LeaseTimeoutError)
