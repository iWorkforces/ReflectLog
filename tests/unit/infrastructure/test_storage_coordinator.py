"""Unit tests for the Portalocker workspace storage coordinator."""

from __future__ import annotations

from pathlib import Path

import pytest

from reflectlog.core.exceptions import GenerationError, LeaseTimeoutError
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
