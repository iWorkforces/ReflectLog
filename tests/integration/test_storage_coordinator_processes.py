"""Cross-process tests for Portalocker workspace coordination."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from reflectlog.core.exceptions import LeaseTimeoutError
from reflectlog.infrastructure.storage_coordinator import (
    PortalockerStorageCoordinator,
)


def _hold_exclusive(
    root: str,
    workspace_id: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    coordinator = PortalockerStorageCoordinator(root, timeout=5.0)
    with coordinator.acquire(workspace_id):
        ready.set()
        _ = release.wait(timeout=30.0)


def _acquire_after_kill(
    root: str,
    workspace_id: str,
    result: multiprocessing.Queue[str],
) -> None:
    coordinator = PortalockerStorageCoordinator(root, timeout=5.0)
    with coordinator.acquire(workspace_id):
        coordinator.publish_generation(workspace_id, 7)
        result.put("acquired")


@pytest.mark.integration
def test_child_kill_releases_lease_without_deleting_lock(tmp_path: Path) -> None:
    root = str(tmp_path / "indexes")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    child = ctx.Process(
        target=_hold_exclusive,
        args=(root, "alpha", ready, release),
    )
    child.start()
    try:
        assert ready.wait(timeout=10.0)
        waiter = PortalockerStorageCoordinator(root, timeout=0.2)
        with pytest.raises(LeaseTimeoutError):
            with waiter.acquire("alpha"):
                raise AssertionError("child still holds the lease")
        lock_path = Path(waiter.paths_for("alpha").lock_path)
        assert lock_path.exists()
        child.kill()
        child.join(timeout=10.0)
        assert not child.is_alive()
        assert lock_path.exists()
        queue: multiprocessing.Queue[str] = ctx.Queue()
        inspector = ctx.Process(
            target=_acquire_after_kill,
            args=(root, "alpha", queue),
        )
        inspector.start()
        inspector.join(timeout=10.0)
        assert inspector.exitcode == 0
        assert queue.get(timeout=1.0) == "acquired"
        parent = PortalockerStorageCoordinator(root, timeout=0.2)
        assert parent.read_generation("alpha") == 7
        assert lock_path.exists()
    finally:
        if child.is_alive():
            release.set()
            child.join(timeout=5.0)
            if child.is_alive():
                child.kill()
                child.join(timeout=5.0)
