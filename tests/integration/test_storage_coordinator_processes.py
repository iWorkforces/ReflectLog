"""Cross-process tests for Portalocker workspace coordination."""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
from pathlib import Path

import pytest

from reflectlog.core.exceptions import LeaseTimeoutError, StorageCoordinationError
from reflectlog.core.storage_coordination import LeaseMode
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


def _shared_upgrade_holder(
    root: str,
    events: tuple[
        multiprocessing.synchronize.Event,
        multiprocessing.synchronize.Event,
        multiprocessing.synchronize.Event,
    ],
    result: multiprocessing.Queue[tuple[str, bool, bool]],
) -> None:
    ready, upgrade, release = events
    coordinator = PortalockerStorageCoordinator(root, timeout=10.0)
    with coordinator.acquire("alpha", LeaseMode.SHARED):
        ready.set()
        if not upgrade.wait(timeout=15.0):
            return
        outcome = "granted"
        try:
            with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
                pass
        except StorageCoordinationError as error:
            outcome = type(error).__name__
        result.put(
            (
                outcome,
                coordinator.is_held("alpha", LeaseMode.SHARED),
                coordinator.is_held("alpha", LeaseMode.EXCLUSIVE),
            )
        )
        _ = release.wait(timeout=20.0)


def _exclusive_probe(
    root: str,
    commands: multiprocessing.Queue[str],
    result: multiprocessing.Queue[str],
) -> None:
    coordinator = PortalockerStorageCoordinator(root, timeout=0.5)
    while commands.get(timeout=20.0) != "stop":
        try:
            with coordinator.acquire("alpha", LeaseMode.EXCLUSIVE):
                result.put("acquired")
        except LeaseTimeoutError:
            result.put("blocked")


@pytest.mark.integration
def test_two_process_shared_upgrades_rejected_without_releasing(tmp_path: Path) -> None:
    root = str(tmp_path / "indexes")
    ctx = multiprocessing.get_context("spawn")
    events = [(ctx.Event(), ctx.Event(), ctx.Event()) for _ in range(2)]
    results: multiprocessing.Queue[tuple[str, bool, bool]] = ctx.Queue()
    commands: multiprocessing.Queue[str] = ctx.Queue()
    probe_results: multiprocessing.Queue[str] = ctx.Queue()
    holders = [
        ctx.Process(target=_shared_upgrade_holder, args=(root, group, results))
        for group in events
    ]
    probe = ctx.Process(target=_exclusive_probe, args=(root, commands, probe_results))
    processes = [*holders, probe]
    for process in processes:
        process.start()
    try:
        for ready, _, _ in events:
            assert ready.wait(timeout=10.0)
        for _, upgrade, _ in events:
            upgrade.set()
        rejected = [results.get(timeout=5.0), results.get(timeout=5.0)]
        assert rejected == [("LeaseUpgradeError", True, False)] * 2
        commands.put("probe")
        assert probe_results.get(timeout=5.0) == "blocked"
        events[0][2].set()
        holders[0].join(timeout=5.0)
        assert holders[0].exitcode == 0
        commands.put("probe")
        assert probe_results.get(timeout=5.0) == "blocked"
        events[1][2].set()
        holders[1].join(timeout=5.0)
        assert holders[1].exitcode == 0
        commands.put("probe")
        assert probe_results.get(timeout=5.0) == "acquired"
        commands.put("stop")
        probe.join(timeout=5.0)
        assert probe.exitcode == 0
    finally:
        for _, upgrade, release in events:
            upgrade.set()
            release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
        for queue in (results, commands, probe_results):
            queue.close()
            queue.join_thread()
