"""Process-scoped Tantivy reader/writer ownership."""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import os
from pathlib import Path

import pytest

from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.tantivy_engine import TantivyConfig, TantivyEngine


def _write_and_die(
    index_path: str,
    workspace_id: str,
    ready: multiprocessing.synchronize.Event,
) -> None:
    coordinator = PortalockerStorageCoordinator(
        os.path.dirname(index_path), timeout=5.0
    )
    engine = TantivyEngine(
        TantivyConfig(workspace_id=workspace_id, index_path=index_path),
        coordinator=coordinator,
    )
    engine.add(workspace_id, "survives")
    engine.commit()
    ready.set()
    os._exit(20)


@pytest.mark.integration
def test_writer_death_releases_and_content_survives(tmp_path: Path) -> None:
    index_path = str(tmp_path / "idx")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(target=_write_and_die, args=(index_path, "ws", ready))
    child.start()
    try:
        assert ready.wait(timeout=20.0)
        child.join(timeout=20.0)
        assert child.exitcode == 20
        coordinator = PortalockerStorageCoordinator(str(tmp_path), timeout=5.0)
        engine = TantivyEngine(
            TantivyConfig(workspace_id="ws", index_path=index_path),
            coordinator=coordinator,
        )
        try:
            engine.add("ws", "after-death")
            engine.commit()
            hits = {memory for memory, _score in engine.search("survives", "ws", 10)}
            assert "survives" in hits
            later = {memory for memory, _score in engine.search("after", "ws", 10)}
            assert "after-death" in later
        finally:
            engine.close()
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5.0)
