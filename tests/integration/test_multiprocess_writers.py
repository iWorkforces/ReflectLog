"""Real-process writer coordination for disjoint and duplicate adds."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest

from reflectlog.core.types import Embeddings
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_USEARCH_CONCURRENCY_TESTS") != "1",
    reason="Set RUN_USEARCH_CONCURRENCY_TESTS=1 to run USearch concurrency tests",
)


class _HashEmbedder(Embeddings):
    def __init__(self, dims: int = 32) -> None:
        super().__init__()
        self.dims = dims

    def embed_query(self, text: str) -> list[float]:
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dims).astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _writer(
    root: str,
    workspace_id: str,
    contents: list[str],
    ready: multiprocessing.synchronize.Event,
    go: multiprocessing.synchronize.Event,
    result: multiprocessing.Queue[int],
) -> None:
    coordinator = PortalockerStorageCoordinator(root, timeout=30.0)
    config = USearchConfig(
        workspace_id=workspace_id,
        index_path=os.path.join(root, workspace_id, "usearch", "vectors.usearch"),
        db_path=os.path.join(root, workspace_id, "usearch", "memories.db"),
        embedding_dims=32,
    )
    engine = USearchEngine(
        config=config, embedder=_HashEmbedder(), coordinator=coordinator
    )
    ready.set()
    if not go.wait(timeout=30.0):
        result.put(-1)
        engine.close()
        return
    added = engine.add_batch(workspace_id, contents, infer=False)
    engine.commit()
    result.put(len(added))
    engine.close()


@pytest.mark.integration
def test_disjoint_and_duplicate_multiprocess_writes(tmp_path: Path) -> None:
    root = str(tmp_path / "indexes")
    workspace_id = "ws"
    ctx = multiprocessing.get_context("spawn")
    ready_events = [ctx.Event() for _ in range(4)]
    go = ctx.Event()
    queue: multiprocessing.Queue[int] = ctx.Queue()
    payloads = [
        [f"w0-{i}" for i in range(200)] + ["shared-dup"],
        [f"w1-{i}" for i in range(200)] + ["shared-dup"],
        [f"w2-{i}" for i in range(200)] + ["shared-dup"],
        [f"w3-{i}" for i in range(200)] + ["shared-dup"],
    ]
    processes = [
        ctx.Process(
            target=_writer,
            args=(root, workspace_id, payload, ready, go, queue),
        )
        for payload, ready in zip(payloads, ready_events, strict=True)
    ]
    for process in processes:
        process.start()
    try:
        assert all(ready.wait(timeout=20.0) for ready in ready_events)
        go.set()
        for process in processes:
            process.join(timeout=60.0)
            assert process.exitcode == 0
        counts = [queue.get(timeout=1.0) for _ in processes]
        assert all(count > 0 for count in counts)
        coordinator = PortalockerStorageCoordinator(root, timeout=5.0)
        config = USearchConfig(
            workspace_id=workspace_id,
            index_path=os.path.join(root, workspace_id, "usearch", "vectors.usearch"),
            db_path=os.path.join(root, workspace_id, "usearch", "memories.db"),
            embedding_dims=32,
        )
        inspector = USearchEngine(
            config=config, embedder=_HashEmbedder(), coordinator=coordinator
        )
        try:
            rows = inspector.get_all(workspace_id)
            assert len(rows) == 801
            assert rows.count("shared-dup") == 1
            assert inspector.get_id_by_content(workspace_id, "shared-dup") is not None
            assert len(inspector.index) == 801
        finally:
            inspector.close()
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
