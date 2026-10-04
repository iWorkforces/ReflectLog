"""Process-death coverage for atomic USearch publication."""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory import replacement_recovery
from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import (
    Embeddings,
    ISemanticSearchEngine,
    ReplacementTransition,
)
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine
from tests.integration.test_memory_manager_usearch import (
    cleanup_manager,
    create_memory_manager,
    create_usearch_config,
)
from tests.integration.test_replacement_recovery import (
    NEW,
    _abandon_without_persist,
    _assert_sqlite_ids_equal_live_keys,
    _crash_after_insert,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_USEARCH_CONCURRENCY_TESTS") != "1",
    reason="Set RUN_USEARCH_CONCURRENCY_TESTS=1 to run USearch concurrency tests",
)


def _reopen_first_add(
    config: Config, barrier: multiprocessing.synchronize.Barrier
) -> None:
    original = replacement_recovery._precompute_add_vectors

    def synchronized(
        pending: list[ReplacementTransition],
        semantic_engine: ISemanticSearchEngine,
        logger: IStructuredLogger,
    ) -> dict[str, list[float]]:
        vectors = original(pending, semantic_engine, logger)
        assert isinstance(semantic_engine, USearchEngine)
        assert semantic_engine._unpublished_bootstrap
        assert len(semantic_engine.index) == 0
        assert not Path(semantic_engine.config.index_path).exists()
        barrier.wait(timeout=30.0)
        return vectors

    with patch.object(replacement_recovery, "_precompute_add_vectors", synchronized):
        manager, _ = create_memory_manager(config)
    try:
        assert manager.get_all() == [NEW]
        assert manager.pending_intent_count() == 0
        engine = manager._semantic_engine
        assert isinstance(engine, USearchEngine)
        assert not engine._unpublished_bootstrap
        engine.verify_index_integrity()
    finally:
        manager.close()


@pytest.mark.integration
def test_two_recoverers_publish_first_add_once(tmp_path: Path) -> None:
    config = create_usearch_config(str(tmp_path))
    manager, _ = create_memory_manager(config)
    generation = manager._coordinator.read_generation(manager.workspace_id)
    with _crash_after_insert(manager), pytest.raises(Exception):
        manager.add_memories([NEW])
    _abandon_without_persist(manager)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    children = [
        ctx.Process(target=_reopen_first_add, args=(config, barrier)) for _ in range(2)
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(timeout=45.0)
            assert child.exitcode == 0
        inspector, _ = create_memory_manager(config)
        try:
            _assert_sqlite_ids_equal_live_keys(inspector)
            assert inspector.get_all() == [NEW]
            assert inspector.pending_intent_count() == 0
            assert (
                inspector._coordinator.read_generation(inspector.workspace_id)
                == generation + 1
            )
            fulltext = inspector._tantivy_engine
            assert fulltext is not None
            assert fulltext.find_by_exact_match(inspector.workspace_id, NEW) == [NEW]
            assert (
                inspector._semantic_engine.memory_store.list_pending_transitions() == []
            )
        finally:
            cleanup_manager(inspector)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(timeout=5.0)


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


def _writer_die_at(
    index_path: str,
    db_path: str,
    workspace_id: str,
    step: str,
    ready: multiprocessing.synchronize.Event,
) -> None:
    def boom(name: str) -> None:
        if name == step:
            os._exit(20)

    config = USearchConfig(
        workspace_id=workspace_id,
        index_path=index_path,
        db_path=db_path,
        embedding_dims=32,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/hash-32",
    )
    first = USearchEngine(config=config, embedder=_HashEmbedder())
    first.add(workspace_id, "kept", infer=False)
    first.commit()
    first.close()
    ready.set()
    second = USearchEngine(config=config, embedder=_HashEmbedder(), publish_hook=boom)
    # MemoryManager journals a pending ADD before it inserts the row.
    store = MemoryStore(db_path=db_path)
    try:
        _ = store.begin_add_intents(workspace_id, ["new-row"])
    finally:
        store.close()
    second.add(workspace_id, "new-row", infer=False)
    second.commit()


@pytest.mark.integration
@pytest.mark.parametrize(
    "step",
    [
        "before_save",
        "after_temp_save",
        "after_temp_validate",
        "after_fsync",
        "before_replace",
    ],
)
def test_kill_at_publish_failpoint_keeps_valid_index(tmp_path: Path, step: str) -> None:
    index_path = str(tmp_path / "vectors.usearch")
    db_path = str(tmp_path / "memories.db")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(
        target=_writer_die_at,
        args=(index_path, db_path, "ws", step, ready),
    )
    child.start()
    try:
        assert ready.wait(timeout=20.0)
        child.join(timeout=30.0)
        assert child.exitcode == 20
        inspector = USearchEngine(
            config=USearchConfig(
                workspace_id="ws",
                index_path=index_path,
                db_path=db_path,
                embedding_dims=32,
                embedder_provider=EmbedderProvider.OPENAI,
                embedding_model="test/hash-32",
            ),
            embedder=_HashEmbedder(),
        )
        try:
            _ = inspector.index
            temps = [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]
            assert temps == []
            rows = inspector.get_all("ws")
            assert "kept" in rows
            assert len(inspector.index) == 1
        finally:
            inspector.close()
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5.0)
