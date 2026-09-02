"""Coordinated shutdown handoff and native signal registration."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.core.exceptions import StorageError
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.server import _start_server


def test_lease_handoff_after_close() -> None:
    from reflectlog.application.config.settings import Config
    from reflectlog.application.memory.manager import MemoryManager
    from reflectlog.application.utils.logging import StructuredLogger
    from reflectlog.application.utils.security import SecretString
    from reflectlog.core.enums import LlmProvider, RerankerEngine
    import logging

    logger = StructuredLogger(logging.getLogger("shutdown-handoff"))
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = PortalockerStorageCoordinator(tmpdir, timeout=1.0)
        config = Config(
            workspace_id="ws",
            openrouter_api_key=SecretString("test"),
            tantivy_index_path_template=os.path.join(tmpdir, "{workspace_id}", "tantivy"),
            enable_hybrid_search=False,
            enable_smart_replace=False,
            llm_provider=LlmProvider.OPENAI,
            reranker_engine=RerankerEngine.NONE,
            embedding_cache_enabled=False,
            eager_initialization=False,
        )
        with (
            patch("reflectlog.application.memory.manager.USearchEngine") as usearch_cls,
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            usearch_cls.return_value = MagicMock()
            first = MemoryManager(config, logger, coordinator=coordinator)
            first.close()
            with pytest.raises(StorageError, match="closed"):
                first.get_all()
            second = MemoryManager(config, logger, coordinator=coordinator)
            assert second._coordinator is coordinator
            second.close()


def test_graceful_signal_registers_posix_handlers() -> None:
    registered: dict[int, object] = {}

    def _capture(signum: int, handler: object) -> None:
        registered[signum] = handler

    with (
        patch("reflectlog.server.signal.signal", side_effect=_capture),
        patch("reflectlog.server._server_cls") as server_cls,
    ):
        server_cls.return_value = lambda: MagicMock()
        _ = _start_server(sys.stderr, 0.0, {})
    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered
    if sys.platform == "win32":
        assert signal.SIGBREAK in registered


def test_second_signal_restores_default() -> None:
    registered: dict[int, object] = {}

    def _capture(signum: int, handler: object) -> object:
        registered[signum] = handler
        return None

    with (
        patch("reflectlog.server.signal.signal", side_effect=_capture),
        patch("reflectlog.server.signal.raise_signal") as raise_signal,
        patch("reflectlog.server._server_cls") as server_cls,
        patch("reflectlog.server.sys.exit"),
    ):
        server_cls.return_value = lambda: MagicMock()
        _ = _start_server(sys.stderr, 0.0, {})
        handler = registered[signal.SIGINT]
        assert callable(handler)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)
        raise_signal.assert_called_once_with(signal.SIGINT)
