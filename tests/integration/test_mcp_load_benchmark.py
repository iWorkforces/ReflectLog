from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import anyio
from fastmcp import Client
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import EmbedderProvider, RerankerEngine
from reflectlog.infrastructure.embeddings.qwen3_embedding import LangchainQwenEmbeddings
from tests.load.locustfile import run_benchmark


@pytest.fixture
def deterministic_embeddings() -> Iterator[None]:
    def vector(_embedder: LangchainQwenEmbeddings, _text: str) -> list[float]:
        return [1.0, *([0.0] * 127)]

    def vectors(
        embedder: LangchainQwenEmbeddings, texts: list[str]
    ) -> list[list[float]]:
        return [vector(embedder, text) for text in texts]

    async def async_vectors(
        embedder: LangchainQwenEmbeddings, texts: list[str]
    ) -> list[list[float]]:
        return vectors(embedder, texts)

    with (
        patch.object(LangchainQwenEmbeddings, "embed_documents", vectors),
        patch.object(LangchainQwenEmbeddings, "aembed_documents", async_vectors),
        patch.object(LangchainQwenEmbeddings, "embed_query", vector),
    ):
        yield


@pytest.fixture
def server_config(tmp_path: Path, set_env_vars: None) -> Config:
    return Config(
        workspace_id="",
        openrouter_api_key=SecretString("test-key"),
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="openai/text-embedding-3-small",
        embedding_dims=128,
        reranker_engine=RerankerEngine.NONE,
        enable_smart_replace=False,
        embedding_cache_enabled=False,
        eager_initialization=False,
        tantivy_index_path_template=str(tmp_path / "{workspace_id}" / "tantivy"),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_measures_real_mcp_tools_without_setup_calls(
    server_config: Config, deterministic_embeddings: None
) -> None:
    workspace_id = "benchmark-smoke"
    unrelated = "unrelated pre-existing memory"
    seed_server = FastMCPServer(server_config)
    try:
        async with Client(seed_server.mcp) as client:
            await client.call_tool(
                "add", {"workspace_id": workspace_id, "memories": [unrelated]}
            )
    finally:
        await seed_server.aclose()
    server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(server.mcp, workspace_id, clients=2, iterations=2)
        reopened = FastMCPServer(server_config)
        try:
            async with Client(reopened.mcp) as client:
                page = await client.call_tool("get_all", {"workspace_id": workspace_id})
        finally:
            await reopened.aclose()
        assert report["workspace_id"] == workspace_id
        assert len(report["connection_ms"]) == 2
        assert len(report["warmup_ms"]) == 2
        assert report["measured_seconds"] > 0
        assert report["requests_per_second"] > 0
        assert report["cleanup_errors"] == 0
        assert set(report["tools"]) == {"add", "search", "get_all", "health_check"}
        assert all(
            metric["count"] == 4
            and metric["errors"] == 0
            and metric["mean_ms"] > 0
            and 0 < metric["p50_ms"] <= metric["p95_ms"]
            and metric["p95_ms"] > 0
            for metric in report["tools"].values()
        )
        assert page.structured_content is not None
        assert page.structured_content["total"] == 1
        memories = page.structured_content["memories"]
        assert isinstance(memories, list)
        assert memories == [unrelated]
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_keeps_clients_open_until_every_cleanup_finishes(
    server_config: Config, deterministic_embeddings: None
) -> None:
    fast_removed = anyio.Event()
    release_slow = anyio.Event()
    client_closed = anyio.Event()
    original_exit = Client.__aexit__
    original_remove = FastMCPServer._remove

    async def remove(
        server: FastMCPServer, memories: list[str], workspace_id: str
    ) -> None:
        if "worker 0 seed" in memories[0]:
            fast_removed.set()
        else:
            await fast_removed.wait()
            await release_slow.wait()
        await original_remove(server, memories, workspace_id)

    async def tracked_exit(
        client: Client,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            await original_exit(client, exc_type, exc_value, traceback)
        finally:
            client_closed.set()

    async def observe_early_exit() -> None:
        nonlocal premature_exit
        await fast_removed.wait()
        with anyio.move_on_after(0.2):
            await client_closed.wait()
        premature_exit = client_closed.is_set()
        release_slow.set()

    premature_exit = False
    report = None
    with patch.object(FastMCPServer, "_remove", remove):
        server = FastMCPServer(server_config)
    try:
        with patch.object(Client, "__aexit__", tracked_exit):
            async with anyio.create_task_group() as group:
                group.start_soon(observe_early_exit)
                report = await run_benchmark(
                    server.mcp, "benchmark-lifecycle", clients=2, iterations=1
                )
        assert report is not None
        assert not premature_exit
        assert report["cleanup_errors"] == 0
        assert all(metric["errors"] == 0 for metric in report["tools"].values())
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_counts_mcp_tool_errors(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def failed_health(
        _server: FastMCPServer, workspace_id: str
    ) -> dict[str, object]:
        raise RuntimeError("health unavailable")

    with patch.object(FastMCPServer, "_health_check", failed_health):
        server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(
            server.mcp, "benchmark-errors", clients=1, iterations=1
        )
        assert report["tools"]["health_check"]["count"] == 1
        assert report["tools"]["health_check"]["errors"] == 1
        assert report["cleanup_errors"] == 0
        assert all(
            report["tools"][name]["errors"] == 0
            for name in ("add", "search", "get_all")
        )
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_counts_timed_out_tool_and_cleans_up(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def slow_health(
        _server: FastMCPServer, workspace_id: str
    ) -> dict[str, object]:
        await anyio.sleep(1)
        return {"workspace_id": workspace_id}

    with patch.object(FastMCPServer, "_health_check", slow_health):
        server = FastMCPServer(server_config)
    workspace_id = "benchmark-timeout"
    try:
        report = await run_benchmark(
            server.mcp, workspace_id, clients=1, iterations=1, timeout_seconds=0.2
        )
        reopened = FastMCPServer(server_config)
        try:
            async with Client(reopened.mcp) as client:
                page = await client.call_tool("get_all", {"workspace_id": workspace_id})
        finally:
            await reopened.aclose()
        assert report["tools"]["health_check"]["errors"] == 1
        assert report["tools"]["health_check"]["count"] == 1
        assert report["cleanup_errors"] == 0
        assert page.structured_content is not None
        assert page.structured_content["total"] == 0
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_reports_cleanup_tool_error(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def failed_remove(
        _server: FastMCPServer, memories: list[str], workspace_id: str
    ) -> None:
        raise RuntimeError("remove unavailable")

    with patch.object(FastMCPServer, "_remove", failed_remove):
        server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(
            server.mcp, "benchmark-cleanup-error", clients=1, iterations=1
        )
        assert report["cleanup_errors"] == 1
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_bounds_cleanup_tool_call(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def slow_remove(
        _server: FastMCPServer, memories: list[str], workspace_id: str
    ) -> None:
        await anyio.sleep(1)

    with patch.object(FastMCPServer, "_remove", slow_remove):
        server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(
            server.mcp,
            "benchmark-slow-cleanup",
            clients=1,
            iterations=1,
            timeout_seconds=0.3,
        )
        assert report["cleanup_errors"] == 1
        assert report["measured_seconds"] < 0.3
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_cleans_seed_after_warmup_error(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def slow_search(
        _server: FastMCPServer, query: str, workspace_id: str
    ) -> list[str]:
        await anyio.sleep(1)
        return []

    with patch.object(FastMCPServer, "_search", slow_search):
        server = FastMCPServer(server_config)
    workspace_id = "benchmark-warmup-timeout"
    try:
        with pytest.raises(ExceptionGroup) as failure:
            await run_benchmark(
                server.mcp,
                workspace_id,
                clients=1,
                iterations=1,
                timeout_seconds=0.3,
            )
        assert any(
            isinstance(error, TimeoutError) for error in failure.value.exceptions
        )
        reopened = FastMCPServer(server_config)
        try:
            async with Client(reopened.mcp) as client:
                page = await client.call_tool("get_all", {"workspace_id": workspace_id})
        finally:
            await reopened.aclose()
        assert page.structured_content is not None
        assert page.structured_content["total"] == 0
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_reports_cleanup_error_when_warmup_fails(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def failed_search(
        _server: FastMCPServer, query: str, workspace_id: str
    ) -> list[str]:
        raise RuntimeError("search unavailable")

    async def failed_remove(
        _server: FastMCPServer, memories: list[str], workspace_id: str
    ) -> None:
        raise RuntimeError("remove unavailable")

    with (
        patch.object(FastMCPServer, "_search", failed_search),
        patch.object(FastMCPServer, "_remove", failed_remove),
    ):
        server = FastMCPServer(server_config)
    try:
        with pytest.raises(ExceptionGroup) as failure:
            await run_benchmark(
                server.mcp, "benchmark-warmup-cleanup-error", clients=2, iterations=1
            )
        assert failure.value.__notes__ == ["benchmark cleanup_errors=2"]
        assert any(
            isinstance(error, RuntimeError) and "warmup search" in str(error)
            for error in failure.value.exceptions
        )
    finally:
        await server.aclose()
