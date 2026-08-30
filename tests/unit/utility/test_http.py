"""Unit tests for HTTP client factory module.

Tests reflectlog.utility.http HttpClientFactory and convenience functions.
"""

import aiohttp
import httpx
import pytest

from reflectlog.utility.http import (
    HttpClientFactory,
    get_pooled_aiohttp_client,
    get_pooled_async_httpx_client,
    get_pooled_httpx_client,
)


@pytest.fixture(autouse=True)
async def reset_singletons():
    """Reset all singleton clients before and after each test."""
    HttpClientFactory._httpx_client = None
    HttpClientFactory._async_httpx_client = None
    HttpClientFactory._aiohttp_client = None

    yield

    if HttpClientFactory._httpx_client is not None:
        HttpClientFactory._httpx_client.close()
        HttpClientFactory._httpx_client = None
    if HttpClientFactory._async_httpx_client is not None:
        await HttpClientFactory._async_httpx_client.aclose()
        HttpClientFactory._async_httpx_client = None
    if HttpClientFactory._aiohttp_client is not None:
        await HttpClientFactory._aiohttp_client.close()
        HttpClientFactory._aiohttp_client = None


class TestHttpFactoryInterface:
    """Test HttpClientFactory exposes expected interface."""

    def test_public_classmethods_exist(self) -> None:
        """Factory has expected public classmethods."""
        expected = {
            "get_httpx_client",
            "get_async_httpx_client",
            "get_aiohttp_client",
            "get_max_connections",
            "get_max_keepalive_connections",
            "get_connect_timeout",
            "get_read_timeout",
            "get_write_timeout",
            "get_httpx_limits",
            "get_httpx_timeout",
            "close_all",
            "close_all_sync",
        }
        for method_name in expected:
            assert hasattr(HttpClientFactory, method_name), (
                f"Missing method: {method_name}"
            )

    def test_class_attributes_exist(self) -> None:
        """Factory defines expected class-level attributes."""
        for attr in (
            "_httpx_client",
            "_async_httpx_client",
            "_aiohttp_client",
            "_create_lock",
        ):
            assert hasattr(HttpClientFactory, attr), f"Missing attribute: {attr}"

    def test_convenience_functions_callable(self) -> None:
        """Module-level convenience functions are callable."""
        assert callable(get_pooled_httpx_client)
        assert callable(get_pooled_async_httpx_client)
        assert callable(get_pooled_aiohttp_client)


class TestDefaultConfiguration:
    """Test default configuration values."""

    def test_default_max_connections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default max_connections is 100."""
        monkeypatch.delenv("HTTP_MAX_CONNECTIONS", raising=False)
        assert HttpClientFactory.get_max_connections() == 100

    def test_default_max_keepalive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default max_keepalive_connections is 20."""
        monkeypatch.delenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", raising=False)
        assert HttpClientFactory.get_max_keepalive_connections() == 20

    def test_default_connect_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default connect timeout is 30.0."""
        monkeypatch.delenv("HTTP_CONNECT_TIMEOUT", raising=False)
        assert HttpClientFactory.get_connect_timeout() == 30.0

    def test_default_read_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default read timeout is 30.0."""
        monkeypatch.delenv("HTTP_READ_TIMEOUT", raising=False)
        assert HttpClientFactory.get_read_timeout() == 30.0

    def test_default_write_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default write timeout is 30.0."""
        monkeypatch.delenv("HTTP_WRITE_TIMEOUT", raising=False)
        assert HttpClientFactory.get_write_timeout() == 30.0


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_env_max_connections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP_MAX_CONNECTIONS env var overrides default."""
        monkeypatch.setenv("HTTP_MAX_CONNECTIONS", "200")
        assert HttpClientFactory.get_max_connections() == 200

    def test_env_max_keepalive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP_MAX_KEEPALIVE_CONNECTIONS env var overrides default."""
        monkeypatch.setenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "50")
        assert HttpClientFactory.get_max_keepalive_connections() == 50

    def test_env_connect_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP_CONNECT_TIMEOUT env var overrides default."""
        monkeypatch.setenv("HTTP_CONNECT_TIMEOUT", "10.0")
        assert HttpClientFactory.get_connect_timeout() == 10.0

    def test_env_read_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP_READ_TIMEOUT env var overrides default."""
        monkeypatch.setenv("HTTP_READ_TIMEOUT", "15.0")
        assert HttpClientFactory.get_read_timeout() == 15.0

    def test_env_write_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP_WRITE_TIMEOUT env var overrides default."""
        monkeypatch.setenv("HTTP_WRITE_TIMEOUT", "45.0")
        assert HttpClientFactory.get_write_timeout() == 45.0


class TestHttpxLimitsAndTimeout:
    """Test httpx Limits and Timeout construction."""

    def test_get_httpx_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_httpx_limits returns configured Limits object."""
        monkeypatch.setenv("HTTP_MAX_CONNECTIONS", "150")
        monkeypatch.setenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "30")
        limits = HttpClientFactory.get_httpx_limits()
        assert isinstance(limits, httpx.Limits)
        assert limits.max_connections == 150
        assert limits.max_keepalive_connections == 30

    def test_get_httpx_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_httpx_timeout returns configured Timeout object."""
        monkeypatch.setenv("HTTP_CONNECT_TIMEOUT", "5.0")
        monkeypatch.setenv("HTTP_READ_TIMEOUT", "10.0")
        monkeypatch.setenv("HTTP_WRITE_TIMEOUT", "15.0")
        timeout = HttpClientFactory.get_httpx_timeout()
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 5.0
        assert timeout.read == 10.0
        assert timeout.write == 15.0
        assert timeout.pool == 5.0  # pool timeout == connect timeout


class TestHttpxClientCreation:
    """Test httpx Client and AsyncClient creation."""

    def test_get_httpx_client_returns_client(self) -> None:
        """get_httpx_client returns an httpx.Client instance."""
        client = HttpClientFactory.get_httpx_client()
        assert isinstance(client, httpx.Client)

    def test_httpx_client_singleton(self) -> None:
        """Subsequent calls return the same client instance."""
        client1 = HttpClientFactory.get_httpx_client()
        client2 = HttpClientFactory.get_httpx_client()
        assert client1 is client2

    def test_get_async_httpx_client_returns_async_client(self) -> None:
        """get_async_httpx_client returns an httpx.AsyncClient instance."""
        client = HttpClientFactory.get_async_httpx_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_async_httpx_client_singleton(self) -> None:
        """Subsequent calls return the same async client instance."""
        client1 = HttpClientFactory.get_async_httpx_client()
        client2 = HttpClientFactory.get_async_httpx_client()
        assert client1 is client2

    def test_httpx_client_custom_params(self) -> None:
        """Custom parameters are applied to client creation."""
        client = HttpClientFactory.get_httpx_client(
            max_connections=50,
            max_keepalive_connections=10,
            connect_timeout=5.0,
            read_timeout=10.0,
            write_timeout=15.0,
        )
        assert isinstance(client, httpx.Client)


class TestAiohttpClientCreation:
    """Test aiohttp ClientSession creation."""

    async def test_get_aiohttp_client_returns_session(self) -> None:
        """get_aiohttp_client returns an aiohttp.ClientSession instance."""
        client = HttpClientFactory.get_aiohttp_client()
        assert isinstance(client, aiohttp.ClientSession)

    async def test_aiohttp_client_singleton(self) -> None:
        """Subsequent calls return the same session instance."""
        client1 = HttpClientFactory.get_aiohttp_client()
        client2 = HttpClientFactory.get_aiohttp_client()
        assert client1 is client2


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_pooled_httpx_client(self) -> None:
        """Convenience function delegates to factory."""
        client = get_pooled_httpx_client()
        assert isinstance(client, httpx.Client)
        assert client is HttpClientFactory._httpx_client

    def test_get_pooled_async_httpx_client(self) -> None:
        """Convenience function delegates to factory."""
        client = get_pooled_async_httpx_client()
        assert isinstance(client, httpx.AsyncClient)
        assert client is HttpClientFactory._async_httpx_client

    async def test_get_pooled_aiohttp_client(self) -> None:
        """Convenience function delegates to factory."""
        client = get_pooled_aiohttp_client()
        assert isinstance(client, aiohttp.ClientSession)
        assert client is HttpClientFactory._aiohttp_client


class TestCloseAll:
    """Test close_all method cleans up all clients."""

    async def test_close_all_resets_singletons(self) -> None:
        """close_all sets all client references to None."""
        HttpClientFactory.get_httpx_client()
        HttpClientFactory.get_async_httpx_client()
        HttpClientFactory.get_aiohttp_client()

        assert HttpClientFactory._httpx_client is not None
        assert HttpClientFactory._async_httpx_client is not None
        assert HttpClientFactory._aiohttp_client is not None

        await HttpClientFactory.close_all()

        assert HttpClientFactory._httpx_client is None
        assert HttpClientFactory._async_httpx_client is None
        assert HttpClientFactory._aiohttp_client is None

    async def test_close_all_idempotent(self) -> None:
        """close_all can be called multiple times safely."""
        await HttpClientFactory.close_all()
        await HttpClientFactory.close_all()  # Should not raise

    def test_close_all_sync_closes_sync_client(self) -> None:
        """close_all_sync releases the pooled sync client."""
        client = HttpClientFactory.get_httpx_client()
        assert client is HttpClientFactory._httpx_client
        HttpClientFactory.close_all_sync()
        assert HttpClientFactory._httpx_client is None

    def test_close_all_sync_closes_async_client_from_running_loop(self) -> None:
        """close_all_sync must join aclose even when a loop is already running."""
        import asyncio

        async def _exercise() -> None:
            client = HttpClientFactory.get_async_httpx_client()
            assert client is HttpClientFactory._async_httpx_client
            HttpClientFactory.close_all_sync()
            assert HttpClientFactory._async_httpx_client is None
            assert client.is_closed

        asyncio.run(_exercise())
