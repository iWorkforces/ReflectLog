"""Unit tests for reflectlog.application.utils.http_client module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest

from reflectlog.application.utils.http_client import (
    HttpClientFactory,
    get_pooled_aiohttp_client,
    get_pooled_async_httpx_client,
    get_pooled_httpx_client,
)


@pytest.fixture(autouse=True)
def reset_factory_singletons():
    """Reset all singleton clients before and after each test."""
    HttpClientFactory._httpx_client = None
    HttpClientFactory._async_httpx_client = None
    HttpClientFactory._aiohttp_client = None
    yield
    HttpClientFactory._httpx_client = None
    HttpClientFactory._async_httpx_client = None
    HttpClientFactory._aiohttp_client = None


class TestGetMaxConnections:
    """Tests for HttpClientFactory.get_max_connections()."""

    def test_default_value(self) -> None:
        """Returns 100 when HTTP_MAX_CONNECTIONS is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_MAX_CONNECTIONS", None)
            assert HttpClientFactory.get_max_connections() == 100

    def test_custom_value(self) -> None:
        """Returns custom value from environment."""
        with patch.dict(os.environ, {"HTTP_MAX_CONNECTIONS": "200"}):
            assert HttpClientFactory.get_max_connections() == 200


class TestGetMaxKeepaliveConnections:
    """Tests for HttpClientFactory.get_max_keepalive_connections()."""

    def test_default_value(self) -> None:
        """Returns 20 when HTTP_MAX_KEEPALIVE_CONNECTIONS is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_MAX_KEEPALIVE_CONNECTIONS", None)
            assert HttpClientFactory.get_max_keepalive_connections() == 20

    def test_custom_value(self) -> None:
        """Returns custom value from environment."""
        with patch.dict(os.environ, {"HTTP_MAX_KEEPALIVE_CONNECTIONS": "50"}):
            assert HttpClientFactory.get_max_keepalive_connections() == 50


class TestGetConnectTimeout:
    """Tests for HttpClientFactory.get_connect_timeout()."""

    def test_default_value(self) -> None:
        """Returns 30.0 when HTTP_CONNECT_TIMEOUT is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_CONNECT_TIMEOUT", None)
            assert HttpClientFactory.get_connect_timeout() == 30.0

    def test_custom_value(self) -> None:
        """Returns custom value from environment."""
        with patch.dict(os.environ, {"HTTP_CONNECT_TIMEOUT": "10.0"}):
            assert HttpClientFactory.get_connect_timeout() == 10.0


class TestGetReadTimeout:
    """Tests for HttpClientFactory.get_read_timeout()."""

    def test_default_value(self) -> None:
        """Returns 30.0 when HTTP_READ_TIMEOUT is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_READ_TIMEOUT", None)
            assert HttpClientFactory.get_read_timeout() == 30.0

    def test_custom_value(self) -> None:
        """Returns custom value from environment."""
        with patch.dict(os.environ, {"HTTP_READ_TIMEOUT": "15.5"}):
            assert HttpClientFactory.get_read_timeout() == 15.5


class TestGetWriteTimeout:
    """Tests for HttpClientFactory.get_write_timeout()."""

    def test_default_value(self) -> None:
        """Returns 30.0 when HTTP_WRITE_TIMEOUT is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_WRITE_TIMEOUT", None)
            assert HttpClientFactory.get_write_timeout() == 30.0

    def test_custom_value(self) -> None:
        """Returns custom value from environment."""
        with patch.dict(os.environ, {"HTTP_WRITE_TIMEOUT": "45.0"}):
            assert HttpClientFactory.get_write_timeout() == 45.0


class TestGetHttpxLimits:
    """Tests for HttpClientFactory.get_httpx_limits()."""

    def test_returns_httpx_limits(self) -> None:
        """Returns httpx.Limits with configured values."""
        with patch.dict(
            os.environ,
            {"HTTP_MAX_CONNECTIONS": "50", "HTTP_MAX_KEEPALIVE_CONNECTIONS": "10"},
        ):
            limits = HttpClientFactory.get_httpx_limits()

            assert isinstance(limits, httpx.Limits)
            assert limits.max_connections == 50
            assert limits.max_keepalive_connections == 10

    def test_uses_defaults(self) -> None:
        """Uses default values when env vars are not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_MAX_CONNECTIONS", None)
            os.environ.pop("HTTP_MAX_KEEPALIVE_CONNECTIONS", None)
            limits = HttpClientFactory.get_httpx_limits()

            assert limits.max_connections == 100
            assert limits.max_keepalive_connections == 20


class TestGetHttpxTimeout:
    """Tests for HttpClientFactory.get_httpx_timeout()."""

    def test_returns_httpx_timeout(self) -> None:
        """Returns httpx.Timeout with configured values."""
        with patch.dict(
            os.environ,
            {
                "HTTP_CONNECT_TIMEOUT": "5.0",
                "HTTP_READ_TIMEOUT": "10.0",
                "HTTP_WRITE_TIMEOUT": "15.0",
            },
        ):
            timeout = HttpClientFactory.get_httpx_timeout()

            assert isinstance(timeout, httpx.Timeout)
            assert timeout.connect == 5.0
            assert timeout.read == 10.0
            assert timeout.write == 15.0
            assert timeout.pool == 5.0

    def test_uses_defaults(self) -> None:
        """Uses default values when env vars are not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_CONNECT_TIMEOUT", None)
            os.environ.pop("HTTP_READ_TIMEOUT", None)
            os.environ.pop("HTTP_WRITE_TIMEOUT", None)
            timeout = HttpClientFactory.get_httpx_timeout()

            assert timeout.connect == 30.0
            assert timeout.read == 30.0
            assert timeout.write == 30.0
            assert timeout.pool == 30.0


class TestGetHttpxClient:
    """Tests for HttpClientFactory.get_httpx_client()."""

    def test_creates_client(self) -> None:
        """Creates an httpx.Client on first call."""
        client = HttpClientFactory.get_httpx_client()

        assert isinstance(client, httpx.Client)

    def test_singleton_returns_same_instance(self) -> None:
        """Subsequent calls return the same client instance."""
        client1 = HttpClientFactory.get_httpx_client()
        client2 = HttpClientFactory.get_httpx_client()

        assert client1 is client2

    def test_custom_limits_passed_to_constructor(self) -> None:
        """Custom max_connections and max_keepalive are passed to httpx.Limits."""
        with patch(
            "reflectlog.application.utils.http_client.httpx.Limits",
        ) as mock_limits:
            mock_limits.return_value = MagicMock()
            with patch(
                "reflectlog.application.utils.http_client.httpx.Timeout",
            ) as mock_timeout:
                mock_timeout.return_value = MagicMock()
                with patch(
                    "reflectlog.application.utils.http_client.httpx.Client",
                ) as mock_client_cls:
                    mock_client_cls.return_value = MagicMock()
                    HttpClientFactory.get_httpx_client(
                        max_connections=50,
                        max_keepalive_connections=5,
                    )

                    limits_kwargs = mock_limits.call_args[1]
                    assert limits_kwargs["max_connections"] == 50
                    assert limits_kwargs["max_keepalive_connections"] == 5

    def test_custom_timeouts(self) -> None:
        """Custom timeout values override environment defaults."""
        client = HttpClientFactory.get_httpx_client(
            connect_timeout=5.0,
            read_timeout=10.0,
            write_timeout=15.0,
        )

        assert client.timeout.connect == 5.0
        assert client.timeout.read == 10.0
        assert client.timeout.write == 15.0

    def test_http2_enabled_by_default(self) -> None:
        """HTTP/2 is enabled by default."""
        with patch(
            "reflectlog.application.utils.http_client.httpx.Client",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            HttpClientFactory.get_httpx_client()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["http2"] is True

    def test_http2_disabled(self) -> None:
        """HTTP/2 can be disabled."""
        with patch(
            "reflectlog.application.utils.http_client.httpx.Client",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            HttpClientFactory.get_httpx_client(http2=False)

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["http2"] is False

    def test_env_based_defaults(self) -> None:
        """Uses environment-based defaults when no overrides given."""
        with patch.dict(
            os.environ,
            {
                "HTTP_MAX_CONNECTIONS": "75",
                "HTTP_CONNECT_TIMEOUT": "5.0",
                "HTTP_READ_TIMEOUT": "10.0",
                "HTTP_WRITE_TIMEOUT": "15.0",
            },
        ):
            client = HttpClientFactory.get_httpx_client()

            assert client.timeout.connect == 5.0
            assert client.timeout.read == 10.0
            assert client.timeout.write == 15.0


class TestGetAsyncHttpxClient:
    """Tests for HttpClientFactory.get_async_httpx_client()."""

    def test_creates_async_client(self) -> None:
        """Creates an httpx.AsyncClient on first call."""
        client = HttpClientFactory.get_async_httpx_client()

        assert isinstance(client, httpx.AsyncClient)

    def test_singleton_returns_same_instance(self) -> None:
        """Subsequent calls return the same async client instance."""
        client1 = HttpClientFactory.get_async_httpx_client()
        client2 = HttpClientFactory.get_async_httpx_client()

        assert client1 is client2

    def test_custom_limits_passed_to_constructor(self) -> None:
        """Custom limits are passed to httpx.Limits for async client."""
        with patch(
            "reflectlog.application.utils.http_client.httpx.Limits",
        ) as mock_limits:
            mock_limits.return_value = MagicMock()
            with patch(
                "reflectlog.application.utils.http_client.httpx.Timeout",
            ) as mock_timeout:
                mock_timeout.return_value = MagicMock()
                with patch(
                    "reflectlog.application.utils.http_client.httpx.AsyncClient",
                ) as mock_client_cls:
                    mock_client_cls.return_value = MagicMock()
                    HttpClientFactory.get_async_httpx_client(
                        max_connections=60,
                        max_keepalive_connections=8,
                    )

                    limits_kwargs = mock_limits.call_args[1]
                    assert limits_kwargs["max_connections"] == 60
                    assert limits_kwargs["max_keepalive_connections"] == 8

    def test_custom_timeouts(self) -> None:
        """Custom timeout values override environment defaults."""
        client = HttpClientFactory.get_async_httpx_client(
            connect_timeout=3.0,
            read_timeout=6.0,
            write_timeout=9.0,
        )

        assert client.timeout.connect == 3.0
        assert client.timeout.read == 6.0
        assert client.timeout.write == 9.0

    def test_http2_enabled_by_default(self) -> None:
        """HTTP/2 is enabled by default for async client."""
        with patch(
            "reflectlog.application.utils.http_client.httpx.AsyncClient",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            HttpClientFactory.get_async_httpx_client()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["http2"] is True

    def test_http2_disabled(self) -> None:
        """HTTP/2 can be disabled for async client."""
        with patch(
            "reflectlog.application.utils.http_client.httpx.AsyncClient",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            HttpClientFactory.get_async_httpx_client(http2=False)

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["http2"] is False

    def test_env_based_defaults(self) -> None:
        """Uses environment-based defaults when no overrides given."""
        with patch.dict(
            os.environ,
            {
                "HTTP_MAX_CONNECTIONS": "80",
                "HTTP_CONNECT_TIMEOUT": "7.0",
                "HTTP_READ_TIMEOUT": "12.0",
                "HTTP_WRITE_TIMEOUT": "18.0",
            },
        ):
            client = HttpClientFactory.get_async_httpx_client()

            assert client.timeout.connect == 7.0
            assert client.timeout.read == 12.0
            assert client.timeout.write == 18.0


class TestGetAiohttpClient:
    """Tests for HttpClientFactory.get_aiohttp_client()."""

    def test_creates_aiohttp_session(self) -> None:
        """Creates an aiohttp.ClientSession via mocked constructors."""
        mock_connector = MagicMock()
        mock_timeout = MagicMock()
        mock_session = MagicMock(spec=aiohttp.ClientSession)

        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
                return_value=mock_connector,
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
                return_value=mock_timeout,
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
                return_value=mock_session,
            ) as mock_session_cls,
        ):
            client = HttpClientFactory.get_aiohttp_client()

            assert client is mock_session
            mock_session_cls.assert_called_once_with(
                connector=mock_connector,
                timeout=mock_timeout,
            )

    def test_singleton_returns_same_instance(self) -> None:
        """Subsequent calls return the same session instance."""
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        HttpClientFactory._aiohttp_client = mock_session

        client1 = HttpClientFactory.get_aiohttp_client()
        client2 = HttpClientFactory.get_aiohttp_client()

        assert client1 is client2
        assert client1 is mock_session

    def test_connector_max_connections(self) -> None:
        """TCPConnector is configured with custom max_connections."""
        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
            ) as mock_conn_cls,
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
            ),
        ):
            mock_conn_cls.return_value = MagicMock()
            HttpClientFactory.get_aiohttp_client(max_connections=75)

            call_kwargs = mock_conn_cls.call_args[1]
            assert call_kwargs["limit"] == 75

    def test_connector_limit_per_host(self) -> None:
        """TCPConnector limit_per_host set from max_keepalive_connections."""
        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
            ) as mock_conn_cls,
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
            ),
        ):
            mock_conn_cls.return_value = MagicMock()
            HttpClientFactory.get_aiohttp_client(max_keepalive_connections=15)

            call_kwargs = mock_conn_cls.call_args[1]
            assert call_kwargs["limit_per_host"] == 15

    def test_connector_dns_cache(self) -> None:
        """TCPConnector has DNS caching enabled with 300s TTL."""
        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
            ) as mock_conn_cls,
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
            ),
        ):
            mock_conn_cls.return_value = MagicMock()
            HttpClientFactory.get_aiohttp_client()

            call_kwargs = mock_conn_cls.call_args[1]
            assert call_kwargs["ttl_dns_cache"] == 300

    def test_connector_cleanup_closed(self) -> None:
        """TCPConnector has enable_cleanup_closed set to True."""
        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
            ) as mock_conn_cls,
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
            ),
        ):
            mock_conn_cls.return_value = MagicMock()
            HttpClientFactory.get_aiohttp_client()

            call_kwargs = mock_conn_cls.call_args[1]
            assert call_kwargs["enable_cleanup_closed"] is True

    def test_custom_timeouts(self) -> None:
        """Custom timeout values override environment defaults."""
        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
            ) as mock_timeout_cls,
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
            ),
        ):
            mock_timeout_cls.return_value = MagicMock()
            HttpClientFactory.get_aiohttp_client(
                connect_timeout=4.0,
                read_timeout=8.0,
            )

            call_kwargs = mock_timeout_cls.call_args[1]
            assert call_kwargs["connect"] == 4.0
            assert call_kwargs["sock_read"] == 8.0

    def test_env_based_defaults(self) -> None:
        """Uses environment-based defaults when no overrides given."""
        with patch.dict(
            os.environ,
            {
                "HTTP_MAX_CONNECTIONS": "120",
                "HTTP_MAX_KEEPALIVE_CONNECTIONS": "30",
                "HTTP_CONNECT_TIMEOUT": "6.0",
                "HTTP_READ_TIMEOUT": "11.0",
            },
        ):
            with (
                patch(
                    "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
                ) as mock_conn_cls,
                patch(
                    "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
                ) as mock_timeout_cls,
                patch(
                    "reflectlog.application.utils.http_client.aiohttp.ClientSession",
                ),
            ):
                mock_conn_cls.return_value = MagicMock()
                mock_timeout_cls.return_value = MagicMock()
                HttpClientFactory.get_aiohttp_client()

                conn_kwargs = mock_conn_cls.call_args[1]
                assert conn_kwargs["limit"] == 120
                assert conn_kwargs["limit_per_host"] == 30

                timeout_kwargs = mock_timeout_cls.call_args[1]
                assert timeout_kwargs["connect"] == 6.0
                assert timeout_kwargs["sock_read"] == 11.0


class TestCloseAll:
    """Tests for HttpClientFactory.close_all()."""

    async def test_close_all_with_no_clients(self) -> None:
        """close_all with no active clients does nothing."""
        await HttpClientFactory.close_all()

        assert HttpClientFactory._httpx_client is None
        assert HttpClientFactory._async_httpx_client is None
        assert HttpClientFactory._aiohttp_client is None

    async def test_close_httpx_client(self) -> None:
        """close_all closes the sync httpx client and resets to None."""
        mock_client = MagicMock(spec=httpx.Client)
        HttpClientFactory._httpx_client = mock_client

        await HttpClientFactory.close_all()

        mock_client.close.assert_called_once()
        assert HttpClientFactory._httpx_client is None

    async def test_close_async_httpx_client(self) -> None:
        """close_all closes the async httpx client and resets to None."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        HttpClientFactory._async_httpx_client = mock_client

        await HttpClientFactory.close_all()

        mock_client.aclose.assert_awaited_once()
        assert HttpClientFactory._async_httpx_client is None

    async def test_close_aiohttp_client(self) -> None:
        """close_all closes the aiohttp client session and resets to None."""
        mock_client = AsyncMock(spec=aiohttp.ClientSession)
        HttpClientFactory._aiohttp_client = mock_client

        await HttpClientFactory.close_all()

        mock_client.close.assert_awaited_once()
        assert HttpClientFactory._aiohttp_client is None

    async def test_close_all_clients_simultaneously(self) -> None:
        """close_all closes all three clients when all are active."""
        mock_httpx = MagicMock(spec=httpx.Client)
        mock_async_httpx = AsyncMock(spec=httpx.AsyncClient)
        mock_aiohttp = AsyncMock(spec=aiohttp.ClientSession)

        HttpClientFactory._httpx_client = mock_httpx
        HttpClientFactory._async_httpx_client = mock_async_httpx
        HttpClientFactory._aiohttp_client = mock_aiohttp

        await HttpClientFactory.close_all()

        mock_httpx.close.assert_called_once()
        mock_async_httpx.aclose.assert_awaited_once()
        mock_aiohttp.close.assert_awaited_once()
        assert HttpClientFactory._httpx_client is None
        assert HttpClientFactory._async_httpx_client is None
        assert HttpClientFactory._aiohttp_client is None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_pooled_httpx_client(self) -> None:
        """get_pooled_httpx_client delegates to HttpClientFactory."""
        client = get_pooled_httpx_client()

        assert isinstance(client, httpx.Client)

    def test_get_pooled_httpx_client_singleton(self) -> None:
        """get_pooled_httpx_client returns the same singleton."""
        client1 = get_pooled_httpx_client()
        client2 = get_pooled_httpx_client()

        assert client1 is client2

    def test_get_pooled_httpx_client_with_kwargs(self) -> None:
        """get_pooled_httpx_client passes kwargs to factory."""
        client = get_pooled_httpx_client(max_connections=42)

        assert isinstance(client, httpx.Client)

    def test_get_pooled_async_httpx_client(self) -> None:
        """get_pooled_async_httpx_client delegates to HttpClientFactory."""
        client = get_pooled_async_httpx_client()

        assert isinstance(client, httpx.AsyncClient)

    def test_get_pooled_async_httpx_client_singleton(self) -> None:
        """get_pooled_async_httpx_client returns the same singleton."""
        client1 = get_pooled_async_httpx_client()
        client2 = get_pooled_async_httpx_client()

        assert client1 is client2

    def test_get_pooled_async_httpx_client_with_kwargs(self) -> None:
        """get_pooled_async_httpx_client passes kwargs to factory."""
        client = get_pooled_async_httpx_client(max_connections=33)

        assert isinstance(client, httpx.AsyncClient)

    def test_get_pooled_aiohttp_client(self) -> None:
        """get_pooled_aiohttp_client delegates to HttpClientFactory."""
        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
                return_value=MagicMock(spec=aiohttp.ClientSession),
            ),
        ):
            client = get_pooled_aiohttp_client()

            assert client is not None

    def test_get_pooled_aiohttp_client_singleton(self) -> None:
        """get_pooled_aiohttp_client returns the same singleton."""
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        HttpClientFactory._aiohttp_client = mock_session

        client1 = get_pooled_aiohttp_client()
        client2 = get_pooled_aiohttp_client()

        assert client1 is client2

    def test_get_pooled_aiohttp_client_with_kwargs(self) -> None:
        """get_pooled_aiohttp_client passes kwargs to factory."""
        with (
            patch(
                "reflectlog.application.utils.http_client.aiohttp.TCPConnector",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientTimeout",
            ),
            patch(
                "reflectlog.application.utils.http_client.aiohttp.ClientSession",
                return_value=MagicMock(spec=aiohttp.ClientSession),
            ),
        ):
            client = get_pooled_aiohttp_client(max_connections=55)

            assert client is not None
