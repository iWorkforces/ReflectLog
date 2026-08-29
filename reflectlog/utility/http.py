"""HTTP client factory with connection pooling support.

This module provides centralized HTTP client creation with configurable
connection pooling to optimize performance and reduce resource overhead.

Key features:
- Connection pooling with configurable limits
- HTTP/2 support for httpx clients
- Configurable timeouts
- Singleton pattern for reusability

Example:
    # Get pooled httpx client
    httpx_client = HttpClientFactory.get_httpx_client(
        max_connections=100,
        max_keepalive_connections=20,
    )

    # Get pooled aiohttp client
    aiohttp_client = HttpClientFactory.get_aiohttp_client(
        max_connections=100,
        max_keepalive_connections=20,
    )
"""

import os
import threading
from typing import Any

import aiohttp
import httpx


class HttpClientFactory:
    """Factory for creating HTTP clients with connection pooling.

    Implements singleton pattern to reuse clients across the application.
    Connection pooling reduces TCP handshake overhead and improves performance.

    Configuration via environment variables:
    - HTTP_MAX_CONNECTIONS: Maximum total connections (default: 100)
    - HTTP_MAX_KEEPALIVE_CONNECTIONS: Maximum keepalive connections (default: 20)
    - HTTP_CONNECT_TIMEOUT: Connection timeout in seconds (default: 30.0)
    - HTTP_READ_TIMEOUT: Read timeout in seconds (default: 30.0)
    - HTTP_WRITE_TIMEOUT: Write timeout in seconds (default: 30.0)
    """

    _httpx_client: httpx.Client | None = None
    _async_httpx_client: httpx.AsyncClient | None = None
    _aiohttp_client: aiohttp.ClientSession | None = None
    _create_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_max_connections(cls) -> int:
        """Get maximum connections from environment or default."""
        return int(os.getenv("HTTP_MAX_CONNECTIONS", "100"))

    @classmethod
    def get_max_keepalive_connections(cls) -> int:
        """Get maximum keepalive connections from environment or default."""
        return int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "20"))

    @classmethod
    def get_connect_timeout(cls) -> float:
        """Get connection timeout from environment or default."""
        return float(os.getenv("HTTP_CONNECT_TIMEOUT", "30.0"))

    @classmethod
    def get_read_timeout(cls) -> float:
        """Get read timeout from environment or default."""
        return float(os.getenv("HTTP_READ_TIMEOUT", "30.0"))

    @classmethod
    def get_write_timeout(cls) -> float:
        """Get write timeout from environment or default."""
        return float(os.getenv("HTTP_WRITE_TIMEOUT", "30.0"))

    @classmethod
    def get_httpx_limits(cls) -> httpx.Limits:
        """Get httpx connection limits from configuration.

        Returns:
            httpx.Limits with configured connection pool settings.
        """
        return httpx.Limits(
            max_connections=cls.get_max_connections(),
            max_keepalive_connections=cls.get_max_keepalive_connections(),
        )

    @classmethod
    def get_httpx_timeout(cls) -> httpx.Timeout:
        """Get httpx timeout from configuration.

        Returns:
            httpx.Timeout with configured timeout settings.
        """
        return httpx.Timeout(
            connect=cls.get_connect_timeout(),
            read=cls.get_read_timeout(),
            write=cls.get_write_timeout(),
            pool=cls.get_connect_timeout(),
        )

    @classmethod
    def get_httpx_client(
        cls,
        *,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
        http2: bool = True,
    ) -> httpx.Client:
        """Get or create pooled httpx client.

        Returns singleton client on first call to enable connection reuse.

        Args:
            max_connections: Maximum total connections (default: from env or 100).
            max_keepalive_connections: Maximum keepalive connections (default: from env or 20).
            connect_timeout: Connection timeout in seconds (default: from env or 30.0).
            read_timeout: Read timeout in seconds (default: from env or 30.0).
            write_timeout: Write timeout in seconds (default: from env or 30.0).
            http2: Enable HTTP/2 support (default: True).

        Returns:
            Pooled httpx.Client instance.
        """
        if cls._httpx_client is not None:
            return cls._httpx_client

        with cls._create_lock:
            if cls._httpx_client is not None:
                return cls._httpx_client

            _ = (
                max_connections,
                max_keepalive_connections,
                connect_timeout,
                read_timeout,
                write_timeout,
                http2,
            )
            cls._httpx_client = httpx.Client(
                limits=cls.get_httpx_limits(),
                timeout=cls.get_httpx_timeout(),
                http2=True,
            )

            return cls._httpx_client

    @classmethod
    def get_async_httpx_client(
        cls,
        *,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
        http2: bool = True,
    ) -> httpx.AsyncClient:
        """Get or create pooled async httpx client.

        Returns singleton client on first call to enable connection reuse.

        Args:
            max_connections: Maximum total connections (default: from env or 100).
            max_keepalive_connections: Maximum keepalive connections (default: from env or 20).
            connect_timeout: Connection timeout in seconds (default: from env or 30.0).
            read_timeout: Read timeout in seconds (default: from env or 30.0).
            write_timeout: Write timeout in seconds (default: from env or 30.0).
            http2: Enable HTTP/2 support (default: True).

        Returns:
            Pooled httpx.AsyncClient instance.
        """
        if cls._async_httpx_client is not None:
            return cls._async_httpx_client

        with cls._create_lock:
            if cls._async_httpx_client is not None:
                return cls._async_httpx_client

            _ = (
                max_connections,
                max_keepalive_connections,
                connect_timeout,
                read_timeout,
                write_timeout,
                http2,
            )
            cls._async_httpx_client = httpx.AsyncClient(
                limits=cls.get_httpx_limits(),
                timeout=cls.get_httpx_timeout(),
                http2=True,
            )

            return cls._async_httpx_client

    @classmethod
    def get_aiohttp_client(
        cls,
        *,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        http2: bool = False,
    ) -> aiohttp.ClientSession:
        """Get or create pooled aiohttp client.

        Returns singleton client on first call to enable connection reuse.
        Note: aiohttp does not support HTTP/2.

        Args:
            max_connections: Maximum total connections (default: from env or 100).
            max_keepalive_connections: Maximum keepalive connections (default: from env or 20).
            connect_timeout: Connection timeout in seconds (default: from env or 30.0).
            read_timeout: Read timeout in seconds (default: from env or 30.0).
            http2: Ignored - aiohttp does not support HTTP/2.

        Returns:
            Pooled aiohttp.ClientSession instance.
        """
        if cls._aiohttp_client is not None:
            return cls._aiohttp_client

        with cls._create_lock:
            if cls._aiohttp_client is not None:
                return cls._aiohttp_client

            connector = aiohttp.TCPConnector(
                limit=max_connections or cls.get_max_connections(),
                limit_per_host=max_keepalive_connections
                or cls.get_max_keepalive_connections(),
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(
                connect=connect_timeout or cls.get_connect_timeout(),
                sock_read=read_timeout or cls.get_read_timeout(),
            )

            cls._aiohttp_client = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )

            return cls._aiohttp_client

    @classmethod
    async def close_all(cls) -> None:
        """Close all pooled HTTP clients.

        Should be called on application shutdown to release resources.
        """
        with cls._create_lock:
            sync_client = cls._httpx_client
            async_client = cls._async_httpx_client
            aiohttp_client = cls._aiohttp_client
            cls._httpx_client = None
            cls._async_httpx_client = None
            cls._aiohttp_client = None

        if sync_client is not None:
            sync_client.close()
        if async_client is not None:
            await async_client.aclose()
        if aiohttp_client is not None:
            await aiohttp_client.close()

    @classmethod
    def close_all_sync(cls) -> None:
        """Close pooled clients from a sync shutdown path (SIGINT/SIGTERM)."""
        import asyncio

        with cls._create_lock:
            sync_client = cls._httpx_client
            async_client = cls._async_httpx_client
            aiohttp_client = cls._aiohttp_client
            cls._httpx_client = None
            cls._async_httpx_client = None
            cls._aiohttp_client = None

        if sync_client is not None:
            sync_client.close()

        async def _aclose() -> None:
            if async_client is not None:
                await async_client.aclose()
            if aiohttp_client is not None:
                await aiohttp_client.close()

        if async_client is None and aiohttp_client is None:
            return

        def _run_aclose() -> None:
            asyncio.run(_aclose())

        closer = threading.Thread(target=_run_aclose, name="http-aclose", daemon=False)
        closer.start()
        closer.join(timeout=5.0)


def get_pooled_httpx_client(**kwargs: Any) -> httpx.Client:
    """Convenience function to get pooled httpx client.

    Args:
        **kwargs: Optional overrides for connection limits and timeouts.

    Returns:
        Pooled httpx.Client instance.
    """
    return HttpClientFactory.get_httpx_client(**kwargs)


def get_pooled_async_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
    """Convenience function to get pooled async httpx client.

    Args:
        **kwargs: Optional overrides for connection limits and timeouts.

    Returns:
        Pooled httpx.AsyncClient instance.
    """
    return HttpClientFactory.get_async_httpx_client(**kwargs)


def get_pooled_aiohttp_client(**kwargs: Any) -> aiohttp.ClientSession:
    """Convenience function to get pooled aiohttp client.

    Args:
        **kwargs: Optional overrides for connection limits and timeouts.

    Returns:
        Pooled aiohttp.ClientSession instance.
    """
    return HttpClientFactory.get_aiohttp_client(**kwargs)
