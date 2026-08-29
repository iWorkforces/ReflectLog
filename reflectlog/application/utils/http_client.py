"""Compatibility re-export of the production HTTP client factory.

Production code uses ``reflectlog.utility.http``. This module exists so
legacy imports keep resolving to the same singleton.
"""

from reflectlog.utility.http import (
    HttpClientFactory,
    get_pooled_aiohttp_client,
    get_pooled_async_httpx_client,
    get_pooled_httpx_client,
)

__all__ = [
    "HttpClientFactory",
    "get_pooled_aiohttp_client",
    "get_pooled_async_httpx_client",
    "get_pooled_httpx_client",
]
