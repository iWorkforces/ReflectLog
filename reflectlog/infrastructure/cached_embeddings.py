"""Cached embeddings wrapper for query embedding LRU caching."""

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reflectlog.core import IStructuredLogger

from cachetools import LRUCache
from pydantic import BaseModel, ConfigDict, PrivateAttr

from reflectlog.application.types import Embeddings


class CachedEmbeddings(BaseModel):
    """LRU caching wrapper for any Embeddings provider.

    This wrapper caches `embed_query()` results using a SHA-256 hash of the query
    text as the cache key. This is useful for search operations where the same
    query may be executed multiple times (e.g., during result refinement).

    Note: `embed_documents()` is NOT cached since document embeddings are
    typically computed once during ingestion.

    Thread-safety: Uses cachetools.LRUCache which is thread-safe by default.

    Example:
        ```python
        from reflectlog.infrastructure import LangchainQwenEmbeddings, CachedEmbeddings

        base_embedder = LangchainQwenEmbeddings({...})
        cached_embedder = CachedEmbeddings(
            embedder=base_embedder,
            cache_size=100,
            enabled=True,
        )

        # First call computes embedding
        embedding1 = cached_embedder.embed_query("Python programming")

        # Second call returns cached result
        embedding2 = cached_embedder.embed_query("Python programming")
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Wrapped embeddings provider
    embedder: Embeddings

    # Cache configuration
    cache_size: int = 100  # Maximum number of cached embeddings
    enabled: bool = True  # Enable/disable caching

    # Optional logger for cache hit/miss stats
    logger: IStructuredLogger | None = None

    # Private cache state (LRUCache from cachetools - thread-safe by default)
    _cache: LRUCache[str, list[float]] = PrivateAttr(
        default_factory=lambda: LRUCache(maxsize=100)
    )
    _hits: int = PrivateAttr(default=0)
    _misses: int = PrivateAttr(default=0)

    def _hash_query(self, text: str) -> str:
        """Compute SHA-256 hash of query text as cache key.

        SHA-256 is cryptographically secure and avoids MD5 collision risks.

        Args:
            text: Query text to hash.

        Returns:
            SHA-256 hex digest of the text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, cache_key: str) -> list[float] | None:
        """Get cached embedding if exists (LRU access).

        Args:
            cache_key: SHA-256 hash of the query text.

        Returns:
            Cached embedding if found, None otherwise.
        """
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._hits += 1
        else:
            self._misses += 1
        return cached

    def _set_cached(self, cache_key: str, embedding: list[float]) -> None:
        """Cache an embedding with LRU eviction (automatic).

        Args:
            cache_key: SHA-256 hash of the query text.
            embedding: Embedding vector to cache.
        """
        self._cache[cache_key] = embedding

    def embed_query(self, text: str) -> list[float]:
        """Embed query text with LRU caching.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector (from cache or freshly computed).
        """
        if not self.enabled:
            return self.embedder.embed_query(text)

        cache_key = self._hash_query(text)
        cached = self._get_cached(cache_key)

        if cached is not None:
            if self.logger:
                self.logger.debug(
                    "Embedding cache HIT",
                    extra={
                        "cache_key": cache_key[:8],
                        "hits": self._hits,
                        "misses": self._misses,
                    },
                )
            return cached

        # Cache miss - compute embedding
        embedding = self.embedder.embed_query(text)
        self._set_cached(cache_key, embedding)

        if self.logger:
            self.logger.debug(
                "Embedding cache MISS",
                extra={
                    "cache_key": cache_key[:8],
                    "hits": self._hits,
                    "misses": self._misses,
                    "cache_size": self._cache.currsize,
                },
            )

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (not cached - pass through to wrapped embedder).

        Document embeddings are typically computed once during ingestion,
        so caching provides little benefit and could consume significant memory.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embedder.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async version of embed_query with LRU caching.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector (from cache or freshly computed).
        """
        if not self.enabled:
            return await self.embedder.aembed_query(text)

        cache_key = self._hash_query(text)
        cached = self._get_cached(cache_key)

        if cached is not None:
            if self.logger:
                self.logger.debug(
                    "Embedding cache HIT (async)",
                    extra={
                        "cache_key": cache_key[:8],
                        "hits": self._hits,
                        "misses": self._misses,
                    },
                )
            return cached

        # Cache miss - compute embedding asynchronously
        embedding = await self.embedder.aembed_query(text)
        self._set_cached(cache_key, embedding)

        if self.logger:
            self.logger.debug(
                "Embedding cache MISS (async)",
                extra={
                    "cache_key": cache_key[:8],
                    "hits": self._hits,
                    "misses": self._misses,
                    "cache_size": self._cache.currsize,
                },
            )

        return embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_documents (not cached - pass through).

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self.embedder.aembed_documents(texts)

    def get_cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, and current cache size.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self._cache.currsize,
            "max_size": self.cache_size,
        }

    def clear_cache(self) -> None:
        """Clear cache and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
