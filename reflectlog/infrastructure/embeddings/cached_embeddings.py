"""Cached embeddings wrapper for query embedding LRU caching."""

import hashlib
import threading

from cachetools import LRUCache
from pydantic import BaseModel, ConfigDict, PrivateAttr

from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import Embeddings


class CachedEmbeddings(BaseModel):
    """LRU caching wrapper for any Embeddings provider.

    This wrapper caches `embed_query()` results using a SHA-256 hash of the query
    text as the cache key. This is useful for search operations where the same
    query may be executed multiple times (e.g., during result refinement).

    `embed_documents()` consults the same per-text LRU so add-path Phase 2
    query embeds can be reused during Phase 3 persist.

    Thread-safety: cachetools.LRUCache is not thread-safe; access is locked.

    Example:
        ```python
        from reflectlog.infrastructure.cached_embeddings import CachedEmbeddings
        from reflectlog.infrastructure.qwen3_embedding import LangchainQwenEmbeddings

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

    _cache: LRUCache[str, list[float]] = PrivateAttr(
        default_factory=lambda: LRUCache(maxsize=100)
    )
    _cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _hits: int = PrivateAttr(default=0)
    _misses: int = PrivateAttr(default=0)

    def model_post_init(self, _context: object, /) -> None:
        """Bind LRU capacity to the configured cache_size."""
        self._cache = LRUCache(maxsize=max(1, self.cache_size))

    def _normalize_text(self, text: str) -> str:
        """Match embedder newline collapsing so cache keys align."""
        return text.replace("\n", " ")

    def _hash_query(self, text: str) -> str:
        """Compute SHA-256 hash of query text as cache key.

        SHA-256 is cryptographically secure and avoids MD5 collision risks.

        Args:
            text: Query text to hash.

        Returns:
            SHA-256 hex digest of the text.
        """
        return hashlib.sha256(self._normalize_text(text).encode("utf-8")).hexdigest()

    def _get_cached(self, cache_key: str) -> list[float] | None:
        """Get cached embedding if exists (LRU access).

        Args:
            cache_key: SHA-256 hash of the query text.

        Returns:
            Cached embedding if found, None otherwise.
        """
        with self._cache_lock:
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
        with self._cache_lock:
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
        if not embedding:
            raise RuntimeError("Embedding produced an empty vector")
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
        """Embed documents, reusing per-text LRU entries from embed_query.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self.enabled or not texts:
            return self.embedder.embed_documents(texts)

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for idx, text in enumerate(texts):
            cached = self._get_cached(self._hash_query(text))
            if cached is not None:
                results[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            computed = self.embedder.embed_documents(miss_texts)
            if len(computed) != len(miss_texts):
                raise RuntimeError(
                    "Embedding batch size mismatch for cached embed_documents"
                )
            for idx, embedding in zip(miss_indices, computed, strict=True):
                if not embedding:
                    raise RuntimeError("Empty embedding returned for cached document")
                self._set_cached(self._hash_query(texts[idx]), embedding)
                results[idx] = embedding

        filled: list[list[float]] = []
        for embedding in results:
            if embedding is None:
                raise RuntimeError("Missing cached embedding slot")
            filled.append(embedding)
        return filled

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
        if not embedding:
            raise RuntimeError("Embedding produced an empty vector")
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
        """Async version of embed_documents with the same per-text LRU.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self.enabled or not texts:
            return await self.embedder.aembed_documents(texts)

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for idx, text in enumerate(texts):
            cached = self._get_cached(self._hash_query(text))
            if cached is not None:
                results[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            computed = await self.embedder.aembed_documents(miss_texts)
            if len(computed) != len(miss_texts):
                raise RuntimeError(
                    "Embedding batch size mismatch for cached aembed_documents"
                )
            for idx, embedding in zip(miss_indices, computed, strict=True):
                if not embedding:
                    raise RuntimeError("Empty embedding returned for cached document")
                self._set_cached(self._hash_query(texts[idx]), embedding)
                results[idx] = embedding

        filled: list[list[float]] = []
        for embedding in results:
            if embedding is None:
                raise RuntimeError("Missing cached embedding slot")
            filled.append(embedding)
        return filled

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
