'''Unit tests for CachedEmbeddings LRU caching wrapper.'''

import hashlib
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from cachetools import LRUCache

from reflectlog.core.types import Embeddings
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.cached_embeddings import CachedEmbeddings


def create_mock_logger() -> IStructuredLogger:
    '''Create a properly typed mock logger for testing.'''
    return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))


@pytest.fixture
def mock_embedder() -> MagicMock:
    '''Create a mock Embeddings provider satisfying runtime_checkable protocol.'''
    embedder = MagicMock(spec=Embeddings)
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    embedder.aembed_query = AsyncMock(return_value=[0.4, 0.5, 0.6])
    embedder.aembed_documents = AsyncMock(return_value=[[0.7, 0.8], [0.9, 1.0]])
    return embedder


@pytest.fixture
def cached(mock_embedder: MagicMock) -> CachedEmbeddings:
    '''Create CachedEmbeddings with default settings.'''
    return CachedEmbeddings(embedder=mock_embedder, cache_size=5, enabled=True)


@pytest.fixture
def cached_disabled(mock_embedder: MagicMock) -> CachedEmbeddings:
    '''Create CachedEmbeddings with caching disabled.'''
    return CachedEmbeddings(embedder=mock_embedder, cache_size=5, enabled=False)


@pytest.fixture
def cached_with_logger(mock_embedder: MagicMock) -> CachedEmbeddings:
    '''Create CachedEmbeddings with a mock logger.'''
    logger = create_mock_logger()
    return CachedEmbeddings(
        embedder=mock_embedder, cache_size=5, enabled=True, logger=logger
    )


class TestCachedEmbeddingsInit:
    '''Test CachedEmbeddings initialization.'''

    def test_default_cache_size(self, mock_embedder: MagicMock) -> None:
        '''Test default cache_size is 100.'''
        cached = CachedEmbeddings(embedder=mock_embedder)
        assert cached.cache_size == 100

    def test_default_enabled(self, mock_embedder: MagicMock) -> None:
        '''Test caching is enabled by default.'''
        cached = CachedEmbeddings(embedder=mock_embedder)
        assert cached.enabled is True

    def test_custom_cache_size(self, mock_embedder: MagicMock) -> None:
        '''Test custom cache_size.'''
        cached = CachedEmbeddings(embedder=mock_embedder, cache_size=50)
        assert cached.cache_size == 50

    def test_initial_stats_are_zero(self, cached: CachedEmbeddings) -> None:
        '''Test initial cache stats are all zero.'''
        stats = cached.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["max_size"] == 5


class TestHashQuery:
    '''Test _hash_query method.'''

    def test_returns_sha256_hex(self, cached: CachedEmbeddings) -> None:
        '''Test that _hash_query returns SHA-256 hex digest.'''
        text = "hello world"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert cached._hash_query(text) == expected

    def test_different_texts_produce_different_hashes(
        self, cached: CachedEmbeddings
    ) -> None:
        '''Test that different texts produce different hashes.'''
        hash1 = cached._hash_query("text one")
        hash2 = cached._hash_query("text two")
        assert hash1 != hash2

    def test_same_text_produces_same_hash(self, cached: CachedEmbeddings) -> None:
        '''Test that same text always produces same hash.'''
        hash1 = cached._hash_query("deterministic")
        hash2 = cached._hash_query("deterministic")
        assert hash1 == hash2


class TestGetCached:
    '''Test _get_cached method.'''

    def test_cache_miss_returns_none(self, cached: CachedEmbeddings) -> None:
        '''Test cache miss returns None and increments misses.'''
        result = cached._get_cached("nonexistent_key")
        assert result is None
        assert cached._misses == 1
        assert cached._hits == 0

    def test_cache_hit_returns_value(self, cached: CachedEmbeddings) -> None:
        '''Test cache hit returns stored value and increments hits.'''
        cached._set_cached("key1", [1.0, 2.0])
        result = cached._get_cached("key1")
        assert result == [1.0, 2.0]
        assert cached._hits == 1

    def test_multiple_misses_increment(self, cached: CachedEmbeddings) -> None:
        '''Test multiple misses increment counter correctly.'''
        cached._get_cached("a")
        cached._get_cached("b")
        cached._get_cached("c")
        assert cached._misses == 3


class TestSetCached:
    '''Test _set_cached method.'''

    def test_stores_value(self, cached: CachedEmbeddings) -> None:
        '''Test that _set_cached stores a value retrievable by key.'''
        cached._set_cached("k", [0.5])
        assert cached._cache["k"] == [0.5]

    def test_overwrites_existing(self, cached: CachedEmbeddings) -> None:
        '''Test that _set_cached overwrites existing value for same key.'''
        cached._set_cached("k", [1.0])
        cached._set_cached("k", [2.0])
        assert cached._cache["k"] == [2.0]


class TestEmbedQuery:
    '''Test embed_query method with caching.'''

    def test_cache_miss_calls_embedder(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test that cache miss calls the underlying embedder.'''
        result = cached.embed_query("test query")
        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once_with("test query")

    def test_cache_hit_skips_embedder(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test that cache hit does not call embedder again.'''
        cached.embed_query("same query")
        cached.embed_query("same query")
        mock_embedder.embed_query.assert_called_once_with("same query")

    def test_cache_hit_returns_same_result(self, cached: CachedEmbeddings) -> None:
        '''Test that cache hit returns the same embedding.'''
        result1 = cached.embed_query("query")
        result2 = cached.embed_query("query")
        assert result1 == result2

    def test_disabled_bypasses_cache(
        self, cached_disabled: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test that disabled caching always calls embedder.'''
        cached_disabled.embed_query("q1")
        cached_disabled.embed_query("q1")
        assert mock_embedder.embed_query.call_count == 2

    def test_disabled_does_not_populate_cache(
        self, cached_disabled: CachedEmbeddings
    ) -> None:
        '''Test that disabled caching does not store in cache.'''
        cached_disabled.embed_query("q1")
        stats = cached_disabled.get_cache_stats()
        assert stats["size"] == 0

    def test_different_queries_both_cached(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test that different queries are cached independently.'''
        mock_embedder.embed_query.side_effect = [[1.0], [2.0]]
        r1 = cached.embed_query("alpha")
        r2 = cached.embed_query("beta")
        assert r1 == [1.0]
        assert r2 == [2.0]
        assert cached.get_cache_stats()["size"] == 2

    def test_cache_miss_with_logger(self, cached_with_logger: CachedEmbeddings) -> None:
        '''Test that cache miss logs debug message with logger.'''
        cached_with_logger.embed_query("logged query")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        mock_logger.debug.assert_called_once()
        args = mock_logger.debug.call_args
        assert "MISS" in args[0][0]

    def test_cache_hit_with_logger(self, cached_with_logger: CachedEmbeddings) -> None:
        '''Test that cache hit logs debug message with logger.'''
        cached_with_logger.embed_query("logged hit")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        mock_logger.debug.reset_mock()
        cached_with_logger.embed_query("logged hit")
        args = mock_logger.debug.call_args
        assert "HIT" in args[0][0]

    def test_cache_miss_log_includes_extra(
        self, cached_with_logger: CachedEmbeddings
    ) -> None:
        '''Test cache miss log includes cache_key, hits, misses, cache_size.'''
        cached_with_logger.embed_query("extra check")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        call_kwargs = mock_logger.debug.call_args
        extra = call_kwargs[1]["extra"]
        assert "cache_key" in extra
        assert "hits" in extra
        assert "misses" in extra
        assert "cache_size" in extra

    def test_cache_hit_log_includes_extra(
        self, cached_with_logger: CachedEmbeddings
    ) -> None:
        '''Test cache hit log includes cache_key, hits, misses.'''
        cached_with_logger.embed_query("extra hit")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        mock_logger.debug.reset_mock()
        cached_with_logger.embed_query("extra hit")
        call_kwargs = mock_logger.debug.call_args
        extra = call_kwargs[1]["extra"]
        assert "cache_key" in extra
        assert "hits" in extra
        assert "misses" in extra

    def test_no_logger_no_error_on_miss(self, cached: CachedEmbeddings) -> None:
        '''Test that no logger does not cause error on miss.'''
        assert cached.logger is None
        cached.embed_query("no logger miss")

    def test_no_logger_no_error_on_hit(self, cached: CachedEmbeddings) -> None:
        '''Test that no logger does not cause error on hit.'''
        assert cached.logger is None
        cached.embed_query("no logger hit")
        cached.embed_query("no logger hit")


class TestEmbedDocuments:
    '''Test embed_documents pass-through method.'''

    def test_passes_through_to_embedder(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test embed_documents delegates directly to embedder.'''
        texts = ["doc1", "doc2"]
        result = cached.embed_documents(texts)
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_embedder.embed_documents.assert_called_once_with(texts)

    def test_reuses_cache_on_second_call(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test embed_documents populates the per-text LRU.'''
        cached.embed_documents(["doc"])
        cached.embed_documents(["doc"])
        assert mock_embedder.embed_documents.call_count == 1
        assert cached.get_cache_stats()["size"] == 1


class TestAembedQuery:
    '''Test async aembed_query method with caching.'''

    async def test_cache_miss_calls_embedder(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test async cache miss calls the underlying embedder.'''
        result = await cached.aembed_query("async query")
        assert result == [0.4, 0.5, 0.6]
        mock_embedder.aembed_query.assert_awaited_once_with("async query")

    async def test_cache_hit_skips_embedder(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test async cache hit does not call embedder again.'''
        await cached.aembed_query("async same")
        await cached.aembed_query("async same")
        mock_embedder.aembed_query.assert_awaited_once_with("async same")

    async def test_disabled_bypasses_cache(
        self, cached_disabled: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test async disabled caching always calls embedder.'''
        await cached_disabled.aembed_query("q")
        await cached_disabled.aembed_query("q")
        assert mock_embedder.aembed_query.await_count == 2

    async def test_cache_miss_with_logger(
        self, cached_with_logger: CachedEmbeddings
    ) -> None:
        '''Test async cache miss logs debug with logger.'''
        await cached_with_logger.aembed_query("async logged")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        args = mock_logger.debug.call_args
        assert "MISS" in args[0][0]
        assert "async" in args[0][0]

    async def test_cache_hit_with_logger(
        self, cached_with_logger: CachedEmbeddings
    ) -> None:
        '''Test async cache hit logs debug with logger.'''
        await cached_with_logger.aembed_query("async hit log")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        mock_logger.debug.reset_mock()
        await cached_with_logger.aembed_query("async hit log")
        args = mock_logger.debug.call_args
        assert "HIT" in args[0][0]
        assert "async" in args[0][0]

    async def test_cache_miss_log_includes_extra(
        self, cached_with_logger: CachedEmbeddings
    ) -> None:
        '''Test async cache miss log includes expected extra fields.'''
        await cached_with_logger.aembed_query("async extra")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        call_kwargs = mock_logger.debug.call_args
        extra = call_kwargs[1]["extra"]
        assert "cache_key" in extra
        assert "cache_size" in extra

    async def test_cache_hit_log_includes_extra(
        self, cached_with_logger: CachedEmbeddings
    ) -> None:
        '''Test async cache hit log includes expected extra fields.'''
        await cached_with_logger.aembed_query("async extra hit")
        mock_logger = cast(MagicMock, cached_with_logger.logger)
        mock_logger.debug.reset_mock()
        await cached_with_logger.aembed_query("async extra hit")
        call_kwargs = mock_logger.debug.call_args
        extra = call_kwargs[1]["extra"]
        assert "cache_key" in extra
        assert "hits" in extra

    async def test_no_logger_no_error(self, cached: CachedEmbeddings) -> None:
        '''Test async no logger does not cause error.'''
        await cached.aembed_query("no logger async")
        await cached.aembed_query("no logger async")


class TestAembedDocuments:
    '''Test async aembed_documents pass-through method.'''

    async def test_passes_through_to_embedder(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test aembed_documents delegates directly to embedder.'''
        texts = ["adoc1", "adoc2"]
        result = await cached.aembed_documents(texts)
        assert result == [[0.7, 0.8], [0.9, 1.0]]
        mock_embedder.aembed_documents.assert_awaited_once_with(texts)

    async def test_reuses_cache_on_second_call(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test aembed_documents populates the per-text LRU.'''
        await cached.aembed_documents(["adoc"])
        await cached.aembed_documents(["adoc"])
        assert mock_embedder.aembed_documents.await_count == 1


class TestLRUEviction:
    '''Test LRU cache eviction behavior.'''

    def test_evicts_oldest_when_full(self, mock_embedder: MagicMock) -> None:
        '''Test that LRU evicts least recently used entry when full.'''
        cached = CachedEmbeddings(embedder=mock_embedder, cache_size=3)
        cached._cache = LRUCache(maxsize=3)
        mock_embedder.embed_query.side_effect = lambda t: [float(ord(t[0]))]

        cached.embed_query("a")
        cached.embed_query("b")
        cached.embed_query("c")
        assert cached.get_cache_stats()["size"] == 3

        # Add 4th - should evict 'a' (LRU)
        cached.embed_query("d")
        assert cached.get_cache_stats()["size"] == 3

        # 'a' should be evicted - calling it again should re-compute
        mock_embedder.embed_query.reset_mock()
        mock_embedder.embed_query.side_effect = lambda t: [float(ord(t[0]))]
        cached.embed_query("a")
        mock_embedder.embed_query.assert_called_once_with("a")

    def test_accessing_refreshes_lru_order(self, mock_embedder: MagicMock) -> None:
        '''Test that accessing an entry refreshes its LRU position.'''
        cached = CachedEmbeddings(embedder=mock_embedder, cache_size=3)
        cached._cache = LRUCache(maxsize=3)
        mock_embedder.embed_query.side_effect = lambda t: [float(ord(t[0]))]

        cached.embed_query("a")
        cached.embed_query("b")
        cached.embed_query("c")

        # Access 'a' to refresh its position
        cached.embed_query("a")  # cache hit - refreshes 'a'

        # Now add 'd' - should evict 'b' (now LRU), not 'a'
        mock_embedder.embed_query.reset_mock()
        mock_embedder.embed_query.side_effect = lambda t: [float(ord(t[0]))]
        cached.embed_query("d")

        # 'b' should be evicted
        mock_embedder.embed_query.reset_mock()
        mock_embedder.embed_query.side_effect = lambda t: [float(ord(t[0]))]
        cached.embed_query("b")
        mock_embedder.embed_query.assert_called_once_with("b")

        # 'a' should still be cached
        mock_embedder.embed_query.reset_mock()
        cached.embed_query("a")
        mock_embedder.embed_query.assert_not_called()


class TestGetCacheStats:
    '''Test get_cache_stats method.'''

    def test_initial_stats(self, cached: CachedEmbeddings) -> None:
        '''Test stats are correct initially.'''
        stats = cached.get_cache_stats()
        assert stats == {"hits": 0, "misses": 0, "size": 0, "max_size": 5}

    def test_stats_after_operations(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test stats reflect operations correctly.'''
        mock_embedder.embed_query.side_effect = lambda t: [1.0]
        cached.embed_query("x")  # miss
        cached.embed_query("y")  # miss
        cached.embed_query("x")  # hit

        stats = cached.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2
        assert stats["max_size"] == 5


class TestClearCache:
    '''Test clear_cache method.'''

    def test_clears_all_entries(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test that clear_cache removes all cached entries.'''
        mock_embedder.embed_query.side_effect = lambda t: [1.0]
        cached.embed_query("a")
        cached.embed_query("b")
        assert cached.get_cache_stats()["size"] == 2

        cached.clear_cache()
        assert cached.get_cache_stats()["size"] == 0

    def test_resets_statistics(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test that clear_cache resets hit/miss counters.'''
        mock_embedder.embed_query.side_effect = lambda t: [1.0]
        cached.embed_query("a")
        cached.embed_query("a")

        cached.clear_cache()
        stats = cached.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_miss_after_clear(
        self, cached: CachedEmbeddings, mock_embedder: MagicMock
    ) -> None:
        '''Test that previously cached queries miss after clear.'''
        mock_embedder.embed_query.side_effect = lambda t: [1.0]
        cached.embed_query("a")
        mock_embedder.embed_query.reset_mock()
        mock_embedder.embed_query.side_effect = lambda t: [1.0]

        cached.clear_cache()
        cached.embed_query("a")
        mock_embedder.embed_query.assert_called_once_with("a")
