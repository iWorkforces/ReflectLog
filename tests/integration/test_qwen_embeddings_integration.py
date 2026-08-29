'''Integration tests for LangchainQwenEmbeddings with real API calls.'''

import os

import pytest

from reflectlog.infrastructure.embeddings.qwen3_embedding import LangchainQwenEmbeddings

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RUN_REAL_API_TESTS = os.getenv("RUN_REAL_API_TESTS") == "1"

# Skip all tests unless explicitly enabled with a real API key.
pytestmark = pytest.mark.skipif(
    not RUN_REAL_API_TESTS or not OPENROUTER_API_KEY,
    reason="Set RUN_REAL_API_TESTS=1 and OPENROUTER_API_KEY to run real API tests",
)


@pytest.mark.integration
class TestRealAPIEmbedQuery:
    '''Test embed_query with real API calls.'''

    @pytest.fixture
    def embeddings(self) -> LangchainQwenEmbeddings:
        '''Create LangchainQwenEmbeddings instance with real configuration.'''
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "openai_base_url": os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        }
        return LangchainQwenEmbeddings(config=config)

    def test_real_api_embed_query(self, embeddings: LangchainQwenEmbeddings) -> None:
        '''Test real API call for embed_query.'''
        result = embeddings.embed_query("Hello, world!")

        assert isinstance(result, list)
        assert len(result) == 1536  # Expected embedding dimensions
        assert all(isinstance(x, float) for x in result)

    def test_real_api_embed_query_with_newlines(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test that newlines are handled correctly in real API calls.'''
        text_with_newlines = "Hello\nWorld\nThis is a test"
        result = embeddings.embed_query(text_with_newlines)

        assert isinstance(result, list)
        assert len(result) == 1536


@pytest.mark.integration
class TestRealAPIEmbedDocuments:
    '''Test embed_documents with real API calls.'''

    @pytest.fixture
    def embeddings(self) -> LangchainQwenEmbeddings:
        '''Create LangchainQwenEmbeddings instance with real configuration.'''
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "openai_base_url": os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        }
        return LangchainQwenEmbeddings(config=config)

    def test_real_api_embed_documents_single(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test real API call for single document.'''
        texts = ["This is a test document."]
        result = embeddings.embed_documents(texts)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 1536

    def test_real_api_embed_documents_multiple(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test real API call for multiple documents.'''
        texts = [
            "First document about machine learning.",
            "Second document about natural language processing.",
            "Third document about embeddings.",
        ]
        result = embeddings.embed_documents(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) == 1536 for emb in result)

    def test_real_api_embed_documents_empty(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test that empty list returns empty result.'''
        result = embeddings.embed_documents([])
        assert result == []

    def test_real_api_embed_documents_batch(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test batch processing with real API.'''
        texts = [f"Document {i} for testing batch processing." for i in range(10)]
        result = embeddings.embed_documents(texts)

        assert isinstance(result, list)
        assert len(result) == 10
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) == 1536 for emb in result)


@pytest.mark.integration
class TestRealAPIAsyncEmbedQuery:
    '''Test aembed_query with real async API calls.'''

    @pytest.fixture
    def embeddings(self) -> LangchainQwenEmbeddings:
        '''Create LangchainQwenEmbeddings instance with real configuration.'''
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "openai_base_url": os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        }
        return LangchainQwenEmbeddings(config=config)

    @pytest.mark.asyncio
    async def test_real_api_aembed_query(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test real async API call for aembed_query.'''
        result = await embeddings.aembed_query("Hello, async world!")

        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_real_api_aembed_query_multiple_calls(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test multiple async API calls.'''
        import anyio

        results: list[list[float]] = [[] for _ in range(3)]
        queries = ["First query", "Second query", "Third query"]

        async def run_query(index: int, query: str) -> None:
            results[index] = await embeddings.aembed_query(query)

        async with anyio.create_task_group() as tg:
            for i, query in enumerate(queries):
                tg.start_soon(run_query, i, query)

        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
        assert all(len(result) == 1536 for result in results)


@pytest.mark.integration
class TestRealAPIAsyncEmbedDocuments:
    '''Test aembed_documents with real async API calls.'''

    @pytest.fixture
    def embeddings(self) -> LangchainQwenEmbeddings:
        '''Create LangchainQwenEmbeddings instance with real configuration.'''
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "openai_base_url": os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        }
        return LangchainQwenEmbeddings(config=config)

    @pytest.mark.asyncio
    async def test_real_api_aembed_documents(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test real async API call for aembed_documents.'''
        texts = [
            "Async document one.",
            "Async document two.",
            "Async document three.",
        ]
        result = await embeddings.aembed_documents(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) == 1536 for emb in result)

    @pytest.mark.asyncio
    async def test_real_api_aembed_documents_large_batch(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test async batch processing with many documents.'''
        texts = [f"Document {i} for async batch testing." for i in range(20)]
        result = await embeddings.aembed_documents(texts)

        assert isinstance(result, list)
        assert len(result) == 20
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) == 1536 for emb in result)


@pytest.mark.integration
class TestSyncVsAsync:
    '''Compare sync and async implementations.'''

    @pytest.fixture
    def embeddings(self) -> LangchainQwenEmbeddings:
        '''Create LangchainQwenEmbeddings instance with real configuration.'''
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "openai_base_url": os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        }
        return LangchainQwenEmbeddings(config=config)

    @pytest.mark.asyncio
    async def test_sync_and_async_produce_same_results(
        self, embeddings: LangchainQwenEmbeddings
    ) -> None:
        '''Test that sync and async methods produce identical results.'''
        text = "Test text for comparison"

        sync_result = embeddings.embed_query(text)
        async_result = await embeddings.aembed_query(text)

        # Results should be very similar (may have minor floating point differences)
        assert len(sync_result) == len(async_result)
        # Check that most values are very close
        differences = sum(
            abs(a - b) for a, b in zip(sync_result, async_result, strict=True)
        )
        avg_difference = differences / len(sync_result)
        assert avg_difference < 0.01  # Average difference should be very small
