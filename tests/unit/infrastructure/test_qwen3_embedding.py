"""Unit tests for LangchainQwenEmbeddings."""

import anyio
import os
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from openmemories.infrastructure.qwen3_embedding import (
    EmbedderConfig,
    LangchainQwenEmbeddings,
)


class TestLangchainQwenEmbeddingsInitialization:
    """Test initialization of LangchainQwenEmbeddings."""

    def test_initialization_with_dict_config(self) -> None:
        """Test initialization with dictionary configuration."""
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": "test-api-key",
            "openai_base_url": "https://openrouter.ai/api/v1",
        }

        with (
            patch(
                "openmemories.infrastructure.qwen3_embedding.OpenAI"
            ) as mock_sync_client,
            patch(
                "openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"
            ) as mock_async_client,
        ):
            embeddings = LangchainQwenEmbeddings(config=config)

            assert embeddings.config.model == "qwen/qwen-2.5-3b-instruct"
            assert embeddings.config.embedding_dims == 1536
            # Both clients should be initialized
            mock_sync_client.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1",
                http_client=ANY,
                timeout=60.0,
            )
            mock_async_client.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1",
                http_client=ANY,
                timeout=60.0,
            )

    def test_initialization_with_embedder_config(self) -> None:
        """Test initialization with EmbedderConfig."""
        config = EmbedderConfig(
            model="qwen/qwen-2.5-3b-instruct",
            embedding_dims=1536,
            api_key="test-api-key",
            openai_base_url="https://openrouter.ai/api/v1",
        )

        with (
            patch(
                "openmemories.infrastructure.qwen3_embedding.OpenAI"
            ) as mock_sync_client,
            patch(
                "openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"
            ) as mock_async_client,
        ):
            embeddings = LangchainQwenEmbeddings(config=config)

            assert embeddings.config.model == "qwen/qwen-2.5-3b-instruct"
            assert embeddings.config.embedding_dims == 1536
            # Both clients should be initialized
            mock_sync_client.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1",
                http_client=ANY,
                timeout=60.0,
            )
            mock_async_client.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1",
                http_client=ANY,
                timeout=60.0,
            )

    def test_initialization_with_empty_config(self) -> None:
        """Test initialization with empty configuration."""
        with (
            patch("openmemories.infrastructure.qwen3_embedding.OpenAI"),
            patch("openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"),
        ):
            embeddings = LangchainQwenEmbeddings(config={})
            assert embeddings.config.embedding_dims == 1536  # Default value

    def test_api_key_from_environment(self) -> None:
        """Test API key from environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-api-key"}):
            with (
                patch(
                    "openmemories.infrastructure.qwen3_embedding.OpenAI"
                ) as mock_sync_client,
                patch(
                    "openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"
                ) as mock_async_client,
            ):
                LangchainQwenEmbeddings(config={})
                mock_sync_client.assert_called_once()
                mock_async_client.assert_called_once()
                sync_kwargs = mock_sync_client.call_args[1]
                async_kwargs = mock_async_client.call_args[1]
                assert sync_kwargs["api_key"] == "env-api-key"
                assert async_kwargs["api_key"] == "env-api-key"

    def test_base_url_from_environment(self) -> None:
        """Test base URL from environment variable."""
        with patch.dict(
            os.environ, {"OPENROUTER_BASE_URL": "https://custom.url/api/v1"}
        ):
            with (
                patch(
                    "openmemories.infrastructure.qwen3_embedding.OpenAI"
                ) as mock_sync_client,
                patch(
                    "openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"
                ) as mock_async_client,
            ):
                LangchainQwenEmbeddings(config={})
                mock_sync_client.assert_called_once()
                mock_async_client.assert_called_once()
                sync_kwargs = mock_sync_client.call_args[1]
                async_kwargs = mock_async_client.call_args[1]
                assert sync_kwargs["base_url"] == "https://custom.url/api/v1"
                assert async_kwargs["base_url"] == "https://custom.url/api/v1"

    def test_base_url_default_value(self) -> None:
        """Test default base URL."""
        with patch.dict(os.environ, {}, clear=True):
            with (
                patch(
                    "openmemories.infrastructure.qwen3_embedding.OpenAI"
                ) as mock_sync_client,
                patch(
                    "openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"
                ) as mock_async_client,
            ):
                LangchainQwenEmbeddings(config={})
                mock_sync_client.assert_called_once()
                mock_async_client.assert_called_once()
                sync_kwargs = mock_sync_client.call_args[1]
                async_kwargs = mock_async_client.call_args[1]
                assert sync_kwargs["base_url"] == "https://openrouter.ai/api/v1"
                assert async_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_deprecated_openai_api_base_warning(self) -> None:
        """Test deprecation warning for OPENAI_API_BASE."""
        with patch.dict(os.environ, {"OPENAI_API_BASE": "https://old.url"}):
            with (
                patch("openmemories.infrastructure.qwen3_embedding.OpenAI"),
                patch("openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"),
            ):
                with pytest.warns(DeprecationWarning, match="OPENAI_API_BASE"):
                    LangchainQwenEmbeddings(config={})


class TestSyncEmbedQuery:
    """Test synchronous embed_query method."""

    @pytest.fixture
    def mock_embeddings(self) -> LangchainQwenEmbeddings:
        """Create a mocked LangchainQwenEmbeddings instance."""
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": "test-key",
        }
        with (
            patch("openmemories.infrastructure.qwen3_embedding.OpenAI"),
            patch("openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"),
        ):
            embeddings = LangchainQwenEmbeddings(config=config)
            return embeddings

    def test_embed_query_success(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test successful embedding query."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_embeddings._client = MagicMock()
        mock_embeddings._client.embeddings.create.return_value = mock_response

        result = mock_embeddings.embed_query("test text")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_query_replaces_newlines(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test that newlines are replaced with spaces."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_embeddings._client = MagicMock()
        mock_embeddings._client.embeddings.create.return_value = mock_response

        mock_embeddings.embed_query("test\ntext\nwith\nnewlines")

        # Verify the input was cleaned
        call_kwargs = mock_embeddings._client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == ["test text with newlines"]

    def test_embed_query_client_not_initialized(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test error when client is not initialized."""
        mock_embeddings._client = None

        with pytest.raises(
            RuntimeError, match="Embedding request failed after retries"
        ):
            mock_embeddings.embed_query("test")


class TestSyncEmbedDocuments:
    """Test synchronous embed_documents method."""

    @pytest.fixture
    def mock_embeddings(self) -> LangchainQwenEmbeddings:
        """Create a mocked LangchainQwenEmbeddings instance."""
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": "test-key",
        }
        with (
            patch("openmemories.infrastructure.qwen3_embedding.OpenAI"),
            patch("openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"),
        ):
            embeddings = LangchainQwenEmbeddings(config=config)
            return embeddings

    def test_embed_documents_empty_list(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test embedding empty list returns empty list."""
        result = mock_embeddings.embed_documents([])
        assert result == []

    def test_embed_documents_single_text(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test embedding a single text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_embeddings._client = MagicMock()
        mock_embeddings._client.embeddings.create.return_value = mock_response

        result = mock_embeddings.embed_documents(["text1"])
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    def test_embed_documents_multiple_texts(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test embedding multiple texts."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.2, 0.4, 0.6]),
            MagicMock(embedding=[0.3, 0.6, 0.9]),
        ]

        mock_embeddings._client = MagicMock()
        mock_embeddings._client.embeddings.create.return_value = mock_response

        result = mock_embeddings.embed_documents(["text1", "text2", "text3"])
        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.2, 0.4, 0.6]
        assert result[2] == [0.3, 0.6, 0.9]


class TestAsyncEmbedQuery:
    """Test asynchronous aembed_query method."""

    @pytest.fixture
    def mock_embeddings(self) -> LangchainQwenEmbeddings:
        """Create a mocked LangchainQwenEmbeddings instance."""
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": "test-key",
        }
        with (
            patch("openmemories.infrastructure.qwen3_embedding.OpenAI"),
            patch("openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"),
        ):
            embeddings = LangchainQwenEmbeddings(config=config)
            return embeddings

    @pytest.mark.asyncio
    async def test_aembed_query_success(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test successful async embedding query."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_embeddings._async_client = AsyncMock()
        mock_embeddings._async_client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_embeddings.aembed_query("test text")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_query_replaces_newlines(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test that newlines are replaced in async version."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_embeddings._async_client = AsyncMock()
        mock_embeddings._async_client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        await mock_embeddings.aembed_query("test\ntext")

        call_args = mock_embeddings._async_client.embeddings.create.call_args
        assert call_args[1]["input"] == ["test text"]

    @pytest.mark.asyncio
    async def test_aembed_query_uses_correct_params(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test that aembed_query uses correct parameters."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_embeddings._async_client = AsyncMock()
        mock_embeddings._async_client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        await mock_embeddings.aembed_query("test")

        mock_embeddings._async_client.embeddings.create.assert_called_once_with(
            input=["test"],
            model="qwen/qwen-2.5-3b-instruct",
            dimensions=1536,
            encoding_format="float",
        )

    @pytest.mark.asyncio
    async def test_aembed_query_client_not_initialized(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test error when async client is not initialized."""
        mock_embeddings._async_client = None

        with pytest.raises(
            RuntimeError, match="Async embedding request failed after retries"
        ):
            await mock_embeddings.aembed_query("test")


class TestAsyncEmbedDocuments:
    """Test asynchronous aembed_documents method."""

    @pytest.fixture
    def mock_embeddings(self) -> LangchainQwenEmbeddings:
        """Create a mocked LangchainQwenEmbeddings instance."""
        config = {
            "model": "qwen/qwen-2.5-3b-instruct",
            "embedding_dims": 1536,
            "api_key": "test-key",
        }
        with (
            patch("openmemories.infrastructure.qwen3_embedding.OpenAI"),
            patch("openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"),
        ):
            embeddings = LangchainQwenEmbeddings(config=config)
            return embeddings

    @pytest.mark.asyncio
    async def test_aembed_documents_empty_list(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test async embedding of empty list."""
        result = await mock_embeddings.aembed_documents([])
        assert result == []

    @pytest.mark.asyncio
    async def test_aembed_documents_single_text(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test async embedding of single text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_embeddings._async_client = AsyncMock()
        mock_embeddings._async_client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_embeddings.aembed_documents(["text1"])
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_documents_multiple_texts(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test async embedding of multiple texts with batching.

        With default batch_size=512, all 3 texts fit in a single batch,
        so only 1 API call should be made.
        """
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await anyio.sleep(0.01)  # Simulate async work
            # Return embeddings for all texts in the batch
            input_texts = kwargs.get("input", [])
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)])
                for i in range(len(input_texts))
            ]
            return mock_response

        mock_embeddings._async_client = AsyncMock()
        mock_embeddings._async_client.embeddings.create = mock_create

        result = await mock_embeddings.aembed_documents(["text1", "text2", "text3"])
        assert len(result) == 3
        # With batch_size=512, all 3 texts fit in 1 batch = 1 API call
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_aembed_documents_concurrency_limit(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test that batch concurrency is limited by EMBEDDING_MAX_CONCURRENT_BATCHES.

        With batch_size=1, each text becomes its own batch.
        Default max_concurrent_batches=4 should limit concurrent API calls.
        """
        max_concurrent = 0
        current_concurrent = 0

        async def mock_create(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await anyio.sleep(0.01)
            current_concurrent -= 1
            input_texts = kwargs.get("input", [])
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in input_texts
            ]
            return mock_response

        mock_embeddings._async_client = AsyncMock()
        mock_embeddings._async_client.embeddings.create = mock_create

        # Use batch_size=1 so each text is a separate batch
        # max_concurrent_batches=4 (default) should limit concurrent API calls
        with patch.dict(
            os.environ,
            {"EMBEDDING_BATCH_SIZE": "1", "EMBEDDING_MAX_CONCURRENT_BATCHES": "4"},
        ):
            texts = [f"text{i}" for i in range(10)]
            await mock_embeddings.aembed_documents(texts)

        # Max concurrent should not exceed 4 (EMBEDDING_MAX_CONCURRENT_BATCHES)
        assert max_concurrent <= 4

    @pytest.mark.asyncio
    async def test_aembed_documents_parallel_processing(
        self, mock_embeddings: LangchainQwenEmbeddings
    ) -> None:
        """Test that batches are processed in parallel.

        With batch_size=1, each text becomes its own batch.
        All batches should start within a short time window (parallel).
        """
        call_times: list[float] = []

        async def mock_create(*args, **kwargs):
            import time

            call_times.append(time.time())
            await anyio.sleep(0.01)
            input_texts = kwargs.get("input", [])
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in input_texts
            ]
            return mock_response

        mock_embeddings._async_client = AsyncMock()
        mock_embeddings._async_client.embeddings.create = mock_create

        # Use batch_size=1 so we get multiple batches to parallelize
        with patch.dict(os.environ, {"EMBEDDING_BATCH_SIZE": "1"}):
            await mock_embeddings.aembed_documents(["text1", "text2", "text3"])

        # All calls should start within a very short time window (parallel)
        if len(call_times) > 1:
            time_diff = max(call_times) - min(call_times)
            assert time_diff < 0.1  # All should start within 100ms


class TestConfigurationPriority:
    """Test configuration priority (config vs environment variables)."""

    def test_config_takes_priority_over_env(self) -> None:
        """Test that config values take priority over environment variables."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            config = {"api_key": "config-key"}
            with (
                patch(
                    "openmemories.infrastructure.qwen3_embedding.OpenAI"
                ) as mock_sync_client,
                patch(
                    "openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"
                ) as mock_async_client,
            ):
                LangchainQwenEmbeddings(config=config)
                sync_kwargs = mock_sync_client.call_args[1]
                async_kwargs = mock_async_client.call_args[1]
                assert sync_kwargs["api_key"] == "config-key"
                assert async_kwargs["api_key"] == "config-key"

    def test_env_used_when_config_not_provided(self) -> None:
        """Test that environment variables are used when config not provided."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            with (
                patch(
                    "openmemories.infrastructure.qwen3_embedding.OpenAI"
                ) as mock_sync_client,
                patch(
                    "openmemories.infrastructure.qwen3_embedding.AsyncOpenAI"
                ) as mock_async_client,
            ):
                LangchainQwenEmbeddings(config={})
                sync_kwargs = mock_sync_client.call_args[1]
                async_kwargs = mock_async_client.call_args[1]
                assert sync_kwargs["api_key"] == "env-key"
                assert async_kwargs["api_key"] == "env-key"
