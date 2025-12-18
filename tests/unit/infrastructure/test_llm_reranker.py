"""Unit tests for LLMReranker."""

import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from ccmemories.infrastructure.llm_reranker import (
    LLMReranker,
    LLMRerankerConfig,
    RelevanceScore,
)


class TestRelevanceScore:
    """Test RelevanceScore Pydantic schema."""

    def test_valid_score_middle_range(self) -> None:
        """Test valid score in middle of range."""
        score = RelevanceScore(score=0.85)
        assert score.score == 0.85

    def test_valid_score_zero(self) -> None:
        """Test minimum valid score."""
        score = RelevanceScore(score=0.0)
        assert score.score == 0.0

    def test_valid_score_one(self) -> None:
        """Test maximum valid score."""
        score = RelevanceScore(score=1.0)
        assert score.score == 1.0

    def test_invalid_score_below_zero(self) -> None:
        """Test that score below 0.0 is rejected."""
        with pytest.raises(ValueError):
            RelevanceScore(score=-0.1)

    def test_invalid_score_above_one(self) -> None:
        """Test that score above 1.0 is rejected."""
        with pytest.raises(ValueError):
            RelevanceScore(score=1.1)

    def test_json_schema_generation(self) -> None:
        """Test that JSON schema is generated correctly."""
        schema = RelevanceScore.model_json_schema()
        assert "score" in schema.get("properties", {})
        assert schema["properties"]["score"]["type"] == "number"

    def test_json_serialization(self) -> None:
        """Test JSON serialization round-trip."""
        score = RelevanceScore(score=0.75)
        json_str = score.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["score"] == 0.75


class TestLLMRerankerConfig:
    """Test LLMRerankerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
        )

        assert config.score_threshold == 0.5
        assert config.max_concurrency == 5
        assert config.timeout == 30.0

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://custom.url/api/v1",
            model="custom/model",
            score_threshold=0.7,
            max_concurrency=10,
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.url/api/v1"
        assert config.model == "custom/model"
        assert config.score_threshold == 0.7
        assert config.max_concurrency == 10

    def test_from_app_config(self) -> None:
        """Test factory method from application config."""
        mock_app_config = MagicMock()
        mock_app_config.openrouter_api_key.get_secret_value.return_value = "api-key"
        mock_app_config.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_app_config.llm_model = "x-ai/grok-4.1-fast"
        mock_app_config.search_score_threshold = 0.6
        mock_app_config.rerank_max_concurrency = 8

        config = LLMRerankerConfig.from_app_config(mock_app_config)

        assert config.api_key == "api-key"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "x-ai/grok-4.1-fast"
        assert config.score_threshold == 0.6
        assert config.max_concurrency == 8


class TestLLMRerankerInitialization:
    """Test LLMReranker initialization."""

    def test_initialization_creates_async_client(self) -> None:
        """Test that initialization creates AsyncOpenAI client."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
        )

        with patch(
            "ccmemories.infrastructure.llm_reranker.AsyncOpenAI"
        ) as mock_async_client:
            reranker = LLMReranker(config=config)

            mock_async_client.assert_called_once_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                http_client=ANY,
                timeout=30.0,
            )
            assert reranker._client is not None


class TestScoreSingle:
    """Test _score_single method."""

    @pytest.fixture
    def mock_reranker(self) -> LLMReranker:
        """Create a mocked LLMReranker instance."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
        )

        with patch("ccmemories.infrastructure.llm_reranker.AsyncOpenAI"):
            reranker = LLMReranker(config=config)
            return reranker

    @pytest.mark.asyncio
    async def test_score_single_success(self, mock_reranker: LLMReranker) -> None:
        """Test successful single document scoring."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.85}'

        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        doc, score = await mock_reranker._score_single(
            query="Python tutorials",
            document="Learn Python programming basics",
            fallback_score=0.5,
        )

        assert doc == "Learn Python programming basics"
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_single_clamps_out_of_range(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test that out-of-range scores are clamped to 0-1."""
        # Test score > 1.0
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 1.5}'

        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        doc, score = await mock_reranker._score_single(
            query="test", document="test doc", fallback_score=0.5
        )
        assert score == 1.0

        # Test score < 0.0
        mock_response.choices[0].message.content = '{"score": -0.5}'
        doc, score = await mock_reranker._score_single(
            query="test", document="test doc", fallback_score=0.5
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_single_invalid_json_uses_fallback(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test that invalid JSON response uses fallback score."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"

        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_logger = MagicMock()
        mock_reranker.logger = mock_logger

        doc, score = await mock_reranker._score_single(
            query="test", document="test doc", fallback_score=0.7
        )

        assert score == 0.7
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_single_empty_response_uses_fallback(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test that empty response uses fallback score."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_logger = MagicMock()
        mock_reranker.logger = mock_logger

        doc, score = await mock_reranker._score_single(
            query="test", document="test doc", fallback_score=0.6
        )

        assert score == 0.6

    @pytest.mark.asyncio
    async def test_score_single_api_error_uses_fallback(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test that API error uses fallback score."""
        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        mock_logger = MagicMock()
        mock_reranker.logger = mock_logger

        doc, score = await mock_reranker._score_single(
            query="test", document="test doc", fallback_score=0.8
        )

        assert score == 0.8
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_single_client_not_initialized(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test error when client is not initialized."""
        mock_reranker._client = None

        mock_logger = MagicMock()
        mock_reranker.logger = mock_logger

        doc, score = await mock_reranker._score_single(
            query="test", document="test doc", fallback_score=0.5
        )

        # Should use fallback when client not initialized
        assert score == 0.5


class TestRerank:
    """Test rerank method."""

    @pytest.fixture
    def mock_reranker(self) -> LLMReranker:
        """Create a mocked LLMReranker instance."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            score_threshold=0.5,
            max_concurrency=5,
            batch_normalize=False,  # Disable to test raw score behavior
        )

        with patch("ccmemories.infrastructure.llm_reranker.AsyncOpenAI"):
            reranker = LLMReranker(config=config)
            return reranker

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self, mock_reranker: LLMReranker) -> None:
        """Test reranking empty candidate list."""
        result = await mock_reranker.rerank("test query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_successful_scoring(self, mock_reranker: LLMReranker) -> None:
        """Test successful reranking with multiple candidates."""

        async def mock_score_single(query, document, fallback_score):
            scores = {
                "doc1": 0.9,
                "doc2": 0.3,  # Below threshold
                "doc3": 0.7,
            }
            return (document, scores.get(document, fallback_score))

        mock_reranker._score_single = mock_score_single  # type: ignore[method-assign]

        candidates = [
            ("doc1", 0.8),
            ("doc2", 0.7),
            ("doc3", 0.6),
        ]

        result = await mock_reranker.rerank("test query", candidates)

        # doc2 should be filtered out (score 0.3 < threshold 0.5)
        assert len(result) == 2
        # Results should be sorted by score descending
        assert result[0] == ("doc1", 0.9)
        assert result[1] == ("doc3", 0.7)

    @pytest.mark.asyncio
    async def test_rerank_threshold_filtering(self, mock_reranker: LLMReranker) -> None:
        """Test that results below threshold are filtered out."""
        mock_reranker.config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test-model",
            score_threshold=0.7,
            max_concurrency=5,
            batch_normalize=False,  # Disable to test raw score threshold behavior
        )

        async def mock_score_single(query, document, fallback_score):
            return (document, 0.5)  # All scores below threshold

        mock_reranker._score_single = mock_score_single  # type: ignore[method-assign]

        candidates = [("doc1", 0.8), ("doc2", 0.7)]
        result = await mock_reranker.rerank("test query", candidates)

        assert len(result) == 0  # All filtered out

    @pytest.mark.asyncio
    async def test_rerank_preserves_documents(self, mock_reranker: LLMReranker) -> None:
        """Test that document content is preserved through reranking."""

        async def mock_score_single(query, document, fallback_score):
            return (document, 0.8)

        mock_reranker._score_single = mock_score_single  # type: ignore[method-assign]

        original_doc = "This is a very long document with specific content"
        candidates = [(original_doc, 0.5)]

        result = await mock_reranker.rerank("test query", candidates)

        assert len(result) == 1
        assert result[0][0] == original_doc

    @pytest.mark.asyncio
    async def test_rerank_concurrent_execution(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test that scoring runs concurrently."""
        import time

        call_times = []

        async def mock_score_single(query, document, fallback_score):
            import anyio

            call_times.append(time.time())
            await anyio.sleep(0.01)  # Simulate async work
            return (document, 0.8)

        mock_reranker._score_single = mock_score_single  # type: ignore[method-assign]

        candidates = [(f"doc{i}", 0.5) for i in range(5)]

        await mock_reranker.rerank("test query", candidates)

        # All calls should start within a short time window (parallel)
        assert len(call_times) == 5
        if len(call_times) > 1:
            time_diff = max(call_times) - min(call_times)
            assert time_diff < 0.1  # All should start within 100ms

    @pytest.mark.asyncio
    async def test_rerank_concurrency_limit(self, mock_reranker: LLMReranker) -> None:
        """Test that concurrency is limited by semaphore."""
        import anyio

        max_concurrent = 0
        current_concurrent = 0

        async def mock_score_single(query, document, fallback_score):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await anyio.sleep(0.01)
            current_concurrent -= 1
            return (document, 0.8)

        mock_reranker._score_single = mock_score_single  # type: ignore[method-assign]
        mock_reranker.config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test-model",
            score_threshold=0.5,
            max_concurrency=3,  # Limit to 3
            batch_normalize=False,  # Disable to test raw score behavior
        )

        candidates = [(f"doc{i}", 0.5) for i in range(10)]

        await mock_reranker.rerank("test query", candidates)

        # Max concurrent should not exceed configured limit
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_rerank_with_logger(self, mock_reranker: LLMReranker) -> None:
        """Test that reranking logs appropriately."""

        async def mock_score_single(query, document, fallback_score):
            return (document, 0.8)

        mock_reranker._score_single = mock_score_single  # type: ignore[method-assign]

        mock_logger = MagicMock()
        mock_reranker.logger = mock_logger

        candidates = [("doc1", 0.5)]
        await mock_reranker.rerank("test query", candidates)

        # Should log debug message about completion
        mock_logger.debug.assert_called()


class TestStructuredOutputFallback:
    """Test structured output with fallback to json_object mode."""

    @pytest.fixture
    def mock_reranker(self) -> LLMReranker:
        """Create a mocked LLMReranker instance."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
        )

        with patch("ccmemories.infrastructure.llm_reranker.AsyncOpenAI"):
            reranker = LLMReranker(config=config)
            return reranker

    @pytest.mark.asyncio
    async def test_structured_output_success(self, mock_reranker: LLMReranker) -> None:
        """Test that structured output call succeeds."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.85}'

        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await mock_reranker._call_llm_with_structured_output("test prompt")

        # Verify structured output format was used
        mock_reranker._client.chat.completions.create.assert_called_once()
        call_kwargs = mock_reranker._client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert "json_schema" in call_kwargs["response_format"]
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_fallback_on_json_schema_not_supported(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test fallback to json_object mode when structured outputs not supported."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.75}'

        # First call fails with json_schema error, second succeeds
        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("json_schema is not supported for this model"),
                mock_response,
            ]
        )

        mock_logger = MagicMock()
        mock_reranker.logger = mock_logger

        await mock_reranker._call_llm_with_structured_output("test prompt")

        # Verify fallback was called
        assert mock_reranker._client.chat.completions.create.call_count == 2

        # Verify second call used json_object mode
        second_call_kwargs = (
            mock_reranker._client.chat.completions.create.call_args_list[1][1]
        )
        assert second_call_kwargs["response_format"]["type"] == "json_object"

        # Verify warning was logged
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_structured_output_error(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test fallback when error mentions 'structured'."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.8}'

        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("structured outputs are not available"),
                mock_response,
            ]
        )

        mock_logger = MagicMock()
        mock_reranker.logger = mock_logger

        await mock_reranker._call_llm_with_structured_output("test prompt")

        # Verify fallback occurred
        assert mock_reranker._client.chat.completions.create.call_count == 2
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_on_unrelated_error(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test that unrelated errors are not caught for fallback."""
        mock_reranker._client = AsyncMock()
        mock_reranker._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Network timeout error")
        )

        with pytest.raises(Exception, match="Network timeout"):
            await mock_reranker._call_llm_with_structured_output("test prompt")

    @pytest.mark.asyncio
    async def test_client_not_initialized_raises_error(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test that RuntimeError is raised when client is not initialized."""
        mock_reranker._client = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await mock_reranker._call_llm_with_structured_output("test prompt")


class TestLLMRerankerIntegration:
    """Integration-style tests for full reranking flow."""

    @pytest.mark.asyncio
    async def test_full_reranking_flow(self) -> None:
        """Test complete reranking flow with mocked LLM."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            score_threshold=0.5,
            batch_normalize=False,  # Disable to test raw score behavior
        )

        with patch("ccmemories.infrastructure.llm_reranker.AsyncOpenAI"):
            reranker = LLMReranker(config=config, logger=MagicMock())

            # Mock the client to return different scores
            call_count = 0

            async def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                scores = [0.9, 0.4, 0.7, 0.3]  # Only 0.9 and 0.7 pass threshold
                score = scores[(call_count - 1) % len(scores)]
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = json.dumps({"score": score})
                return mock_response

            reranker._client = AsyncMock()
            reranker._client.chat.completions.create = mock_create

            candidates = [
                ("High relevance doc", 0.8),
                ("Low relevance doc", 0.7),
                ("Medium relevance doc", 0.6),
                ("Very low relevance doc", 0.5),
            ]

            result = await reranker.rerank("Python tutorials", candidates)

            # Should have 2 results (scores 0.9 and 0.7 pass threshold 0.5)
            assert len(result) == 2
            # Should be sorted by score descending
            assert result[0][1] == 0.9
            assert result[1][1] == 0.7
