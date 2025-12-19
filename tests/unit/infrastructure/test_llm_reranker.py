"""Unit tests for LLMReranker."""

import json
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest

from ccmemories.infrastructure.llm_reranker import (
    AnthropicRerankerProvider,
    LLMReranker,
    LLMRerankerConfig,
    OpenAIRerankerProvider,
    RelevanceScore,
    create_reranker_provider,
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
        assert config.provider == "anthropic"  # Default provider

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://custom.url/api/v1",
            model="custom/model",
            score_threshold=0.7,
            max_concurrency=10,
            provider="openai",
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.url/api/v1"
        assert config.model == "custom/model"
        assert config.score_threshold == 0.7
        assert config.max_concurrency == 10
        assert config.provider == "openai"

    def test_from_app_config(self) -> None:
        """Test factory method from application config."""
        mock_app_config = MagicMock()
        mock_app_config.openrouter_api_key.get_secret_value.return_value = "api-key"
        mock_app_config.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_app_config.llm_model = "x-ai/grok-4.1-fast"
        mock_app_config.search_score_threshold = 0.6
        mock_app_config.rerank_max_concurrency = 8
        mock_app_config.reranker_min_results = 0
        mock_app_config.reranker_batch_normalize = True
        mock_app_config.llm_provider = "openai"

        config = LLMRerankerConfig.from_app_config(mock_app_config)

        assert config.api_key == "api-key"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "x-ai/grok-4.1-fast"
        assert config.score_threshold == 0.6
        assert config.max_concurrency == 8
        assert config.provider == "openai"


class TestCreateRerankerProvider:
    """Test create_reranker_provider factory function."""

    def test_create_openai_provider(self) -> None:
        """Test creating OpenAI provider."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            provider="openai",
        )

        with patch(
            "ccmemories.infrastructure.llm_reranker.AsyncOpenAI"
        ) as mock_client_class:
            provider = create_reranker_provider(config)

            assert isinstance(provider, OpenAIRerankerProvider)
            mock_client_class.assert_called_once()

    def test_create_anthropic_provider(self) -> None:
        """Test creating Anthropic provider."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="claude-3-sonnet",
            provider="anthropic",
        )

        with patch("ccmemories.utility.init_credentials") as mock_init:
            provider = create_reranker_provider(config)

            assert isinstance(provider, AnthropicRerankerProvider)
            mock_init.assert_called_once_with(verbose=False)

    def test_create_unsupported_provider_raises(self) -> None:
        """Test that unsupported provider raises ValueError."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test-model",
            provider="unsupported",
        )

        with pytest.raises(ValueError, match="Unsupported provider"):
            create_reranker_provider(config)


class TestOpenAIRerankerProvider:
    """Test OpenAIRerankerProvider class."""

    @pytest.fixture
    def mock_provider(self) -> OpenAIRerankerProvider:
        """Create a mocked OpenAIRerankerProvider instance."""
        with patch(
            "ccmemories.infrastructure.llm_reranker.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            provider = OpenAIRerankerProvider(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="x-ai/grok-4.1-fast",
            )
            # Replace the client with our mock for testing
            provider._client = mock_client
            return provider

    @pytest.mark.asyncio
    async def test_score_document_success(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test successful document scoring."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.85}'

        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        doc, score = await mock_provider.score_document(
            query="Python tutorials",
            document="Learn Python programming basics",
            fallback_score=0.5,
        )

        assert doc == "Learn Python programming basics"
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_document_clamps_out_of_range(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test that out-of-range scores are clamped to 0-1."""
        # Test score > 1.0
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 1.5}'

        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        doc, score = await mock_provider.score_document(
            query="test", document="test doc", fallback_score=0.5
        )
        assert score == 1.0

        # Test score < 0.0
        mock_response.choices[0].message.content = '{"score": -0.5}'
        doc, score = await mock_provider.score_document(
            query="test", document="test doc", fallback_score=0.5
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_document_invalid_json_uses_fallback(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test that invalid JSON response uses fallback score."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"

        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_logger = MagicMock()
        mock_provider._logger = mock_logger

        doc, score = await mock_provider.score_document(
            query="test", document="test doc", fallback_score=0.7
        )

        assert score == 0.7
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_document_empty_response_uses_fallback(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test that empty response uses fallback score."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_logger = MagicMock()
        mock_provider._logger = mock_logger

        doc, score = await mock_provider.score_document(
            query="test", document="test doc", fallback_score=0.6
        )

        assert score == 0.6

    @pytest.mark.asyncio
    async def test_score_document_api_error_uses_fallback(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test that API error uses fallback score."""
        mock_provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        mock_logger = MagicMock()
        mock_provider._logger = mock_logger

        doc, score = await mock_provider.score_document(
            query="test", document="test doc", fallback_score=0.8
        )

        assert score == 0.8
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_with_structured_output_success(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test that structured output call succeeds."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.85}'

        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await mock_provider._call_llm_with_structured_output("test prompt")

        # Verify structured output format was used
        mock_provider._client.chat.completions.create.assert_called_once()
        call_kwargs = mock_provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert "json_schema" in call_kwargs["response_format"]
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_fallback_on_json_schema_not_supported(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test fallback to json_object mode when structured outputs not supported."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.75}'

        # First call fails with json_schema error, second succeeds
        mock_provider._client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("json_schema is not supported for this model"),
                mock_response,
            ]
        )

        mock_logger = MagicMock()
        mock_provider._logger = mock_logger

        await mock_provider._call_llm_with_structured_output("test prompt")

        # Verify fallback was called
        assert mock_provider._client.chat.completions.create.call_count == 2

        # Verify second call used json_object mode
        second_call_kwargs = (
            mock_provider._client.chat.completions.create.call_args_list[1][1]
        )
        assert second_call_kwargs["response_format"]["type"] == "json_object"

        # Verify warning was logged
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_structured_output_error(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test fallback when error mentions 'structured'."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.8}'

        mock_provider._client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("structured outputs are not available"),
                mock_response,
            ]
        )

        mock_logger = MagicMock()
        mock_provider._logger = mock_logger

        await mock_provider._call_llm_with_structured_output("test prompt")

        # Verify fallback occurred
        assert mock_provider._client.chat.completions.create.call_count == 2
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_on_unrelated_error(
        self, mock_provider: OpenAIRerankerProvider
    ) -> None:
        """Test that unrelated errors are not caught for fallback."""
        mock_provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Network timeout error")
        )

        with pytest.raises(Exception, match="Network timeout"):
            await mock_provider._call_llm_with_structured_output("test prompt")


class TestAnthropicRerankerProvider:
    """Test AnthropicRerankerProvider class."""

    @pytest.fixture
    def mock_provider(self) -> AnthropicRerankerProvider:
        """Create a mocked AnthropicRerankerProvider instance."""
        with patch("ccmemories.utility.init_credentials") as mock_init:
            provider = AnthropicRerankerProvider(
                model="claude-3-sonnet",
                logger=MagicMock(),
            )
            mock_init.assert_called_once_with(verbose=False)
            return provider

    def test_extract_json_from_response_pure_json(
        self, mock_provider: AnthropicRerankerProvider
    ) -> None:
        """Test extracting JSON from pure JSON response."""
        response = '{"score": 0.85}'
        result = mock_provider._extract_json_from_response(response)
        assert result["score"] == 0.85

    def test_extract_json_from_response_markdown_block(
        self, mock_provider: AnthropicRerankerProvider
    ) -> None:
        """Test extracting JSON from markdown code block."""
        response = """Here is the score:
```json
{"score": 0.75}
```
"""
        result = mock_provider._extract_json_from_response(response)
        assert result["score"] == 0.75

    def test_extract_json_from_response_embedded(
        self, mock_provider: AnthropicRerankerProvider
    ) -> None:
        """Test extracting embedded JSON from text."""
        response = 'The relevance score is {"score": 0.9} based on analysis.'
        result = mock_provider._extract_json_from_response(response)
        assert result["score"] == 0.9

    def test_extract_json_from_response_failure(
        self, mock_provider: AnthropicRerankerProvider
    ) -> None:
        """Test that extraction fails on non-JSON text."""
        response = "This is just plain text with no JSON"
        with pytest.raises(ValueError, match="Could not extract JSON"):
            mock_provider._extract_json_from_response(response)

    @pytest.mark.asyncio
    async def test_score_document_success(
        self, mock_provider: AnthropicRerankerProvider
    ) -> None:
        """Test successful document scoring with Anthropic."""
        with patch("ccmemories.utility.generate_content") as mock_generate:
            mock_generate.return_value = '{"score": 0.85}'

            doc, score = await mock_provider.score_document(
                query="Python tutorials",
                document="Learn Python programming basics",
                fallback_score=0.5,
            )

            assert doc == "Learn Python programming basics"
            assert score == 0.85
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_document_error_uses_fallback(
        self, mock_provider: AnthropicRerankerProvider
    ) -> None:
        """Test that errors use fallback score."""
        with patch("ccmemories.utility.generate_content") as mock_generate:
            mock_generate.side_effect = Exception("API Error")

            doc, score = await mock_provider.score_document(
                query="test",
                document="test doc",
                fallback_score=0.6,
            )

            assert score == 0.6
            mock_provider._logger.warning.assert_called_once()


class TestLLMRerankerInitialization:
    """Test LLMReranker initialization."""

    def test_initialization_creates_openai_provider(self) -> None:
        """Test that initialization creates OpenAI provider when configured."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            provider="openai",
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
            assert reranker._provider is not None
            assert isinstance(reranker._provider, OpenAIRerankerProvider)

    def test_initialization_creates_anthropic_provider(self) -> None:
        """Test that initialization creates Anthropic provider when configured."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="claude-3-sonnet",
            provider="anthropic",
        )

        with patch("ccmemories.utility.init_credentials") as mock_init:
            reranker = LLMReranker(config=config)

            mock_init.assert_called_once_with(verbose=False)
            assert reranker._provider is not None
            assert isinstance(reranker._provider, AnthropicRerankerProvider)


class TestScoreSingle:
    """Test _score_single method."""

    @pytest.fixture
    def mock_reranker(self) -> LLMReranker:
        """Create a mocked LLMReranker instance."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            provider="openai",
        )

        with patch("ccmemories.infrastructure.llm_reranker.AsyncOpenAI"):
            reranker = LLMReranker(config=config)
            return reranker

    @pytest.mark.asyncio
    async def test_score_single_success(self, mock_reranker: LLMReranker) -> None:
        """Test successful single document scoring."""
        # Mock the provider's score_document method
        mock_reranker._provider = Mock()
        mock_reranker._provider.score_document = AsyncMock(
            return_value=("Learn Python programming basics", 0.85)
        )

        doc, score = await mock_reranker._score_single(
            query="Python tutorials",
            document="Learn Python programming basics",
            fallback_score=0.5,
        )

        assert doc == "Learn Python programming basics"
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_single_provider_not_initialized(
        self, mock_reranker: LLMReranker
    ) -> None:
        """Test fallback when provider is not initialized."""
        mock_reranker._provider = None

        doc, score = await mock_reranker._score_single(
            query="test", document="test doc", fallback_score=0.5
        )

        # Should use fallback when provider not initialized
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
            provider="openai",
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
            provider="openai",
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
            provider="openai",
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


class TestLLMRerankerIntegration:
    """Integration-style tests for full reranking flow."""

    @pytest.mark.asyncio
    async def test_full_reranking_flow(self) -> None:
        """Test complete reranking flow with mocked provider."""
        config = LLMRerankerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            score_threshold=0.5,
            batch_normalize=False,  # Disable to test raw score behavior
            provider="openai",
        )

        with patch("ccmemories.infrastructure.llm_reranker.AsyncOpenAI"):
            reranker = LLMReranker(config=config, logger=MagicMock())

            # Mock the provider's score_document method
            call_count = 0
            scores = [0.9, 0.4, 0.7, 0.3]  # Only 0.9 and 0.7 pass threshold

            async def mock_score_document(query, document, fallback_score):
                nonlocal call_count
                score = scores[call_count % len(scores)]
                call_count += 1
                return (document, score)

            reranker._provider.score_document = mock_score_document  # type: ignore[method-assign]

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
