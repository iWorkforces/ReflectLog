"""Unit tests for BaseOpenAIProvider."""

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.llm_provider_base import (
    BaseOpenAIProvider,
    IStructuredOutputSchema,
)


def create_mock_logger() -> IStructuredLogger:
    """Create a properly typed mock logger for testing."""
    return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))


class FakeSchema:
    """Fake Pydantic-like schema implementing IStructuredOutputSchema."""

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return a minimal JSON schema."""
        return {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        }


def _make_chat_response(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestBaseOpenAIProviderInit:
    """Test __init__ and attribute storage."""

    def test_stores_all_parameters(self) -> None:
        """Test that constructor stores all provided parameters."""
        logger = create_mock_logger()
        provider = BaseOpenAIProvider(
            api_key="sk-test",
            base_url="https://api.example.com",
            model="gpt-4",
            timeout=60.0,
            logger=logger,
        )
        assert provider._api_key == "sk-test"
        assert provider._base_url == "https://api.example.com"
        assert provider._model == "gpt-4"
        assert provider._timeout == 60.0
        assert provider._logger is logger
        assert provider._client is None

    def test_default_timeout(self) -> None:
        """Test default timeout is 30.0."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._timeout == 30.0

    def test_default_logger_none(self) -> None:
        """Test logger defaults to None."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._logger is None


class TestGetClient:
    """Test lazy client initialization."""

    @patch("reflectlog.infrastructure.llm_provider_base.HttpClientFactory")
    @patch("reflectlog.infrastructure.llm_provider_base.AsyncOpenAI")
    def test_creates_client_on_first_call(
        self, mock_openai_cls: MagicMock, mock_http_factory: MagicMock
    ) -> None:
        """Test that _get_client creates AsyncOpenAI on first call."""
        mock_httpx = MagicMock()
        mock_http_factory.get_async_httpx_client.return_value = mock_httpx
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        provider = BaseOpenAIProvider(
            api_key="sk-test",
            base_url="https://api.example.com",
            model="gpt-4",
            timeout=45.0,
        )
        client = provider._get_client()

        assert client is mock_client
        mock_http_factory.get_async_httpx_client.assert_called_once_with(http2=True)
        mock_openai_cls.assert_called_once_with(
            api_key="sk-test",
            base_url="https://api.example.com",
            http_client=mock_httpx,
            timeout=45.0,
        )

    @patch("reflectlog.infrastructure.llm_provider_base.HttpClientFactory")
    @patch("reflectlog.infrastructure.llm_provider_base.AsyncOpenAI")
    def test_reuses_client_on_subsequent_calls(
        self, mock_openai_cls: MagicMock, mock_http_factory: MagicMock
    ) -> None:
        """Test that _get_client returns cached client on second call."""
        mock_openai_cls.return_value = MagicMock()
        mock_http_factory.get_async_httpx_client.return_value = MagicMock()

        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        first = provider._get_client()
        second = provider._get_client()

        assert first is second
        assert mock_openai_cls.call_count == 1


class TestCallLLMWithStructuredOutput:
    """Test the _call_llm_with_structured_output method."""

    async def test_successful_structured_output(self) -> None:
        """Test successful call with json_schema response format."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        response = _make_chat_response('{"score": 0.9}')
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        result = await provider._call_llm_with_structured_output(
            prompt="Rate this",
            response_schema=FakeSchema,
        )

        assert result == {"score": 0.9}
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "m"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 150
        assert call_kwargs["response_format"]["type"] == "json_schema"

    async def test_custom_max_tokens(self) -> None:
        """Test that custom max_tokens is passed through."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        response = _make_chat_response('{"score": 0.5}')
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        await provider._call_llm_with_structured_output(
            prompt="p", response_schema=FakeSchema, max_tokens=300
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 300

    async def test_fallback_to_json_object_on_json_schema_error(self) -> None:
        """Test fallback to json_object when json_schema is unsupported."""
        logger = create_mock_logger()
        provider = BaseOpenAIProvider(
            api_key="k", base_url="http://x", model="m", logger=logger
        )
        mock_client = MagicMock()

        # First call raises json_schema error, second succeeds
        fallback_response = _make_chat_response('{"score": 0.7}')
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("json_schema is not supported"),
                fallback_response,
            ]
        )
        provider._client = mock_client

        result = await provider._call_llm_with_structured_output(
            prompt="p", response_schema=FakeSchema
        )

        assert result == {"score": 0.7}
        assert mock_client.chat.completions.create.call_count == 2
        # Verify fallback call uses json_object format
        fallback_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert fallback_kwargs["response_format"] == {"type": "json_object"}
        # Logger warned about fallback
        cast(MagicMock, logger).warning.assert_called_once()

    async def test_fallback_on_structured_keyword_in_error(self) -> None:
        """Test fallback triggered by 'structured' keyword in error."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        fallback_response = _make_chat_response('{"score": 0.3}')
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("structured output not available"),
                fallback_response,
            ]
        )
        provider._client = mock_client

        result = await provider._call_llm_with_structured_output(
            prompt="p", response_schema=FakeSchema
        )
        assert result == {"score": 0.3}

    async def test_fallback_without_logger(self) -> None:
        """Test fallback works when no logger is set."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        fallback_response = _make_chat_response('{"score": 0.6}')
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("json_schema unsupported"),
                fallback_response,
            ]
        )
        provider._client = mock_client

        result = await provider._call_llm_with_structured_output(
            prompt="p", response_schema=FakeSchema
        )
        assert result == {"score": 0.6}

    async def test_raises_non_fallback_error(self) -> None:
        """Test that non-schema errors are re-raised."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("connection timeout")
        )
        provider._client = mock_client

        with pytest.raises(Exception, match="connection timeout"):
            await provider._call_llm_with_structured_output(
                prompt="p", response_schema=FakeSchema
            )

    async def test_raises_on_empty_response(self) -> None:
        """Test ValueError raised when LLM returns None content."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        response = _make_chat_response("not used")
        response.choices[0].message.content = None
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        with pytest.raises(ValueError, match="Empty response from LLM"):
            await provider._call_llm_with_structured_output(
                prompt="p", response_schema=FakeSchema
            )

    async def test_raises_on_invalid_json(self) -> None:
        """Test json.JSONDecodeError when LLM returns invalid JSON."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        response = _make_chat_response("not valid json")
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        with pytest.raises(json.JSONDecodeError):
            await provider._call_llm_with_structured_output(
                prompt="p", response_schema=FakeSchema
            )

    async def test_schema_name_from_class(self) -> None:
        """Test that schema name is derived from class name (lowered)."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        response = _make_chat_response('{"score": 1.0}')
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        await provider._call_llm_with_structured_output(
            prompt="p", response_schema=FakeSchema
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = call_kwargs["response_format"]
        assert rf["json_schema"]["name"] == "fakeschema"
        assert rf["json_schema"]["strict"] is True

    async def test_prompt_passed_as_user_message(self) -> None:
        """Test that prompt is sent as user role message."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        mock_client = MagicMock()
        response = _make_chat_response('{"score": 0.5}')
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        await provider._call_llm_with_structured_output(
            prompt="test prompt", response_schema=FakeSchema
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]


class TestClampFloat:
    """Test _clamp_float utility method."""

    def test_value_within_range(self) -> None:
        """Test value already in range is unchanged."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._clamp_float(0.5) == 0.5

    def test_value_below_min(self) -> None:
        """Test value below min is clamped to min."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._clamp_float(-0.5) == 0.0

    def test_value_above_max(self) -> None:
        """Test value above max is clamped to max."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._clamp_float(1.5) == 1.0

    def test_custom_range(self) -> None:
        """Test clamping with custom min/max values."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._clamp_float(5.0, min_value=2.0, max_value=8.0) == 5.0
        assert provider._clamp_float(1.0, min_value=2.0, max_value=8.0) == 2.0
        assert provider._clamp_float(10.0, min_value=2.0, max_value=8.0) == 8.0

    def test_boundary_values(self) -> None:
        """Test exact boundary values are returned."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._clamp_float(0.0) == 0.0
        assert provider._clamp_float(1.0) == 1.0


class TestExtractStringField:
    """Test _extract_string_field utility method."""

    def test_existing_field(self) -> None:
        """Test extraction of existing string field."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_string_field({"name": "hello"}, "name") == "hello"

    def test_missing_field_default(self) -> None:
        """Test default empty string for missing field."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_string_field({}, "name") == ""

    def test_custom_default(self) -> None:
        """Test custom default for missing field."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert (
            provider._extract_string_field({}, "name", default="unknown") == "unknown"
        )

    def test_non_string_value_converted(self) -> None:
        """Test non-string values are converted via str()."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_string_field({"n": 42}, "n") == "42"


class TestExtractFloatField:
    """Test _extract_float_field utility method."""

    def test_existing_field_clamped(self) -> None:
        """Test existing float field with clamping."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_float_field({"score": 0.8}, "score") == 0.8

    def test_above_one_clamped(self) -> None:
        """Test value above 1.0 is clamped."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_float_field({"score": 1.5}, "score") == 1.0

    def test_below_zero_clamped(self) -> None:
        """Test value below 0.0 is clamped."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_float_field({"score": -0.5}, "score") == 0.0

    def test_no_clamp(self) -> None:
        """Test extraction without clamping."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_float_field({"v": 2.5}, "v", clamp=False) == 2.5

    def test_missing_field_default(self) -> None:
        """Test default value for missing field."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_float_field({}, "score") == 0.0

    def test_custom_default(self) -> None:
        """Test custom default for missing field."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_float_field({}, "score", default=0.5) == 0.5


class TestExtractBoolField:
    """Test _extract_bool_field utility method."""

    def test_true_value(self) -> None:
        """Test extraction of True value."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_bool_field({"flag": True}, "flag") is True

    def test_false_value(self) -> None:
        """Test extraction of False value."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_bool_field({"flag": False}, "flag") is False

    def test_missing_field_default_false(self) -> None:
        """Test default False for missing field."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_bool_field({}, "flag") is False

    def test_custom_default_true(self) -> None:
        """Test custom default True for missing field."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_bool_field({}, "flag", default=True) is True

    def test_truthy_value(self) -> None:
        """Test truthy non-bool value converted via bool()."""
        provider = BaseOpenAIProvider(api_key="k", base_url="http://x", model="m")
        assert provider._extract_bool_field({"flag": 1}, "flag") is True
        assert provider._extract_bool_field({"flag": 0}, "flag") is False


class TestIStructuredOutputSchemaProtocol:
    """Test that IStructuredOutputSchema protocol works correctly."""

    def test_fake_schema_satisfies_protocol(self) -> None:
        """Test our FakeSchema matches the protocol."""
        assert hasattr(FakeSchema, "model_json_schema")
        schema = FakeSchema.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
