"""Base class for OpenAI-compatible LLM providers.

This module provides BaseOpenAIProvider, which extracts common functionality
shared between OpenAIRerankerProvider and OpenAIReplacementProvider:

- AsyncOpenAI client initialization with DefaultAioHttpClient (HTTP/2 support)
- Structured output with json_schema fallback to json_object
- Retry decorator for exponential backoff on transient errors

Example:
    class MyProvider(BaseOpenAIProvider):
        async def call_llm(self, prompt: str) -> dict:
            response = await self._call_llm_with_structured_output(
                prompt=prompt,
                response_schema=MyResponseSchema,
            )
            return response
"""

import json
from typing import Any, Protocol

from openai import AsyncOpenAI
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)

from reflectlog.application.utils.http_client import HttpClientFactory
from reflectlog.core.logging import IStructuredLogger


class IStructuredOutputSchema(Protocol):
    """Protocol for Pydantic schemas used in structured output."""

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Generate JSON schema for structured output."""
        ...


class BaseOpenAIProvider:
    """Base class for OpenAI-compatible LLM providers.

    Provides common infrastructure:
    - AsyncOpenAI client with HTTP/2 support
    - Structured output with fallback to json_object
    - Safe JSON parsing with clamping

    Subclasses should implement protocol-specific methods (e.g., score_document
    for IRerankerProvider, detect_replacement for IReplacementProvider).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        logger: IStructuredLogger | None = None,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: OpenRouter/OpenAI API key.
            base_url: API base URL.
            model: LLM model identifier.
            timeout: HTTP request timeout in seconds.
            logger: Optional structured logger.
        """
        self._client: AsyncOpenAI | None = None
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._model = model
        self._logger = logger

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            httpx_client = HttpClientFactory.get_async_httpx_client(http2=False)
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx_client,
                timeout=self._timeout,
            )
        return self._client

    async def _call_llm_with_structured_output(
        self,
        prompt: str,
        response_schema: type[IStructuredOutputSchema],
        max_tokens: int = 150,
    ) -> dict[str, Any]:
        """Call LLM with structured output, falling back to json_object if unsupported.

        Tries json_schema mode first for guaranteed schema compliance.
        Falls back to json_object mode if the model doesn't support structured outputs.

        Args:
            prompt: The formatted prompt for the LLM.
            response_schema: Pydantic schema class for structured output.
            max_tokens: Maximum tokens in response (default: 150).

        Returns:
            Parsed JSON dictionary from LLM response.

        Raises:
            Exception: If both structured output and json_object fallback fail.
        """
        # Build structured output response format using Pydantic schema
        schema_name = response_schema.__name__.lower()
        structured_response_format = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema={
                "name": schema_name,
                "strict": True,
                "schema": response_schema.model_json_schema(),
            },
        )

        try:
            # Try structured outputs first (guaranteed schema compliance)
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                response_format=structured_response_format,
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Fallback to simple json_object mode for unsupported models
            if "json_schema" in error_msg or "structured" in error_msg:
                if self._logger:
                    self._logger.warning(
                        "Model doesn't support structured outputs, "
                        "falling back to json_object",
                        extra={"model": self._model, "error": str(e)},
                    )
                client = self._get_client()
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                raise

        # Parse response content
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from LLM")

        result = json.loads(content)
        return result

    def _clamp_float(
        self, value: float, min_value: float = 0.0, max_value: float = 1.0
    ) -> float:
        """Clamp a float value to a specified range.

        Args:
            value: The value to clamp.
            min_value: Minimum allowed value (default: 0.0).
            max_value: Maximum allowed value (default: 1.0).

        Returns:
            Clamped value within [min_value, max_value].
        """
        return max(min_value, min(max_value, value))

    def _extract_string_field(
        self,
        data: dict[str, Any],
        field: str,
        default: str = "",
    ) -> str:
        """Extract a string field from JSON data with default fallback.

        Args:
            data: Parsed JSON dictionary.
            field: Field name to extract.
            default: Default value if field missing (default: "").

        Returns:
            String value or default.
        """
        return str(data.get(field, default))

    def _extract_float_field(
        self,
        data: dict[str, Any],
        field: str,
        default: float = 0.0,
        clamp: bool = True,
    ) -> float:
        """Extract a float field from JSON data with optional clamping.

        Args:
            data: Parsed JSON dictionary.
            field: Field name to extract.
            default: Default value if field missing (default: 0.0).
            clamp: Whether to clamp to [0.0, 1.0] range (default: True).

        Returns:
            Float value (clamped if clamp=True).
        """
        value = float(data.get(field, default))
        return self._clamp_float(value) if clamp else value

    def _extract_bool_field(
        self,
        data: dict[str, Any],
        field: str,
        default: bool = False,
    ) -> bool:
        """Extract a boolean field from JSON data with default fallback.

        Args:
            data: Parsed JSON dictionary.
            field: Field name to extract.
            default: Default value if field missing (default: False).

        Returns:
            Boolean value or default.
        """
        return bool(data.get(field, default))
