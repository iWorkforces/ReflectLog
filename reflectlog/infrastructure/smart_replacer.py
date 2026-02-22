"""Smart memory replacement detector using LLM."""

import asyncio
from dataclasses import dataclass
import json
import re
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from reflectlog.application.config import REPLACEMENT_DETECTION_PROMPT, Config
from reflectlog.application.utils.retry import async_retry_with_backoff
from reflectlog.infrastructure.llm_provider_base import (
    BaseOpenAIProvider,
)


class ReplacementDecision(BaseModel):
    """Schema for LLM replacement detection response.

    Used with OpenAI Structured Outputs to guarantee valid JSON responses
    from the smart replacer.

    Attributes:
        should_replace: Whether the new memory should replace the old one.
        confidence: Confidence score from 0.0 to 1.0.
        reason: Brief explanation of the decision.
    """

    should_replace: bool = Field(
        ...,
        description="Whether the new memory should replace the old one",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    reason: str = Field(
        ...,
        description="Brief explanation of the decision",
    )


class AnthropicReplacementResponse(TypedDict, total=False):
    should_replace: bool
    confidence: float
    reason: str


@dataclass(frozen=True)
class SmartReplacerConfig:
    """Configuration for SmartReplacer.

    Attributes:
        api_key: OpenRouter API key for authentication.
        base_url: OpenRouter API base URL.
        model: LLM model identifier (e.g., 'x-ai/grok-4.1-fast').
        threshold: Minimum confidence score to trigger replacement (0.0-1.0).
        enabled: Whether smart replacement is enabled.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retry attempts for LLM calls.
        retry_delay: Base delay in seconds for exponential backoff.
        provider: LLM provider ('openai' or 'anthropic').
    """

    api_key: str
    base_url: str
    model: str
    threshold: float = 0.7
    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    provider: str = "openai"  # Provider: "openai" or "anthropic"

    @classmethod
    def from_app_config(cls, config: Config) -> SmartReplacerConfig:
        """Create SmartReplacerConfig from application Config.

        Args:
            config: Application configuration object.

        Returns:
            SmartReplacerConfig instance configured from app settings.
        """
        return cls(
            api_key=config.openrouter_api_key.get_secret_value(),
            base_url=config.openrouter_base_url,
            model=config.llm_model,
            threshold=config.smart_replace_threshold,
            enabled=config.enable_smart_replace,
            max_retries=config.smart_replace_max_retries,
            retry_delay=config.smart_replace_retry_delay,
            provider=config.llm_provider,
        )


class IReplacementProvider(Protocol):
    """Protocol for replacement detection providers.

    Defines the interface for LLM providers used in smart memory replacement.
    """

    async def detect_replacement(
        self,
        prompt: str,
        max_retries: int,
        retry_delay: float,
    ) -> tuple[bool, float, str]:
        """Detect if replacement should occur.

        Args:
            prompt: The formatted replacement detection prompt.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay in seconds for exponential backoff.

        Returns:
            Tuple of (should_replace, confidence, reason):
                - should_replace: Whether replacement is recommended
                - confidence: Confidence score (0.0-1.0)
                - reason: Brief explanation of the decision
        """
        ...


class OpenAIReplacementProvider(BaseOpenAIProvider):
    """OpenAI/OpenRouter-based replacement detection provider.

    Uses AsyncOpenAI client with structured JSON output for reliable parsing.
    Supports fallback to json_object mode for models without structured output.

    Inherits common functionality from BaseOpenAIProvider:
    - AsyncOpenAI client initialization with HTTP/2 support
    - Structured output with fallback to json_object
    - Safe JSON parsing with clamping

    Retry logic is handled by the retry decorator.
    """

    async def _detect_replacement_once(
        self,
        prompt: str,
    ) -> tuple[bool, float, str]:
        """Call LLM with structured output once.

        Args:
            prompt: The formatted replacement detection prompt.

        Returns:
            Tuple of (should_replace, confidence, reason).

        Raises:
            Exception: If LLM call fails.
        """
        result = await self._call_llm_with_structured_output(
            prompt=prompt,
            response_schema=ReplacementDecision,
            max_tokens=150,
        )

        should_replace = self._extract_bool_field(
            result, "should_replace", default=False
        )
        confidence = self._extract_float_field(result, "confidence", default=0.0)
        reason = self._extract_string_field(
            result, "reason", default="No reason provided"
        )

        return (should_replace, confidence, reason)

    async def detect_replacement(
        self,
        prompt: str,
        max_retries: int,
        retry_delay: float,
    ) -> tuple[bool, float, str]:
        """Detect if replacement should occur using OpenAI API.

        Args:
            prompt: The formatted replacement detection prompt.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay in seconds for exponential backoff.

        Returns:
            Tuple of (should_replace, confidence, reason).
        """
        try:
            retry = async_retry_with_backoff(
                max_retries=max_retries, base_delay=retry_delay
            )
            return await retry(self._detect_replacement_once)(prompt)

        except Exception as e:
            if self._logger:
                self._logger.warning(
                    f"OpenAI replacement detection failed after {max_retries} retries",
                    extra={
                        "max_retries": max_retries,
                        "error": str(e),
                    },
                )

            error_msg = str(e) if e else "Unknown error"
            return (False, 0.0, f"Error: {error_msg}")


class AnthropicReplacementProvider:
    """Anthropic Claude-based replacement detection provider.

    Uses Claude Agent SDK via utility module for LLM calls.
    Parses JSON from plain text responses with multiple fallback strategies.
    """

    def __init__(
        self,
        model: str | None = None,
        logger: Any = None,
    ):
        """Initialize Anthropic replacement provider.

        Calls init_credentials() to set up OAuth credentials.

        Args:
            model: LLM model identifier (passed to generate_content).
            logger: Optional structured logger.
        """
        super().__init__()

        # Lazy import to avoid dependency issues
        from reflectlog.utility import init_credentials

        _ = init_credentials(verbose=False)
        self._model = model
        self._logger = logger

    def _extract_json_from_response(
        self, response_text: str
    ) -> AnthropicReplacementResponse:
        """Extract JSON from plain text response.

        Handles multiple response formats:
        1. Pure JSON
        2. Markdown code blocks (```json ... ```)
        3. Embedded JSON in text

        Args:
            response_text: Raw response text from LLM.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If JSON extraction fails.
        """
        text = response_text.strip()

        # Strategy 1: Direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Markdown code block
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(code_block_pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find embedded JSON object
        json_pattern = r"\{[\s\S]*?\}"
        matches = list(re.finditer(json_pattern, text))
        for match in matches:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not extract JSON from response: {text[:200]}")

    async def detect_replacement(
        self,
        prompt: str,
        max_retries: int,
        retry_delay: float,
    ) -> tuple[bool, float, str]:
        """Detect if replacement should occur using Anthropic Claude.

        Args:
            prompt: The formatted replacement detection prompt.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay in seconds for exponential backoff.

        Returns:
            Tuple of (should_replace, confidence, reason).
        """
        # Lazy import to avoid dependency issues
        from reflectlog.utility import generate_content

        last_exception: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                # Call generate_content with model parameter
                response_text = await generate_content(
                    prompt=prompt,
                    model=self._model,
                    allowed_tools=[],
                )

                # Parse JSON from response
                result = self._extract_json_from_response(response_text)
                should_replace = bool(result.get("should_replace", False))
                confidence = float(result.get("confidence", 0.0))
                reason = str(result.get("reason", "No reason provided"))
                confidence = max(0.0, min(1.0, confidence))

                return (should_replace, confidence, reason)

            except Exception as e:
                last_exception = e

                if self._logger:
                    self._logger.warning(
                        f"Anthropic replacement detection failed "
                        f"(attempt {attempt}/{max_retries})",
                        extra={
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "error": str(e),
                        },
                    )

                if attempt < max_retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    if self._logger:
                        self._logger.debug(
                            f"Retrying in {delay:.1f}s",
                            extra={"delay": delay, "next_attempt": attempt + 1},
                        )
                    await asyncio.sleep(delay)

        # All retries exhausted - return safe defaults
        error_msg = str(last_exception) if last_exception else "Unknown error"
        return (False, 0.0, f"Error: {error_msg}")


def create_replacement_provider(
    config: SmartReplacerConfig,
    logger: Any = None,
) -> IReplacementProvider:
    """Create a replacement provider based on configuration.

    Factory function that returns the appropriate provider implementation
    based on the provider setting in config.

    Args:
        config: SmartReplacerConfig with provider selection.
        logger: Optional structured logger.

    Returns:
        An IReplacementProvider implementation.

    Raises:
        ValueError: If provider is not supported.
    """
    if config.provider == "openai":
        return OpenAIReplacementProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            logger=logger,
        )
    elif config.provider == "anthropic":
        return AnthropicReplacementProvider(
            model=config.model,
            logger=logger,
        )
    else:
        raise ValueError(
            f"Unsupported provider: '{config.provider}'. "
            f"Valid options: 'openai', 'anthropic'"
        )


class SmartReplacer(BaseModel):
    """LLM-based smart memory replacement detector.

    This class determines if a new memory should replace an existing one
    by analyzing semantic similarity and contextual updates using an LLM.

    Supports multiple LLM providers via the provider abstraction:
    - 'openai': OpenRouter/OpenAI API with structured JSON output
    - 'anthropic': Claude Agent SDK via utility module

    Attributes:
        config: SmartReplacerConfig with API credentials and settings.
        logger: Optional structured logger for debug/info memory details.

    Example:
        >>> config = SmartReplacerConfig.from_app_config(app_config)
        >>> replacer = SmartReplacer(config=config, logger=logger)
        >>> should_replace, confidence, reason = await replacer.check_replacement(
        ...     new_memory="I don't like cats anymore",
        ...     existing_memory="I like cats"
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SmartReplacerConfig
    logger: Any = None

    _provider: IReplacementProvider | None = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        """Initialize SmartReplacer with appropriate provider."""
        super().__init__(**data)

        if self.config.enabled:
            self._provider = create_replacement_provider(self.config, self.logger)

    async def check_replacement(
        self,
        new_memory: str,
        existing_memory: str,
    ) -> tuple[bool, float, str]:
        """Check if new memory should replace existing memory.

        Uses LLM to analyze whether the new memory semantically replaces
        the existing memory (e.g., updated preference, contradictory statement).

        Args:
            new_memory: The new memory being added.
            existing_memory: An existing memory to compare against.

        Returns:
            Tuple of (should_replace, confidence, reason):
                - should_replace: True if replacement should occur
                - confidence: LLM confidence score (0.0-1.0)
                - reason: Brief explanation of the decision
        """
        if not self.config.enabled:
            return (False, 0.0, "Smart replacement disabled")

        if self._provider is None:
            return (False, 0.0, "Provider not initialized")

        try:
            # Format the replacement detection prompt
            prompt = REPLACEMENT_DETECTION_PROMPT.format(
                old_memory=existing_memory,
                new_memory=new_memory,
            )

            # Delegate to provider
            (
                should_replace,
                confidence,
                reason,
            ) = await self._provider.detect_replacement(
                prompt=prompt,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )

            # Only trigger replacement if confidence meets threshold
            final_should_replace = (
                should_replace and confidence >= self.config.threshold
            )

            if self.logger:
                self.logger.debug(
                    f"Smart replacer decision: should_replace={should_replace}, "
                    f"confidence={confidence:.2f}, threshold={self.config.threshold}",
                    extra={
                        "should_replace": should_replace,
                        "final_should_replace": final_should_replace,
                        "confidence": confidence,
                        "threshold": self.config.threshold,
                        "reason": reason[:100],
                        "provider": self.config.provider,
                    },
                )

            return (final_should_replace, confidence, reason)

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(
                    f"Invalid JSON from LLM for replacement detection: {e}",
                    extra={"error": str(e), "provider": self.config.provider},
                )
            return (False, 0.0, f"JSON parse error: {e}")

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Smart replacement check failed: {e}",
                    extra={"error": str(e), "provider": self.config.provider},
                )
            return (False, 0.0, f"Error: {e}")
