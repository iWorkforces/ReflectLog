"""Smart memory replacement detector using LLM."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from openai import AsyncOpenAI, DefaultAioHttpClient
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ccmemories.application.config import REPLACEMENT_DETECTION_PROMPT, Config


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
    def from_app_config(cls, config: Config) -> "SmartReplacerConfig":
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
            provider=config.smart_replace_provider,
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
    ) -> Tuple[bool, float, str]:
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


class OpenAIReplacementProvider:
    """OpenAI/OpenRouter-based replacement detection provider.

    Uses AsyncOpenAI client with structured JSON output for reliable parsing.
    Supports fallback to json_object mode for models without structured output.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        logger: Any = None,
    ):
        """Initialize OpenAI replacement provider.

        Args:
            api_key: OpenRouter/OpenAI API key.
            base_url: API base URL.
            model: LLM model identifier.
            timeout: HTTP request timeout in seconds.
            logger: Optional structured logger.
        """
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAioHttpClient(),
            timeout=timeout,
        )
        self._model = model
        self._logger = logger

    async def detect_replacement(
        self,
        prompt: str,
        max_retries: int,
        retry_delay: float,
    ) -> Tuple[bool, float, str]:
        """Detect if replacement should occur using OpenAI API.

        Args:
            prompt: The formatted replacement detection prompt.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay in seconds for exponential backoff.

        Returns:
            Tuple of (should_replace, confidence, reason).
        """
        # Build structured output response format using Pydantic schema
        structured_response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "replacement_decision",
                "strict": True,
                "schema": ReplacementDecision.model_json_schema(),
            },
        }

        last_exception: Exception | None = None
        use_fallback_format = False

        for attempt in range(1, max_retries + 1):
            try:
                if use_fallback_format:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=150,
                        response_format={"type": "json_object"},
                    )
                else:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=150,
                        response_format=structured_response_format,
                    )

                # Parse response
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Empty response from LLM")

                result = json.loads(content)
                should_replace = bool(result.get("should_replace", False))
                confidence = float(result.get("confidence", 0.0))
                reason = str(result.get("reason", "No reason provided"))
                confidence = max(0.0, min(1.0, confidence))

                return (should_replace, confidence, reason)

            except Exception as e:
                error_msg = str(e).lower()

                # Check if this is a structured output compatibility issue
                if not use_fallback_format and (
                    "json_schema" in error_msg or "structured" in error_msg
                ):
                    if self._logger:
                        self._logger.warning(
                            "Model doesn't support structured outputs, "
                            "falling back to json_object",
                            extra={"model": self._model, "error": str(e)},
                        )
                    use_fallback_format = True
                    continue

                last_exception = e

                if self._logger:
                    self._logger.warning(
                        f"OpenAI replacement detection failed "
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
        # Lazy import to avoid dependency issues
        from ccmemories.utility import init_credentials

        init_credentials(verbose=False)
        self._model = model
        self._logger = logger

    def _extract_json_from_response(self, response_text: str) -> dict[str, Any]:
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
    ) -> Tuple[bool, float, str]:
        """Detect if replacement should occur using Anthropic Claude.

        Args:
            prompt: The formatted replacement detection prompt.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay in seconds for exponential backoff.

        Returns:
            Tuple of (should_replace, confidence, reason).
        """
        # Lazy import to avoid dependency issues
        from ccmemories.utility import generate_content

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
        logger: Optional structured logger for debug/info messages.

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
    ) -> Tuple[bool, float, str]:
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
