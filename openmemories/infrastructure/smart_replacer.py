"""Smart memory replacement detector using LLM."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Tuple

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
    """

    api_key: str
    base_url: str
    model: str
    threshold: float = 0.7
    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

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
        )


class SmartReplacer(BaseModel):
    """LLM-based smart memory replacement detector.

    This class determines if a new memory should replace an existing one
    by analyzing semantic similarity and contextual updates using an LLM.

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

    _client: AsyncOpenAI | None = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        """Initialize SmartReplacer with AsyncOpenAI client."""
        super().__init__(**data)

        if self.config.enabled:
            # Initialize async OpenAI client with HTTP/2 support and timeout
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                http_client=DefaultAioHttpClient(),
                timeout=self.config.timeout,
            )

    async def _call_llm_with_structured_output(
        self,
        prompt: str,
    ) -> Any:
        """Call LLM with structured output, falling back to json_object if unsupported.

        Implements exponential backoff retry for transient failures.

        Args:
            prompt: The formatted replacement detection prompt.

        Returns:
            The API response object.

        Raises:
            RuntimeError: If client is not initialized.
            Exception: If all retry attempts fail.
        """
        if self._client is None:
            raise RuntimeError("Smart replacer client is not initialized.")

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
        use_fallback_format = False  # Track if we need to use json_object fallback

        for attempt in range(1, self.config.max_retries + 1):
            try:
                if use_fallback_format:
                    # Use json_object fallback for models that don't support structured outputs
                    return await self._client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=150,
                        response_format={"type": "json_object"},
                    )
                else:
                    # Try structured outputs first (guaranteed schema compliance)
                    return await self._client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,  # Deterministic scoring
                        max_tokens=150,  # Need room for reason field
                        response_format=structured_response_format,
                    )
            except Exception as e:
                error_msg = str(e).lower()

                # Check if this is a structured output compatibility issue
                if not use_fallback_format and (
                    "json_schema" in error_msg or "structured" in error_msg
                ):
                    if self.logger:
                        self.logger.warning(
                            "Model doesn't support structured outputs, falling back to json_object",
                            extra={"model": self.config.model, "error": str(e)},
                        )
                    use_fallback_format = True
                    # Don't count this as a retry attempt, try immediately with fallback
                    continue

                last_exception = e

                # Log the retry attempt
                if self.logger:
                    self.logger.warning(
                        f"Smart replacer LLM call failed (attempt {attempt}/{self.config.max_retries})",
                        extra={
                            "attempt": attempt,
                            "max_retries": self.config.max_retries,
                            "error": str(e),
                        },
                    )

                # If we have more retries, wait with exponential backoff
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** (attempt - 1))
                    if self.logger:
                        self.logger.debug(
                            f"Retrying smart replacer in {delay:.1f}s",
                            extra={"delay": delay, "next_attempt": attempt + 1},
                        )
                    await asyncio.sleep(delay)

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError("Smart replacer LLM call failed after all retries")

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

        if self._client is None:
            return (False, 0.0, "Client not initialized")

        try:
            # Format the replacement detection prompt
            prompt = REPLACEMENT_DETECTION_PROMPT.format(
                old_memory=existing_memory,
                new_memory=new_memory,
            )

            # Call LLM with structured output (or fallback)
            response = await self._call_llm_with_structured_output(prompt)

            # Parse JSON response
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")

            result = json.loads(content)

            # Extract fields with defaults
            should_replace = bool(result.get("should_replace", False))
            confidence = float(result.get("confidence", 0.0))
            reason = str(result.get("reason", "No reason provided"))

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

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
                    },
                )

            return (final_should_replace, confidence, reason)

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(
                    f"Invalid JSON from LLM for replacement detection: {e}",
                    extra={"error": str(e)},
                )
            return (False, 0.0, f"JSON parse error: {e}")

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Smart replacement check failed: {e}",
                    extra={"error": str(e)},
                )
            return (False, 0.0, f"Error: {e}")
