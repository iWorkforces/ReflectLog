"""Configuration module for ReflectLogMCP Server."""

from .prompts import (
    INSTRUCTIONS_HEADER,
    MCP_INSTRUCTIONS,
    REPLACEMENT_DETECTION_PROMPT,
    SCORING_PROMPT,
    SCORING_PROMPT_WITH_AGE,
    TOOL_ORDER,
    build_instructions,
)
from .settings import Config, TransportMode, config
from .validation import (
    ConfigurationValidator,
    ValidationError,
    validate_config,
)

__all__ = [
    # SCREAMING_SNAKE_CASE
    "INSTRUCTIONS_HEADER",
    "MCP_INSTRUCTIONS",
    "REPLACEMENT_DETECTION_PROMPT",
    "SCORING_PROMPT",
    "SCORING_PROMPT_WITH_AGE",
    "TOOL_ORDER",
    # CamelCase
    "Config",
    "ConfigurationValidator",
    "TransportMode",
    "ValidationError",
    # snake_case
    "build_instructions",
    "config",
    "validate_config",
]
