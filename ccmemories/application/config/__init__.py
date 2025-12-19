"""Configuration module for CCMemoriesMCP Server."""

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

__all__ = [
    "Config",
    "TransportMode",
    "config",
    "MCP_INSTRUCTIONS",
    "INSTRUCTIONS_HEADER",
    "REPLACEMENT_DETECTION_PROMPT",
    "SCORING_PROMPT",
    "SCORING_PROMPT_WITH_AGE",
    "TOOL_ORDER",
    "build_instructions",
]
