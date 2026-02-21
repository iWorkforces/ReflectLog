'''Configuration module for ReflectLogMCP Server.'''

from .presets import (
    apply_preset_to_env,
    get_active_preset,
    get_preset_summary,
)
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
from .validation import ConfigurationValidator, ValidationError

__all__ = [
    # SCREAMING_SNAKE_CASE
    'INSTRUCTIONS_HEADER',
    'MCP_INSTRUCTIONS',
    'REPLACEMENT_DETECTION_PROMPT',
    'SCORING_PROMPT',
    'SCORING_PROMPT_WITH_AGE',
    'TOOL_ORDER',
    # CamelCase
    'Config',
    'ConfigurationValidator',
    'TransportMode',
    'ValidationError',
    # snake_case
    'apply_preset_to_env',
    'build_instructions',
    'config',
    'get_active_preset',
    'get_preset_summary',
]
