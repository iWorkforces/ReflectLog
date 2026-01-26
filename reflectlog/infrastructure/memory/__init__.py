"""Memory storage implementations for ReflectLogMCP.

This subpackage contains implementations of memory storage backends.
These components handle message persistence and smart replacement detection.

Note: The actual implementations are in the parent infrastructure/
directory. This module re-exports them for backward compatibility.
"""

# Re-export from parent module for backward compatibility
from reflectlog.infrastructure.message_store import MessageStore, MessageRecord
from reflectlog.infrastructure.smart_replacer import (
    SmartReplacer,
    SmartReplacerConfig,
    IReplacementProvider,
    OpenAIReplacementProvider,
    AnthropicReplacementProvider,
    ReplacementDecision,
    create_replacement_provider,
)

__all__ = [
    "MessageStore",
    "MessageRecord",
    "SmartReplacer",
    "SmartReplacerConfig",
    "IReplacementProvider",
    "OpenAIReplacementProvider",
    "AnthropicReplacementProvider",
    "ReplacementDecision",
    "create_replacement_provider",
]
