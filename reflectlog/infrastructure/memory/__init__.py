"""Memory storage implementations for ReflectLogMCP.

This subpackage contains implementations of memory storage backends.
These components handle message persistence and smart replacement detection.

Note: The actual implementations are in the parent infrastructure/
directory. This module re-exports them for backward compatibility.
"""

# Re-export from parent module for backward compatibility
from reflectlog.infrastructure.message_store import MessageRecord, MessageStore
from reflectlog.infrastructure.smart_replacer import (
    AnthropicReplacementProvider,
    IReplacementProvider,
    OpenAIReplacementProvider,
    ReplacementDecision,
    SmartReplacer,
    SmartReplacerConfig,
    create_replacement_provider,
)

__all__ = [
    "AnthropicReplacementProvider",
    "create_replacement_provider",
    "IReplacementProvider",
    "MessageRecord",
    "MessageStore",
    "OpenAIReplacementProvider",
    "ReplacementDecision",
    "SmartReplacer",
    "SmartReplacerConfig",
]
