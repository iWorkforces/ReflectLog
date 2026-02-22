"""Type stubs for claude_agent_sdk package."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

class TextBlock:
    """Text content block in assistant messages."""

    text: str

class AssistantMessage:
    """Assistant message containing content blocks."""

    content: list[TextBlock | object]

class ResultMessage:
    """Result message produced by the Claude Agent SDK."""

    is_error: bool
    subtype: str

@dataclass
class ClaudeAgentOptions:
    """Options passed to Claude Agent SDK queries."""

    model: str | None = None
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None
    permission_mode: str = "bypassPermissions"

def query(
    *,
    prompt: str,
    options: ClaudeAgentOptions,
) -> AsyncIterator[AssistantMessage | ResultMessage]: ...

__all__ = [
    "AssistantMessage",
    "ClaudeAgentOptions",
    "ResultMessage",
    "TextBlock",
    "query",
]
