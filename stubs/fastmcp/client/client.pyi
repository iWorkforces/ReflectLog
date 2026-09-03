"""Type stubs for fastmcp.client.client module."""

from typing import Literal

class TextContent:
    """Text payload returned by an MCP tool."""

    type: Literal["text"]
    text: str

class ImageContent:
    """Image payload returned by an MCP tool."""

    type: Literal["image"]
    data: str
    mimeType: str
    text: str

class EmbeddedResource:
    """Resource payload returned by an MCP tool."""

    type: Literal["resource"]
    uri: str
    mimeType: str | None
    text: str

type ToolContent = TextContent | ImageContent | EmbeddedResource

class CallToolResult:
    """Result from calling an MCP tool."""

    content: list[ToolContent]
    isError: bool

    def __init__(
        self,
        content: list[ToolContent] | None = None,
        isError: bool = False,
    ) -> None: ...

__all__ = [
    "CallToolResult",
    "EmbeddedResource",
    "ImageContent",
    "TextContent",
    "ToolContent",
]
