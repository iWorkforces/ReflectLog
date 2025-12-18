"""Type stubs for fastmcp.client.client module."""

from typing import Any

class CallToolResult:
    """Result from calling an MCP tool."""

    content: list[Any]
    isError: bool

    def __init__(
        self, content: list[Any] | None = None, isError: bool = False
    ) -> None: ...

__all__ = ["CallToolResult"]
