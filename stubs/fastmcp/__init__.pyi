"""Type stubs for fastmcp library."""

from typing import Any, Callable

class Tool:
    """FastMCP Tool with registered function."""

    name: str
    fn: Callable[..., Any]

class FastMCP:
    """FastMCP server instance."""
    def __init__(self, name: str, instructions: str | None = None) -> None: ...
    def tool(self, func: Callable[..., Any] | None = None, **kwargs: Any) -> Any: ...
    def run(self, **kwargs: Any) -> None: ...
    @property
    def _tool_manager(self) -> ToolManager: ...

class ToolManager:
    """Manages registered tools."""

    _tools: dict[str, Tool]

class Client:
    """FastMCP Client for connecting to MCP servers."""
    def __init__(self, server_url: str) -> None: ...
    async def __aenter__(self) -> Client: ...
    async def __aexit__(self, *args: Any) -> None: ...
    async def list_tools(self) -> list[Any]: ...
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> Any: ...

__all__ = ["FastMCP", "Tool", "ToolManager", "Client"]
