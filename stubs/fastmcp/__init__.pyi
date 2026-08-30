"""Type stubs for fastmcp library."""

from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeAlias, TypeVar, overload

from fastmcp.client.client import CallToolResult

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


class Context:
    """FastMCP context for tool execution.

    Provides access to request metadata and client communication.
    """

    @property
    def request_id(self) -> str | None:
        """Request identifier."""
        ...

    async def report_progress(
        self,
        progress: float,
        total: float | None = None,
    ) -> None:
        """Report progress for long-running operations.

        Args:
            progress: Current progress value.
            total: Optional total value for percentage calculation.
        """
        ...

    def log(
        self,
        level: str,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log a message to the MCP client.

        Args:
            level: Log level ('debug', 'info', 'warning', 'error').
            message: Log message.
            **kwargs: Additional structured data.
        """
        ...

class Tool:
    """FastMCP Tool with registered function."""

    name: str
    description: str
    fn: Callable[..., Awaitable[object]]

class FastMCP:
    """FastMCP server instance."""
    def __init__(self, name: str, instructions: str | None = None) -> None: ...
    @overload
    def tool(
        self,
        func: Callable[_P, _R],
        **kwargs: object,
    ) -> Callable[_P, _R]: ...
    @overload
    def tool(
        self,
        func: None = None,
        **kwargs: object,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
    def tool(
        self,
        func: Callable[_P, _R] | None = None,
        **kwargs: object,
    ) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
    def run(self, **kwargs: object) -> None: ...
    def add_middleware(self, middleware: object) -> object: ...
    @property
    def _tool_manager(self) -> ToolManager: ...

class ToolManager:
    """Manages registered tools."""

    _tools: dict[str, Tool]

class Client:
    """FastMCP Client for connecting to MCP servers."""
    def __init__(self, server_url: str) -> None: ...
    async def __aenter__(self) -> Client: ...
    async def __aexit__(self, *args: object) -> None: ...
    async def list_tools(self) -> list[Tool]: ...
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, JSONValue] | dict[str, object] | None = None,
    ) -> CallToolResult: ...

__all__ = ["Client", "FastMCP", "Tool", "ToolManager"]
