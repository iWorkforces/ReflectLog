"""Tool registration protocols for ReflectLogMCP.

This module defines protocols for MCP tool implementations and registries.
The tool abstraction enables discoverable, configurable tools that can
be added without modifying core application logic.
"""

from typing import Protocol, runtime_checkable, Optional, Any, Callable
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete definition of an MCP tool."""

    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable[..., Any]


@dataclass
class IToolResult(BaseModel):
    """Result from a tool execution."""

    success: bool
    content: list[str]
    error: str | None = None


@runtime_checkable
class ITool(Protocol):
    """Protocol for MCP tool implementations.

    This protocol defines the interface that all MCP tools must implement.
    Tools are self-contained units of functionality that can be discovered
    and registered at runtime.

    Attributes:
        name: Tool identifier for registration and invocation.
        description: Human-readable description for MCP clients.
    """

    @property
    def name(self) -> str:
        """Tool identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """OpenAI-style JSON Schema for parameters."""
        ...

    async def execute(self, **kwargs: Any) -> IToolResult:
        """Execute the tool with given arguments.

        Args:
            kwargs: Tool arguments from MCP client.

        Returns:
            Tool execution result.
        """
        ...

    def get_instruction_snippet(self) -> str:
        """Get documentation snippet for MCP instructions.

        Returns:
            Documentation string for tool inclusion in server instructions.
        """
        ...


@runtime_checkable
class IToolRegistry(Protocol):
    """Protocol for tool registration and discovery.

    This protocol defines the interface for managing tool registrations.
    Registries support static registration, discovery, and dynamic loading
    from plugins.
    """

    def register(self, tool: ITool) -> None:
        """Register a tool instance.

        Args:
            tool: Tool to register.

        Raises:
            ValueError: If tool with same name already registered.
        """
        ...

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: Tool name to remove.

        Returns:
            True if removed, False if not found.
        """
        ...

    def get(self, name: str) -> Optional[ITool]:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance or None if not found.
        """
        ...

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        ...

    def list_all(self) -> list[ITool]:
        """List all registered tools.

        Returns:
            List of tool instances.
        """
        ...

    def clear(self) -> None:
        """Remove all registrations."""
        ...

    def count(self) -> int:
        """Get number of registered tools.

        Returns:
            Number of registered tools.
        """
        ...


@runtime_checkable
class IToolLoader(Protocol):
    """Protocol for loading tools from external sources.

    This protocol defines the interface for discovering and loading
    tools from plugins, entry points, or other external sources.
    """

    async def discover(self) -> list[type[ITool]]:
        """Discover tool classes from configured sources.

        Returns:
            List of tool classes ready for instantiation.
        """
        ...

    async def load(self, tool_class: type[ITool]) -> ITool:
        """Instantiate a tool class.

        Args:
            tool_class: Tool class to instantiate.

        Returns:
            Tool instance.
        """
        ...
