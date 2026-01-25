"""Plugin registry for managing registered plugins.

This module provides the PluginRegistry class that tracks registered
plugins and their capabilities, enabling queries like "find all tools"
or "find reranker plugins".
"""

from dataclasses import dataclass, field
from typing import Optional, Generic, TypeVar, Protocol, runtime_checkable, cast
from enum import Enum
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Get current UTC time in a timezone-aware format.

    This function replaces the deprecated datetime.utcnow() and returns
    a timezone-aware datetime object in UTC.
    """
    return datetime.now(timezone.utc)


class PluginState(Enum):
    """Plugin lifecycle states."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    UNLOADED = "unloaded"
    ERROR = "error"


@dataclass
class PluginCapability:
    """A capability provided by a plugin."""

    name: str
    version: str = "0.0.0"


@dataclass
class PluginMetadata:
    """Metadata about a registered plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    capabilities: list[PluginCapability] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    state: PluginState = PluginState.DISCOVERED
    error_message: Optional[str] = None
    discovered_at: datetime = field(default_factory=utc_now)
    loaded_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None


T = TypeVar("T")


@runtime_checkable
class IPluggable(Protocol):
    """Protocol for pluggable components."""

    @property
    def plugin_name(self) -> str:
        """Plugin identifier."""
        ...

    @property
    def plugin_version(self) -> str:
        """Plugin version."""
        ...


class PluginRegistry(Generic[T]):
    """Registry for managing plugins and their capabilities.

    This class provides methods for registering, unregistering, and
    querying plugins by various criteria.

    Example:
        registry = PluginRegistry[ITool]()
        registry.register(my_tool)
        tools = registry.list_by_type(ITool)
    """

    def __init__(self):
        """Initialize empty plugin registry."""
        self._plugins: dict[str, PluginMetadata] = {}
        self._instances: dict[str, T] = {}

    def register(
        self,
        plugin: T,
        metadata: Optional[PluginMetadata] = None,
    ) -> PluginMetadata:
        """Register a plugin instance.

        Args:
            plugin: Plugin instance to register.
            metadata: Optional metadata (auto-generated if not provided).

        Returns:
            PluginMetadata for the registered plugin.
        """
        if metadata is None:
            # Auto-generate metadata from plugin
            # Use getattr with default to satisfy type checker
            name = (
                getattr(plugin, "plugin_name", None)
                or getattr(plugin, "__class__", type(plugin)).__name__
            )

            version = (
                getattr(plugin, "plugin_version", None)
                or getattr(plugin, "__class__", type(plugin)).__module__.split(".")[-1]
            )

            metadata = PluginMetadata(
                name=name,
                version=version,
            )

        metadata.state = PluginState.LOADED
        metadata.loaded_at = utc_now()

        self._plugins[metadata.name] = metadata
        self._instances[metadata.name] = plugin

        return metadata

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name.

        Args:
            name: Plugin name to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        if name not in self._plugins:
            return False

        del self._plugins[name]
        if name in self._instances:
            del self._instances[name]

        return True

    def get(self, name: str) -> Optional[T]:
        """Get a plugin instance by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance or None if not found.
        """
        return self._instances.get(name)

    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name.

        Args:
            name: Plugin name.

        Returns:
            PluginMetadata or None if not found.
        """
        return self._plugins.get(name)

    def list_all(self) -> list[str]:
        """List all registered plugin names.

        Returns:
            List of plugin names.
        """
        return list(self._plugins.keys())

    def list_by_state(self, state: PluginState) -> list[str]:
        """List plugins by state.

        Args:
            state: Plugin state to filter by.

        Returns:
            List of plugin names in the given state.
        """
        return [name for name, meta in self._plugins.items() if meta.state == state]

    def list_by_capability(self, capability_name: str) -> list[str]:
        """List plugins that provide a capability.

        Args:
            capability_name: Name of capability.

        Returns:
            List of plugin names with the capability.
        """
        return [
            name
            for name, meta in self._plugins.items()
            if any(cap.name == capability_name for cap in meta.capabilities)
        ]

    def list_by_type(self, plugin_type: type) -> list[T]:
        """List plugins that are instances of a type.

        Args:
            plugin_type: Type to filter by.

        Returns:
            List of plugin instances of the given type.
        """
        return [
            plugin
            for plugin in self._instances.values()
            if isinstance(plugin, plugin_type)
        ]

    def activate(self, name: str) -> bool:
        """Activate a registered plugin.

        Args:
            name: Plugin name to activate.

        Returns:
            True if activated, False if not found or already active.
        """
        if name not in self._plugins:
            return False

        meta = self._plugins[name]
        if meta.state != PluginState.LOADED:
            return False

        meta.state = PluginState.ACTIVATED
        meta.activated_at = utc_now()
        return True

    def deactivate(self, name: str) -> bool:
        """Deactivate a registered plugin.

        Args:
            name: Plugin name to deactivate.

        Returns:
            True if deactivated, False if not found or not active.
        """
        if name not in self._plugins:
            return False

        meta = self._plugins[name]
        if meta.state != PluginState.ACTIVATED:
            return False

        meta.state = PluginState.DEACTIVATED
        return True

    def set_error(self, name: str, error_message: str) -> bool:
        """Set plugin to error state.

        Args:
            name: Plugin name.
            error_message: Error description.

        Returns:
            True if set, False if not found.
        """
        if name not in self._plugins:
            return False

        meta = self._plugins[name]
        meta.state = PluginState.ERROR
        meta.error_message = error_message
        return True

    def clear(self) -> None:
        """Remove all registered plugins."""
        self._plugins.clear()
        self._instances.clear()

    def count(self) -> int:
        """Get number of registered plugins.

        Returns:
            Number of registered plugins.
        """
        return len(self._plugins)

    @property
    def plugins(self) -> dict[str, PluginMetadata]:
        """Get copy of plugin metadata registry."""
        return self._plugins.copy()

    @property
    def instances(self) -> dict[str, T]:
        """Get copy of plugin instances registry."""
        return self._instances.copy()


class ToolRegistry(PluginRegistry[T]):
    """Specialized registry for MCP tools.

    This registry provides additional methods for querying and
    managing MCP tool plugins.
    """

    def __init__(self):
        """Initialize tool registry."""
        super().__init__()
        self._tool_names: set[str] = set()

    def register_tool(
        self,
        tool: T,
        name: str,
        description: str = "",
        version: str = "0.0.0",
    ) -> PluginMetadata:
        """Register a tool with metadata.

        Args:
            tool: Tool instance.
            name: Tool name.
            description: Tool description.
            version: Tool version.

        Returns:
            PluginMetadata for the registered tool.
        """
        if name in self._tool_names:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tool_names.add(name)

        metadata = PluginMetadata(
            name=name,
            version=version,
            description=description,
            capabilities=[
                PluginCapability(name="tool", version=version),
            ],
        )

        return self.register(tool, metadata)

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: Tool name.

        Returns:
            True if unregistered, False if not found.
        """
        if name not in self._tool_names:
            return False

        self._tool_names.discard(name)
        return self.unregister(name)

    def get_tool(self, name: str) -> Optional[T]:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance or None if not found.
        """
        return self.get(name)

    def list_tool_names(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return sorted(self._tool_names)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._tool_names
