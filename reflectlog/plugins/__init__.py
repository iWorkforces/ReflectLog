"""Plugin system for ReflectLogMCP.

This package provides mechanisms for discovering, registering, and loading
plugins at runtime. It enables extensibility without modifying core code.

Modules:
    discovery: Plugin discovery mechanisms (entry points, directory scan, static)
    registry: Plugin registry for managing registered plugins
    loading: Plugin loading and lifecycle management

Example:
    # Discover plugins from entry points
    discovery = EntryPointDiscovery(
        group="reflectlog.plugins",
        plugin_type=ITool,
    )

    # Create registry and loader
    registry = PluginRegistry[ITool]()
    loader = PluginLoader(discovery, registry)

    # Discover, load, and activate
    await loader.discover()
    await loader.load_all()
    await loader.activate_all()
"""

from .discovery import (
    DiscoveredPlugin,
    PluginDiscoveryStrategy,
    EntryPointDiscovery,
    DirectoryScanDiscovery,
    StaticRegistration,
    CompositeDiscovery,
    PluginDiscoverer,
    load_plugin,
)
from .registry import (
    PluginState,
    PluginCapability,
    PluginMetadata,
    IPluggable,
    PluginRegistry,
    ToolRegistry,
)
from .loading import (
    LifecycleHooks,
    IPluginLifecycle,
    PluginLoader,
)

__all__ = [
    # Discovery
    "DiscoveredPlugin",
    "PluginDiscoveryStrategy",
    "EntryPointDiscovery",
    "DirectoryScanDiscovery",
    "StaticRegistration",
    "CompositeDiscovery",
    "PluginDiscoverer",
    "load_plugin",
    # Registry
    "PluginState",
    "PluginCapability",
    "PluginMetadata",
    "IPluggable",
    "PluginRegistry",
    "ToolRegistry",
    # Loading
    "LifecycleHooks",
    "IPluginLifecycle",
    "PluginLoader",
]
