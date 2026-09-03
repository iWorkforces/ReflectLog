"""Plugin loading and lifecycle management.

This module provides the PluginLoader class that handles plugin lifecycle:
loading, initialization, activation, deactivation, and unloading.
"""

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from .discovery import DiscoveredPlugin, PluginDiscoverer, PluginDiscoveryStrategy
from .registry import PluginRegistry, PluginState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class LifecycleHooks:
    """Hooks for plugin lifecycle events."""

    on_load: Callable[[str], None] | None = None
    on_initialize: Callable[[str], None] | None = None
    on_activate: Callable[[str], None] | None = None
    on_deactivate: Callable[[str], None] | None = None
    on_unload: Callable[[str], None] | None = None


@runtime_checkable
class IPluginLifecycle(Protocol):
    """Protocol for lifecycle-aware plugins."""

    async def initialize(self) -> None:
        """Initialize the plugin after loading."""
        ...

    async def activate(self) -> None:
        """Activate the plugin for use."""
        ...

    async def deactivate(self) -> None:
        """Deactivate the plugin."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources before unloading."""
        ...


class PluginLoader[T]:
    """Loader for managing plugin lifecycle.

    This class handles the full plugin lifecycle: discovery, loading,
    initialization, activation, deactivation, and unloading.

    Example:
        loader = PluginLoader[ITool](
            discovery=entry_point_discovery,
            registry=tool_registry,
            hooks=lifecycle_hooks,
        )
        await loader.load_all()
        await loader.activate_all()
    """

    def __init__(
        self,
        discovery_strategy: PluginDiscoveryStrategy[T],
        registry: PluginRegistry[T],
        hooks: LifecycleHooks | None = None,
    ) -> None:
        """Initialize plugin loader.

        Args:
            discovery_strategy: Strategy for discovering plugins.
            registry: Registry for managing plugins.
            hooks: Optional lifecycle hooks.
        """
        self._discovery_strategy = discovery_strategy
        self._registry = registry
        self._hooks = hooks or LifecycleHooks()
        self._discoverer = PluginDiscoverer(discovery_strategy)

    async def discover(self) -> list[DiscoveredPlugin]:
        """Discover available plugins.

        Returns:
            List of discovered plugins.
        """
        return await self._discoverer.discover_plugins()

    async def load_plugin(
        self,
        name: str,
        instance: T | None = None,
    ) -> bool:
        """Load a discovered plugin.

        Args:
            name: Plugin name to load.
            instance: Optional pre-existing instance.

        Returns:
            True if loaded successfully, False otherwise.
        """
        # Get discovered plugin
        discovered = None
        for p in self._discoverer.discovered_plugins:
            if p.name == name:
                discovered = p
                break

        if discovered is None:
            logger.error(f"Plugin '{name}' not discovered")
            return False

        # Load instance
        if instance is None:
            try:
                from .discovery import load_plugin

                instance = cast("T", await load_plugin(discovered))
            except Exception as e:
                logger.error(f"Failed to load plugin '{name}': {e}")
                _ = self._registry.set_error(name, str(e))
                return False

        # Register in registry
        assert instance is not None, "Plugin instance should not be None at this point"
        metadata = self._registry.register(instance)

        # Call load hook
        if self._hooks.on_load:
            self._hooks.on_load(name)

        logger.info(f"Loaded plugin '{name}' (v{metadata.version})")
        return True

    async def initialize_plugin(self, name: str) -> bool:
        """Initialize a loaded plugin.

        Args:
            name: Plugin name to initialize.

        Returns:
            True if initialized successfully, False otherwise.
        """
        plugin = self._registry.get(name)
        if plugin is None:
            logger.error(f"Plugin '{name}' not registered")
            return False

        # Check if plugin supports lifecycle
        if isinstance(plugin, IPluginLifecycle):
            try:
                await plugin.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize plugin '{name}': {e}")
                _ = self._registry.set_error(name, str(e))
                return False

        # Call initialize hook
        if self._hooks.on_initialize:
            self._hooks.on_initialize(name)

        logger.info(f"Initialized plugin '{name}'")
        return True

    async def activate_plugin(self, name: str) -> bool:
        """Activate a registered plugin.

        Args:
            name: Plugin name to activate.

        Returns:
            True if activated successfully, False otherwise.
        """
        if not self._registry.activate(name):
            logger.error(f"Failed to activate plugin '{name}'")
            return False

        # Call activate hook
        if self._hooks.on_activate:
            self._hooks.on_activate(name)

        logger.info(f"Activated plugin '{name}'")
        return True

    async def deactivate_plugin(self, name: str) -> bool:
        """Deactivate an active plugin.

        Args:
            name: Plugin name to deactivate.

        Returns:
            True if deactivated successfully, False otherwise.
        """
        plugin = self._registry.get(name)
        if plugin is None:
            logger.error(f"Plugin '{name}' not registered")
            return False

        # Check if plugin supports lifecycle
        if isinstance(plugin, IPluginLifecycle):
            try:
                await plugin.deactivate()
            except Exception as e:
                logger.warning(f"Error during deactivate of '{name}': {e}")

        if not self._registry.deactivate(name):
            return False

        # Call deactivate hook
        if self._hooks.on_deactivate:
            self._hooks.on_deactivate(name)

        logger.info(f"Deactivated plugin '{name}'")
        return True

    async def unload_plugin(self, name: str) -> bool:
        """Unload a registered plugin.

        Args:
            name: Plugin name to unload.

        Returns:
            True if unloaded successfully, False otherwise.
        """
        plugin = self._registry.get(name)
        if plugin is None:
            logger.error(f"Plugin '{name}' not registered")
            return False

        # Cleanup if supported
        if isinstance(plugin, IPluginLifecycle):
            try:
                await plugin.cleanup()
            except Exception as e:
                logger.warning(f"Error during cleanup of '{name}': {e}")

        # Unload from registry
        _ = self._registry.unregister(name)

        # Call unload hook
        if self._hooks.on_unload:
            self._hooks.on_unload(name)

        logger.info(f"Unloaded plugin '{name}'")
        return True

    async def load_all(self) -> int:
        """Load all discovered plugins.

        Returns:
            Number of plugins loaded.
        """
        loaded = 0
        for plugin in self._discoverer.discovered_plugins:
            if await self.load_plugin(plugin.name):
                loaded += 1
        return loaded

    async def initialize_all(self) -> int:
        """Initialize all loaded plugins.

        Returns:
            Number of plugins initialized.
        """
        initialized = 0
        for name in self._registry.list_by_state(PluginState.LOADED):
            if await self.initialize_plugin(name):
                initialized += 1
        return initialized

    async def activate_all(self) -> int:
        """Activate all registered plugins.

        Returns:
            Number of plugins activated.
        """
        activated = 0
        for name in self._registry.list_all():
            if await self.activate_plugin(name):
                activated += 1
        return activated

    async def deactivate_all(self) -> int:
        """Deactivate all active plugins.

        Returns:
            Number of plugins deactivated.
        """
        deactivated = 0
        for name in self._registry.list_by_state(PluginState.ACTIVATED):
            if await self.deactivate_plugin(name):
                deactivated += 1
        return deactivated

    async def unload_all(self) -> int:
        """Unload all registered plugins.

        Returns:
            Number of plugins unloaded.
        """
        unloaded = 0
        for name in list(self._registry.list_all()):
            if await self.unload_plugin(name):
                unloaded += 1
        return unloaded

    async def shutdown(self) -> None:
        """Gracefully shutdown all plugins.

        This method deactivates and unloads all plugins.
        """
        _ = await self.deactivate_all()
        _ = await self.unload_all()
        logger.info("Plugin loader shutdown complete")

    @property
    def registry(self) -> PluginRegistry[T]:
        """Get the plugin registry."""
        return self._registry

    @property
    def discoverer(self) -> PluginDiscoverer[T]:
        """Get the plugin discoverer."""
        return self._discoverer
