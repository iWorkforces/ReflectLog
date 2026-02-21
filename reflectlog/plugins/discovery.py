'''Plugin discovery mechanisms for ReflectLogMCP.

This module provides mechanisms for discovering plugins from various sources:
1. Entry point discovery (via package metadata)
2. Directory scanning (package-based distribution)
3. Static registration (explicit configuration)
'''

from dataclasses import dataclass
import importlib
import importlib.metadata
from pathlib import Path
import pkgutil
from typing import TypeVar, cast

T = TypeVar('T')


@dataclass
class DiscoveredPlugin:
    '''A discovered plugin with its load function and metadata.'''

    name: str
    module_path: str
    class_name: str
    version: str = '0.0.0'
    entry_point: str | None = None


class PluginDiscoveryStrategy[T]:
    '''Base class for plugin discovery strategies.'''

    async def discover(self) -> list[DiscoveredPlugin]:
        '''Discover plugins using this strategy.

        Returns:
            List of discovered plugins.
        '''
        raise NotImplementedError


class EntryPointDiscovery(PluginDiscoveryStrategy[T]):
    '''Discover plugins via Python entry points.

    This strategy uses the standard Python entry point mechanism to
    discover plugins distributed through package indexes.
    '''

    def __init__(
        self,
        group: str,
        plugin_type: type[T],
    ):
        '''Initialize entry point discovery.

        Args:
            group: Entry point group name (e.g., 'reflectlog.plugins').
            plugin_type: Base class or protocol for type checking.
        '''
        self._group = group
        self._plugin_type = plugin_type

    async def discover(self) -> list[DiscoveredPlugin]:
        '''Discover plugins from entry points.

        Returns:
            List of discovered plugins.
        '''
        discovered: list[DiscoveredPlugin] = []

        try:
            eps = importlib.metadata.entry_points(group=self._group)
        except TypeError:
            # Python 3.9 compatibility - entry_points() returns dict-like
            # SelectableGroups
            eps = importlib.metadata.entry_points()
            # Filter to our group if SelectableGroups
            if hasattr(eps, 'select'):
                eps = eps.select(group=self._group)
            else:
                # Python 3.9 without select, iterate all and filter
                eps = [ep for ep in eps if ep.group == self._group]

        for ep in eps:
            # Parse module path and class name from entry point value
            # Format: "module.path:ClassName"
            value = ep.value
            if ':' in value:
                module_path, class_name = value.rsplit(':', 1)
            else:
                module_path = value
                class_name = ''

            discovered.append(
                DiscoveredPlugin(
                    name=ep.name,
                    module_path=module_path,
                    class_name=class_name,
                    entry_point=str(ep),
                )
            )

        return discovered


class DirectoryScanDiscovery(PluginDiscoveryStrategy[T]):
    '''Discover plugins by scanning directories.

    This strategy scans configured Python packages for modules
    containing plugin implementations.
    '''

    def __init__(
        self,
        package_names: list[str],
        plugin_base_class: type[T],
        module_pattern: str = 'plugin_*.py',
    ):
        '''Initialize directory scan discovery.

        Args:
            package_names: List of package names to scan.
            plugin_base_class: Base class for type checking.
            module_pattern: Glob pattern for plugin modules.
        '''
        self._package_names = package_names
        self._plugin_base_class = plugin_base_class
        self._module_pattern = module_pattern

    async def discover(self) -> list[DiscoveredPlugin]:
        '''Discover plugins by scanning packages.

        Returns:
            List of discovered plugins.
        '''
        discovered: list[DiscoveredPlugin] = []

        for package_name in self._package_names:
            try:
                pkg = importlib.import_module(package_name)
                pkg_file = pkg.__file__
                if pkg_file is None:
                    # Namespace package - skip
                    continue
                pkg_path = Path(pkg_file).parent

                # Find all modules matching the pattern
                for _finder, name, ispkg in pkgutil.iter_modules(
                    [str(pkg_path)],
                    prefix=f'{package_name}.',
                ):
                    if ispkg:
                        continue  # Skip packages for now

                    # Try to find plugin classes in the module
                    module = importlib.import_module(name)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name, None)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, self._plugin_base_class)
                            and attr is not self._plugin_base_class
                        ):
                            discovered.append(
                                DiscoveredPlugin(
                                    name=f'{package_name}_{attr_name}',
                                    module_path=name,
                                    class_name=attr_name,
                                )
                            )

            except ImportError:
                # Package not installed, skip
                continue

        return discovered


class StaticRegistration(PluginDiscoveryStrategy[T]):
    '''Static plugin registration via explicit configuration.

    This strategy allows plugins to be registered explicitly without
    automatic discovery, useful for development and testing.
    '''

    def __init__(
        self,
        registered_plugins: list[DiscoveredPlugin],
    ):
        '''Initialize static registration.

        Args:
            registered_plugins: List of pre-registered plugins.
        '''
        self._registered = registered_plugins

    async def discover(self) -> list[DiscoveredPlugin]:
        '''Return statically registered plugins.

        Returns:
            List of registered plugins.
        '''
        return self._registered.copy()


class CompositeDiscovery(PluginDiscoveryStrategy[T]):
    '''Combine multiple discovery strategies.

    This class combines multiple discovery strategies and returns
    all plugins discovered by any strategy.
    '''

    def __init__(
        self,
        strategies: list[PluginDiscoveryStrategy[T]],
    ):
        '''Initialize composite discovery.

        Args:
            strategies: List of discovery strategies to combine.
        '''
        self._strategies = strategies

    async def discover(self) -> list[DiscoveredPlugin]:
        '''Discover plugins using all strategies.

        Returns:
            Combined list of discovered plugins.
        '''
        all_discovered: dict[str, DiscoveredPlugin] = {}

        for strategy in self._strategies:
            for plugin in await strategy.discover():
                # Deduplicate by name, first discovery wins
                if plugin.name not in all_discovered:
                    all_discovered[plugin.name] = plugin

        return list(all_discovered.values())


async def load_plugin(plugin: DiscoveredPlugin) -> T:
    '''Load and instantiate a discovered plugin.

    Args:
        plugin: Discovered plugin information.

    Returns:
        Instantiated plugin.
    '''
    module = importlib.import_module(plugin.module_path)

    if plugin.class_name:
        plugin_class = getattr(module, plugin.class_name)
        return plugin_class()
    else:
        # Return the module itself if no class specified
        # Type cast needed because module can be returned when no class specified
        return cast(T, module)


class PluginDiscoverer[T]:
    '''Main class for plugin discovery and loading.

    This class orchestrates plugin discovery using configured strategies
    and provides methods for loading and managing plugins.

    Type Parameters:
        T: Plugin type that this discoverer manages.
    '''

    def __init__(
        self,
        discovery_strategy: PluginDiscoveryStrategy[T],
    ):
        '''Initialize plugin discoverer.

        Args:
            discovery_strategy: Strategy for discovering plugins.
        '''
        self._strategy = discovery_strategy
        self._discovered: list[DiscoveredPlugin] = []
        self._loaded: dict[str, T] = {}

    async def discover_plugins(self) -> list[DiscoveredPlugin]:
        '''Discover available plugins.

        Returns:
            List of discovered plugins.
        '''
        self._discovered = await self._strategy.discover()
        return self._discovered

    async def load_plugin(self, name: str) -> T | None:
        '''Load a plugin by name.

        Args:
            name: Plugin name to load.

        Returns:
            Loaded plugin instance or None if not found.
        '''
        # Check if already loaded
        if name in self._loaded:
            return self._loaded[name]

        # Find plugin
        plugin = None
        for p in self._discovered:
            if p.name == name:
                plugin = p
                break

        if plugin is None:
            return None

        # Load plugin
        instance = await load_plugin(plugin)
        self._loaded[name] = instance
        return instance

    async def load_all_plugins(self) -> list[T]:
        '''Load all discovered plugins.

        Returns:
            List of loaded plugin instances.
        '''
        loaded = []
        for plugin in self._discovered:
            instance = await self.load_plugin(plugin.name)
            if instance is not None:
                loaded.append(instance)
        return loaded

    @property
    def discovered_plugins(self) -> list[DiscoveredPlugin]:
        '''Get list of discovered plugins.'''
        return self._discovered.copy()

    @property
    def loaded_plugins(self) -> dict[str, T]:
        '''Get dict of loaded plugins by name.'''
        return self._loaded.copy()
