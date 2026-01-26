# Agent Guidelines for reflectlog/plugins/

This directory contains the plugin system for ReflectLogMCP. It enables extensibility by discovering, registering, and loading plugins at runtime without modifying core code.

## Directory Structure

```
plugins/
├── __init__.py            # Package exports and public API
├── discovery.py           # Plugin discovery mechanisms
├── registry.py            # Plugin registry for managing plugins
└── loading.py             # Plugin loading and lifecycle management
```

## Core Responsibilities

### Discovery (discovery.py)

The `discovery.py` module provides multiple strategies for discovering plugins:

1. **EntryPointDiscovery**: Discover plugins via Python entry points (package metadata)
2. **DirectoryScanDiscovery**: Scan directories for plugin modules
3. **StaticRegistration**: Explicit static registration for testing
4. **CompositeDiscovery**: Combine multiple discovery strategies
5. **PluginDiscoverer**: Main class orchestrating discovery and loading

### Registry (registry.py)

The `registry.py` module manages plugin registration and queries:

- **PluginRegistry**: Generic registry for any plugin type
- **ToolRegistry**: Specialized registry for MCP tools
- Track plugin state (discovered, loaded, activated, etc.)
- Query plugins by state, capability, or type

### Loading (loading.py)

The `loading.py` module handles full plugin lifecycle:

- **PluginLoader**: Manages loading, initialization, activation, deactivation, unloading
- **LifecycleHooks**: Callbacks for plugin lifecycle events
- **IPluginLifecycle**: Protocol for lifecycle-aware plugins

## Key Components

### Plugin Discovery Strategies

#### EntryPointDiscovery

Discover plugins via standard Python entry point mechanism:

```python
discovery = EntryPointDiscovery(
    group="reflectlog.plugins",
    plugin_type=ITool,
)

plugins = await discovery.discover()
```

#### DirectoryScanDiscovery

Scan packages for plugin modules:

```python
discovery = DirectoryScanDiscovery(
    package_names=["my_package.plugins"],
    plugin_base_class=ITool,
    module_pattern="plugin_*.py",
)

plugins = await discovery.discover()
```

#### CompositeDiscovery

Combine multiple discovery strategies:

```python
discovery = CompositeDiscovery([
    EntryPointDiscovery(group="reflectlog.plugins", plugin_type=ITool),
    DirectoryScanDiscovery(package_names=["local.plugins"], plugin_base_class=ITool),
])

plugins = await discovery.discover()  # All plugins from both sources
```

### Plugin Registry

#### Generic Registry

```python
registry = PluginRegistry[ITool]()
metadata = registry.register(my_tool)
plugins = registry.list_by_type(ITool)
```

#### Tool Registry (Specialized)

```python
registry = ToolRegistry[ITool]()
registry.register_tool(
    tool=my_tool,
    name="my_custom_tool",
    description="A custom tool for X",
    version="1.0.0",
)

has_tool = registry.has_tool("my_custom_tool")
tool = registry.get_tool("my_custom_tool")
```

### Plugin Loader

```python
discovery = EntryPointDiscovery(
    group="reflectlog.plugins",
    plugin_type=ITool,
)
registry = ToolRegistry[ITool]()

# Create loader with lifecycle hooks
hooks = LifecycleHooks(
    on_load=lambda name: print(f"Loaded {name}"),
    on_activate=lambda name: print(f"Activated {name}"),
)

loader = PluginLoader(discovery, registry, hooks)

# Discover, load, initialize, and activate
await loader.discover()
loaded = await loader.load_all()
initialized = await loader.initialize_all()
activated = await loader.activate_all()

# Graceful shutdown
await loader.shutdown()
```

## Key Patterns

### Lifecycle-Aware Plugins

Plugins can implement `IPluginLifecycle` for lifecycle hooks:

```python
@runtime_checkable
class IPluginLifecycle(Protocol):
    """Protocol for lifecycle-aware plugins."""

    async def initialize(self) -> None:
        """Initialize plugin after loading."""
        ...

    async def activate(self) -> None:
        """Activate plugin for use."""
        ...

    async def deactivate(self) -> None:
        """Deactivate plugin."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources before unloading."""
        ...

class MyTool(ITool, IPluginLifecycle):
    """Tool with lifecycle management."""

    async def initialize(self) -> None:
        """Initialize resources."""
        self._client = await create_client()

    async def activate(self) -> None:
        """Register with external service."""
        await self._client.register()

    async def deactivate(self) -> None:
        """Unregister from service."""
        await self._client.unregister()

    async def cleanup(self) -> None:
        """Release resources."""
        await self._client.close()
```

### Plugin Metadata

Track rich metadata for each plugin:

```python
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
    error_message: str | None = None
    discovered_at: datetime
    loaded_at: datetime | None = None
    activated_at: datetime | None = None
```

### Plugin Capabilities

Declare capabilities for plugin discovery:

```python
@dataclass
class PluginCapability:
    """A capability provided by a plugin."""

    name: str
    version: str = "0.0.0"

# Example: Tool capability
capability = PluginCapability(name="tool", version="1.0.0")

# Example: Search backend capability
capability = PluginCapability(name="search_backend", version="2.0.0")
```

## Plugin States

Plugins transition through lifecycle states:

```
DISCOVERED → LOADED → ACTIVATED
                ↓           ↓
             DEACTIVATED ←──┘
                ↓
            UNLOADED
                ↓
             ERROR
```

**State meanings:**
- **DISCOVERED**: Found but not yet loaded
- **LOADED**: Loaded into memory, not yet active
- **ACTIVATED**: Ready for use
- **DEACTIVATED**: Loaded but inactive
- **UNLOADED**: Removed from memory
- **ERROR**: Failed during lifecycle

## Testing Guidelines

### Unit Tests

Test plugin discovery, registry, and loading in isolation:

```python
@pytest.fixture
def static_discovery():
    return StaticRegistration([
        DiscoveredPlugin(
            name="test_plugin",
            module_path="test_module",
            class_name="TestClass",
        )
    ])

@pytest.fixture
def plugin_registry():
    return PluginRegistry[ITool]()

def test_discovery(static_discovery):
    plugins = await static_discovery.discover()
    assert len(plugins) == 1
    assert plugins[0].name == "test_plugin"

def test_registration(plugin_registry):
    tool = MockTool()
    metadata = plugin_registry.register(tool)
    assert metadata.name == "mock_tool"
    assert metadata.state == PluginState.LOADED
```

### Integration Tests

Test full plugin lifecycle:

```python
@pytest.mark.asyncio
async def test_plugin_lifecycle():
    discovery = EntryPointDiscovery(group="test.plugins", plugin_type=ITool)
    registry = ToolRegistry[ITool]()
    loader = PluginLoader(discovery, registry)

    await loader.discover()
    assert len(loader.discoverer.discovered_plugins) > 0

    loaded = await loader.load_all()
    assert loaded > 0

    initialized = await loader.initialize_all()
    assert initialized == loaded

    activated = await loader.activate_all()
    assert activated == loaded

    await loader.shutdown()
    assert len(registry.list_all()) == 0
```

## Dependencies

### Internal Dependencies

- `core/tools.py`: `ITool`, `IToolRegistry` protocols
- `application/exceptions.py`: Plugin-related exceptions
- `application/utils/logging.py`: Structured logging

### External Dependencies

- `importlib`: Module importing and metadata
- `importlib.metadata`: Entry point discovery
- `pkgutil`: Package scanning
- `dataclasses`: Metadata structures

## Important Notes

### Entry Point Configuration

Plugins distributed as packages should register via entry points in `pyproject.toml`:

```toml
[project.entry-points."reflectlog.plugins"]
my_tool = "my_package.plugins:MyTool"
```

### Error Handling

Plugin failures should not crash the server:

```python
try:
    await loader.load_all()
except Exception as e:
    logger.error(f"Plugin loading failed: {e}")
    # Continue loading other plugins
```

### Thread Safety

- Registry is not thread-safe by default
- Use external locking if accessing from multiple threads
- Plugin loader methods are async-safe

### Resource Management

- Always implement `cleanup()` for lifecycle-aware plugins
- Call `shutdown()` to gracefully unload all plugins
- Registry tracks timestamps for debugging

## Future Expansion

### Plugin Discovery Sources

Potential future discovery strategies:
- **HTTPDiscovery**: Discover from HTTP endpoints
- **GitDiscovery**: Discover from Git repositories
- **DockerDiscovery**: Discover from Docker containers

### Plugin Types

Current support:
- MCP tools (`ITool`)

Potential future types:
- Search backends (`ISearchBackend`)
- Rerankers (`IReranker`)
- Embedders (`IEmbedder`)

### Dependency Resolution

Implement plugin dependency resolution:
- Check dependencies before activation
- Load dependencies first
- Validate version constraints
