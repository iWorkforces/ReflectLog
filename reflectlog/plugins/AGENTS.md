# Agent Guidelines for reflectlog/plugins/

## OVERVIEW
Three-tier plugin system for runtime extensibility without core code modification.

## STRUCTURE
```
plugins/
├── discovery.py     # Entry points, directory scan, static registration
├── registry.py      # Plugin registration, state tracking, querying
└── loading.py       # Lifecycle management (load/activate/deactivate/unload)
```

## WHERE TO LOOK
| Task | Location | Notes |
|-------|----------|--------|
| Discovery strategies | discovery.py | EntryPoint, DirectoryScan, Static, Composite |
| Plugin registration | registry.py | PluginRegistry[T], ToolRegistry[T] |
| Lifecycle management | loading.py | PluginLoader, LifecycleHooks, IPluginLifecycle |
| State transitions | registry.py | DISCOVERED → LOADED → ACTIVATED |
| Metadata tracking | registry.py | PluginMetadata with timestamps |

## CONVENTIONS

**Three Discovery Mechanisms**:
- EntryPointDiscovery: Standard Python entry points (`reflectlog.plugins` group)
- DirectoryScanDiscovery: Scan packages by module pattern
- StaticRegistration: Explicit registration for testing

**Lifecycle Protocol**: Implement `IPluginLifecycle` for plugins needing resource management:
```python
async def initialize()  # Post-load setup
async def activate()    # Register for use
async def deactivate()  # Unregister
async def cleanup()     # Release resources
```

**State Machine**: DISCOVERED → LOADED → ACTIVATED → DEACTIVATED → UNLOADED. Errors propagate to ERROR state.

**Thread Safety**: Registry not thread-safe by default. Use external locking for concurrent access. Loader methods are async-safe.

**Entry Point Config** (in pyproject.toml):
```toml
[project.entry-points."reflectlog.plugins"]
my_tool = "my_package.plugins:MyTool"
```

**Graceful Shutdown**: Always call `await loader.shutdown()` to clean up all plugins and release resources.
