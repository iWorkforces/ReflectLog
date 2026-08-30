# reflectlog/plugins/

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Discovery/registry/loader exist. **Not imported by `server.py` or `FastMCPServer`.** Startup tools come from `AVAILABLE_TOOL_CLASSES` only.

## STRUCTURE

```
plugins/
├── discovery.py   # EntryPointDiscovery, DirectoryScanDiscovery, StaticRegistration, CompositeDiscovery
├── registry.py    # PluginRegistry[T], ToolRegistry[T], PluginState, PluginMetadata, IPluggable
└── loading.py     # PluginLoader, LifecycleHooks, IPluginLifecycle
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Discover | discovery.py | `PluginDiscoveryStrategy.discover()` → `DiscoveredPlugin` |
| Register | registry.py | `PluginState`: DISCOVERED → LOADED → ACTIVATED → DEACTIVATED → UNLOADED; ERROR |
| Lifecycle | loading.py | `IPluginLifecycle`: `initialize` / `activate` / `deactivate` / `cleanup` |
| Shutdown | loading.py | `PluginLoader.shutdown()` deactivates then unloads |

## CONVENTIONS

- `IPluggable`: `plugin_name`, `plugin_version`.
- `EntryPointDiscovery(group, plugin_type)` — docstring group `reflectlog.plugins`. **No `pyproject.toml` entry-points registered.**
- Callers (tests only): `PluginLoader.load_all` / `activate_all` / `shutdown`.
- Unit tests under `tests/unit/plugins/`.

## ANTI-PATTERNS

- Do not wire this package into `server.py` / `mcp_server.py` without an explicit product change.
- Do not invent a live plugin API at startup.
- Do not assume entry points exist in the installed package.
