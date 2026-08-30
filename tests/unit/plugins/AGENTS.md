# Plugin Unit Tests

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Unit tests for discovery, registry, and loader lifecycle. Production plugins are **not** wired at startup; tests still cover the package. External imports mocked.

## STRUCTURE
```
tests/unit/plugins/
├── test_discovery.py   # Entry points, directory scan, static, composite
├── test_loading.py     # PluginLoader state machine
└── test_plugins.py     # Registry, metadata, package surface
```

## WHERE TO LOOK
| File | Purpose |
|------|---------|
| `test_discovery.py` | `EntryPointDiscovery`, `DirectoryScanDiscovery`, `StaticRegistration`, `load_plugin` |
| `test_loading.py` | load → initialize → activate → deactivate → unload |
| `test_plugins.py` | `PluginRegistry`, `PluginState`, `ToolRegistry` |

## CONVENTIONS
- Patch `importlib.metadata.entry_points`; never import real plugin packages.
- Lifecycle tests must clean up to `UNLOADED` / registry empty.
- `MagicMock(spec=...)` only. Put missing methods on a protocol, not on the mock.

## ANTI-PATTERNS
- Never import real plugins in unit tests.
- Never skip lifecycle cleanup.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`.

## NOTES
Plugins exist under `reflectlog/plugins/` but are not started by `server.py`. Keep tests aligned with that unused-at-runtime package.
