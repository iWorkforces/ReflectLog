# Plugin Unit Tests

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

## OVERVIEW

Unit tests for three-tier plugin system: discovery, registration, lifecycle. All external imports mocked.

## STRUCTURE

```
tests/unit/plugins/
└── test_plugins.py         # 1500+ lines, comprehensive plugin tests
```

## WHERE TO LOOK

| Test Section | Purpose |
|--------------|---------|
| EntryPointDiscovery | Python entry point discovery |
| DirectoryScanDiscovery | Module pattern scanning |
| StaticRegistration | Explicit registration |
| PluginRegistry | State management, metadata |
| PluginLoader | Lifecycle orchestration |

## KEY PATTERNS

### Entry Point Mock
```python
@patch('importlib.metadata.entry_points')
def test_entry_point_discovery(mock_entry_points):
    mock_entry_points.return_value = [MockEP(name="test_plugin", value="mod:Cls")]
    discovered = EntryPointDiscovery().discover()
    assert len(discovered) == 1
```

### Lifecycle State Machine
```python
async def test_lifecycle_transitions():
    loader = PluginLoader(registry, discoverer)
    await loader.load_all()
    await loader.initialize_all()
    await loader.activate_all()
    assert registry.get_state("plugin") == PluginState.ACTIVATED
```

## ANTI-PATTERNS

- Never import real plugins in unit tests
- Never skip cleanup in lifecycle tests

## NOTES

- **Comprehensive coverage**: All 3 discovery mechanisms tested
- **Async lifecycle**: Tests use anyio for async operations
