# Unit Tests Root

**Generated:** 2026-02-22
**Commit:** 6c2d6fa
**Branch:** develop

## OVERVIEW

Unit test root directory. Contains server entry point tests and subdirectories for application, infrastructure, plugins, and utility layers.

## STRUCTURE

```
tests/unit/
├── test_server.py          # CLI entry, signal handling, startup (26KB)
├── application/            # Business logic tests (see application/AGENTS.md)
├── infrastructure/         # Infrastructure tests (see infrastructure/AGENTS.md)
├── plugins/                # Plugin system tests (see plugins/AGENTS.md)
└── utility/                # Platform utility tests (see utility/AGENTS.md)
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_server.py | `server.py:main()` CLI, Numba warmup, FastMCP initialization |

## KEY PATTERNS

### Server Startup Mock
```python
@patch('reflectlog.server.FastMCPServer')
@patch('reflectlog.server.warmup_numba')
def test_server_startup(mock_warmup, mock_server_class):
    with pytest.raises(SystemExit):
        main(["--transport", "stdio"])
    mock_warmup.assert_called_once()
```

### Signal Handler Test
```python
def test_graceful_shutdown_signal(mock_server):
    import signal
    handler = signal.getsignal(signal.SIGINT)
    handler(signal.SIGINT, None)
    mock_server.close.assert_called_once()
```

## ANTI-PATTERNS

- Never test with real MemoryManager
- Never skip config reset between tests

## NOTES

- **CLAUDE.md present**: Additional patterns documented there
- **Fast tests**: All tests <100ms each
