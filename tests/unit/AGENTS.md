# Unit Tests Root

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

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

## ANTI-PATTERNS

- Never test with real MemoryManager
- Never skip config reset between tests

## NOTES
- All tests <100ms, all dependencies mocked
