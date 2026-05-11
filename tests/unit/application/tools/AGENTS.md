# MCP Tool Unit Tests

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

## OVERVIEW

Unit tests for MCP tool implementations: add, search, remove, and base tool patterns. All memory operations mocked.

## STRUCTURE

```
tests/unit/application/tools/
├── conftest.py             # Shared tool test fixtures
├── test_add.py             # Add tool tests
├── test_base.py            # Base tool patterns
├── test_remove.py          # Remove tool tests
└── test_search.py          # Search tool tests
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_add.py | Memory add tool validation, batching |
| test_search.py | Search tool query handling, result formatting |
| test_remove.py | Memory removal, exact match verification |
| test_base.py | Shared tool behavior, error handling |

## ANTI-PATTERNS

- Never call real MemoryManager in tool tests
- Never skip input validation tests

## NOTES

- **All mocked**: No real infrastructure calls
- **Fast**: Each test <100ms
