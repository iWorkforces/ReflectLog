# MCP Tool Unit Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
Unit tests for MCP tools. Tools go through `MemoryManager` only — engines stay mocked.

## STRUCTURE
```
tests/unit/application/tools/
├── conftest.py              # mock_config, mock_memory_manager, tool fixtures
├── test_add.py
├── test_search.py
├── test_remove.py
├── test_health_check.py
└── test_base.py
```

## WHERE TO LOOK
| Test | Purpose |
|------|---------|
| `test_add.py` | `add_memories_async` counts / dry_run |
| `test_search.py` | Query handling, `SearchError` |
| `test_remove.py` | `delete_memories` → `list[str]`; not-found = set difference |
| `test_health_check.py` | `search_engine_status` / `EngineReadiness` |
| `test_base.py` | `BaseTool` name / handler / validation |

## CONVENTIONS
- `MagicMock(spec=MemoryManager)` in conftest. Put methods on the protocol.
- No real engines. No real `MemoryManager` construction.
- Tools must not import infrastructure engines.

## ANTI-PATTERNS
- Never call a real `MemoryManager` or engine.
- Never skip input validation tests.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`.

## NOTES
Production also has `get_all.py`; this folder does not yet have `test_get_all.py`. Registration is `application/mcp_server.py`. Local: MagicMock(spec=MemoryManager) only.
