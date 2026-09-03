# MCP Tools

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

MCP tool implementations. Each `BaseTool` subclass validates input, logs redacted metadata, and delegates to `MemoryManager`. No engine imports.

## STRUCTURE

```
tools/
├── base.py           # ABC + log + InconsistentStateError re-raise
├── add.py            # add(memories, dry_run=False) → counts dict
├── get_all.py        # paged get_all(limit, offset)
├── search.py         # hybrid search(query) → list[str]
├── remove.py         # remove(memories); exact match
└── health_check.py   # read-only component status
```

Registry lives in `application/mcp_server.py` as `AVAILABLE_TOOL_CLASSES`.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add | `add.py` | `validate_memories` + `validate_add_batch`; `add_memories_async` |
| Search | `search.py` | Manager runs RRF + optional CE; tool returns texts |
| Remove | `remove.py` | `delete_memories` → `list[str]`; missing ids are a no-op |
| Get all | `get_all.py` | Cap `Config.get_all_limit` (default 1000) |
| Health | `health_check.py` | Read-only; leftovers are not reconciled here |
| Errors | `base.py` | `InconsistentStateError` is re-raised, not wrapped |

## CONVENTIONS

- Tools never touch USearch/Tantivy/SQLite. Go through `MemoryManager` only.
- Handlers are typed. `get_handler()` returns `Callable[..., Awaitable[...]]`:
  - `add` → `Awaitable[dict[str, object]]` (`memories: list[str]`, `dry_run: bool`)
  - `search` → `Awaitable[list[str]]`
  - `remove` → `Awaitable[None]`
  - `get_all` → `Awaitable[dict[str, object]]`
  - `health_check` → `Awaitable[dict[str, Any]]`
- Tests mock `MemoryManager` (`MagicMock(spec=MemoryManager)`), not engines.
- `delete_memories` returns deleted contents (`list[str]`), not an int.
- `BaseTool._raise_tool_error`: if `isinstance(error, InconsistentStateError): raise error` — never flatten to `StorageError` / `SearchError`.
- Names come from `core.enums.ToolName`. Instruction snippets feed dynamic MCP instructions.

## ANTI-PATTERNS

- Never import infrastructure engines from this package.
- Never log memory text or secrets.
- Never wrap or flatten `InconsistentStateError`.
- Never assume both backends return the same hit count.
- Never run reconcile or compact from `health_check`.
