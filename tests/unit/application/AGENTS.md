# Application Unit Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

MCP, config, tools, memory facade. Engines mocked. Tools mock `MemoryManager` only.

## STRUCTURE

```
tests/unit/application/
├── test_mcp_server.py
├── test_mcp_server_error_handling.py
├── test_memory_manager.py             # Facade (sibling of memory/test_manager.py)
├── test_graceful_degradation.py
├── test_ranx_fusion.py
├── test_rrf_fusion_toggle.py
├── test_search_hypothesis.py
├── test_validation.py
├── test_dynamic_instructions.py
├── test_logging_utils.py
├── memory/                            # Pipeline + MemoryManager internals
├── tools/                             # MCP tools; mock MemoryManager
├── config/
└── utils/
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_mcp_server.py` | Registry; never flatten `InconsistentStateError` → `StorageError` |
| `test_memory_manager.py` | Hybrid facade (mocked engines) |
| `memory/test_manager.py` | Locks/delete/inconsistency |
| `tools/` | `MagicMock(spec=MemoryManager)` |

## DUAL MANAGER TESTS

- `test_memory_manager.py` — facade
- `memory/test_manager.py` — internals; patches `MODULE = "reflectlog.application.memory.manager"`

Do not merge or delete one without checking the other.

## CONVENTIONS

- Patch `reflectlog.application.memory.manager.USearchEngine`.
- Tools: `MagicMock(spec=MemoryManager)`; never patch engines from a tool test.
- Stub `is_ready=False` and `add_batch` `side_effect`. `delete_memories` → `list[str]`.

## ANTI-PATTERNS

- Never import real infrastructure engines here.
- Never flatten `InconsistentStateError` to `StorageError`.
- Never mark this tree `@pytest.mark.integration`.
