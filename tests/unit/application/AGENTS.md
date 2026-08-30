# Application Unit Tests

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW

Unit tests for MCP, config, tools, and the memory facade. All engines mocked. Tools mock `MemoryManager` only.

## STRUCTURE

```
tests/unit/application/
├── test_mcp_server.py                 # FastMCPServer registry + transports
├── test_mcp_server_error_handling.py  # Error translation
├── test_memory_manager.py             # Facade (sibling of memory/test_manager.py)
├── test_graceful_degradation.py       # Engine-down paths
├── test_ranx_fusion.py                # RRF / CombSUM / MNZ
├── test_rrf_fusion_toggle.py          # enable_rrf_fusion
├── test_search_hypothesis.py          # Hypothesis search properties
├── test_validation.py                 # Input validation
├── test_dynamic_instructions.py       # MCP_INSTRUCTIONS from tools
├── test_logging_utils.py              # Application logging helpers
├── memory/                            # Pipeline + MemoryManager internals
├── tools/                             # MCP tools; mock MemoryManager
├── config/                            # Frozen Config + presets
└── utils/                             # logging, security, validation, reload
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_mcp_server.py` | Tool registration; `InconsistentStateError` must not flatten to `StorageError` |
| `test_memory_manager.py` | Hybrid facade orchestration (mocked engines) |
| `memory/test_manager.py` | Lock/delete/inconsistency coverage of `manager.py` |
| `tools/` | Add/search/remove/health; fixtures stub `MemoryManager` |

## DUAL MANAGER TESTS

Both exist and both run:

- `test_memory_manager.py` — hybrid facade, older Mock(spec=Config) style
- `memory/test_manager.py` — line-targeted coverage; patches `MODULE = "reflectlog.application.memory.manager"`

Do not merge or delete one without checking the other.

## CONVENTIONS

- Patch `reflectlog.application.memory.manager.USearchEngine` (and Tantivy/embedder there).
- Tools: `MagicMock(spec=MemoryManager)`; never patch engines from a tool test.
- Stub `is_ready=False` and `add_batch` `side_effect`. `delete_memories` returns `list[str]`.

## ANTI-PATTERNS

- Never import real infrastructure engines in these tests.
- Never flatten `InconsistentStateError` to `StorageError` in expected outcomes.
- Never use `type(obj).__dict__.get(...)`.
- Never mark this tree `@pytest.mark.integration`.
