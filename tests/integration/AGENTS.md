# Integration Tests

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW

Real USearch + Tantivy + SQLite identity. In default pytest `testpaths`. NUMBA JIT is enabled here (JIT disable is only in two unit conftests). Marked `@pytest.mark.integration`; some are `@pytest.mark.slow`.

## STRUCTURE

```
tests/integration/
├── test_memory_manager_usearch.py   # Shared create/cleanup helpers
├── test_mcp_workflows.py            # Add → search → remove via tools
├── test_concurrent_operations.py    # Parallel add/search/delete
├── test_thread_safety.py            # Lock hierarchy under threads
├── test_chaos.py                    # Randomized op sequences
├── test_replacement_recovery.py     # Crash inject + restart converge
├── test_layer_boundaries.py         # AST import-direction checks
└── test_qwen_embeddings_integration.py
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_memory_manager_usearch.py` | SoT USearch/SQLite; `get_all()` / `count()` |
| `test_replacement_recovery.py` | Journal later-write-wins after crash |
| `test_layer_boundaries.py` | Infra/core must not import application (except documented adapter) |
| `test_mcp_workflows.py` | Tools go through `MemoryManager` only |

## CONVENTIONS

- One manager per test; always `cleanup_manager` / close indexes.
- `delete_memories` returns `list[str]` of deleted contents.
- Dual-engine split after USearch success is `InconsistentStateError`, not `StorageError`.
- Identity is unique `(workspace_id, content)` in SQLite. Tantivy is not exact match.
- Embed-API tests need a live endpoint or must skip; do not mock the engines instead.

## ANTI-PATTERNS

- Never mock USearch/Tantivy here.
- Never share a `MemoryManager` across tests.
- Never compact-on-delete assertions (no compact in `delete`/`delete_batch`).
- Never skip cleanup (leaks HNSW/SQLite files).

## NOTES

Slow (seconds to a minute). `./start-unittest.sh` runs `pytest tests/`, which also picks up root demo `test_*.py` outside this folder. Coverage fail-under 90% is suite-wide.
