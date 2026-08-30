# Memory Pipeline Unit Tests

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW

Unit tests for 3-phase add, 4-step search, factory, and journal replay. Engines mocked. This directory’s `conftest.py` sets `NUMBA_DISABLE_JIT=1` and reloads numba.

## STRUCTURE

```
tests/unit/application/memory/
├── conftest.py                  # NUMBA_DISABLE_JIT=1; purge + reload numba
├── test_manager.py              # MemoryManager internals (locks, delete)
├── test_add_phases.py           # Dedup / smart replace / persist
├── test_search_strategies.py    # Fusion, threshold, CE skip
├── test_search_pipeline.py      # SearchPipeline + overfetch
├── test_engine_factory.py       # from_config() engine wiring
├── test_replacement_recovery.py # Journal replay converge
└── reranking/test_normalization.py
```

Sibling (not here): `../test_memory_manager.py` — dual facade file; both collect.

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_manager.py` | `_write_lock` then `_lock`; `delete_memories` → `list[str]`; Tantivy fail → `InconsistentStateError` |
| `test_add_phases.py` | Embed outside write lock; persist NEW then OLD |
| `test_search_pipeline.py` | Canonical `SearchPipeline` / `SearchContext` |
| `test_replacement_recovery.py` | Restart journal kinds `add\|delete\|replace` |

## CONVENTIONS

Patch constructors on the manager module:

```python
MODULE = "reflectlog.application.memory.manager"
patch(f"{MODULE}.USearchEngine")
patch(f"{MODULE}.TantivyEngine")
```

Stub `is_ready.return_value = False` and `add_batch.side_effect`. Do not treat MagicMock auto-attrs as real APIs.

`conftest.py` must run before ranx/numba import. Collection-order flake if a JIT module is imported first (e.g. another test file imported scoring before this conftest). The other JIT disable is `../utils/conftest.py` only.

## ANTI-PATTERNS

- Never enable JIT in this tree (breaks coverage).
- Never wrap `InconsistentStateError` as `StorageError`.
- Never use `type(obj).__dict__.get(...)` (banned).
- Never construct real USearch/Tantivy here.

## NOTES

`delete_memories` returns deleted contents (`list[str]`), not a count. CE is skipped when ≤1 hit.
