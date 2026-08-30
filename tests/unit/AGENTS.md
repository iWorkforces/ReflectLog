# Unit Tests

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW

Mocked unit tree. Layout mirrors `reflectlog/`. No real USearch/Tantivy/LLM. Default pytest `testpaths` includes this directory.

## STRUCTURE

```
tests/unit/
├── test_server.py       # CLI, signals, Numba warmup, FastMCP boot
├── application/         # MCP, manager, tools, config, utils
├── core/                # Adapters, prompts
├── infrastructure/      # Engine + embedder + store unit tests
├── plugins/             # Discovery/loading (plugins not wired at startup)
└── utility/             # HTTP factory, retry, scoring helpers
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_server.py` | `reflectlog.server:main`, SIGINT/SIGTERM persist, warmup |
| `application/` | Facade + tools + config; see child AGENTS.md |
| `infrastructure/` | USearch/Tantivy/SQLite/CE with mocks |
| `utility/` | `HttpClientFactory`, retry; scoring lives in `reflectlog/utility/scoring.py` |

## CONVENTIONS

- Patch constructors on the **import site**: `reflectlog.application.memory.manager.USearchEngine` (not the infrastructure module).
- Stub engine methods: `is_ready=False`, `add_batch` `side_effect`. Do not rely on MagicMock auto-attrs.
- `NUMBA_DISABLE_JIT` is set only in `application/memory/conftest.py` and `application/utils/conftest.py`. Importing JIT modules first flakes collection.
- Dual manager files: `application/test_memory_manager.py` vs `application/memory/test_manager.py`.

## ANTI-PATTERNS

- Never construct a real `MemoryManager` here.
- Never skip `reset_env_after_test` / env isolation.
- Never mark these `@pytest.mark.integration`.
- Never use `type(obj).__dict__.get(...)` (banned).

## NOTES

- `asyncio_mode=auto`. Coverage fail-under is 90% at the suite level.
- `start-unittest.sh` runs `pytest tests/`, which also collects root demo `test_*.py` outside this tree.
