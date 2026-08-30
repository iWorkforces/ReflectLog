# Test Suite

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW

Pytest suite for ReflectLog. Default collection is `tests/unit` + `tests/integration` + `tests/security`. Coverage **fails under 90%** (`[tool.coverage.report] fail_under` and `COVERAGE_MIN` in `start-unittest.sh`). `filterwarnings = ["error", ...]`.

## STRUCTURE

```
tests/
├── conftest.py          # Shared fixtures; reset_env_after_test autouse
├── unit/                # Mirrors reflectlog/; engines mocked
├── integration/         # Real USearch + Tantivy
├── security/            # In default testpaths
├── load/                # Locust; not in pytest testpaths
└── test_*.py            # Root demos; not in default testpaths
```

## COLLECTION

| Invoker | Collects |
|---------|----------|
| Bare `pytest` / `testpaths` | `unit/` + `integration/` + `security/` |
| `./start-unittest.sh` (`pytest tests/`) | Also root `test_*.py` demos |

`tests/load/` is Locust, not pytest.

## FIXTURES

1. `tests/conftest.py` — `mock_usearch_engine` (`is_ready=False`, `add_batch` `side_effect`), `mock_memory_class` (patches `reflectlog.application.memory.manager.USearchEngine`), `set_env_vars`, `reset_env_after_test`.
2. `tests/unit/application/memory/conftest.py` — `NUMBA_DISABLE_JIT=1` + purge/reload numba.
3. `tests/unit/application/utils/conftest.py` — same JIT disable.

JIT disable lives in those two unit conftests only. Collection-order flake if a JIT-compiled module is imported before they run.

## CONVENTIONS

- Async: `asyncio_mode=auto`; no extra `@pytest.mark.asyncio`.
- MagicMock: stub real methods (`is_ready=False`, `add_batch` `side_effect`). Put APIs on protocols. Do **not** use `type(obj).__dict__.get(...)` (banned).
- Patch constructors on the manager module: `reflectlog.application.memory.manager.USearchEngine`.
- Tools mock `MemoryManager`, not engines.
- Dual manager tests: `unit/application/test_memory_manager.py` and `unit/application/memory/test_manager.py` both exist.

## ANTI-PATTERNS

- Never treat `fail_under` as a warning. 90% is a hard fail.
- Never invent MagicMock auto-attrs as production APIs.
- Never log memory text or secrets in assertions/helpers.

## RUNNING

```bash
./start-unittest.sh --coverage   # fail_under=90 via coverage.py
./start-unittest.sh --parallel
pytest                           # testpaths only
pytest tests/                    # also collects root demos
```
