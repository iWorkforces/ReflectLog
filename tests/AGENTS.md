# Test Suite

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Pytest suite. Default `testpaths`: `tests/unit`, `tests/integration`, `tests/security`. Coverage `fail_under = 90` (`[tool.coverage.report]` + `COVERAGE_MIN` in `start-unittest.sh`). `filterwarnings = ["error", ...]`.

## STRUCTURE

```
tests/
├── conftest.py          # Shared fixtures; reset_env_after_test autouse
├── unit/                # Mirrors reflectlog/; app mocked, infra real tmp
├── integration/         # Real USearch + Tantivy
├── security/            # In default testpaths
├── load/                # Locust; not pytest
└── test_*.py            # Root demos; not in default testpaths
```

## COLLECTION

| Invoker | Collects |
|---------|----------|
| Bare `pytest` / default `./start-unittest.sh` | `testpaths` only |
| `--parallel` / `--pattern` (`pytest tests/`) | Also root `test_*.py` demos |

`tests/load/` is Locust, not pytest.

## FIXTURES

1. `tests/conftest.py` — `mock_usearch_engine` (`is_ready=False`, `add_batch` `side_effect`), `mock_memory_class` (patches `reflectlog.application.memory.manager.USearchEngine`), `set_env_vars`, `reset_env_after_test`.
2. `tests/unit/application/memory/conftest.py` — `NUMBA_DISABLE_JIT=1` + purge/reload numba.
3. `tests/unit/application/utils/conftest.py` — same JIT disable.

## CONVENTIONS

- Async: `asyncio_mode=auto`.
- MagicMock: `spec=` real types; stub `is_ready=False`, `add_batch` `side_effect`. No auto-attrs as APIs.
- Patch constructors on the manager import site.
- Tools mock `MemoryManager`, not engines.
- Dual manager tests: `unit/application/test_memory_manager.py` and `unit/application/memory/test_manager.py`.

## ANTI-PATTERNS

- Never treat `fail_under` 90 as a warning.
- Never invent MagicMock auto-attrs.
- Never log memory text or secrets.

## CI

Focused workflow `.github/workflows/platform-storage.yml` exists. It runs `scripts/run_platform_gates.py --focused` plus a **subset** of coordinator/engine tests. It does **not** run the full suite, lint, typecheck, or coverage.

## RUNNING

```bash
./start-unittest.sh --coverage   # fail_under=90
./start-unittest.sh --parallel
pytest                           # testpaths only
```
