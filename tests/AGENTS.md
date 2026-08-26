# ReflectLog Test Suite

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW

Comprehensive test suite covering unit and integration tests for the ReflectLog memory system. Coverage is reported, but the current runner has no enforced fail-under gate.

## STRUCTURE

tests/ - 88 files, 34k lines
├── conftest.py - Shared fixtures (15+)
├── unit/ - Unit tests (mirrors reflectlog/)
├── integration/ - Real engine tests
├── load/ - Locust load tests
├── security/ - Security tests
└── test_*.py - Standalone test scripts

## TEST CONFIGURATION

**Pytest:** `asyncio_mode = "auto"`, `asyncio_default_fixture_loop_scope = "function"`
**Coverage:** 2 decimal precision, HTML + term-missing reports; the wrapper warns below 75% but does not fail
**Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

## FIXTURE LOCATIONS

1. `tests/conftest.py` - Shared fixtures (memory_manager, mock engines, reset_env_after_test)
2. `tests/unit/application/memory/conftest.py` - NUMBA_DISABLE_JIT
3. `tests/unit/application/utils/conftest.py` - NUMBA_DISABLE_JIT

## CONVENTIONS

- Async tests: mode=auto (no manual `@pytest.mark.asyncio` needed)
- Mock strategy: `MagicMock(spec=...)` for spec-based mocking
- Config reset: `reset_env_after_test` autouse fixture
- NUMBA JIT disabled in memory/utils unit tests for coverage

## ANTI-PATTERNS

- Never use `@type: ignore`, `as any`, bare `except:`
- Never use legacy typing (`List`, `Optional`, `Union`) - use native syntax
- Never suppress type errors - build fails on violations

## RUNNING TESTS

```bash
./start-unittest.sh --coverage   # HTML + terminal coverage report
./start-unittest.sh --parallel   # Parallel execution
pytest tests/unit/ -v           # Unit only
pytest tests/integration/ -v    # Integration only
```
