# Utility Unit Tests

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW

Unit tests for cross-cutting utilities: logging, metrics, circuit breaker, HTTP client, security validation.

## STRUCTURE

```
tests/unit/application/utils/
├── test_circuit_breaker.py          # Circuit state transitions (31KB)
├── test_http_client.py              # HttpClientFactory (25KB)
├── test_metrics.py                  # Prometheus metrics (23KB)
├── test_numba_utils.py              # JIT functions (17KB)
├── test_config_reload.py            # Hot reload (14KB)
├── test_logging.py                  # Structured logging (11KB)
├── test_security.py                 # SQL injection, path traversal (11KB)
└── conftest.py                      # Shared fixtures
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_circuit_breaker.py | Open/half-open/closed transitions |
| test_http_client.py | Connection pooling, retry |
| test_security.py | `validate_project_id()`, SQL patterns |

## KEY PATTERNS

### Circuit Breaker State Machine
```python
async def test_circuit_opens_after_failures():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CircuitState.OPEN
```

### Security Validation
```python
def test_sql_injection_blocked():
    malicious = "'; DROP TABLE memories; --"
    with pytest.raises(ValidationError):
        validate_messages([malicious])
```

### Metrics Registration
```python
def test_metrics_registered():
    metrics = MemoryMetrics()
    assert 'memory_add_total' in metrics._registry
```

## ANTI-PATTERNS

- Never skip SQL injection tests
- Never test circuit breaker without time progression
- Never mock numba functions (test actual JIT)

## NOTES

- **Numba tests**: Some test actual JIT compilation
- **Security focus**: SQL injection, path traversal coverage
- **Async utilities**: All tests are async
