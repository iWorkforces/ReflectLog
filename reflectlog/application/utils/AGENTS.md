# Agent Guidelines for reflectlog/application/utils/

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop
Cross-cutting utilities for logging, metrics, resilience, security, and performance.

## STRUCTURE

```
utils/
├── logging.py           # StructuredLogger + sanitize_for_logging
├── security.py         # SecretString, redact_dict_secrets, validate_workspace_id
├── validation.py       # validate_memories, validate_add_batch
├── config_reload.py    # unused SIGHUP helper (not registered at startup)
└── __init__.py
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
| Structured logging | logging.py | StructuredLogger with auto-redaction |
| Metrics collection | metrics.py | Prometheus export, timers |
| Retry logic | retry.py | Exponential backoff + jitter |
| Circuit breaker | circuit_breaker.py | LLM API protection |
| Secret redaction | security.py | SecretString, redact_dict_secrets |
| HTTP pooling | http_client.py | Singleton httpx/aiohttp |

## CONVENTIONS

**Resilience Patterns**: Use `async_retry_with_backoff` for transient failures (ConnectionError, TimeoutError). Default `max_retries=3`, `base_delay=1.0`, jitter ±20%.

**Circuit Breaker**: Wrap external LLM API calls with `CircuitBreaker`. Default `failure_threshold=5`, `timeout=60.0`. States: CLOSED (normal), OPEN (fail-fast), HALF_OPEN (test recovery).

**Metrics**: Use `MetricsRegistry` with thread-safe operations. `increment()` for counters, `set()` for gauges, `observe()` for histograms, `timer()` context for durations. Export via `export_prometheus()`.

**Secret Handling**: Store secrets in `SecretString`. Use `redact_dict_secrets()` before logging configs. Auto-redacts API keys, passwords, tokens via regex patterns.

**Input Validation**: Use `validate_messages()` for user input. Checks length, type, SQL injection patterns, control characters. `validate_workspace_id()` prevents path traversal.

**Numba JIT**: Functions now live in `reflectlog/utility/scoring.py`. Call `warmup_numba_functions()` at startup. First-call latency 50-200ms otherwise.

**HTTP Pooling**: Production factory is `reflectlog.utility.http.HttpClientFactory` (`close_all_sync` on SIGINT). This package's `http_client.py` / `retry.py` / `metrics.py` / `circuit_breaker.py` have no runtime callers.

**Logging**: `StructuredLogger` auto-redacts secrets via `redact_dict_secrets`. Use `operation()` context manager for multi-step operations.

## ANTI-PATTERNS

- Never log secrets directly - use `SecretString` or `redact_dict_secrets()`
- Never retry non-transient exceptions - specify `exceptions` parameter
- Never skip Numba warmup (`reflectlog/utility/scoring.py`) - first calls pay compilation cost
- Never create new HTTP clients per request - use factory singleton
- Never use bare `except:` in retry/circuit breaker - catch specific exceptions
- Never validate workspace_id without `validate_workspace_id()` - path traversal risk
- SQLite is parameterized; do not reject memory text with SQL-like words
- Never acquire locks out of order - `_write_lock` before `_lock` (root-level, relevant here)
- Never use circuit breaker for local operations - only external services
- Never forget `close_all()` on shutdown - releases HTTP connections
