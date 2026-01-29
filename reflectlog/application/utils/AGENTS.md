# Agent Guidelines for reflectlog/application/utils/

Cross-cutting utilities for logging, metrics, resilience, security, and performance.

## STRUCTURE

```
utils/
├── logging.py           # StructuredLogger, auto-redaction, operation context
├── metrics.py           # MetricsRegistry (counters/gauges/histograms), Prometheus export
├── retry.py            # async_retry_with_backoff (exponential + jitter)
├── circuit_breaker.py   # CircuitBreaker (CLOSED/OPEN/HALF_OPEN states)
├── security.py         # SecretString, redact_dict_secrets, validate_project_id
├── http_client.py       # HttpClientFactory (pooled httpx/aiohttp)
├── numba_utils.py      # JIT-compiled RRF, normalization, filtering
├── validation.py       # validate_messages, truncate_message, SQL injection detection
├── config_reload.py    # ConfigReloadManager (SIGHUP-based runtime reload)
└── __init__.py         # Package exports
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
| JIT acceleration | numba_utils.py | RRF, normalization, filtering |

## CONVENTIONS

**Resilience Patterns**: Use `async_retry_with_backoff` for transient failures (ConnectionError, TimeoutError). Default `max_retries=3`, `base_delay=1.0`, jitter ±20%.

**Circuit Breaker**: Wrap external LLM API calls with `CircuitBreaker`. Default `failure_threshold=5`, `timeout=60.0`. States: CLOSED (normal), OPEN (fail-fast), HALF_OPEN (test recovery).

**Metrics**: Use `MetricsRegistry` with thread-safe operations. `increment()` for counters, `set()` for gauges, `observe()` for histograms, `timer()` context for durations. Export via `export_prometheus()`.

**Secret Handling**: Store secrets in `SecretString`. Use `redact_dict_secrets()` before logging configs. Auto-redacts API keys, passwords, tokens via regex patterns.

**Input Validation**: Use `validate_messages()` for user input. Checks length, type, SQL injection patterns, control characters. `validate_project_id()` prevents path traversal.

**Numba JIT**: Call `warmup_numba_functions()` at startup to pre-compile `_find_minmax`, `_normalize_inplace`, `compute_rrf_scores_batch`, `_filter_by_threshold`. First-call latency 50-200ms otherwise.

**HTTP Pooling**: Use `HttpClientFactory` singleton. Default `max_connections=100`, `max_keepalive=20`. HTTP/2 enabled for httpx.

**Logging**: `StructuredLogger` auto-redacts secrets via `redact_dict_secrets`. Use `operation()` context manager for multi-step operations.

## ANTI-PATTERNS

- Never log secrets directly - use `SecretString` or `redact_dict_secrets()`
- Never retry non-transient exceptions - specify `exceptions` parameter
- Never skip Numba warmup - first calls pay compilation cost
- Never create new HTTP clients per request - use factory singleton
- Never use bare `except:` in retry/circuit breaker - catch specific exceptions
- Never validate project_id without `validate_project_id()` - path traversal risk
- Never allow SQL injection patterns - `validate_messages()` checks them
- Never acquire locks out of order - `_write_lock` before `_lock` (root-level, relevant here)
- Never use circuit breaker for local operations - only external services
- Never forget `close_all()` on shutdown - releases HTTP connections
