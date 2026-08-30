# Application Utils

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW

Small leftover helpers used by tools/config. HTTP, retry, circuit breaker, metrics, and Numba scoring are **not** in this package.

Removed from this directory (do not reintroduce): `http_client.py`, `circuit_breaker.py`, `metrics.py`, `retry.py`.

## STRUCTURE

```
utils/
├── logging.py         # StructuredLogger + auto-redaction
├── security.py        # SecretString, redact_dict_secrets, validate_workspace_id
├── validation.py      # validate_memories, validate_add_batch, truncate_memory
├── config_reload.py   # SIGHUP helper; not registered at startup
└── __init__.py
```

Moved homes:

| Concern | Location |
|---------|----------|
| HTTP pool | `reflectlog/utility/http.py` (`HttpClientFactory`) |
| Retry | `reflectlog/utility/` |
| Numba RRF / min-max / recency | `reflectlog/utility/scoring.py` |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Log without leaking | `logging.py` | `StructuredLogger`; redacts via `security` |
| Secrets | `security.py` | `SecretString`; `__str__`/`__repr__` are `***REDACTED***` |
| Add payload caps | `validation.py` | Count + total chars before embed |
| Workspace path safety | `security.validate_workspace_id` | Alphanumeric / `_.-`; reject `..` |

## CONVENTIONS

- SQLite is parameterized. Do **not** reject memory text because it looks like SQL.
- `validate_memories` checks type, empty/whitespace, min/max length, control chars (tab/LF/CR allowed).
- `config_reload.setup_signal_handler` is only wired if `settings.setup_config_reload()` is called; `server.py` does not call it.

## ANTI-PATTERNS

- Never log `SecretString.get_secret_value()`.
- Never add `http_client.py` / `retry.py` / `metrics.py` / `circuit_breaker.py` back here.
- Never put JIT scoring in this package; call `warmup_numba_functions()` from `utility/scoring.py`.
- Never treat this folder as the HTTP or resilience layer.
