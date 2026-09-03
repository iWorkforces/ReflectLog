# Application Utils

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Small leftover helpers used by tools/config. HTTP, retry, circuit breaker, metrics, and Numba scoring are **not** in this package.

## STRUCTURE

```
utils/
├── logging.py         # StructuredLogger + auto-redaction
├── security.py        # SecretString, redact_dict_secrets, validate_workspace_id
├── validation.py      # validate_memories, validate_add_batch, validate_remove_*
├── config_reload.py   # SIGHUP helper; unused at startup
└── __init__.py
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Log without leaking | `logging.py` | `StructuredLogger`; redacts via `security` |
| Secrets | `security.py` | `SecretString`; `__str__`/`__repr__` are `***REDACTED***` |
| Add payload caps | `validation.py` | `validate_memories` + `validate_add_batch` |
| Remove payload | `validation.py` | `validate_remove_memories` + `validate_remove_batch` |
| Workspace path safety | `security.validate_workspace_id` | Alphanumeric / `_.-`; reject `..` |
| Unused SIGHUP | `config_reload.py` | `setup_signal_handler`; not registered |

## CONVENTIONS

- SQLite is parameterized. Do **not** reject memory text because it looks like SQL.
- `validate_memories` checks type, empty/whitespace, min/max length, control chars (tab/LF/CR allowed).
- `config_reload.setup_signal_handler` is only wired if `settings.setup_config_reload()` is called; `server.py` does not call it. SIGHUP stays unused.

## ANTI-PATTERNS

- Never log `SecretString.get_secret_value()`.
- Never add `http_client.py` / `retry.py` / `metrics.py` / `circuit_breaker.py` / `truncate_memory` back here.
- Never put JIT scoring in this package; call `warmup_numba_functions()` from `utility/scoring.py`.
- Never treat this folder as the HTTP or resilience layer.
- Never register SIGHUP from `server.py` / `FastMCPServer`.
