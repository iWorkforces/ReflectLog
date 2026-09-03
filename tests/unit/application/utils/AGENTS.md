# Application Utils Unit Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
Unit tests for logging, security helpers, config reload, and Numba scoring wrappers. HTTP / circuit-breaker / metrics test modules are gone from this folder.

## STRUCTURE
```
tests/unit/application/utils/
├── conftest.py                 # NUMBA_DISABLE_JIT=1 + reload
├── test_logging.py             # StructuredLogger, redaction
├── test_security.py            # SecretString, validate_workspace_id
├── test_config_reload.py       # ConfigReloadManager / SIGHUP
└── test_numba_utils.py         # reflectlog.utility.scoring JIT helpers
```

## WHERE TO LOOK
| Test | Purpose |
|------|---------|
| `test_security.py` | `SecretString` str=`***REDACTED***`; workspace_id pattern |
| `test_logging.py` | Structured fields; no memory text |
| `test_numba_utils.py` | RRF / min-max / filter; imports `utility.scoring` |
| `test_config_reload.py` | Reload manager + signal handler |

## CONVENTIONS
- `conftest.py` purges `numba` and reloads with `NUMBA_DISABLE_JIT=1` so coverage can see bodies.
- Never log secrets or memory text.
- Recency is not applied before CE normalize/threshold.

## ANTI-PATTERNS
- Never skip SQL / path-traversal cases in `validate_workspace_id`.
- Never mock Numba away when testing scoring math.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`.

## NOTES
Production HTTP is `reflectlog.utility.http`. Do not resurrect `test_http_client.py` here. Local: this conftest disables JIT for scoring coverage.
