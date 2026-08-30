# Utility Unit Tests

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Unit tests for platform credential retrieval, HTTP pool, retry, and workspace-id equivalence. Subprocess and network mocked.

## STRUCTURE
```
tests/unit/utility/
├── test_utility.py    # get_claude_code_api_key / platform factory
├── test_http.py       # HttpClientFactory singletons
├── test_retry.py      # async_retry_with_backoff
└── test_security.py   # validate_workspace_id vs application.utils.security
```

## WHERE TO LOOK
| Test | Purpose |
|------|---------|
| `test_utility.py` | Retriever factory; parse OAuth / legacy / raw |
| `test_http.py` | Pooled httpx / aiohttp; reset singletons |
| `test_retry.py` | Transient exceptions + backoff |
| `test_security.py` | Dual `validate_workspace_id` characterization |

## CONVENTIONS
- Mock `platform.system` and `subprocess.run`. Never invoke `security` / `secret-tool` / PowerShell.
- Darwin/Linux timeout **10s**; Windows **30s**.
- Linux config-file path parses inline (does not call `parse_credential()`).
- Retrievers return `None` on error. Never raise or log tokens.

## ANTI-PATTERNS
- Never log credential values in assertions or fixtures.
- Never treat Windows timeout as 10s.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`.

## NOTES
HTTP factory is `reflectlog.utility.http` (no leftover `http_client.py`). Scoring lives in `utility/scoring.py` and is covered under application/utils numba tests.
