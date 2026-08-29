# ReflectLog Knowledge Base - Utility

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW

Platform abstraction for secure credential retrieval from OS-specific stores, plus JIT-compiled scoring functions.

## STRUCTURE

```
utility/
├── __init__.py          # Package exports (OAuth helpers, token prefixes)
├── types.py             # Constants (TOKEN_PREFIX, SERVICE_NAME, ApiKeyResult)
├── utility.py           # Credential retrieval functions (get_anthropic_api_key, init_credentials)
├── scoring.py           # JIT-compiled RRF, normalization, filtering (Numba)
├── http.py              # Production HttpClientFactory (SIGINT close_all_sync)
├── retry.py             # Production async_retry_with_backoff (SmartReplacer)
└── platforms/           # OS-specific implementations
    ├── __init__.py      # get_platform_retriever() factory
    ├── base.py          # CredentialRetriever ABC with parse_credential()
    ├── darwin.py        # macOS Keychain via security CLI
    ├── linux.py         # Linux secret-tool + env fallback
    └── windows.py       # Windows Credential Manager
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|----------|--------|
| Credential factory | platforms/__init__.py | get_platform_retriever() using platform.system() |
| Credential parsing | platforms/base.py | 3 formats: OAuth JSON, legacy JSON, raw token |
| Main retrieval | utility.py | get_anthropic_api_key(), init_credentials() |
| Type definitions | types.py | TOKEN_PREFIX, SERVICE_NAME constants |
| Scoring functions | scoring.py | JIT-compiled RRF, batch normalization, recency decay, filtering |

## CONVENTIONS

- Factory pattern: get_platform_retriever() returns platform-specific subclass
- Graceful degradation: All get_credential() return None on errors (no exceptions)
- Three credential formats: OAuth JSON → legacy JSON → raw token (fallback chain)
- 10s timeout on subprocess calls to prevent hanging
- Never log/raise on credential retrieval failures

## ANTI-PATTERNS

- Never raise exceptions on credential retrieval - always return None
- Never log credential values or partial tokens
- Never bypass parse_credential() validation
- Never call get_credential() without subprocess timeout
- Never use platform-specific imports at module level (lazy import in factory)
