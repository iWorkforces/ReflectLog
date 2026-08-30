# ReflectLog Utility

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
HTTP pool, Numba scoring, retry, OS credentials. **Not** `application/utils/` (logging / `SecretString` / `validate_memories`).

## STRUCTURE

```
utility/
├── http.py          # HttpClientFactory (httpx + aiohttp pool)
├── scoring.py       # Numba RRF, min-max, recency, warmup_numba_functions
├── retry.py         # async_retry_with_backoff (SmartReplacer)
├── security.py      # validate_workspace_id
├── utility.py       # get_anthropic_api_key, init_credentials
├── types.py         # TOKEN_PREFIX, SERVICE_NAME, ApiKeyResult
└── platforms/       # OS credential retrievers (see child AGENTS)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| HTTP pool | http.py | `HttpClientFactory`; `close_all_sync` on SIGINT |
| RRF / CE math | scoring.py | `compute_rrf_scores_batch`, `normalize_reranker_scores`, `apply_recency_decay` |
| Retry | retry.py | production retry; no `getattr` for callable names |
| Credentials | utility.py + platforms/ | `get_anthropic_api_key()` → `ApiKeyResult` |

## CONVENTIONS

- This layer: HTTP / scoring / retry / credentials.
- App layer `application/utils/`: `StructuredLogger`, `SecretString`, `validate_memories`.
- No leftover `http_client.py` here or under `application/utils/`.
- `access.py` deleted. No `optional_attr` / `invoke_if_callable`.

## ANTI-PATTERNS

- Do not log tokens or memory text.
- Do not add metrics/circuit-breaker modules here (removed).
- Do not import `application/` from this package.
- Credential parse/timeout rules live in `platforms/` — do not duplicate here.
