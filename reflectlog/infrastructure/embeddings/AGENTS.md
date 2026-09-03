# Agent Guidelines for reflectlog/infrastructure/embeddings/

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
Only populated child of `infrastructure/`. Qwen OpenRouter client + LRU cache.

## STRUCTURE

```
embeddings/
├── cached_embeddings.py  # SHA-256 LRU; fail-closed batches
└── qwen3_embedding.py    # OpenAI-compat OpenRouter client
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| LRU wrap | `cached_embeddings.py` | `embed_query` + per-text `embed_documents` |
| Qwen HTTP | `qwen3_embedding.py` | `LangchainQwenEmbeddings` name leftover |
| HTTP pool | `utility/http.py` | `HttpClientFactory` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `CachedEmbeddings` | Class | `cached_embeddings.py` | LRU; fail-closed short/empty |
| `LangchainQwenEmbeddings` | Class | `qwen3_embedding.py` | OpenRouter OpenAI-compat |

## CONVENTIONS

- Imports: `reflectlog.infrastructure.embeddings.*`. Not at package root.
- Qwen is OpenAI-compatible OpenRouter. Langchain name is leftover.
- Fail-closed on short/empty batches. Cache raises `RuntimeError` on size mismatch or empty vector. Qwen raises if `len(results) != len(texts)` or any empty item.
- `aembed_documents` may init slots as `[]`; leftover empty slots raise. Never treat that as a successful pad.
- No pad with `[]` to fake a complete batch.

## ANTI-PATTERNS

- Never put these modules back on `infrastructure/` root.
- Never treat empty/short embed batches as success.
- Never pad missing vectors with `[]`.
- Never log query text or API keys.

## NOTES

Parent engines stay FLAT. Empty sibling markers `llm/`, `memory/`, `reranking/` get no guides.
