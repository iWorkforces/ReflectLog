# Agent Guidelines for reflectlog/infrastructure/embeddings/

**Generated:** 2026-08-30
**Commit:** 062b44f
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
| Engine consume | `../usearch_engine.py` | Embed outside write lock |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `CachedEmbeddings` | Class | `cached_embeddings.py` | LRU; fail-closed short/empty |
| `LangchainQwenEmbeddings` | Class | `qwen3_embedding.py` | OpenRouter OpenAI-compat |

## CONVENTIONS

- Not at package root. Imports: `reflectlog.infrastructure.embeddings.*`.
- Qwen is OpenAI-compatible OpenRouter. Langchain name is leftover.
- `CachedEmbeddings` fail-closed on short/empty batches. No pad with `[]`.
- Factories: `from_config()`, not `from_app_config()`.
- No `getattr` / `optional_attr`.

## ANTI-PATTERNS

- Never put `cached_embeddings.py` / `qwen3_embedding.py` back on `infrastructure/` root.
- Never treat empty/short embed batches as success.
- Never log query text or API keys.
- Never use `from_app_config()`.

## NOTES

Parent engines stay FLAT. This is the only extra guide besides `search/`.
Empty sibling markers `llm/`, `memory/`, `reranking/` get no guides.
