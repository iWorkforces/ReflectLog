# Infrastructure Layer

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW
Protocol wrappers. Live engines stay FLAT here. `embeddings/` is the only populated child.

## STRUCTURE

```
infrastructure/
├── usearch_engine.py           # HNSW + SQLite SoT
├── tantivy_engine.py           # FTS + tombstones
├── memory_store.py             # Identity + journal
├── cross_encoder_reranker.py   # Local FlagReranker
├── reranker_post_processor.py  # CE + temporal compose
├── smart_replacer.py           # LLM replacement
├── llm_provider_base.py        # OpenAI provider base
├── embeddings/                 # ONLY populated child
│   ├── cached_embeddings.py
│   └── qwen3_embedding.py
├── search/                     # empty marker + guide
├── llm/                        # empty marker (no guide)
├── memory/                     # empty marker (no guide)
└── reranking/                  # empty marker (no guide)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Semantic + SoT | `usearch_engine.py` | `get_all`/`count` via MemoryStore |
| FTS | `tantivy_engine.py` | Tombstone+commit; `compact()` only |
| Identity/journal | `memory_store.py` | unique `(workspace_id, content)` |
| CE Step 4 | `cross_encoder_reranker.py` | `rerank` always scores |
| CE skip ≤1 | `application/memory/search_strategies.py` | SearchPipeline, not CE |
| Embed cache | `embeddings/cached_embeddings.py` | SHA-256 LRU |
| Qwen client | `embeddings/qwen3_embedding.py` | OpenRouter OpenAI-compat |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `USearchEngine` | Class | `usearch_engine.py` | HNSW + SQLite SoT |
| `TantivyEngine` | Class | `tantivy_engine.py` | FTS + tombs |
| `MemoryStore` | Class | `memory_store.py` | Identity + journal |
| `CrossEncoderReranker` | Class | `cross_encoder_reranker.py` | Search Step 4 |
| `CachedEmbeddings` | Class | `embeddings/cached_embeddings.py` | LRU; fail-closed batches |
| `LangchainQwenEmbeddings` | Class | `embeddings/qwen3_embedding.py` | OpenRouter client |

## CONVENTIONS

- Factories: `from_config()`, not `from_app_config()`.
- `get_all()` / `count()` SoT is MemoryStore via USearch.
- HNSW fail-closed if SQLite missing/empty/unreadable.
- Persist: SQLite insert then vectors; rollback HNSW+rows on embed/index fail.
- Tantivy delete = tombstone+commit. `compact()` is maintenance only.
- Search `OSError` → `SearchError`, never `[]`.
- Journal kinds `add|delete|replace`. Never dedupe add intents by `old_memory_id=0`.
- Protocols at boundaries. `raise ... from e`. No `getattr` / `optional_attr`.

## ANTI-PATTERNS

- No `cached_embeddings.py` / `qwen3_embedding.py` at this package root.
- No HNSW load when SQLite is missing/empty/unreadable.
- No compact-on-delete. No leftover rows after mid-batch `index.add` fail.
- No CE skip ≤1 inside `CrossEncoderReranker.rerank` — SearchPipeline owns that.
- No journal dedupe of `kind=add` via `old_memory_id=0`.
- No `getattr`, `optional_attr()`, `from_app_config()`.
- No extra AGENTS.md under `llm/`, `memory/`, `reranking/`.

## NOTES

Children: only `embeddings/` (full) and `search/` (marker). Empty `llm/`, `memory/`, `reranking/` stay markers.
