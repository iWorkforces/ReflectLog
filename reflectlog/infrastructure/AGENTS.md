# Infrastructure Layer

External integrations wrapping third-party libraries with protocol-based interfaces.

## STRUCTURE

```
infrastructure/
├── search/           # Search engines (USearch, Tantivy)
├── embeddings/       # Embedding providers (Qwen3)
├── reranking/        # Rerankers (LLM, cross-encoder)
├── memory/           # Storage (MessageStore, SmartReplacer)
└── llm/              # LLM provider protocols
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|--------|
| USearch vector search | search/usearch_engine.py | HNSW, batch ops, dual modes |
| Tantivy full-text | search/tantivy_engine.py | Soft-delete, tombstone LRU cache |
| LLM reranking | reranking/llm_reranker.py | Provider abstraction, temporal scoring |
| Cross-encoder | reranking/cross_encoder.py | Local BAAI/bge-reranker-v2-m3 |
| Message storage | memory/message_store.py | SQLite CRUD, archival/recovery |
| Smart replacement | memory/smart_replacer.py | LLM update detection, 0.7 threshold |
| Embedding cache | embeddings/cached.py | LRU cache, 100 entry default |
| Qwen3 embeddings | embeddings/langchain_qwen.py | Langchain integration |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|--------|------|
| USearchEngine | Class | search/usearch_engine.py | High | Semantic search, libSQL storage |
| TantivyEngine | Class | search/tantivy_engine.py | High | Full-text search, soft-delete |
| LLMReranker | Class | reranking/llm_reranker.py | Medium | LLM-based scoring, temporal |
| CrossEncoderReranker | Class | reranking/cross_encoder.py | Low | Local reranking |
| MessageStore | Class | memory/message_store.py | Medium | SQLite persistence |
| SmartReplacer | Class | memory/smart_replacer.py | Medium | LLM memory update detection |
| CachedEmbeddings | Class | embeddings/cached.py | Medium | LRU query cache |

## CONVENTIONS

**Protocol Wrappers** - All external libs wrapped in classes implementing core protocols (ISearchBackend, IReranker, IRerankerProvider).

**Lazy Initialization** - Expensive resources (embedders, rerankers, LLM providers) initialized on-demand with thread-safe patterns.

**Factory Methods** - Components created via `from_config()` or `from_app_config()` class methods.

**Exception Wrapping** - Third-party errors wrapped in domain exceptions (VectorSearchError, RerankerError) with `from e` chaining.

**Soft-Delete Pattern** - Tantivy uses O(1) tombstone flag (is_deleted) instead of O(n) rebuild. Compacts at 20% tombstones.

**LRU Caching** - Query embeddings cached (default 100 entries) using MD5 hash keys. Tantivy tombstone cache for fast skip.

**Batch Operations** - USearch supports batch add/remove for bulk operations. LLM reranking uses batch scoring.

**Connection Pooling** - SQLite connections reused. Tantivy writers context-managed.

## ANTI-PATTERNS

- Never assume USearch is thread-safe - serialize all writes with _write_lock
- Never cache LLM responses in search results (only embeddings)
- Never use bare `except:` - catch specific third-party exceptions (usearch.SearchError, tantivy.TantivyError)
- Never mix LLM providers in same reranker instance (provider set once in __init__)
- Never skip tombstone compaction check in Tantivy (memory leak risk)
- Never create new MessageStore per request - reuse instance
- Never call LLM provider sync - all methods async
- Never use hard-coded model names - pull from config
