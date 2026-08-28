# Infrastructure Layer

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop
External integrations wrapping third-party libraries with protocol-based interfaces.

## STRUCTURE

```
infrastructure/
├── usearch_engine.py     # Semantic vector search (HNSW, batch, dual modes)
├── tantivy_engine.py     # Full-text search, soft-delete, tombstone cache
├── llm_reranker.py       # LLM reranking, provider abstraction, temporal
├── cross_encoder_reranker.py  # Local BAAI/bge-reranker-v2-m3
├── memory_store.py       # SQLite CRUD, archival/recovery
├── smart_replacer.py     # LLM update detection, 0.7 threshold
├── cached_embeddings.py  # LRU query cache (100 entries)
├── qwen3_embedding.py    # Qwen3 Langchain embeddings
├── llm_provider_base.py  # Base OpenAI provider protocol
├── search/              # Package marker only (live engines are in this directory)
├── reranker_post_processor.py  # Post-search reranking composition (LLM + cross-encoder + temporal)
└── embeddings/          # Re-exports (future expansion)

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|--------|
| USearch vector search | `usearch_engine.py` | HNSW, batch ops, dual modes |
| Tantivy full-text | `tantivy_engine.py` | Soft-delete, tombstone LRU cache |
| LLM reranking | `llm_reranker.py` | Provider abstraction, temporal scoring |
| Cross-encoder | `cross_encoder_reranker.py` | Local BAAI/bge-reranker-v2-m3 |
| Memory storage | `memory_store.py` | SQLite CRUD, archival/recovery |
| Smart replacement | `smart_replacer.py` | LLM update detection, 0.7 threshold |
| Embedding cache | `cached_embeddings.py` | LRU cache, 100 entries |
| Qwen3 embeddings | `qwen3_embedding.py` | Langchain integration |
| Reranker composition | `reranker_post_processor.py` | LLM + cross-encoder + temporal |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|--------|------|
| USearchEngine | Class | usearch_engine.py | High | Semantic search, HNSW, batch ops |
| TantivyEngine | Class | tantivy_engine.py | High | Full-text search, soft-delete |
| LLMReranker | Class | llm_reranker.py | Medium | LLM-based scoring, temporal |
| CrossEncoderReranker | Class | cross_encoder_reranker.py | Low | Local reranking |
| MemoryStore | Class | memory_store.py | Medium | SQLite persistence |
| SmartReplacer | Class | smart_replacer.py | Medium | LLM memory update detection |
| CachedEmbeddings | Class | cached_embeddings.py | Medium | LRU query cache |
| RerankerPostProcessor | Class | reranker_post_processor.py | Medium | Post-search reranking composition |

## CONVENTIONS

**Protocol Wrappers** - All external libs wrapped in classes implementing core protocols (ISemanticSearchEngine, IReranker, IRerankerProvider).

**Lazy Initialization** - Expensive resources (embedders, rerankers, LLM providers) initialized on-demand with thread-safe patterns.

**Factory Methods** - Components created via `from_config()` class methods.

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
- Never create new MemoryStore per request - reuse instance
- Never call LLM provider sync - all methods async
- Never use hard-coded model names - pull from config
