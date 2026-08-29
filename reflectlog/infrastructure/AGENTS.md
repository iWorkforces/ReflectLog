# Infrastructure Layer

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop
External integrations wrapping third-party libraries with protocol-based interfaces.

## STRUCTURE

```
infrastructure/
├── usearch_engine.py     # Semantic vector search (HNSW, batch, dual modes)
├── tantivy_engine.py     # Full-text search, soft-delete, tombstone cache
├── cross_encoder_reranker.py  # Local BAAI/bge-reranker-v2-m3
├── memory_store.py       # SQLite CRUD, archival/recovery
├── smart_replacer.py     # LLM update detection, 0.7 threshold
├── cached_embeddings.py  # LRU query cache (100 entries)
├── qwen3_embedding.py    # Qwen3 Langchain embeddings
├── llm_provider_base.py  # Base OpenAI provider protocol
├── search/              # Package marker only (live engines are in this directory)
├── reranker_post_processor.py  # Post-search reranking composition (cross-encoder + temporal)
└── embeddings/          # Re-exports (future expansion)

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|--------|
| USearch vector search | `usearch_engine.py` | HNSW, batch ops, dual modes |
| Tantivy full-text | `tantivy_engine.py` | Soft-delete, tombstone LRU cache |
| Cross-encoder | `cross_encoder_reranker.py` | Local BAAI/bge-reranker-v2-m3 |
| Memory storage | `memory_store.py` | SQLite CRUD, archival/recovery |
| Smart replacement | `smart_replacer.py` | LLM update detection, 0.7 threshold |
| Embedding cache | `cached_embeddings.py` | LRU cache, 100 entries |
| Qwen3 embeddings | `qwen3_embedding.py` | Langchain integration |
| Reranker composition | `reranker_post_processor.py` | Cross-encoder + temporal |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|--------|------|
| USearchEngine | Class | usearch_engine.py | High | Semantic search, HNSW, batch ops |
| TantivyEngine | Class | tantivy_engine.py | High | Full-text search, soft-delete |
| CrossEncoderReranker | Class | cross_encoder_reranker.py | High | Local reranking |
| MemoryStore | Class | memory_store.py | Medium | SQLite persistence |
| SmartReplacer | Class | smart_replacer.py | Medium | LLM memory update detection |
| CachedEmbeddings | Class | cached_embeddings.py | Medium | LRU query cache |
| RerankerPostProcessor | Class | reranker_post_processor.py | Medium | Post-search reranking composition |

## CONVENTIONS

**Protocol Wrappers** - All external libs wrapped in classes implementing core protocols (ISemanticSearchEngine, IReranker, IRerankerProvider).

**Lazy Initialization** - Expensive resources (embedders, rerankers, LLM providers) initialized on-demand with thread-safe patterns.

**Factory Methods** - Components created via `from_config()` class methods.

**Exception Wrapping** - Third-party errors wrapped in domain exceptions (VectorSearchError, RerankerError) with `from e` chaining.

**Soft-Delete** - Tantivy tombstones (`is_deleted`). A text is FTS-dead iff `tomb_count >= live_count`. One `delete()` plants enough tombs to hide every live copy; search skips tomb docs and dedupes. Disk errors during search raise `SearchError` (not `[]`). `compact()` is maintenance (ratio≥0.2 or count≥10000) and rebuilds with `_get_doc_limit()`.

**Fail-closed embeds** - Short/empty `embed_documents` raise. USearch: SQLite first, then vectors; mid-add rolls back HNSW keys + rows.

**LRU Caching** - `CachedEmbeddings` SHA-256 + lock. Tantivy tombstone LRU skip.

**Batch Operations** - USearch supports batch add/remove for bulk operations. Cross-encoder reranking uses batched inference.

**Connection Pooling** - SQLite connections reused. Tantivy writers context-managed.

## ANTI-PATTERNS

- Never assume USearch is thread-safe - serialize all writes with _write_lock
- Never cache LLM responses in search results (only embeddings)
- Never use bare `except:` - catch specific third-party exceptions (usearch.SearchError, tantivy.TantivyError)
- Never compact Tantivy inside request-path delete
- Never leave SQLite rows after a mid-batch `index.add` failure
- Never create new MemoryStore per request - reuse instance
- Never call LLM provider sync - all methods async
- Never use hard-coded model names - pull from config
