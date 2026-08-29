# Infrastructure Unit Tests

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW

Unit tests for infrastructure layer: USearch, Tantivy, memory store, rerankers, embeddings. Uses file fixtures and mocks external APIs.

## STRUCTURE

```
tests/unit/infrastructure/
├── test_tantivy_engine.py           # Full-text search (95KB)
├── test_memory_store.py             # SQLite CRUD (61KB)
├── test_usearch_engine.py           # Vector search (47KB)
├── test_cross_encoder_reranker.py   # Cross-encoder (38KB)
├── test_smart_replacer.py           # Replacement detection (36KB)
├── test_qwen3_embedding.py          # Qwen embeddings (26KB)
└── test_cached_embeddings.py        # Embedding cache (18KB)
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_tantivy_engine.py | Soft-delete, tombstone cache, compaction |
| test_memory_store.py | Batch CRUD, archive/restore |
| test_usearch_engine.py | Exact vs approximate search |

## ANTI-PATTERNS

- Never use real LLM API calls in unit tests
- Never skip tombstone cleanup verification
- Never share tmp_path between tests

## NOTES
- Large test files (test_tantivy_engine.py is 95KB), file-based fixtures via tmp_path
