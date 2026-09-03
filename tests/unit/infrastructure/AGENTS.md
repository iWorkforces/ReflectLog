# Infrastructure Unit Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Real USearch, Tantivy, SQLite store on `tempfile.TemporaryDirectory` (or `tmp_path`). Embeddings/CE/LLM provider mocked or local. No live LLM HTTP.

## STRUCTURE

```
tests/unit/infrastructure/
├── test_tantivy_engine.py
├── test_memory_store.py
├── test_usearch_engine.py
├── test_storage_coordinator.py
├── test_cross_encoder_reranker.py
├── test_smart_replacer.py
├── test_qwen3_embedding.py
├── test_cached_embeddings.py
├── test_replacement_transitions.py
├── test_reranker_post_processor.py
├── test_llm_provider_base.py
└── test_import_patterns.py
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_tantivy_engine.py` | FTS, tombstones, no compact-on-delete |
| `test_memory_store.py` | Identity + journal `add\|delete\|replace` |
| `test_usearch_engine.py` | HNSW + SQLite SoT; fail-closed empty SQLite |
| `test_storage_coordinator.py` | Portalocker lease; focused CI includes this |
| `test_cached_embeddings.py` | LRU; short-batch raise (no pad with `[]`) |

## ANTI-PATTERNS

- Never call a real LLM API.
- Never share tmp dirs across tests.
- Never compact-on-delete. Never HNSW-load when SQLite is missing/empty.
- Never add `_rebuild_index_with_docs` tests (API removed).
- No MagicMock auto-attrs as engine APIs.

## NOTES

`get_all()` / `count()` SoT is USearch/SQLite. Journal later-write-wins. Focused CI runs a subset of these files, not the whole folder.
