# Infrastructure Unit Tests

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Unit tests for USearch, Tantivy, SQLite store, embeddings, CE reranker, and LLM provider base. File fixtures via `tmp_path`. No live LLM HTTP.

## STRUCTURE
```
tests/unit/infrastructure/
├── test_tantivy_engine.py
├── test_memory_store.py
├── test_usearch_engine.py
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
| `test_cached_embeddings.py` | LRU; short-batch raise (no pad with `[]`) |
| `test_cross_encoder_reranker.py` | Skip CE if ≤1 hit; recency after normalize |
| `test_replacement_transitions.py` | Durable journal transitions |
| `test_import_patterns.py` | No `infrastructure` → `application` imports |

## ANTI-PATTERNS
- Never call a real LLM API.
- Never share `tmp_path` across tests.
- Never compact-on-delete. Never HNSW-load when SQLite is missing/empty.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`. No MagicMock auto-attrs as engine APIs.

## NOTES
`get_all()` / `count()` SoT is USearch/SQLite. Journal later-write-wins.
