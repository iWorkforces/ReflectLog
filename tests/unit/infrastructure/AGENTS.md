# Infrastructure Unit Tests

**Generated:** 2026-02-21
**Commit:** 4c3af26
**Branch:** develop

## OVERVIEW

Unit tests for infrastructure layer: USearch, Tantivy, memory store, rerankers, embeddings. Uses file fixtures and mocks external APIs.

## STRUCTURE

```
tests/unit/infrastructure/
├── test_tantivy_engine.py           # Full-text search (95KB)
├── test_memory_store.py             # SQLite CRUD (61KB)
├── test_usearch_engine.py           # Vector search (47KB)
├── test_llm_reranker.py             # LLM reranking (46KB)
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
| test_llm_reranker.py | Provider abstraction, parallel scoring |

## KEY PATTERNS

### Temporary Index Fixtures
```python
@pytest.fixture
def tantivy_engine(tmp_path):
    engine = TantivyEngine(index_path=tmp_path / "tantivy")
    yield engine
    engine.close()
```

### Mock LLM Provider
```python
@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock(spec=ILLMProvider)
    provider.complete.return_value = '{"score": 0.85}'
    return provider
```

### Soft-Delete Verification
```python
def test_soft_delete_creates_tombstone(tantivy_engine):
    tantivy_engine.delete("message text")
    stats = tantivy_engine.get_tombstone_stats()
    assert stats.count == 1
```

## ANTI-PATTERNS

- Never use real LLM API calls in unit tests
- Never skip tombstone cleanup verification
- Never share tmp_path between tests

## NOTES

- **Large test files**: test_tantivy_engine.py is 95KB
- **File-based fixtures**: Uses tmp_path for isolation
- **Thread-safety**: Some tests verify lock behavior
