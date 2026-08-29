# Memory Pipeline Unit Tests

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW

Unit tests for 3-phase add pipeline and 4-step search pipeline. Tests manager, strategies, and pipeline orchestration.

## STRUCTURE

```
tests/unit/application/memory/
├── test_add_phases.py               # Phase 1/2/3 implementations
├── test_search_strategies.py        # 4-step search
├── test_replacement_recovery.py     # Pending transition reconcile
├── test_search_pipeline.py          # Search orchestration (37KB)
├── test_manager.py                  # MemoryManager unit tests (35KB)
├── test_engine_factory.py           # Engine creation (26KB)
└── conftest.py                      # NUMBA_DISABLE_JIT=1
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_add_phases.py | DuplicateDetection, SmartReplacement, Storage phases |
| test_search_strategies.py | RRF fusion, threshold filtering, reranking |
| test_manager.py | Facade coordination, lock hierarchy |
| test_engine_factory.py | USearch/Tantivy lazy initialization |

## KEY PATTERNS

### NUMBA Disable for Coverage
```python
# conftest.py
os.environ["NUMBA_DISABLE_JIT"] = "1"
# Required for ranx + coverage compatibility
```

### Phase Testing
```python
async def test_duplicate_detection_phase():
    phase = DuplicateDetectionPhase(store, engine)
    result = await phase.run(messages)
    assert result.duplicates == expected_duplicates
    assert result.new_messages == expected_new
```

### Pipeline Verification
```python
async def test_search_pipeline_returns_fused_memories():
    pipeline = SearchPipeline(
        semantic_engine=semantic,
        tantivy_engine=tantivy,
        fusion_engine=fusion,
        config=config,
        logger=logger,
        memory_manager=manager,
    )
    result = await pipeline.execute(SearchContext(...))
    assert result.memories == expected
```

## ANTI-PATTERNS

- Never enable NUMBA JIT in these tests (breaks coverage)
- Never bypass phase isolation - test phases independently
- Never mock internal phase dependencies incorrectly

## NOTES

- **NUMBA disabled**: Required for coverage reporting
- **Phase isolation**: Each phase testable independently
- **Pipeline order matters**: Stages execute sequentially
