# tests/unit/application/memory/

This directory contains unit tests for the memory management subsystem.

## Structure

```
memory/
└── reranking/                    # Reranking-specific tests
    ├── __init__.py               # Package marker
    └── test_normalization.py     # Score normalization tests
```

## Purpose

Tests for the `openmemories/application/memory/` module components:

- **Score normalization** (`reranking/`): Tests for `normalize_reranker_scores()` and `apply_threshold_with_safety_net()`

Note: The main `MemoryManager` tests are in `tests/unit/application/test_memory_manager.py` for historical reasons.

## Test Files

### `reranking/test_normalization.py`

Tests for score normalization utilities:

```python
class TestNormalizeRerankerScores:
    """Tests for normalize_reranker_scores()"""
    # - Empty input handling
    # - Single result handling
    # - All equal scores handling
    # - Various score distributions
    # - LLM-like score ranges (0.7-0.9)
    # - CrossEncoder-like score ranges (0.001-0.17)

class TestApplyThresholdWithSafetyNet:
    """Tests for apply_threshold_with_safety_net()"""
    # - Basic threshold filtering
    # - Safety net behavior (min_results > 0)
    # - Edge cases (empty, all below threshold)
    # - Boundary conditions
```

## Running Tests

```bash
# All memory tests
uv run pytest tests/unit/application/memory/ -v

# Specific test file
./start-unittest.sh --file tests/unit/application/memory/reranking/test_normalization.py

# With pattern matching
./start-unittest.sh --pattern "normalize"
```

## Related Test Files

- `tests/unit/application/test_memory_manager.py` - Main MemoryManager unit tests
- `tests/unit/application/test_ranx_fusion.py` - RRF fusion algorithm tests
- `tests/integration/test_memory_manager_usearch.py` - Integration tests with real engines

## Mocking Strategy

Score normalization tests typically don't require mocking since they're pure functions:

```python
def test_normalize_llm_like_scores():
    """Test normalization of LLM-like score distribution."""
    scored = [("doc1", 0.9), ("doc2", 0.85), ("doc3", 0.7)]
    normalized = normalize_reranker_scores(scored)

    # Best score = 1.0, worst score = 0.0
    assert normalized[0][1] == 1.0
    assert normalized[2][1] == 0.0
```

## Test Coverage Goals

Target **95%+ coverage** for normalization utilities:
- All function branches
- All edge cases
- All error paths
