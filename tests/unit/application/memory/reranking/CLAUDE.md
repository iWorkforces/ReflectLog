# tests/unit/application/memory/reranking/

This directory contains unit tests for the reranker score normalization and recency decay utilities.

## Structure

```
reranking/
├── __init__.py              # Package marker
└── test_normalization.py    # Score normalization and recency decay tests
```

## Purpose

Tests for the `ccmemories/application/memory/reranking/` module components:

- **`normalize_reranker_scores()`**: Batch min-max normalization to [0, 1] range
- **`apply_threshold_with_safety_net()`**: Threshold filtering with optional safety net
- **`calculate_recency_factor()`**: Exponential decay factor based on memory age
- **`apply_recency_decay()`**: Apply recency decay to scored results and re-sort

## Test File: `test_normalization.py`

### Test Classes

#### `TestNormalizeRerankerScores`

Tests for the normalization function that transforms diverse reranker score ranges into a consistent [0, 1] range:

| Test | Purpose |
|------|---------|
| `test_empty_input` | Empty list returns empty list |
| `test_single_result` | Single result gets score 1.0 (best by definition) |
| `test_two_results` | Best=1.0, worst=0.0 |
| `test_cross_encoder_range` | CrossEncoder range (0.001-0.17) normalizes correctly |
| `test_llm_range` | LLM range (0.7-0.9) normalizes correctly |
| `test_all_equal_scores` | All equal scores become 1.0 |
| `test_preserves_document_order` | Input order is preserved |
| `test_negative_scores` | Negative scores are handled |

#### `TestApplyThresholdWithSafetyNet`

Tests for the threshold filtering function with safety net behavior:

| Test | Purpose |
|------|---------|
| `test_empty_input` | Empty list returns empty list |
| `test_all_above_threshold` | All scores above threshold are kept |
| `test_some_below_threshold` | Scores below threshold are filtered |
| `test_all_below_threshold_no_safety_net` | Returns empty when all below and min_results=0 |
| `test_safety_net_when_enabled` | Returns min_results when all below threshold |
| `test_safety_net_partial_filter` | Safety net kicks in when filtered too much |
| `test_safety_net_not_enough_candidates` | Returns all available if less than min_results |
| `test_threshold_zero` | Threshold 0.0 keeps all results |
| `test_threshold_one` | Threshold 1.0 only keeps perfect scores |
| `test_threshold_at_boundary` | Score equal to threshold is kept (inclusive) |
| `test_min_results_zero_default` | Default min_results=0 allows empty results |

#### `TestIntegration`

Integration tests combining normalization and threshold filtering:

| Test | Purpose |
|------|---------|
| `test_cross_encoder_workflow` | Simulates CrossEncoder reranking workflow |
| `test_llm_reranker_workflow` | Simulates LLM reranking workflow |

#### `TestCalculateRecencyFactor`

Tests for the exponential decay calculation:

| Test | Purpose |
|------|---------|
| `test_very_recent_memory` | Brand new memory gets factor ≈ 1.0 |
| `test_one_hour_old` | 1 hour old with rate=0.01 ≈ 0.99 |
| `test_half_life_decay` | At half-life (69h for rate=0.01), factor ≈ 0.5 |
| `test_very_old_memory` | Very old memory gets factor near 0 |
| `test_zero_decay_rate` | Zero rate always returns 1.0 (no decay) |
| `test_higher_decay_rate` | Rate=0.1 decays faster |
| `test_invalid_timestamp` | Invalid timestamp returns 1.0 gracefully |
| `test_future_timestamp` | Future timestamps clamped to 1.0 |

#### `TestApplyRecencyDecay`

Tests for applying decay to scored results:

| Test | Purpose |
|------|---------|
| `test_empty_input` | Empty list returns empty list |
| `test_no_timestamps` | Missing timestamps, results unchanged |
| `test_all_timestamps_present` | Decay applied and re-sorted |
| `test_partial_timestamps` | Some missing, only apply to available |
| `test_reordering_by_decay` | Old high-score beaten by new low-score |
| `test_zero_decay_rate` | Rate=0, no changes to scores |
| `test_preserves_message_content` | Messages not mutated |

## Running Tests

```bash
# All reranking tests
uv run pytest tests/unit/application/memory/reranking/ -v

# Specific test file
./start-unittest.sh --file tests/unit/application/memory/reranking/test_normalization.py

# Specific test class
uv run pytest tests/unit/application/memory/reranking/test_normalization.py::TestNormalizeRerankerScores -v

# With pattern matching
./start-unittest.sh --pattern "threshold"
```

## Key Test Patterns

### Pure Function Testing

The normalization functions are pure and don't require mocking:

```python
def test_normalize_scores_basic():
    """Test basic min-max normalization."""
    scored = [("doc1", 0.9), ("doc2", 0.1)]
    normalized = normalize_reranker_scores(scored)

    # Best score -> 1.0, worst score -> 0.0
    doc1_result = next(r for r in normalized if r[0] == "doc1")
    doc2_result = next(r for r in normalized if r[0] == "doc2")

    assert doc1_result[1] == 1.0
    assert doc2_result[1] == 0.0
```

### Parameterized Testing with pytest.approx

For floating-point comparisons:

```python
# Middle should be 0.75 ((0.85 - 0.70) / (0.90 - 0.70))
doc2_result = next(r for r in result if r[0] == "doc2")
assert doc2_result[1] == pytest.approx(0.75)
```

### Safety Net Testing Pattern

```python
def test_safety_net_behavior():
    # Results sorted by score descending
    scored = [("doc1", 0.4), ("doc2", 0.3)]  # All below 0.5

    # With safety net disabled
    result_no_safety = apply_threshold_with_safety_net(scored, 0.5, min_results=0)
    assert result_no_safety == []

    # With safety net enabled
    result_with_safety = apply_threshold_with_safety_net(scored, 0.5, min_results=1)
    assert len(result_with_safety) == 1
    assert result_with_safety[0][0] == "doc1"  # Top result returned
```

### Recency Decay Testing Pattern

```python
from datetime import datetime, timedelta

def test_recency_decay_reorders():
    """Newer memory with lower base score can beat older higher score."""
    now = datetime.now()

    scored = [
        ("old_memory", 0.9),  # Higher base score
        ("new_memory", 0.8),  # Lower base score but newer
    ]
    timestamp_map = {
        "old_memory": (now - timedelta(hours=100)).isoformat(),  # Old
        "new_memory": (now - timedelta(hours=1)).isoformat(),    # Recent
    }

    decayed = apply_recency_decay(scored, timestamp_map, decay_rate=0.01)

    # New memory should now rank higher after decay applied
    assert decayed[0][0] == "new_memory"
```

## Test Data Patterns

### CrossEncoder Score Range

Typical range: 0.001-0.17 (very low values)

```python
scored = [
    ("doc1", 0.17),   # Best
    ("doc2", 0.05),   # Middle
    ("doc3", 0.001),  # Worst
]
```

### LLM Reranker Score Range

Typical range: 0.7-0.9 (narrow, high range)

```python
scored = [
    ("doc1", 0.90),  # Best
    ("doc2", 0.85),  # Middle
    ("doc3", 0.70),  # Worst
]
```

## Test Coverage Goals

Target **95%+ coverage** for normalization utilities:

- All function branches
- All edge cases (empty, single, all equal)
- Boundary conditions (threshold at boundary)
- Safety net behavior
- Different score distributions

## Related Files

- **Source**: `ccmemories/application/memory/reranking/normalization.py`
- **Parent tests**: `tests/unit/application/memory/CLAUDE.md`
- **Integration tests**: `tests/integration/test_memory_manager_usearch.py`
