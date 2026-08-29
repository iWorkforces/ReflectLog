# Reranking Unit Tests

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW

Unit tests for batch min-max normalization and temporal recency decay. Pure functions, no mocking needed.

## STRUCTURE

```
tests/unit/application/memory/reranking/
└── test_normalization.py   # Score normalization, threshold filtering, decay
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_normalize_* | Batch min-max normalization edge cases |
| test_threshold_* | Safety net minimum results |
| test_recency_* | Exponential decay calculations |

## KEY PATTERNS

### Normalization Edge Cases
```python
def test_normalize_single_item():
    result = normalize_reranker_scores([("doc", 0.5)])
    assert result == [("doc", 1.0)]  # Single item = 1.0

def test_normalize_equal_scores():
    result = normalize_reranker_scores([("a", 0.5), ("b", 0.5)])
    assert all(s == 0.5 for _, s in result)  # All equal = 0.5 (neutral)
```

### Recency Decay Formula
```python
def test_decay_rate():
    # rate=0.01, 24 hours old
    # factor = exp(-0.01 * 24) = 0.787
    factor = calculate_recency_factor(created_at, decay_rate=0.01)
    assert 0.78 < factor < 0.79
```

## ANTI-PATTERNS

- Never test normalization individually - batch only
- Never apply decay before normalization

## NOTES

- **Pure functions**: No mocking required
- **Deterministic**: Time mocked for decay tests
