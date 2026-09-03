# Reranking Unit Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Tests `reflectlog.utility.scoring`: batch min-max, threshold safety net, recency decay. This package is the pointer only.

## STRUCTURE

```
tests/unit/application/memory/reranking/
└── test_normalization.py
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_normalize_*` | Batch min-max to `[0, 1]` |
| `test_all_equal_scores` | All equal → 1.0 (not 0.5) |
| `test_threshold_*` | Safety net keeps `min_results` |
| `test_recency_*` | `exp(-rate * hours_old)` after CE normalize |

## KEY PATTERNS

```python
def test_normalize_single_item() -> None:
    result = normalize_reranker_scores([("doc", 0.5)])
    assert result == [("doc", 1.0)]
```

## ANTI-PATTERNS

- Never document all-equal as 0.5.
- Never normalize one score at a time.
- Never apply recency before CE normalize/threshold.
- Never skip CE tests for ≤1 hit (production skips CE).

## NOTES

Skip CE if ≤1 hit. Recency after threshold. Pure functions; mock time only for decay.
