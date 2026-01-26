# Agent Guidelines for reflectlog/application/memory/reranking/

This directory contains score normalization and temporal decay utilities for rerankers. It provides batch-level min-max normalization to transform diverse reranker score ranges into a consistent [0, 1] range, enabling unified threshold semantics.

## Directory Structure

```
reranking/
├── __init__.py          # Package exports (normalize_reranker_scores, apply_threshold_with_safety_net, calculate_recency_factor, apply_recency_decay)
└── normalization.py     # Min-max normalization and recency decay functions
```

## Core Responsibilities

### Score Normalization

Different reranking engines produce fundamentally different score distributions:

| Reranker | Typical Score Range | Characteristics |
|----------|---------------------|-----------------|
| LLMReranker | 0.7-0.9 | Prompt-calibrated, narrow range |
| CrossEncoderReranker | 0.001-0.17 | Sigmoid-normalized logits, very low values |

This module provides batch-level min-max normalization to transform these diverse score ranges into a consistent `[0, 1]` range, enabling:

- **Unified thresholds**: A threshold of 0.5 consistently means "above median relevance"
- **Cross-reranker comparability**: Results from different rerankers can be meaningfully compared
- **Safety nets**: Guarantee minimum results even when all scores fall below threshold

### Temporal Decay

Applies exponential decay based on memory age to favor recent memories:

```
decayed_score = normalized_score * exp(-rate * hours_old)
```

## Key Functions

### normalize_reranker_scores

Transforms reranker scores to `[0, 1]` using batch min-max normalization:

```python
def normalize_reranker_scores(
    scored_results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    '''Normalize reranker scores to 0-1 range using batch min-max.

    Args:
        scored_results: List of (document, score) tuples.

    Returns:
        Normalized list with scores in 0-1 range.
        Best score = 1.0, worst score = 0.0.
    '''
```

**Behavior:**
- Best score in batch = 1.0
- Worst score in batch = 0.0
- Equal scores share normalized positions appropriately

### apply_threshold_with_safety_net

Applies threshold filtering with guaranteed minimum results:

```python
def apply_threshold_with_safety_net(
    normalized_results: list[tuple[str, float]],
    threshold: float = 0.5,
    min_results: int = 0,
) -> list[tuple[str, float]]:
    '''Apply threshold with safety net for minimum results.

    Args:
        normalized_results: Normalized (document, score) tuples.
        threshold: Minimum score to keep (default: 0.5).
        min_results: Minimum results to return (default: 0, disabled).

    Returns:
        Filtered results meeting threshold, or top min_results if safety net triggers.
    '''
```

**Safety Net Behavior:**
- If fewer than `min_results` pass threshold, return top `min_results`
- Guarantees results even when all scores are low
- Configurable per deployment needs

### calculate_recency_factor

Computes exponential decay factor based on memory age:

```python
def calculate_recency_factor(
    created_at: str,
    decay_rate: float = 0.01,
    now: datetime | None = None,
) -> float:
    '''Calculate recency decay factor for a memory.

    Args:
        created_at: ISO timestamp of memory creation.
        decay_rate: Exponential decay rate per hour (default: 0.01).
        now: Current time for calculation (default: now()).

    Returns:
        Decay factor between 0 and 1.
        1.0 = just created, approaches 0 as age increases.
    '''
```

**Decay Formula:**
```
factor = exp(-decay_rate * hours_old)
```

### apply_recency_decay

Applies recency decay to normalized scores:

```python
def apply_recency_decay(
    scored_results: list[tuple[str, float]],
    timestamp_map: dict[str, str],
    decay_rate: float = 0.01,
) -> list[tuple[str, float]]:
    '''Apply recency decay to reranked results.

    Args:
        scored_results: Normalized (document, score) tuples.
        timestamp_map: Mapping of document to creation timestamp.
        decay_rate: Exponential decay rate per hour.

    Returns:
        Results with decayed scores, re-sorted by decayed score.
    '''
```

**Flow:**
```
Reranker Score → [Batch Normalize] → [Apply Decay] → [Re-sort] → Results
                  0-1 range          score * exp(-rate * hours)   by decayed score
```

## Usage Examples

### Basic Normalization

```python
from reflectlog.application.memory.reranking import normalize_reranker_scores

# CrossEncoder typical range (0.001-0.17)
scored = [("doc1", 0.17), ("doc2", 0.05), ("doc3", 0.001)]
normalized = normalize_reranker_scores(scored)
# normalized[0] = ("doc1", 1.0)  # best score
# normalized[2] = ("doc3", 0.0)  # worst score
```

### Threshold with Safety Net

```python
from reflectlog.application.memory.reranking import apply_threshold_with_safety_net

# All scores below threshold
results = [("doc1", 0.3), ("doc2", 0.25), ("doc3", 0.2)]
filtered = apply_threshold_with_safety_net(results, threshold=0.5, min_results=2)
# Returns top 2 results since none pass threshold
```

### Recency Decay

```python
from reflectlog.application.memory.reranking import apply_recency_decay

results = [("doc1", 1.0), ("doc2", 0.9), ("doc3", 0.8)]
timestamp_map = {
    "doc1": "2024-01-01T00:00:00Z",  # 24 hours old
    "doc2": "2024-01-01T12:00:00Z",  # 12 hours old
    "doc3": "2024-01-02T00:00:00Z",  # 0 hours old (newest)
}

decayed = apply_recency_decay(results, timestamp_map, decay_rate=0.01)
# doc3 (newest) may rank higher despite lower original score
```

## Decay Rate Configuration

The decay rate controls how quickly older memories lose relevance:

| Rate | Half-life | Use Case |
|------|-----------|----------|
| 0.001 | ~693 hours (~29 days) | Long-term preferences |
| 0.01 | ~69 hours (~3 days) | General memories |
| 0.05 | ~14 hours | Fast-changing context |
| 0.1 | ~7 hours | Session-specific data |

**Formula:** `score = original_score * exp(-rate * hours_old)`

### Examples

```python
# After 24 hours with rate=0.01
factor = exp(-0.01 * 24) = exp(-0.24) = 0.787
# Score reduced to 78.7% of original

# After 72 hours (3 days) with rate=0.01
factor = exp(-0.01 * 72) = exp(-0.72) = 0.487
# Score reduced to 48.7% of original
```

## Key Patterns

### Pipeline Integration

Integrate normalization and decay into the search pipeline:

```python
def _rerank(
    self,
    query: str,
    candidates: list[tuple[str, float]],
    timestamp_map: dict[str, str],
) -> list[tuple[str, float]]:
    # Get reranker scores
    reranked = self._reranker.rerank(query, [c[0] for c in candidates])

    # Normalize scores to 0-1 range
    normalized = normalize_reranker_scores(reranked)

    # Apply recency decay
    decayed = apply_recency_decay(normalized, timestamp_map)

    # Sort by decayed score
    decayed.sort(key=lambda x: x[1], reverse=True)

    return decayed
```

### Batch Processing

Process all candidates in a batch for normalization:

```python
def _batch_rerank(
    self,
    query: str,
    candidates: list[str],
) -> list[tuple[str, float]]:
    # Get scores for all candidates
    scored = self._reranker.rerank_batch(query, candidates)

    # Normalize as a batch
    normalized = normalize_reranker_scores(scored)

    return normalized
```

## Error Handling

### Empty Input

```python
def normalize_reranker_scores(scored_results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not scored_results:
        return []

    if len(scored_results) == 1:
        # Single item is always 1.0
        return [(scored_results[0][0], 1.0)]

    # Normalize batch
    ...
```

### Invalid Timestamps

```python
def calculate_recency_factor(
    created_at: str,
    decay_rate: float = 0.01,
    now: datetime | None = None,
) -> float:
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        # Invalid timestamp, return full factor (no decay)
        return 1.0

    hours_old = (now - created).total_seconds() / 3600
    return max(0.0, min(1.0, math.exp(-decay_rate * hours_old)))
```

## Testing Guidelines

### Unit Tests

- Test normalization with known score distributions
- Verify threshold filtering behavior
- Test recency decay calculations
- Validate edge cases (empty input, single item, all equal scores)

### Test Cases

```python
def test_normalize_single_item():
    '''Single item should normalize to 1.0.'''
    result = normalize_reranker_scores([("doc", 0.5)])
    assert result == [("doc", 1.0)]

def test_normalize_equal_scores():
    '''Equal scores should all normalize to 1.0.'''
    result = normalize_reranker_scores([("doc1", 0.5), ("doc2", 0.5)])
    assert result[0][1] == 1.0
    assert result[1][1] == 1.0

def test_threshold_safety_net():
    '''Safety net should return min_results when threshold not met.'''
    results = [("doc1", 0.1), ("doc2", 0.05)]
    filtered = apply_threshold_with_safety_net(results, threshold=0.5, min_results=1)
    assert len(filtered) == 1

def test_recency_decay():
    '''Newer memories should have higher decayed scores.'''
    now = datetime.now()
    old = [("old_doc", 1.0)]
    new = [("new_doc", 0.8)]

    timestamp_map = {
        "old_doc": (now - timedelta(hours=72)).isoformat(),
        "new_doc": (now - timedelta(hours=1)).isoformat(),
    }

    # Newer doc should rank higher despite lower original score
    decayed = apply_recency_decay(old + new, timestamp_map)
    assert decayed[0][0] == "new_doc"
```

## Dependencies

### Internal Dependencies

- `application/memory/protocols.py`: Search result types
- `application/utils/`: Logging utilities

### External Dependencies

- `math`: For exponential calculations
- `datetime`: For timestamp handling

## Important Notes

### Score Distribution Independence

After normalization, the actual score ranges don't matter:

| Reranker | Raw Range | After Normalization |
|----------|-----------|---------------------|
| LLMReranker | 0.7-0.9 | 0-1 |
| CrossEncoderReranker | 0.001-0.17 | 0-1 |

This enables a single threshold value (e.g., 0.5) to work consistently across different reranker types.

### Batch Normalization Important

Normalization must be done as a batch, not individually:

```python
# Correct - batch normalization
all_scores = [("doc1", 0.17), ("doc2", 0.05), ("doc3", 0.001)]
normalized = normalize_reranker_scores(all_scores)

# Incorrect - individual normalization loses relative information
doc1 = normalize_single(0.17)  # Would be 1.0
doc2 = normalize_single(0.05)  # Would be 1.0
# Lost the distinction between doc1 and doc2
```

### Recency Decay Order

Apply normalization before recency decay:

1. Reranker produces raw scores (0.001-0.17 for CrossEncoder)
2. Normalize to 0-1 range
3. Apply recency decay to normalized scores
4. Re-sort by decayed score

This ensures fair comparison across documents with different ages.
