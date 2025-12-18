# ccmemories/application/memory/reranking/

This directory contains score normalization utilities for rerankers to enable unified threshold semantics.

## Structure

```
reranking/
├── __init__.py          # Package exports (normalize_reranker_scores, apply_threshold_with_safety_net)
└── normalization.py     # Min-max batch normalization functions
```

## Purpose

Different reranking engines produce fundamentally different score distributions:

| Reranker | Typical Score Range | Characteristics |
|----------|---------------------|-----------------|
| LLMReranker | 0.7-0.9 | Prompt-calibrated, narrow range |
| CrossEncoderReranker | 0.001-0.17 | Sigmoid-normalized logits, very low values |

This module provides batch-level min-max normalization to transform these diverse score ranges into a consistent `[0, 1]` range, enabling:

- **Unified thresholds**: A threshold of 0.5 consistently means "above median relevance"
- **Cross-reranker comparability**: Results from different rerankers can be meaningfully compared
- **Safety nets**: Guarantee minimum results even when all scores fall below threshold

## Key Functions

### `normalize_reranker_scores(scored_results) -> List[Tuple[str, float]]`

Transforms reranker scores to `[0, 1]` using batch min-max normalization:

```python
from ccmemories.application.memory.reranking import normalize_reranker_scores

# CrossEncoder typical range (0.001-0.17)
scored = [("doc1", 0.17), ("doc2", 0.05), ("doc3", 0.001)]
normalized = normalize_reranker_scores(scored)
# normalized[0] = ("doc1", 1.0)  # best score
# normalized[2] = ("doc3", 0.0)  # worst score
```

**Behavior**:
- Best score in batch = 1.0
- Worst score in batch = 0.0
- Single result = 1.0 (best by definition)
- All equal scores = 1.0 (all equally good)

### `apply_threshold_with_safety_net(scored_results, threshold, min_results) -> List[Tuple[str, float]]`

Filters results by score threshold with optional safety net:

```python
from ccmemories.application.memory.reranking import apply_threshold_with_safety_net

# With safety net disabled (default)
scored = [("doc1", 0.4), ("doc2", 0.3)]  # all below 0.5
filtered = apply_threshold_with_safety_net(scored, 0.5, min_results=0)
# filtered = []  (all filtered out)

# With safety net enabled
filtered = apply_threshold_with_safety_net(scored, 0.5, min_results=1)
# filtered = [("doc1", 0.4)]  (top 1 returned despite being below threshold)
```

**Parameters**:
- `threshold`: Minimum score (inclusive) to keep results
- `min_results`: Safety net - guarantee at least N results (0 = disabled)

## Configuration

Related environment variables (via `Config`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_BATCH_NORMALIZE` | true | Enable batch min-max normalization |
| `RERANKER_MIN_RESULTS` | 0 | Safety net: min results to return (0 = disabled) |
| `SEARCH_SCORE_THRESHOLD` | 0.5 | Min score after normalization |
| `CROSS_ENCODER_SCORE_THRESHOLD` | 0.0 | Min cross-encoder score |

## Implementation Details

### Numba Acceleration

The core normalization uses `normalize_scores_minmax()` from `ccmemories.application.utils.numba_utils`:

```python
from ccmemories.application.utils.numba_utils import normalize_scores_minmax

# Numba JIT-compiled for performance
scores = np.array([0.17, 0.05, 0.001], dtype=np.float64)
normalized = normalize_scores_minmax(scores)  # [1.0, 0.29, 0.0]
```

### Edge Cases

1. **Empty input**: Returns empty list
2. **Single result**: Returns score of 1.0 (best by definition)
3. **All equal scores**: Returns 1.0 for all (all equally good)
4. **Safety net + empty threshold results**: Returns top N results

## Testing

Tests are located in `tests/unit/application/memory/reranking/test_normalization.py`:

```bash
./start-unittest.sh --file tests/unit/application/memory/reranking/test_normalization.py
```

Key test scenarios:
- Various score distributions (LLM-like, CrossEncoder-like)
- Edge cases (empty, single, all equal)
- Safety net behavior
- Numba function correctness

## Usage in MemoryManager

The normalization functions are called during the reranking step (Step 4):

```python
# In MemoryManager.search():
if self.config.reranker_batch_normalize:
    scored_results = normalize_reranker_scores(scored_results)

filtered_results = apply_threshold_with_safety_net(
    scored_results,
    threshold=self.config.search_score_threshold,
    min_results=self.config.reranker_min_results,
)
```

## Dependencies

- `numpy`: Array operations
- `ccmemories.application.utils.numba_utils`: JIT-compiled normalization
