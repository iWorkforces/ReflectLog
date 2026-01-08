# ccmemories/application/memory/reranking/

This directory contains score normalization and temporal decay utilities for rerankers.

## Structure

```
reranking/
├── __init__.py          # Package exports (normalize_reranker_scores, apply_threshold_with_safety_net, calculate_recency_factor, apply_recency_decay)
└── normalization.py     # Min-max normalization and recency decay functions
```

## Purpose

This module provides two key capabilities:

### 1. Score Normalization

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

### 2. Recency Decay

The module also provides temporal-aware scoring to handle contradictory memories (e.g., "I like cats" vs "I don't like cats anymore").

### `calculate_recency_factor(timestamp_iso: str, decay_rate: float, now: datetime | None = None) -> float`

Calculates exponential decay factor based on memory age:

```python
from ccmemories.application.memory.reranking import calculate_recency_factor

# 10 hours old with default decay rate
factor = calculate_recency_factor("2024-01-15T10:00:00", 0.01)
# factor ≈ 0.905  (exp(-0.01 * 10))

# Very old memory
factor = calculate_recency_factor("2024-01-01T00:00:00", 0.01)
# factor ≈ 0.0  (very low, memory is old)

# With explicit 'now' for testing
from datetime import datetime, timezone
now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
factor = calculate_recency_factor("2024-01-15T10:00:00", 0.01, now=now)
```

**Parameters**:
- `timestamp_iso`: ISO 8601 timestamp string (e.g., "2024-01-15T10:30:00+00:00")
- `decay_rate`: Decay rate per hour (e.g., 0.01 for ~50% decay at 69 hours)
- `now`: Optional datetime for testing. If None, uses `datetime.now(timezone.utc)`

**Formula**: `factor = exp(-decay_rate * hours_old)`

**Decay Rate Examples**:

| Rate | Half-life (hours) | Use Case |
|------|-------------------|----------|
| 0.001 | ~693 | Long-term preferences |
| 0.01 (default) | ~69 | General memories |
| 0.05 | ~14 | Fast-changing context |
| 0.1 | ~7 | Session-specific data |

### `apply_recency_decay(scored_results, timestamp_map, decay_rate, now=None) -> List[Tuple[str, float]]`

Applies recency decay to scored results and re-sorts by decayed score:

```python
from ccmemories.application.memory.reranking import apply_recency_decay

scored = [("new memory", 0.8), ("old memory", 0.9)]
timestamp_map = {
    "new memory": "2024-01-15T12:00:00",  # 2 hours ago
    "old memory": "2024-01-10T12:00:00",  # 5 days ago
}

decayed = apply_recency_decay(scored, timestamp_map, decay_rate=0.01)
# "new memory" gets higher decayed score despite lower base score
# because it's more recent
```

**Parameters**:
- `scored_results`: List of (document, score) tuples from reranker
- `timestamp_map`: Dict mapping document text to ISO timestamp strings
- `decay_rate`: Decay rate per hour (e.g., 0.01 for ~50% decay at 69 hours)
- `now`: Optional datetime for testing. If None, uses `datetime.now(timezone.utc)`

**Behavior**:
- Multiplies each score by `calculate_recency_factor(created_at, decay_rate, now)`
- Re-sorts results by decayed score (descending)
- Preserves original score for documents without timestamps in the map
- Graceful handling: missing timestamps use factor of 1.0 (no decay)

## Configuration

Related environment variables (via `Config`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_BATCH_NORMALIZE` | true | Enable batch min-max normalization |
| `RERANKER_MIN_RESULTS` | 0 | Safety net: min results to return (0 = disabled) |
| `SEARCH_SCORE_THRESHOLD` | 0.5 | Min score after normalization |
| `CROSS_ENCODER_SCORE_THRESHOLD` | 0.0 | Min cross-encoder score |
| `ENABLE_RECENCY_BOOST` | true | Enable temporal context in reranking |
| `RECENCY_DECAY_RATE` | 0.01 | Exponential decay rate per hour |

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
- Recency decay calculation (various ages and decay rates)
- Recency decay application (re-sorting, missing timestamps)

## Usage in MemoryManager

The normalization and decay functions are called during the reranking step (Step 4):

```python
# In MemoryManager.search():

# Step 4a: Batch normalization (if enabled)
if self.config.reranker_batch_normalize:
    scored_results = normalize_reranker_scores(scored_results)

# Step 4b: Recency decay (if enabled and timestamps available)
if (self.config.enable_recency_boost and
    self.config.recency_decay_rate > 0 and
    timestamp_map):
    scored_results = apply_recency_decay(
        scored_results,
        timestamp_map,
        self.config.recency_decay_rate,
    )

# Step 4c: Threshold filtering
filtered_results = apply_threshold_with_safety_net(
    scored_results,
    threshold=self.config.search_score_threshold,
    min_results=self.config.reranker_min_results,
)
```

**Note**: Recency decay is applied AFTER normalization and BEFORE threshold filtering. This ensures:
1. Scores are in a consistent [0, 1] range before decay
2. Decay-adjusted scores are used for threshold comparison
3. Results are already sorted by decayed score

## Usage in Rerankers

Both `LLMReranker` and `CrossEncoderReranker` support recency decay directly:

```python
# Rerankers accept timestamp_map parameter
scored = reranker.rerank(query, candidates, timestamp_map=timestamp_map)

# Internally, they call apply_recency_decay() after scoring
```

## Dependencies

- `numpy`: Array operations
- `ccmemories.application.utils.numba_utils`: JIT-compiled normalization


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Dec 21, 2025

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #177 | 4:12 PM | 🔵 | Complete CLAUDE.md Documentation Inventory | ~680 |
| #662 | 4:09 PM | ✅ | CLAUDE.md Files Aligned | ~196 |

### Jan 8, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #664 | 8:48 AM | 🔵 | Temporal-Aware Reranking Implementation | ~178 |
</claude-mem-context>