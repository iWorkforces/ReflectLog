# Agent Guidelines for reflectlog/application/memory/fusion/

This directory contains fusion engine implementations for hybrid search ranking. It implements Reciprocal Rank Fusion (RRF) and other fusion algorithms to combine results from multiple search engines into a single ranked list.

## Directory Structure

```
fusion/
├── __init__.py          # Package exports and factory function
├── base.py              # FusionEngine protocol
└── ranx_fusion.py       # RanxFusionEngine implementation
```

## Core Responsibilities

### FusionEngine Protocol

The `base.py` module defines the interface all fusion engines must implement:

```python
class FusionEngine(Protocol):
    @property
    def method(self) -> str:
        '''Fusion method name (e.g., 'rrf', 'sum').'''
        ...

    @property
    def normalization(self) -> str | None:
        '''Normalization strategy (e.g., 'min-max', 'rank').'''
        ...

    def fuse(
        self,
        *result_sets: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        '''Fuse ranked lists into single ranking.'''
        ...
```

### RanxFusionEngine

The `ranx_fusion.py` module provides the RRF fusion implementation using the `ranx` library:

- Implements Reciprocal Rank Fusion algorithm
- Supports multiple fusion algorithms (CombSUM, CombMNZ, CombMAX, BordaFuse)
- Provides configurable score normalization
- Normalizes output scores to 0-1 range

## Fusion Algorithms

### Reciprocal Rank Fusion (RRF)

The RRF algorithm combines multiple ranked lists by computing the reciprocal rank of each document:

```
RRF_score(doc) = sum over rankings of: 1 / (k + rank(doc))
```

Where `k` is a constant (default: 60) that prevents division by zero and reduces the impact of low ranks.

**Characteristics:**
- Rank-based rather than score-based
- Robust to score scale differences between engines
- Naturally handles documents appearing in multiple lists
- Configurable `k` parameter for tuning

### CombSUM

CombSUM adds normalized scores from each ranking:

```
CombSUM_score(doc) = sum over rankings of: normalized_score(doc)
```

**Characteristics:**
- Score-based approach
- Requires score normalization before fusion
- Favors documents with high scores across multiple engines

### CombMNZ

CombMNZ (Combination with Number of Zeros) multiplies the CombSUM score by the number of engines that returned the document:

```
CombMNZ_score(doc) = CombSUM_score(doc) * count(engines_returning_doc)
```

**Characteristics:**
- Favors documents appearing in multiple rankings
- Useful when high overlap indicates relevance

### BordaFuse

BordaFuse converts rankings to points based on position:

```
BordaFuse_score(doc) = sum over rankings of: (max_rank - rank(doc))
```

**Characteristics:**
- Rank-based transformation
- Similar to voting systems
- Less sensitive to score distributions

## Score Normalization

### Min-Max Normalization

Transforms scores to 0-1 range:

```
normalized = (score - min) / (max - min)
```

### Max Normalization

Normalizes by maximum score:

```
normalized = score / max
```

### Rank Normalization

Converts scores to ranks:

```
ranked[i] = position of score[i] in sorted order
```

### Z-Score Normalization

Standardizes scores to zero mean and unit variance:

```
normalized = (score - mean) / std
```

## Key Patterns

### Protocol-Based Design

Implement the FusionEngine protocol for swappable fusion strategies:

```python
class RanxFusionEngine:
    def __init__(
        self,
        method: str = "rrf",
        normalization: str = "min-max",
        k: int = 60,
    ):
        self._method = method
        self._normalization = normalization
        self._k = k

    @property
    def method(self) -> str:
        return self._method

    @property
    def normalization(self) -> str | None:
        return self._normalization

    def fuse(
        self,
        *result_sets: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        # Convert to ranx format
        runs = [convert_to_ranx(results) for results in result_sets]

        # Create fusion model
        model = ranx.FusionModel(runs)
        model.fuse(method=self._method, norm=self._normalization)

        # Extract results
        fused = model.get_fused_scores()
        return [(doc, score) for doc, score in fused]
```

### Factory Function

Provide a factory function for creating fusion engines:

```python
def create_fusion_engine(
    method: str = "rrf",
    normalization: str = "min-max",
    k: int = 60,
) -> FusionEngine:
    '''Create a fusion engine with specified parameters.'''
    return RanxFusionEngine(method=method, normalization=normalization, k=k)
```

## Usage Examples

### Basic RRF Fusion

```python
from reflectlog.application.memory.fusion import create_fusion_engine

# Create RRF fusion engine
fusion_engine = create_fusion_engine(method="rrf", k=60)

# Fuse results from two search engines
usearch_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
tantivy_results = [("doc2", 0.85), ("doc4", 0.75), ("doc1", 0.7)]

# Fuse ranked lists
fused = fusion_engine.fuse(usearch_results, tantivy_results)
# Returns: [("doc1", 0.92), ("doc2", 0.90), ("doc3", 0.45), ("doc4", 0.43)]
```

### Custom Fusion Method

```python
# Create CombSUM fusion engine with max normalization
fusion_engine = create_fusion_engine(method="sum", normalization="max")

# Fuse with custom parameters
fused = fusion_engine.fuse(results1, results2, results3)
```

## Configuration

### Fusion Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FUSION_METHOD` | rrf | Fusion algorithm (rrf, sum, mnz, max, borda) |
| `FUSION_NORMALIZATION` | min-max | Score normalization strategy |
| `FUSION_RRF_K` | 60 | RRF constant for reciprocal ranking |
| `FUSION_RANKING_THRESHOLD` | 0.8 | Minimum score to keep after fusion |

### Method Selection Guide

| Method | Best For | Requires Normalization |
|--------|----------|------------------------|
| RRF | General purpose, robust | No |
| CombSUM | Score-based fusion | Yes |
| CombMNZ | High overlap scenarios | Yes |
| CombMAX | Conservative fusion | Yes |
| BordaFuse | Rank-based voting | No |

## Performance Considerations

### Algorithm Complexity

| Method | Complexity | Notes |
|--------|------------|-------|
| RRF | O(n log n) | Sorting dominates |
| CombSUM | O(n) | Linear combination |
| CombMNZ | O(n) | Linear with count |
| BordaFuse | O(n log n) | Ranking transformation |

### Optimization Strategies

- **Pre-normalization**: Normalize scores before fusion to improve numerical stability
- **Deduplication**: Remove duplicate documents before fusion to avoid double-counting
- **Filtering**: Apply threshold filters after fusion to reduce downstream processing

## Error Handling

### Invalid Input

```python
def fuse(
    self,
    *result_sets: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    # Validate input
    if not result_sets:
        raise ValueError("At least one result set required")

    for i, results in enumerate(result_sets):
        if not results:
            continue
        if not all(isinstance(r, tuple) and len(r) == 2 for r in results):
            raise ValueError(f"Invalid result format in set {i}")

    # Proceed with fusion
    ...
```

### Empty Results

Handle empty result sets gracefully:

```python
def fuse(
    self,
    *result_sets: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    # Filter out empty result sets
    valid_sets = [s for s in result_sets if s]

    if not valid_sets:
        return []

    # Fuse only valid sets
    ...
```

## Testing Guidelines

### Unit Tests

- Test fusion with known expected outputs
- Verify RRF ranking correctness
- Test edge cases (empty inputs, single document)
- Validate score normalization

### Test Cases

```python
def test_rrf_fusion():
    '''RRF should rank documents appearing in multiple lists higher.'''
    usearch = [("A", 0.9), ("B", 0.8)]
    tantivy = [("B", 0.7), ("C", 0.6)]

    fused = fusion_engine.fuse(usearch, tantivy)

    # B appears in both, should rank highest
    assert fused[0][0] == "B"

def test_empty_results():
    '''Empty results should return empty list.'''
    result = fusion_engine.fuse([])
    assert result == []

def test_single_list():
    '''Single result list should be returned as-is (normalized).'''
    results = [("A", 0.9), ("B", 0.8)]
    fused = fusion_engine.fuse(results)
    assert len(fused) == 2
```

## Dependencies

### Internal Dependencies

- `application/memory/protocols.py`: Search result types
- `application/utils/`: Logging utilities

### External Dependencies

- `ranx`: RRF fusion and fusion algorithms library
- `numpy`: Numerical operations (if needed)

## Important Notes

### Score Interpretation

- Fusion scores are normalized to 0-1 range
- Higher scores indicate better overall relevance
- Scores from different fusion methods are not directly comparable

### Document Identity

- Documents are identified by their content string
- Duplicate documents across result sets are automatically merged
- Final ranking reflects combined relevance from all sources

### Parameter Tuning

- Start with default RRF parameters (k=60)
- Adjust `k` based on ranking sensitivity needs
- Lower `k` gives more weight to top ranks
- Higher `k` provides more balanced fusion
