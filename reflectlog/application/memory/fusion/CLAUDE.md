# reflectlog/application/memory/fusion/

This directory contains the fusion engine implementations for hybrid search ranking.

## Structure

```
fusion/
├── __init__.py          # Package exports and factory function
├── base.py              # FusionEngine protocol
└── ranx_fusion.py       # RanxFusionEngine implementation
```

## Purpose

The `fusion/` module implements **Reciprocal Rank Fusion (RRF)** and other fusion algorithms to combine results from multiple search engines (USearch semantic + Tantivy full-text) into a single ranked list.

Key capabilities:
- **Multiple fusion algorithms**: RRF, CombSUM, CombMNZ, CombMAX, BordaFuse
- **Score normalization**: min-max, max, sum, zmuv, rank, borda
- **Output normalization**: All fusion scores normalized to 0-1 range
- **Protocol-based design**: Enables swappable fusion implementations

## Architecture

### FusionEngine Protocol (`base.py`)

Defines the interface all fusion engines must implement:

```python
class FusionEngine(Protocol):
    @property
    def method(self) -> str:
        """Fusion method name (e.g., 'rrf', 'sum')."""
        ...

    @property
    def normalization(self) -> Optional[str]:
        """Normalization strategy (e.g., 'min-max', 'rank')."""
        ...

    def fuse(
        self,
        *result_sets: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Fuse ranked lists into single ranking."""
        ...
```

### RanxFusionEngine (`ranx_fusion.py`)

Production implementation using the [ranx](https://github.com/AmenRa/ranx) library:

```python
engine = RanxFusionEngine(
    method="rrf",        # Fusion algorithm
    normalization=None,  # Input normalization (auto-selected)
    rrf_k=60,            # RRF constant
    logger=logger,       # Optional structured logger
)

# Fuse semantic and full-text results
fused = engine.fuse(
    [("doc1", 0.9), ("doc2", 0.8)],  # Semantic results
    [("doc2", 0.7), ("doc3", 0.6)],  # Full-text results
)
# Returns: [("doc2", 1.0), ("doc1", 0.67), ("doc3", 0.33)]
```

### Factory Function

```python
from reflectlog.application.memory.fusion import create_fusion_engine

engine = create_fusion_engine(
    method="rrf",
    normalization=None,
    rrf_k=60,
    logger=logger,
)
```

## Supported Algorithms

### Fusion Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `rrf` | Reciprocal Rank Fusion (default) | General hybrid search |
| `sum` | CombSUM - score addition | Similar score ranges |
| `mnz` | CombMNZ - weighted addition | Multi-source voting |
| `max` | CombMAX - maximum score | High-confidence matches |
| `bordafuse` | Borda voting fusion | Rank-based aggregation |

### Normalization Strategies

| Strategy | Description |
|----------|-------------|
| `min-max` | Scale to [0, 1] using min/max |
| `max` | Divide by max score |
| `sum` | Divide by sum of scores |
| `zmuv` | Zero-mean unit-variance |
| `rank` | Rank-based normalization |
| `borda` | Borda count normalization |

**Note**: RRF uses rank positions, not scores, so input normalization is typically not needed.

## RRF Algorithm

Reciprocal Rank Fusion formula:

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```

Where:
- `k` = constant (default: 60)
- `rank_i(doc)` = rank of document in ranking list `i`
- Lower `k` = more weight to top ranks
- Higher `k` = more balanced weighting

**Example**:
```
Semantic: [A@1, B@2, C@3]  →  A: 1/61, B: 1/62, C: 1/63
Fulltext: [B@1, C@2, D@3]  →  B: 1/61, C: 1/62, D: 1/63

Fused scores:
  B: 1/61 + 1/62 = 0.0328  (appears in both, highest)
  A: 1/61 = 0.0164
  C: 1/62 + 1/63 = 0.0320
  D: 1/63 = 0.0159

Normalized to 0-1: B=1.0, C=0.95, A=0.03, D=0.0
```

## Implementation Details

### Single Result Set Handling

When only one result set is non-empty, fusion is skipped and scores are normalized directly:

```python
if len(runs) == 1:
    sorted_results = self._convert_from_run(runs[0])
    return self._normalize_output_scores(sorted_results)
```

### Duplicate Handling

Within a single result set, only the first occurrence is kept:

```python
for msg, score in result_set:
    if msg not in doc_scores:
        doc_scores[msg] = score
```

### Output Normalization

All fusion output is min-max normalized to 0-1:

```python
def _normalize_output_scores(self, results):
    min_score = min(scores)
    max_score = max(scores)
    return [(msg, (score - min_score) / (max_score - min_score))
            for msg, score in results]
```

## Configuration

Via environment variables (in `MemoryManager`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FUSION_RRF_K` | 60 | RRF constant |
| `FUSION_RANKING_THRESHOLD` | 0.8 | Min normalized score to keep |

## Usage in MemoryManager

```python
from reflectlog.application.memory.fusion import create_fusion_engine

class MemoryManager:
    def __init__(self, config, logger):
        self.fusion_engine = create_fusion_engine(
            method="rrf",
            rrf_k=config.fusion_rrf_k,
            logger=logger,
        )

    def _fuse_hybrid_results(self, usearch_results, tantivy_results):
        fused = self.fusion_engine.fuse(usearch_results, tantivy_results)
        # Apply threshold
        return [(msg, score) for msg, score in fused
                if score >= self.config.fusion_ranking_threshold]
```

## Adding a New Fusion Engine

1. Create `reflectlog/application/memory/fusion/new_engine.py`:
   ```python
   class NewFusionEngine:
       def __init__(self, method: str = "custom", logger=None):
           self._method = method
           self.logger = logger

       @property
       def method(self) -> str:
           return self._method

       @property
       def normalization(self) -> Optional[str]:
           return None

       def fuse(self, *result_sets) -> List[Tuple[str, float]]:
           # Custom fusion logic
           ...
   ```

2. Export in `__init__.py`:
   ```python
   from .new_engine import NewFusionEngine
   __all__ = [..., "NewFusionEngine"]
   ```

3. Update factory function if needed

## Testing

### Unit Test Scenarios

```python
def test_rrf_fusion_basic():
    """RRF should combine rankings correctly."""
    engine = RanxFusionEngine(method="rrf", rrf_k=60)
    result = engine.fuse(
        [("A", 0.9), ("B", 0.8)],
        [("B", 0.7), ("C", 0.6)],
    )
    # B appears in both, should rank highest
    assert result[0][0] == "B"

def test_single_result_set():
    """Single result set should normalize without fusion."""
    engine = RanxFusionEngine()
    result = engine.fuse([("A", 0.5), ("B", 0.3)])
    assert result[0] == ("A", 1.0)
    assert result[1] == ("B", 0.0)

def test_empty_result_sets():
    """Empty inputs should return empty list."""
    engine = RanxFusionEngine()
    assert engine.fuse([], []) == []

def test_unsupported_method():
    """Invalid method should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported fusion method"):
        RanxFusionEngine(method="invalid")
```

## Dependencies

- `ranx>=0.3.16`: Fusion algorithm implementations
- Python typing: `Protocol`, `List`, `Tuple`, `Optional`

## Performance Considerations

- **ranx.fuse() overhead**: Minimal for typical result set sizes (<100 items)
- **Conversion cost**: Tuple ↔ Run object conversion is O(n)
- **Memory**: Run objects created per query, garbage collected after
- **Normalization**: O(n) pass over results

## Constants Reference

```python
# ranx_fusion.py

SUPPORTED_METHODS = {"rrf", "sum", "mnz", "max", "bordafuse"}

SUPPORTED_NORMALIZATIONS = {"min-max", "max", "sum", "zmuv", "rank", "borda"}

DEFAULT_NORMALIZATIONS = {
    "rrf": None,        # RRF uses ranks, not scores
    "sum": None,
    "mnz": None,
    "max": None,
    "bordafuse": None,
}
```
