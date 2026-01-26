# Agent Guidelines for reflectlog/infrastructure/reranking/

This directory is a placeholder for reranking implementations.

## Directory Structure

```
reranking/
└── __init__.py            # Placeholder - re-exports from parent
```

## Current Status

This directory is currently empty. Reranking implementations are located in the parent `infrastructure/` directory and re-exported here for backward compatibility when this directory is populated in the future.

## Re-Exports

The parent `infrastructure/` directory provides:
- `LLMReranker`: LLM-based relevance scoring
- `CrossEncoderReranker`: Local cross-encoder reranking

## Future Expansion

When implementing new reranking algorithms, add them to this directory:
1. Create algorithm-specific files (e.g., `mmr.py`, `diverse_rerank.py`)
2. Implement `IReranker` protocol from `core/reranking.py`
3. Re-export in `__init__.py`
4. Update this documentation

## Key Patterns

### Reranker Protocol

All rerankers must implement `IReranker` protocol:

```python
@runtime_checkable
class IReranker(Protocol):
    """Protocol for reranking interface."""

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank candidate documents by relevance."""
        ...
```

### Temporal-Aware Scoring

Consider recency when scoring for handling contradictory memories:

```python
def _apply_temporal_decay(self, scores: list[float], timestamps: list[str]) -> list[float]:
    """Apply recency decay to scores."""
    now = datetime.now(timezone.utc)
    adjusted = []

    for score, ts in zip(scores, timestamps):
        age_hours = (now - datetime.fromisoformat(ts)).total_seconds() / 3600
        decay_factor = math.exp(-0.01 * age_hours)  # Decay over time
        adjusted.append(score * decay_factor)

    return adjusted
```

## Dependencies

### Internal Dependencies

- `core/reranking.py`: `IReranker` protocol
- `application/config/`: Configuration dataclasses
- `application/exceptions.py`: `RerankingError` exception

### External Dependencies

- `sentence-transformers`: Cross-encoder models
- `openai`: OpenAI API (for LLM reranking)
- `anthropic`: Anthropic API (for LLM reranking)
- `numpy`: Numerical operations
