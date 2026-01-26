# Agent Guidelines for reflectlog/infrastructure/llm/

This directory is a placeholder for LLM provider implementations.

## Directory Structure

```
llm/
└── __init__.py            # Placeholder - re-exports from parent
```

## Current Status

This directory is currently empty. LLM provider implementations are located in the parent `infrastructure/` directory and re-exported here for backward compatibility when this directory is populated in the future.

## Re-Exports

The parent `infrastructure/` directory provides:
- `LLMProvider`: Base class for LLM providers
- `IRerankerProvider`: Protocol for reranking providers
- `OpenRouterLLM`: OpenRouter API wrapper

## Future Expansion

When implementing new LLM providers, add them to this directory:
1. Create provider-specific files (e.g., `openai.py`, `anthropic.py`, `google.py`)
2. Implement `IRerankerProvider` protocol from `core/reranking.py`
3. Re-export in `__init__.py`
4. Update this documentation

## Key Patterns

### Provider Abstraction

All LLM providers must implement `IRerankerProvider` protocol:

```python
@runtime_checkable
class IRerankerProvider(Protocol):
    """Protocol for LLM-based reranking providers."""

    async def score(
        self,
        query: str,
        document: str,
        prompt: str,
    ) -> float:
        """Score a single document's relevance to query."""
        ...

    async def score_batch(
        self,
        query: str,
        documents: list[str],
        prompt: str,
    ) -> list[float]:
        """Score multiple documents efficiently."""
        ...
```

### Error Handling

Wrap API errors with domain-specific exceptions:

```python
try:
    response = await self._client.chat.completions.create(...)
except openai.APIError as e:
    raise LLMError(f"OpenRouter API error: {e}") from e
```

## Dependencies

### Internal Dependencies

- `core/reranking.py`: `IRerankerProvider` protocol
- `application/config/`: Configuration dataclasses
- `application/exceptions.py`: `LLMError` exception

### External Dependencies

- `openai`: OpenAI API client
- `anthropic`: Anthropic API client
- `httpx`: Async HTTP client
