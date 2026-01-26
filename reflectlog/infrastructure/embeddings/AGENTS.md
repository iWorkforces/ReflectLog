# Agent Guidelines for reflectlog/infrastructure/embeddings/

This directory is a placeholder for embedding provider implementations.

## Directory Structure

```
embeddings/
└── __init__.py            # Placeholder - re-exports from parent
```

## Current Status

This directory is currently empty. Embedding implementations are located in the parent `infrastructure/` directory and re-exported here for backward compatibility when this directory is populated in the future.

## Re-Exports

The parent `infrastructure/` directory provides:
- `LangchainQwenEmbeddings`: Qwen3 embedding implementation
- `CachedEmbeddings`: LRU cache for query embeddings

## Future Expansion

When implementing new embedding providers, add them to this directory:
1. Create provider-specific files (e.g., `openai.py`, `anthropic.py`)
2. Implement `IEmbedder` protocol from `core/reranking.py`
3. Re-export in `__init__.py`
4. Update this documentation

## Dependencies

### Internal Dependencies

- `core/reranking.py`: `IEmbedder` protocol
- `application/config/`: Configuration dataclasses

### External Dependencies

- `langchain`: Embeddings interface
- `openai`: OpenAI embeddings (when implemented)
- `anthropic`: Anthropic embeddings (when implemented)
