"""Embedding provider implementations for ReflectLogMCP.

This subpackage contains implementations of embedding providers.
Each provider implements the embedding protocol from core.

Note: The actual implementations are in the parent infrastructure/
directory. This module re-exports them for convenience.
"""

# Re-export from parent module for convenience
from reflectlog.infrastructure.cached_embeddings import CachedEmbeddings
from reflectlog.infrastructure.qwen3_embedding import LangchainQwenEmbeddings

__all__ = [
    "CachedEmbeddings",
    "LangchainQwenEmbeddings",
]
