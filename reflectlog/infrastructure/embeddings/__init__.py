"""Embedding provider implementations for ReflectLogMCP.

This subpackage contains implementations of embedding providers.
Each provider implements the embedding protocol from core.

Note: The actual implementations are in the parent infrastructure/
directory. This module re-exports them for backward compatibility.
"""

# Re-export from parent module for backward compatibility
from reflectlog.infrastructure.qwen3_embedding import LangchainQwenEmbeddings
from reflectlog.infrastructure.cached_embeddings import CachedEmbeddings

__all__ = [
    "LangchainQwenEmbeddings",
    "CachedEmbeddings",
]
