"""Reranker implementations for ReflectLogMCP.

This subpackage contains implementations of relevance scoring rerankers.
Each reranker implements the IReranker protocol from core.reranking.

Note: The actual implementations are in the parent infrastructure/
directory. This module re-exports them for backward compatibility.
"""

# Re-export from parent module for backward compatibility
from reflectlog.infrastructure.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
)
from reflectlog.infrastructure.llm_reranker import (
    AnthropicRerankerProvider,
    IRerankerProvider,
    LLMReranker,
    LLMRerankerConfig,
    OpenAIRerankerProvider,
    RelevanceScore,
    create_reranker_provider,
    format_memory_age,
)

__all__ = [
    "LLMReranker",
    "LLMRerankerConfig",
    "IRerankerProvider",
    "OpenAIRerankerProvider",
    "AnthropicRerankerProvider",
    "RelevanceScore",
    "create_reranker_provider",
    "format_memory_age",
    "CrossEncoderReranker",
    "CrossEncoderConfig",
]
