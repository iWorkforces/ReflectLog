"""Infrastructure layer for external library wrappers.

This module provides:
- BaseOpenAIProvider: Base class for OpenAI-compatible LLM providers
- CachedEmbeddings: LRU caching wrapper for query embedding caching
- LangchainQwenEmbeddings: OpenRouter embedding provider with async support
- LLMReranker: LLM-based document reranker for search results
- CrossEncoderReranker: FlagReranker-based local reranking (BGE models)
- SmartReplacer: LLM-based smart memory replacement detector
- TantivyEngine: Full-text search engine wrapper
- USearchEngine: USearch-based semantic search engine
- MessageStore: SQLite-backed message storage for USearch
"""

from .cached_embeddings import CachedEmbeddings
from .cross_encoder_reranker import CrossEncoderConfig, CrossEncoderReranker
from .llm_provider_base import BaseOpenAIProvider, IStructuredOutputSchema
from .llm_reranker import (
    AnthropicRerankerProvider,
    IRerankerProvider,
    LLMReranker,
    LLMRerankerConfig,
    OpenAIRerankerProvider,
    RelevanceScore,
    create_reranker_provider,
    format_memory_age,
)
from .message_store import MessageRecord, MessageStore
from .qwen3_embedding import LangchainQwenEmbeddings
from .smart_replacer import (
    AnthropicReplacementProvider,
    IReplacementProvider,
    OpenAIReplacementProvider,
    ReplacementDecision,
    SmartReplacer,
    SmartReplacerConfig,
    create_replacement_provider,
)
from .tantivy_engine import TantivyConfig, TantivyEngine
from .usearch_engine import USearchConfig, USearchEngine

__all__ = [
    "AnthropicRerankerProvider",
    "AnthropicReplacementProvider",
    "BaseOpenAIProvider",
    "CachedEmbeddings",
    "IStructuredOutputSchema",
    "create_reranker_provider",
    "create_replacement_provider",
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "format_memory_age",
    "IRerankerProvider",
    "IReplacementProvider",
    "LangchainQwenEmbeddings",
    "LLMReranker",
    "LLMRerankerConfig",
    "MessageRecord",
    "MessageStore",
    "OpenAIRerankerProvider",
    "OpenAIReplacementProvider",
    "RelevanceScore",
    "ReplacementDecision",
    "SmartReplacer",
    "SmartReplacerConfig",
    "TantivyConfig",
    "TantivyEngine",
    "USearchConfig",
    "USearchEngine",
]
