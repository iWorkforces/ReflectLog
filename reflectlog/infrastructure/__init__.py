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
- MemoryStore: SQLite-backed memory storage for USearch

This module also re-exports from subpackages:
- infrastructure.search: Search engine implementations
- infrastructure.embeddings: Embedding provider implementations
- infrastructure.reranking: Reranker implementations
- infrastructure.memory: Memory storage implementations
- infrastructure.llm: LLM provider implementations
"""

# Re-export from subpackages for convenient imports
from . import embeddings, llm, memory, reranking, search
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
from .memory_store import MemoryRecord, MemoryStore
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
    "AnthropicReplacementProvider",
    "AnthropicRerankerProvider",
    "BaseOpenAIProvider",
    "CachedEmbeddings",
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "IReplacementProvider",
    "IRerankerProvider",
    "IStructuredOutputSchema",
    "LLMReranker",
    "LLMRerankerConfig",
    "LangchainQwenEmbeddings",
    "MemoryRecord",
    "MemoryStore",
    "OpenAIReplacementProvider",
    "OpenAIRerankerProvider",
    "RelevanceScore",
    "ReplacementDecision",
    "SmartReplacer",
    "SmartReplacerConfig",
    "TantivyConfig",
    "TantivyEngine",
    "USearchConfig",
    "USearchEngine",
    "create_replacement_provider",
    "create_reranker_provider",
    "embeddings",
    "format_memory_age",
    "llm",
    "memory",
    "reranking",
    "search",
]
