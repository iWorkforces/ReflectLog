"""Infrastructure layer for external library wrappers.

This module provides:
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
from .llm_reranker import LLMReranker, LLMRerankerConfig, RelevanceScore
from .message_store import MessageRecord, MessageStore
from .qwen3_embedding import LangchainQwenEmbeddings
from .smart_replacer import ReplacementDecision, SmartReplacer, SmartReplacerConfig
from .tantivy_engine import TantivyConfig, TantivyEngine
from .usearch_engine import USearchConfig, USearchEngine

__all__ = [
    "CachedEmbeddings",
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "LangchainQwenEmbeddings",
    "LLMReranker",
    "LLMRerankerConfig",
    "MessageRecord",
    "MessageStore",
    "RelevanceScore",
    "ReplacementDecision",
    "SmartReplacer",
    "SmartReplacerConfig",
    "TantivyConfig",
    "TantivyEngine",
    "USearchConfig",
    "USearchEngine",
]
