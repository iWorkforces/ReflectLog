"""Engine factory for search engine initialization.

This module provides the EngineFactory class that encapsulates the creation
and configuration of search engine instances based on application configuration.
It enables testing with mock engines and supports new engine types without
modifying the factory interface.
"""

from typing import Optional
from dataclasses import dataclass

from reflectlog.application.config import Config
from reflectlog.application.utils import StructuredLogger
from reflectlog.infrastructure import (
    CachedEmbeddings,
    CrossEncoderConfig,
    CrossEncoderReranker,
    LangchainQwenEmbeddings,
    LLMReranker,
    LLMRerankerConfig,
    SmartReplacer,
    SmartReplacerConfig,
    TantivyConfig,
    TantivyEngine,
    USearchConfig,
    USearchEngine,
)
from reflectlog.application.memory.fusion import FusionEngine, create_fusion_engine


@dataclass
class EngineFactoryResult:
    """Result of engine factory initialization."""

    semantic_engine: USearchEngine
    tantivy_engine: Optional[TantivyEngine]
    fusion_engine: FusionEngine
    reranker_engine: str
    enable_hybrid_search: bool


class EngineFactory:
    """Factory for creating and configuring search engine instances.

    This factory encapsulates all engine initialization logic, making it
    easy to test with mock engines and add support for new engine types.

    Example:
        factory = EngineFactory()
        result = factory.create_engines(config, logger)
        semantic = result.semantic_engine
        tantivy = result.tantivy_engine
    """

    def __init__(self):
        """Initialize the engine factory."""
        pass

    def create_engines(
        self,
        config: Config,
        logger: StructuredLogger,
    ) -> EngineFactoryResult:
        """Create and configure all search engines based on configuration.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            EngineFactoryResult with all initialized engines.
        """
        # Create USearch semantic engine
        semantic_engine = self._create_semantic_engine(config, logger)

        # Create Tantivy full-text engine (if hybrid search enabled)
        tantivy_engine = self._create_tantivy_engine(config, logger)

        # Create fusion engine for hybrid ranking
        fusion_engine = self._create_fusion_engine(config, logger)

        return EngineFactoryResult(
            semantic_engine=semantic_engine,
            tantivy_engine=tantivy_engine,
            fusion_engine=fusion_engine,
            reranker_engine=config.reranker_engine,
            enable_hybrid_search=config.enable_hybrid_search,
        )

    def _create_semantic_engine(
        self,
        config: Config,
        logger: StructuredLogger,
    ) -> USearchEngine:
        """Create and configure USearch semantic engine.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            Configured USearchEngine instance.
        """
        usearch_config = USearchConfig.from_app_config(config)
        embedder = self._create_embedder(config, logger)
        return USearchEngine(usearch_config, embedder=embedder, logger=logger)

    def _create_embedder(
        self,
        config: Config,
        logger: StructuredLogger,
    ) -> LangchainQwenEmbeddings | CachedEmbeddings:
        """Create embedder with optional caching.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            Embedder instance (possibly wrapped with caching).
        """
        base_embedder = LangchainQwenEmbeddings(
            config={
                "model": config.embedding_model,
                "embedding_dims": config.qwen_embedding_dims
                if config.embedder_provider == "langchain"
                else config.embedding_dims,
                "api_key": config.openrouter_api_key.get_secret_value(),
                "openai_base_url": config.openrouter_base_url,
                "batch_size": config.embedding_batch_size,
                "max_concurrent_batches": config.embedding_max_concurrent_batches,
            }
        )

        if config.embedding_cache_enabled:
            return CachedEmbeddings(
                embedder=base_embedder,
                cache_size=config.embedding_cache_size,
                enabled=True,
                logger=logger,
            )
        return base_embedder

    def _create_tantivy_engine(
        self,
        config: Config,
        logger: StructuredLogger,
    ) -> Optional[TantivyEngine]:
        """Create Tantivy full-text engine if hybrid search is enabled.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            TantivyEngine instance or None if hybrid search is disabled.
        """
        if not config.enable_hybrid_search:
            return None

        tantivy_config = TantivyConfig(
            project_id=config.project_id,
            index_path=config.tantivy_index_path_template.format(
                project_id=config.project_id
            ).lower(),
            normalize_scores=config.tantivy_normalize_scores,
        )
        return TantivyEngine(tantivy_config, logger=logger)

    def _create_fusion_engine(
        self,
        config: Config,
        logger: StructuredLogger,
    ) -> FusionEngine:
        """Create fusion engine for hybrid ranking.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            Configured FusionEngine instance.
        """
        return create_fusion_engine(
            method=config.fusion_method,
            normalization=config.fusion_normalization,
            rrf_k=config.fusion_rrf_k,
            logger=logger,
        )


def create_llm_reranker(
    config: Config,
    logger: StructuredLogger,
) -> Optional[LLMReranker]:
    """Create LLM reranker if configured.

    Args:
        config: Application configuration.
        logger: Structured logger instance.

    Returns:
        LLMReranker instance or None if not configured.
    """
    if config.reranker_engine != "llm":
        return None

    reranker_config = LLMRerankerConfig.from_app_config(config)
    return LLMReranker(config=reranker_config, logger=logger)


def create_cross_encoder_reranker(
    config: Config,
    logger: StructuredLogger,
) -> Optional[CrossEncoderReranker]:
    """Create CrossEncoder reranker if configured.

    Args:
        config: Application configuration.
        logger: Structured logger instance.

    Returns:
        CrossEncoderReranker instance or None if not configured.
    """
    if config.reranker_engine != "cross_encoder":
        return None

    ce_config = CrossEncoderConfig.from_app_config(config)
    return CrossEncoderReranker(config=ce_config, logger=logger)


def create_smart_replacer(
    config: Config,
    logger: StructuredLogger,
) -> Optional[SmartReplacer]:
    """Create SmartReplacer if enabled.

    Args:
        config: Application configuration.
        logger: Structured logger instance.

    Returns:
        SmartReplacer instance or None if disabled.
    """
    if not config.enable_smart_replace:
        return None

    replacer_config = SmartReplacerConfig.from_app_config(config)
    return SmartReplacer(config=replacer_config, logger=logger)
