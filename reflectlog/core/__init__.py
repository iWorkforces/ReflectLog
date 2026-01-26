"""ReflectLogMCP Core Package.

This package contains protocol definitions and abstractions that define the
interface contracts for the application layer. All components depend on
these protocols rather than concrete implementations, enabling:

- Runtime component substitution
- Compile-time type checking
- Dependency injection
- Testability through mock implementations

Modules:
    config: Configuration protocols
    memory: Memory operation protocols
    search: Search engine protocols
    reranking: Reranker protocols
    tools: Tool registration protocols
    logging: Logging protocols
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Re-export all protocols for type checkers
    from .config import (
        IAppConfig,
        IEmbedderConfig,
        IReplacementConfig,
        IRerankerConfig,
        ISearchConfig,
        IServerConfig,
        IStorageConfig,
    )
    from .config_adapters import (
        ConfigAdapter,
        EmbedderConfigAdapter,
        ReplacementConfigAdapter,
        RerankerConfigAdapter,
        SearchConfigAdapter,
        ServerConfigAdapter,
        StorageConfigAdapter,
        create_config_adapter,
        create_embedder_config_adapter,
        create_replacement_config_adapter,
        create_reranker_config_adapter,
        create_search_config_adapter,
        create_server_config_adapter,
        create_storage_config_adapter,
    )
    from .logging import (
        ILoggingService,
        ILogSink,
        LogLevel,
    )
    from .memory import (
        IMemoryBackend,
        IMemoryManager,
        IMemoryStore,
    )
    from .reranking import (
        IRankingResult,
        IReranker,
        IRerankerProvider,
    )
    from .search import (
        IFusionAlgorithm,
        ISearchBackend,
        ISearchResult,
    )
    from .tools import (
        ITool,
        IToolRegistry,
        IToolResult,
    )

__all__ = [
    # Configuration protocols
    "IAppConfig",
    "IServerConfig",
    "ISearchConfig",
    "IStorageConfig",
    "IRerankerConfig",
    "IEmbedderConfig",
    "IReplacementConfig",
    # Configuration adapters
    "ConfigAdapter",
    "ServerConfigAdapter",
    "SearchConfigAdapter",
    "StorageConfigAdapter",
    "RerankerConfigAdapter",
    "EmbedderConfigAdapter",
    "ReplacementConfigAdapter",
    "create_config_adapter",
    "create_server_config_adapter",
    "create_search_config_adapter",
    "create_storage_config_adapter",
    "create_reranker_config_adapter",
    "create_embedder_config_adapter",
    "create_replacement_config_adapter",
    # Memory protocols
    "IMemoryStore",
    "IMemoryBackend",
    "IMemoryManager",
    # Search protocols
    "ISearchBackend",
    "ISearchResult",
    "IFusionAlgorithm",
    # Reranking protocols
    "IReranker",
    "IRerankerProvider",
    "IRankingResult",
    # Tool protocols
    "ITool",
    "IToolRegistry",
    "IToolResult",
    # Logging protocols
    "ILoggingService",
    "ILogSink",
    "LogLevel",
]
