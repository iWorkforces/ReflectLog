# ReflectLog Knowledge Base - Core Protocols

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW

Protocol definitions layer defining all abstractions for protocol-based dependency injection throughout the system.

## STRUCTURE

```
core/
├── config.py            # Configuration protocols (IServerConfig, ISearchConfig, IStorageConfig, IRerankerConfig, IEmbedderConfig, IReplacementConfig, IAppConfig)
├── config_adapters.py   # Protocol adapters wrapping Config dataclass (ConfigAdapter + 6 fine-grained adapters)
├── memory.py            # Memory protocols (IMemoryStore, IMemoryBackend, IMemoryManager)
├── reranking.py         # Reranking protocols (IReranker, IRerankerProvider, IRankingResult, IRerankerConfig)
├── tools.py             # Tool protocols (ITool, IToolRegistry, IToolLoader, ToolParameter, ToolDefinition)
├── logging.py           # Logging protocols (ILoggingService, ILogSink, LogLevel)
└── types.py             # Canonical types (ISemanticSearchEngine, MemoryRecord, Embeddings, IArchiveMemoryStore)
```

## WHERE TO LOOK

| Task | File | Key Protocols |
|------|------|---------------|
| Server configuration | config.py | IServerConfig, IAppConfig |
| Live search | application/memory/search_strategies.py | SearchPipeline, SearchContext, SearchResult |
| Semantic engine | types.py | ISemanticSearchEngine (sync tuple search) |
| Fusion | application/memory/fusion/base.py | FusionEngine |
| Memory operations | memory.py | IMemoryStore, IMemoryManager |
| Reranking interfaces | reranking.py | IReranker, IRerankerProvider |
| Config adaptation | config_adapters.py | ConfigAdapter, SearchConfigAdapter, etc. |
| Tool registration | tools.py | ITool, IToolRegistry |
| Logging abstraction | logging.py | ILoggingService, ILogSink |
| Canonical types | types.py | ISemanticSearchEngine, MemoryRecord, Embeddings, IArchiveMemoryStore |

## CONVENTIONS

**Protocol-Based DI** - Components depend on protocols from `core/`, not concrete implementations. Enables runtime substitution and test mocking via `@runtime_checkable`.

**Structural Typing** - Use protocols (not ABCs) for duck typing. Runtime verification enabled with `@runtime_checkable`.

**Adapter Pattern** - `ConfigAdapter` wraps `Config` dataclass to satisfy `IAppConfig`. Fine-grained adapters (`SearchConfigAdapter`, `StorageConfigAdapter`, etc.) expose subset interfaces.

**Protocol Composition** - `IAppConfig` combines 6 sub-protocols: `IServerConfig`, `ISearchConfig`, `IStorageConfig`, `IRerankerConfig`, `IEmbedderConfig`, `IReplacementConfig`.

**Factory Functions** - Adapter creation via `create_*_adapter()` factory functions.

## ANTI-PATTERNS

- Never depend on concrete `Config` class in application layer - use protocols
- Never create direct instances of protocols - use concrete implementations
- Never mix `Protocol` and `ABC` - choose one pattern per abstraction
- Never skip `@runtime_checkable` if runtime isinstance() checks needed
- Never expose implementation details through protocol interfaces
