# ccmemories/

This directory contains the main source code for the CCMemoriesMCP server.

## Structure

```
ccmemories/
├── __init__.py           # Package metadata (__version__) and exception exports
├── server.py             # CLI entry point and argument parsing
├── application/          # Application layer (business logic)
│   ├── __init__.py       # Application exports
│   ├── exceptions.py     # Custom exception hierarchy (CCMemoriesError, etc.)
│   ├── mcp_server.py     # FastMCPServer orchestrator
│   ├── types.py          # Type definitions (ISemanticSearchEngine protocol)
│   ├── config/           # Configuration management
│   │   ├── settings.py   # Config dataclass from environment
│   │   └── prompts.py    # MCP_INSTRUCTIONS, SCORING_PROMPT, SCORING_PROMPT_WITH_AGE, REPLACEMENT_DETECTION_PROMPT
│   ├── memory/           # Memory management
│   │   ├── manager.py    # MemoryManager (USearch + Tantivy)
│   │   ├── protocols.py  # Search engine protocols
│   │   ├── fusion/       # Hybrid ranking
│   │   │   ├── base.py   # FusionEngine protocol
│   │   │   └── ranx_fusion.py # RanxFusionEngine (RRF)
│   │   └── reranking/    # Score normalization utilities
│   │       └── normalization.py # Min-max batch normalization
│   ├── tools/            # Modular MCP tool implementations
│   │   ├── base.py       # BaseTool abstract class
│   │   ├── add.py        # AddTool
│   │   ├── get_all.py    # GetAllTool
│   │   ├── search.py     # SearchTool
│   │   └── remove.py     # RemoveTool
│   └── utils/            # Utilities
│       ├── logging.py    # StructuredLogger, format_fusion_score_status
│       ├── numba_utils.py # Numba JIT functions (normalization, distance)
│       ├── retry.py      # async_retry_with_backoff decorator
│       ├── security.py   # SecretString, redact_dict_secrets
│       └── validation.py # Message validation helpers
├── infrastructure/       # External integrations
│   ├── cached_embeddings.py   # CachedEmbeddings (LRU query cache)
│   ├── cross_encoder_reranker.py # CrossEncoderReranker (local FlagReranker)
│   ├── llm_provider_base.py   # BaseOpenAIProvider (shared LLM provider base class)
│   ├── llm_reranker.py        # LLMReranker (AI relevance scoring)
│   ├── message_store.py       # MessageStore (libSQL for USearch)
│   ├── qwen3_embedding.py     # LangchainQwenEmbeddings
│   ├── smart_replacer.py      # SmartReplacer (LLM memory replacement)
│   ├── tantivy_engine.py      # TantivyEngine (full-text search)
│   └── usearch_engine.py      # USearchEngine (semantic search)
└── utility/              # Cross-platform credential retrieval
    ├── __init__.py       # Package exports
    ├── types.py          # Token prefix constants and types
    ├── utility.py        # Core credential retrieval functions
    └── platforms/        # Platform-specific implementations
        ├── base.py       # Abstract CredentialRetriever
        ├── darwin.py     # macOS Keychain retrieval
        ├── linux.py      # Linux config/secret-tool
        └── windows.py    # Windows Credential Manager
```

## Purpose

This is the top-level package that ties together:
1. **CLI interface** (`server.py`) - Handles command-line arguments, environment setup, and stdio/stderr routing
2. **Application logic** (`application/`) - Contains the FastMCPServer, MemoryManager, and modular MCP tools
3. **Infrastructure** (`infrastructure/`) - External service integrations (embedding providers, rerankers, search engines)
4. **Utility** (`utility/`) - Cross-platform Anthropic API key retrieval for Claude Code credentials

## Entry Point Flow

When `ccmemories` command is run:
1. `server.py::main()` is invoked (defined as entry point in `pyproject.toml`)
2. CLI arguments are parsed with `argparse`
3. Environment variables are set based on arguments (priority: CLI args > env vars > defaults)
4. Output stream is selected (stderr for stdio mode, stdout otherwise)
5. `FastMCPServer` is instantiated (loads config, initializes MemoryManager, registers tools)
6. `server.run()` starts the MCP server with configured transport

## Key Files

### `__init__.py`
- Exports `main` function and `__version__`
- **Exports exception hierarchy** for structured error handling:
  - `CCMemoriesError` (base exception)
  - `ConfigurationError`, `ValidationError`, `InitializationError`
  - `StorageError`, `DuplicateError`, `InconsistentStateError`
  - `SearchError`, `EmbeddingError`, `RerankerError`
- Enables both CLI usage (`ccmemories`) and programmatic usage (`from ccmemories import main`)

### `server.py`
- **Responsibility**: CLI argument parsing and environment configuration
- **Critical detail**: Lines 110-111 handle stdio transport requirement that ALL output must go to stderr (not stdout) to avoid corrupting JSON-RPC protocol
- **Transport modes**: stdio (default), http, sse, streamable-http
- **Configuration priority**: CLI args > env vars > defaults

**CLI Arguments**:
```
--version           Show version and exit
--transport         Transport protocol (stdio, http, sse, streamable-http)
--port              Server port for non-stdio transports (default: 9103)
--host              Server host (default: 127.0.0.1)
--path              Server path (default: /mcp)
```

## Module Responsibilities

### application/
- **exceptions.py**: Structured exception hierarchy (`CCMemoriesError` and subclasses)
- **mcp_server.py**: `FastMCPServer` class that orchestrates initialization
- **types.py**: Core type definitions and `ISemanticSearchEngine` protocol
- **config/**: Centralized configuration from environment variables
- **memory/**: `MemoryManager` with hybrid USearch + Tantivy engines
- **memory/fusion/**: `RanxFusionEngine` for RRF hybrid ranking
- **memory/reranking/**: Score normalization and recency decay utilities for rerankers
- **tools/**: Modular tool implementations following `BaseTool` pattern
- **utils/**: Logging, validation, security, and retry utilities

### infrastructure/
- **usearch_engine.py**: `USearchEngine` class for semantic vector search (HNSW)
- **tantivy_engine.py**: `TantivyEngine` class for full-text search with soft-delete
- **message_store.py**: `MessageStore` libSQL storage for message text
- **qwen3_embedding.py**: `LangchainQwenEmbeddings` class for custom embeddings
- **cached_embeddings.py**: LRU caching wrapper for query embeddings
- **llm_provider_base.py**: `BaseOpenAIProvider` base class for OpenAI-compatible LLM providers with structured output
- **llm_reranker.py**: `LLMReranker` for AI-powered relevance scoring (with provider abstraction via `IRerankerProvider`)
- **cross_encoder_reranker.py**: `CrossEncoderReranker` for local FlagReranker-based scoring (with recency decay support)
- **smart_replacer.py**: `SmartReplacer` for LLM-based memory replacement detection (with retry logic)
- Supports both sync and async operations
- HTTP/2 enabled via `DefaultAioHttpClient`
- Concurrency control with `anyio.Semaphore`

### utility/
- **utility.py**: Core credential retrieval functions (`get_anthropic_api_key`, etc.)
- **types.py**: Token prefix constants (`TOKEN_PREFIX`, `OAUTH_TOKEN_PREFIX`)
- **platforms/**: Platform-specific credential retrievers (macOS, Windows, Linux)

## Adding New Modules

When adding new modules to this package:
1. Create them at the appropriate level:
   - Business logic: `application/`
   - External integrations: `infrastructure/`
2. Export public APIs through `__init__.py` if needed
3. Keep domain logic separate from infrastructure concerns
4. Follow the existing pattern: CLI/config at top level, business logic in subdirectories

## Dependencies

This directory directly imports:
- Standard library: `sys`, `os`, `argparse`
- Internal: `ccmemories.__version__`, `application.mcp_server.FastMCPServer`

## Conventions

- **Imports**: Absolute imports preferred (`from ccmemories.application import ...`)
- **Path manipulation**: `sys.path.insert()` used in `server.py:7` for direct script execution
- **Logging**: Use stderr for stdio mode (see `server.py:110-111`)
- **Error messages**: Include help text with examples (see `server.py:23-40`)

## Environment Variables

Key variables used by the package:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROJECT_ID` | Yes | - | Unique project identifier |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `MCP_TRANSPORT` | No | `stdio` | Transport protocol |
| `MCP_PORT` | No | `9103` | Server port |
| `MCP_HOST` | No | `127.0.0.1` | Server host |
| `MCP_PATH` | No | `/mcp` | Server path |
| `ENABLE_RRF_FUSION` | No | `true` | Enable RRF fusion for hybrid search |
| `RERANKER_ENGINE` | No | `llm` | Reranking engine: `llm`, `cross_encoder`, `none` |
| `LLM_PROVIDER` | No | `anthropic` | LLM provider: `openai` or `anthropic` |
| `ENABLE_RECENCY_BOOST` | No | `true` | Include memory age in reranking context |
| `RECENCY_DECAY_RATE` | No | `0.01` | Exponential decay rate per hour |
| `LOG_LEVEL` | No | `INFO` | Logging level |

See `application/config/settings.py` for the complete configuration reference.


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Dec 26, 2025

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #458 | 11:27 AM | ✅ | Enhanced Module Documentation with New Patterns | ~226 |
| #457 | " | ✅ | Updated Package Structure Documentation | ~201 |
| #456 | " | 🔵 | Package Structure Documentation Gap Identified | ~206 |
| #455 | " | 🔵 | Package Structure Documentation Review | ~84 |
| #329 | 10:44 AM | 🔵 | Codebase Analysis Summary | ~361 |
| #322 | 10:42 AM | 🔵 | Server Entry Point Analysis | ~285 |

### Jan 8, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #742 | 8:57 AM | 🔵 | Server Configuration Analysis | ~158 |
</claude-mem-context>