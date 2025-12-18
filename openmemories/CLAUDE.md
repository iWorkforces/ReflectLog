# openmemories/

This directory contains the main source code for the OpenMemoriesMCP MCP server.

## Structure

```
openmemories/
├── __init__.py           # Package metadata (__version__ = "0.1.0") and exports
├── server.py             # CLI entry point and argument parsing
├── application/          # Application layer (business logic)
│   ├── mcp_server.py     # FastMCPServer orchestrator
│   ├── types.py          # Type definitions (ISemanticSearchEngine protocol)
│   ├── config/           # Configuration management
│   │   ├── settings.py   # Config dataclass from environment
│   │   └── prompts.py    # MCP_INSTRUCTIONS, SCORING_PROMPT
│   ├── memory/           # Memory management
│   │   ├── manager.py    # MemoryManager (USearch + Tantivy)
│   │   ├── protocols.py  # Search engine protocols
│   │   └── fusion/       # Hybrid ranking
│   │       ├── base.py   # FusionEngine protocol
│   │       └── ranx_fusion.py # RanxFusionEngine (RRF)
│   ├── tools/            # Modular MCP tool implementations
│   │   ├── base.py       # BaseTool abstract class
│   │   ├── add.py        # AddTool
│   │   ├── get_all.py    # GetAllTool
│   │   ├── search.py     # SearchTool
│   │   └── remove.py     # RemoveTool
│   └── utils/            # Utilities
│       ├── logging.py    # StructuredLogger, format_fusion_score_status
│       ├── numba_utils.py # Numba JIT functions (normalization, distance)
│       ├── security.py   # SecretString, redact_dict_secrets
│       └── validation.py # Message validation helpers
└── infrastructure/       # External integrations
    ├── message_store.py   # MessageStore (libSQL for USearch)
    ├── qwen3_embedding.py # LangchainQwenEmbeddings
    ├── tantivy_engine.py  # TantivyEngine (full-text search)
    └── usearch_engine.py  # USearchEngine (semantic search)
```

## Purpose

This is the top-level package that ties together:
1. **CLI interface** (`server.py`) - Handles command-line arguments, environment setup, and stdio/stderr routing
2. **Application logic** (`application/`) - Contains the FastMCPServer, MemoryManager, and modular MCP tools
3. **Infrastructure** (`infrastructure/`) - External service integrations (embedding providers)

## Entry Point Flow

When `openmemories` command is run:
1. `server.py::main()` is invoked (defined as entry point in `pyproject.toml`)
2. CLI arguments are parsed with `argparse`
3. Environment variables are set based on arguments (priority: CLI args > env vars > defaults)
4. Output stream is selected (stderr for stdio mode, stdout otherwise)
5. `FastMCPServer` is instantiated (loads config, initializes MemoryManager, registers tools)
6. `server.run()` starts the MCP server with configured transport

## Key Files

### `__init__.py`
- Exports `main` function and `__version__`
- Version is sourced here and used throughout the application
- Enables both CLI usage (`openmemories`) and programmatic usage (`from openmemories import main`)

### `server.py`
- **Responsibility**: CLI argument parsing and environment configuration
- **Critical detail**: Lines 110-111 handle stdio transport requirement that ALL output must go to stderr (not stdout) to avoid corrupting JSON-RPC protocol
- **Transport modes**: stdio (default), http, sse, streamable-http
- **Configuration priority**: CLI args > env vars > defaults

**CLI Arguments**:
```
--version           Show version and exit
--transport         Transport protocol (stdio, http, sse, streamable-http)
--port              Server port for non-stdio transports (default: 9104)
--host              Server host (default: 127.0.0.1)
--path              Server path (default: /mcp)
```

## Module Responsibilities

### application/
- **mcp_server.py**: `FastMCPServer` class that orchestrates initialization
- **types.py**: Core type definitions and `ISemanticSearchEngine` protocol
- **config/**: Centralized configuration from environment variables
- **memory/**: `MemoryManager` with hybrid USearch + Tantivy engines
- **memory/fusion/**: `RanxFusionEngine` for RRF hybrid ranking
- **tools/**: Modular tool implementations following `BaseTool` pattern
- **utils/**: Logging, validation, and security utilities

### infrastructure/
- **usearch_engine.py**: `USearchEngine` class for semantic vector search (HNSW)
- **tantivy_engine.py**: `TantivyEngine` class for full-text search
- **message_store.py**: `MessageStore` libSQL storage for message text
- **qwen3_embedding.py**: `LangchainQwenEmbeddings` class for custom embeddings
- Supports both sync and async operations
- HTTP/2 enabled via `DefaultAioHttpClient`
- Concurrency control with `anyio.Semaphore`

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
- Internal: `openmemories.__version__`, `application.mcp_server.FastMCPServer`

## Conventions

- **Imports**: Absolute imports preferred (`from openmemories.application import ...`)
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
| `MCP_PORT` | No | `9104` | Server port |
| `MCP_HOST` | No | `127.0.0.1` | Server host |
| `MCP_PATH` | No | `/mcp` | Server path |
| `LOG_LEVEL` | No | `INFO` | Logging level |

See `application/config/settings.py` for the complete configuration reference.
