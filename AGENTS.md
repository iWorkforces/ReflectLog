<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

## Always open `@/openspec/AGENTS.md` when the request

- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

## Use `@/openspec/AGENTS.md` to learn

- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

## Leverages MCP (Model Context Protocol) servers for enhanced capabilities

1. **Sequential Thinking Tools MCP** (`mcp__sequentialthinking-tools__sequentialthinking_tools`)
   - Used for structured analysis and validation
   - Provides step-by-step reasoning for impact scoring and recommendations

2. **Tavily MCP** (`mcp__tavily-mcp__tavily-search`, `mcp__tavily-mcp__tavily-extract`)
   - Used for researching industry best practices and standards
   - Provides real-time validation against OWASP, NIST, WCAG, and other standards
   - Enables anti-pattern detection and production readiness checks

3. **Context7 MCP** (`mcp__context7__resolve-library-id`, `mcp__context7__get-library-docs`)
   - Used for technology-specific guidance
   - Provides framework and library best practices
   - Enables stack-aware question generation

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# ReflectLogMCP - Agent Coding Guidelines

## Build / Lint / Test Commands

```bash
# Install dependencies
uv sync

# Type checking (ty - strict mypy alternative)
./start-type-check.sh

# Linting (ruff)
./start-lint.sh --all        # Check, fix, and format
./start-lint.sh --check      # Check only
./start-lint.sh --fix        # Fix issues automatically

# Testing
./start-unittest.sh                      # Run all tests
./start-unittest.sh --coverage           # With coverage report
./start-unittest.sh --parallel           # Parallel execution
./start-unittest.sh --file tests/unit/application/test_memory_manager.py  # Single file
./start-unittest.sh --pattern test_add   # Tests matching pattern

# Run server
uv run reflectlog --transport http --port 9103
```

## Code Style Guidelines

### Imports
- Use absolute imports: `from reflectlog.application.config import Config`
- Group imports: stdlib → third-party → local (separated by blank lines)
- Use `TYPE_CHECKING` guard for type-only imports

### Formatting
- Line length: 120 characters (ruff default)
- Use triple single quotes for docstrings: `'''docstring'''`
- Run `./start-lint.sh --format` before committing

### Types
- Python 3.14+ required (no type union syntax like `List[str] | None`, use `str | None`)
- Use `ty` for static type checking (strict mode)
- Avoid `Any`, `Union`, `Optional` where native union syntax works
- Private attributes use `PrivateAttr` in Pydantic models

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `MemoryManager`, `SearchError`)
- **Functions/Methods**: `snake_case` (e.g., `get_messages()`, `_init_engine()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `LOG_ADD_MESSAGE_PREVIEW_LIMIT`)
- **Private members**: Leading underscore (e.g., `_lock`, `_client`)
- **Type variables**: `PascalCase` with `T` prefix (e.g., `TResult`, `TConfig`)

### Error Handling
- Use custom exception hierarchy (see `reflectlog/application/exceptions.py`)
- Always chain exceptions with `from e`: `raise XError(...) from e`
- Never use bare `except:` - catch specific exceptions
- Log errors with `StructuredLogger` before raising

### Async Code
- Use `asyncio_mode = auto` in pytest
- Lazy initialization for expensive resources (embedders, rerankers)
- Use `anyio` for cross-platform async operations

### Thread Safety
- Follow lock hierarchy: `_write_lock` before `_lock` (see `MemoryManager` docstring)
- USearch is not thread-safe: serialize writes with `_write_lock`
- RLock for methods that may call other protected methods

### Testing
- Test files: `tests/unit/` and `tests/integration/`
- Use `pytest.mark.unit` / `pytest.mark.integration` markers
- Fixtures in `conftest.py` at test directory root
- Mock external services (LLM APIs, embedding services)

### Project Structure
```
reflectlog/
├── core/              # Protocol definitions and abstractions
│   ├── __init__.py    # Package exports
│   ├── config.py      # Configuration protocols (IServerConfig, ISearchConfig, etc.)
│   ├── config_adapters.py  # Config adapters for protocol-based DI
│   ├── memory.py      # Memory operation protocols (IMemoryStore, IMemoryManager)
│   ├── search.py      # Search engine protocols (ISearchBackend, IFusionAlgorithm)
│   ├── reranking.py   # Reranker protocols (IReranker, IRerankerProvider)
│   ├── tools.py       # Tool registration protocols (ITool, IToolRegistry)
│   └── logging.py     # Logging protocols (ILoggingService, LogLevel)
│
├── application/       # Business logic (depends on core)
│   ├── mcp_server.py  # MCP server orchestration
│   ├── memory/        # Memory management
│   │   ├── manager.py              # MemoryManager (facade)
│   │   ├── engine_factory.py       # Engine factory for search engines
│   │   ├── search_pipeline.py      # Search pipeline with pluggable stages
│   │   ├── add_pipeline.py         # Add pipeline with pluggable phases
│   │   ├── fusion/                 # RRF fusion algorithms
│   │   ├── search_strategies.py    # Original search strategies
│   │   ├── add_phases.py           # Original add phases
│   │   └── match_utils.py          # Match utilities
│   ├── tools/        # MCP tool implementations
│   ├── config/       # Configuration
│   └── utils/        # Utilities
│
├── infrastructure/   # Implementations (depend on core)
│   ├── search/       # Search engine implementations (re-exports from parent)
│   ├── embeddings/   # Embedding provider implementations
│   ├── reranking/    # Reranker implementations
│   ├── memory/       # Memory storage implementations
│   └── llm/          # LLM provider implementations
│
├── plugins/          # Plugin system for extensibility
│   ├── discovery.py  # Plugin discovery mechanisms
│   ├── registry.py   # Plugin registry
│   └── loading.py    # Plugin loading and lifecycle
│
├── utility/          # Platform-specific utilities
└── server.py         # CLI entry point
```

### Key Patterns

1. **Protocol-Based Design**: Components depend on protocols from `core/` rather than concrete implementations
2. **Pluggable Pipelines**: Search and add operations use composable stages with protocol interfaces
3. **Plugin Architecture**: Discovery via entry points, directory scan, or static registration
4. **Factory Pattern**: EngineFactory creates and configures search engines
5. **Dependency Injection**: Components receive dependencies through constructor parameters
