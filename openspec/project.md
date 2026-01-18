# Project Context

## Purpose
ReflectLogMCP is an MCP (Model Context Protocol) server that provides persistent, project-based memory storage for Claude Code and other AI agents. It combines:
- **Semantic search** via USearch HNSW vector embeddings (OpenRouter embeddings)
- **Full-text search** via Tantivy (stemmed + exact matching)
- **Hybrid ranking** via RRF (Reciprocal Rank Fusion)
- **Optional AI reranking** via LLM-based relevance scoring

Data persists across restarts in `indexes/{project_id}/usearch/` and `indexes/{project_id}/tantivy/`.

## Tech Stack

- **Language**: Python ≥3.13
- **Package Manager**: uv
- **MCP Framework**: fastmcp ≥2.13.1
- **Semantic Memory**: usearch ≥2.17.0
- **Full-Text Search**: tantivy ≥0.25.0
- **Hybrid Fusion**: ranx ≥0.3.21
- **LLM/Embeddings**: openai ≥2.8.1, langchain ≥1.1.0 (via OpenRouter)
- **Web Framework**: FastAPI ≥0.122.0, uvicorn ≥0.38.0 (HTTP transport)
- **Validation**: pydantic ≥2.12.4
- **Async**: anyio ≥4.11.0
- **Dev Tools**: ruff (linting), ty (type checking), pytest (testing)

## Project Conventions

### Code Style
- **Type Hints**: Comprehensive throughout; ty strict mode enforced
- **Type Aliases**: Use `TypeAlias` for semantic domain types (e.g., `MemoryRecord`, `SearchResult`)
- **Union Types**: Modern `X | None` syntax (not `Optional[X]`)
- **Docstrings**: Google/NumPy style with Args/Returns/Raises/Examples sections
- **Naming**:
  - Classes: `PascalCase` (e.g., `FastMCPServer`, `BaseTool`)
  - Functions/methods: `snake_case` (e.g., `get_handler`, `log_invocation`)
  - Constants: `SCREAMING_SNAKE_CASE` (e.g., `MCP_INSTRUCTIONS`, `SCORING_PROMPT`)
  - Private methods: `_snake_case` prefix (e.g., `_initialize_tools`)
- **Imports**: Organized in order: stdlib → third-party → local/relative (blank lines between)
- **Error Handling**: Consistent catch-log-re-raise pattern with `RuntimeError` wrapping
- **Logging**: Structured via `extra` dict (never string interpolation); includes tool name, parameters, counts

### Architecture Patterns
- **Layered Architecture** (Clean Architecture):
  - `server.py` - CLI/Entry Point Layer
  - `application/` - Application/Business Logic Layer
  - `infrastructure/` - External Integrations Layer
- **Dependency Injection**: Tools, MemoryManager, Logger injected via constructor
- **Registry Pattern**: `AVAILABLE_TOOL_CLASSES` maps tool names to implementations
- **Orchestrator Pattern**: `FastMCPServer` coordinates initialization of all components
- **Facade Pattern**: `MemoryManager` hides complexity of dual-engine storage
- **Template Method + Strategy**: `BaseTool` ABC defines contract; concrete tools provide logic

### Testing Strategy
- **Framework**: pytest + pytest-asyncio
- **Structure**:
  - `tests/unit/` - Isolated tests with mocked dependencies (USearch, Tantivy, OpenRouter)
  - `tests/integration/` - Real USearch/Tantivy indices, stubbed/real OpenRouter API
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- **Naming**: `test_{method}_{scenario}_{expected_result}()`
- **Fixtures**: Centralized in `tests/conftest.py` with factory fixtures
- **Mocking**: `unittest.mock` (Mock, MagicMock, patch)
- **Coverage**: Minimum 80% (target 90%)
- **Pattern**: AAA (Arrange, Act, Assert)
- **Commands**:
  ```bash
  ./start-unittest.sh                   # Run all tests
  ./start-unittest.sh --coverage        # With coverage report
  ./start-unittest.sh --parallel        # Parallel execution
  uv run pytest tests/unit/ -v          # Unit tests only
  uv run pytest -m "not slow"           # Exclude slow tests
  ```

### Git Workflow
- **Branches**: `main` (production), `develop` (active development)
- **Commit Messages**: Lowercase imperative form (e.g., "Enhance logging", "Fix search index")
- **Git Hooks**: Pre-push hook in `scripts/git-hooks/` runs:
  1. Type checking (`./start-type-check.sh`)
  2. Linting (`./start-lint.sh --all`)
- **Hook Installation**: `./scripts/setup-git-hooks.sh`
- **No CI/CD**: Validation is developer responsibility via git hooks

## Domain Context

### Core Concepts
- **Hybrid Search**: Combines semantic similarity (USearch) + exact phrase matching (Tantivy)
- **RRF Fusion**: `RRF_score(doc) = sum(1 / (k + rank(doc)))` - merges rankings from both engines
- **Two-Stage Filtering**:
  1. `FUSION_RANKING_THRESHOLD` (default 0.8) - filters after RRF fusion
  2. `SEARCH_SCORE_THRESHOLD` (default 0.5) - filters after AI reranking
- **Source of Truth**: USearch/SQLite for `get_all()` operations; Tantivy mirrors data

### MCP Tools
1. **add(messages: list[str])** - Store messages in both engines (validates 1-30720 chars)
2. **get_all() → list[str]** - Retrieve all messages from USearch
3. **search(query: str) → list[str]** - Hybrid search with optional AI reranking
4. **remove(messages: list[str])** - Exact-match deletion from both engines

### Transport Modes
- `stdio` (default) - JSON-RPC over stdin/stdout for MCP clients
- `http` - HTTP server with FastAPI
- `sse` - Server-Sent Events
- `streamable-http` - Streamable HTTP transport

## Important Constraints

### Required Configuration
- `PROJECT_ID` - Unique project identifier (alphanumeric, underscore, hyphen, dot; max 64 chars)
- `OPENROUTER_API_KEY` - OpenRouter API credentials for LLM and embeddings

### Output Stream Handling
- **Critical for stdio transport**: All non-JSON-RPC output must go to stderr only (see `server.py:110-111`)

### Performance Considerations
- Concurrent embedding requests limited via `anyio.Semaphore` (max 32)
- Batch embedding processing (batch size 64)
- HTTP/2 enabled for OpenAI client

### Storage
- Indices persist in `indexes/{project_id}/` directory
- USearch: `indexes/{project_id}/usearch/` (vectors.usearch + messages.db)
- Tantivy: `indexes/{project_id}/tantivy/`

## External Dependencies

### Required Services
| Service | Purpose | Configuration |
|---------|---------|---------------|
| **OpenRouter API** | LLM + Embeddings | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` |

### Storage Engines (Local)
| Engine | Purpose | Storage Location |
|--------|---------|------------------|
| **USearch** | Vector storage | `indexes/{project_id}/usearch/` |
| **Tantivy** | Full-text index | `indexes/{project_id}/tantivy/` |

### Key Environment Variables
```bash
# Required
PROJECT_ID=my-project
OPENROUTER_API_KEY=sk-or-...

# LLM/Embedding
LLM_MODEL=x-ai/grok-4.1-fast
EMBEDDING_MODEL=openai/text-embedding-3-large
EMBEDDER_PROVIDER=openai  # or langchain

# Search Tuning
SEARCH_LIMIT=5
RERANKER_ENGINE=llm  # or cross_encoder, none
SEARCH_SCORE_THRESHOLD=0.5
ENABLE_HYBRID_SEARCH=true
FUSION_RRF_K=60
FUSION_RANKING_THRESHOLD=0.8

# Server
MCP_TRANSPORT=stdio
MCP_PORT=9103
LOG_LEVEL=INFO
```
