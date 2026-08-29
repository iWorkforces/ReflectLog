# ReflectLog

> An Agentic Memory Layer For Coding Agents

[![Python](https://img.shields.io/badge/python-%E2%89%A53.14.2-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ReflectLog is an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that provides persistent, project-based memory storage for Claude Code and other AI agents. It combines semantic vector search with full-text search for intelligent memory retrieval.

## Features

- **Hybrid Search**: Combines semantic similarity (USearch) + exact phrase matching (Tantivy)
- **RRF Fusion**: Reciprocal Rank Fusion for optimal result ranking
- **Pluggable Reranking**: Local cross-encoder relevance scoring (or none)
- **Temporal-Aware Scoring**: Recency decay for handling contradictory memories
- **Smart Memory Replacement**: LLM-based detection of memory updates
- **Multiple Transport Modes**: stdio, HTTP, SSE, streamable-http
- **Lazy Initialization**: Fast startup with on-demand component loading

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ReflectLog.git
cd ReflectLog

# Install dependencies using uv
uv sync
```

### Configuration

Create a `.env` file:

```bash
WORKSPACE_ID=my-workspace_id
OPENROUTER_API_KEY=sk-or-your-key-here
```

### Running the Server

```bash
# Start with stdio transport (default for MCP clients)
uv run reflectlog

# Start with HTTP transport
uv run reflectlog --transport http --port 9103
```

## Usage

### MCP Tools

ReflectLog provides five MCP tools:

1. **add(memories: list[str])** - Store memories with semantic embeddings
2. **get_all() -> list[str]** - Retrieve all stored memories
3. **search(query: str) -> list[str]** - Hybrid semantic + full-text search
4. **remove(memories: list[str])** - Remove memories by exact match
5. **health_check() -> dict** - Get server health status

### Example Usage

```python
# Add memories
await add(["I prefer Python for web development", "I use FastAPI for APIs"])

# Search semantically
results = await search("web frameworks")
# Returns: ["I prefer Python for web development"]

# Get all memories
all = await get_all()
# Returns all stored memories

# Remove memories
await remove(["I use FastAPI for APIs"])
```

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `WORKSPACE_ID` | Unique workspace identifier |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM/embeddings |

### Optional Configuration

```bash
# Search Settings
SEARCH_LIMIT=5                    # Max results per search
RERANKER_ENGINE=cross_encoder     # cross_encoder or none
ENABLE_HYBRID_SEARCH=true         # Enable full-text search

# Memory Replacement
ENABLE_SMART_REPLACE=true         # LLM-based memory replacement
SMART_REPLACE_THRESHOLD=0.7       # Confidence threshold

# Server
MCP_TRANSPORT=stdio               # stdio, http, sse, streamable-http
MCP_PORT=9103                     # Port for HTTP transport
LOG_LEVEL=INFO                    # Logging level
```

See `.env.example` for all available options.

## Architecture

```
ReflectLog/
├── reflectlog/
│   ├── server.py              # CLI entry point
│   ├── application/           # Business logic
│   │   ├── mcp_server.py      # FastMCPServer orchestrator
│   │   ├── memory/            # Memory management
│   │   ├── tools/             # MCP tool implementations
│   │   └── config/            # Configuration management
│   └── infrastructure/        # External integrations
│       ├── usearch_engine.py  # Semantic vector search
│       ├── tantivy_engine.py  # Full-text search
│       └── cross_encoder_reranker.py  # Local cross-encoder reranking
```

### Data Persistence

- **USearch**: `indexes/{workspace_id}/usearch/` - Vector index + SQLite messages
- **Tantivy**: `indexes/{workspace_id}/tantivy/` - Full-text index

## Development

### Commands

```bash
# Type checking
./start-type-check.sh

# Linting
./start-lint.sh --all

# Testing
./start-unittest.sh
./start-unittest.sh --coverage

# Run server
./start-reflectlog-mcp-server.sh --workspace_id my-workspace_id
```

### Testing

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit/ -v

# Run with coverage
./start-unittest.sh --coverage
```

## Performance

- **Exact Search**: SIMD-optimized brute-force (default, best for <10K vectors)
- **Approximate Search**: HNSW algorithm (for large collections)
- **Phased Parallel Add**: 3-phase pipeline for 5-8x speedup on bulk operations
- **LRU Query Cache**: Reduces embedding API calls

## Documentation

- [CLAUDE.md](CLAUDE.md) - Comprehensive developer documentation
- [openspec/AGENTS.md](openspec/AGENTS.md) - OpenSpec workflow guide
- [.env.example](.env.example) - Full configuration reference

## Contributing

1. Install git hooks: `./scripts/setup-git-hooks.sh`
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `./start-unittest.sh`
5. Commit and push

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [USearch](https://github.com/unum-cloud/usearch) - Vector search engine
- [Tantivy](https://github.com/tantivy-search/tantivy-py) - Full-text search
- [ranx](https://github.com/AmenRa/ranx) - Ranking fusion algorithms
