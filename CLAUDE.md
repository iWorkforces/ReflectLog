<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CCMemoriesMCP is an MCP (Model Context Protocol) server that provides persistent, project-based memory storage for intelligent AI Agents with **hybrid semantic + full-text search**. It combines:

- **Semantic vector search**: USearch HNSW algorithm with libSQL text storage
- **Full-text search**: Tantivy (stemmed + exact matching via `en_stem` tokenizer)
- **RRF fusion**: Reciprocal Rank Fusion via ranx library for hybrid ranking
- **Pluggable reranking**: LLM-based or local cross-encoder relevance scoring (default: LLM)

Data persists across restarts in `indexes/{project_id}/usearch/` and `indexes/{project_id}/tantivy/`.

## Development Commands

### Package Management
This project uses `uv` for fast, reliable Python package management. All commands should be run with `uv`:

```bash
uv sync                    # Install/sync all dependencies
uv pip install <package>   # Install a package
uv run python <script>     # Run a Python script
```

### Running the Server

**Command-line tool** (after installation):
```bash
ccmemories                              # stdio transport (default)
ccmemories --transport http --port 9103 # HTTP transport
```

**Direct execution** (development):
```bash
uv run python ccmemories/server.py --transport http
./start-ccmemories-mcp-server.sh --project_id my-project
```

### Code Quality

**Linting** (uses ruff):
```bash
./start-lint.sh          # Check for issues
./start-lint.sh --fix    # Auto-fix issues
./start-lint.sh --format # Format code
./start-lint.sh --all    # Check, fix, and format
```

**Type Checking** (uses ty with strict rules):
```bash
./start-type-check.sh           # Run type check
./start-type-check.sh --concise # Run with concise output (for CI)
./start-type-check.sh --stats   # Show statistics
```

**Testing** (uses pytest):
```bash
./start-unittest.sh                     # Run all tests
./start-unittest.sh --coverage          # With coverage report
./start-unittest.sh --parallel          # Run in parallel
./start-unittest.sh --file tests/...    # Run specific file
./start-unittest.sh --pattern <name>    # Run tests matching pattern
```

## Architecture

### Transport Modes
The server supports multiple transport protocols:
- **stdio**: Default for MCP clients (JSON-RPC over stdin/stdout)
- **http**: HTTP server mode
- **sse**: Server-Sent Events
- **streamable-http**: Streamable HTTP transport

Transport can be configured via:
1. CLI args: `--transport`, `--port`, `--host`, `--path`
2. Environment variables: `MCP_TRANSPORT`, `MCP_PORT`, `MCP_HOST`, `MCP_PATH`
3. Defaults: stdio transport, port 9103, host 127.0.0.1

### Project Structure

```
CCMemoriesMCP/
├── ccmemories/              # Main package
│   ├── __init__.py           # Package metadata (__version__ = "0.1.0")
│   ├── server.py             # CLI entry point
│   ├── application/          # Application layer
│   │   ├── mcp_server.py     # FastMCPServer orchestrator
│   │   ├── types.py          # Type definitions (ISemanticSearchEngine protocol)
│   │   ├── config/           # Configuration management
│   │   │   ├── settings.py   # Config dataclass from env vars
│   │   │   └── prompts.py    # MCP_INSTRUCTIONS, SCORING_PROMPT
│   │   ├── memory/           # Memory management
│   │   │   ├── manager.py    # MemoryManager (USearch + Tantivy)
│   │   │   ├── protocols.py  # Search engine protocols
│   │   │   ├── fusion/       # Hybrid ranking
│   │   │   │   ├── base.py   # FusionEngine protocol
│   │   │   │   └── ranx_fusion.py # RanxFusionEngine (RRF)
│   │   │   └── reranking/    # Score normalization utilities
│   │   │       └── normalization.py # Min-max normalization for rerankers
│   │   ├── tools/            # MCP tool implementations
│   │   │   ├── base.py       # BaseTool abstract class
│   │   │   ├── add.py        # AddTool
│   │   │   ├── get_all.py    # GetAllTool
│   │   │   ├── search.py     # SearchTool
│   │   │   └── remove.py     # RemoveTool
│   │   └── utils/            # Utilities
│   │       ├── logging.py    # StructuredLogger, format_fusion_score_status
│   │       ├── numba_utils.py # Numba JIT functions (normalization, distance)
│   │       ├── security.py   # SecretString, redact_dict_secrets
│   │       └── validation.py # Message validation
│   └── infrastructure/       # External integrations
│       ├── cross_encoder_reranker.py # CrossEncoderReranker (FlagReranker-based local reranking)
│       ├── llm_reranker.py    # LLMReranker (AI relevance scoring)
│       ├── message_store.py   # MessageStore (libSQL for USearch)
│       ├── qwen3_embedding.py # LangchainQwenEmbeddings
│       ├── smart_replacer.py  # SmartReplacer (LLM-based memory replacement detection)
│       ├── tantivy_engine.py  # TantivyEngine (full-text search wrapper)
│       └── usearch_engine.py  # USearchEngine (USearch/SQLite wrapper)
├── tests/                    # Test suites
│   ├── conftest.py           # Pytest fixtures
│   ├── unit/                 # Unit tests (mocked deps)
│   │   ├── application/      # Application layer tests
│   │   └── infrastructure/   # Infrastructure layer tests
│   └── integration/          # Integration tests (real engines)
├── stubs/                    # Type stubs for ty
│   ├── fastmcp/              # FastMCP library stubs
│   ├── libsql/               # libSQL library stubs
│   ├── numba/                # Numba JIT compiler stubs
│   ├── ranx/                 # RRF ranking library stubs
│   └── usearch/              # USearch library stubs
├── scripts/                  # Development scripts
│   ├── benchmark_engines.py  # Performance benchmarking
│   └── git-hooks/            # Version-controlled git hooks
└── indexes/                  # Persisted indices (gitignored)
    └── {project_id}/
        ├── usearch/          # USearch vector + SQLite
        └── tantivy/          # Tantivy full-text index
```

### Hybrid Memory Storage (MemoryManager)

**Core Engine**: `ccmemories/application/memory/manager.py`

**Infrastructure Layer**:

- Uses dedicated engine classes from `ccmemories/infrastructure/`
- **USearchEngine**: Semantic vector search with USearch (HNSW) + libSQL text storage
- **TantivyEngine**: Full-text search with English stemming

**Semantic Search** (USearchEngine):

- Infrastructure: `ccmemories.infrastructure.USearchEngine`
- Backend: USearch HNSW index + libSQL `MessageStore` for text
- Embeddings: `LangchainQwenEmbeddings` (Qwen 4096 dims) or OpenAI compatible
- Storage: `indexes/{project_id}/usearch/` (vectors.usearch + messages.db)
- **Source of truth** for `get_all()` - returns all stored messages
- **Search modes**: Supports both exact (brute-force) and approximate (HNSW) search:
  - **Approximate (default)**: Uses HNSW algorithm for fast nearest neighbor search
  - **Exact**: Bypasses HNSW and uses SIMD-optimized brute-force search from SimSIMD
  - Configure via `USEARCH_EXACT_SEARCH` or auto-switch with `USEARCH_EXACT_SEARCH_THRESHOLD`

**Full-text Search** (TantivyEngine):

- Infrastructure: `ccmemories.infrastructure.TantivyEngine`
- Schema: `project_id` (raw tokenizer), `message` (en_stem tokenizer)
- Storage: `indexes/{project_id}/tantivy/`
- Optimized for exact phrase matching and keyword search

**Hybrid Search with RRF Fusion**:

- USearchEngine + TantivyEngine → **RanxFusionEngine** → optional **Reranker**
- RRF formula: `RRF_score(doc) = sum(1 / (k + rank(doc)))`
- Implementation: `ccmemories/application/memory/fusion/ranx_fusion.py`
- Configurable `k` parameter via `FUSION_RRF_K` (default: 60)

**Pluggable Reranking** (LLMReranker or CrossEncoderReranker):

- Controlled by `RERANKER_ENGINE` env var: `llm`, `cross_encoder`, or `none`
- **LLMReranker** (`reranker_engine=llm`, default):
  - Infrastructure: `ccmemories.infrastructure.LLMReranker`
  - Purpose: AI-powered relevance scoring via OpenRouter API
  - Uses `SCORING_PROMPT` template from `config/prompts.py`
  - Parallel scoring with concurrency control via `anyio.Semaphore`
  - HTTP/2 enabled via AsyncOpenAI with `DefaultAioHttpClient`
  - Graceful fallback: returns fusion score if LLM call fails
- **CrossEncoderReranker** (`reranker_engine=cross_encoder`):
  - Infrastructure: `ccmemories.infrastructure.CrossEncoderReranker`
  - Purpose: Fast local reranking using FlagEmbedding's FlagReranker
  - Default model: `BAAI/bge-reranker-v2-m3` (multilingual, high quality)
  - No API costs, runs locally on CPU/GPU/MPS
  - Built-in FP16 support for faster inference
  - Built-in score normalization (sigmoid to 0-1 range)
  - Lazy model loading with thread-safe initialization
  - Score threshold filtering before returning results

**Smart Memory Replacement** (SmartReplacer):

- Infrastructure: `ccmemories.infrastructure.SmartReplacer`
- Purpose: Detect when a new memory semantically replaces an existing one
- Example: "I like cats" → "I don't like cats anymore, I like dogs"
- Uses LLM (via `LLM_MODEL`) with structured JSON output for replacement detection
- Configurable confidence threshold (default: 0.7)
- Enabled by default, can be disabled via `ENABLE_SMART_REPLACE=false`
- Graceful degradation: if LLM call fails, memory is added normally
- Detailed logging: old/new memory preview, confidence score, reason

**Config Options** (via env vars):

- `PROJECT_ID` (required): Unique project identifier
- `OPENROUTER_API_KEY` (required): OpenRouter API key for LLM/embeddings
- `ENABLE_HYBRID_SEARCH`: Enable Tantivy full-text (default: true)
- `DEDUPLICATE_MESSAGES`: Skip exact duplicates on add (default: true)
- `ENABLE_SMART_REPLACE`: Enable smart memory replacement detection (default: true)
- `SMART_REPLACE_THRESHOLD`: Min LLM confidence to trigger replacement (default: 0.7)
- `SMART_REPLACE_MIN_SIMILARITY`: Min embedding similarity to trigger LLM check (default: 0.9)
- `SMART_REPLACE_CANDIDATE_LIMIT`: Max candidates to check for replacement (default: 3)
- `SMART_REPLACE_ARCHIVE_TTL_DAYS`: Days to keep archived memories, 0=permanent (default: 30)
- `SMART_REPLACE_MAX_RETRIES`: Max LLM call retries with exponential backoff (default: 3)
- `SMART_REPLACE_RETRY_DELAY`: Base delay in seconds for exponential backoff (default: 1.0)
- `SMART_REPLACE_PROVIDER`: LLM provider for smart replacement: `openai` or `anthropic` (default: openai)
- `SEARCH_LIMIT`: Max results returned (default: 5)
- `ENABLE_RRF_FUSION`: Enable RRF fusion for hybrid search (default: true)
- `FUSION_RRF_K`: RRF constant (default: 60)
- `FUSION_RANKING_THRESHOLD`: Min normalized RRF score to keep, only when RRF enabled (default: 0.8)
- `RERANKER_ENGINE`: Reranking engine: `llm`, `cross_encoder`, or `none` (default: llm)
- `SEARCH_SCORE_THRESHOLD`: Min LLM relevance score to keep (default: 0.5)
- `RERANK_MAX_CONCURRENCY`: Max parallel LLM calls for reranking (default: 10)
- `LLM_MODEL`: LLM model for reranking (default: `x-ai/grok-4.1-fast`)
- `CROSS_ENCODER_MODEL`: Cross-encoder model (default: `BAAI/bge-reranker-v2-m3`)
- `CROSS_ENCODER_TOP_K`: Max results after cross-encoder reranking (default: 20)
- `CROSS_ENCODER_DEVICE`: Inference device: `cpu`, `cuda`, `mps` (default: cpu)
- `CROSS_ENCODER_BATCH_SIZE`: Batch size for inference (default: 32)
- `CROSS_ENCODER_SCORE_THRESHOLD`: Min cross-encoder score to keep (default: 0.0)
- `CROSS_ENCODER_USE_FP16`: Enable FP16 for faster inference (default: true)
- `CROSS_ENCODER_NORMALIZE`: Normalize scores to 0-1 with sigmoid (default: true)
- `CROSS_ENCODER_MAX_LENGTH`: Max token length for query-doc pairs (default: 512)
- `RERANKER_MIN_RESULTS`: Safety net: min results to return, 0 = disabled (default: 0)
- `RERANKER_BATCH_NORMALIZE`: Enable batch min-max normalization for both rerankers (default: true)
- `REMOVE_SCORE_THRESHOLD`: Min score for remove candidates (default: 0.9)
- `EMBEDDING_MODEL`: Embedding model (default: `qwen/qwen3-embedding-8b`)
- `QWEN_EMBEDDING_DIMS`: Embedding dimensions (default: 4096)
- `EMBEDDER_PROVIDER`: `langchain` (default: langchain)
- `ALLOWED_TOOLS`: Comma-separated tools to enable (default: all)
- `USEARCH_EXACT_SEARCH`: Force exact brute-force search (default: true)
- `USEARCH_EXACT_SEARCH_THRESHOLD`: Auto-switch to exact when index < threshold (default: 0, disabled)
- `ADD_MAX_CONCURRENCY`: Max concurrent tasks for phased parallel add (default: 4)
- `EMBEDDING_BATCH_SIZE`: Batch size for async embedding operations (default: 512)
- `EMBEDDING_MAX_CONCURRENT_BATCHES`: Max parallel batch requests (default: 4)
- `EMBEDDING_CACHE_ENABLED`: Enable LRU cache for query embeddings (default: true)
- `EMBEDDING_CACHE_SIZE`: LRU cache size for query embeddings (default: 100)
- `OVERFETCH_MULTIPLIER`: Base multiplier for fetching candidates (default: 3)
- `OVERFETCH_ADAPTIVE`: Enable adaptive overfetch based on index size (default: true)
- `OVERFETCH_MIN_MULTIPLIER`: Min multiplier for large indexes (default: 1.5)
- `OVERFETCH_MAX_MULTIPLIER`: Max multiplier for small indexes (default: 3.0)
- `TANTIVY_SOFT_DELETE_ENABLED`: Use tombstone marking instead of index rebuild (default: true)
- `TANTIVY_COMPACTION_THRESHOLD_RATIO`: Compact when tombstones > ratio of live docs (default: 0.2)
- `TANTIVY_COMPACTION_MAX_TOMBSTONES`: Force compaction above this count (default: 10000)
- `TANTIVY_TOMBSTONE_TTL_DAYS`: Days before tombstones eligible for removal (default: 7)
- `EAGER_INITIALIZATION`: Pre-warm engines during startup (default: true)

### MCP Tools

Four modular FastMCP tools backed by `MemoryManager`:

1. **add(messages: list[str])**: Store messages in hybrid storage
   - Validates message length (1-30720 chars)
   - **Phased parallel processing** (3 phases for optimal performance):
     - Phase 1: Parallel duplicate detection + batch deduplication
     - Phase 2: Parallel smart replacement detection
     - Phase 3: Sequential database writes (SQLite constraint)
   - **Smart replacement**: Detects if new memory replaces an existing one (via LLM)
   - Deduplicates exact matches (via TantivyEngine or O(log n) database lookup)
   - Dual engine storage: USearchEngine (vectors) + TantivyEngine (full-text)

2. **get_all() -> list[str]**: Retrieve all messages
   - **Source of truth**: USearchEngine (libSQL MessageStore)
   - Returns defensive copy (new list each time)
   - Empty list `[]` if no messages stored

3. **search(query: str) -> list[str]**: Hybrid semantic + full-text search
   - Pydantic validation: `Field(min_length=1)`
   - Hybrid search: USearch + Tantivy → RRF fusion → optional LLM rerank
   - Score threshold filtering (default: 0.5)
   - Configurable limit (default: 5)

4. **remove(messages: list[str])**: Exact match search + delete
   - Uses USearch (source of truth) for finding messages
   - Case-sensitive exact string matching
   - Removes ALL occurrences of each message
   - Silently ignores non-existent messages (logs but no error)

### Entry Point Flow

1. `ccmemories/server.py::main()`: CLI parsing, env setup
2. `FastMCPServer.__init__()`: Init `MemoryManager`, register tools
3. `FastMCPServer._initialize_tools()`: Create tool instances
4. `FastMCPServer._register_tools()`: Register with FastMCP
5. `FastMCPServer.run()`: Start server with configured transport

### Output Stream Handling

**Critical for stdio**: Non-JSON-RPC output to stderr only (see `server.py:110-111`).

## Code Style

- **Type hints**: Everywhere (ty strict rules)
- **Docstrings**: Public functions with Args/Returns/Raises
- **Logging**: Structured with `extra` dict (project_id, tool, etc.)
- **Errors**: Catch/log/re-raise as `RuntimeError`

## Testing

**Structure**:

```
tests/
├── conftest.py        # Shared fixtures
├── unit/              # Isolated (mocked engines)
│   ├── application/   # Application layer tests (mcp_server, memory_manager, etc.)
│   └── infrastructure/# Infrastructure tests (engines, embeddings)
└── integration/       # Real USearch/Tantivy/OpenRouter
    └── test_memory_manager_usearch.py  # End-to-end with real engines
```

**Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

**Running**:

```bash
uv run pytest tests/unit/ -v              # Unit tests only
uv run pytest tests/integration/ -v       # Integration tests only
uv run pytest -m "not slow"               # Exclude slow tests
uv run pytest tests/unit/infrastructure/  # Infrastructure unit tests only
```

## Dependencies

**Runtime** (from pyproject.toml):

- Python ≥3.13
- fastmcp ≥2.13.2
- tantivy ≥0.25.1
- usearch ≥2.21.0
- libsql ≥0.1.11
- openai ≥2.9.0
- langchain ≥1.1.3
- langchain-openai ≥1.1.1
- pydantic ≥2.12.5
- ranx ≥0.3.21 (RRF fusion)
- anyio ≥4.11.0

**Dev**: ruff, ty, pytest (-asyncio, -cov, -xdist)

## Common Tasks

### Adding a New MCP Tool

1. Create `ccmemories/application/tools/new_tool.py`:

   ```python
   from .base import BaseTool

   class NewTool(BaseTool):
       def get_name(self) -> str:
           return "new_tool"

       def get_handler(self):
           def new_tool(param: str) -> str:
               """Docstring with Args/Returns/Raises."""
               self.log_invocation("new_tool", param=param)
               # Implementation
               self.log_completion("new_tool")
               return result
           return new_tool
   ```

2. Register in `mcp_server.py:AVAILABLE_TOOL_CLASSES`:

   ```python
   AVAILABLE_TOOL_CLASSES: Dict[str, Type[BaseTool]] = {
       "add": AddTool,
       "get_all": GetAllTool,
       "search": SearchTool,
       "remove": RemoveTool,
       "new_tool": NewTool,  # Add here
   }
   ```

3. Update `prompts.py:MCP_INSTRUCTIONS` with tool documentation

4. Add tests in `tests/unit/application/test_new_tool.py`

### Tuning Hybrid Search

- Disable hybrid search entirely: `ENABLE_HYBRID_SEARCH=false`
- Disable RRF fusion (concatenate instead): `ENABLE_RRF_FUSION=false`
- Adjust RRF k parameter: `FUSION_RRF_K=30` (lower = more weight to top ranks)
- Adjust fusion threshold: `FUSION_RANKING_THRESHOLD=0.3` (permissive) or `0.7` (strict)

**RRF Fusion vs Concatenation**:

| Setting | Behavior | Use Case |
|---------|----------|----------|
| `ENABLE_RRF_FUSION=true` | Combines semantic + full-text using RRF algorithm | Best quality ranking |
| `ENABLE_RRF_FUSION=false` | Concatenates results (semantic first, then full-text deduped) | Faster, simpler |

When RRF is disabled:
- Step 2 concatenates results with semantic priority
- Step 3 (filtering) is skipped (scores from different engines aren't comparable)
- Reranking becomes Step 3 instead of Step 4

**Reranker Engine Selection**:

- Use LLM reranking (default): `RERANKER_ENGINE=llm`
- Use local cross-encoder: `RERANKER_ENGINE=cross_encoder`
- Disable reranking: `RERANKER_ENGINE=none`

**LLM Reranking Tuning** (when `RERANKER_ENGINE=llm`):

- Adjust threshold: `SEARCH_SCORE_THRESHOLD=0.7`
- Adjust parallelism: `RERANK_MAX_CONCURRENCY=15` (default: 10)
- Change model: `LLM_MODEL=openai/gpt-4o-mini`

**Cross-Encoder Tuning** (when `RERANKER_ENGINE=cross_encoder`):

- Change model: `CROSS_ENCODER_MODEL=BAAI/bge-reranker-base` (faster, smaller)
- Adjust results: `CROSS_ENCODER_TOP_K=10`
- Use GPU: `CROSS_ENCODER_DEVICE=cuda` or `CROSS_ENCODER_DEVICE=mps`
- Adjust threshold: `CROSS_ENCODER_SCORE_THRESHOLD=0.5`
- Disable FP16: `CROSS_ENCODER_USE_FP16=false` (for full precision)
- Disable normalization: `CROSS_ENCODER_NORMALIZE=false` (raw scores)

**Smart Memory Replacement Tuning**:

- Disable feature: `ENABLE_SMART_REPLACE=false`
- Adjust confidence threshold: `SMART_REPLACE_THRESHOLD=0.8` (stricter) or `0.5` (permissive)
- Adjust similarity pre-filter: `SMART_REPLACE_MIN_SIMILARITY=0.5` (check more candidates)
- Increase candidates: `SMART_REPLACE_CANDIDATE_LIMIT=5` (check more memories)
- Adjust archive retention: `SMART_REPLACE_ARCHIVE_TTL_DAYS=90` (keep longer) or `0` (permanent)
- Configure retry behavior: `SMART_REPLACE_MAX_RETRIES=5`, `SMART_REPLACE_RETRY_DELAY=2.0`
- Change LLM model: `LLM_MODEL=openai/gpt-4o-mini` (shared with reranking)
- Change LLM provider: `SMART_REPLACE_PROVIDER=anthropic` (uses claude-agent-sdk)

**Smart Replacement Flow** (on add):
```
New Memory → [Step 1: Semantic Search] → [Step 2: Similarity Filter] → [Step 3: LLM Check] → [Step 4: Replace/Add]
              Find top N similar          Score >= 0.5?               Should replace?          Archive old + Add new
```

**Archiving**:
- Replaced memories are soft-deleted to `archived_messages` table before removal
- Archives include: original memory, replacement memory, confidence, reason, timestamp
- Retention period configurable via `SMART_REPLACE_ARCHIVE_TTL_DAYS` (default: 30 days)

### Tuning USearch Exact vs Approximate Search

USearch supports two search modes:

- **Exact (brute-force)**: Default. Guaranteed exact results using SIMD-optimized SimSIMD. Best for small collections.
- **Approximate (HNSW)**: Fast for large collections, slight accuracy tradeoff.

Configuration options:

- Keep defaults (exact): `USEARCH_EXACT_SEARCH=true` (recommended for small databases)
- Switch to approximate: `USEARCH_EXACT_SEARCH=false` (for large collections)
- Auto-switch based on size: `USEARCH_EXACT_SEARCH_THRESHOLD=10000` (use exact when index < 10k vectors)

**When to use exact search (default):**

- Small collections (< 10,000 vectors)
- When 100% recall is critical
- Development/testing environments

**When to use approximate search:**

- Large collections (> 10,000 vectors)
- Production environments prioritizing speed
- When slight recall reduction is acceptable

**Search Pipeline (4 Steps):**

```
Query → [Step 1: Parallel Search] → [Step 2: RRF Fusion] → [Step 3: Fusion Filter] → [Step 4: Rerank] → Results
         USearch + Tantivy           RanxFusionEngine       threshold >= 0.8        LLM/CrossEncoder/None
```

**Step 4 Reranker Options** (via `RERANKER_ENGINE`):

- `llm`: LLMReranker - API-based semantic scoring (default)
- `cross_encoder`: CrossEncoderReranker - Local model inference
- `none`: Skip reranking, use fusion scores directly

**Two-Stage Threshold Filtering:**

| Setting | Stage | Purpose |
|---------|-------|---------|
| `FUSION_RANKING_THRESHOLD` | Step 3: After RRF fusion | Filter low-confidence fusion matches (default: 0.8) |
| `SEARCH_SCORE_THRESHOLD` | Step 4: After LLM reranking | Filter low-relevance LLM results (default: 0.5) |
| `CROSS_ENCODER_SCORE_THRESHOLD` | Step 4: After cross-encoder | Filter low-relevance cross-encoder results (default: 0.0, normalized 0-1 when CROSS_ENCODER_NORMALIZE=true) |

### Using Different Embedding Models

**Langchain/Qwen (default)**:
```bash
EMBEDDER_PROVIDER=langchain
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
QWEN_EMBEDDING_DIMS=4096
```

The default uses `LangchainQwenEmbeddings` from `ccmemories/infrastructure/qwen3_embedding.py`.

## Environment Setup

Create a `.env` file:
```bash
PROJECT_ID=my-project
OPENROUTER_API_KEY=sk-or-...
# Optional overrides
LLM_MODEL=x-ai/grok-4.1-fast
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
SEARCH_LIMIT=10
RERANKER_ENGINE=llm  # or "cross_encoder" or "none"
ENABLE_SMART_REPLACE=true  # Smart memory replacement (default: true)
SMART_REPLACE_THRESHOLD=0.7  # Min confidence for replacement (default: 0.7)
SMART_REPLACE_PROVIDER=openai  # or "anthropic" (uses claude-agent-sdk)
LOG_LEVEL=INFO
```

## Performance Optimizations

The server includes several performance optimizations for high-throughput scenarios.

### Phased Parallel Add Processing

When adding multiple messages, the `add_messages_async()` method uses a 3-phase parallel approach:

```
Messages → [Phase 1: Parallel Dedup] → [Phase 2: Parallel Replace] → [Phase 3: Sequential Write] → Stored
           Batch dedup + Storage check   Smart replacement LLM calls   SQLite/Tantivy writes
```

**Phase 1: Parallel Duplicate Detection**
- Deduplicates messages within the batch itself (O(1) hash lookup)
- Parallel duplicate checks against existing storage (via semaphore)
- Uses O(log n) database lookup when Tantivy is disabled

**Phase 2: Parallel Smart Replacement Detection**
- Runs LLM replacement checks in parallel for non-duplicate messages
- Controlled by `ADD_MAX_CONCURRENCY` (default: 4)
- Graceful fallback if individual LLM calls fail

**Phase 3: Sequential Database Writes**
- Sequential writes to avoid SQLite corruption
- Writes to both USearch (vectors) and Tantivy (full-text)

**Configuration**:
```bash
ADD_MAX_CONCURRENCY=4   # Max parallel tasks (default: 4)
```

**Performance Impact**: 5-8x speedup for batch adds compared to sequential processing.

### Async Embedding Batching

The `LangchainQwenEmbeddings` class supports batch embedding with configurable batch size:

```python
# Batch embeddings for multiple texts
embeddings = await embedder.aembed_documents(texts)  # Batched by EMBEDDING_BATCH_SIZE
```

**Configuration**:
```bash
EMBEDDING_BATCH_SIZE=512  # Texts per batch (default: 512)
```

**Performance Impact**: ~100x fewer API calls for large batches.

### Query Embedding LRU Cache

Repeated query embeddings are cached to avoid redundant API calls:

```bash
EMBEDDING_CACHE_SIZE=100  # Max cached embeddings (default: 100)
```

### Direct Database Lookup for Deduplication

When hybrid search is disabled (`ENABLE_HYBRID_SEARCH=false`), duplicate detection uses O(log n) database lookup via `MessageStore.get_id_by_message()` instead of expensive embedding API calls.

### Parallel Smart Replacement

Smart replacement checks run in parallel when multiple candidates exist:

```
New Memory → [Semantic Search] → [Parallel LLM Checks] → Best Match
             Find top N similar   Concurrent confidence scoring
```

**Configuration**:
```bash
SMART_REPLACE_CANDIDATE_LIMIT=3  # Max candidates to check (default: 3)
ADD_MAX_CONCURRENCY=4            # Parallel LLM calls (default: 4)
```

### Tantivy Soft-Delete

Tantivy uses O(1) tombstone marking instead of O(n) index rebuild for deletions:

```
Delete Request → [Mark as Deleted] → [Background Compaction]
                 O(1) tombstone       Async cleanup
```

**Benefits**:
- Delete operations complete in < 100ms instead of 50+ seconds for large indexes
- Deleted documents filtered at query time
- Background compaction removes tombstones when threshold is reached

**Configuration**:
```bash
TANTIVY_SOFT_DELETE_ENABLED=true       # Enable soft-delete (default: true)
TANTIVY_COMPACTION_THRESHOLD_RATIO=0.2 # Compact when tombstones > 20% of docs
TANTIVY_COMPACTION_MAX_TOMBSTONES=10000 # Force compaction above this count
TANTIVY_TOMBSTONE_TTL_DAYS=7           # Days before tombstones eligible for removal
```

### Adaptive Overfetch

Search uses adaptive overfetch multiplier based on index size for better RRF fusion:

```
Small index (≤100 docs)   → 3.0x multiplier (more candidates)
Large index (≥10k docs)   → 1.5x multiplier (fewer candidates)
Medium index              → Logarithmic interpolation
```

**Configuration**:
```bash
OVERFETCH_ADAPTIVE=true       # Enable adaptive overfetch (default: true)
OVERFETCH_MIN_MULTIPLIER=1.5  # For large indexes
OVERFETCH_MAX_MULTIPLIER=3.0  # For small indexes
```

### Eager Initialization

Pre-warm all engines at startup to avoid cold-start latency:

```bash
EAGER_INITIALIZATION=true  # Pre-warm USearch + Tantivy + embeddings (default: true)
```

When disabled, engines initialize lazily on first use (faster startup, slower first request).
