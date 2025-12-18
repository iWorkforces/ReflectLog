# openmemories/application/config/

This directory contains configuration management and system prompts for OpenMemoriesMCP.

## Structure

```
config/
├── __init__.py          # Configuration exports
├── settings.py          # Config dataclass from environment variables
└── prompts.py           # MCP_INSTRUCTIONS, SCORING_PROMPT
```

## Purpose

The `config/` module centralizes:
- Environment variable management via dataclass
- Configuration validation and defaults
- LLM prompt templates (MCP instructions, scoring)
- Transport mode enumeration

## Configuration Architecture

### Config Dataclass (`settings.py`)

The `Config` class loads all settings from environment variables with sensible defaults:

```python
@dataclass
class Config:
    """Centralized configuration from environment variables."""

    # Required - will fail if not set
    project_id: str

    # OpenRouter API
    openrouter_api_key: str | None

    # Transport settings
    transport: str = "stdio"
    port: int = 9104
    host: str = "127.0.0.1"
    path: str = "/mcp"

    # LLM configuration
    llm_model: str = "x-ai/grok-4.1-fast"
    embedding_model: str = "openai/text-embedding-3-large"
    embedder_provider: str = "openai"  # "openai" or "langchain"
    embedding_dims: int = 3072
    qwen_embedding_dims: int = 4096

    # Embedding performance settings
    embedding_batch_size: int = 512        # Texts per API request for async batching
    embedding_max_concurrent_batches: int = 4  # Max parallel batch requests
    embedding_cache_enabled: bool = True   # Enable LRU cache for query embeddings
    embedding_cache_size: int = 100        # Max cached embeddings

    # Search configuration
    search_limit: int = 5
    remove_search_limit: int = 5

    # Reranker engine selection
    reranker_engine: str = "llm"  # "llm", "cross_encoder", or "none"

    # LLM reranking settings (used when reranker_engine="llm")
    search_score_threshold: float = 0.5
    rerank_max_concurrency: int = 10  # Increased from 5 for better parallelism

    # Cross-encoder reranking settings (used when reranker_engine="cross_encoder")
    # Uses FlagEmbedding's FlagReranker (optimized for BGE reranker models)
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_top_k: int = 20
    cross_encoder_device: str = "cpu"
    cross_encoder_batch_size: int = 32
    cross_encoder_score_threshold: float = 0.0
    cross_encoder_use_fp16: bool = True      # FP16 for faster inference
    cross_encoder_normalize: bool = True      # Sigmoid normalization to 0-1
    cross_encoder_max_length: int = 512       # Max token length for pairs

    # Unified reranker settings (apply to both LLM and CrossEncoder)
    reranker_min_results: int = 0            # Safety net: min results to return (0 = disabled)
    reranker_batch_normalize: bool = True    # Enable batch min-max normalization

    remove_score_threshold: float = 0.9

    # Hybrid search configuration
    enable_hybrid_search: bool = True
    fusion_rrf_k: int = 60  # RRF k parameter (lower = more weight to top ranks)
    fusion_ranking_threshold: float = 0.8  # Min normalized RRF score to keep
    tantivy_index_path_template: str = "indexes/{project_id}/tantivy"
    usearch_index_path_template: str = "indexes/{project_id}/usearch"

    # Overfetch settings (adaptive multiplier for better fusion)
    overfetch_multiplier: int = 3          # Base multiplier
    overfetch_adaptive: bool = True        # Enable adaptive overfetch based on index size
    overfetch_min_multiplier: float = 1.5  # Multiplier for large indexes (≥10k docs)
    overfetch_max_multiplier: float = 3.0  # Multiplier for small indexes (≤100 docs)

    # Tantivy soft-delete settings (O(1) delete vs O(n) rebuild)
    tantivy_soft_delete_enabled: bool = True        # Use tombstone marking
    tantivy_compaction_threshold_ratio: float = 0.2 # Compact when tombstones > 20%
    tantivy_compaction_max_tombstones: int = 10000  # Force compaction above this
    tantivy_tombstone_ttl_days: int = 7             # Days before tombstone removal

    # USearch exact search settings
    usearch_exact_search: bool = True      # Force exact brute-force search
    usearch_exact_search_threshold: int = 10000  # Auto-switch threshold

    # Message handling
    max_message_length: int = 30720
    min_message_length: int = 1
    deduplicate_messages: bool = True

    # Smart replacement configuration
    enable_smart_replace: bool = True         # Enable smart memory replacement
    smart_replace_threshold: float = 0.7      # Min LLM confidence to trigger replacement
    smart_replace_min_similarity: float = 0.5 # Min embedding similarity to trigger LLM check
    smart_replace_candidate_limit: int = 3    # Max candidates to check for replacement
    smart_replace_archive_ttl_days: int = 30  # Days to keep archived memories (0 = permanent)
    smart_replace_max_retries: int = 3        # Max LLM call retries with exponential backoff
    smart_replace_retry_delay: float = 1.0    # Base delay in seconds for exponential backoff

    # Concurrency settings
    add_max_concurrency: int = 4             # Max concurrent message additions (Phase 1)

    # Initialization settings
    eager_initialization: bool = True        # Pre-warm engines during MemoryManager init

    # LLM inference
    enable_llm_infer: bool = False

    # Tool selection
    allowed_tools: list[str] | None = None  # None means all tools

    # Logging
    log_level: str = "INFO"
    log_search_results_verbose: bool = False  # Log individual search results
    log_search_result_limit: int = 3          # Max results to log when verbose
```

### Configuration Priority

1. **Environment Variables**: Primary source (loaded via `os.getenv`)
2. **Defaults**: Fallback values defined in dataclass

### Singleton Pattern

A global `config` singleton is created on module import:

```python
# settings.py (bottom)
config = Config()  # Singleton instance
```

Usage throughout the application:

```python
from openmemories.application.config import config

# Access configuration
print(config.project_id)
print(config.search_limit)
```

## Prompt Templates (`prompts.py`)

### MCP_INSTRUCTIONS

Server instructions for Claude clients describing available tools:

```python
MCP_INSTRUCTIONS = """
OpenMemoriesMCP MCP Server - Project-based memory storage for Claude Code.

Available Tools:
    • add(messages: list[str])
      Add messages with semantic embeddings. Empty lists are no-op.
      Messages must be 1-30720 characters, non-whitespace.

    • get_all() -> list[str]
      Retrieve all stored messages.

    • search(query: str) -> list[str]
      Semantic search with optional AI reranking.

    • remove(messages: list[str])
      Remove messages using exact string matching (case-sensitive).

Note: Data persists in USearch index until manually cleared.
"""
```

### SCORING_PROMPT

The scoring prompt is used for AI reranking during search:

```python
SCORING_PROMPT = """You are a relevance scoring system. Score how relevant a document is to a query.

CRITICAL OUTPUT REQUIREMENTS:
Your output MUST be a SINGLE NUMBER between 0.0 and 1.0 (inclusive).
• VALID range: 0.0 ≤ score ≤ 1.0
• Output ONLY the number, NO other text

SCORING SCALE:
• 1.0   = Perfect match - Document directly answers the query
• 0.9   = Very good match - Has the specific information needed
• 0.7   = Good match - Related with partial coverage
• 0.5   = Moderate match - Same domain, different focus
• 0.3   = Weak match - Minimal overlap with query
• 0.0   = No match - Completely unrelated

Query: "{query}"
Document: "{document}"
"""
```

## Environment Variable Reference

### Required Variables

| Variable | Type | Description |
|----------|------|-------------|
| `PROJECT_ID` | str | Unique project identifier for index naming |
| `OPENROUTER_API_KEY` | str | OpenRouter API credentials |

### Optional Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MCP_TRANSPORT` | str | `stdio` | Transport protocol |
| `MCP_PORT` | int | `9104` | Server port |
| `MCP_HOST` | str | `127.0.0.1` | Server host |
| `MCP_PATH` | str | `/mcp` | Server path |
| `LLM_MODEL` | str | `x-ai/grok-4.1-fast` | LLM for reranking/inference |
| `EMBEDDING_MODEL` | str | `openai/text-embedding-3-large` | Embedding model |
| `EMBEDDER_PROVIDER` | str | `openai` | `openai` or `langchain` |
| `EMBEDDING_DIMS` | int | `3072` | OpenAI embedding dimensions |
| `QWEN_EMBEDDING_DIMS` | int | `4096` | Qwen embedding dimensions |
| `SEARCH_LIMIT` | int | `5` | Maximum search results |
| `RERANKER_ENGINE` | str | `llm` | Reranker: `llm`, `cross_encoder`, `none` |
| `SEARCH_SCORE_THRESHOLD` | float | `0.5` | Min LLM relevance score |
| `RERANK_MAX_CONCURRENCY` | int | `10` | Max parallel LLM calls |
| `CROSS_ENCODER_MODEL` | str | `BAAI/bge-reranker-v2-m3` | FlagReranker model (BGE reranker) |
| `CROSS_ENCODER_TOP_K` | int | `20` | Max cross-encoder results |
| `CROSS_ENCODER_DEVICE` | str | `cpu` | Device: `cpu`, `cuda`, `mps` |
| `CROSS_ENCODER_BATCH_SIZE` | int | `32` | Batch size for inference |
| `CROSS_ENCODER_SCORE_THRESHOLD` | float | `0.0` | Min cross-encoder score |
| `CROSS_ENCODER_USE_FP16` | bool | `true` | Enable FP16 inference |
| `CROSS_ENCODER_NORMALIZE` | bool | `true` | Normalize scores (sigmoid) |
| `CROSS_ENCODER_MAX_LENGTH` | int | `512` | Max token length for pairs |
| `RERANKER_MIN_RESULTS` | int | `0` | Safety net: min results (0 = disabled) |
| `RERANKER_BATCH_NORMALIZE` | bool | `true` | Enable batch min-max normalization |
| `REMOVE_SEARCH_LIMIT` | int | `5` | Candidates for removal |
| `REMOVE_SCORE_THRESHOLD` | float | `0.9` | Min score for remove candidates |
| `ENABLE_HYBRID_SEARCH` | bool | `true` | Enable Tantivy full-text |
| `FUSION_RRF_K` | int | `60` | RRF k parameter (lower = more weight to top ranks) |
| `FUSION_RANKING_THRESHOLD` | float | `0.8` | Min RRF score to keep after fusion |
| `TANTIVY_INDEX_PATH_TEMPLATE` | str | `indexes/{project_id}/tantivy` | Tantivy index path |
| `USEARCH_INDEX_PATH_TEMPLATE` | str | `indexes/{project_id}/usearch` | USearch index path |
| `MAX_MESSAGE_LENGTH` | int | `30720` | Max message length |
| `MIN_MESSAGE_LENGTH` | int | `1` | Min message length |
| `DEDUPLICATE_MESSAGES` | bool | `true` | Skip duplicate messages |
| `ENABLE_LLM_INFER` | bool | `false` | Enable LLM message processing |
| `ALLOWED_TOOLS` | str | `*` | Comma-separated tool list |
| `LOG_LEVEL` | str | `INFO` | Logging verbosity |
| `ENABLE_SMART_REPLACE` | bool | `true` | Enable smart memory replacement |
| `SMART_REPLACE_THRESHOLD` | float | `0.7` | Min LLM confidence for replacement |
| `SMART_REPLACE_MIN_SIMILARITY` | float | `0.5` | Min embedding similarity for LLM check |
| `SMART_REPLACE_CANDIDATE_LIMIT` | int | `3` | Max candidates to check |
| `SMART_REPLACE_ARCHIVE_TTL_DAYS` | int | `30` | Days to keep archived memories |
| `SMART_REPLACE_MAX_RETRIES` | int | `3` | Max LLM call retries |
| `SMART_REPLACE_RETRY_DELAY` | float | `1.0` | Base retry delay in seconds |
| `EMBEDDING_BATCH_SIZE` | int | `512` | Texts per API request for async batching |
| `EMBEDDING_MAX_CONCURRENT_BATCHES` | int | `4` | Max parallel batch requests |
| `EMBEDDING_CACHE_ENABLED` | bool | `true` | Enable LRU cache for query embeddings |
| `EMBEDDING_CACHE_SIZE` | int | `100` | Max cached embeddings |
| `OVERFETCH_ADAPTIVE` | bool | `true` | Enable adaptive overfetch based on index size |
| `OVERFETCH_MIN_MULTIPLIER` | float | `1.5` | Multiplier for large indexes (≥10k) |
| `OVERFETCH_MAX_MULTIPLIER` | float | `3.0` | Multiplier for small indexes (≤100) |
| `TANTIVY_SOFT_DELETE_ENABLED` | bool | `true` | O(1) tombstone marking vs O(n) rebuild |
| `TANTIVY_COMPACTION_THRESHOLD_RATIO` | float | `0.2` | Compact when tombstones > 20% |
| `TANTIVY_COMPACTION_MAX_TOMBSTONES` | int | `10000` | Force compaction above this count |
| `TANTIVY_TOMBSTONE_TTL_DAYS` | int | `7` | Days before tombstone removal |
| `USEARCH_EXACT_SEARCH` | bool | `true` | Force exact brute-force search |
| `USEARCH_EXACT_SEARCH_THRESHOLD` | int | `10000` | Auto-switch to exact when index < threshold |
| `ADD_MAX_CONCURRENCY` | int | `4` | Max concurrent message additions |
| `EAGER_INITIALIZATION` | bool | `true` | Pre-warm engines during startup |

## TransportMode Enum

```python
class TransportMode(str, Enum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"
```

## Usage Examples

### Accessing Configuration

```python
from openmemories.application.config import config

# Check reranker engine settings
if config.reranker_engine == "llm":
    print(f"LLM reranking enabled with threshold: {config.search_score_threshold}")
elif config.reranker_engine == "cross_encoder":
    print(f"CrossEncoder reranking with model: {config.cross_encoder_model}")
else:
    print("Reranking disabled")

# Get index paths
tantivy_path = config.tantivy_index_path_template.format(project_id=config.project_id)
usearch_path = config.usearch_index_path_template.format(project_id=config.project_id)
```

### Generating USearch Configuration

```python
from openmemories.application.config import config
from openmemories.infrastructure import USearchConfig

def get_usearch_config() -> USearchConfig:
    """Generate USearch configuration from app config."""
    return USearchConfig(
        project_id=config.project_id,
        index_path=f"indexes/{config.project_id}/usearch/vectors.usearch",
        db_path=f"indexes/{config.project_id}/usearch/messages.db",
        embedding_dims=config.qwen_embedding_dims,
        metric="cos",  # Cosine similarity
    )
```

## Configuration Testing

### Unit Tests

```python
class TestConfig:
    def test_default_values(self):
        """Configuration should have sensible defaults."""
        # Mock required env vars
        with patch.dict(os.environ, {"PROJECT_ID": "test", "OPENROUTER_API_KEY": "key"}):
            cfg = Config.from_environment()
            assert cfg.search_limit == 5
            assert cfg.reranker_engine == "llm"
            assert cfg.cross_encoder_model == "BAAI/bge-reranker-v2-m3"

    def test_environment_override(self):
        """Environment variables should override defaults."""
        with patch.dict(os.environ, {
            "PROJECT_ID": "test",
            "OPENROUTER_API_KEY": "key",
            "RERANKER_ENGINE": "cross_encoder"
        }):
            cfg = Config.from_environment()
            assert cfg.reranker_engine == "cross_encoder"
```

## Best Practices

1. **Environment Variables**: Use environment variables for all configurable values
2. **Sensible Defaults**: Provide reasonable defaults for optional settings
3. **Type Safety**: Use proper types for configuration values
4. **Singleton Access**: Use `from config import config` for consistent access
5. **Security**: Never log sensitive configuration values (API keys)
6. **Validation**: Validate configuration at startup (PROJECT_ID required)
