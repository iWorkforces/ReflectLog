# ccmemories/application/

This directory contains the application layer - the core business logic for the CCMemoriesMCP server.

## Structure

```
application/
├── __init__.py           # Package exports
├── exceptions.py         # Custom exception hierarchy (CCMemoriesError, etc.)
├── mcp_server.py         # FastMCPServer orchestrator
├── types.py              # Type definitions (ISemanticSearchEngine protocol)
├── config/               # Configuration management
│   ├── __init__.py       # Config exports
│   ├── settings.py       # Config dataclass from environment
│   └── prompts.py        # MCP_INSTRUCTIONS, SCORING_PROMPT, SCORING_PROMPT_WITH_AGE, REPLACEMENT_DETECTION_PROMPT
├── memory/               # Memory management
│   ├── __init__.py       # Memory exports
│   ├── manager.py        # MemoryManager (USearch + Tantivy)
│   ├── protocols.py      # Search engine protocols
│   ├── fusion/           # Hybrid ranking
│   │   ├── __init__.py   # Fusion exports
│   │   ├── base.py       # FusionEngine protocol
│   │   └── ranx_fusion.py # RanxFusionEngine (RRF)
│   └── reranking/        # Score normalization and recency decay utilities
│       ├── __init__.py   # Reranking exports
│       └── normalization.py # Min-max batch normalization and recency decay
├── tools/                # Modular MCP tool implementations
│   ├── __init__.py       # Tool exports
│   ├── base.py           # BaseTool abstract class
│   ├── add.py            # AddTool
│   ├── get_all.py        # GetAllTool
│   ├── health_check.py   # HealthCheckTool
│   ├── search.py         # SearchTool
│   └── remove.py         # RemoveTool
└── utils/                # Utilities
    ├── __init__.py       # Utility exports
    ├── logging.py        # StructuredLogger, format_fusion_score_status
    ├── numba_utils.py    # Numba JIT functions (normalization, distance)
    ├── retry.py          # async_retry_with_backoff decorator
    ├── security.py       # SecretString, redact_dict_secrets
    └── validation.py     # Message validation helpers
```

## Purpose

This layer implements the MCP server using the FastMCP framework, including:
- **Exception hierarchy** (`exceptions.py`) for structured error handling
- Server initialization and configuration
- Memory storage setup with hybrid USearch + Tantivy engines
- Five modular MCP tools: `add`, `get_all`, `search`, `remove`, `health_check`
- Message validation logic
- Structured logging with context
- Security utilities for API key redaction

## Exception Hierarchy

The `exceptions.py` module provides a structured exception hierarchy:

```
CCMemoriesError (base)
├── ConfigurationError     - Invalid or missing configuration
├── ValidationError        - Input validation failures
├── InitializationError    - Engine/client initialization failures
├── StorageError           - Storage operation failures
│   ├── DuplicateError     - Duplicate entry detected
│   └── InconsistentStateError - Dual-engine state mismatch
├── SearchError            - Search operation failures
├── EmbeddingError         - Embedding generation failures
└── RerankerError          - LLM reranking failures
```

Usage:
```python
from ccmemories.application.exceptions import (
    CCMemoriesError,
    ConfigurationError,
    StorageError,
)

try:
    memory_manager.add_messages(messages)
except StorageError as e:
    # Handle storage-specific error
except CCMemoriesError as e:
    # Handle any CCMemoriesMCP error
```

## Key Architecture Decisions

### Modular Tool Design

Tools are implemented as separate classes inheriting from `BaseTool`:

```python
# tools/base.py
class BaseTool(ABC):
    def __init__(self, config: Config, memory_manager: MemoryManager, logger: StructuredLogger):
        self.config = config
        self.memory = memory_manager
        self.logger = logger

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_handler(self) -> Callable: ...

    def log_invocation(self, tool_name: str, **kwargs): ...
    def log_completion(self, tool_name: str, **kwargs): ...
    def log_error(self, tool_name: str, error: Exception, **kwargs): ...
```

### FastMCPServer Class

**Core orchestrator** (`mcp_server.py`):

1. **Tool Registry** (line 14-19):
   ```python
   AVAILABLE_TOOL_CLASSES: Dict[str, Type[BaseTool]] = {
       "add": AddTool,
       "get_all": GetAllTool,
       "search": SearchTool,
       "remove": RemoveTool,
   }
   ```

2. **Initialization** (`__init__`):
   - Creates `StructuredLogger` with project context
   - Initializes `MemoryManager` (hybrid USearch + Tantivy)
   - Creates and registers tool instances

3. **Tool Selection** (`_determine_tool_selection`):
   - Respects `ALLOWED_TOOLS` env var for selective registration
   - Supports flexible token matching (snake_case, kebab-case, etc.)
   - `ALLOWED_TOOLS=all` or `ALLOWED_TOOLS=*` enables all tools

4. **Transport** (`run`):
   - Routes to appropriate FastMCP transport based on config
   - stdio for MCP clients, http/sse/streamable-http for HTTP

## Tool Implementation Pattern

Each tool follows this pattern in its handler:

```python
def get_handler(self):
    def tool_name(params) -> ReturnType:
        """Docstring with Args/Returns/Raises."""
        try:
            # 1. Validation
            if not valid:
                self.log_error("tool_name", ValueError("reason"))
                raise ValueError("reason")

            # 2. Pre-logging
            self.log_invocation("tool_name", param=value)

            # 3. Operation via MemoryManager
            result = self.memory.operation()

            # 4. Post-logging
            self.log_completion("tool_name", result_count=len(result))

            return result

        except Exception as e:
            # 5. Error handling
            self.log_error("tool_name", e)
            raise RuntimeError(f"Failed to execute tool: {e}") from e

    return tool_name
```

## Tool Details

### `AddTool` (`add.py`)

**Purpose**: Store messages with hybrid USearch + Tantivy engines

```python
def add(messages: List[str]) -> None
```

- **Validation**: Uses `validate_messages()` from utils
- **Behavior**: Empty list is no-op (returns without error)
- **Deduplication**: Skips exact matches when `DEDUPLICATE_MESSAGES=true`
- **Dual Storage**: Stores in both USearch (semantic) and Tantivy (full-text)

### `GetAllTool` (`get_all.py`)

**Purpose**: Retrieve all stored messages

```python
def get_all() -> List[str]
```

- **Source of Truth**: USearch engine (SQLite MessageStore)
- **Returns**: Defensive copy via `.copy()`
- **Empty case**: Returns `[]` if no messages stored

### `SearchTool` (`search.py`)

**Purpose**: Hybrid semantic + full-text search

```python
def search(query: Annotated[str, Field(min_length=1)]) -> List[str]
```

- **Pydantic validation**: `min_length=1` enforced on query
- **Hybrid search**: USearch semantic + Tantivy full-text → RRF fusion
- **Score filtering**: Applies threshold when reranking enabled
- **Configurable**: `SEARCH_LIMIT`, `RERANKER_ENGINE`, `SEARCH_SCORE_THRESHOLD`

### `RemoveTool` (`remove.py`)

**Purpose**: Remove messages by exact string matching

```python
def remove(messages: List[str]) -> None
```

- **Strategy**: Tantivy search for candidates + exact match filter
- **Exact matching**: Uses `item["memory"] == message` (case-sensitive)
- **Multiple occurrences**: Deletes ALL exact matches
- **Not found**: Silently ignores (logs but doesn't error)

## Configuration

**Required environment variables**:
- `PROJECT_ID`: Used for index naming (lowercased)
- `OPENROUTER_API_KEY`: OpenRouter API key for LLM/embeddings

**Optional environment variables**:
| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_LIMIT` | 5 | Max search results |
| `RERANKER_ENGINE` | llm | Reranking engine: `llm`, `cross_encoder`, or `none` |
| `SEARCH_SCORE_THRESHOLD` | 0.5 | Min reranking score |
| `REMOVE_SEARCH_LIMIT` | 5 | Candidates for removal |
| `REMOVE_SCORE_THRESHOLD` | 0.9 | Min score for remove candidates |
| `MAX_MESSAGE_LENGTH` | 30720 | Max chars per message |
| `MIN_MESSAGE_LENGTH` | 1 | Min chars per message |
| `ENABLE_LLM_INFER` | false | LLM message transformation |
| `DEDUPLICATE_MESSAGES` | true | Skip exact duplicates |
| `LOG_LEVEL` | INFO | Logging level |
| `ALLOWED_TOOLS` | all | Comma-separated tool list |
| `ENABLE_HYBRID_SEARCH` | true | Enable Tantivy full-text |
| `FUSION_RRF_K` | 60 | RRF fusion constant |
| `LLM_PROVIDER` | anthropic | LLM provider: `openai` or `anthropic` |
| `ENABLE_RECENCY_BOOST` | true | Include memory age in reranking context |
| `RECENCY_DECAY_RATE` | 0.01 | Exponential decay per hour: `exp(-rate * hours_old)` |
| `RERANKER_BATCH_NORMALIZE` | true | Enable batch min-max normalization |
| `RERANKER_MIN_RESULTS` | 0 | Safety net: min results to return (0 = disabled) |

## Logging Strategy

All logging uses structured logging with `extra` parameter:

```python
self.logger.info(
    "Tool invoked",
    extra={
        "tool": "add",
        "count": len(messages),
        "project_id": self.config.project_id,
    }
)
```

**Log levels**:
- `INFO`: Tool invocations, success messages, search results
- `WARNING`: Non-critical issues (e.g., unknown tool tokens)
- `ERROR`: Validation failures, operation failures

## Adding a New Tool

1. Create `ccmemories/application/tools/new_tool.py`:
   ```python
   from .base import BaseTool

   class NewTool(BaseTool):
       def get_name(self) -> str:
           return "new_tool"

       def get_handler(self):
           def new_tool(param: str) -> str:
               """Comprehensive docstring."""
               self.log_invocation("new_tool", param=param)
               result = self.memory.some_operation(param)
               self.log_completion("new_tool", result=result)
               return result
           return new_tool
   ```

2. Export in `tools/__init__.py`:
   ```python
   from .new_tool import NewTool
   __all__ = [..., "NewTool"]
   ```

3. Register in `mcp_server.py:AVAILABLE_TOOL_CLASSES`:
   ```python
   AVAILABLE_TOOL_CLASSES = {
       ...,
       "new_tool": NewTool,
   }
   ```

4. Update `config/prompts.py:MCP_INSTRUCTIONS`

5. Add tests in `tests/unit/application/test_new_tool.py`

## Testing Considerations

When testing tools:
- Mock `MemoryManager` instance
- Mock `StructuredLogger`
- Test validation edge cases (empty, too long, too short, whitespace)
- Test error handling (exceptions from MemoryManager)
- Verify defensive copying (for `get_all`)
- Verify exact matching (for `remove`)

## Performance Notes

- **Index persistence**: Stored in `indexes/{project_id}/usearch/` and `indexes/{project_id}/tantivy/`
- **Reranking overhead**: Set `RERANKER_ENGINE=none` to disable AI reranking
- **Search limits**: Lower `SEARCH_LIMIT` for faster searches
- **Batch operations**: `add()` accepts lists - batch messages when possible
- **RRF overfetch**: Hybrid search fetches 3x limit for better fusion quality


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Dec 21, 2025

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #177 | 4:12 PM | 🔵 | Complete CLAUDE.md Documentation Inventory | ~680 |
| #662 | 4:09 PM | ✅ | CLAUDE.md Files Aligned | ~196 |

### Dec 26, 2025

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #459 | 11:27 AM | ✅ | Updated Application Layer Structure Documentation | ~196 |
| #388 | 11:05 AM | 🟣 | HealthCheckTool Registered with Server | ~185 |
| #387 | " | ✅ | HealthCheckTool Import Added to Server | ~118 |
| #386 | 11:04 AM | 🔵 | MCP Server Tool Registry | ~184 |
| #363 | 10:50 AM | 🔵 | Exception Hierarchy Analysis | ~207 |
| #329 | 10:44 AM | 🔵 | Codebase Analysis Summary | ~361 |
| #326 | 10:43 AM | 🔵 | Exception Hierarchy Analysis | ~355 |
| #315 | 10:40 AM | 🔵 | Code Quality Analysis Complete | ~339 |
| #310 | 10:39 AM | 🔵 | Architecture Analysis Complete | ~288 |
| #305 | 10:33 AM | 🔵 | Structured exception hierarchy for CCMemoriesMCP | ~276 |
| #300 | 10:32 AM | 🔵 | FastMCP server orchestrator implementation | ~246 |
| #297 | 10:31 AM | 🔵 | Type definitions and Clean Architecture interfaces | ~226 |

### Jan 8, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #749 | 8:57 AM | 🔵 | Memory Manager Initialization Critical Path | ~65 |
| #746 | " | 🔵 | Server Architecture Analysis | ~158 |
| #722 | 8:52 AM | ✅ | Tool Count Updated | ~212 |

### Jan 12, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #803 | 10:14 PM | 🔵 | 项目依赖范围最终确认 | ~128 |
</claude-mem-context>