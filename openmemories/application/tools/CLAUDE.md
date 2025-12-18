# ccmemories/application/tools/

This directory contains the modular MCP tool implementations that provide the external interface to the memory system.

## Structure

```
tools/
├── __init__.py          # Package exports (BaseTool, AddTool, etc.)
├── base.py              # BaseTool abstract base class
├── add.py               # AddTool - add(messages) implementation
├── get_all.py           # GetAllTool - get_all() implementation
├── search.py            # SearchTool - search(query) implementation
└── remove.py            # RemoveTool - remove(messages) implementation
```

## Purpose

Each tool file implements one MCP tool with:
- Input validation and sanitization
- Structured logging with context
- Error handling and graceful degradation
- Integration with MemoryManager
- Consistent response formatting

## Tool Architecture

### BaseTool Abstract Class (`base.py`)

All tools inherit from `BaseTool`:

```python
class BaseTool(ABC):
    """Abstract base class for MCP tools."""

    def __init__(
        self,
        config: Config,
        memory_manager: MemoryManager,
        logger: StructuredLogger
    ):
        self.config = config
        self.memory = memory_manager
        self.logger = logger

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name for registration."""
        pass

    @abstractmethod
    def get_handler(self) -> Callable:
        """Get the tool handler function."""
        pass

    def log_invocation(self, tool_name: str, **kwargs: Any) -> None:
        """Log tool invocation with context."""
        self.logger.info(f"Tool '{tool_name}' invoked", extra={"tool": tool_name, **kwargs})

    def log_completion(self, tool_name: str, **kwargs: Any) -> None:
        """Log tool completion with results."""
        self.logger.info(f"Tool '{tool_name}' completed successfully", extra={"tool": tool_name, **kwargs})

    def log_error(self, tool_name: str, error: Exception, **kwargs: Any) -> None:
        """Log tool error with context."""
        self.logger.error(f"Tool '{tool_name}' failed: {error}", extra={"tool": tool_name, "error": str(error), **kwargs})
```

### Tool Implementation Pattern

Each tool's `get_handler()` returns a closure that captures `self`:

```python
def get_handler(self):
    def tool_name(params) -> ReturnType:
        """Comprehensive docstring with Args/Returns/Raises/Examples."""
        try:
            # 1. Validation (if needed)
            if not valid:
                self.log_error("tool_name", ValueError("reason"))
                raise ValueError("reason")

            # 2. Pre-logging
            self.log_invocation("tool_name", param=value)

            # 3. Operation via self.memory
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

## Individual Tools

### AddTool (`add.py`)

**Purpose**: Store messages with semantic embeddings, full-text indexing, and smart replacement detection

```python
def add(messages: List[str], dry_run: bool = False) -> AddResult
```

**Key Features**:
- Validates message length (1-30720 chars)
- Rejects whitespace-only messages
- Empty list is no-op (not an error)
- Deduplication via MemoryManager
- Dual storage: USearch + Tantivy
- **Smart replacement**: Detects when new memories update/replace existing ones
- **dry_run mode**: Test replacement detection without modifying storage

**Return Type** (`AddResult`):
```python
@dataclass
class AddResult:
    stored_count: int                    # Number of new messages stored
    skipped_count: int                   # Number of duplicates skipped
    replaced_count: int                  # Number of memories replaced
    replacements: List[ReplacementInfo]  # Details of each replacement

@dataclass
class ReplacementInfo:
    old_memory: str    # The memory that was replaced
    new_memory: str    # The new memory that replaced it
    confidence: float  # LLM confidence score (0.0-1.0)
    reason: str        # LLM explanation for replacement
```

**Implementation**:
```python
def get_handler(self):
    async def add(messages: List[str], dry_run: bool = False) -> AddResult:
        if not messages:
            self.log_invocation("add", count=0)
            return AddResult(stored_count=0, skipped_count=0, replaced_count=0, replacements=[])

        is_valid, error_msg = validate_messages(
            messages, self.config.min_message_length, self.config.max_message_length
        )
        if not is_valid:
            raise ValueError(f"Invalid message: {error_msg}")

        self.log_invocation("add", count=len(messages), dry_run=dry_run)

        result = await self.memory.add_messages_async(messages, dry_run=dry_run)

        self.log_completion(
            "add",
            requested=len(messages),
            stored=result.stored_count,
            replaced=result.replaced_count,
            dry_run=dry_run,
        )

        return result

    return add
```

### GetAllTool (`get_all.py`)

**Purpose**: Retrieve all stored messages

```python
def get_all() -> List[str]
```

**Key Features**:
- Returns USearch results as source of truth
- Defensive copying (new list each time)
- Empty list if no messages stored
- No validation required

**Implementation**:
```python
def get_handler(self):
    def get_all() -> List[str]:
        self.log_invocation("get_all")

        messages = self.memory.get_all()

        for idx, message in enumerate(messages, 1):
            self.logger.info(f"[{idx}/{len(messages)}] Message: {truncate_message(message)}")

        self.log_completion("get_all", count=len(messages))

        return messages.copy()  # Defensive copy

    return get_all
```

### SearchTool (`search.py`)

**Purpose**: Hybrid semantic + full-text search

```python
def search(query: Annotated[str, Field(min_length=1)]) -> List[str]
```

**Key Features**:
- Pydantic validation: `Field(min_length=1)` for query
- Configurable search limit (default: 5)
- Optional AI reranking (default: true)
- Score threshold filtering (default: 0.5)
- RRF fusion for hybrid results

**Implementation**:
```python
def get_handler(self):
    def search(
        query: Annotated[str, Field(min_length=1, description="Search query")]
    ) -> List[str]:
        self.log_invocation("search", query=query)

        results = self.memory.search(
            query,
            limit=self.config.search_limit,
            score_threshold=self.config.search_score_threshold,
        )

        self.log_completion("search", query=query, result_count=len(results))

        return results

    return search
```

### RemoveTool (`remove.py`)

**Purpose**: Remove messages by exact string matching

```python
def remove(messages: List[str]) -> None
```

**Key Features**:
- Uses USearch (source of truth) for finding candidates
- Exact string matching (case-sensitive)
- Removes ALL occurrences of each message
- Silently ignores non-existent messages
- Guarantees finding messages regardless of Tantivy index state

**Implementation**:
```python
def get_handler(self):
    def remove(messages: List[str]) -> None:
        if not messages:
            self.log_invocation("remove", count=0)
            return

        self.log_invocation("remove", requested_count=len(messages))

        actual_removed = 0
        for message in messages:
            removed_count = self._remove_single_message(message)
            actual_removed += removed_count

        self.log_completion("remove", requested=len(messages), removed=actual_removed)

    return remove

def _remove_single_message(self, message: str) -> int:
    """Remove a single message and all its occurrences."""
    # search_for_removal uses USearch (source of truth) for exact matching
    candidates = self.memory.search_for_removal(message)
    exact_matches = [item for item in candidates if item["memory"] == message]

    for match in exact_matches:
        self.memory.delete_by_message(match["memory"])

    return len(exact_matches)
```

## Validation System

### Message Validation (`utils/validation.py`)

```python
def validate_messages(
    messages: List[str],
    min_length: int = 1,
    max_length: int = 30720
) -> Tuple[bool, str]:
    """Validate message lists for add tool."""
    if not isinstance(messages, list):
        return False, "Messages must be a list"

    for i, msg in enumerate(messages):
        if not isinstance(msg, str):
            return False, f"Message {i} is not a string"
        if len(msg) < min_length:
            return False, f"Message {i} is too short"
        if len(msg) > max_length:
            return False, f"Message {i} is too long"
        if not msg.strip():
            return False, f"Message {i} contains only whitespace"

    return True, ""
```

### Pydantic Validation

For search queries:
```python
from pydantic import Field
from typing import Annotated

def search(
    query: Annotated[str, Field(min_length=1, description="Search query")]
) -> List[str]:
    ...
```

## Configuration

### Environment Variables

Tools read configuration from `Config`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_LIMIT` | 5 | Max search results |
| `RERANKER_ENGINE` | llm | Reranking engine: `llm`, `cross_encoder`, `none` |
| `SEARCH_SCORE_THRESHOLD` | 0.5 | Score threshold |
| `REMOVE_SEARCH_LIMIT` | 5 | Candidates for removal |
| `REMOVE_SCORE_THRESHOLD` | 0.9 | Min score for remove candidates |
| `MAX_MESSAGE_LENGTH` | 30720 | Max chars per message |
| `MIN_MESSAGE_LENGTH` | 1 | Min chars per message |
| `ENABLE_SMART_REPLACE` | true | Enable smart memory replacement |
| `SMART_REPLACE_THRESHOLD` | 0.7 | Min LLM confidence for replacement |
| `SMART_REPLACE_MIN_SIMILARITY` | 0.5 | Min embedding similarity for LLM check |
| `SMART_REPLACE_CANDIDATE_LIMIT` | 3 | Max candidates to check |
| `SMART_REPLACE_ARCHIVE_TTL_DAYS` | 30 | Days to keep archived memories |

### Tool Registration

Tools are registered in `FastMCPServer`:

```python
# mcp_server.py
AVAILABLE_TOOL_CLASSES: Dict[str, Type[BaseTool]] = {
    "add": AddTool,
    "get_all": GetAllTool,
    "search": SearchTool,
    "remove": RemoveTool,
}

# During initialization
for tool_name in selected_names:
    tool_class = AVAILABLE_TOOL_CLASSES[tool_name]
    tool = tool_class(
        config=self.config,
        memory_manager=self.memory_manager,
        logger=self.logger,
    )
    self.tools.append(tool)

# Registration with FastMCP
for tool in self.tools:
    handler = tool.get_handler()
    self.mcp.tool(handler)
```

## Error Handling

### Exception Types

1. **ValueError**: Invalid input parameters
2. **RuntimeError**: Operation failures (storage, search, etc.)

### Error Responses

All errors are wrapped in `RuntimeError` with descriptive messages:

```python
try:
    result = self.memory.operation()
except Exception as e:
    self.log_error("tool_name", e)
    raise RuntimeError(f"Failed to execute tool: {e}") from e
```

## Logging Strategy

### Structured Logging

All tools use consistent logging format:

```python
self.logger.info(
    "Tool operation",
    extra={
        "tool": "tool_name",
        "project_id": self.config.project_id,
        "count": len(messages),
    }
)
```

### Log Levels

- **INFO**: Tool invocations, successes, search counts
- **ERROR**: Validation failures, operation failures

## Testing

### Unit Test Structure

```python
class TestAddTool:
    def test_add_valid_messages(self, mock_memory_manager):
        # Test successful addition

    def test_add_invalid_messages(self, mock_memory_manager):
        # Test validation failures

    def test_add_empty_list(self, mock_memory_manager):
        # Test no-op behavior

class TestSearchTool:
    def test_search_hybrid_results(self, mock_memory_manager):
        # Test RRF fusion

    def test_search_with_reranking(self, mock_memory_manager):
        # Test AI reranking
```

### Mocking Strategy

- Mock `MemoryManager` for all tool tests
- Mock `StructuredLogger` to verify logging
- Use fixtures for common test setup

## Adding a New Tool

1. Create `ccmemories/application/tools/new_tool.py`:
   ```python
   from .base import BaseTool

   class NewTool(BaseTool):
       def get_name(self) -> str:
           return "new_tool"

       def get_handler(self):
           def new_tool(param: str) -> str:
               """Docstring with Args/Returns/Raises/Examples."""
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

3. Register in `mcp_server.py:AVAILABLE_TOOL_CLASSES`

4. Update `config/prompts.py:MCP_INSTRUCTIONS`

5. Add tests in `tests/unit/application/test_new_tool.py`

## Best Practices

1. **Validation First**: Always validate inputs before operations
2. **Structured Logging**: Include context in all logs
3. **Error Wrapping**: Wrap specific exceptions in RuntimeError
4. **Defensive Copying**: Return new collections, don't expose internal state
5. **Graceful Degradation**: Handle engine failures gracefully
6. **Consistent APIs**: Follow established patterns across all tools
