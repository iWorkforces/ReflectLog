# Agent Guidelines for reflectlog/application/tools/

This directory contains the modular MCP tool implementations that provide the external interface to the memory system. Each tool file implements one MCP tool with input validation, structured logging, error handling, and integration with MemoryManager.

## Directory Structure

```
tools/
├── __init__.py          # Package exports (BaseTool, AddTool, etc.)
├── base.py              # BaseTool abstract base class
├── add.py               # AddTool - add(messages) implementation
├── get_all.py           # GetAllTool - get_all() implementation
├── health_check.py      # HealthCheckTool - health_check() implementation
├── search.py            # SearchTool - search(query) implementation
└── remove.py            # RemoveTool - remove(messages) implementation
```

## Core Responsibilities

### Tool Architecture

Each tool inherits from `BaseTool` and implements:

- **Input validation and sanitization**: Validate tool arguments before processing
- **Structured logging**: Log operations with consistent context
- **Error handling**: Graceful degradation with meaningful error messages
- **MemoryManager integration**: Call appropriate memory operations
- **Response formatting**: Return consistent response formats

### BaseTool Abstract Class

All tools inherit from `BaseTool`:

```python
class BaseTool(ABC):
    '''Abstract base class for MCP tools.'''

    def __init__(
        self,
        config: Config,
        memory_manager: MemoryManager,
        logger: StructuredLogger,
    ):
        self.config = config
        self.memory = memory_manager
        self.logger = logger

    @abstractmethod
    def get_name(self) -> str:
        '''Get the tool name for registration.'''
        pass

    @abstractmethod
    def get_description(self) -> str:
        '''Get the tool description for the MCP server.'''
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        '''Get the tool parameters schema.'''
        pass

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        '''Execute the tool logic.'''
        pass
```

## Tool Implementations

### AddTool (add.py)

Stores messages with semantic embeddings using the 3-phase add pipeline.

**Parameters:**
```python
{
    "messages": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of messages to store",
        "minItems": 1,
    }
}
```

**Response:**
```python
{
    "stored_count": int,      # Number of messages stored
    "skipped_count": int,     # Number of duplicates skipped
    "replaced_count": int,    # Number of memories replaced
    "replacements": list[dict],  # Details of each replacement
}
```

**Implementation Pattern:**
```python
class AddTool(BaseTool):
    async def run(self, messages: list[str]) -> dict:
        self.logger.info("Adding messages", extra={"count": len(messages)})

        # Validate input
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Execute add operation
        result = await self.memory.add_messages_async(messages)

        self.logger.info(
            "Messages added",
            extra={
                "stored": result.stored_count,
                "skipped": result.skipped_count,
                "replaced": result.replaced_count,
            }
        )

        return {
            "stored_count": result.stored_count,
            "skipped_count": result.skipped_count,
            "replaced_count": result.replaced_count,
            "replacements": [
                {
                    "old_memory": r.old_memory,
                    "new_memory": r.new_memory,
                    "confidence": r.confidence,
                    "reason": r.reason,
                }
                for r in result.replacements
            ],
        }
```

### GetAllTool (get_all.py)

Retrieves all stored messages from the memory system.

**Parameters:** None (empty object)

**Response:**
```python
{
    "messages": list[str],  # All stored messages
    "count": int,           # Number of messages
}
```

**Implementation Pattern:**
```python
class GetAllTool(BaseTool):
    async def run(self) -> dict:
        messages = self.memory.get_all()

        self.logger.info(
            "Retrieved all messages",
            extra={"count": len(messages)}
        )

        return {
            "messages": messages,
            "count": len(messages),
        }
```

### SearchTool (search.py)

Performs hybrid semantic + full-text search with RRF fusion and optional reranking.

**Parameters:**
```python
{
    "query": {
        "type": "string",
        "description": "Search query",
        "minLength": 1,
    },
    "limit": {
        "type": "integer",
        "description": "Maximum results to return",
        "default": 5,
        "minimum": 1,
        "maximum": 20,
    }
}
```

**Response:**
```python
{
    "results": list[str],   # Matching messages
    "query": str,           # Echoed query
    "count": int,           # Number of results
}
```

**Implementation Pattern:**
```python
class SearchTool(BaseTool):
    async def run(self, query: str, limit: int = 5) -> dict:
        self.logger.info(
            "Searching memories",
            extra={"query": query[:100], "limit": limit}
        )

        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Execute search
        results = self.memory.search(query.strip(), limit=limit)

        self.logger.info(
            "Search completed",
            extra={
                "query": query[:100],
                "result_count": len(results),
            }
        )

        return {
            "results": results,
            "query": query,
            "count": len(results),
        }
```

### RemoveTool (remove.py)

Removes messages by exact match from both search engines.

**Parameters:**
```python
{
    "messages": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Messages to remove",
        "minItems": 1,
    }
}
```

**Response:**
```python
{
    "removed_count": int,   # Number of messages removed
    "not_found_count": int, # Number of messages not found
}
```

**Implementation Pattern:**
```python
class RemoveTool(BaseTool):
    async def run(self, messages: list[str]) -> dict:
        self.logger.info(
            "Removing messages",
            extra={"count": len(messages)}
        )

        removed = 0
        not_found = 0

        for message in messages:
            if self.memory.delete_by_message(message):
                removed += 1
            else:
                not_found += 1

        self.logger.info(
            "Messages removed",
            extra={
                "removed": removed,
                "not_found": not_found,
            }
        )

        return {
            "removed_count": removed,
            "not_found_count": not_found,
        }
```

### HealthCheckTool (health_check.py)

Returns server health status including memory usage and configuration.

**Parameters:** None (empty object)

**Response:**
```python
{
    "status": str,              # "healthy", "degraded", "unhealthy"
    "project_id": str,          # Current project ID
    "message_count": int,       # Total stored messages
    "transport": str,           # Current transport mode
    "components": dict,         # Component health status
}
```

**Implementation Pattern:**
```python
class HealthCheckTool(BaseTool):
    async def run(self) -> dict:
        # Check memory count
        try:
            message_count = len(self.memory.get_all())
        except Exception:
            message_count = -1

        # Determine overall status
        if message_count >= 0:
            status = "healthy"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "project_id": self.config.project_id,
            "message_count": message_count,
            "transport": self.config.transport,
            "components": {
                "usearch": "healthy" if message_count >= 0 else "unhealthy",
                "tantivy": "healthy" if message_count >= 0 else "unhealthy",
            },
        }
```

## Key Patterns

### Input Validation

Always validate inputs before processing:

```python
def _validate_messages(self, messages: list[str]) -> None:
    if not messages:
        raise ValueError("Messages list cannot be empty")

    for i, message in enumerate(messages):
        if not isinstance(message, str):
            raise TypeError(f"Message {i} must be a string")
        if len(message) > self.config.max_message_length:
            raise ValueError(
                f"Message {i} exceeds maximum length "
                f"({len(message)} > {self.config.max_message_length})"
            )
```

### Structured Logging

Use structured logging with consistent context:

```python
self.logger.info(
    "Tool operation completed",
    extra={
        "tool": self.get_name(),
        "project_id": self.config.project_id,
        "duration_ms": duration,
    }
)
```

### Error Handling

Provide meaningful error messages:

```python
try:
    result = await self.memory.add_messages_async(messages)
except MemoryStorageError as e:
    self.logger.error(
        "Failed to store messages",
        extra={"error": str(e)}
    )
    raise ToolExecutionError(
        f"Failed to store messages: {e}"
    ) from e
```

### Response Consistency

Maintain consistent response formats:

```python
# Always include these fields
{
    "success": True,  # or include error field
    # ... tool-specific fields
}
```

## Tool Registration

Tools are registered with the MCP server in `mcp_server.py`:

```python
def register_tools(server: FastMCPServer, memory_manager: MemoryManager) -> None:
    config = Config.from_env()
    logger = create_logger(__name__, config.project_id, config.log_level)

    # Create and register each tool
    server.add_tool(AddTool(config, memory_manager, logger))
    server.add_tool(GetAllTool(config, memory_manager, logger))
    server.add_tool(SearchTool(config, memory_manager, logger))
    server.add_tool(RemoveTool(config, memory_manager, logger))
    server.add_tool(HealthCheckTool(config, memory_manager, logger))
```

## Testing Guidelines

### Unit Tests

- Mock MemoryManager for isolated testing
- Test input validation for each tool
- Verify response formats
- Test error handling paths

### Test Cases

```python
@pytest.fixture
def add_tool(config, memory_manager, logger):
    return AddTool(config, memory_manager, logger)

@pytest.mark.asyncio
async def test_add_tool_stores_messages(add_tool, memory_manager):
    messages = ["test message 1", "test message 2"]
    result = await add_tool.run(messages)

    assert result["stored_count"] == 2
    assert result["skipped_count"] == 0
    assert len(memory_manager.get_all()) == 2

@pytest.mark.asyncio
async def test_add_tool_rejects_empty_messages(add_tool):
    with pytest.raises(ValueError):
        await add_tool.run([])
```

## Dependencies

### Internal Dependencies

- `application/memory/`: MemoryManager for storage operations
- `application/config/`: Config for validation
- `application/utils/logging.py`: StructuredLogger
- `application/exceptions.py`: ToolExecutionError

### External Dependencies

- `fastmcp`: MCP server framework (for tool decorators if used)

## Important Notes

### Idempotency

- AddTool should be idempotent (duplicate messages are skipped)
- RemoveTool should be idempotent (missing messages are counted as not found)

### Response Size

- Limit response size for large result sets
- Use pagination for get_all() if necessary
- Log response sizes for monitoring

### Security

- Never log message content in production
- Sanitize error messages to avoid information leakage
- Validate input lengths to prevent DoS
