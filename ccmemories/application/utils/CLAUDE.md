# ccmemories/application/utils/

This directory contains utility functions and classes that support the core memory management system.

## Structure

```
utils/
├── __init__.py          # Package exports
├── logging.py           # Structured logging utilities
├── numba_utils.py       # Numba JIT-compiled numerical utilities
├── retry.py             # Retry decorator with exponential backoff
├── security.py          # Secret redaction and secure logging
└── validation.py        # Input validation helpers
```

## Purpose

Utility modules provide:
- Consistent logging across the application
- Shared validation logic
- Security utilities for API key protection
- Type conversion helpers
- Error handling utilities
- Retry logic with exponential backoff

## Logging Utilities (`logging.py`)

### StructuredLogger Class

```python
class StructuredLogger:
    """Provides structured logging with consistent context."""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)

    def info(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log info level with structured context."""
        self.logger.info(message, **(extra or {}))

    def warning(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log warning level with structured context."""
        self.logger.warning(message, **(extra or {}))

    def error(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log error level with structured context."""
        self.logger.error(message, **(extra or {}))

    def debug(self, message: str, extra: Dict[str, Any] = None) -> None:
        """Log debug level with structured context."""
        self.logger.debug(message, **(extra or {}))
```

### Log Context Helpers

```python
def get_base_log_context(project_id: str, tool: str = None) -> Dict[str, Any]:
    """Get base context for all logs."""
    context = {
        "timestamp": datetime.utcnow().isoformat(),
        "project_id": project_id,
    }
    if tool:
        context["tool"] = tool
    return context

def format_score_status(score: float, threshold: float) -> Tuple[str, str]:
    """Format score status for logging."""
    if score >= threshold:
        status = "[KEEP]"
        interpretation = "Relevant"
    else:
        status = "[FILTER]"
        interpretation = "Not relevant"

    return status, interpretation
```

### Performance Timing

```python
import time
from contextlib import contextmanager

@contextmanager
def log_operation_timing(operation_name: str, logger: StructuredLogger, extra: Dict[str, Any] = None):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        log_data = (extra or {}).copy()
        log_data.update({
            "operation": operation_name,
            "duration_ms": round(duration, 2)
        })
        logger.info(f"Operation {operation_name} completed", extra=log_data)
```

## Numba Utilities (`numba_utils.py`)

JIT-compiled numerical utilities for performance-critical operations using numba.

### Key Functions

```python
def normalize_scores_minmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize scores to 0-1 range using min-max normalization."""

def filter_scores_by_threshold(
    scores: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Filter scores by threshold and return indices and filtered scores."""

def compute_rrf_scores_batch(
    ranks_matrix: NDArray[np.int64],
    k: int = 60,
) -> NDArray[np.float64]:
    """Compute RRF scores for multiple documents across multiple rankings."""

def distance_to_similarity_cosine(
    distances: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Convert cosine distances to similarity scores."""

def warmup_numba_functions() -> None:
    """Pre-compile all numba JIT functions to avoid first-call latency."""
```

### Performance Characteristics

- **JIT Compilation**: Functions are compiled on first call with caching enabled
- **Caching**: Subsequent calls use cached machine code for near-native performance
- **Parallel Processing**: `compute_rrf_scores_batch` uses `prange` for parallel execution
- **In-place Operations**: Internal functions use in-place updates for efficiency

### Usage Pattern

```python
from ccmemories.application.utils.numba_utils import (
    normalize_scores_minmax,
    filter_scores_by_threshold,
    compute_rrf_scores_batch,
    warmup_numba_functions,
)

# Warm up during server startup to avoid first-call latency
warmup_numba_functions()

# Min-max normalization
scores = np.array([0.1, 0.5, 0.9])
normalized = normalize_scores_minmax(scores)  # [0.0, 0.5, 1.0]

# Threshold filtering
indices, filtered = filter_scores_by_threshold(scores, 0.6)

# RRF scoring (parallel)
ranks = np.array([[1, 2], [2, 0]], dtype=np.int64)
rrf_scores = compute_rrf_scores_batch(ranks, k=60)
```

## Validation Utilities (`validation.py`)

### Message Validation

```python
def validate_message_length(message: str, min_length: int = 1, max_length: int = 30720) -> Tuple[bool, str]:
    """Validate individual message length and content."""
    if not isinstance(message, str):
        return False, "Message must be a string"

    if len(message) < min_length:
        return False, f"Message too short (minimum {min_length} characters)"

    if len(message) > max_length:
        return False, f"Message too long (maximum {max_length} characters)"

    if not message.strip():
        return False, "Message cannot contain only whitespace"

    return True, ""

def validate_message_list(messages: List[str], min_length: int = 1, max_length: int = 30720) -> Tuple[bool, str]:
    """Validate list of messages."""
    if not isinstance(messages, list):
        return False, "Messages must be a list"

    for i, message in enumerate(messages):
        is_valid, error = validate_message_length(message, min_length, max_length)
        if not is_valid:
            return False, f"Message {i}: {error}"

    return True, ""
```

### Query Validation

```python
def validate_search_query(query: str, min_length: int = 1) -> Tuple[bool, str]:
    """Validate search query parameters."""
    if not isinstance(query, str):
        return False, "Query must be a string"

    if len(query) < min_length:
        return False, f"Query too short (minimum {min_length} characters)"

    if not query.strip():
        return False, "Query cannot be empty or whitespace only"

    return True, ""

def validate_search_parameters(limit: int = None, score_threshold: float = None) -> Tuple[bool, str]:
    """Validate search parameters."""
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            return False, "Limit must be a positive integer"

    if score_threshold is not None:
        if not isinstance(score_threshold, (int, float)) or not (0.0 <= score_threshold <= 1.0):
            return False, "Score threshold must be between 0.0 and 1.0"

    return True, ""
```

### Configuration Validation

```python
def validate_project_id(project_id: str) -> Tuple[bool, str]:
    """Validate project ID format."""
    if not project_id:
        return False, "Project ID cannot be empty"

    # Check for valid characters (alphanumeric, hyphens, underscores)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        return False, "Project ID can only contain letters, numbers, hyphens, and underscores"

    if len(project_id) > 100:
        return False, "Project ID too long (maximum 100 characters)"

    return True, ""
```

## Formatting Utilities (`formatting.py`)

### Result Formatting

```python
def format_search_results(results: List[str], query: str = None) -> Dict[str, Any]:
    """Format search results for consistent output."""
    return {
        "query": query,
        "result_count": len(results),
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }

def format_memory_entry(message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format memory entry with metadata."""
    entry = {
        "message": message,
        "created_at": datetime.utcnow().isoformat(),
        "message_length": len(message),
        "word_count": len(message.split())
    }

    if metadata:
        entry.update(metadata)

    return entry
```

### Score Formatting

```python
def format_score_distribution(scores: List[float], query: str, threshold: float) -> Dict[str, Any]:
    """Format score distribution for logging."""
    if not scores:
        return {
            "query": query,
            "scores": [],
            "statistics": {},
            "threshold": threshold,
            "pass_count": 0,
            "total_count": 0
        }

    import statistics

    stats = {
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std_dev": round(statistics.stdev(scores) if len(scores) > 1 else 0, 4)
    }

    pass_count = sum(1 for score in scores if score >= threshold)

    return {
        "query": query,
        "scores": [round(score, 4) for score in scores],
        "statistics": stats,
        "threshold": threshold,
        "pass_count": pass_count,
        "total_count": len(scores)
    }
```

### Error Formatting

```python
def format_error_response(error: Exception, context: str = None) -> Dict[str, Any]:
    """Format error for consistent error responses."""
    return {
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
```

## Type Conversion Utilities

### Safe Type Conversion

```python
def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    if isinstance(value, int):
        return value != 0
    return default
```

## Retry Utilities (`retry.py`)

### `async_retry_with_backoff()` Decorator

Decorator for async functions with exponential backoff retry logic:

```python
def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Decorator for async functions with exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts (including initial attempt).
        base_delay: Base delay in seconds between retries.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorated async function with retry logic.

    Raises:
        Last exception encountered after all retries are exhausted.
    """
```

### Retry Behavior

The decorator implements exponential backoff with the following delay formula:

```
delay = base_delay * (2 ** (attempt - 1))
```

**Example** (with `max_retries=3`, `base_delay=1.0`):
- Attempt 1: Execute immediately
- Attempt 2 (if 1 fails): Wait `1.0 * 2^0 = 1.0s`, then retry
- Attempt 3 (if 2 fails): Wait `1.0 * 2^1 = 2.0s`, then retry
- Raise last exception if all 3 attempts fail

### Usage Example

```python
from ccmemories.application.utils.retry import async_retry_with_backoff
from ccmemories.infrastructure import BaseOpenAIProvider

class MyLLMProvider(BaseOpenAIProvider):
    @async_retry_with_backoff(max_retries=3, base_delay=1.0)
    async def call_llm_with_retry(self, prompt: str) -> dict:
        """Call LLM with automatic retry on transient errors."""
        return await self._call_llm_with_structured_output(
            prompt=prompt,
            response_schema=MySchema,
        )
```

### Configuration Examples

**Default retries (3 attempts, 1s base delay):**
```python
@async_retry_with_backoff()
async def my_function():
    ...
```

**More retries with longer delays:**
```python
@async_retry_with_backoff(max_retries=5, base_delay=2.0)
async def my_function():
    ...
```

**Specific exceptions only:**
```python
@async_retry_with_backoff(max_retries=3, exceptions=(TimeoutError, ConnectionError))
async def my_function():
    ...
```

### Design Rationale

**Why exponential backoff?**
- **Network resilience**: Temporary issues (rate limits, hiccups) often resolve quickly
- **Server-friendly**: Exponential delay avoids overwhelming struggling services
- **Balance**: 3 retries with exponential delay provides reasonable recovery without excessive wait times

**When to use:**
- LLM API calls (can have transient failures)
- Network requests to external services
- Database operations with connection errors
- Any operation that may fail transiently

## Testing Utilities

### Test Fixtures

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Go is great for system programming"
    ]

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger

@pytest.fixture
def test_config():
    """Test configuration."""
    config = Mock()
    config.project_id = "test-project"
    config.search_limit = 5
    config.reranker_engine = "llm"
    config.hybrid_fusion_k = 60
    return config
```

### Test Helpers

```python
def create_mock_memory_result(memory: str, score: float = 0.8) -> Mock:
    """Create mock memory search result."""
    result = Mock()
    result.memory = memory
    result.score = score
    result.__str__ = Mock(return_value=memory)
    result.lower = Mock(return_value=memory.lower())
    return result

def assert_log_contains(log_mock, message: str, level: str = "info") -> None:
    """Assert that log contains specific message."""
    calls = getattr(log_mock, level).call_args_list
    assert any(message in str(call) for call in calls), f"Log does not contain '{message}'"
```

## Best Practices

### Logging Best Practices

1. **Consistent Context**: Always include `project_id` and relevant operation context
2. **Structured Data**: Use dictionaries for extra context, not string formatting
3. **Performance**: Use timing context managers for performance-critical operations
4. **Security**: Never log sensitive data like API keys
5. **Levels**: Use appropriate log levels (INFO for operations, ERROR for failures)

### Validation Best Practices

1. **Fail Fast**: Validate inputs at function entry points
2. **Clear Messages**: Provide specific error messages with field names
3. **Type Safety**: Check types before processing
4. **Boundary Testing**: Test min/max values and edge cases
5. **Consistency**: Use same validation logic across similar functions

### Formatting Best Practices

1. **Consistent Structure**: Use same format for similar data types
2. **Precision**: Round floating point numbers appropriately
3. **Metadata**: Include timestamps and context in formatted output
4. **Error Handling**: Gracefully handle formatting errors
5. **Performance**: Cache expensive formatting operations when possible

## Integration with Application

### Usage in MemoryManager

```python
# In memory/manager.py
from ccmemories.application.utils import StructuredLogger, validate_message_list

class MemoryManager:
    def __init__(self, config, logger):
        self.logger = logger  # StructuredLogger instance
        self.config = config

    def add_messages(self, messages: List[str]) -> int:
        # Validation
        is_valid, error = validate_message_list(messages)
        if not is_valid:
            raise ValueError(f"Invalid messages: {error}")

        # Logging with context
        self.logger.info(
            "Adding messages to hybrid storage",
            extra={
                "project_id": self.project_id,
                "message_count": len(messages),
                "message_lengths": [len(msg) for msg in messages]
            }
        )
```

### Usage in Tools

```python
# In tools/add.py
from ccmemories.application.utils import validate_message_list, format_error_response

@self.mcp.tool
def add(self, messages: List[str]) -> None:
    try:
        # Validation
        is_valid, error = validate_message_list(messages)
        if not is_valid:
            raise ValueError(error)

        # Operation
        stored_count = self.memory_manager.add_messages(messages)

        # Success logging
        self.logger.info(
            "Messages added successfully",
            extra={
                "tool": "add",
                "project_id": self.project_id,
                "stored_count": stored_count,
                "total_count": len(messages)
            }
        )

    except Exception as e:
        # Error formatting and logging
        error_data = format_error_response(e, context="add_messages")
        self.logger.error(
            "Failed to add messages",
            extra={
                "tool": "add",
                "project_id": self.project_id,
                "error_data": error_data
            }
        )
        raise RuntimeError(f"Failed to add messages: {str(e)}") from e
```

## Testing Strategy

### Unit Tests

```python
class TestValidationUtils:
    def test_validate_message_length(self):
        # Test valid messages
        assert validate_message_length("Hello")[0] is True

        # Test invalid messages
        assert validate_message_length("")[0] is False
        assert validate_message_length("x" * 30721)[0] is False

class TestLoggingUtils:
    def test_structured_logger(self, mock_logger):
        logger = StructuredLogger("test")
        logger.logger = mock_logger

        logger.info("Test message", extra={"key": "value"})

        mock_logger.info.assert_called_once_with("Test message", key="value")
```

### Integration Tests

```python
def test_validation_integration():
    """Test validation with real data."""
    valid_messages = ["Valid message", "Another valid message"]
    invalid_messages = ["", "x" * 30721, 123]

    assert validate_message_list(valid_messages)[0] is True
    assert validate_message_list(invalid_messages)[0] is False
```

This utility system provides the foundation for consistent, maintainable code across the entire CCMemoriesMCP application.
