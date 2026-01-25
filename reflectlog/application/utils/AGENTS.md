# Agent Guidelines for reflectlog/application/utils/

This directory contains utility functions and classes that support the core memory management system. These utilities provide consistent logging, validation, retry logic, security features, and performance utilities used throughout the application.

## Directory Structure

```
utils/
├── __init__.py          # Package exports
├── logging.py           # Structured logging utilities
├── numba_utils.py       # Numba JIT-compiled numerical utilities
├── retry.py             # Retry decorator with exponential backoff
├── security.py          # Secret redaction and secure logging
├── validation.py        # Input validation helpers
├── circuit_breaker.py   # CircuitBreaker for external service resilience
└── metrics.py           # Prometheus-style metrics collection
```

## Core Responsibilities

### Structured Logging

The `logging.py` module provides consistent structured logging across the application:

```python
class StructuredLogger:
    '''Provides structured logging with consistent context.'''

    def __init__(self, name: str, project_id: str | None = None, log_level: str = "INFO"):
        self.logger = structlog.get_logger(name)
        self.project_id = project_id
        self.log_level = log_level

    def info(self, message: str, extra: dict[str, Any] | None = None) -> None:
        '''Log info level with structured context.'''
        self.logger.info(message, **(extra or {}))

    def warning(self, message: str, extra: dict[str, Any] | None = None) -> None:
        '''Log warning level with structured context.'''
        self.logger.warning(message, **(extra or {}))

    def error(self, message: str, extra: dict[str, Any] | None = None) -> None:
        '''Log error level with structured context.'''
        self.logger.error(message, **(extra or {}))
```

### Retry Logic

The `retry.py` module provides a decorator for retrying operations with exponential backoff:

```python
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (IOError, TimeoutError),
) -> Callable:
    '''Decorator for retrying functions with exponential backoff.'''
```

### Circuit Breaker

The `circuit_breaker.py` module provides resilience for external service calls:

```python
class CircuitBreaker:
    '''Circuit breaker for external service resilience.'''

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

    async def __aenter__(self) -> None:
        '''Enter context manager.'''
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        '''Exit context manager, handle exceptions.'''
        ...
```

### Security

The `security.py` module provides utilities for protecting sensitive data:

```python
def redact_secrets(text: str, secrets: list[str] | None = None) -> str:
    '''Redact secrets from text for safe logging.'''
    ...

class SecureLogger:
    '''Logger that automatically redacts secrets.'''
```

### Metrics

The `metrics.py` module provides Prometheus-style metrics collection:

```python
class MetricsCollector:
    '''Collects and exports metrics.'''

    def increment(self, name: str, tags: dict[str, str] | None = None) -> None:
        '''Increment a counter metric.'''

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        '''Set a gauge metric.'''

    def histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        '''Observe a histogram metric.'''
```

### Numba Utilities

The `numba_utils.py` module provides JIT-compiled numerical utilities:

```python
from numba import jit

@jit(nopython=True)
def compute_similarity(vectors: np.ndarray) -> np.ndarray:
    '''JIT-compiled similarity computation.'''
    ...
```

## Key Patterns

### Retry Decorator Usage

```python
@retry(max_attempts=3, base_delay=1.0, retryable_exceptions=(aiohttp.ClientError,))
async def fetch_embedding(text: str) -> list[float]:
    '''Fetch embedding from API with retry.'''
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"text": text}) as response:
            return await response.json()
```

### Circuit Breaker Usage

```python
async with CircuitBreaker(failure_threshold=5, recovery_timeout=60.0):
    result = await external_api_call()
```

### Secure Logging

```python
logger = SecureLogger()
logger.info(
    "API request completed",
    extra={
        "url": "/api/endpoint",
        "api_key": "[REDACTED]",  # Auto-redacted
        "status": 200,
    }
)
```

### Metrics Collection

```python
metrics = MetricsCollector()

# Track operation duration
start_time = time.perf_counter()
result = await operation()
duration = (time.perf_counter() - start_time) * 1000

metrics.histogram(
    "operation_duration_ms",
    duration,
    tags={"operation": "search"},
)
```

## Utility Modules

### Validation Helpers

```python
def validate_message_length(message: str, max_length: int = 10000) -> None:
    '''Validate message length.'''
    if len(message) > max_length:
        raise ValueError(f"Message exceeds maximum length ({len(message)} > {max_length})")

def sanitize_query(query: str) -> str:
    '''Sanitize search query.'''
    return query.strip()[:1000]  # Limit length

def validate_project_id(project_id: str) -> None:
    '''Validate project ID format.'''
    if not project_id or len(project_id) < 3:
        raise ValueError("Project ID must be at least 3 characters")
```

### Logging Best Practices

Use structured logging with consistent context:

```python
# Good - structured with context
logger.info(
    "Search completed",
    extra={
        "query": query[:100],  # Truncate for safety
        "result_count": len(results),
        "duration_ms": duration,
        "project_id": self.project_id,
    }
)

# Avoid - unstructured logging
logger.info(f"Search completed with {len(results)} results in {duration}ms")
```

### Retry Best Practices

- Use exponential backoff to avoid overwhelming services
- Set reasonable max attempts based on operation criticality
- Only retry transient errors
- Log retry attempts for debugging

### Circuit Breaker Best Practices

- Set failure threshold based on normal error rates
- Configure recovery timeout appropriately
- Use half-open state to test recovery
- Monitor circuit state for debugging

## Error Handling

### Custom Exceptions

Define utility-specific exceptions:

```python
class RetryExhaustedError(Exception):
    '''Raised when all retry attempts are exhausted.'''

class CircuitOpenError(Exception):
    '''Raised when circuit breaker is open.'''
```

### Exception Transformation

Transform external exceptions to internal types:

```python
try:
    await external_call()
except aiohttp.ClientError as e:
    raise ServiceUnavailableError(f"External service error: {e}") from e
```

## Testing Guidelines

### Unit Tests

- Test retry logic with mocked time
- Test circuit breaker state transitions
- Test secure logging redaction
- Test validation helpers

### Test Cases

```python
def test_retry_exponential_backoff():
    '''Retry should increase delay exponentially.'''
    call_times = []

    @retry(base_delay=0.1, exponential_base=2.0)
    def failing_function():
        call_times.append(time.perf_counter())
        if len(call_times) < 3:
            raise ValueError("Temporary failure")
        return "success"

    start = time.perf_counter()
    result = failing_function()
    elapsed = time.perf_counter() - start

    # Should have 3 calls with increasing delays
    assert len(call_times) == 3
    # Total time should be approximately 0.1 + 0.2 = 0.3s
    assert elapsed >= 0.3

def test_circuit_opens_after_failures():
    '''Circuit should open after failure threshold.'''
    cb = CircuitBreaker(failure_threshold=3)

    for _ in range(3):
        with pytest.raises(CircuitOpenError):
            cb.call(failing_function)

    assert cb.state == CircuitState.OPEN
```

## Dependencies

### Internal Dependencies

- `application/config/`: Configuration values
- `application/exceptions.py`: Exception classes

### External Dependencies

- `structlog`: Structured logging
- `numba`: JIT compilation for numerical operations
- `prometheus-client`: Metrics collection (if used)

## Important Notes

### Secret Redaction

Never log sensitive information:

```python
# Always redact API keys, passwords, etc.
logger.info(
    "API call",
    extra={
        "api_key": "[REDACTED]",
        "request_body": "[REDACTED]" if has_secrets else body,
    }
)
```

### Performance

- Use JIT compilation for numerical hotspots
- Batch metrics collection to reduce overhead
- Avoid expensive operations in logging paths

### Thread Safety

- Metrics collectors should be thread-safe
- Circuit breaker state should be protected
- Use locks for shared mutable state
