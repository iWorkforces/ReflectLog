"""Circuit breaker pattern for protecting external API calls.

The circuit breaker pattern prevents cascading failures by:
1. Tracking failures to external services
2. Opening the circuit (blocking calls) after threshold is reached
3. Allowing a half-open state for testing recovery
4. Closing the circuit when the service recovers

This is especially important for LLM API calls which can be slow or
unreliable, preventing the entire memory system from hanging.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, TypeVar

from .logging import StructuredLogger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Number of failures before opening
    timeout: float = 60.0  # Seconds to wait before trying again (OPEN -> HALF_OPEN)
    success_threshold: int = (
        2  # Successes needed to close circuit (HALF_OPEN -> CLOSED)
    )
    exception_types: tuple[type[Exception], ...] = (Exception,)  # Exceptions to track


T = TypeVar("T")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit is open and requests are blocked."""

    def __init__(self, last_failure_time: float, timeout: float):
        """Initialize error with context.

        Args:
            last_failure_time: Timestamp of the failure that opened the circuit
            timeout: Seconds until circuit will try again
        """
        self.last_failure_time = last_failure_time
        self.timeout = timeout
        remaining = last_failure_time + timeout - time.time()
        super().__init__(
            f"Circuit is open after failure. Try again in {remaining:.1f} seconds."
        )


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests fail fast without calling the service
    - HALF_OPEN: Testing if service has recovered, allows limited requests

    Example:
        ```python
        circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=3,
                timeout=30.0,
            ),
            name="llm_api",
            logger=logger,
        )

        @circuit_breaker.call
        async def call_llm_api(prompt: str) -> dict:
            return await external_llm_api.call(prompt)
        ```
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        name: str,
        logger: StructuredLogger,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            name: Name for logging (e.g., "llm_reranker_api")
            logger: Structured logger instance
        """
        self._config = config
        self._name = name
        self._logger = logger

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._opened_at: float | None = None

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset.

        Returns:
            True if circuit should transition to HALF_OPEN
        """
        if self._opened_at is None:
            return False

        elapsed = time.time() - self._opened_at
        return elapsed >= self._config.timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        self._success_count += 1
        self._failure_count = 0
        self._last_success_time = time.time()

        if (
            self._state == CircuitState.HALF_OPEN
            and self._success_count >= self._config.success_threshold
        ):
            self._state = CircuitState.CLOSED
            self._success_count = 0
            self._logger.info(
                f"Circuit breaker '{self._name}' closed after {self._config.success_threshold} "
                f"successful call(s) in HALF_OPEN state",
                extra={
                    "circuit_breaker": self._name,
                    "state": self._state.value,
                    "success_count": self._success_count,
                },
            )

    def _record_failure(self, exception: BaseException) -> None:
        """Record a failed call.

        Args:
            exception: The exception that caused the failure
        """
        self._failure_count += 1
        self._success_count = 0
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failed in HALF_OPEN, go back to OPEN
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
            self._logger.warning(
                f"Circuit breaker '{self._name}' returned to OPEN state after failure "
                f"in HALF_OPEN",
                extra={
                    "circuit_breaker": self._name,
                    "state": self._state.value,
                    "exception_type": type(exception).__name__,
                    "failure_count": self._failure_count,
                },
            )
        elif self._failure_count >= self._config.failure_threshold:
            # Threshold reached, open the circuit
            if self._state == CircuitState.CLOSED:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                self._logger.warning(
                    f"Circuit breaker '{self._name}' opened after "
                    f"{self._failure_count} consecutive failure(s)",
                    extra={
                        "circuit_breaker": self._name,
                        "state": self._state.value,
                        "failure_count": self._failure_count,
                        "failure_threshold": self._config.failure_threshold,
                        "exception_type": type(exception).__name__,
                    },
                )

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: The function to call (async or sync)
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            The result of func(*args, **kwargs)

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN and blocking requests
            Exception: The exception from func if it fails
        """
        # Check if circuit is OPEN
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                # Transition to HALF_OPEN to test recovery
                self._state = CircuitState.HALF_OPEN
                self._logger.info(
                    f"Circuit breaker '{self._name}' transitioning to HALF_OPEN state",
                    extra={
                        "circuit_breaker": self._name,
                        "state": self._state.value,
                        "open_duration": time.time() - (self._opened_at or 0),
                    },
                )
            else:
                # Circuit is still OPEN, fail fast
                raise CircuitBreakerOpenError(
                    last_failure_time=self._last_failure_time or time.time(),
                    timeout=self._config.timeout,
                )

        # Execute the function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            self._record_success()

            return result

        except self._config.exception_types as e:
            # Record failure
            self._record_failure(e)
            raise

    def get_state(self) -> CircuitState:
        """Get the current circuit state.

        Returns:
            The current circuit state (CLOSED, OPEN, or HALF_OPEN)
        """
        # Auto-transition from OPEN to HALF_OPEN if timeout has passed
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            self._state = CircuitState.HALF_OPEN

        return self._state

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary containing circuit breaker state and metrics
        """
        return {
            "name": self._name,
            "state": self.get_state().value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "last_success_time": self._last_success_time,
            "opened_at": self._opened_at,
            "failure_threshold": self._config.failure_threshold,
            "timeout": self._config.timeout,
            "success_threshold": self._config.success_threshold,
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state.

        This is useful for testing or manual recovery procedures.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_success_time = None
        self._opened_at = None

        self._logger.info(
            f"Circuit breaker '{self._name}' manually reset to CLOSED state",
            extra={"circuit_breaker": self._name, "state": self._state.value},
        )


def circuit_breaker_decorator(
    circuit_breaker: CircuitBreaker,
) -> Callable:
    """Create a decorator that wraps a function with circuit breaker protection.

    Args:
        circuit_breaker: The circuit breaker instance

    Returns:
        A decorator function

    Example:
        ```python
        circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(failure_threshold=3, timeout=30.0),
            name="llm_api",
            logger=logger,
        )

        @circuit_breaker_decorator(circuit_breaker)
        async def call_llm_api(prompt: str) -> dict:
            return await external_llm_api.call(prompt)
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            return await circuit_breaker.call(func, *args, **kwargs)

        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # For sync functions, we need to run the async call method
            # Create a simple event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(circuit_breaker.call(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        else:
            return sync_wrapper

    return decorator


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "circuit_breaker_decorator",
]
