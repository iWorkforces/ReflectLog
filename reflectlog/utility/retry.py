"""Retry decorator with exponential backoff for async functions."""

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
import logging
import random
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

# Default logger for retry operations
_retry_logger = logging.getLogger(__name__)

# Transient exceptions that are safe to retry
# Connection errors, timeouts, and temporary server issues
_TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,  # Includes socket errors
    asyncio.TimeoutError,
)


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] | None = None,
    logger: logging.Logger | None = None,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]
]:
    """Decorator for async functions with exponential backoff retry logic.

    Features:
    - Exponential backoff: delay increases with each retry
    - Random jitter: prevents thundering herd problem
    - Max delay cap: prevents excessively long waits
    - Selective retry: only retries on transient exceptions by default
    - Optional logging: logs retry attempts for debugging

    Args:
        max_retries: Maximum number of retry attempts (including initial attempt).
        base_delay: Base delay in seconds between retries (will be multiplied by 2^(attempt-1)).
        max_delay: Maximum delay in seconds (default: 60.0).
        exceptions: Tuple of exception types to catch and retry on.
            If None, uses default transient exceptions (ConnectionError, TimeoutError, etc.)
        logger: Optional logger for retry logging. If None, uses module logger.

    Returns:
        Decorated async function with retry logic.

    Raises:
        Last exception encountered after all retries are exhausted.

    Example:
        ```python
        from reflectlog.utility.retry import async_retry_with_backoff

        @async_retry_with_backoff(max_retries=3, base_delay=1.0)
        async def fetch_data(url: str) -> dict:
            async with session.get(url) as response:
                return await response.json()
        ```
    """

    # Use default transient exceptions if none specified
    if exceptions is None:
        exceptions = _TRANSIENT_EXCEPTIONS

    # Use module logger if none specified
    if logger is None:
        logger = _retry_logger

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # If this is the last attempt, raise the exception
                    if attempt >= max_retries:
                        if logger:
                            func_name = getattr(func, "__name__", "<unknown>")
                            logger.warning(
                                f"All {max_retries} retry attempts exhausted for {func_name}",
                                extra={
                                    "function": func_name,
                                    "final_error": str(e),
                                    "error_type": type(e).__name__,
                                },
                            )
                        break

                    # Calculate delay with exponential backoff
                    delay = base_delay * (2 ** (attempt - 1))

                    # Add random jitter (±20%) to prevent thundering herd
                    jitter = random.uniform(0.8, 1.2)
                    delay = delay * jitter

                    # Cap the maximum delay
                    delay = min(delay, max_delay)

                    # Log retry attempt
                    if logger:
                        func_name = getattr(func, "__name__", "<unknown>")
                        logger.info(
                            f"Retry attempt {attempt + 1}/{max_retries} for {func_name} after {delay:.2f}s delay",
                            extra={
                                "function": func_name,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "delay": delay,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )

                    # Sleep before next retry
                    await asyncio.sleep(delay)

            # All retries exhausted - raise the last exception
            if last_exception is not None:
                raise last_exception

            # Should never reach here, but type checker needs it
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator
