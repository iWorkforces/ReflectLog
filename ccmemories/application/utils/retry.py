"""Retry decorator with exponential backoff for async functions."""

import asyncio
from functools import wraps
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

# Generic types for async function decorator
P = ParamSpec("P")
T = TypeVar("T")


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]
]:
    """Decorator for async functions with exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts (including initial attempt).
        base_delay: Base delay in seconds between retries (will be multiplied by 2^(attempt-1)).
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorated async function with retry logic.

    Raises:
        Last exception encountered after all retries are exhausted.

    Example:
        ```python
        @async_retry_with_backoff(max_retries=3, base_delay=1.0)
        async def fetch_data(url: str) -> dict:
            async with aiohttp.get(url) as response:
                return await response.json()
        ```
    """

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
                        break

                    # Calculate delay with exponential backoff
                    delay = base_delay * (2 ** (attempt - 1))

                    # Sleep before next retry
                    await asyncio.sleep(delay)

            # All retries exhausted - raise the last exception
            if last_exception is not None:
                raise last_exception

            # Should never reach here, but type checker needs it
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator
