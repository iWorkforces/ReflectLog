"""Unit tests for reflectlog.utility.retry module."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reflectlog.utility.retry import (
    _TRANSIENT_EXCEPTIONS,
    async_retry_with_backoff,
)


class TestRetryInterface:
    """Verify retry module interface."""

    def test_transient_exceptions_contents(self) -> None:
        """Default transient exceptions include expected types."""
        expected = (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError)
        assert _TRANSIENT_EXCEPTIONS == expected

    def test_decorator_preserves_function_name(self) -> None:
        """Decorator uses @wraps to preserve the original function name."""

        @async_retry_with_backoff(max_retries=1)
        async def my_func() -> str:
            return "ok"

        assert my_func.__name__ == "my_func"


class TestRetryBehavior:
    """Test actual retry behavior."""

    async def test_success_no_retry(self) -> None:
        """Function succeeds on first call - no retries."""
        call_count = 0

        @async_retry_with_backoff(max_retries=3)
        async def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_transient_exception(self) -> None:
        """Retries on ConnectionError (default transient) then succeeds."""
        call_count = 0

        @async_retry_with_backoff(max_retries=3, base_delay=0.001)
        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection lost")
            return "recovered"

        result = await flaky()
        assert result == "recovered"
        assert call_count == 3

    async def test_raises_after_max_retries(self) -> None:
        """Raises the last exception after all retries exhausted."""
        call_count = 0

        @async_retry_with_backoff(max_retries=2, base_delay=0.001)
        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise TimeoutError(f"attempt {call_count}")

        with pytest.raises(TimeoutError, match="attempt 2"):
            await always_fail()
        assert call_count == 2

    async def test_no_retry_on_non_transient_exception(self) -> None:
        """Does not retry on exceptions not in the transient list."""
        call_count = 0

        @async_retry_with_backoff(max_retries=3, base_delay=0.001)
        async def value_error_fn() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await value_error_fn()
        assert call_count == 1

    async def test_custom_exceptions(self) -> None:
        """Retries on custom exception types when specified."""
        call_count = 0

        @async_retry_with_backoff(
            max_retries=2, base_delay=0.001, exceptions=(ValueError,)
        )
        async def custom_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("custom")

        with pytest.raises(ValueError, match="custom"):
            await custom_fail()
        assert call_count == 2

    async def test_max_retries_one_no_retry(self) -> None:
        """max_retries=1 means only one attempt, no retries."""
        call_count = 0

        @async_retry_with_backoff(max_retries=1, base_delay=0.001)
        async def single_shot() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await single_shot()
        assert call_count == 1

    async def test_max_delay_cap(self) -> None:
        """Delay is capped at max_delay regardless of exponential growth."""
        delays: list[float] = []

        async def mock_sleep(delay: float) -> None:
            delays.append(delay)

        @async_retry_with_backoff(max_retries=5, base_delay=10.0, max_delay=15.0)
        async def always_timeout() -> str:
            raise TimeoutError("timeout")

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(TimeoutError):
                await always_timeout()

        # All delays should be <= max_delay (15.0)
        for d in delays:
            assert d <= 15.0, f"Delay {d} exceeds max_delay 15.0"

    async def test_exponential_backoff_pattern(self) -> None:
        """Delays follow exponential backoff pattern with jitter."""
        delays: list[float] = []

        async def mock_sleep(delay: float) -> None:
            delays.append(delay)

        @async_retry_with_backoff(max_retries=4, base_delay=1.0, max_delay=100.0)
        async def always_fail() -> str:
            raise ConnectionError("fail")

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(ConnectionError):
                await always_fail()

        # 3 delays (4 attempts, 3 sleeps between them)
        assert len(delays) == 3

        # Expected base delays: 1.0, 2.0, 4.0 (before jitter)
        # With jitter ±20%: 0.8-1.2, 1.6-2.4, 3.2-4.8
        assert 0.8 <= delays[0] <= 1.2
        assert 1.6 <= delays[1] <= 2.4
        assert 3.2 <= delays[2] <= 4.8

    async def test_logging_on_retry(self) -> None:
        """Logger is called on retry attempts."""
        mock_logger = MagicMock(spec=logging.Logger)
        call_count = 0

        @async_retry_with_backoff(max_retries=2, base_delay=0.001, logger=mock_logger)
        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("oops")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError):
                await flaky()

        # Should log info for retry attempt and warning for exhaustion
        assert mock_logger.info.called
        assert mock_logger.warning.called

    async def test_logging_exhaustion_message(self) -> None:
        """Logger warning includes function name on exhaustion."""
        mock_logger = MagicMock(spec=logging.Logger)

        @async_retry_with_backoff(max_retries=1, base_delay=0.001, logger=mock_logger)
        async def named_func() -> str:
            raise OSError("disk error")

        with pytest.raises(OSError):
            await named_func()

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "named_func" in warning_msg
        assert "1" in warning_msg  # max_retries count

    async def test_passes_args_and_kwargs(self) -> None:
        """Decorated function receives positional and keyword arguments."""
        received_args: list = []
        received_kwargs: list = []

        @async_retry_with_backoff(max_retries=1)
        async def capture(a: int, b: str, *, c: bool = False) -> str:
            received_args.append((a, b))
            received_kwargs.append({"c": c})
            return f"{a}-{b}-{c}"

        result = await capture(42, "hello", c=True)
        assert result == "42-hello-True"
        assert received_args == [(42, "hello")]
        assert received_kwargs == [{"c": True}]

    async def test_oserror_is_retried(self) -> None:
        """OSError (includes socket errors) is in default transient set."""
        call_count = 0

        @async_retry_with_backoff(max_retries=2, base_delay=0.001)
        async def socket_fail() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("socket error")
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await socket_fail()
        assert result == "ok"
        assert call_count == 2

    async def test_asyncio_timeout_error_is_retried(self) -> None:
        """asyncio.TimeoutError is in default transient set."""
        call_count = 0

        @async_retry_with_backoff(max_retries=2, base_delay=0.001)
        async def timeout_fail() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await timeout_fail()
        assert result == "ok"
        assert call_count == 2
