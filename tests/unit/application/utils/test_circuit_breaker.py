"""Unit tests for reflectlog.application.utils.circuit_breaker module."""

import asyncio
import time
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    circuit_breaker_decorator,
)
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_closed_value(self) -> None:
        """Test CLOSED state has correct value."""
        assert CircuitState.CLOSED.value == "closed"

    def test_open_value(self) -> None:
        """Test OPEN state has correct value."""
        assert CircuitState.OPEN.value == "open"

    def test_half_open_value(self) -> None:
        """Test HALF_OPEN state has correct value."""
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_all_states_exist(self) -> None:
        """Test all three states are defined."""
        states = list(CircuitState)
        assert len(states) == 3


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.timeout == 60.0
        assert config.success_threshold == 2
        assert config.exception_types == (Exception,)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=30.0,
            success_threshold=1,
            exception_types=(ValueError, TypeError),
        )
        assert config.failure_threshold == 3
        assert config.timeout == 30.0
        assert config.success_threshold == 1
        assert config.exception_types == (ValueError, TypeError)


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError exception."""

    def test_initialization(self) -> None:
        """Test error stores last_failure_time and timeout."""
        now = time.time()
        error = CircuitBreakerOpenError(last_failure_time=now, timeout=60.0)
        assert error.last_failure_time == now
        assert error.timeout == 60.0

    def test_message_contains_remaining_time(self) -> None:
        """Test error message includes remaining seconds."""
        now = time.time()
        error = CircuitBreakerOpenError(last_failure_time=now, timeout=60.0)
        assert "Circuit is open after failure" in str(error)
        assert "Try again in" in str(error)
        assert "seconds" in str(error)

    def test_is_exception(self) -> None:
        """Test CircuitBreakerOpenError is an Exception subclass."""
        error = CircuitBreakerOpenError(last_failure_time=time.time(), timeout=30.0)
        assert isinstance(error, Exception)


class TestCircuitBreakerInit:
    """Tests for CircuitBreaker initialization."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    def test_initial_state_is_closed(self, mock_logger: MagicMock) -> None:
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test_api",
            logger=mock_logger,
        )
        assert cb._state == CircuitState.CLOSED

    def test_initial_counters_are_zero(self, mock_logger: MagicMock) -> None:
        """Test failure and success counters start at zero."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test_api",
            logger=mock_logger,
        )
        assert cb._failure_count == 0
        assert cb._success_count == 0

    def test_initial_timestamps_are_none(self, mock_logger: MagicMock) -> None:
        """Test timestamp fields start as None."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test_api",
            logger=mock_logger,
        )
        assert cb._last_failure_time is None
        assert cb._last_success_time is None
        assert cb._opened_at is None

    def test_stores_config_and_name(self, mock_logger: MagicMock) -> None:
        """Test config and name are stored correctly."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config, name="my_service", logger=mock_logger)
        assert cb._config is config
        assert cb._name == "my_service"


class TestShouldAttemptReset:
    """Tests for CircuitBreaker._should_attempt_reset."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    def test_returns_false_when_opened_at_is_none(self, mock_logger: MagicMock) -> None:
        """Test returns False when circuit was never opened."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(timeout=10.0),
            name="test",
            logger=mock_logger,
        )
        assert cb._should_attempt_reset() is False

    def test_returns_false_before_timeout(self, mock_logger: MagicMock) -> None:
        """Test returns False when timeout has not elapsed."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(timeout=60.0),
            name="test",
            logger=mock_logger,
        )
        cb._opened_at = time.time()  # Just opened
        assert cb._should_attempt_reset() is False

    def test_returns_true_after_timeout(self, mock_logger: MagicMock) -> None:
        """Test returns True when timeout has elapsed."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(timeout=1.0),
            name="test",
            logger=mock_logger,
        )
        cb._opened_at = time.time() - 2.0  # Opened 2s ago, timeout is 1s
        assert cb._should_attempt_reset() is True


class TestRecordSuccess:
    """Tests for CircuitBreaker._record_success."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    def test_increments_success_count(self, mock_logger: MagicMock) -> None:
        """Test success count increments."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._record_success()
        assert cb._success_count == 1

    def test_resets_failure_count(self, mock_logger: MagicMock) -> None:
        """Test failure count resets to zero on success."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._failure_count = 3
        cb._record_success()
        assert cb._failure_count == 0

    def test_sets_last_success_time(self, mock_logger: MagicMock) -> None:
        """Test last success time is set."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._record_success()
        assert cb._last_success_time is not None

    def test_transitions_half_open_to_closed(self, mock_logger: MagicMock) -> None:
        """Test transition from HALF_OPEN to CLOSED after success threshold."""
        config = CircuitBreakerConfig(success_threshold=2)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)
        cb._state = CircuitState.HALF_OPEN

        cb._record_success()  # 1st success
        assert cb._state == CircuitState.HALF_OPEN

        cb._record_success()  # 2nd success - meets threshold
        assert cb._state == CircuitState.CLOSED

    def test_logs_on_transition_to_closed(self, mock_logger: MagicMock) -> None:
        """Test info log emitted when transitioning to CLOSED."""
        config = CircuitBreakerConfig(success_threshold=1)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)
        cb._state = CircuitState.HALF_OPEN

        cb._record_success()
        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        assert "closed" in log_msg.lower()

    def test_resets_success_count_on_transition(self, mock_logger: MagicMock) -> None:
        """Test success count resets when transitioning to CLOSED."""
        config = CircuitBreakerConfig(success_threshold=1)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)
        cb._state = CircuitState.HALF_OPEN

        cb._record_success()
        assert cb._success_count == 0

    def test_no_transition_in_closed_state(self, mock_logger: MagicMock) -> None:
        """Test no state transition when already CLOSED."""
        config = CircuitBreakerConfig(success_threshold=1)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)
        cb._state = CircuitState.CLOSED

        cb._record_success()
        assert cb._state == CircuitState.CLOSED
        mock_logger.info.assert_not_called()


class TestRecordFailure:
    """Tests for CircuitBreaker._record_failure."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    def test_increments_failure_count(self, mock_logger: MagicMock) -> None:
        """Test failure count increments."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._record_failure(ValueError("boom"))
        assert cb._failure_count == 1

    def test_resets_success_count(self, mock_logger: MagicMock) -> None:
        """Test success count resets on failure."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._success_count = 3
        cb._record_failure(ValueError("boom"))
        assert cb._success_count == 0

    def test_sets_last_failure_time(self, mock_logger: MagicMock) -> None:
        """Test last failure time is set."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._record_failure(ValueError("boom"))
        assert cb._last_failure_time is not None

    def test_transitions_closed_to_open_at_threshold(
        self, mock_logger: MagicMock
    ) -> None:
        """Test circuit opens when failure threshold is reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)

        for _ in range(2):
            cb._record_failure(ValueError("boom"))
        assert cb._state == CircuitState.CLOSED

        cb._record_failure(ValueError("boom"))  # 3rd failure
        assert cb._state == CircuitState.OPEN

    def test_sets_opened_at_on_transition(self, mock_logger: MagicMock) -> None:
        """Test opened_at timestamp is set when circuit opens."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)

        cb._record_failure(ValueError("boom"))
        assert cb._opened_at is not None

    def test_logs_warning_on_transition_to_open(self, mock_logger: MagicMock) -> None:
        """Test warning log on CLOSED -> OPEN transition."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)

        cb._record_failure(ValueError("boom"))
        mock_logger.warning.assert_called_once()
        log_msg = mock_logger.warning.call_args[0][0]
        assert "opened" in log_msg.lower()

    def test_half_open_to_open_on_failure(self, mock_logger: MagicMock) -> None:
        """Test failure in HALF_OPEN returns to OPEN."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._state = CircuitState.HALF_OPEN

        cb._record_failure(RuntimeError("service down"))
        assert cb._state == CircuitState.OPEN
        assert cb._opened_at is not None

    def test_half_open_failure_logs_warning(self, mock_logger: MagicMock) -> None:
        """Test warning log when HALF_OPEN fails back to OPEN."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._state = CircuitState.HALF_OPEN

        cb._record_failure(RuntimeError("service down"))
        mock_logger.warning.assert_called_once()
        log_msg = mock_logger.warning.call_args[0][0]
        assert "HALF_OPEN" in log_msg

    def test_below_threshold_stays_closed(self, mock_logger: MagicMock) -> None:
        """Test circuit stays CLOSED when below failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config=config, name="test", logger=mock_logger)

        for _ in range(4):
            cb._record_failure(ValueError("boom"))
        assert cb._state == CircuitState.CLOSED
        mock_logger.warning.assert_not_called()


class TestCall:
    """Tests for CircuitBreaker.call async method."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    @pytest.fixture
    def circuit_breaker(self, mock_logger: MagicMock) -> CircuitBreaker:
        """Create a circuit breaker with low thresholds for testing."""
        return CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=2,
                timeout=0.1,
                success_threshold=1,
            ),
            name="test_service",
            logger=mock_logger,
        )

    async def test_passes_through_in_closed_state(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test async function calls pass through in CLOSED state."""

        async def my_func(x: int) -> int:
            return x * 2

        result = await circuit_breaker.call(my_func, 5)
        assert result == 10

    async def test_passes_through_sync_function(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test sync function calls pass through in CLOSED state."""

        def my_sync_func(x: int) -> int:
            return x + 1

        result = await circuit_breaker.call(my_sync_func, 5)
        assert result == 6

    async def test_raises_on_open_circuit(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test CircuitBreakerOpenError raised when circuit is OPEN."""
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.time()  # Just opened
        circuit_breaker._last_failure_time = time.time()

        async def my_func() -> str:
            return "ok"

        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(my_func)  # type: ignore

    async def test_open_circuit_without_last_failure_time(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test open circuit uses current time when last_failure_time is None."""
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.time()
        circuit_breaker._last_failure_time = None  # Edge case

        async def my_func() -> str:
            return "ok"

        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(my_func)  # type: ignore

    async def test_transitions_to_half_open_after_timeout(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit transitions from OPEN to HALF_OPEN after timeout."""
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.time() - 1.0  # Well past 0.1s timeout

        async def my_func() -> str:
            return "recovered"

        result = await circuit_breaker.call(my_func)
        assert result == "recovered"
        assert (
            circuit_breaker._state == CircuitState.CLOSED
        )  # After success threshold=1

    async def test_half_open_logs_transition(
        self, circuit_breaker: CircuitBreaker, mock_logger: MagicMock
    ) -> None:
        """Test info log when transitioning to HALF_OPEN."""
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.time() - 1.0

        async def my_func() -> str:
            return "ok"

        await circuit_breaker.call(my_func)  # type: ignore
        # Should have logged transition to HALF_OPEN
        info_calls = mock_logger.info.call_args_list
        half_open_logged = any("HALF_OPEN" in str(call) for call in info_calls)
        assert half_open_logged

    async def test_records_failure_on_exception(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test failure is recorded when function raises."""

        async def failing_func() -> None:
            raise ValueError("service error")

        with pytest.raises(ValueError, match="service error"):
            await circuit_breaker.call(failing_func)  # type: ignore

        assert circuit_breaker._failure_count == 1

    async def test_circuit_opens_after_threshold_failures(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit opens after failure_threshold consecutive failures."""

        async def failing_func() -> None:
            raise ValueError("error")

        for _ in range(2):  # threshold is 2
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)  # type: ignore

        assert circuit_breaker._state == CircuitState.OPEN

    async def test_only_catches_configured_exceptions(
        self, mock_logger: MagicMock
    ) -> None:
        """Test only configured exception types trigger circuit breaker."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=1,
                exception_types=(ValueError,),
            ),
            name="test",
            logger=mock_logger,
        )

        async def raise_type_error() -> None:
            raise TypeError("not tracked")

        # TypeError is not in exception_types, so it should propagate
        # but NOT be recorded by the circuit breaker
        with pytest.raises(TypeError):
            await cb.call(raise_type_error)  # type: ignore

        assert cb._failure_count == 0
        assert cb._state == CircuitState.CLOSED

    async def test_configured_exception_opens_circuit(
        self, mock_logger: MagicMock
    ) -> None:
        """Test configured exception type does trigger circuit breaker."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=1,
                exception_types=(ValueError,),
            ),
            name="test",
            logger=mock_logger,
        )

        async def raise_value_error() -> None:
            raise ValueError("tracked")

        with pytest.raises(ValueError):
            await cb.call(raise_value_error)  # type: ignore

        assert cb._failure_count == 1
        assert cb._state == CircuitState.OPEN

    async def test_passes_args_and_kwargs(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test positional and keyword arguments are forwarded."""

        async def func(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        result = await circuit_breaker.call(func, 1, 2, c=3)
        assert result == 6

    async def test_full_lifecycle_closed_open_half_open_closed(
        self, mock_logger: MagicMock
    ) -> None:
        """Test full state transition lifecycle."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=2,
                timeout=0.05,
                success_threshold=1,
            ),
            name="lifecycle",
            logger=mock_logger,
        )

        # Start CLOSED
        assert cb._state == CircuitState.CLOSED

        # Fail until OPEN
        async def fail() -> None:
            raise ValueError("err")

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(fail)  # type: ignore
        assert cb._state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.1)

        # Next call transitions to HALF_OPEN then succeeds -> CLOSED
        async def succeed() -> str:
            return "ok"

        result = await cb.call(succeed)
        assert result == "ok"
        assert cb._state == CircuitState.CLOSED

    async def test_half_open_failure_returns_to_open(
        self, mock_logger: MagicMock
    ) -> None:
        """Test failure in HALF_OPEN transitions back to OPEN."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout=0.05,
                success_threshold=1,
            ),
            name="test",
            logger=mock_logger,
        )

        # Open the circuit
        async def fail() -> None:
            raise ValueError("err")

        with pytest.raises(ValueError):
            await cb.call(fail)  # type: ignore
        assert cb._state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.1)

        # Fail again in HALF_OPEN -> back to OPEN
        with pytest.raises(ValueError):
            await cb.call(fail)  # type: ignore
        assert cb._state == CircuitState.OPEN


class TestGetState:
    """Tests for CircuitBreaker.get_state."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    def test_returns_closed_initially(self, mock_logger: MagicMock) -> None:
        """Test returns CLOSED when just created."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        assert cb.get_state() == CircuitState.CLOSED

    def test_returns_open_state(self, mock_logger: MagicMock) -> None:
        """Test returns OPEN when circuit is open and timeout not elapsed."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(timeout=60.0),
            name="test",
            logger=mock_logger,
        )
        cb._state = CircuitState.OPEN
        cb._opened_at = time.time()  # Just opened
        assert cb.get_state() == CircuitState.OPEN

    def test_auto_transitions_to_half_open(self, mock_logger: MagicMock) -> None:
        """Test auto-transition from OPEN to HALF_OPEN after timeout."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(timeout=0.01),
            name="test",
            logger=mock_logger,
        )
        cb._state = CircuitState.OPEN
        cb._opened_at = time.time() - 1.0  # Well past timeout
        assert cb.get_state() == CircuitState.HALF_OPEN

    def test_returns_half_open_state(self, mock_logger: MagicMock) -> None:
        """Test returns HALF_OPEN when already in that state."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._state = CircuitState.HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN


class TestGetStats:
    """Tests for CircuitBreaker.get_stats."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    def test_returns_complete_stats(self, mock_logger: MagicMock) -> None:
        """Test stats dictionary contains all expected keys."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout=60.0,
                success_threshold=2,
            ),
            name="my_api",
            logger=mock_logger,
        )

        stats = cb.get_stats()
        assert stats["name"] == "my_api"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0
        assert stats["last_failure_time"] is None
        assert stats["last_success_time"] is None
        assert stats["opened_at"] is None
        assert stats["failure_threshold"] == 5
        assert stats["timeout"] == 60.0
        assert stats["success_threshold"] == 2

    def test_stats_reflect_current_state(self, mock_logger: MagicMock) -> None:
        """Test stats reflect mutations to the circuit breaker."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(failure_threshold=2),
            name="test",
            logger=mock_logger,
        )
        cb._record_failure(ValueError("err"))
        cb._record_failure(ValueError("err"))

        stats = cb.get_stats()
        assert stats["state"] == "open"
        assert stats["failure_count"] == 2
        assert stats["last_failure_time"] is not None
        assert stats["opened_at"] is not None


class TestReset:
    """Tests for CircuitBreaker.reset."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    def test_resets_state_to_closed(self, mock_logger: MagicMock) -> None:
        """Test reset sets state to CLOSED."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._state = CircuitState.OPEN
        cb.reset()
        assert cb._state == CircuitState.CLOSED

    def test_resets_all_counters(self, mock_logger: MagicMock) -> None:
        """Test reset clears all counters."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._failure_count = 5
        cb._success_count = 3
        cb.reset()
        assert cb._failure_count == 0
        assert cb._success_count == 0

    def test_resets_all_timestamps(self, mock_logger: MagicMock) -> None:
        """Test reset clears all timestamps."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb._last_failure_time = time.time()
        cb._last_success_time = time.time()
        cb._opened_at = time.time()
        cb.reset()
        assert cb._last_failure_time is None
        assert cb._last_success_time is None
        assert cb._opened_at is None

    def test_logs_reset(self, mock_logger: MagicMock) -> None:
        """Test reset logs info message."""
        cb = CircuitBreaker(
            config=CircuitBreakerConfig(),
            name="test",
            logger=mock_logger,
        )
        cb.reset()
        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        assert "reset" in log_msg.lower()
        assert "CLOSED" in log_msg


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker_decorator function."""

    @pytest.fixture
    def mock_logger(self) -> IStructuredLogger:
        """Create a mock StructuredLogger."""
        return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))

    @pytest.fixture
    def circuit_breaker(self, mock_logger: MagicMock) -> CircuitBreaker:
        """Create a circuit breaker for decorator tests."""
        return CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=2,
                timeout=0.1,
                success_threshold=1,
            ),
            name="decorated_service",
            logger=mock_logger,
        )

    async def test_decorates_async_function(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test decorator wraps async function correctly."""

        @circuit_breaker_decorator(circuit_breaker)
        async def my_async_func(x: int) -> int:
            return x * 3

        result = await my_async_func(4)
        assert result == 12

    async def test_decorated_async_records_failure(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test decorated async function failures are tracked."""

        @circuit_breaker_decorator(circuit_breaker)
        async def failing_async() -> None:
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            await failing_async()

        assert circuit_breaker._failure_count == 1

    async def test_decorated_async_opens_circuit(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test decorated async function opens circuit after threshold."""

        @circuit_breaker_decorator(circuit_breaker)
        async def failing_async() -> None:
            raise ValueError("error")

        for _ in range(2):  # threshold is 2
            with pytest.raises(ValueError):
                await failing_async()

        assert circuit_breaker._state == CircuitState.OPEN

        # Next call should fail fast
        with pytest.raises(CircuitBreakerOpenError):
            await failing_async()

    def test_decorates_sync_function(self, circuit_breaker: CircuitBreaker) -> None:
        """Test decorator wraps sync function correctly."""

        @circuit_breaker_decorator(circuit_breaker)
        def my_sync_func(x: int) -> int:
            return x + 10

        result = my_sync_func(5)
        assert result == 15

    def test_decorated_sync_records_failure(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test decorated sync function failures are tracked."""

        @circuit_breaker_decorator(circuit_breaker)
        def failing_sync() -> None:
            raise ValueError("sync error")

        with pytest.raises(ValueError, match="sync error"):
            failing_sync()

        assert circuit_breaker._failure_count == 1

    def test_decorated_sync_opens_circuit(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test decorated sync function opens circuit after threshold."""

        @circuit_breaker_decorator(circuit_breaker)
        def failing_sync() -> None:
            raise ValueError("error")

        for _ in range(2):  # threshold is 2
            with pytest.raises(ValueError):
                failing_sync()

        assert circuit_breaker._state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            failing_sync()

    def test_sync_decorator_creates_event_loop_if_needed(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test sync decorator handles missing event loop."""

        # We patch to simulate RuntimeError on get_event_loop
        @circuit_breaker_decorator(circuit_breaker)
        def my_sync_func() -> str:
            return "result"

        with patch(
            "reflectlog.application.utils.circuit_breaker.asyncio.get_event_loop",
            side_effect=RuntimeError("no loop"),
        ):
            new_loop = asyncio.new_event_loop()
            with patch(
                "reflectlog.application.utils.circuit_breaker.asyncio.new_event_loop",
                return_value=new_loop,
            ):
                with patch(
                    "reflectlog.application.utils.circuit_breaker.asyncio.set_event_loop"
                ) as mock_set:
                    result = my_sync_func()
                    assert result == "result"
                    mock_set.assert_called_once_with(new_loop)
            new_loop.close()
