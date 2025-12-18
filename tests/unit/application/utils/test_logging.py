"""Unit tests for ccmemories.application.utils.logging module."""

import logging
from unittest.mock import MagicMock, call, patch

import pytest

from ccmemories.application.utils.logging import (
    StructuredLogger,
    create_logger,
    format_fusion_score_status,
)


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Create a mock Python logger."""
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def structured_logger(self, mock_logger: MagicMock) -> StructuredLogger:
        """Create a StructuredLogger with mock underlying logger."""
        return StructuredLogger(
            logger=mock_logger, default_extra={"project_id": "test-project"}
        )

    def test_init_with_default_extra(self, mock_logger: MagicMock) -> None:
        """Test initialization with default extra fields."""
        logger = StructuredLogger(logger=mock_logger, default_extra={"key": "value"})
        assert logger.default_extra == {"key": "value"}

    def test_init_without_default_extra(self, mock_logger: MagicMock) -> None:
        """Test initialization without default extra fields."""
        logger = StructuredLogger(logger=mock_logger)
        assert logger.default_extra == {}

    def test_merge_extra_with_none(self, structured_logger: StructuredLogger) -> None:
        """Test _merge_extra returns default_extra when extra is None."""
        result = structured_logger._merge_extra(None)
        assert result == {"project_id": "test-project"}

    def test_merge_extra_with_values(self, structured_logger: StructuredLogger) -> None:
        """Test _merge_extra merges default and provided extras."""
        result = structured_logger._merge_extra({"custom": "value"})
        assert result == {"project_id": "test-project", "custom": "value"}

    def test_merge_extra_override_default(
        self, structured_logger: StructuredLogger
    ) -> None:
        """Test _merge_extra allows overriding default extra fields."""
        result = structured_logger._merge_extra({"project_id": "override"})
        assert result == {"project_id": "override"}

    def test_is_enabled_for(self, mock_logger: MagicMock) -> None:
        """Test isEnabledFor delegates to underlying logger."""
        mock_logger.isEnabledFor.return_value = True
        logger = StructuredLogger(logger=mock_logger)
        assert logger.isEnabledFor(logging.DEBUG) is True
        mock_logger.isEnabledFor.assert_called_once_with(logging.DEBUG)

    def test_info(
        self, structured_logger: StructuredLogger, mock_logger: MagicMock
    ) -> None:
        """Test info method logs at INFO level."""
        structured_logger.info("test message", extra={"key": "value"})
        mock_logger.info.assert_called_once_with(
            "test message",
            extra={"project_id": "test-project", "key": "value"},
            exc_info=False,
        )

    def test_info_with_exc_info(
        self, structured_logger: StructuredLogger, mock_logger: MagicMock
    ) -> None:
        """Test info method with exc_info=True."""
        structured_logger.info("test message", exc_info=True)
        mock_logger.info.assert_called_once_with(
            "test message",
            extra={"project_id": "test-project"},
            exc_info=True,
        )

    def test_error(
        self, structured_logger: StructuredLogger, mock_logger: MagicMock
    ) -> None:
        """Test error method logs at ERROR level."""
        structured_logger.error("error message", extra={"err": "details"})
        mock_logger.error.assert_called_once_with(
            "error message",
            extra={"project_id": "test-project", "err": "details"},
            exc_info=False,
        )

    def test_warning(
        self, structured_logger: StructuredLogger, mock_logger: MagicMock
    ) -> None:
        """Test warning method logs at WARNING level."""
        structured_logger.warning("warning message")
        mock_logger.warning.assert_called_once_with(
            "warning message",
            extra={"project_id": "test-project"},
            exc_info=False,
        )

    def test_debug(
        self, structured_logger: StructuredLogger, mock_logger: MagicMock
    ) -> None:
        """Test debug method logs at DEBUG level."""
        structured_logger.debug("debug message", extra={"debug_key": "debug_value"})
        mock_logger.debug.assert_called_once_with(
            "debug message",
            extra={"project_id": "test-project", "debug_key": "debug_value"},
            exc_info=False,
        )


class TestStructuredLoggerOperation:
    """Tests for StructuredLogger.operation() context manager."""

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Create a mock Python logger."""
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def structured_logger(self, mock_logger: MagicMock) -> StructuredLogger:
        """Create a StructuredLogger with mock underlying logger."""
        return StructuredLogger(
            logger=mock_logger, default_extra={"project_id": "test"}
        )

    def test_operation_success(
        self, structured_logger: StructuredLogger, mock_logger: MagicMock
    ) -> None:
        """Test operation context manager logs start and completion."""
        with structured_logger.operation("test_op", count=5) as ctx:
            ctx["result"] = "success"

        # Verify info was called twice (start and completion)
        assert mock_logger.info.call_count == 2

        # Check start log
        start_call = mock_logger.info.call_args_list[0]
        assert start_call == call(
            "Starting test_op",
            extra={"project_id": "test", "operation": "test_op", "count": 5},
            exc_info=False,
        )

        # Check completion log (includes updated context)
        completion_call = mock_logger.info.call_args_list[1]
        assert completion_call == call(
            "Completed test_op",
            extra={
                "project_id": "test",
                "operation": "test_op",
                "count": 5,
                "result": "success",
            },
            exc_info=False,
        )

    def test_operation_failure(
        self, structured_logger: StructuredLogger, mock_logger: MagicMock
    ) -> None:
        """Test operation context manager logs error on exception."""
        with pytest.raises(ValueError):
            with structured_logger.operation("failing_op"):
                raise ValueError("Test error")

        # Verify info called for start, error called for failure
        assert mock_logger.info.call_count == 1
        assert mock_logger.error.call_count == 1

        # Check error log
        error_call = mock_logger.error.call_args
        assert "Failed failing_op: Test error" in str(error_call)

    def test_operation_yields_context(
        self, structured_logger: StructuredLogger
    ) -> None:
        """Test operation yields mutable context dict."""
        with structured_logger.operation("test_op", initial="value") as ctx:
            assert ctx["operation"] == "test_op"
            assert ctx["initial"] == "value"
            ctx["added"] = "new_value"
            assert ctx["added"] == "new_value"


class TestCreateLogger:
    """Tests for create_logger function."""

    @patch("fastmcp.utilities.logging.get_logger")
    def test_create_logger_basic(self, mock_get_logger: MagicMock) -> None:
        """Test create_logger creates StructuredLogger with correct config."""
        mock_underlying = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_underlying

        result = create_logger("test_name", "test-project", "DEBUG")

        mock_get_logger.assert_called_once_with("test_name")
        mock_underlying.setLevel.assert_called_once_with(logging.DEBUG)
        assert isinstance(result, StructuredLogger)
        assert result.default_extra == {"project_id": "test-project"}

    @patch("fastmcp.utilities.logging.get_logger")
    def test_create_logger_default_level(self, mock_get_logger: MagicMock) -> None:
        """Test create_logger uses INFO as default level."""
        mock_underlying = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_underlying

        create_logger("test_name", "test-project")

        mock_underlying.setLevel.assert_called_once_with(logging.INFO)

    @patch("fastmcp.utilities.logging.get_logger")
    def test_create_logger_case_insensitive_level(
        self, mock_get_logger: MagicMock
    ) -> None:
        """Test create_logger handles case-insensitive log levels."""
        mock_underlying = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_underlying

        create_logger("test_name", "test-project", "warning")

        mock_underlying.setLevel.assert_called_once_with(logging.WARNING)


class TestFormatFusionScoreStatus:
    """Tests for format_fusion_score_status function."""

    def test_high_score_above_threshold(self) -> None:
        """Test high score (>= 0.8) above threshold."""
        status, interpretation = format_fusion_score_status(0.9, 0.5)
        assert status == "KEEP"
        assert interpretation == "High fusion confidence"

    def test_high_score_at_boundary(self) -> None:
        """Test score at 0.8 boundary."""
        status, interpretation = format_fusion_score_status(0.8, 0.5)
        assert status == "KEEP"
        assert interpretation == "High fusion confidence"

    def test_moderate_score(self) -> None:
        """Test moderate score (0.6-0.8)."""
        status, interpretation = format_fusion_score_status(0.7, 0.5)
        assert status == "KEEP"
        assert interpretation == "Moderate fusion confidence"

    def test_low_score_above_threshold(self) -> None:
        """Test low score (0.4-0.6) above threshold."""
        status, interpretation = format_fusion_score_status(0.5, 0.4)
        assert status == "KEEP"
        assert interpretation == "Low fusion confidence"

    def test_minimal_score_below_threshold(self) -> None:
        """Test minimal score (< 0.4) below threshold."""
        status, interpretation = format_fusion_score_status(0.3, 0.5)
        assert status == "FILTER"
        assert interpretation == "Minimal fusion overlap"

    def test_score_at_threshold_exact(self) -> None:
        """Test score exactly at threshold."""
        status, interpretation = format_fusion_score_status(0.5, 0.5)
        assert status == "KEEP"

    def test_score_just_below_threshold(self) -> None:
        """Test score just below threshold."""
        status, interpretation = format_fusion_score_status(0.499, 0.5)
        assert status == "FILTER"

    def test_zero_score(self) -> None:
        """Test zero score."""
        status, interpretation = format_fusion_score_status(0.0, 0.5)
        assert status == "FILTER"
        assert interpretation == "Minimal fusion overlap"

    def test_perfect_score(self) -> None:
        """Test perfect score of 1.0."""
        status, interpretation = format_fusion_score_status(1.0, 0.5)
        assert status == "KEEP"
        assert interpretation == "High fusion confidence"
