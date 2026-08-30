"""Unit tests for RerankerPostProcessor."""

from typing import cast
from unittest.mock import MagicMock

import pytest

from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.reranker_post_processor import RerankerPostProcessor


def create_mock_logger() -> tuple[IStructuredLogger, MagicMock]:
    """Create a properly typed mock logger and its mock reference for testing."""
    mock = MagicMock(spec=StructuredLogger)
    return cast(IStructuredLogger, mock), mock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Test __init__ and slot storage."""

    def test_stores_all_parameters(self) -> None:
        """Test that all constructor parameters are stored correctly."""
        logger, _ = create_mock_logger()
        pp = RerankerPostProcessor(
            min_results=3,
            batch_normalize=True,
            logger=logger,
            normalize_log_level="info",
        )
        assert pp._min_results == 3
        assert pp._batch_normalize is True
        assert pp._logger is logger
        assert pp._normalize_log_level == "info"

    def test_defaults(self) -> None:
        """Test default optional parameters."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        assert pp._logger is None
        assert pp._normalize_log_level == "debug"

    def test_slots_defined(self) -> None:
        """Test that __slots__ are defined (no __dict__)."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        assert not hasattr(pp, "__dict__")


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


class TestNormalize:
    """Test batch min-max normalization step."""

    def test_empty_scores_returns_empty(self) -> None:
        """Empty list is returned as-is regardless of batch_normalize."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=True)
        assert pp.normalize([]) == []

    def test_batch_normalize_disabled_returns_unchanged(self) -> None:
        """When batch_normalize=False, scores pass through unchanged."""
        scores: list[tuple[str, float]] = [("a", 0.3), ("b", 0.7)]
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        assert pp.normalize(scores) is scores  # identity — same object

    def test_single_score_normalizes_to_one(self) -> None:
        """A single result normalizes to score 1.0."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=True)
        result = pp.normalize([("doc", 0.42)])
        assert len(result) == 1
        assert result[0][0] == "doc"
        assert result[0][1] == pytest.approx(1.0)

    def test_two_scores_minmax(self) -> None:
        """Two scores → min maps to 0, max maps to 1."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=True)
        result = pp.normalize([("low", 0.2), ("high", 0.8)])
        lookup = dict(result)
        assert lookup["low"] == pytest.approx(0.0)
        assert lookup["high"] == pytest.approx(1.0)

    def test_three_scores_proportional(self) -> None:
        """Middle value normalizes proportionally."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=True)
        result = pp.normalize([("a", 0.0), ("b", 0.5), ("c", 1.0)])
        lookup = dict(result)
        assert lookup["a"] == pytest.approx(0.0)
        assert lookup["b"] == pytest.approx(0.5)
        assert lookup["c"] == pytest.approx(1.0)

    def test_all_equal_scores(self) -> None:
        """All equal scores normalize to 1.0 (equally good)."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=True)
        result = pp.normalize([("a", 0.5), ("b", 0.5), ("c", 0.5)])
        for _, score in result:
            assert score == pytest.approx(1.0)

    def test_logs_when_logger_present(self) -> None:
        """Logger is called with the configured log level after normalization."""
        logger, mock = create_mock_logger()
        pp = RerankerPostProcessor(
            min_results=1,
            batch_normalize=True,
            logger=logger,
            normalize_log_level="debug",
        )
        pp.normalize([("a", 0.2), ("b", 0.8)])
        mock.debug.assert_called_once()
        args, kwargs = mock.debug.call_args
        assert "Batch normalization" in args[0]
        assert kwargs["extra"]["batch_normalize"] is True

    def test_logs_with_info_level(self) -> None:
        """Configurable log level routes to the correct logger method."""
        logger, mock = create_mock_logger()
        pp = RerankerPostProcessor(
            min_results=1,
            batch_normalize=True,
            logger=logger,
            normalize_log_level="info",
        )
        pp.normalize([("a", 0.1), ("b", 0.9)])
        mock.info.assert_called_once()
        mock.debug.assert_not_called()

    def test_logs_with_warning_level(self) -> None:
        """Warning-level logging works via getattr dispatch."""
        logger, mock = create_mock_logger()
        pp = RerankerPostProcessor(
            min_results=1,
            batch_normalize=True,
            logger=logger,
            normalize_log_level="warning",
        )
        pp.normalize([("x", 0.3), ("y", 0.7)])
        mock.warning.assert_called_once()

    def test_no_logging_without_logger(self) -> None:
        """No error when logger is None."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=True)
        # Should not raise
        result = pp.normalize([("a", 0.2), ("b", 0.8)])
        assert len(result) == 2

    def test_log_extra_contains_raw_range(self) -> None:
        """Log extra dict contains raw_min and raw_max."""
        logger, mock = create_mock_logger()
        pp = RerankerPostProcessor(
            min_results=1,
            batch_normalize=True,
            logger=logger,
        )
        pp.normalize([("a", 0.1), ("b", 0.5), ("c", 0.9)])
        extra = mock.debug.call_args.kwargs["extra"]
        assert extra["raw_min"] == pytest.approx(0.1)
        assert extra["raw_max"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# apply_decay()
# ---------------------------------------------------------------------------


class TestApplyDecay:
    """Test recency decay step."""

    def test_empty_scores_returns_empty(self) -> None:
        """Empty input is returned as-is."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        result = pp.apply_decay([], {"a": "2024-01-01T00:00:00+00:00"}, 0.01, True)
        assert result == []

    def test_disabled_returns_unchanged(self) -> None:
        """When enabled=False, scores pass through unchanged."""
        scores: list[tuple[str, float]] = [("a", 0.9)]
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        result = pp.apply_decay(scores, {"a": "2024-01-01T00:00:00+00:00"}, 0.01, False)
        assert result is scores

    def test_zero_decay_rate_returns_unchanged(self) -> None:
        """When decay_rate=0, scores pass through unchanged."""
        scores: list[tuple[str, float]] = [("a", 0.9)]
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        result = pp.apply_decay(scores, {"a": "2024-01-01T00:00:00+00:00"}, 0.0, True)
        assert result is scores

    def test_none_timestamp_map_returns_unchanged(self) -> None:
        """When timestamp_map is None, scores pass through unchanged."""
        scores: list[tuple[str, float]] = [("a", 0.9)]
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        result = pp.apply_decay(scores, None, 0.01, True)
        assert result is scores

    def test_empty_timestamp_map_returns_unchanged(self) -> None:
        """When timestamp_map is empty dict, scores pass through unchanged."""
        scores: list[tuple[str, float]] = [("a", 0.9)]
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        result = pp.apply_decay(scores, {}, 0.01, True)
        assert result is scores

    def test_decay_reduces_scores(self) -> None:
        """Decay applied to old documents reduces their scores."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        scores: list[tuple[str, float]] = [("old", 0.9), ("new", 0.8)]
        timestamps = {
            "old": "2020-01-01T00:00:00+00:00",
            "new": "2026-04-11T00:00:00+00:00",
        }
        result = pp.apply_decay(scores, timestamps, 0.01, True)
        lookup = dict(result)
        # Old doc should have decayed significantly
        assert lookup["old"] < 0.9
        # New doc should retain most of its score
        assert lookup["new"] <= 0.8

    def test_decay_reorders_by_decayed_score(self) -> None:
        """Results are sorted by decayed score descending."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        scores: list[tuple[str, float]] = [("old", 0.95), ("new", 0.8)]
        timestamps = {
            "old": "2020-01-01T00:00:00+00:00",
            "new": "2026-04-11T00:00:00+00:00",
        }
        result = pp.apply_decay(scores, timestamps, 0.01, True)
        # New doc should now rank first because old decayed heavily
        assert result[0][0] == "new"

    def test_decay_logs_when_logger_present(self) -> None:
        """Logger.debug is called when decay is applied."""
        logger, mock = create_mock_logger()
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False, logger=logger)
        scores: list[tuple[str, float]] = [("a", 0.9)]
        timestamps = {"a": "2020-01-01T00:00:00+00:00"}
        pp.apply_decay(scores, timestamps, 0.01, True)
        mock.debug.assert_called_once()
        args, kwargs = mock.debug.call_args
        assert "Recency decay" in args[0]
        assert kwargs["extra"]["recency_decay"] is True
        assert kwargs["extra"]["decay_rate"] == 0.01

    def test_decay_no_logging_without_logger(self) -> None:
        """No error when logger is None and decay is applied."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        scores: list[tuple[str, float]] = [("a", 0.9)]
        timestamps = {"a": "2020-01-01T00:00:00+00:00"}
        # Should not raise
        result = pp.apply_decay(scores, timestamps, 0.01, True)
        assert len(result) == 1

    def test_negative_decay_rate_returns_unchanged(self) -> None:
        """Negative decay_rate does not satisfy > 0 guard."""
        scores: list[tuple[str, float]] = [("a", 0.9)]
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        result = pp.apply_decay(scores, {"a": "2020-01-01T00:00:00+00:00"}, -0.01, True)
        assert result is scores


# ---------------------------------------------------------------------------
# filter_by_threshold()
# ---------------------------------------------------------------------------


class TestFilterByThreshold:
    """Test threshold filtering with safety-net minimum."""

    def test_empty_scores_returns_empty(self) -> None:
        """Empty input returns empty list."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        assert pp.filter_by_threshold([], 0.5) == []

    def test_all_above_threshold(self) -> None:
        """All scores above threshold are retained."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        scores: list[tuple[str, float]] = [("a", 0.9), ("b", 0.7), ("c", 0.6)]
        result = pp.filter_by_threshold(scores, 0.5)
        assert len(result) == 3

    def test_some_below_threshold(self) -> None:
        """Only scores >= threshold are returned."""
        pp = RerankerPostProcessor(min_results=0, batch_normalize=False)
        scores: list[tuple[str, float]] = [("a", 0.9), ("b", 0.4), ("c", 0.2)]
        result = pp.filter_by_threshold(scores, 0.5)
        assert len(result) == 1
        assert result[0][0] == "a"

    def test_all_below_threshold_no_safety_net(self) -> None:
        """All filtered out when min_results=0 (no safety net)."""
        pp = RerankerPostProcessor(min_results=0, batch_normalize=False)
        scores: list[tuple[str, float]] = [("a", 0.3), ("b", 0.2)]
        result = pp.filter_by_threshold(scores, 0.5)
        assert result == []

    def test_safety_net_returns_min_results(self) -> None:
        """Safety net returns top min_results when all below threshold."""
        pp = RerankerPostProcessor(min_results=2, batch_normalize=False)
        scores: list[tuple[str, float]] = [("a", 0.4), ("b", 0.3), ("c", 0.2)]
        result = pp.filter_by_threshold(scores, 0.9)
        assert len(result) == 2
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_safety_net_not_enough_candidates(self) -> None:
        """When fewer candidates than min_results, return all available."""
        pp = RerankerPostProcessor(min_results=5, batch_normalize=False)
        scores: list[tuple[str, float]] = [("a", 0.3), ("b", 0.2)]
        result = pp.filter_by_threshold(scores, 0.9)
        assert len(result) == 2

    def test_single_score_above_threshold(self) -> None:
        """Single score above threshold is kept."""
        pp = RerankerPostProcessor(min_results=0, batch_normalize=False)
        result = pp.filter_by_threshold([("a", 0.8)], 0.5)
        assert len(result) == 1

    def test_single_score_below_threshold_no_safety(self) -> None:
        """Single score below threshold with min_results=0 returns empty."""
        pp = RerankerPostProcessor(min_results=0, batch_normalize=False)
        result = pp.filter_by_threshold([("a", 0.3)], 0.5)
        assert result == []

    def test_single_score_below_threshold_with_safety(self) -> None:
        """Single score below threshold with safety net returns it."""
        pp = RerankerPostProcessor(min_results=1, batch_normalize=False)
        result = pp.filter_by_threshold([("a", 0.3)], 0.5)
        assert len(result) == 1
        assert result[0][0] == "a"

    def test_exact_threshold_included(self) -> None:
        """Score equal to threshold is included (>=)."""
        pp = RerankerPostProcessor(min_results=0, batch_normalize=False)
        result = pp.filter_by_threshold([("a", 0.5)], 0.5)
        assert len(result) == 1

    def test_zero_threshold_keeps_all(self) -> None:
        """Threshold of 0.0 keeps everything."""
        pp = RerankerPostProcessor(min_results=0, batch_normalize=False)
        scores: list[tuple[str, float]] = [("a", 0.9), ("b", 0.1), ("c", 0.0)]
        result = pp.filter_by_threshold(scores, 0.0)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Integration: full pipeline sequence
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Test normalize → decay → filter used together."""

    def test_full_pipeline(self) -> None:
        """Three steps compose correctly in sequence."""
        logger, mock = create_mock_logger()
        pp = RerankerPostProcessor(
            min_results=1,
            batch_normalize=True,
            logger=logger,
            normalize_log_level="debug",
        )

        raw: list[tuple[str, float]] = [("a", 0.1), ("b", 0.5), ("c", 0.9)]
        timestamps = {
            "a": "2026-04-11T00:00:00+00:00",
            "b": "2026-04-11T00:00:00+00:00",
            "c": "2026-04-11T00:00:00+00:00",
        }

        # Step 1: normalize
        normalized = pp.normalize(raw)
        assert len(normalized) == 3

        # Step 2: decay (recent timestamps, minimal effect)
        decayed = pp.apply_decay(normalized, timestamps, 0.01, True)
        assert len(decayed) == 3

        # Step 3: filter
        filtered = pp.filter_by_threshold(decayed, 0.3)
        assert len(filtered) >= 1

        # Logger was invoked for both normalize and decay
        assert mock.debug.call_count == 2
