"""Structured metrics collection for ReflectLogMCP Server.

This module provides a Prometheus-style metrics collection system for monitoring
the health and performance of the memory management system.

Metrics are tracked for:
- Operation counts (add, search, remove)
- Operation latencies (histograms)
- Error rates
- Index sizes
- Cache hit rates
"""

from collections import defaultdict
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass
class MetricValue:
    """A single metric value with timestamp."""

    value: float
    timestamp: float
    labels: dict[str, str]

    def __str__(self) -> str:
        """String representation for Prometheus export."""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if labels_str:
            return f"{{{labels_str}}} {self.value}"
        return f"{self.value}"


class MetricsRegistry:
    """Thread-safe registry for collecting application metrics.

    This class tracks various metrics for monitoring the health and performance
    of the ReflectLogMCP server. Metrics can be exported in Prometheus format.

    Example:
        ```python
        metrics = MetricsRegistry()

        # Record a metric
        metrics.increment("add_messages_count", labels={"status": "success"})

        # Time an operation
        with metrics.timer("search_duration", labels={"engine": "hybrid"}):
            results = memory_manager.search(query)

        # Export metrics
        prometheus_text = metrics.export_prometheus()
        ```
    """

    def __init__(self) -> None:
        """Initialize the metrics registry."""
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._histograms: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._label_keys: dict[str, set[str]] = defaultdict(set)

    def _make_label_key(self, labels: dict[str, str]) -> str:
        """Create a key from labels dictionary.

        Args:
            labels: Dictionary of label names to values

        Returns:
            A string key suitable for dictionary lookups
        """
        if not labels:
            return ""
        # Sort labels for consistent keys
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric.

        Counters are monotonically increasing values used for things like
        request counts, error counts, etc.

        Args:
            name: Metric name (e.g., "add_messages_total")
            value: Amount to increment by (default: 1.0)
            labels: Optional labels for this metric (e.g., {"status": "success"})
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)

        with self._lock:
            self._counters[name][label_key] += value
            # Track label keys for export
            for key in labels:
                self._label_keys[name].add(key)

    def set(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric.

        Gauges can go up or down and are used for things like current index size,
        cache hit rate, etc.

        Args:
            name: Metric name (e.g., "index_size")
            value: Value to set
            labels: Optional labels for this metric
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)

        with self._lock:
            self._gauges[name][label_key] = value
            # Track label keys for export
            for key in labels:
                self._label_keys[name].add(key)

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a value in a histogram metric.

        Histograms track distributions of values like operation latencies.
        Summary statistics (count, sum, avg) are computed on export.

        Args:
            name: Metric name (e.g., "search_duration_seconds")
            value: Value to observe (e.g., latency in seconds)
            labels: Optional labels for this metric
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)

        with self._lock:
            self._histograms[name][label_key].append(value)
            # Track label keys for export
            for key in labels:
                self._label_keys[name].add(key)

    @contextlib.contextmanager
    def timer(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ):
        """Context manager for timing operations.

        Records the duration of the wrapped code block in a histogram.

        Args:
            name: Metric name for the timer (e.g., "search_duration_seconds")
            labels: Optional labels for this metric

        Yields:
            None

        Example:
            ```python
            with metrics.timer("add_messages_duration", labels={"batch_size": "10"}):
                memory_manager.add_messages(messages)
            ```
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(name, duration, labels)

    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get the current value of a counter.

        Args:
            name: Metric name
            labels: Optional labels for the metric

        Returns:
            Current counter value
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)

        with self._lock:
            return self._counters[name][label_key]

    def get_gauge(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get the current value of a gauge.

        Args:
            name: Metric name
            labels: Optional labels for the metric

        Returns:
            Current gauge value
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)

        with self._lock:
            return self._gauges[name][label_key]

    def get_histogram_stats(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> dict[str, float] | None:
        """Get summary statistics for a histogram.

        Args:
            name: Metric name
            labels: Optional labels for the metric

        Returns:
            Dictionary with count, sum, avg, min, max, or None if no data
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)

        with self._lock:
            values = self._histograms[name][label_key]
            if not values:
                return None

            return {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    def reset(self) -> None:
        """Reset all metrics to zero/empty.

        This is useful for testing or periodic metric cleanup.
        """
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._label_keys.clear()

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format.

        Returns:
            Prometheus-compatible text representation of all metrics

        Example output:
            ```
            # HELP add_messages_total Total number of add operations
            # TYPE add_messages_total counter
            add_messages_total{status="success"} 42.0
            add_messages_total{status="error"} 1.0

            # HELP search_duration_seconds Search operation duration
            # TYPE search_duration_seconds histogram
            search_duration_seconds{quantile="0.5"} 0.123
            search_duration_seconds{quantile="0.9"} 0.456
            search_duration_seconds_count 100
            search_duration_seconds_sum 12.3
            ```
        """
        lines: list[str] = []

        with self._lock:
            # Export counters
            for name, label_sets in sorted(self._counters.items()):
                if label_sets:
                    lines.append(f"# HELP {name} Counter metric")
                    lines.append(f"# TYPE {name} counter")
                    for label_key, value in sorted(label_sets.items()):
                        if label_key:
                            # Parse back the labels from the key
                            labels = dict(
                                item.split("=") for item in label_key.split(",")
                            )
                            labels_str = ",".join(
                                f'{k}="{v}"' for k, v in sorted(labels.items())
                            )
                            lines.append(f"{name}{{{labels_str}}} {value}")
                        else:
                            lines.append(f"{name} {value}")

            # Export gauges
            for name, label_sets in sorted(self._gauges.items()):
                if label_sets:
                    lines.append(f"# HELP {name} Gauge metric")
                    lines.append(f"# TYPE {name} gauge")
                    for label_key, value in sorted(label_sets.items()):
                        if label_key:
                            labels = dict(
                                item.split("=") for item in label_key.split(",")
                            )
                            labels_str = ",".join(
                                f'{k}="{v}"' for k, v in sorted(labels.items())
                            )
                            lines.append(f"{name}{{{labels_str}}} {value}")
                        else:
                            lines.append(f"{name} {value}")

            # Export histograms
            for name, label_sets in sorted(self._histograms.items()):
                if label_sets:
                    lines.append(f"# HELP {name} Histogram metric")
                    lines.append(f"# TYPE {name} histogram")
                    for label_key, values in sorted(label_sets.items()):
                        if not values:
                            continue

                        values_sorted = sorted(values)
                        count = len(values_sorted)
                        total = sum(values_sorted)

                        # Calculate quantiles
                        quantiles = [0.5, 0.9, 0.95, 0.99]
                        for q in quantiles:
                            idx = int(q * (count - 1))
                            quantile_value = values_sorted[max(0, min(idx, count - 1))]
                            if label_key:
                                labels = dict(
                                    item.split("=") for item in label_key.split(",")
                                )
                                labels_str = ",".join(
                                    f'{k}="{v}"' for k, v in sorted(labels.items())
                                )
                                lines.append(
                                    f'{name}{{{labels_str},quantile="{q}"}} {quantile_value}'
                                )
                            else:
                                lines.append(
                                    f'{name}{{quantile="{q}"}} {quantile_value}'
                                )

                        # Export count and sum
                        if label_key:
                            labels = dict(
                                item.split("=") for item in label_key.split(",")
                            )
                            labels_str = ",".join(
                                f'{k}="{v}"' for k, v in sorted(labels.items())
                            )
                            lines.append(f"{name}{{{labels_str}_count}} {count}")
                            lines.append(f"{name}{{{labels_str}_sum}} {total}")
                        else:
                            lines.append(f"{name}_count {count}")
                            lines.append(f"{name}_sum {total}")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get a summary of all tracked metrics.

        Returns:
            Dictionary containing metric names and their label sets
        """
        with self._lock:
            return {
                "counters": {
                    name: dict(label_sets)
                    for name, label_sets in self._counters.items()
                },
                "gauges": {
                    name: dict(label_sets) for name, label_sets in self._gauges.items()
                },
                "histograms": {
                    name: {
                        label_key: {
                            "count": len(values),
                            "sum": sum(values),
                            "avg": sum(values) / len(values) if values else 0,
                        }
                        for label_key, values in label_sets.items()
                    }
                    for name, label_sets in self._histograms.items()
                },
            }


def timed(
    metrics: MetricsRegistry,
    name: str,
    labels: dict[str, str] | None = None,
) -> Callable:
    """Decorator for timing function execution.

    Args:
        metrics: MetricsRegistry instance
        name: Metric name for the timer
        labels: Optional labels for this metric

    Returns:
        Decorator function

    Example:
        ```python
        metrics = MetricsRegistry()

        @timed(metrics, "search_duration", labels={"engine": "hybrid"})
        def search(query: str) -> list[str]:
            return memory_manager.search(query)
        ```
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with metrics.timer(name, labels):
                return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "MetricValue",
    "MetricsRegistry",
    "timed",
]
