'''Unit tests for reflectlog.application.utils.metrics module.'''

import threading
import time
from unittest.mock import patch

import pytest

from reflectlog.application.utils.metrics import (
    MetricValue,
    MetricsRegistry,
    timed,
)


# ---------------------------------------------------------------------------
# MetricValue
# ---------------------------------------------------------------------------


class TestMetricValue:
    '''Tests for MetricValue dataclass.'''

    def test_str_without_labels(self) -> None:
        '''String representation without labels omits braces.'''
        mv = MetricValue(value=42.0, timestamp=0.0, labels={})
        assert str(mv) == "42.0"

    def test_str_with_single_label(self) -> None:
        '''String representation with one label includes braces.'''
        mv = MetricValue(value=1.5, timestamp=0.0, labels={"status": "ok"})
        assert str(mv) == '{status="ok"} 1.5'

    def test_str_with_multiple_labels(self) -> None:
        '''Labels appear comma-separated in iteration order.'''
        labels = {"engine": "hybrid", "status": "success"}
        mv = MetricValue(value=3.0, timestamp=0.0, labels=labels)
        result = str(mv)
        assert 'engine="hybrid"' in result
        assert 'status="success"' in result
        assert result.endswith(" 3.0")

    def test_dataclass_fields(self) -> None:
        '''All dataclass fields are accessible.'''
        mv = MetricValue(value=1.0, timestamp=99.9, labels={"a": "b"})
        assert mv.value == 1.0
        assert mv.timestamp == 99.9
        assert mv.labels == {"a": "b"}


# ---------------------------------------------------------------------------
# MetricsRegistry — initialisation & helpers
# ---------------------------------------------------------------------------


class TestMetricsRegistryInit:
    '''Tests for MetricsRegistry.__init__ and _make_label_key.'''

    def test_initial_state_is_empty(self) -> None:
        '''Freshly created registry has no metrics.'''
        reg = MetricsRegistry()
        stats = reg.get_stats()
        assert stats["counters"] == {}
        assert stats["gauges"] == {}
        assert stats["histograms"] == {}

    def test_make_label_key_empty(self) -> None:
        '''Empty labels produce empty string key.'''
        reg = MetricsRegistry()
        assert reg._make_label_key({}) == ""

    def test_make_label_key_single(self) -> None:
        '''Single label produces simple key.'''
        reg = MetricsRegistry()
        assert reg._make_label_key({"status": "ok"}) == "status=ok"

    def test_make_label_key_sorted(self) -> None:
        '''Multiple labels are sorted alphabetically.'''
        reg = MetricsRegistry()
        key = reg._make_label_key({"z": "2", "a": "1"})
        assert key == "a=1,z=2"


# ---------------------------------------------------------------------------
# MetricsRegistry — counters
# ---------------------------------------------------------------------------


class TestMetricsRegistryCounters:
    '''Tests for increment / get_counter.'''

    @pytest.fixture
    def registry(self) -> MetricsRegistry:
        '''Fresh registry per test.'''
        return MetricsRegistry()

    def test_increment_default(self, registry: MetricsRegistry) -> None:
        '''Default increment adds 1.'''
        registry.increment("ops_total")
        assert registry.get_counter("ops_total") == 1.0

    def test_increment_custom_value(self, registry: MetricsRegistry) -> None:
        '''Custom value is added.'''
        registry.increment("ops_total", value=5.0)
        assert registry.get_counter("ops_total") == 5.0

    def test_increment_accumulates(self, registry: MetricsRegistry) -> None:
        '''Multiple increments accumulate.'''
        registry.increment("ops_total")
        registry.increment("ops_total")
        registry.increment("ops_total", value=3.0)
        assert registry.get_counter("ops_total") == 5.0

    def test_increment_with_labels(self, registry: MetricsRegistry) -> None:
        '''Labelled counters are tracked separately.'''
        registry.increment("ops_total", labels={"status": "ok"})
        registry.increment("ops_total", labels={"status": "err"})
        registry.increment("ops_total", labels={"status": "ok"})

        assert registry.get_counter("ops_total", labels={"status": "ok"}) == 2.0
        assert registry.get_counter("ops_total", labels={"status": "err"}) == 1.0

    def test_get_counter_nonexistent(self, registry: MetricsRegistry) -> None:
        '''Non-existent counter returns 0.'''
        assert registry.get_counter("missing") == 0.0

    def test_get_counter_wrong_labels(self, registry: MetricsRegistry) -> None:
        '''Counter with different labels returns 0.'''
        registry.increment("ops_total", labels={"status": "ok"})
        assert registry.get_counter("ops_total", labels={"status": "bad"}) == 0.0

    def test_increment_no_labels_vs_empty_labels(
        self, registry: MetricsRegistry
    ) -> None:
        '''None labels and empty dict labels map to same key.'''
        registry.increment("c1", labels=None)
        registry.increment("c1", labels={})
        assert registry.get_counter("c1") == 2.0


# ---------------------------------------------------------------------------
# MetricsRegistry — gauges
# ---------------------------------------------------------------------------


class TestMetricsRegistryGauges:
    '''Tests for set / get_gauge.'''

    @pytest.fixture
    def registry(self) -> MetricsRegistry:
        '''Fresh registry per test.'''
        return MetricsRegistry()

    def test_set_and_get(self, registry: MetricsRegistry) -> None:
        '''Set stores value, get retrieves it.'''
        registry.set("index_size", 100.0)
        assert registry.get_gauge("index_size") == 100.0

    def test_set_overwrites(self, registry: MetricsRegistry) -> None:
        '''Successive sets overwrite the value.'''
        registry.set("index_size", 100.0)
        registry.set("index_size", 200.0)
        assert registry.get_gauge("index_size") == 200.0

    def test_set_with_labels(self, registry: MetricsRegistry) -> None:
        '''Labelled gauges tracked separately.'''
        registry.set("cache_hit_rate", 0.8, labels={"cache": "query"})
        registry.set("cache_hit_rate", 0.6, labels={"cache": "embed"})
        assert registry.get_gauge("cache_hit_rate", labels={"cache": "query"}) == 0.8
        assert registry.get_gauge("cache_hit_rate", labels={"cache": "embed"}) == 0.6

    def test_get_gauge_nonexistent(self, registry: MetricsRegistry) -> None:
        '''Non-existent gauge returns 0.'''
        assert registry.get_gauge("missing") == 0.0

    def test_set_no_labels_vs_empty_labels(self, registry: MetricsRegistry) -> None:
        '''None labels and empty dict produce same key.'''
        registry.set("g1", 10.0, labels=None)
        assert registry.get_gauge("g1") == 10.0
        registry.set("g1", 20.0, labels={})
        assert registry.get_gauge("g1") == 20.0


# ---------------------------------------------------------------------------
# MetricsRegistry — histograms
# ---------------------------------------------------------------------------


class TestMetricsRegistryHistograms:
    '''Tests for observe / get_histogram_stats.'''

    @pytest.fixture
    def registry(self) -> MetricsRegistry:
        '''Fresh registry per test.'''
        return MetricsRegistry()

    def test_observe_single(self, registry: MetricsRegistry) -> None:
        '''Single observation returns correct stats.'''
        registry.observe("latency", 0.5)
        stats = registry.get_histogram_stats("latency")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["sum"] == pytest.approx(0.5)
        assert stats["avg"] == pytest.approx(0.5)
        assert stats["min"] == pytest.approx(0.5)
        assert stats["max"] == pytest.approx(0.5)

    def test_observe_multiple(self, registry: MetricsRegistry) -> None:
        '''Multiple observations produce correct aggregate stats.'''
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            registry.observe("latency", v)

        stats = registry.get_histogram_stats("latency")
        assert stats is not None
        assert stats["count"] == 5
        assert stats["sum"] == pytest.approx(1.5)
        assert stats["avg"] == pytest.approx(0.3)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.5)

    def test_observe_with_labels(self, registry: MetricsRegistry) -> None:
        '''Labelled histograms tracked separately.'''
        registry.observe("latency", 0.1, labels={"engine": "usearch"})
        registry.observe("latency", 0.5, labels={"engine": "tantivy"})

        us = registry.get_histogram_stats("latency", labels={"engine": "usearch"})
        tv = registry.get_histogram_stats("latency", labels={"engine": "tantivy"})
        assert us is not None and us["avg"] == pytest.approx(0.1)
        assert tv is not None and tv["avg"] == pytest.approx(0.5)

    def test_get_histogram_stats_no_data(self, registry: MetricsRegistry) -> None:
        '''Non-existent histogram returns None.'''
        assert registry.get_histogram_stats("missing") is None

    def test_get_histogram_stats_empty_after_access(
        self, registry: MetricsRegistry
    ) -> None:
        '''Accessing a defaultdict-created but empty list returns None.'''
        # Access the key to trigger defaultdict creation
        registry._histograms["empty"][""]  # noqa: B018
        assert registry.get_histogram_stats("empty") is None


# ---------------------------------------------------------------------------
# MetricsRegistry — timer context manager
# ---------------------------------------------------------------------------


class TestMetricsRegistryTimer:
    '''Tests for timer context manager.'''

    def test_timer_records_duration(self) -> None:
        '''Timer records a positive duration.'''
        registry = MetricsRegistry()
        with registry.timer("op_duration"):
            time.sleep(0.01)

        stats = registry.get_histogram_stats("op_duration")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["sum"] > 0

    def test_timer_with_labels(self) -> None:
        '''Timer passes labels through to observe.'''
        registry = MetricsRegistry()
        with registry.timer("op_duration", labels={"op": "search"}):
            pass

        stats = registry.get_histogram_stats("op_duration", labels={"op": "search"})
        assert stats is not None
        assert stats["count"] == 1

    def test_timer_records_on_exception(self) -> None:
        '''Duration recorded even if body raises.'''
        registry = MetricsRegistry()
        with pytest.raises(ValueError, match="boom"):
            with registry.timer("op_duration"):
                raise ValueError("boom")

        stats = registry.get_histogram_stats("op_duration")
        assert stats is not None
        assert stats["count"] == 1


# ---------------------------------------------------------------------------
# MetricsRegistry — reset
# ---------------------------------------------------------------------------


class TestMetricsRegistryReset:
    '''Tests for reset.'''

    def test_reset_clears_all(self) -> None:
        '''Reset clears counters, gauges, and histograms.'''
        registry = MetricsRegistry()
        registry.increment("c1")
        registry.set("g1", 10.0)
        registry.observe("h1", 0.5)

        registry.reset()

        assert registry.get_counter("c1") == 0.0
        assert registry.get_gauge("g1") == 0.0
        assert registry.get_histogram_stats("h1") is None

    def test_reset_clears_stats(self) -> None:
        '''get_stats returns empty after reset.'''
        registry = MetricsRegistry()
        registry.increment("c1")
        registry.reset()
        stats = registry.get_stats()
        assert stats["counters"] == {}
        assert stats["gauges"] == {}
        assert stats["histograms"] == {}


# ---------------------------------------------------------------------------
# MetricsRegistry — export_prometheus
# ---------------------------------------------------------------------------


class TestMetricsRegistryExportPrometheus:
    '''Tests for Prometheus text export.'''

    @pytest.fixture
    def registry(self) -> MetricsRegistry:
        '''Fresh registry per test.'''
        return MetricsRegistry()

    def test_empty_export(self, registry: MetricsRegistry) -> None:
        '''Empty registry exports empty string.'''
        assert registry.export_prometheus() == ""

    def test_counter_no_labels(self, registry: MetricsRegistry) -> None:
        '''Counter without labels exports value line.'''
        registry.increment("requests_total", value=5.0)
        output = registry.export_prometheus()
        assert "# HELP requests_total Counter metric" in output
        assert "# TYPE requests_total counter" in output
        assert "requests_total 5.0" in output

    def test_counter_with_labels(self, registry: MetricsRegistry) -> None:
        '''Counter with labels includes label set in export.'''
        registry.increment("requests_total", labels={"status": "ok"})
        output = registry.export_prometheus()
        assert 'requests_total{status="ok"} 1.0' in output

    def test_gauge_no_labels(self, registry: MetricsRegistry) -> None:
        '''Gauge without labels exported correctly.'''
        registry.set("index_size", 42.0)
        output = registry.export_prometheus()
        assert "# TYPE index_size gauge" in output
        assert "index_size 42.0" in output

    def test_gauge_with_labels(self, registry: MetricsRegistry) -> None:
        '''Gauge with labels exported correctly.'''
        registry.set("cache_rate", 0.95, labels={"cache": "query"})
        output = registry.export_prometheus()
        assert 'cache_rate{cache="query"} 0.95' in output

    def test_histogram_no_labels(self, registry: MetricsRegistry) -> None:
        '''Histogram without labels exports quantiles, count, sum.'''
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            registry.observe("latency", v)

        output = registry.export_prometheus()
        assert "# TYPE latency histogram" in output
        assert 'latency{quantile="0.5"}' in output
        assert 'latency{quantile="0.9"}' in output
        assert 'latency{quantile="0.95"}' in output
        assert 'latency{quantile="0.99"}' in output
        assert "latency_count 5" in output
        assert "latency_sum" in output

    def test_histogram_with_labels(self, registry: MetricsRegistry) -> None:
        '''Histogram with labels exports quantiles with label set.'''
        registry.observe("latency", 0.3, labels={"engine": "usearch"})
        output = registry.export_prometheus()
        assert 'engine="usearch"' in output
        assert 'quantile="0.5"' in output

    def test_histogram_with_labels_count_sum(self, registry: MetricsRegistry) -> None:
        '''Histogram with labels exports count and sum with label set.'''
        registry.observe("latency", 0.3, labels={"engine": "usearch"})
        output = registry.export_prometheus()
        # With labels, count/sum lines include the label set
        assert "_count}" in output
        assert "_sum}" in output

    def test_export_multiple_metric_types(self, registry: MetricsRegistry) -> None:
        '''All three metric types appear in single export.'''
        registry.increment("c1")
        registry.set("g1", 5.0)
        registry.observe("h1", 0.1)

        output = registry.export_prometheus()
        assert "# TYPE c1 counter" in output
        assert "# TYPE g1 gauge" in output
        assert "# TYPE h1 histogram" in output

    def test_export_sorted_by_name(self, registry: MetricsRegistry) -> None:
        '''Metrics within each type are sorted alphabetically.'''
        registry.increment("z_counter")
        registry.increment("a_counter")
        output = registry.export_prometheus()
        a_pos = output.index("a_counter")
        z_pos = output.index("z_counter")
        assert a_pos < z_pos

    def test_counter_multiple_label_sets(self, registry: MetricsRegistry) -> None:
        '''Multiple label sets for the same counter exported.'''
        registry.increment("req", labels={"method": "GET"})
        registry.increment("req", labels={"method": "POST"})
        output = registry.export_prometheus()
        assert 'method="GET"' in output
        assert 'method="POST"' in output

    def test_histogram_empty_values_skipped(self, registry: MetricsRegistry) -> None:
        '''Histogram label set with empty values list is skipped in export.'''
        # Force-create a histogram entry with an empty list
        registry._histograms["latency"][""]  # noqa: B018
        output = registry.export_prometheus()
        # HELP/TYPE headers appear because label_sets is non-empty,
        # but no quantile/count/sum lines are emitted
        assert "quantile" not in output


# ---------------------------------------------------------------------------
# MetricsRegistry — get_stats
# ---------------------------------------------------------------------------


class TestMetricsRegistryGetStats:
    '''Tests for get_stats summary.'''

    def test_get_stats_counters(self) -> None:
        '''Counters appear in stats.'''
        reg = MetricsRegistry()
        reg.increment("c1")
        reg.increment("c1", labels={"l": "v"})
        stats = reg.get_stats()
        assert "c1" in stats["counters"]
        assert "" in stats["counters"]["c1"]
        assert "l=v" in stats["counters"]["c1"]

    def test_get_stats_gauges(self) -> None:
        '''Gauges appear in stats.'''
        reg = MetricsRegistry()
        reg.set("g1", 10.0)
        stats = reg.get_stats()
        assert "g1" in stats["gauges"]
        assert stats["gauges"]["g1"][""] == 10.0

    def test_get_stats_histograms(self) -> None:
        '''Histograms appear with count/sum/avg.'''
        reg = MetricsRegistry()
        reg.observe("h1", 0.2)
        reg.observe("h1", 0.4)
        stats = reg.get_stats()
        h = stats["histograms"]["h1"][""]
        assert h["count"] == 2
        assert h["sum"] == pytest.approx(0.6)
        assert h["avg"] == pytest.approx(0.3)

    def test_get_stats_histogram_empty_values(self) -> None:
        '''Histogram with empty values list reports avg=0.'''
        reg = MetricsRegistry()
        # Force defaultdict creation without any observations
        reg._histograms["empty"][""]  # noqa: B018
        stats = reg.get_stats()
        h = stats["histograms"]["empty"][""]
        assert h["count"] == 0
        assert h["avg"] == 0


# ---------------------------------------------------------------------------
# MetricsRegistry — thread safety
# ---------------------------------------------------------------------------


class TestMetricsRegistryThreadSafety:
    '''Basic thread safety tests.'''

    def test_concurrent_increments(self) -> None:
        '''Concurrent increments produce correct total.'''
        registry = MetricsRegistry()
        n_threads = 10
        n_per_thread = 1000
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(n_per_thread):
                registry.increment("total")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert registry.get_counter("total") == n_threads * n_per_thread

    def test_concurrent_mixed_ops(self) -> None:
        '''Concurrent mixed operations don't crash.'''
        registry = MetricsRegistry()
        barrier = threading.Barrier(4)

        def inc_worker() -> None:
            barrier.wait()
            for _ in range(100):
                registry.increment("c1")

        def set_worker() -> None:
            barrier.wait()
            for i in range(100):
                registry.set("g1", float(i))

        def obs_worker() -> None:
            barrier.wait()
            for i in range(100):
                registry.observe("h1", float(i) * 0.01)

        def read_worker() -> None:
            barrier.wait()
            for _ in range(100):
                registry.get_stats()
                registry.export_prometheus()

        threads = [
            threading.Thread(target=inc_worker),
            threading.Thread(target=set_worker),
            threading.Thread(target=obs_worker),
            threading.Thread(target=read_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No crash and counter is correct
        assert registry.get_counter("c1") == 100.0


# ---------------------------------------------------------------------------
# timed decorator
# ---------------------------------------------------------------------------


class TestTimedDecorator:
    '''Tests for the timed() decorator function.'''

    def test_timed_records_duration(self) -> None:
        '''Decorated function records a histogram observation.'''
        registry = MetricsRegistry()

        @timed(registry, "func_duration")
        def my_func() -> str:
            return "ok"

        result = my_func()
        assert result == "ok"

        stats = registry.get_histogram_stats("func_duration")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["sum"] > 0

    def test_timed_with_labels(self) -> None:
        '''Decorated function with labels records in correct bucket.'''
        registry = MetricsRegistry()

        @timed(registry, "func_duration", labels={"op": "search"})
        def my_func() -> int:
            return 42

        my_func()
        stats = registry.get_histogram_stats("func_duration", labels={"op": "search"})
        assert stats is not None
        assert stats["count"] == 1

    def test_timed_passes_args(self) -> None:
        '''Decorated function receives positional and keyword args.'''
        registry = MetricsRegistry()

        @timed(registry, "func_duration")
        def add(a: int, b: int, offset: int = 0) -> int:
            return a + b + offset

        assert add(1, 2, offset=10) == 13

    def test_timed_propagates_exception(self) -> None:
        '''Exception propagates and duration is still recorded.'''
        registry = MetricsRegistry()

        @timed(registry, "func_duration")
        def boom() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            boom()

        stats = registry.get_histogram_stats("func_duration")
        assert stats is not None
        assert stats["count"] == 1

    def test_timed_multiple_calls(self) -> None:
        '''Multiple calls accumulate observations.'''
        registry = MetricsRegistry()

        @timed(registry, "func_duration")
        def noop() -> None:
            pass

        for _ in range(5):
            noop()

        stats = registry.get_histogram_stats("func_duration")
        assert stats is not None
        assert stats["count"] == 5

    def test_timed_without_labels(self) -> None:
        '''Decorator works when labels=None (default).'''
        registry = MetricsRegistry()

        @timed(registry, "func_duration", labels=None)
        def noop() -> None:
            pass

        noop()
        stats = registry.get_histogram_stats("func_duration")
        assert stats is not None
        assert stats["count"] == 1
