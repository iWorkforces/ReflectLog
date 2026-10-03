"""Unit tests for HealthCheckTool."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.health_check import HealthCheckTool
from reflectlog.core.enums import EngineReadiness, HealthStatus
from reflectlog.core.search_health import (
    ComponentFailureSnapshot,
    SearchFailureSnapshot,
)


@pytest.mark.unit
class TestHealthCheckToolStatus:
    """Health check must use the public readiness API."""

    @pytest.mark.parametrize("eager", [False, True])
    @pytest.mark.parametrize("pending", [0, 2])
    @pytest.mark.parametrize(
        "semantic",
        [
            EngineReadiness.INITIALIZED,
            EngineReadiness.PENDING,
            EngineReadiness.NOT_INITIALIZED,
        ],
    )
    @pytest.mark.parametrize(
        "tantivy",
        [
            EngineReadiness.INITIALIZED,
            EngineReadiness.PENDING,
            EngineReadiness.NOT_INITIALIZED,
            EngineReadiness.DISABLED,
        ],
    )
    async def test_existing_status_truth_table(
        self,
        mock_config: Config,
        mock_memory_manager: MagicMock,
        mock_tool_logger: MagicMock,
        eager: bool,
        pending: int,
        semantic: EngineReadiness,
        tantivy: EngineReadiness,
    ) -> None:
        config = replace(mock_config, eager_initialization=eager)
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": semantic,
            "tantivy_engine": tantivy,
        }
        mock_memory_manager.pending_intent_count.return_value = pending
        expected = HealthStatus.DEGRADED if pending else HealthStatus.HEALTHY
        if semantic in {
            EngineReadiness.PENDING,
            EngineReadiness.NOT_INITIALIZED,
        } or tantivy in {EngineReadiness.PENDING, EngineReadiness.NOT_INITIALIZED}:
            expected = HealthStatus.UNHEALTHY if eager else HealthStatus.DEGRADED

        result = await HealthCheckTool(
            config, mock_memory_manager, mock_tool_logger
        ).get_handler()()

        assert result["status"] == expected
        assert result["pending_intent_count"] == pending
        assert result["hybrid_search_enabled"] == (tantivy != EngineReadiness.DISABLED)
        mock_memory_manager.reconcile_pending_replacements.assert_not_called()

    async def test_reports_search_failures_without_changing_status(
        self,
        mock_config: Config,
        mock_memory_manager: MagicMock,
        mock_tool_logger: MagicMock,
    ) -> None:
        snapshot = SearchFailureSnapshot(
            semantic=ComponentFailureSnapshot(1, "2026-10-03T00:00:00+00:00", "OSError")
        )
        mock_memory_manager.search_failure_snapshot = MagicMock(return_value=snapshot)
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": EngineReadiness.INITIALIZED,
            "tantivy_engine": EngineReadiness.INITIALIZED,
        }
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)

        result = await tool.get_handler()()

        assert result["search_failures"] == snapshot.to_dict()
        assert result["status"] == HealthStatus.HEALTHY
        assert set(result) == {
            "status",
            "workspace_id",
            "semantic_engine",
            "tantivy_engine",
            "reranker_engine",
            "hybrid_search_enabled",
            "rrf_fusion_enabled",
            "recency_boost_enabled",
            "pending_intent_count",
            "search_failures",
        }
        mock_memory_manager.search_failure_snapshot.assert_called_once_with()
        mock_memory_manager.reconcile_pending_replacements.assert_not_called()

    async def test_reports_pending_when_engines_are_not_ready(
        self,
        mock_config: Config,
        mock_memory_manager: MagicMock,
        mock_tool_logger: MagicMock,
    ) -> None:
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": EngineReadiness.PENDING,
            "tantivy_engine": EngineReadiness.DISABLED,
        }
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] in {HealthStatus.UNHEALTHY, HealthStatus.DEGRADED}
        assert result["semantic_engine"] == EngineReadiness.PENDING
        assert result["tantivy_engine"] == EngineReadiness.DISABLED
        mock_memory_manager.search_engine_status.assert_called_once()
        mock_memory_manager.reconcile_pending_replacements.assert_not_called()

    async def test_degraded_when_pending_replacements_remain(
        self,
        mock_config: Config,
        mock_memory_manager: MagicMock,
        mock_tool_logger: MagicMock,
    ) -> None:
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": EngineReadiness.INITIALIZED,
            "tantivy_engine": EngineReadiness.INITIALIZED,
        }
        mock_memory_manager.pending_intent_count.return_value = 2
        mock_memory_manager.pending_replacement_count.return_value = 2
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] == HealthStatus.DEGRADED
        assert result["pending_intent_count"] == 2
        mock_memory_manager.reconcile_pending_replacements.assert_not_called()
        mock_memory_manager.pending_intent_count.assert_called_once()

    async def test_healthy_when_no_pending_replacements(
        self,
        mock_config: Config,
        mock_memory_manager: MagicMock,
        mock_tool_logger: MagicMock,
    ) -> None:
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": EngineReadiness.INITIALIZED,
            "tantivy_engine": EngineReadiness.INITIALIZED,
        }
        mock_memory_manager.pending_intent_count.return_value = 0
        mock_memory_manager.pending_replacement_count.return_value = 0
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] == HealthStatus.HEALTHY
        assert result["pending_intent_count"] == 0
        mock_memory_manager.reconcile_pending_replacements.assert_not_called()

    async def test_unhealthy_does_not_reenter_status(
        self,
        mock_config: Config,
        mock_memory_manager: MagicMock,
        mock_tool_logger: MagicMock,
    ) -> None:
        mock_memory_manager.search_engine_status.side_effect = RuntimeError(
            "status boom"
        )
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] == HealthStatus.UNHEALTHY
        assert result["error_type"] == "RuntimeError"
        assert result["search_failures"] == SearchFailureSnapshot().to_dict()
        assert result["diagnostics"]["semantic_engine"] == EngineReadiness.UNKNOWN
        assert result["diagnostics"]["tantivy_engine"] == EngineReadiness.UNKNOWN
        assert mock_memory_manager.search_engine_status.call_count == 1

    async def test_unhealthy_when_journal_cannot_be_listed(
        self,
        mock_config: Config,
        mock_memory_manager: MagicMock,
        mock_tool_logger: MagicMock,
    ) -> None:
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": EngineReadiness.INITIALIZED,
            "tantivy_engine": EngineReadiness.INITIALIZED,
        }
        mock_memory_manager.pending_intent_count.side_effect = RuntimeError(
            "journal locked"
        )
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] == HealthStatus.UNHEALTHY
        assert result["error_type"] == "RuntimeError"
