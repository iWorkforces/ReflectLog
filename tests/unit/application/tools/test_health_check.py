"""Unit tests for HealthCheckTool."""

from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.health_check import HealthCheckTool
from reflectlog.core.enums import EngineReadiness, HealthStatus


@pytest.mark.unit
class TestHealthCheckToolStatus:
    """Health check must use the public readiness API."""

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
