"""Unit tests for HealthCheckTool."""

import pytest

from reflectlog.application.tools.health_check import HealthCheckTool


@pytest.mark.unit
class TestHealthCheckToolStatus:
    """Health check must use the public readiness API."""

    async def test_reports_pending_when_engines_are_not_ready(
        self, mock_config, mock_memory_manager, mock_tool_logger
    ) -> None:
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": "pending",
            "tantivy_engine": "disabled",
        }
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] == "healthy"
        assert result["semantic_engine"] == "pending"
        assert result["tantivy_engine"] == "disabled"
        mock_memory_manager.search_engine_status.assert_called_once()

    async def test_degraded_when_replacements_are_pending(
        self, mock_config, mock_memory_manager, mock_tool_logger
    ) -> None:
        mock_memory_manager.search_engine_status.return_value = {
            "semantic_engine": "initialized",
            "tantivy_engine": "initialized",
        }
        mock_memory_manager.pending_replacement_count.return_value = 2
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] == "degraded"
        assert result["pending_replacement_transitions"] == 2

    async def test_unhealthy_does_not_reenter_status(
        self, mock_config, mock_memory_manager, mock_tool_logger
    ) -> None:
        mock_memory_manager.search_engine_status.side_effect = RuntimeError("status boom")
        mock_memory_manager.startup_metrics = None
        tool = HealthCheckTool(mock_config, mock_memory_manager, mock_tool_logger)
        result = await tool.get_handler()()

        assert result["status"] == "unhealthy"
        assert result["error_type"] == "RuntimeError"
        assert result["diagnostics"]["semantic_engine"] == "unknown"
        assert result["diagnostics"]["tantivy_engine"] == "unknown"
        assert mock_memory_manager.search_engine_status.call_count == 1
