"""Shared fixtures for MCP tool unit tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.add_phases import AddResult
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.tools.add import AddTool
from reflectlog.application.tools.base import BaseTool
from reflectlog.application.tools.remove import RemoveTool
from reflectlog.application.tools.search import SearchTool


@pytest.fixture
def mock_config(set_env_vars) -> Config:
    """Create a Config from test environment variables."""
    return Config.from_environment()


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock MemoryManager with common methods stubbed."""
    mm = MagicMock(spec=MemoryManager)
    mm.add_memories_async = AsyncMock(
        return_value=AddResult(
            stored_count=1, skipped_count=0, replaced_count=0, replacements=[]
        )
    )
    mm.search = AsyncMock(return_value=[])
    mm.search_for_removal = MagicMock(return_value=[])
    mm.delete_by_memory = MagicMock(return_value=True)
    mm.delete_by_message = MagicMock(return_value=True)
    return mm


@pytest.fixture
def mock_tool_logger() -> MagicMock:
    """Create a mock structured logger for tool tests."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def add_tool_instance(mock_config, mock_memory_manager, mock_tool_logger) -> AddTool:
    """Create an AddTool instance with mocked dependencies."""
    return AddTool(
        config=mock_config, memory_manager=mock_memory_manager, logger=mock_tool_logger
    )


@pytest.fixture
def search_tool_instance(
    mock_config, mock_memory_manager, mock_tool_logger
) -> SearchTool:
    """Create a SearchTool instance with mocked dependencies."""
    return SearchTool(
        config=mock_config, memory_manager=mock_memory_manager, logger=mock_tool_logger
    )


@pytest.fixture
def remove_tool_instance(
    mock_config, mock_memory_manager, mock_tool_logger
) -> RemoveTool:
    """Create a RemoveTool instance with mocked dependencies."""
    return RemoveTool(
        config=mock_config, memory_manager=mock_memory_manager, logger=mock_tool_logger
    )
