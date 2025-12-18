"""Unit tests for error handling in mcp_server.py."""
# mypy: disable-error-code="misc,var-annotated,method-assign"

import os
import pytest
from unittest.mock import MagicMock, patch

from openmemories.application.exceptions import (
    ConfigurationError,
    SearchError,
    StorageError,
)


@pytest.fixture
def mock_usearch_engine():
    """Create a mock USearchEngine instance."""
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)
    # USearchEngine.get_all returns List[str] directly
    engine.get_all = MagicMock(return_value=[])
    # USearchEngine.search returns List[Tuple[str, float]]
    engine.search = MagicMock(return_value=[])
    engine.delete = MagicMock(return_value=None)
    engine.commit = MagicMock(return_value=None)
    return engine


@pytest.mark.unit
class TestServerInitializationErrors:
    """Test error handling in server initialization."""

    @patch("openmemories.application.mcp_server.PROJECT_ID", None)
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_init_without_project_id_raises_error(
        self, mock_usearch_engine_class, mock_embeddings
    ):
        """Test server initialization fails without PROJECT_ID."""
        from openmemories.application.mcp_server import FastMCPServer

        with pytest.raises(ConfigurationError) as exc_info:
            FastMCPServer()

        assert "PROJECT_ID" in str(exc_info.value)

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_init_with_memory_config_error(
        self, mock_usearch_engine_class, mock_embeddings
    ):
        """Test server initialization handles Memory.from_config errors."""
        # Make USearchEngine raise an exception
        mock_usearch_engine_class.side_effect = Exception(
            "Memory initialization failed"
        )

        from openmemories.application.mcp_server import FastMCPServer

        with pytest.raises(Exception) as exc_info:
            FastMCPServer()

        assert "Memory initialization failed" in str(exc_info.value)


@pytest.mark.unit
class TestAddToolErrorHandling:
    """Test error handling in add tool."""

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_add_memory_storage_failure(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test add tool handles memory storage failures."""
        # Configure mock to raise exception on add
        mock_usearch_engine.add.side_effect = Exception("Storage failure")
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        add_tool = mcp_server.mcp._tool_manager._tools["add"].fn

        with pytest.raises(StorageError) as exc_info:
            add_tool(["Test message"])

        assert "Failed to add messages" in str(exc_info.value)
        assert "Storage failure" in str(exc_info.value)

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_add_validation_error_with_non_string(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test add tool validation error with non-string message."""
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        add_tool = mcp_server.mcp._tool_manager._tools["add"].fn

        with pytest.raises(ValueError) as exc_info:
            add_tool([123, "Valid message"])

        assert "not a string" in str(exc_info.value)

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_add_validation_error_with_empty_string(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test add tool validation error with empty string."""
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        add_tool = mcp_server.mcp._tool_manager._tools["add"].fn

        with pytest.raises(ValueError) as exc_info:
            add_tool([""])

        assert "too short" in str(exc_info.value).lower()


@pytest.mark.unit
class TestGetAllToolErrorHandling:
    """Test error handling in get_all tool."""

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_get_all_memory_retrieval_failure(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test get_all tool handles retrieval failures."""
        # Configure mock to raise exception on get_all
        mock_usearch_engine.get_all.side_effect = Exception("Retrieval failure")
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        get_all_tool = mcp_server.mcp._tool_manager._tools["get_all"].fn

        with pytest.raises(StorageError) as exc_info:
            get_all_tool()

        assert "Failed to retrieve messages" in str(exc_info.value)
        assert "Retrieval failure" in str(exc_info.value)


@pytest.mark.unit
class TestSearchToolErrorHandling:
    """Test error handling in search tool."""

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_search_memory_failure(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test search tool handles search failures."""
        # Configure mock to raise exception on search
        mock_usearch_engine.search.side_effect = Exception("Search failure")
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        search_tool = mcp_server.mcp._tool_manager._tools["search"].fn

        with pytest.raises(SearchError) as exc_info:
            search_tool("test query")

        assert "Failed to search" in str(exc_info.value)
        assert "Search failure" in str(exc_info.value)


@pytest.mark.unit
class TestRemoveToolErrorHandling:
    """Test error handling in remove tool."""

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_remove_with_empty_list(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test remove tool with empty list (no-op)."""
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        remove_tool = mcp_server.mcp._tool_manager._tools["remove"].fn

        # Should not raise any errors
        remove_tool([])

        # Verify delete was never called
        mock_usearch_engine.delete.assert_not_called()

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_remove_memory_delete_failure(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test remove tool handles delete failures."""
        # Configure mock to return a search result (USearchEngine returns List[Tuple[str, float]])
        mock_usearch_engine.search.return_value = [("Test message", 0.9)]
        # Configure mock to raise exception on delete
        mock_usearch_engine.delete.side_effect = Exception("Delete failure")
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        remove_tool = mcp_server.mcp._tool_manager._tools["remove"].fn

        with pytest.raises(StorageError) as exc_info:
            remove_tool(["Test message"])

        assert "Failed to remove messages" in str(exc_info.value)
        assert "Delete failure" in str(exc_info.value)

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_remove_message_not_found(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test remove tool when message is not found."""
        # Configure mock to return no results (USearchEngine returns List[Tuple[str, float]])
        mock_usearch_engine.search.return_value = []
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        remove_tool = mcp_server.mcp._tool_manager._tools["remove"].fn

        # Should not raise any errors, just silently skip
        remove_tool(["Nonexistent message"])

        # Verify delete was never called
        mock_usearch_engine.delete.assert_not_called()


@pytest.mark.unit
class TestRunMethodCoverage:
    """Test coverage for run() method."""

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch.dict(os.environ, {"MCP_TRANSPORT": "http"}, clear=False)
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_run_method_with_http_transport(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test run() method with HTTP transport."""
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()

        # Mock the mcp.run() method to avoid actually starting the server
        mock_run = MagicMock()
        object.__setattr__(mcp_server.mcp, "run", mock_run)

        mcp_server.run()

        # Verify run was called
        mock_run.assert_called_once()

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}, clear=False)
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_run_method_with_stdio_transport(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test run() method with stdio transport."""
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        mock_run = MagicMock()
        object.__setattr__(mcp_server.mcp, "run", mock_run)

        mcp_server.run()

        mock_run.assert_called_once()

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}, clear=False)
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_run_method_with_sse_transport(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test run() method with SSE transport."""
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        mock_run = MagicMock()
        object.__setattr__(mcp_server.mcp, "run", mock_run)

        mcp_server.run()

        mock_run.assert_called_once()

    @patch("openmemories.application.mcp_server.PROJECT_ID", "test_project")
    @patch.dict(
        os.environ,
        {
            "MCP_TRANSPORT": "streamable-http",
            "MCP_PORT": "8080",
            "MCP_HOST": "localhost",
            "MCP_PATH": "/custom",
        },
        clear=False,
    )
    @patch("openmemories.application.memory.manager.LangchainQwenEmbeddings")
    @patch("openmemories.application.memory.manager.USearchEngine")
    def test_run_method_with_custom_config(
        self, mock_usearch_engine_class, mock_embeddings, mock_usearch_engine
    ):
        """Test run() method with custom configuration."""
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from openmemories.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        mock_run = MagicMock()
        object.__setattr__(mcp_server.mcp, "run", mock_run)

        mcp_server.run()

        mock_run.assert_called_once()
