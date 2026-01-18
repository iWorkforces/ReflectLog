"""Shared fixtures and configuration for all tests."""

from unittest.mock import MagicMock, patch

import pytest


class MockMemorySearchResult:
    """Mock implementation of MemorySearchResult for testing."""

    def __init__(self, text: str, index: int | str):
        """Initialize mock search result.

        Args:
            text: The message text content
            index: Unique identifier for the result
        """
        self._text = text
        self.index = index

    def __str__(self) -> str:
        """Return string representation."""
        return self._text

    def lower(self) -> str:
        """Return lowercase version of the text."""
        return self._text.lower()


@pytest.fixture
def mock_search_result():
    """Factory fixture for creating mock search results.

    Returns:
        Callable that creates MockMemorySearchResult instances
    """

    def _create_result(text: str, index: int | str = 0) -> MockMemorySearchResult:
        return MockMemorySearchResult(text, index)

    return _create_result


@pytest.fixture
def mock_usearch_engine():
    """Mock USearchEngine instance for testing.

    Returns:
        MagicMock configured to behave like a USearchEngine instance
    """
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)
    # USearchEngine.get_all returns List[str] directly
    engine.get_all = MagicMock(return_value=[])
    # USearchEngine.search returns List[Tuple[str, float]]
    engine.search = MagicMock(return_value=[])
    engine.delete = MagicMock(return_value=None)
    engine.commit = MagicMock(return_value=None)
    engine.ensure_initialized = MagicMock(return_value=None)
    return engine


@pytest.fixture
def mock_memory_class(mock_usearch_engine):
    """Mock USearchEngine class for patching imports.

    Yields:
        Patched USearchEngine class that returns mock_usearch_engine instance
    """
    with patch("reflectlog.application.memory.manager.USearchEngine") as mock_cls:
        mock_cls.return_value = mock_usearch_engine
        yield mock_cls


@pytest.fixture
def set_env_vars(monkeypatch):
    """Set required environment variables for testing.

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        Dictionary of set environment variables
    """
    env_vars = {
        "PROJECT_ID": "test_project",
        "OPENROUTER_API_KEY": "test_key",
        "MCP_TRANSPORT": "stdio",
        "MCP_PORT": "9103",
        "MCP_HOST": "127.0.0.1",
        "MCP_PATH": "/mcp",
        "SEARCH_LIMIT": "5",
        "REMOVE_SEARCH_LIMIT": "5",
        "MAX_MESSAGE_LENGTH": "30720",
        "MIN_MESSAGE_LENGTH": "1",
        "LOG_LEVEL": "INFO",
        "DEDUPLICATE_MESSAGES": "true",
        "ADD_MAX_CONCURRENCY": "8",
        # Disable embedding cache in tests to avoid issues with mocked embedders
        "EMBEDDING_CACHE_ENABLED": "false",
        # Disable eager initialization in tests to avoid issues with mocked engines
        "EAGER_INITIALIZATION": "false",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def unset_project_id(monkeypatch):
    """Remove PROJECT_ID from environment for testing missing config.

    Args:
        monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.delenv("PROJECT_ID", raising=False)


@pytest.fixture
def mcp_server(set_env_vars, mock_usearch_engine):
    """Create FastMCPServer instance with mocked dependencies.

    Args:
        set_env_vars: Environment variables fixture
        mock_usearch_engine: Mocked USearchEngine instance

    Returns:
        FastMCPServer instance for testing
    """
    with (
        patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_cls,
        patch(
            "reflectlog.application.memory.manager.TantivyEngine"
        ) as mock_tantivy_cls,
        patch(
            "reflectlog.application.memory.manager.LangchainQwenEmbeddings"
        ) as mock_embedder_cls,
        patch(
            "reflectlog.application.memory.manager.CachedEmbeddings"
        ) as mock_cached_embedder_cls,
    ):
        # Configure USearchEngine mock
        mock_usearch_cls.return_value = mock_usearch_engine

        # Configure TantivyEngine mock (basic mock, not used in most tests)
        mock_tantivy = MagicMock()
        mock_tantivy.ensure_initialized = MagicMock(return_value=None)
        mock_tantivy_cls.return_value = mock_tantivy

        # Configure embedder mock
        mock_embedder = MagicMock()
        mock_embedder_cls.return_value = mock_embedder

        # Configure CachedEmbeddings mock to return the base embedder mock
        # This prevents Pydantic validation errors in tests
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = mock_embedder
        mock_cached_embedder_cls.return_value = mock_cached_embedder

        from reflectlog.application.mcp_server import FastMCPServer

        server = FastMCPServer()
        # Add backwards-compatible 'memory_manager' attribute for tests (accessing private attribute)
        server.memory_manager = server._memory_manager  # type: ignore[attr-defined]
        # Add backwards-compatible 'memory' attribute for tests
        server.memory_manager.memory = mock_usearch_engine  # type: ignore[attr-defined]
        return server


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing.

    Returns:
        Dictionary of sample message lists for various test scenarios
    """
    return {
        "single": ["Hello, World!"],
        "multiple": [
            "First message",
            "Second message",
            "Third message",
        ],
        "with_special_chars": [
            "Message with special chars: !@#$%^&*()",
            "Unicode message: 你好世界 🌍",
            "Newlines\nand\ttabs",
        ],
        "duplicates": [
            "Duplicate message",
            "Unique message",
            "Duplicate message",
        ],
        "edge_cases": {
            "min_length": "a",  # 1 character
            "max_length": "x" * 30720,  # 30720 characters
            "whitespace": "   spaces   ",
        },
        "invalid": {
            "empty": "",
            "whitespace_only": "   ",
            "too_long": "x" * 30721,  # 30721 characters
        },
    }


@pytest.fixture
def mock_logger():
    """Mock logger for testing logging behavior.

    Returns:
        MagicMock configured as a logger
    """
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    return logger


@pytest.fixture(autouse=True)
def reset_env_after_test():
    """Automatically reset environment after each test.

    This fixture runs after every test to ensure clean state.
    """
    yield
    # Cleanup happens here if needed
    pass


@pytest.fixture
def create_search_results(mock_search_result):
    """Factory for creating lists of search results in USearchEngine format.

    Args:
        mock_search_result: Factory fixture for single results (unused, kept for compatibility)

    Returns:
        Callable that creates lists of tuples (message, score) for USearchEngine.search()
    """

    def _create_results(messages: list[str]) -> list[tuple[str, float]]:
        # USearchEngine.search returns List[Tuple[str, float]]
        # Default scores decrease from 0.9 for ranking
        return [(msg, 0.9 - (idx * 0.1)) for idx, msg in enumerate(messages)]

    return _create_results


@pytest.fixture
def create_search_response():
    """Factory for creating search response dictionaries.

    Returns:
        Callable that creates search response format: {'results': [{'memory': '...', 'id': '...'}]}
    """

    def _create_response(messages: list[str], include_ids: bool = True) -> dict:
        results = []
        for idx, msg in enumerate(messages):
            result = {"memory": msg}
            if include_ids:
                result["id"] = f"id_{idx}"
            results.append(result)
        return {"results": results}

    return _create_response


@pytest.fixture
def get_tool_func(mcp_server):
    """Helper to get tool function by name.

    Args:
        mcp_server: FastMCPServer instance

    Returns:
        Callable that retrieves tool functions by name
    """

    def _get_tool(tool_name: str):
        """Get tool function by name."""
        # Access the internal _tools dictionary directly to avoid async call
        if hasattr(mcp_server.mcp._tool_manager, "_tools"):
            tool = mcp_server.mcp._tool_manager._tools.get(tool_name)
            if tool is None:
                raise ValueError(f"Tool '{tool_name}' not found")
            return tool.fn
        raise RuntimeError("Tool manager not properly initialized")

    return _get_tool


@pytest.fixture
def add_tool(get_tool_func):
    """Get the add tool function."""
    return get_tool_func("add")


@pytest.fixture
def get_all_tool(get_tool_func):
    """Get the get_all tool function."""
    return get_tool_func("get_all")


@pytest.fixture
def search_tool(get_tool_func):
    """Get the search tool function."""
    return get_tool_func("search")


@pytest.fixture
def remove_tool(get_tool_func):
    """Get the remove tool function."""
    return get_tool_func("remove")
