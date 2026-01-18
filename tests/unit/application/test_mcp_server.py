"""Tests for MCP server tools: add, get_all, search, remove."""
# mypy: disable-error-code="misc,var-annotated"
# Tests are async because tool handlers are async functions

from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.exceptions import (
    ConfigurationError,
    SearchError,
    StorageError,
)
from reflectlog.application.memory.manager import AddResult


@pytest.mark.unit
class TestFastMCPServerInitialization:
    """Test suite for FastMCPServer initialization."""

    def test_server_initialization_success(self, mcp_server):
        """Test successful server initialization."""
        assert mcp_server is not None
        assert mcp_server.mcp is not None
        assert mcp_server.memory_manager is not None
        # Memory manager should have a mock semantic engine
        assert mcp_server.memory_manager.memory is not None

    def test_server_initialization_missing_project_id(self):
        """Test Config.from_environment fails without PROJECT_ID.

        Note: FastMCPServer uses a module-level config singleton, so we test
        the Config class directly to verify PROJECT_ID validation.
        """
        import os

        from reflectlog.application.config import Config

        # Save original and clear PROJECT_ID
        original = os.environ.pop("PROJECT_ID", None)
        try:
            with pytest.raises(
                ConfigurationError, match="PROJECT_ID environment variable"
            ):
                Config.from_environment()
        finally:
            # Restore original
            if original is not None:
                os.environ["PROJECT_ID"] = original

    def test_memory_config_structure(self, set_env_vars):
        """Test USearchEngine is initialized with correct config."""
        from reflectlog.application.mcp_server import FastMCPServer

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_cls:
            mock_usearch_cls.return_value = MagicMock()

            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_cls:
                    mock_tantivy_cls.return_value = MagicMock()

                    FastMCPServer()

                    # Verify USearchEngine was called
                    mock_usearch_cls.assert_called_once()
                    call_args = mock_usearch_cls.call_args

                    # First positional arg should be USearchConfig
                    usearch_config = call_args[0][0]

                    # Verify config attributes (USearchConfig uses project_id)
                    assert usearch_config.project_id == "test_project"


@pytest.mark.unit
class TestAddTool:
    """Test suite for add() tool."""

    async def test_add_single_message_success(self, mcp_server, sample_messages):
        """Test adding a single valid message."""
        messages = sample_messages["single"]

        # Get the add tool function from registered tools
        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None, "add tool not found"

        # Execute add (async)
        result = await add_func(messages)

        # Verify behavior
        assert result is None  # add returns None
        mcp_server.memory_manager.memory.add.assert_called_once()
        # USearchEngine.add is called with keyword arguments
        call_kwargs = mcp_server.memory_manager.memory.add.call_args.kwargs
        assert call_kwargs["message"] == messages[0]
        assert call_kwargs["project_id"] == mcp_server.config.project_id

    async def test_add_multiple_messages_success(self, mcp_server, sample_messages):
        """Test adding multiple valid messages."""
        messages = sample_messages["multiple"]

        # Get the add tool function
        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        # Execute add (async)
        assert add_func is not None
        await add_func(messages)

        # Verify memory.add was called with correct messages
        assert mcp_server.memory_manager.memory.add.call_count == len(messages)
        # USearchEngine.add uses keyword arguments
        added_messages = [
            call.kwargs["message"]
            for call in mcp_server.memory_manager.memory.add.call_args_list
        ]
        assert added_messages == messages

    async def test_add_empty_list_noop(self, mcp_server):
        """Test adding empty list is no-op (no error, no call to memory)."""
        messages = []

        # Get the add tool function
        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        # Execute add (async)
        assert add_func is not None
        result = await add_func(messages)

        # Verify no-op: returns None and doesn't call memory.add
        assert result is None
        mcp_server.memory_manager.memory.add.assert_not_called()

    async def test_add_message_at_min_length(self, mcp_server, sample_messages):
        """Test adding message at minimum length (1 character)."""
        messages = [sample_messages["edge_cases"]["min_length"]]

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(messages)
        mcp_server.memory_manager.memory.add.assert_called_once()

    async def test_add_message_at_max_length(self, mcp_server, sample_messages):
        """Test adding message at maximum length (30720 characters)."""
        messages = [sample_messages["edge_cases"]["max_length"]]

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(messages)
        mcp_server.memory_manager.memory.add.assert_called_once()

    async def test_add_messages_with_special_characters(
        self, mcp_server, sample_messages
    ):
        """Test adding messages with special characters."""
        messages = sample_messages["with_special_chars"]

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(messages)
        assert mcp_server.memory_manager.memory.add.call_count == len(messages)
        added_messages = [
            call.kwargs["message"]
            for call in mcp_server.memory_manager.memory.add.call_args_list
        ]
        assert added_messages == messages

    async def test_add_non_string_message_raises_value_error(self, mcp_server):
        """Test adding non-string message raises ValueError."""
        messages = [123]  # Non-string

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="not a string"):
            await add_func(messages)

        mcp_server.memory_manager.memory.add.assert_not_called()

    async def test_add_empty_string_raises_value_error(self, mcp_server):
        """Test adding empty string raises ValueError."""
        messages = [""]

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="too short"):
            await add_func(messages)

        mcp_server.memory_manager.memory.add.assert_not_called()

    async def test_add_whitespace_only_raises_value_error(self, mcp_server):
        """Test adding whitespace-only message raises ValueError."""
        messages = ["   "]

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="only whitespace"):
            await add_func(messages)

        mcp_server.memory_manager.memory.add.assert_not_called()

    async def test_add_message_too_long_raises_value_error(
        self, mcp_server, sample_messages
    ):
        """Test adding message exceeding max length raises ValueError."""
        messages = [sample_messages["invalid"]["too_long"]]

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="too long"):
            await add_func(messages)

        mcp_server.memory_manager.memory.add.assert_not_called()

    async def test_add_mixed_valid_invalid_raises_value_error(self, mcp_server):
        """Test adding mix of valid and invalid messages raises ValueError."""
        messages = ["Valid message", ""]  # Second is invalid

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError):
            await add_func(messages)

        mcp_server.memory_manager.memory.add.assert_not_called()

    async def test_add_memory_storage_failure_raises_storage_error(self, mcp_server):
        """Test memory storage failure raises StorageError."""
        messages = ["Valid message"]

        # Configure mock to raise exception
        mcp_server.memory_manager.memory.add.side_effect = Exception("Storage failed")

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(StorageError, match="Failed to add messages"):
            await add_func(messages)

    @pytest.mark.parametrize(
        "messages",
        [
            ["Single message"],
            ["First", "Second"],
            ["Message with\nnewlines"],
            ["Unicode: 你好"],
        ],
    )
    async def test_add_various_valid_messages(self, mcp_server, messages):
        """Test adding various types of valid messages."""
        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(messages)
        assert mcp_server.memory_manager.memory.add.call_count == len(messages)
        added_messages = [
            call.kwargs["message"]
            for call in mcp_server.memory_manager.memory.add.call_args_list
        ]
        assert added_messages == messages

    async def test_add_skips_duplicates(self, mcp_server):
        """Test that duplicate messages are not stored twice."""
        message = "Duplicate message"
        # Mock add_messages_async to return AddResult with skipped (deduplication happened)
        from unittest.mock import AsyncMock

        mcp_server.memory_manager.add_messages_async = AsyncMock(
            return_value=AddResult(
                stored_count=0, skipped_count=1, replaced_count=0, replacements=[]
            )
        )

        add_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func([message])

        # MemoryManager.add_messages_async was called but returned 0 (duplicate skipped)
        mcp_server.memory_manager.add_messages_async.assert_called_once()


@pytest.mark.unit
class TestGetAllTool:
    """Test suite for get_all() tool."""

    async def test_get_all_empty_store(self, mcp_server):
        """Test get_all returns empty list when no messages stored."""
        mcp_server.memory_manager.memory.get_all.return_value = []

        # Get the get_all tool function
        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None, "get_all tool not found"

        result = await get_all_func()

        assert result == []
        mcp_server.memory_manager.memory.get_all.assert_called_once()

    async def test_get_all_single_message(self, mcp_server, sample_messages):
        """Test get_all returns single message."""
        messages = sample_messages["single"]
        mcp_server.memory_manager.memory.get_all.return_value = messages

        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert result == messages
        mcp_server.memory_manager.memory.get_all.assert_called_once()

    async def test_get_all_multiple_messages(self, mcp_server, sample_messages):
        """Test get_all returns multiple messages."""
        messages = sample_messages["multiple"]
        mcp_server.memory_manager.memory.get_all.return_value = messages

        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert result == messages
        assert len(result) == 3
        mcp_server.memory_manager.memory.get_all.assert_called_once()

    async def test_get_all_returns_copy(self, mcp_server, sample_messages):
        """Test get_all returns a copy of messages (prevents mutation)."""
        messages = sample_messages["multiple"].copy()
        mcp_server.memory_manager.memory.get_all.return_value = messages

        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        # Modify result
        if result:
            result.append("New message")

        # Original should be unchanged if copy works correctly
        # Note: In practice, we can't test true immutability here since
        # we're mocking, but we verify the .copy() call would be made
        assert "New message" not in messages

    async def test_get_all_with_special_characters(self, mcp_server, sample_messages):
        """Test get_all handles messages with special characters."""
        messages = sample_messages["with_special_chars"]
        mcp_server.memory_manager.memory.get_all.return_value = messages

        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert result == messages
        assert len(result) == 3

    async def test_get_all_large_dataset(self, mcp_server):
        """Test get_all with large number of messages."""
        messages = [f"Message {i}" for i in range(1000)]
        mcp_server.memory_manager.memory.get_all.return_value = messages

        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert len(result) == 1000
        assert result == messages

    async def test_get_all_memory_failure_raises_storage_error(self, mcp_server):
        """Test memory retrieval failure raises StorageError."""
        mcp_server.memory_manager.memory.get_all.side_effect = Exception(
            "Retrieval failed"
        )

        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        with pytest.raises(StorageError, match="Failed to retrieve messages"):
            await get_all_func()

    async def test_get_all_called_multiple_times(self, mcp_server, sample_messages):
        """Test get_all can be called multiple times."""
        messages = sample_messages["multiple"]
        mcp_server.memory_manager.memory.get_all.return_value = messages

        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result1 = await get_all_func()
        result2 = await get_all_func()

        assert result1 == messages
        assert result2 == messages
        assert mcp_server.memory_manager.memory.get_all.call_count == 2


@pytest.mark.unit
class TestSearchTool:
    """Test suite for search() tool."""

    async def test_search_exact_match(self, mcp_server):
        """Test search returns all semantic matches."""
        from unittest.mock import AsyncMock

        query = "Hello"
        expected_results = ["Hello, World!", "Hello"]

        # Mock MemoryManager.search() directly (returns List[str]) - now async
        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        # Get the search tool function
        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None, "search tool not found"

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results without filtering
        assert len(result) == 2
        assert "Hello, World!" in result
        assert "Hello" in result
        mcp_server.memory_manager.search.assert_called_once()

    async def test_search_semantic_results(self, mcp_server):
        """Test search returns all semantic results without substring filtering."""
        from unittest.mock import AsyncMock

        query = "World"
        expected_results = ["Hello, World!", "Goodbye", "World"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results, no substring filtering
        assert len(result) == 3
        assert "Hello, World!" in result
        assert "World" in result
        assert (
            "Goodbye" in result
        )  # Now included even though it doesn't contain "World"

    async def test_search_returns_all_semantic_matches(self, mcp_server):
        """Test search returns all semantic matches without filtering."""
        from unittest.mock import AsyncMock

        query = "programming"
        expected_results = ["Python tutorial", "JavaScript guide", "Database design"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results regardless of substring content
        assert len(result) == 3
        assert "Python tutorial" in result
        assert "JavaScript guide" in result
        assert "Database design" in result

    async def test_search_with_semantic_matches(self, mcp_server):
        """Test search returns all semantic matches even if no substring matches."""
        from unittest.mock import AsyncMock

        query = "Programming"
        expected_results = ["Hello", "World", "Test"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results even though none contain "Programming" substring
        assert len(result) == 3
        assert "Hello" in result
        assert "World" in result
        assert "Test" in result

    async def test_search_with_empty_semantic_results(self, mcp_server):
        """Test search returns empty list when semantic search returns no results."""
        from unittest.mock import AsyncMock

        query = "Python"
        expected_results = []

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Empty semantic results means empty final results
        assert result == []

    async def test_search_with_special_characters(self, mcp_server):
        """Test search with special characters."""
        from unittest.mock import AsyncMock

        query = "@user"
        expected_results = ["Message to @user", "Another message", "@user replied"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results, no substring filtering
        assert len(result) == 3
        assert "Message to @user" in result
        assert "@user replied" in result
        assert "Another message" in result

    async def test_search_with_unicode(self, mcp_server):
        """Test search with unicode characters."""
        from unittest.mock import AsyncMock

        query = "你好"
        expected_results = ["你好世界", "Hello World", "你好 friend"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results, no substring filtering
        assert len(result) == 3
        assert "你好世界" in result
        assert "你好 friend" in result
        assert "Hello World" in result  # Included even without Chinese characters

    async def test_search_single_character(self, mcp_server):
        """Test search with single character query."""
        from unittest.mock import AsyncMock

        query = "a"
        expected_results = ["Apple", "Banana", "Cherry"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results, no substring filtering
        assert len(result) == 3
        assert "Apple" in result
        assert "Banana" in result
        assert "Cherry" in result  # Included even without 'a'

    async def test_search_empty_result_from_memory(self, mcp_server):
        """Test search when memory returns no semantic results."""
        from unittest.mock import AsyncMock

        query = "test"
        mcp_server.memory_manager.search = AsyncMock(return_value=[])

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        assert result == []

    async def test_search_memory_failure_raises_search_error(self, mcp_server):
        """Test memory search failure raises SearchError."""
        from unittest.mock import AsyncMock

        query = "test"
        mcp_server.memory_manager.search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        with pytest.raises(SearchError, match="Failed to search memory store"):
            await search_func(query)

    @pytest.mark.parametrize(
        "query,messages,expected_count",
        [
            ("test", ["This is a test", "testing", "no match"], 3),
            ("Python", ["Python tutorial", "Java guide", "Python examples"], 3),
            ("🌍", ["Hello 🌍", "World 🌎", "🌍 emoji"], 3),
        ],
    )
    async def test_search_various_queries(
        self, mcp_server, query, messages, expected_count
    ):
        """Test search returns all semantic results without substring filtering."""
        from unittest.mock import AsyncMock

        # Mock MemoryManager.search() to return all messages (semantic search behavior)
        mcp_server.memory_manager.search = AsyncMock(return_value=messages)

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        assert len(result) == expected_count


@pytest.mark.unit
class TestRemoveTool:
    """Test suite for remove() tool."""

    async def test_remove_single_message(self, mcp_server):
        """Test removing a single message."""
        message_to_remove = "Message to remove"
        mock_candidates = [{"id": "1", "memory": message_to_remove, "score": 0.95}]

        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        # Get the remove tool function
        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None, "remove tool not found"

        assert remove_func is not None
        result = await remove_func([message_to_remove])

        assert result is None  # remove returns None
        mcp_server.memory_manager.search_for_removal.assert_called_once_with(
            message_to_remove
        )
        mcp_server.memory_manager.delete_by_message.assert_called_once_with(
            message_to_remove
        )

    async def test_remove_multiple_messages(self, mcp_server):
        """Test removing multiple messages."""
        messages_to_remove = ["Message 1", "Message 2"]

        def search_side_effect(query):
            if query == "Message 1":
                return [{"id": "1", "memory": "Message 1", "score": 0.95}]
            elif query == "Message 2":
                return [{"id": "2", "memory": "Message 2", "score": 0.95}]
            return []

        mcp_server.memory_manager.search_for_removal = MagicMock(
            side_effect=search_side_effect
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        await remove_func(messages_to_remove)

        assert mcp_server.memory_manager.search_for_removal.call_count == 2
        assert mcp_server.memory_manager.delete_by_message.call_count == 2

    async def test_remove_empty_list_noop(self, mcp_server):
        """Test removing empty list is no-op (no error, no calls)."""
        messages = []

        # Mock the methods to verify they're not called
        mcp_server.memory_manager.search_for_removal = MagicMock()
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        result = await remove_func(messages)

        assert result is None
        mcp_server.memory_manager.search_for_removal.assert_not_called()
        mcp_server.memory_manager.delete_by_message.assert_not_called()

    async def test_remove_non_existent_message_silently_ignored(self, mcp_server):
        """Test removing non-existent message is silently ignored."""
        message = "Non-existent message"

        # Semantic search returns results, but none match exactly
        mock_candidates = [
            {"id": "1", "memory": "Similar message", "score": 0.85},
            {"id": "2", "memory": "Another message", "score": 0.80},
        ]
        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        # Should not raise error
        assert remove_func is not None
        result = await remove_func([message])

        assert result is None
        mcp_server.memory_manager.search_for_removal.assert_called_once_with(message)
        mcp_server.memory_manager.delete_by_message.assert_not_called()  # No exact match, no delete

    async def test_remove_duplicate_messages_all_removed(self, mcp_server):
        """Test removing message that appears multiple times removes all."""
        message = "Duplicate message"

        # Return multiple results with same message and different IDs
        mock_candidates = [
            {"id": "1", "memory": message, "score": 0.95},
            {"id": "2", "memory": message, "score": 0.95},
            {"id": "3", "memory": message, "score": 0.95},
        ]

        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        await remove_func([message])

        # Should delete all 3 occurrences
        assert mcp_server.memory_manager.delete_by_message.call_count == 3

    async def test_remove_case_sensitive_exact_match(self, mcp_server):
        """Test remove uses case-sensitive exact matching."""
        message = "Hello"

        # Semantic search returns multiple variations, but only exact match should be deleted
        mock_candidates = [
            {"id": "1", "memory": "Hello", "score": 0.95},
            {"id": "2", "memory": "hello", "score": 0.90},
            {"id": "3", "memory": "HELLO", "score": 0.90},
        ]
        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        await remove_func([message])

        # Only exact match "Hello" should be deleted (1 call)
        assert mcp_server.memory_manager.delete_by_message.call_count == 1
        mcp_server.memory_manager.delete_by_message.assert_called_once_with("Hello")

    async def test_remove_mixed_existent_and_non_existent(self, mcp_server):
        """Test removing mix of existent and non-existent messages."""

        def search_side_effect(query):
            if query == "Exists":
                return [{"id": "1", "memory": "Exists", "score": 0.95}]
            else:
                return []  # Non-existent returns empty

        mcp_server.memory_manager.search_for_removal = MagicMock(
            side_effect=search_side_effect
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        messages = ["Exists", "Does not exist"]
        assert remove_func is not None
        await remove_func(messages)

        assert mcp_server.memory_manager.search_for_removal.call_count == 2
        assert (
            mcp_server.memory_manager.delete_by_message.call_count == 1
        )  # Only one existed

    async def test_remove_memory_search_failure_raises_storage_error(self, mcp_server):
        """Test memory search failure during remove raises StorageError."""
        message = "Message"
        mcp_server.memory_manager.search_for_removal = MagicMock(
            side_effect=Exception("Search failed")
        )

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        with pytest.raises(StorageError, match="Failed to remove messages"):
            await remove_func([message])

    async def test_remove_memory_delete_failure_raises_storage_error(self, mcp_server):
        """Test memory delete failure raises StorageError."""
        message = "Message"
        mock_candidates = [{"id": "1", "memory": message, "score": 0.95}]

        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock(
            side_effect=Exception("Delete failed")
        )

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        with pytest.raises(StorageError, match="Failed to remove messages"):
            await remove_func([message])

    async def test_remove_with_special_characters(self, mcp_server):
        """Test removing messages with special characters."""
        message = "Message with special chars: !@#$%"
        mock_candidates = [{"id": "1", "memory": message, "score": 0.95}]

        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        await remove_func([message])

        mcp_server.memory_manager.delete_by_message.assert_called_once()

    async def test_remove_with_unicode(self, mcp_server):
        """Test removing messages with unicode characters."""
        message = "Unicode message: 你好世界 🌍"
        mock_candidates = [{"id": "1", "memory": message, "score": 0.95}]

        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        await remove_func([message])

        mcp_server.memory_manager.delete_by_message.assert_called_once()

    @pytest.mark.parametrize(
        "message,should_delete",
        [
            ("Exact match", True),
            ("EXACT MATCH", False),  # Case-sensitive
            ("exact match", False),  # Case-sensitive
        ],
    )
    async def test_remove_case_sensitivity(self, mcp_server, message, should_delete):
        """Test remove is case-sensitive for matching."""
        # Semantic search always returns "Exact match"
        mock_candidates = [{"id": "1", "memory": "Exact match", "score": 0.95}]
        mcp_server.memory_manager.search_for_removal = MagicMock(
            return_value=mock_candidates
        )
        mcp_server.memory_manager.delete_by_message = MagicMock()

        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "remove":
                remove_func = tool.fn
                break

        assert remove_func is not None
        await remove_func([message])

        if should_delete:
            mcp_server.memory_manager.delete_by_message.assert_called_once()
        else:
            mcp_server.memory_manager.delete_by_message.assert_not_called()


@pytest.mark.unit
class TestToolRegistrationConfiguration:
    """Test suite for configurable tool registration."""

    def _build_server(self, monkeypatch, allowed_value: str | None):
        """Helper to create a FastMCPServer with patched dependencies."""
        monkeypatch.setenv("PROJECT_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")

        if allowed_value is None:
            monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
        else:
            monkeypatch.setenv("ALLOWED_TOOLS", allowed_value)

        with (
            patch("reflectlog.application.mcp_server.FastMCP") as mock_fastmcp,
            patch(
                "reflectlog.application.mcp_server.MemoryManager"
            ) as mock_memory_manager,
            patch(
                "reflectlog.application.mcp_server.create_logger"
            ) as mock_create_logger,
        ):
            fastmcp_instance = MagicMock()
            mock_fastmcp.return_value = fastmcp_instance
            mock_memory_manager.return_value = MagicMock()
            logger = MagicMock()
            mock_create_logger.return_value = logger

            from reflectlog.application.config import Config
            from reflectlog.application.mcp_server import FastMCPServer

            server_config = Config.from_environment()
            server = FastMCPServer(server_config=server_config)

        return server, logger, fastmcp_instance

    def test_allowed_tools_subset(self, monkeypatch):
        """Only tools listed in ALLOWED_TOOLS should register."""
        server, logger, fastmcp_instance = self._build_server(
            monkeypatch, "add,get_all"
        )

        tool_names = [tool.get_name() for tool in server.tools]
        assert tool_names == ["add", "get_all"]
        assert fastmcp_instance.tool.call_count == 2
        logger.warning.assert_not_called()

    def test_allowed_tools_invalid_tokens(self, monkeypatch):
        """Unknown tool names are ignored with a warning."""
        server, logger, fastmcp_instance = self._build_server(
            monkeypatch, "add,unknown,remove_tool"
        )

        tool_names = [tool.get_name() for tool in server.tools]
        assert tool_names == ["add", "remove"]
        assert fastmcp_instance.tool.call_count == 2

        warnings = logger.warning.call_args_list
        assert warnings, "Expected warning for invalid tool tokens"
        first_warning = warnings[0]
        extra = first_warning.kwargs.get("extra", {})
        assert "invalid_tools" in extra
        assert "unknown" in extra["invalid_tools"]
        assert "Ignoring unknown tool identifiers" in first_warning.args[0]

    def test_allowed_tools_none_keyword(self, monkeypatch):
        """'none' disables all tools and surfaces a warning."""
        server, logger, fastmcp_instance = self._build_server(monkeypatch, "none")

        assert server.tools == []
        assert fastmcp_instance.tool.call_count == 0

        warnings = logger.warning.call_args_list
        assert warnings, "Expected warning when no tools are registered"
        first_warning = warnings[0]
        assert "No MCP tools selected" in first_warning.args[0]
        extra = first_warning.kwargs.get("extra", {})
        assert "available_tools" in extra


@pytest.mark.unit
class TestHealthCheckTool:
    """Test suite for health_check() tool."""

    async def test_health_check_returns_healthy_status(self, mcp_server):
        """Test health check returns healthy status with all components."""
        # Mock semantic and tantivy engines as initialized
        mcp_server.memory_manager._semantic_engine = MagicMock()
        mcp_server.memory_manager._tantivy_engine = MagicMock()

        health_check_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        assert result["status"] == "healthy"
        assert result["project_id"] == "test_project"
        assert result["semantic_engine"] == "initialized"
        assert result["tantivy_engine"] == "initialized"
        assert result["reranker_engine"] == "llm"
        assert result["hybrid_search_enabled"] is True
        assert result["rrf_fusion_enabled"] is True
        assert result["recency_boost_enabled"] is True

    async def test_health_check_with_tantivy_disabled(self, mcp_server):
        """Test health check when Tantivy is disabled."""
        # Mock semantic engine as initialized, tantivy as None
        mcp_server.memory_manager._semantic_engine = MagicMock()
        mcp_server.memory_manager._tantivy_engine = None

        health_check_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        assert result["status"] == "healthy"
        assert result["semantic_engine"] == "initialized"
        assert result["tantivy_engine"] == "disabled"

    async def test_health_check_no_semantic_engine(self, mcp_server):
        """Test health check when semantic engine is not initialized."""
        mcp_server.memory_manager._semantic_engine = None
        mcp_server.memory_manager._tantivy_engine = None

        health_check_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        assert result["semantic_engine"] == "not_initialized"
        assert result["tantivy_engine"] == "disabled"

    async def test_health_check_with_different_reranker(self, mcp_server):
        """Test health check reports configured reranker engine."""
        # The default reranker is "llm" - verify it's reported correctly
        mcp_server._memory_manager._semantic_engine = MagicMock()
        mcp_server._memory_manager._tantivy_engine = MagicMock()

        health_check_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        # Default reranker should be "llm"
        assert result["reranker_engine"] == "llm"
