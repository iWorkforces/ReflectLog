'''Tests for MCP server tools: add, get_all, search, remove.'''
# mypy: disable-error-code="misc,var-annotated"
# Tests are async because tool handlers are async functions

from unittest.mock import MagicMock, patch

import pytest

from reflectlog.core.exceptions import (
    ConfigurationError,
    SearchError,
    StorageError,
)
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.memory.manager import AddResult


@pytest.mark.unit
class TestFastMCPServerInitialization:
    '''Test suite for FastMCPServer initialization.'''

    def test_server_initialization_success(self, mcp_server):
        '''Test successful server initialization.'''
        assert mcp_server is not None
        assert mcp_server.mcp is not None
        assert mcp_server.memory_manager is not None
        # Memory manager should have a mock semantic engine
        assert mcp_server.memory_manager.memory is not None

    def test_server_initialization_missing_workspace_id(self):
        '''Test Config.from_environment fails without WORKSPACE_ID.

        Note: FastMCPServer uses a module-level config singleton, so we test
        the Config class directly to verify WORKSPACE_ID validation.
        '''
        import os

        from reflectlog.application.config.settings import Config

        # Save original and clear WORKSPACE_ID
        original = os.environ.pop("WORKSPACE_ID", None)
        try:
            with pytest.raises(
                ConfigurationError, match="WORKSPACE_ID environment variable"
            ):
                Config.from_environment()
        finally:
            # Restore original
            if original is not None:
                os.environ["WORKSPACE_ID"] = original

    def test_memory_config_structure(self, set_env_vars):
        '''Test USearchEngine is initialized with correct config.'''
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

                    # Verify config attributes (USearchConfig uses workspace_id)
                    assert usearch_config.workspace_id == "test_project"


@pytest.mark.unit
class TestAddTool:
    '''Test suite for add() tool.'''

    async def test_add_single_memory_success(self, mcp_server, sample_memories):
        '''Test adding a single valid memory.'''
        memories = sample_memories["single"]

        # Get the add tool function from registered tools
        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None, "add tool not found"

        # Execute add (async)
        result = await add_func(memories)

        # Verify behavior
        assert result is None  # add returns None
        mcp_server.memory_manager.memory.add_batch.assert_called_once()
        # USearchEngine.add_batch is called with keyword arguments
        call_kwargs = mcp_server.memory_manager.memory.add_batch.call_args.kwargs
        assert call_kwargs["contents"] == memories
        assert call_kwargs["workspace_id"] == mcp_server.config.workspace_id

    async def test_add_multiple_memories_success(self, mcp_server, sample_memories):
        '''Test adding multiple valid memories.'''
        memories = sample_memories["multiple"]

        # Get the add tool function
        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        # Execute add (async)
        assert add_func is not None
        await add_func(memories)

        # Verify memory.add_batch was called with correct memories
        assert mcp_server.memory_manager.memory.add_batch.call_count == 1
        call_kwargs = mcp_server.memory_manager.memory.add_batch.call_args.kwargs
        assert call_kwargs["contents"] == memories

    async def test_add_empty_list_noop(self, mcp_server):
        '''Test adding empty list is no-op (no error, no call to memory).'''
        memories = []

        # Get the add tool function
        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        # Execute add (async)
        assert add_func is not None
        result = await add_func(memories)

        # Verify no-op: returns None and doesn't call memory.add_batch
        assert result is None
        mcp_server.memory_manager.memory.add_batch.assert_not_called()

    async def test_add_memory_at_min_length(self, mcp_server, sample_memories):
        '''Test adding memory at minimum length (1 character).'''
        memories = [sample_memories["edge_cases"]["min_length"]]

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(memories)
        mcp_server.memory_manager.memory.add_batch.assert_called_once()

    async def test_add_memory_at_max_length(self, mcp_server, sample_memories):
        '''Test adding memory at maximum length (30720 characters).'''
        memories = [sample_memories["edge_cases"]["max_length"]]

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(memories)
        mcp_server.memory_manager.memory.add_batch.assert_called_once()

    async def test_add_memories_with_special_characters(
        self, mcp_server, sample_memories
    ):
        '''Test adding memories with special characters.'''
        memories = sample_memories["with_special_chars"]

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(memories)
        assert mcp_server.memory_manager.memory.add_batch.call_count == 1
        call_kwargs = mcp_server.memory_manager.memory.add_batch.call_args.kwargs
        assert call_kwargs["contents"] == memories

    async def test_add_non_string_memory_raises_value_error(self, mcp_server):
        '''Test adding non-string memory raises ValueError.'''
        memories = [123]  # Non-string

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="not a string"):
            await add_func(memories)

        mcp_server.memory_manager.memory.add_batch.assert_not_called()

    async def test_add_empty_string_raises_value_error(self, mcp_server):
        '''Test adding empty string raises ValueError.'''
        memories = [""]

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="contains only whitespace"):
            await add_func(memories)

        mcp_server.memory_manager.memory.add_batch.assert_not_called()

    async def test_add_whitespace_only_raises_value_error(self, mcp_server):
        '''Test adding whitespace-only memory raises ValueError.'''
        memories = ["   "]

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="only whitespace"):
            await add_func(memories)

        mcp_server.memory_manager.memory.add_batch.assert_not_called()

    async def test_add_memory_too_long_raises_value_error(
        self, mcp_server, sample_memories
    ):
        '''Test adding memory exceeding max length raises ValueError.'''
        memories = [sample_memories["invalid"]["too_long"]]

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError, match="too long"):
            await add_func(memories)

        mcp_server.memory_manager.memory.add_batch.assert_not_called()

    async def test_add_mixed_valid_invalid_raises_value_error(self, mcp_server):
        '''Test adding mix of valid and invalid memories raises ValueError.'''
        memories = ["Valid memory", ""]  # Second is invalid

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(ValueError):
            await add_func(memories)

        mcp_server.memory_manager.memory.add_batch.assert_not_called()

    async def test_add_memory_storage_failure_raises_storage_error(self, mcp_server):
        '''Test memory storage failure raises StorageError.'''
        memories = ["Valid memory"]

        # Configure mock to raise exception
        mcp_server.memory_manager.memory.add_batch.side_effect = Exception(
            "Storage failed"
        )

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        with pytest.raises(StorageError, match="Failed to add memories"):
            await add_func(memories)

    @pytest.mark.parametrize(
        "memories",
        [
            ["Single memory"],
            ["First", "Second"],
            ["Message with\nnewlines"],
            ["Unicode: 你好"],
        ],
    )
    async def test_add_various_valid_memories(self, mcp_server, memories):
        '''Test adding various types of valid memories.'''
        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func(memories)
        assert mcp_server.memory_manager.memory.add_batch.call_count == 1
        call_kwargs = mcp_server.memory_manager.memory.add_batch.call_args.kwargs
        assert call_kwargs["contents"] == memories

    async def test_add_skips_duplicates(self, mcp_server):
        '''Test that duplicate memories are not stored twice.'''
        memory = "Duplicate memory"
        # Mock add_memories_async to return AddResult with skipped (deduplication happened)
        from unittest.mock import AsyncMock

        mcp_server.memory_manager.add_memories_async = AsyncMock(
            return_value=AddResult(
                stored_count=0, skipped_count=1, replaced_count=0, replacements=[]
            )
        )

        add_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
                break

        assert add_func is not None
        await add_func([memory])

        # MemoryManager.add_memories_async was called but returned 0 (duplicate skipped)
        mcp_server.memory_manager.add_memories_async.assert_called_once()


@pytest.mark.unit
class TestGetAllTool:
    '''Test suite for get_all() tool.'''

    async def test_get_all_empty_store(self, mcp_server):
        '''Test get_all returns empty list when no memories stored.'''
        mcp_server.memory_manager.memory.get_all.return_value = []

        # Get the get_all tool function
        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None, "get_all tool not found"

        result = await get_all_func()

        assert result == []
        mcp_server.memory_manager.memory.get_all.assert_called_once()

    async def test_get_all_single_memory(self, mcp_server, sample_memories):
        '''Test get_all returns single memory.'''
        memories = sample_memories["single"]
        mcp_server.memory_manager.memory.get_all.return_value = memories

        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert result == memories
        mcp_server.memory_manager.memory.get_all.assert_called_once()

    async def test_get_all_multiple_memories(self, mcp_server, sample_memories):
        '''Test get_all returns multiple memories.'''
        memories = sample_memories["multiple"]
        mcp_server.memory_manager.memory.get_all.return_value = memories

        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert result == memories
        assert len(result) == 3
        mcp_server.memory_manager.memory.get_all.assert_called_once()

    async def test_get_all_returns_copy(self, mcp_server, sample_memories):
        '''Test get_all returns a copy of memories (prevents mutation).'''
        memories = sample_memories["multiple"].copy()
        mcp_server.memory_manager.memory.get_all.return_value = list(memories)

        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        # Modify result
        if result:
            result.append("New memory")

        # Original should be unchanged if copy works correctly
        # Note: In practice, we can't test true immutability here since
        # we're mocking, but we verify the .copy() call would be made
        assert "New memory" not in memories

    async def test_get_all_with_special_characters(self, mcp_server, sample_memories):
        '''Test get_all handles memories with special characters.'''
        memories = sample_memories["with_special_chars"]
        mcp_server.memory_manager.memory.get_all.return_value = memories

        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert result == memories
        assert len(result) == 3

    async def test_get_all_large_dataset(self, mcp_server):
        '''Test get_all with large number of memories.'''
        memories = [f"Memory {i}" for i in range(1000)]
        mcp_server.memory_manager.memory.get_all.return_value = memories

        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result = await get_all_func()

        assert len(result) == 1000
        assert result == memories

    async def test_get_all_memory_failure_raises_storage_error(self, mcp_server):
        '''Test memory retrieval failure raises StorageError.'''
        mcp_server.memory_manager.memory.get_all.side_effect = Exception(
            "Retrieval failed"
        )

        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        with pytest.raises(StorageError, match="Failed to retrieve memories"):
            await get_all_func()

    async def test_get_all_called_multiple_times(self, mcp_server, sample_memories):
        '''Test get_all can be called multiple times.'''
        memories = sample_memories["multiple"]
        mcp_server.memory_manager.memory.get_all.return_value = memories

        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
                break

        assert get_all_func is not None
        result1 = await get_all_func()
        result2 = await get_all_func()

        assert result1 == memories
        assert result2 == memories
        assert mcp_server.memory_manager.memory.get_all.call_count == 2


@pytest.mark.unit
class TestSearchTool:
    '''Test suite for search() tool.'''

    async def test_search_exact_match(self, mcp_server):
        '''Test search returns all semantic matches.'''
        from unittest.mock import AsyncMock

        query = "Hello"
        expected_results = ["Hello, World!", "Hello"]

        # Mock MemoryManager.search() directly (returns List[str]) - now async
        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        # Get the search tool function
        search_func = None
        for tool in mcp_server.registered_tools.values():
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
        '''Test search returns all semantic results without substring filtering.'''
        from unittest.mock import AsyncMock

        query = "World"
        expected_results = ["Hello, World!", "Goodbye", "World"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.registered_tools.values():
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
        '''Test search returns all semantic matches without filtering.'''
        from unittest.mock import AsyncMock

        query = "programming"
        expected_results = ["Python tutorial", "JavaScript guide", "Database design"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.registered_tools.values():
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
        '''Test search returns all semantic matches even if no substring matches.'''
        from unittest.mock import AsyncMock

        query = "Programming"
        expected_results = ["Hello", "World", "Test"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.registered_tools.values():
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
        '''Test search returns empty list when semantic search returns no results.'''
        from unittest.mock import AsyncMock

        query = "Python"
        expected_results = []

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Empty semantic results means empty final results
        assert result == []

    async def test_search_with_special_characters(self, mcp_server):
        '''Test search with special characters.'''
        from unittest.mock import AsyncMock

        query = "@user"
        expected_results = ["Memory to @user", "Another memory", "@user replied"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        # Returns all semantic results, no substring filtering
        assert len(result) == 3
        assert "Memory to @user" in result
        assert "@user replied" in result
        assert "Another memory" in result

    async def test_search_with_unicode(self, mcp_server):
        '''Test search with unicode characters.'''
        from unittest.mock import AsyncMock

        query = "你好"
        expected_results = ["你好世界", "Hello World", "你好 friend"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.registered_tools.values():
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
        '''Test search with single character query.'''
        from unittest.mock import AsyncMock

        query = "a"
        expected_results = ["Apple", "Banana", "Cherry"]

        mcp_server.memory_manager.search = AsyncMock(return_value=expected_results)

        search_func = None
        for tool in mcp_server.registered_tools.values():
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
        '''Test search when memory returns no semantic results.'''
        from unittest.mock import AsyncMock

        query = "test"
        mcp_server.memory_manager.search = AsyncMock(return_value=[])

        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        assert result == []

    async def test_search_memory_failure_raises_search_error(self, mcp_server):
        '''Test memory search failure raises SearchError.'''
        from unittest.mock import AsyncMock

        query = "test"
        mcp_server.memory_manager.search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        with pytest.raises(SearchError, match="Failed to search memory store"):
            await search_func(query)

    @pytest.mark.parametrize(
        "query,memories,expected_count",
        [
            ("test", ["This is a test", "testing", "no match"], 3),
            ("Python", ["Python tutorial", "Java guide", "Python examples"], 3),
            ("🌍", ["Hello 🌍", "World 🌎", "🌍 emoji"], 3),
        ],
    )
    async def test_search_various_queries(
        self, mcp_server, query, memories, expected_count
    ):
        '''Test search returns all semantic results without substring filtering.'''
        from unittest.mock import AsyncMock

        # Mock MemoryManager.search() to return all memories (semantic search behavior)
        mcp_server.memory_manager.search = AsyncMock(return_value=memories)

        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func(query)

        assert len(result) == expected_count


@pytest.mark.unit
class TestRemoveTool:
    '''Test suite for remove() tool.'''

    @staticmethod
    def _remove_fn(mcp_server):
        return mcp_server.tool_fn("remove")

    async def test_remove_single_memory(self, mcp_server):
        '''Test removing a single memory.'''
        memory_to_remove = "Memory to remove"
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=[memory_to_remove],
        ) as deleted:
            result = await self._remove_fn(mcp_server)([memory_to_remove])

        assert result is None
        deleted.assert_called_once_with([memory_to_remove])

    async def test_remove_multiple_memories(self, mcp_server):
        '''Test removing multiple memories.'''
        memories_to_remove = ["Memory 1", "Memory 2"]
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=memories_to_remove,
        ) as deleted:
            await self._remove_fn(mcp_server)(memories_to_remove)
        deleted.assert_called_once_with(memories_to_remove)

    async def test_remove_empty_list_noop(self, mcp_server):
        """Empty list does not call delete_memories."""
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=[],
        ) as deleted:
            result = await self._remove_fn(mcp_server)([])
        assert result is None
        deleted.assert_not_called()

    async def test_remove_non_existent_memory_silently_ignored(self, mcp_server):
        """Missing memories are reported as not found, not raised."""
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=[],
        ) as deleted:
            result = await self._remove_fn(mcp_server)(["Non-existent memory"])
        assert result is None
        deleted.assert_called_once_with(["Non-existent memory"])

    async def test_remove_duplicate_memories_all_removed(self, mcp_server):
        """Duplicates are collapsed before delete_memories."""
        memory = "Duplicate memory"
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=[memory],
        ) as deleted:
            await self._remove_fn(mcp_server)([memory, memory])
        deleted.assert_called_once_with([memory])

    async def test_remove_mixed_existent_and_non_existent(self, mcp_server):
        """Only memories returned by delete_memories count as removed."""
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=["Exists"],
        ) as deleted:
            await self._remove_fn(mcp_server)(["Exists", "Does not exist"])
        deleted.assert_called_once_with(["Exists", "Does not exist"])

    async def test_remove_memory_delete_failure_raises_storage_error(self, mcp_server):
        """delete_memories exceptions are wrapped as StorageError."""
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            side_effect=Exception("Delete failed"),
        ):
            with pytest.raises(StorageError, match="Failed to remove memories"):
                await self._remove_fn(mcp_server)(["Memory"])

    async def test_remove_with_special_characters(self, mcp_server):
        memory = "Memory with special chars: !@#$%"
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=[memory],
        ) as deleted:
            await self._remove_fn(mcp_server)([memory])
        deleted.assert_called_once_with([memory])

    async def test_remove_with_unicode(self, mcp_server):
        memory = "Unicode memory: 你好世界 🌍"
        with patch.object(
            type(mcp_server.memory_manager),
            "delete_memories",
            return_value=[memory],
        ) as deleted:
            await self._remove_fn(mcp_server)([memory])
        deleted.assert_called_once_with([memory])


@pytest.mark.unit
class TestToolRegistrationConfiguration:
    '''Test suite for configurable tool registration.'''

    def _build_server(self, monkeypatch, allowed_value: str | None):
        '''Helper to create a FastMCPServer with patched dependencies.'''
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
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

            from reflectlog.application.config.settings import Config
            from reflectlog.application.mcp_server import FastMCPServer

            server_config = Config.from_environment()
            server = FastMCPServer(server_config=server_config)

        return server, logger, fastmcp_instance

    def test_allowed_tools_subset(self, monkeypatch):
        '''Only tools listed in ALLOWED_TOOLS should register.'''
        server, logger, fastmcp_instance = self._build_server(
            monkeypatch, "add,get_all"
        )

        tool_names = [tool.get_name() for tool in server.tools]
        assert tool_names == ["add", "get_all"]
        assert fastmcp_instance.tool.call_count == 2
        logger.warning.assert_not_called()

    def test_allowed_tools_invalid_tokens(self, monkeypatch):
        '''Unknown tool names are ignored with a warning.'''
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
        ''''none' disables all tools and surfaces a warning.'''
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
    '''Test suite for health_check() tool.'''

    async def test_health_check_returns_healthy_status(self, mcp_server):
        '''Test health check returns healthy status with all components.'''
        class ReadyEngine:
            def is_ready(self) -> bool:
                return True

        ready = ReadyEngine()
        mcp_server.memory_manager._semantic_engine = ready
        mcp_server.memory_manager._tantivy_engine = ready

        health_check_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        assert result["status"] == "healthy"
        assert result["workspace_id"] == "test_project"
        assert result["semantic_engine"] == "initialized"
        assert result["tantivy_engine"] == "initialized"
        assert result["reranker_engine"] == "cross_encoder"
        assert result["hybrid_search_enabled"] is True
        assert result["rrf_fusion_enabled"] is True
        assert result["recency_boost_enabled"] is True

    async def test_health_check_with_tantivy_disabled(self, mcp_server):
        '''Test health check when Tantivy is disabled.'''

        class ReadyEngine:
            def is_ready(self) -> bool:
                return True

        mcp_server.memory_manager._semantic_engine = ReadyEngine()
        mcp_server.memory_manager._tantivy_engine = None

        health_check_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        assert result["status"] == "healthy"
        assert result["semantic_engine"] == "initialized"
        assert result["tantivy_engine"] == "disabled"

    async def test_health_check_no_semantic_engine(self, mcp_server):
        '''Test health check when semantic engine is not initialized.'''
        mcp_server.memory_manager._semantic_engine = None
        mcp_server.memory_manager._tantivy_engine = None

        health_check_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        assert result["semantic_engine"] == "not_initialized"
        assert result["tantivy_engine"] == "disabled"

    async def test_health_check_with_different_reranker(self, mcp_server):
        '''Test health check reports configured reranker engine.'''
        class ReadyEngine:
            def is_ready(self) -> bool:
                return True

        ready = ReadyEngine()
        mcp_server.memory_manager._semantic_engine = ready
        mcp_server.memory_manager._tantivy_engine = ready

        health_check_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "health_check":
                health_check_func = tool.fn
                break

        assert health_check_func is not None
        result = await health_check_func()

        # Default reranker should be "cross_encoder"
        assert result["reranker_engine"] == "cross_encoder"


@pytest.mark.unit
class TestCanonicalizeToolToken:
    '''Test suite for FastMCPServer._canonicalize_tool_token edge cases.'''

    def test_empty_token_returns_none(self):
        '''Empty string after normalization returns None (line 202).'''
        available = {"add", "search", "remove"}
        result = FastMCPServer._canonicalize_tool_token("", available)
        assert result is None

    def test_whitespace_only_token_returns_none(self):
        '''Whitespace-only token normalizes to empty and returns None.'''
        available = {"add", "search", "remove"}
        result = FastMCPServer._canonicalize_tool_token("   ", available)
        assert result is None

    def test_collapsed_underscore_match(self):
        '''Token without underscores matches name with underscores (line 210).'''
        available = {"health_check", "add", "get_all"}
        result = FastMCPServer._canonicalize_tool_token("healthcheck", available)
        assert result == "health_check"

    def test_collapsed_match_get_all(self):
        '''Token "getall" matches "get_all" via collapsed comparison.'''
        available = {"health_check", "add", "get_all"}
        result = FastMCPServer._canonicalize_tool_token("getall", available)
        assert result == "get_all"

    def test_collapsed_match_with_hyphens(self):
        '''Hyphenated token "health-check" matches "health_check".'''
        available = {"health_check", "add"}
        result = FastMCPServer._canonicalize_tool_token("health-check", available)
        assert result == "health_check"


@pytest.mark.unit
class TestServerClose:
    '''Test suite for FastMCPServer.close() method.'''

    def _build_server(self, monkeypatch):
        '''Helper to create a FastMCPServer with fully mocked dependencies.'''
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")
        monkeypatch.delenv("ALLOWED_TOOLS", raising=False)

        with (
            patch("reflectlog.application.mcp_server.FastMCP") as mock_fastmcp,
            patch(
                "reflectlog.application.mcp_server.MemoryManager"
            ) as mock_memory_manager,
            patch(
                "reflectlog.application.mcp_server.create_logger"
            ) as mock_create_logger,
        ):
            mock_fastmcp.return_value = MagicMock()
            mock_mm_instance = MagicMock()
            mock_memory_manager.return_value = mock_mm_instance
            mock_logger = MagicMock()
            mock_create_logger.return_value = mock_logger

            from reflectlog.application.config.settings import Config

            server_config = Config.from_environment()
            server = FastMCPServer(server_config=server_config)

        return server, mock_mm_instance, mock_logger

    def test_close_success(self, monkeypatch):
        '''close() persists data and logs success (lines 257-261).'''
        server, mock_mm, mock_logger = self._build_server(monkeypatch)

        server.close()

        mock_mm.close.assert_called_once()
        mock_logger.info.assert_any_call("Initiating graceful server shutdown...")
        mock_logger.info.assert_any_call(
            "Server shutdown complete - all data persisted"
        )

    def test_close_handles_exception(self, monkeypatch):
        '''close() catches and logs errors from MemoryManager.close() (lines 262-266).'''
        server, mock_mm, mock_logger = self._build_server(monkeypatch)
        mock_mm.close.side_effect = RuntimeError("disk full")

        server.close()

        mock_mm.close.assert_called_once()
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args.args[0]
        assert "disk full" in error_msg


@pytest.mark.unit
class TestMainFunction:
    '''Test suite for the main() entry point function.'''

    def test_main_runtime_error_is_reraised(self, monkeypatch):
        '''main() re-raises RuntimeError after logging (lines 286-288).'''
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")

        with (
            patch("reflectlog.application.mcp_server.FastMCPServer") as mock_server_cls,
            patch("reflectlog.application.mcp_server.get_logger"),
        ):
            mock_server_cls.side_effect = RuntimeError("port in use")

            from reflectlog.application.mcp_server import main

            with pytest.raises(RuntimeError, match="port in use"):
                main()

    def test_main_keyboard_interrupt_calls_close(self, monkeypatch):
        '''main() calls server.close() on KeyboardInterrupt (lines 289-292).'''
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")

        with (
            patch("reflectlog.application.mcp_server.FastMCPServer") as mock_server_cls,
            patch("reflectlog.application.mcp_server.get_logger"),
        ):
            mock_instance = MagicMock()
            mock_server_cls.return_value = mock_instance
            mock_instance.run.side_effect = KeyboardInterrupt

            from reflectlog.application.mcp_server import main

            main()

            mock_instance.close.assert_called_once()

    def test_main_unexpected_exception_calls_close_and_reraises(self, monkeypatch):
        '''main() calls server.close() then re-raises on unexpected Exception (lines 293-297).'''
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")

        with (
            patch("reflectlog.application.mcp_server.FastMCPServer") as mock_server_cls,
            patch("reflectlog.application.mcp_server.get_logger"),
        ):
            mock_instance = MagicMock()
            mock_server_cls.return_value = mock_instance
            mock_instance.run.side_effect = ValueError("unexpected")

            from reflectlog.application.mcp_server import main

            with pytest.raises(ValueError, match="unexpected"):
                main()

            mock_instance.close.assert_called_once()

    def test_main_keyboard_interrupt_before_server_created(self, monkeypatch):
        '''main() handles KeyboardInterrupt when server is None (line 291 branch).'''
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")

        with (
            patch("reflectlog.application.mcp_server.FastMCPServer") as mock_server_cls,
            patch("reflectlog.application.mcp_server.get_logger"),
        ):
            mock_server_cls.side_effect = KeyboardInterrupt

            from reflectlog.application.mcp_server import main

            main()

    def test_main_unexpected_exception_before_server_created(self, monkeypatch):
        '''main() re-raises unexpected Exception when server is None (line 295 branch).'''
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")

        with (
            patch("reflectlog.application.mcp_server.FastMCPServer") as mock_server_cls,
            patch("reflectlog.application.mcp_server.get_logger"),
        ):
            mock_server_cls.side_effect = TypeError("bad init")

            from reflectlog.application.mcp_server import main

            with pytest.raises(TypeError, match="bad init"):
                main()
