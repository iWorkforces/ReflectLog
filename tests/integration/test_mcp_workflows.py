"""Integration tests for MCP server workflows."""
# mypy: disable-error-code="misc,var-annotated"

import pytest


@pytest.mark.integration
class TestMCPWorkflows:
    """Integration tests for complete MCP tool workflows."""

    @pytest.mark.asyncio
    async def test_add_then_get_all_workflow(self, mcp_server, sample_messages):
        """Test adding messages and retrieving them with get_all."""
        messages = sample_messages["multiple"]

        # Setup mocks
        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def get_all_side_effect(**kwargs):
            return {"results": [{"memory": msg} for msg in stored_messages]}

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

        # Get tool functions
        from typing import Callable, Any

        add_func: Callable[..., Any] | None = None
        get_all_func: Callable[..., Any] | None = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn

        # Execute workflow: Add -> Get All
        assert add_func is not None
        assert get_all_func is not None
        await add_func(messages)
        result = await get_all_func()

        # Verify
        assert len(result) == 3
        assert result == messages

    @pytest.mark.asyncio
    async def test_add_then_search_workflow(self, mcp_server, create_search_results):
        """Test adding messages and searching for them."""
        messages = ["Python tutorial", "Java guide", "Python examples"]

        # Setup mocks
        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def search_side_effect(query, **kwargs):
            # Simulate semantic search returning messages containing query
            matching = [msg for msg in stored_messages if query.lower() in msg.lower()]
            return {
                "results": [
                    {"memory": msg, "id": f"id_{i}"} for i, msg in enumerate(matching)
                ]
            }

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect

        # Get tool functions
        add_func = None
        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn

        # Execute workflow: Add -> Search
        assert add_func is not None
        assert search_func is not None
        await add_func(messages)
        result = await search_func("Python")

        # Verify
        assert len(result) == 2
        assert "Python tutorial" in result
        assert "Python examples" in result
        assert "Java guide" not in result

    @pytest.mark.asyncio
    async def test_add_remove_get_all_workflow(self, mcp_server, create_search_results):
        """Test adding, removing, and verifying with get_all."""
        initial_messages = ["Message 1", "Message 2", "Message 3"]
        message_to_remove = "Message 2"

        # Setup mocks
        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def search_side_effect(query, **kwargs):
            # Return exact matches for removal
            matching = [
                (msg, stored_messages.index(msg))
                for msg in stored_messages
                if msg == query
            ]
            return {
                "results": [{"memory": msg, "id": f"id_{idx}"} for msg, idx in matching]
            }

        def delete_side_effect(memory_id=None, project_id=None):
            # Remove by message value (find and remove)
            if memory_id and memory_id.startswith("id_"):
                idx = int(memory_id.split("_")[1])
                if idx < len(stored_messages):
                    stored_messages.pop(idx)

        def get_all_side_effect(**kwargs):
            return {"results": [{"memory": msg} for msg in stored_messages]}

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        mcp_server.memory_manager.memory.delete.side_effect = delete_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

        # Get tool functions
        add_func = None
        remove_func = None
        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "remove":
                remove_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn

        # Execute workflow: Add -> Remove -> Get All
        assert add_func is not None
        assert remove_func is not None
        assert get_all_func is not None
        await add_func(initial_messages)

        # Verify initial state
        all_messages = await get_all_func()
        assert len(all_messages) == 3

        # Remove one message
        await remove_func([message_to_remove])

        # Verify final state
        remaining = await get_all_func()
        assert len(remaining) == 2
        assert "Message 1" in remaining
        assert "Message 3" in remaining
        assert "Message 2" not in remaining

    @pytest.mark.asyncio
    async def test_add_search_remove_search_workflow(
        self, mcp_server, create_search_results
    ):
        """Test complex workflow: add, search, remove subset, search again."""
        messages = [
            "Python tutorial for beginners",
            "Advanced Python techniques",
            "Java programming guide",
            "Python data science",
        ]

        # Setup mocks
        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def search_side_effect(query, **kwargs):
            matching = [msg for msg in stored_messages if query.lower() in msg.lower()]
            return {
                "results": [
                    {"memory": msg, "id": f"id_{i}"} for i, msg in enumerate(matching)
                ]
            }

        def delete_side_effect(memory_id=None, project_id=None):
            if memory_id and memory_id.startswith("id_"):
                idx = int(memory_id.split("_")[1])
                stored_messages.pop(idx) if idx < len(stored_messages) else None

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        mcp_server.memory_manager.memory.delete.side_effect = delete_side_effect

        # Get tool functions
        add_func = None
        search_func = None
        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn
            elif tool.name == "remove":
                remove_func = tool.fn

        # Execute complex workflow
        assert add_func is not None
        assert search_func is not None
        assert remove_func is not None
        # 1. Add all messages
        await add_func(messages)

        # 2. Search for "Python" - should find 3
        python_results_before = await search_func("Python")
        assert len(python_results_before) == 3

        # 3. Remove one Python message
        await remove_func(["Python tutorial for beginners"])

        # 4. Search again - should find 2
        python_results_after = await search_func("Python")
        assert len(python_results_after) == 2
        assert "Advanced Python techniques" in python_results_after
        assert "Python data science" in python_results_after
        assert "Python tutorial for beginners" not in python_results_after

    @pytest.mark.asyncio
    async def test_multiple_add_operations(self, mcp_server):
        """Test multiple add operations in sequence."""
        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def get_all_side_effect(**kwargs):
            return {"results": [{"memory": msg} for msg in stored_messages]}

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

        # Get tool functions
        add_func = None
        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn

        # Add messages in batches
        assert add_func is not None
        assert get_all_func is not None
        await add_func(["Message 1", "Message 2"])
        await add_func(["Message 3"])
        await add_func(["Message 4", "Message 5"])

        result = await get_all_func()
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_empty_store_operations(self, mcp_server):
        """Test operations on empty store."""
        mcp_server.memory_manager.memory.get_all.return_value = []
        mcp_server.memory_manager.memory.search.return_value = []

        # Get tool functions
        get_all_func = None
        search_func = None
        remove_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "get_all":
                get_all_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn
            elif tool.name == "remove":
                remove_func = tool.fn

        # Operations on empty store should not fail
        assert get_all_func is not None
        assert search_func is not None
        assert remove_func is not None
        all_messages = await get_all_func()
        assert all_messages == []

        search_results = await search_func("anything")
        assert search_results == []

        # Remove from empty store (no-op)
        result = await remove_func(["non-existent"])
        assert result is None

    @pytest.mark.asyncio
    async def test_add_duplicate_messages(self, mcp_server, create_search_results):
        """Test adding duplicate messages and removing them."""
        duplicate_message = "Duplicate message"
        messages = [duplicate_message, "Unique message", duplicate_message]

        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def search_side_effect(query, **kwargs):
            matching = [msg for msg in stored_messages if msg == query]
            return {
                "results": [
                    {"memory": msg, "id": f"id_{i}"} for i, msg in enumerate(matching)
                ]
            }

        def delete_side_effect(memory_id=None, project_id=None):
            if memory_id and memory_id.startswith("id_"):
                idx = int(memory_id.split("_")[1])
                stored_messages.pop(idx) if idx < len(stored_messages) else None

        def get_all_side_effect(**kwargs):
            return {"results": [{"memory": msg} for msg in stored_messages]}

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        mcp_server.memory_manager.memory.delete.side_effect = delete_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

        # Get tool functions
        add_func = None
        remove_func = None
        get_all_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "remove":
                remove_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn

        # Add messages with duplicates
        assert add_func is not None
        assert remove_func is not None
        assert get_all_func is not None
        await add_func(messages)

        all_msgs = await get_all_func()
        assert len(all_msgs) == 3

        # Remove duplicates (should remove all occurrences)
        await remove_func([duplicate_message])

        remaining = await get_all_func()
        assert len(remaining) == 1
        assert "Unique message" in remaining

    @pytest.mark.asyncio
    async def test_search_with_no_semantic_matches(self, mcp_server):
        """Test search when semantic search returns no results."""
        mcp_server.memory_manager.memory.search.return_value = []

        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func("nonexistent query")
        assert result == []

    @pytest.mark.asyncio
    async def test_large_dataset_workflow(self, mcp_server, create_search_results):
        """Test workflow with large number of messages."""
        large_dataset = [f"Message {i}" for i in range(100)]

        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def get_all_side_effect(**kwargs):
            return {"results": [{"memory": msg} for msg in stored_messages]}

        def search_side_effect(query, **kwargs):
            matching = [msg for msg in stored_messages if query in msg]
            return create_search_results(matching[:5])  # Limit to 5 as per config

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect

        # Get tool functions
        add_func = None
        get_all_func = None
        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn

        # Add large dataset
        assert add_func is not None
        assert get_all_func is not None
        assert search_func is not None
        await add_func(large_dataset)

        all_messages = await get_all_func()
        assert len(all_messages) == 100

        # Search returns limited results
        search_results = await search_func("Message")
        assert len(search_results) <= 5

    @pytest.mark.asyncio
    async def test_special_characters_workflow(self, mcp_server, create_search_results):
        """Test workflow with messages containing special characters."""
        special_messages = [
            "Email: user@example.com",
            "Price: $100.00",
            "Math: 2 + 2 = 4",
            "Code: function() { return true; }",
        ]

        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def search_side_effect(query, **kwargs):
            matching = [msg for msg in stored_messages if query in msg]
            return {
                "results": [
                    {"memory": msg, "id": f"id_{i}"} for i, msg in enumerate(matching)
                ]
            }

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect

        # Get tool functions
        add_func = None
        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn

        # Add messages with special characters
        assert add_func is not None
        assert search_func is not None
        await add_func(special_messages)

        # Search for special characters
        email_results = await search_func("@")
        assert len(email_results) == 1
        assert "user@example.com" in email_results[0]

        price_results = await search_func("$")
        assert len(price_results) == 1
        assert "$100.00" in price_results[0]

    @pytest.mark.asyncio
    async def test_unicode_workflow(self, mcp_server, create_search_results):
        """Test workflow with unicode messages."""
        unicode_messages = [
            "Hello in Chinese: 你好",
            "Hello in Japanese: こんにちは",
            "Hello in Arabic: مرحبا",
            "Emoji: 😀 🌍 🚀",
        ]

        stored_messages = []

        def add_side_effect(message, project_id=None, infer=True):
            stored_messages.append(message)

        def search_side_effect(query, **kwargs):
            matching = [msg for msg in stored_messages if query.lower() in msg.lower()]
            return {
                "results": [
                    {"memory": msg, "id": f"id_{i}"} for i, msg in enumerate(matching)
                ]
            }

        mcp_server.memory_manager.memory.add.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect

        # Get tool functions
        add_func = None
        search_func = None
        for tool in mcp_server.mcp._tool_manager._tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn

        # Add unicode messages
        assert add_func is not None
        assert search_func is not None
        await add_func(unicode_messages)

        # Search for unicode
        chinese_results = await search_func("你好")
        assert len(chinese_results) == 1

        emoji_results = await search_func("😀")
        assert len(emoji_results) == 1
