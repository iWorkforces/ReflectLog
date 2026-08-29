'''Integration tests for MCP server workflows.'''
# mypy: disable-error-code="misc,var-annotated"

import pytest


@pytest.mark.integration
class TestMCPWorkflows:
    '''Integration tests for complete MCP tool workflows.'''

    @pytest.mark.asyncio
    async def test_add_then_get_all_workflow(self, mcp_server, sample_memories):
        '''Test adding memories and retrieving them with get_all.'''
        memories = sample_memories["multiple"]

        # Setup mocks
        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def get_all_side_effect(**kwargs):
            return stored_memories.copy()

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

        # Get tool functions
        from typing import Any, Callable

        add_func: Callable[..., Any] | None = None
        get_all_func: Callable[..., Any] | None = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn

        # Execute workflow: Add -> Get All
        assert add_func is not None
        assert get_all_func is not None
        await add_func(memories)
        result = await get_all_func()

        # Verify
        assert len(result) == 3
        assert result == memories

    @pytest.mark.asyncio
    async def test_add_then_search_workflow(self, mcp_server, create_search_results):
        '''Test adding memories and searching for them.'''
        memories = ["Python tutorial", "Java guide", "Python examples"]

        # Setup mocks
        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def search_side_effect(query, **kwargs):
            # Simulate semantic search returning memories containing query
            matching = [mem for mem in stored_memories if query.lower() in mem.lower()]
            return create_search_results(matching)

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect

        # Get tool functions
        add_func = None
        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn

        # Execute workflow: Add -> Search
        assert add_func is not None
        assert search_func is not None
        await add_func(memories)
        object.__setattr__(
            mcp_server.memory_manager.config, "enable_rrf_fusion", False
        )
        object.__setattr__(
            mcp_server.memory_manager.config, "reranker_engine", "none"
        )
        result = await search_func("Python")

        # Verify
        assert len(result) == 2
        assert "Python tutorial" in result
        assert "Python examples" in result
        assert "Java guide" not in result

    @pytest.mark.asyncio
    async def test_add_remove_get_all_workflow(self, mcp_server, create_search_results):
        '''Test adding, removing, and verifying with get_all.'''
        initial_memories = ["Memory 1", "Memory 2", "Memory 3"]
        memory_to_remove = "Memory 2"

        # Setup mocks
        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def search_side_effect(query, **kwargs):
            # Return exact matches for removal
            matching = [mem for mem in stored_memories if mem == query]
            return create_search_results(matching)

        def delete_side_effect(memory_id=None, workspace_id=None):
            # Remove by numeric ID (matches MemoryStore auto-increment IDs)
            if memory_id is None:
                return
            try:
                idx = int(memory_id)
            except (TypeError, ValueError):
                return
            if 0 <= idx < len(stored_memories):
                stored_memories.pop(idx)

        def get_all_side_effect(**kwargs):
            return stored_memories.copy()

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        mcp_server.memory_manager.memory.delete.side_effect = delete_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect
        mcp_server.memory_manager.memory.get_id_by_memory.side_effect = (
            lambda workspace_id, memory: stored_memories.index(memory)
            if memory in stored_memories
            else None
        )
        mcp_server.memory_manager.memory.get_id_by_content.side_effect = (
            lambda workspace_id, memory: stored_memories.index(memory)
            if memory in stored_memories
            else None
        )

        # Get tool functions
        add_func = None
        remove_func = None
        get_all_func = None
        for tool in mcp_server.registered_tools.values():
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
        await add_func(initial_memories)

        # Verify initial state
        all_memories = await get_all_func()
        assert len(all_memories) == 3

        # Remove one memory
        await remove_func([memory_to_remove])

        # Verify final state
        remaining = await get_all_func()
        assert len(remaining) == 2
        assert "Memory 1" in remaining
        assert "Memory 3" in remaining
        assert "Memory 2" not in remaining

    @pytest.mark.asyncio
    async def test_add_search_remove_search_workflow(
        self, mcp_server, create_search_results
    ):
        '''Test complex workflow: add, search, remove subset, search again.'''
        memories = [
            "Python tutorial for beginners",
            "Advanced Python techniques",
            "Java programming guide",
            "Python data science",
        ]

        # Setup mocks
        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def search_side_effect(query, **kwargs):
            matching = [mem for mem in stored_memories if query.lower() in mem.lower()]
            return create_search_results(matching)

        def delete_side_effect(memory_id=None, workspace_id=None):
            if memory_id is None:
                return
            try:
                idx = int(memory_id)
            except (TypeError, ValueError):
                return
            if 0 <= idx < len(stored_memories):
                stored_memories.pop(idx)

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        mcp_server.memory_manager.memory.delete.side_effect = delete_side_effect
        mcp_server.memory_manager.memory.get_id_by_memory.side_effect = (
            lambda workspace_id, memory: stored_memories.index(memory)
            if memory in stored_memories
            else None
        )
        mcp_server.memory_manager.memory.get_id_by_content.side_effect = (
            lambda workspace_id, memory: stored_memories.index(memory)
            if memory in stored_memories
            else None
        )

        # Get tool functions
        add_func = None
        search_func = None
        remove_func = None
        for tool in mcp_server.registered_tools.values():
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
        # 1. Add all memories
        await add_func(memories)

        # 2. Search for "Python" - should find 3
        object.__setattr__(
            mcp_server.memory_manager.config, "enable_rrf_fusion", False
        )
        object.__setattr__(
            mcp_server.memory_manager.config, "reranker_engine", "none"
        )
        python_results_before = await search_func("Python")
        assert len(python_results_before) == 3

        # 3. Remove one Python memory
        await remove_func(["Python tutorial for beginners"])

        # 4. Search again - should find 2
        python_results_after = await search_func("Python")
        assert len(python_results_after) == 2
        assert "Advanced Python techniques" in python_results_after
        assert "Python data science" in python_results_after
        assert "Python tutorial for beginners" not in python_results_after

    @pytest.mark.asyncio
    async def test_multiple_add_operations(self, mcp_server):
        '''Test multiple add operations in sequence.'''
        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def get_all_side_effect(**kwargs):
            return stored_memories.copy()

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

        # Get tool functions
        add_func = None
        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn

        # Add memories in batches
        assert add_func is not None
        assert get_all_func is not None
        await add_func(["Memory 1", "Memory 2"])
        await add_func(["Memory 3"])
        await add_func(["Memory 4", "Memory 5"])

        result = await get_all_func()
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_empty_store_operations(self, mcp_server):
        '''Test operations on empty store.'''
        mcp_server.memory_manager.memory.get_all.return_value = []
        mcp_server.memory_manager.memory.search.return_value = []

        # Get tool functions
        get_all_func = None
        search_func = None
        remove_func = None
        for tool in mcp_server.registered_tools.values():
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
        all_memories = await get_all_func()
        assert all_memories == []

        search_results = await search_func("anything")
        assert search_results == []

        # Remove from empty store (no-op)
        result = await remove_func(["non-existent"])
        assert result is None

    @pytest.mark.asyncio
    async def test_add_duplicate_memories(self, mcp_server, create_search_results):
        '''Test adding duplicate memories and removing them.'''
        duplicate_memory = "Duplicate memory"
        memories = [duplicate_memory, "Unique memory", duplicate_memory]

        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            for mem in batch or []:
                if mem not in stored_memories:
                    stored_memories.append(mem)
            return batch or []

        def search_side_effect(query, **kwargs):
            matching = [mem for mem in stored_memories if mem == query]
            return create_search_results(matching)

        def delete_side_effect(memory_id=None, workspace_id=None):
            if memory_id is None:
                return
            try:
                idx = int(memory_id)
            except (TypeError, ValueError):
                return
            if 0 <= idx < len(stored_memories):
                stored_memories.pop(idx)

        def get_all_side_effect(**kwargs):
            return stored_memories.copy()

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        mcp_server.memory_manager.memory.delete.side_effect = delete_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect
        mcp_server.memory_manager.memory.get_id_by_memory.side_effect = (
            lambda workspace_id, memory: stored_memories.index(memory)
            if memory in stored_memories
            else None
        )
        mcp_server.memory_manager.memory.get_id_by_content.side_effect = (
            lambda workspace_id, memory: stored_memories.index(memory)
            if memory in stored_memories
            else None
        )

        # Get tool functions
        add_func = None
        remove_func = None
        get_all_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "remove":
                remove_func = tool.fn
            elif tool.name == "get_all":
                get_all_func = tool.fn

        # Add memories with duplicates
        assert add_func is not None
        assert remove_func is not None
        assert get_all_func is not None
        await add_func(memories)

        all_mems = await get_all_func()
        assert len(all_mems) == 2

        # Remove duplicates (should remove all occurrences)
        await remove_func([duplicate_memory])

        remaining = await get_all_func()
        assert len(remaining) == 1
        assert "Unique memory" in remaining

    @pytest.mark.asyncio
    async def test_search_with_no_semantic_matches(self, mcp_server):
        '''Test search when semantic search returns no results.'''
        mcp_server.memory_manager.memory.search.return_value = []

        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "search":
                search_func = tool.fn
                break

        assert search_func is not None
        result = await search_func("nonexistent query")
        assert result == []

    @pytest.mark.asyncio
    async def test_large_dataset_workflow(self, mcp_server, create_search_results):
        '''Test workflow with large number of memories.'''
        large_dataset = [f"Memory {i}" for i in range(100)]

        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def get_all_side_effect(**kwargs):
            return stored_memories.copy()

        def search_side_effect(query, **kwargs):
            matching = [mem for mem in stored_memories if query in mem]
            return create_search_results(matching[:5])  # Limit to 5 as per config

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect

        # Get tool functions
        add_func = None
        get_all_func = None
        search_func = None
        for tool in mcp_server.registered_tools.values():
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

        all_memories = await get_all_func()
        assert len(all_memories) == 100

        # Search returns limited results
        search_results = await search_func("Memory")
        assert len(search_results) <= 5

    @pytest.mark.asyncio
    async def test_special_characters_workflow(self, mcp_server, create_search_results):
        '''Test workflow with memories containing special characters.'''
        from unittest.mock import AsyncMock, MagicMock

        special_memories = [
            "Email: user@example.com",
            "Price: $100.00",
            "Math: 2 + 2 = 4",
            "Code: function() { return true; }",
        ]

        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def search_side_effect(query, **kwargs):
            matching = [mem for mem in stored_memories if query in mem]
            return create_search_results(matching)

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        object.__setattr__(mcp_server.memory_manager.config, "reranker_engine", "none")
        object.__setattr__(
            mcp_server.memory_manager.config, "fusion_ranking_threshold", 0.0
        )

        def semantic_search(query, **_kwargs):
            matching = [mem for mem in stored_memories if query in mem]
            return [(mem, 0.9, "2026-08-22T00:00:00+00:00") for mem in matching]

        def tantivy_search(query, *_args, **_kwargs):
            matching = [mem for mem in stored_memories if query in mem]
            return [(mem, 0.8) for mem in matching]

        mcp_server.memory_manager._semantic_engine.search.side_effect = semantic_search
        mcp_server.memory_manager._tantivy_engine.search.side_effect = tantivy_search

        add_func = None
        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn

        assert add_func is not None
        assert search_func is not None
        await add_func(special_memories)

        email_results = await search_func("@")
        assert len(email_results) == 1
        assert "user@example.com" in email_results[0]

        price_results = await search_func("$")
        assert len(price_results) == 1
        assert "$100.00" in price_results[0]

    @pytest.mark.asyncio
    async def test_unicode_workflow(self, mcp_server, create_search_results):
        '''Test workflow with unicode memories.'''
        from unittest.mock import AsyncMock, MagicMock

        unicode_memories = [
            "Hello in Chinese: 你好",
            "Hello in Japanese: こんにちは",
            "Hello in Arabic: مرحبا",
            "Emoji: 😀 🌍 🚀",
        ]

        stored_memories = []

        def add_side_effect(
            *,
            workspace_id=None,
            memories=None,
            contents=None,
            infer=True,
            **_kwargs,
        ):
            batch = contents if contents is not None else memories
            stored_memories.extend(batch or [])
            return batch or []

        def search_side_effect(query, **kwargs):
            matching = [mem for mem in stored_memories if query.lower() in mem.lower()]
            return create_search_results(matching)

        mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect
        mcp_server.memory_manager.memory.search.side_effect = search_side_effect
        object.__setattr__(mcp_server.memory_manager.config, "reranker_engine", "none")
        object.__setattr__(
            mcp_server.memory_manager.config, "fusion_ranking_threshold", 0.0
        )

        def semantic_search(query, **_kwargs):
            matching = [
                mem for mem in stored_memories if query.lower() in mem.lower()
            ]
            return [(mem, 0.9, "2026-08-22T00:00:00+00:00") for mem in matching]

        def tantivy_search(query, *_args, **_kwargs):
            matching = [
                mem for mem in stored_memories if query.lower() in mem.lower()
            ]
            return [(mem, 0.8) for mem in matching]

        mcp_server.memory_manager._semantic_engine.search.side_effect = semantic_search
        mcp_server.memory_manager._tantivy_engine.search.side_effect = tantivy_search

        add_func = None
        search_func = None
        for tool in mcp_server.registered_tools.values():
            if tool.name == "add":
                add_func = tool.fn
            elif tool.name == "search":
                search_func = tool.fn

        assert add_func is not None
        assert search_func is not None
        await add_func(unicode_memories)

        chinese_results = await search_func("你好")
        assert len(chinese_results) == 1

        emoji_results = await search_func("😀")
        assert len(emoji_results) == 1
