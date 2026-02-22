'''Demonstration test: Add 10 memories, get_all, and search.'''

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_demo_workflow(
    add_tool, get_all_tool, search_tool, mcp_server, create_search_results
):
    '''Demonstration: Add 10 memories, retrieve all, and search.'''

    # Prepare 10 memories
    memories = [
        "Python tutorial for beginners",
        "Advanced Python programming",
        "JavaScript fundamentals",
        "Python data science guide",
        "Java enterprise patterns",
        "Python machine learning",
        "React frontend development",
        "Python web scraping",
        "Docker containerization",
        "Python automation scripts",
    ]

    print("\n" + "=" * 70)
    print("📝 STEP 1: Adding 10 memories to memory store")
    print("=" * 70)
    for i, mem in enumerate(memories, 1):
        print(f"  {i}. {mem}")

    # Setup mock behavior for add
    stored_memories = []

    def add_side_effect(*, memories=None, **_kwargs):
        memories = memories or []
        stored_memories.extend(memories)
        print(f"\n✅ Successfully added {len(memories)} memorie(s) to store")
        return memories

    mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect

    # Add memories
    await add_tool(memories)

    print(f"\n📊 Total memories in store: {len(stored_memories)}")

    # Setup mock behavior for get_all
    def get_all_side_effect(*_args, **_kwargs):
        return stored_memories.copy()

    mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

    print("\n" + "=" * 70)
    print("📥 STEP 2: Retrieving ALL memories with get_all()")
    print("=" * 70)

    # Get all memories
    all_memories = await get_all_tool()

    print(f"\n✅ Retrieved {len(all_memories)} memorie(s):")
    for i, mem in enumerate(all_memories, 1):
        print(f"  {i}. {mem}")

    # Verify all memories retrieved
    assert len(all_memories) == 10
    assert all_memories == memories

    # Setup mock behavior for search
    def search_side_effect(query, **kwargs):
        # Simulate semantic + substring search
        matching = [mem for mem in stored_memories if query.lower() in mem.lower()]
        return create_search_results(matching)

    mcp_server.memory_manager.memory.search.side_effect = search_side_effect

    print("\n" + "=" * 70)
    print("🔍 STEP 3: Searching for 'Python' memories")
    print("=" * 70)

    # Disable reranking/fusion for deterministic test results
    mcp_server.memory_manager.config.reranker_engine = "none"
    mcp_server.memory_manager.config.enable_rrf_fusion = False
    mcp_server.memory_manager.config.search_limit = len(memories)

    # Search for Python memories
    python_results = await search_tool("Python")

    print(f"\n✅ Found {len(python_results)} memorie(s) containing 'Python':")
    for i, mem in enumerate(python_results, 1):
        print(f"  {i}. {mem}")

    # Verify search results
    assert len(python_results) == 6
    assert "Python tutorial for beginners" in python_results
    assert "Advanced Python programming" in python_results
    assert "Python data science guide" in python_results
    assert "Python machine learning" in python_results
    assert "Python web scraping" in python_results
    assert "Python automation scripts" in python_results

    # Search for JavaScript
    print("\n" + "=" * 70)
    print("🔍 BONUS: Searching for 'JavaScript' memories")
    print("=" * 70)

    js_results = await search_tool("JavaScript")

    print(f"\n✅ Found {len(js_results)} memorie(s) containing 'JavaScript':")
    for i, mem in enumerate(js_results, 1):
        print(f"  {i}. {mem}")

    assert len(js_results) == 1
    assert "JavaScript fundamentals" in js_results

    print("\n" + "=" * 70)
    print("🎉 DEMO COMPLETE - All operations successful!")
    print("=" * 70)
    print(f'''
Summary:
  • Added: {len(memories)} memories
  • Retrieved: {len(all_memories)} memories via get_all()
  • Found: {len(python_results)} Python-related memories
  • Found: {len(js_results)} JavaScript-related memories
    ''')
