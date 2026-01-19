"""Demonstration test: Add 10 messages, get_all, and search."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_demo_workflow(
    add_tool, get_all_tool, search_tool, mcp_server, create_search_results
):
    """Demonstration: Add 10 messages, retrieve all, and search."""

    # Prepare 10 messages
    messages = [
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
    print("📝 STEP 1: Adding 10 messages to memory store")
    print("=" * 70)
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. {msg}")

    # Setup mock behavior for add
    stored_messages = []

    def add_side_effect(*, messages=None, **_kwargs):
        messages = messages or []
        stored_messages.extend(messages)
        print(f"\n✅ Successfully added {len(messages)} message(s) to store")
        return messages

    mcp_server.memory_manager.memory.add_batch.side_effect = add_side_effect

    # Add messages
    await add_tool(messages)

    print(f"\n📊 Total messages in store: {len(stored_messages)}")

    # Setup mock behavior for get_all
    def get_all_side_effect(*_args, **_kwargs):
        return stored_messages.copy()

    mcp_server.memory_manager.memory.get_all.side_effect = get_all_side_effect

    print("\n" + "=" * 70)
    print("📥 STEP 2: Retrieving ALL messages with get_all()")
    print("=" * 70)

    # Get all messages
    all_messages = await get_all_tool()

    print(f"\n✅ Retrieved {len(all_messages)} message(s):")
    for i, msg in enumerate(all_messages, 1):
        print(f"  {i}. {msg}")

    # Verify all messages retrieved
    assert len(all_messages) == 10
    assert all_messages == messages

    # Setup mock behavior for search
    def search_side_effect(query, **kwargs):
        # Simulate semantic + substring search
        matching = [msg for msg in stored_messages if query.lower() in msg.lower()]
        return create_search_results(matching)

    mcp_server.memory_manager.memory.search.side_effect = search_side_effect

    print("\n" + "=" * 70)
    print("🔍 STEP 3: Searching for 'Python' messages")
    print("=" * 70)

    # Disable reranking/fusion for deterministic test results
    mcp_server.memory_manager.config.reranker_engine = "none"
    mcp_server.memory_manager.config.enable_rrf_fusion = False
    mcp_server.memory_manager.config.search_limit = len(messages)

    # Search for Python messages
    python_results = await search_tool("Python")

    print(f"\n✅ Found {len(python_results)} message(s) containing 'Python':")
    for i, msg in enumerate(python_results, 1):
        print(f"  {i}. {msg}")

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
    print("🔍 BONUS: Searching for 'JavaScript' messages")
    print("=" * 70)

    js_results = await search_tool("JavaScript")

    print(f"\n✅ Found {len(js_results)} message(s) containing 'JavaScript':")
    for i, msg in enumerate(js_results, 1):
        print(f"  {i}. {msg}")

    assert len(js_results) == 1
    assert "JavaScript fundamentals" in js_results

    print("\n" + "=" * 70)
    print("🎉 DEMO COMPLETE - All operations successful!")
    print("=" * 70)
    print(f"""
Summary:
  • Added: {len(messages)} messages
  • Retrieved: {len(all_messages)} messages via get_all()
  • Found: {len(python_results)} Python-related messages
  • Found: {len(js_results)} JavaScript-related messages
    """)
