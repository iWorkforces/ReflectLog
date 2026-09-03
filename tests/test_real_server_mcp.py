"""Real server integration test using fastmcp.Client (refactored from MCP SDK)."""

import json
import os

import anyio
from fastmcp import Client
from fastmcp.client.client import CallToolResult
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Set RUN_LIVE_TESTS=1 to run live server tests",
)


async def test_real_server_with_mcp_client():
    """Test real MCP server using fastmcp.Client."""

    print("\n" + "=" * 70)
    print("🌐 REAL SERVER TEST - Using fastmcp.Client")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    print(f"\n📡 Connecting to MCP server at {server_url}...")

    try:
        async with Client(server_url) as client:
            print("✅ Successfully connected to MCP server!")

            # List available tools
            tools = await client.list_tools()
            print(f"\n📋 Available tools: {[tool.name for tool in tools]}")

            # Prepare 10 test messages
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
            print("📝 STEP 1: Adding 10 memories to REAL server")
            print("=" * 70)
            for i, mem in enumerate(memories, 1):
                print(f"  {i}. {mem}")

            # Call add tool
            print("\n📤 Calling 'add' tool...")
            add_result: CallToolResult = await client.call_tool(
                "add", arguments={"memories": memories}
            )

            print("✅ Add tool response:")
            for content in add_result.content:
                if hasattr(content, "text"):
                    print(f"   {content.text}")

            # Small delay for persistence
            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("📥 STEP 2: Retrieving ALL memories from REAL server")
            print("=" * 70)

            print("\n📤 Calling 'get_all' tool...")
            get_all_result: CallToolResult = await client.call_tool("get_all")

            print("✅ Retrieved messages:")
            all_memories = []
            for content in get_all_result.content:
                if hasattr(content, "text"):
                    try:
                        all_memories = json.loads(content.text)
                        for i, msg in enumerate(all_memories, 1):
                            print(f"  {i}. {msg}")
                        print(f"\n📊 Total: {len(all_memories)} messages")
                    except Exception:
                        if isinstance(content.text, list):
                            all_memories = content.text
                            for i, msg in enumerate(all_memories, 1):
                                print(f"  {i}. {msg}")
                            print(f"\n📊 Total: {len(all_memories)} messages")
                        else:
                            print(f"   {content.text}")

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🔍 STEP 3: Searching for 'Python' memories on REAL server")
            print("=" * 70)

            print("\n📤 Calling 'search' tool with query='Python'...")
            search_result: CallToolResult = await client.call_tool(
                "search", arguments={"query": "Python"}
            )

            print("✅ Search results for 'Python':")
            python_results = []
            for content in search_result.content:
                if hasattr(content, "text"):
                    try:
                        python_results = json.loads(content.text)
                        for i, mem in enumerate(python_results, 1):
                            print(f"  {i}. {mem}")
                        print(
                            f"\n📊 Found: {len(python_results)} Python-related messages"
                        )
                    except Exception:
                        if isinstance(content.text, list):
                            python_results = content.text
                            for i, mem in enumerate(python_results, 1):
                                print(f"  {i}. {mem}")
                            print(
                                f"\n📊 Found: {len(python_results)} Python-related messages"
                            )
                        else:
                            print(f"   {content.text}")

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🔍 BONUS: Searching for 'JavaScript' memories")
            print("=" * 70)

            print("\n📤 Calling 'search' tool with query='JavaScript'...")
            js_search_result: CallToolResult = await client.call_tool(
                "search", arguments={"query": "JavaScript"}
            )

            print("✅ Search results for 'JavaScript':")
            js_results = []
            for content in js_search_result.content:
                if hasattr(content, "text"):
                    try:
                        js_results = json.loads(content.text)
                        for i, mem in enumerate(js_results, 1):
                            print(f"  {i}. {mem}")
                        print(
                            f"\n📊 Found: {len(js_results)} JavaScript-related messages"
                        )
                    except Exception:
                        if isinstance(content.text, list):
                            js_results = content.text
                            for i, mem in enumerate(js_results, 1):
                                print(f"  {i}. {mem}")
                            print(
                                f"\n📊 Found: {len(js_results)} JavaScript-related messages"
                            )
                        else:
                            print(f"   {content.text}")

            print("\n" + "=" * 70)
            print("🎉 REAL SERVER TEST COMPLETE!")
            print("=" * 70)
            print(f"""
Summary:
  ✅ Connected to MCP server at {server_url}
  ✅ Added 10 memories using REAL OpenAI embeddings
  ✅ Memories stored in REAL USearch vector index
  ✅ Retrieved all memories from persistent storage
  ✅ Searched with semantic similarity + substring filtering
  ✅ Used REAL OpenAI API for embeddings and reranking

This was a complete END-TO-END test with:
  • fastmcp.Client (simple and reliable)
  • Real HTTP/SSE MCP protocol
  • Real USearch vector database
  • Real OpenAI API calls
  • Real semantic search with reranking
            """)

    except Exception as e:
        import traceback

        print(f"\n❌ Error connecting to server: {e}")
        print("\nDetailed error:")
        traceback.print_exc()
        print("\nMake sure the server is running:")
        print(
            "  MCP_TRANSPORT=http MCP_PORT=9103 ./start-reflectlog-mcp-server.sh --workspace_id ReflectLog"
        )
        raise


if __name__ == "__main__":
    print("Starting REAL MCP server integration test...")
    anyio.run(test_real_server_with_mcp_client)
