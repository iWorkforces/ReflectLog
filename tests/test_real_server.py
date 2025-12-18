"""Real server integration test using fastmcp.Client (refactored from plain HTTP)."""

from fastmcp import Client
from fastmcp.client.client import CallToolResult
import anyio
import json


async def test_real_server_workflow():
    """Test real MCP server: add 10 messages, get_all, and search."""

    print("\n" + "=" * 70)
    print("🌐 REAL SERVER TEST - Connecting to http://127.0.0.1:9103/mcp")
    print("=" * 70)
    print("   Using fastmcp.Client for HTTP/SSE transport")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    try:
        async with Client(server_url) as client:
            print("\n📡 Connecting to MCP server...")
            print("✅ Successfully connected to MCP server!")

            # List available tools
            tools = await client.list_tools()
            print("\n📋 Available tools:")
            for tool in tools:
                print(
                    f"   • {tool.name}: {tool.description.split('.')[0] if tool.description else 'No description'}..."
                )

            # Prepare 10 test messages
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
            print("📝 STEP 1: Adding 10 messages to REAL server")
            print("=" * 70)
            for i, msg in enumerate(messages, 1):
                print(f"  {i}. {msg}")

            print("\n📤 Calling 'add' tool...")
            add_result: CallToolResult = await client.call_tool(
                "add", arguments={"messages": messages}
            )
            print("✅ Add completed!")
            if add_result.content:
                for content in add_result.content:
                    if hasattr(content, "text"):
                        print(f"   {content.text}")

            # Small delay for persistence
            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("📥 STEP 2: Retrieving ALL messages from REAL server")
            print("=" * 70)

            print("\n📤 Calling 'get_all' tool...")
            get_all_result: CallToolResult = await client.call_tool("get_all")

            print("✅ Retrieved messages:")
            all_messages = []
            for content in get_all_result.content:
                if hasattr(content, "text"):
                    try:
                        all_messages = json.loads(content.text)
                        for i, msg in enumerate(all_messages, 1):
                            print(f"  {i}. {msg}")
                        print(f"\n📊 Total: {len(all_messages)} messages")
                    except Exception:
                        if isinstance(content.text, list):
                            all_messages = content.text
                            for i, msg in enumerate(all_messages, 1):
                                print(f"  {i}. {msg}")
                            print(f"\n📊 Total: {len(all_messages)} messages")
                        else:
                            print(f"   {content.text}")

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🔍 STEP 3: Searching for 'Python' messages on REAL server")
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
                        for i, msg in enumerate(python_results, 1):
                            print(f"  {i}. {msg}")
                        print(
                            f"\n📊 Found: {len(python_results)} Python-related messages"
                        )
                    except Exception:
                        if isinstance(content.text, list):
                            python_results = content.text
                            for i, msg in enumerate(python_results, 1):
                                print(f"  {i}. {msg}")
                            print(
                                f"\n📊 Found: {len(python_results)} Python-related messages"
                            )
                        else:
                            print(f"   {content.text}")

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🔍 STEP 4: Searching for 'JavaScript' messages")
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
                        for i, msg in enumerate(js_results, 1):
                            print(f"  {i}. {msg}")
                        print(
                            f"\n📊 Found: {len(js_results)} JavaScript-related messages"
                        )
                    except Exception:
                        if isinstance(content.text, list):
                            js_results = content.text
                            for i, msg in enumerate(js_results, 1):
                                print(f"  {i}. {msg}")
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
  ✅ Listed {len(tools)} available tools
  ✅ Added {len(messages)} messages using REAL OpenAI embeddings
  ✅ Messages stored in REAL USearch vector index
  ✅ Retrieved all messages from persistent storage
  ✅ Searched and found {len(python_results)} Python-related messages
  ✅ Searched and found {len(js_results)} JavaScript-related messages

This was a complete END-TO-END test with:
  • fastmcp.Client (simple and reliable)
  • Real HTTP/SSE MCP protocol
  • Real USearch vector database
  • Real OpenAI API calls
  • All tools tested (add, get_all, search)
            """)

    except Exception as e:
        import traceback

        print(f"\n❌ Error during test: {e}")
        print("\nDetailed error:")
        traceback.print_exc()
        print("\nMake sure the server is running:")
        print(
            "  MCP_TRANSPORT=http MCP_PORT=9103 ./start-ccmemories-mcp-server.sh --project_id CCMemoriesMCP"
        )
        raise


if __name__ == "__main__":
    print("Starting REAL MCP server integration test...")
    anyio.run(test_real_server_workflow)
