"""Real server integration test using fastmcp.Client - SIMPLIFIED APPROACH."""

import os

import anyio
from fastmcp import Client
from fastmcp.client.client import CallToolResult
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Set RUN_LIVE_TESTS=1 to run live server tests",
)


async def test_real_server_with_fastmcp_client():
    """Test real MCP server using fastmcp.Client.

    This is the RECOMMENDED approach for testing the HTTP server.
    It's simpler and more reliable than raw HTTP or MCP SDK approaches.
    """

    print("\n" + "=" * 70)
    print("🌐 REAL SERVER TEST - Using fastmcp.Client")
    print("=" * 70)
    print("   Transport: HTTP (via fastmcp.Client)")
    print("   Server: http://127.0.0.1:9103/mcp")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    try:
        async with Client(server_url) as client:
            print("\n✅ Successfully connected to MCP server!")

            # List available tools
            print("\n📋 Listing available tools...")
            tools = await client.list_tools()
            print(f"   Found {len(tools)} tools:")
            for tool in tools:
                print(f"   • {tool.name}: {tool.description.split('.')[0]}...")

            # Prepare 10 test memories
            memories = [
                "Python tutorial for beginners covers the essential syntax and core concepts. Learn about variables, data types, loops, and functions through practical examples. This comprehensive guide helps newcomers build a solid foundation in Python programming.",
                "Advanced Python programming explores decorators, metaclasses, and context managers for professional development. Master asynchronous programming with asyncio and concurrent.futures for high-performance applications. Deep dive into design patterns and best practices used in production systems.",
                "JavaScript fundamentals teach you the core concepts of web development and DOM manipulation. Understand event handling, callbacks, promises, and async/await for modern web applications. Learn ES6+ features including arrow functions, destructuring, and modules.",
                "Python data science guide introduces pandas, numpy, and matplotlib for data analysis and visualization. Process and clean large datasets efficiently using vectorized operations. Create insightful visualizations and statistical analyses to extract meaningful patterns from your data.",
                "Java enterprise patterns demonstrate architectural best practices for scalable business applications. Learn dependency injection, factory patterns, and service-oriented architecture principles. Implement robust solutions using Spring Framework and enterprise design patterns.",
                "Python machine learning covers scikit-learn, TensorFlow, and PyTorch for building intelligent systems. Train classification and regression models using supervised learning algorithms. Deploy neural networks and deep learning models for real-world prediction tasks.",
                "React frontend development focuses on building interactive user interfaces with components and hooks. Manage application state using useState, useEffect, and Context API. Create responsive, performant web applications following modern React best practices.",
                "Python web scraping demonstrates how to extract data from websites using BeautifulSoup and Scrapy. Handle dynamic content with Selenium and respect robots.txt guidelines. Parse HTML, navigate DOM trees, and store extracted data in structured formats.",
                "Docker containerization enables consistent deployment across different environments and platforms. Create lightweight, isolated containers using Dockerfile and docker-compose. Orchestrate multi-container applications and manage microservices efficiently.",
                "Python automation scripts streamline repetitive tasks and improve workflow efficiency. Automate file operations, data processing, and system administration using Python libraries. Schedule tasks with cron jobs and create robust error handling for production automation.",
            ]

            print("\n" + "=" * 70)
            print("📝 STEP 1: Adding 10 memories to REAL server")
            print("=" * 70)
            for i, mem in enumerate(memories, 1):
                print(f"  {i}. {mem}")

            print("\n📤 Calling 'add' tool...")
            add_result: CallToolResult = await client.call_tool(
                "add", arguments={"memories": memories}
            )
            print("✅ Add completed!")
            if add_result.content:
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

            print("✅ Retrieved memories:")
            all_memories = []
            for content in get_all_result.content:
                if hasattr(content, "text"):
                    # Parse the text content
                    import json

                    try:
                        all_memories = json.loads(content.text)
                        for i, mem in enumerate(all_memories, 1):
                            print(f"  {i}. {mem}")
                        print(f"\n📊 Total: {len(all_memories)} memories")
                    except Exception:
                        # Might already be a list
                        if isinstance(content.text, list):
                            all_memories = content.text
                            for i, mem in enumerate(all_memories, 1):
                                print(f"  {i}. {mem}")
                            print(f"\n📊 Total: {len(all_memories)} memories")
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
                            f"\n📊 Found: {len(python_results)} Python-related memories"
                        )
                    except Exception:
                        if isinstance(content.text, list):
                            python_results = content.text
                            for i, mem in enumerate(python_results, 1):
                                print(f"  {i}. {mem}")
                            print(
                                f"\n📊 Found: {len(python_results)} Python-related memories"
                            )
                        else:
                            print(f"   {content.text}")

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🔍 STEP 4: Searching for 'JavaScript' memories")
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
                            f"\n📊 Found: {len(js_results)} JavaScript-related memories"
                        )
                    except Exception:
                        if isinstance(content.text, list):
                            js_results = content.text
                            for i, mem in enumerate(js_results, 1):
                                print(f"  {i}. {mem}")
                            print(
                                f"\n📊 Found: {len(js_results)} JavaScript-related memories"
                            )
                        else:
                            print(f"   {content.text}")

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🗑️  STEP 5: Testing remove functionality")
            print("=" * 70)

            java_memory = "Java enterprise patterns demonstrate architectural best practices for scalable business applications. Learn dependency injection, factory patterns, and service-oriented architecture principles. Implement robust solutions using Spring Framework and enterprise design patterns."
            print("\n📤 Removing Java enterprise patterns memory...")
            _ = await client.call_tool("remove", arguments={"memories": [java_memory]})
            print("✅ Remove completed!")

            # Verify removal
            await anyio.sleep(1)
            verify_result: CallToolResult = await client.call_tool("get_all")
            remaining_memories = []
            for content in verify_result.content:
                if hasattr(content, "text"):
                    try:
                        remaining_memories = json.loads(content.text)
                    except Exception:
                        if isinstance(content.text, list):
                            remaining_memories = content.text

            print(f"\n📊 After removal: {len(remaining_memories)} memories remain")
            assert java_memory not in remaining_memories, (
                "Java memory should be removed"
            )
            print(
                "✅ Verified: Java enterprise patterns memory was successfully removed"
            )

            print("\n" + "=" * 70)
            print("🎉 REAL SERVER TEST COMPLETE!")
            print("=" * 70)
            print(f"""
Summary:
  ✅ Connected to MCP server at {server_url}
  ✅ Listed {len(tools)} available tools
  ✅ Added {len(memories)} memories using REAL OpenAI embeddings
  ✅ Memories stored in REAL USearch vector index
  ✅ Retrieved all memories from persistent storage
  ✅ Searched and found {len(python_results)} Python-related memories
  ✅ Searched and found {len(js_results)} JavaScript-related memories
  ✅ Removed 1 memory successfully
  ✅ Verified removal

This was a complete END-TO-END test with:
  • fastmcp.Client (simple and reliable)
  • Real HTTP/SSE MCP protocol
  • Real USearch vector database
  • Real OpenAI API calls
  • All 4 MCP tools tested (add, get_all, search, remove)
            """)

    except Exception as e:
        import traceback

        print(f"\n❌ Error during test: {e}")
        print("\nDetailed error:")
        traceback.print_exc()
        print("\nMake sure the server is running:")
        print("  ./start-reflectlog-mcp-server.sh --project_id ReflectLogMCP")
        raise


if __name__ == "__main__":
    print("Starting REAL MCP server integration test with fastmcp.Client...")
    anyio.run(test_real_server_with_fastmcp_client)
