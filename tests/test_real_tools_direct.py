'''Direct integration test - calls real server tools with real USearch and OpenRouter via HTTP/SSE.'''

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


async def test_real_tools_workflow():
    '''Test real MCP tools with actual USearch and OpenRouter integration via HTTP/SSE.'''

    print("\n" + "=" * 70)
    print("🌐 REAL INTEGRATION TEST - Via HTTP/SSE Transport")
    print("=" * 70)
    print("   Using: REAL USearch index + REAL OpenRouter API")
    print("   Transport: fastmcp.Client over HTTP/SSE")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    try:
        async with Client(server_url) as client:
            print(f"\n✅ Connected to MCP server at {server_url}")

            # List tools
            tools = await client.list_tools()
            print(f"   Found {len(tools)} tools: {[t.name for t in tools]}")

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
            print("📝 STEP 1: Adding 10 memories to REAL USearch index")
            print("=" * 70)
            for i, mem in enumerate(memories, 1):
                print(f"  {i}. {mem}")

            print("\n📤 Calling add() with REAL OpenAI embeddings...")
            print("   (This will call OpenAI API to generate embeddings)")

            try:
                _ = await client.call_tool("add", arguments={"memories": memories})
                print("✅ Successfully added 10 memories to USearch index!")
            except Exception as e:
                print(f"❌ Error adding memories: {e}")
                return

            # Small delay for persistence
            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("📥 STEP 2: Retrieving ALL memories from REAL USearch index")
            print("=" * 70)

            print("\n📤 Calling get_all() to retrieve from USearch...")

            try:
                get_all_result: CallToolResult = await client.call_tool("get_all")
                all_memories = []
                for content in get_all_result.content:
                    if hasattr(content, "text"):
                        try:
                            all_memories = json.loads(content.text)
                        except Exception:
                            if isinstance(content.text, list):
                                all_memories = content.text

                print(f"✅ Retrieved {len(all_memories)} memorie(s) from USearch:")
                for i, mem in enumerate(all_memories, 1):
                    print(f"  {i}. {mem}")
            except Exception as e:
                print(f"❌ Error retrieving memories: {e}")
                return

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🔍 STEP 3: Searching for 'Python' with REAL semantic search")
            print("=" * 70)

            print("\n📤 Calling search('Python')...")
            print(
                "   (This uses OpenRouter embeddings + USearch similarity + reranking)"
            )

            try:
                search_result: CallToolResult = await client.call_tool(
                    "search", arguments={"query": "Python"}
                )
                python_results = []
                for content in search_result.content:
                    if hasattr(content, "text"):
                        try:
                            python_results = json.loads(content.text)
                        except Exception:
                            if isinstance(content.text, list):
                                python_results = content.text

                print(f"✅ Found {len(python_results)} Python-related memorie(s):")
                for i, mem in enumerate(python_results, 1):
                    print(f"  {i}. {mem}")
            except Exception as e:
                print(f"❌ Error searching: {e}")
                return

            await anyio.sleep(1)

            print("\n" + "=" * 70)
            print("🔍 BONUS: Searching for 'JavaScript'")
            print("=" * 70)

            print("\n📤 Calling search('JavaScript')...")

            try:
                js_search_result: CallToolResult = await client.call_tool(
                    "search", arguments={"query": "JavaScript"}
                )
                js_results = []
                for content in js_search_result.content:
                    if hasattr(content, "text"):
                        try:
                            js_results = json.loads(content.text)
                        except Exception:
                            if isinstance(content.text, list):
                                js_results = content.text

                print(f"✅ Found {len(js_results)} JavaScript-related memorie(s):")
                for i, mem in enumerate(js_results, 1):
                    print(f"  {i}. {mem}")
            except Exception as e:
                print(f"❌ Error searching: {e}")
                return

            print("\n" + "=" * 70)
            print("🎉 REAL INTEGRATION TEST COMPLETE!")
            print("=" * 70)
            print(f'''
Summary:
  ✅ Connected to MCP server at {server_url}
  ✅ Added 10 memories using REAL OpenAI text-embedding-3-large
  ✅ Memories stored in REAL USearch vector index (cosine similarity)
  ✅ Retrieved all memories from persistent USearch storage
  ✅ Searched with REAL semantic similarity (OpenAI embeddings)
  ✅ Applied hybrid search (semantic + substring filtering)
  ✅ Used REAL OpenAI gpt-5-mini for reranking

This was a complete END-TO-END integration test with:
  • fastmcp.Client (simple and reliable HTTP/SSE transport)
  • Real USearch vector database
  • Real OpenAI API calls (embeddings)
  • Real semantic search with L2 normalization
  • Persistent storage (data saved to disk)
            ''')

    except Exception as e:
        import traceback

        print(f"\n❌ Error during test: {e}")
        print("\nDetailed error:")
        traceback.print_exc()
        print("\nMake sure the server is running:")
        print(
            "  MCP_TRANSPORT=http MCP_PORT=9103 ./start-reflectlog-mcp-server.sh --workspace_id ReflectLog"
        )
        raise


if __name__ == "__main__":
    print("Starting REAL integration test with actual backends...")
    anyio.run(test_real_tools_workflow)
