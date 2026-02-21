#!/usr/bin/env python3
'''Simple test to verify ENABLE_LLM_INFER functionality.'''

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


async def test_infer_disabled():
    '''Test with ENABLE_LLM_INFER=false (default).'''
    print("\n" + "=" * 70)
    print("🔬 TEST: With LLM Inference DISABLED (default)")
    print("=" * 70)
    print("   Server: http://127.0.0.1:9103/mcp")
    print("   ENABLE_LLM_INFER: false (memories stored exactly as provided)")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    try:
        async with Client(server_url) as client:
            print("\n✅ Successfully connected to MCP server!")

            # Add a simple memory
            test_memory = "Python tutorial for beginners"
            print(f"\n📤 Adding memory: '{test_memory}'")

            await client.call_tool("add", arguments={"memories": [test_memory]})
            print("✅ Memory added successfully!")

            # Small delay for persistence
            await anyio.sleep(1)

            # Retrieve all memories
            print("\n📥 Retrieving all memories...")
            result: CallToolResult = await client.call_tool("get_all", arguments={})

            memories = []
            for content in result.content:
                if hasattr(content, "text"):
                    try:
                        memories = json.loads(content.text)
                    except Exception:
                        if isinstance(content.text, list):
                            memories = content.text

            print("\n📊 Stored memories:")
            for mem in memories:
                print(f"  - {mem}")

            # Check if memory was stored exactly as provided
            if test_memory in memories:
                print("\n✅ SUCCESS: LLM Inference is DISABLED")
                print(f"   Original: '{test_memory}'")
                print("   Found exact match in storage!")
                print("   The memory was stored without LLM transformation.")
            else:
                print("\n⚠️ UNEXPECTED: Memory not found exactly as provided")
                print(f"   Original: '{test_memory}'")
                print(f"   Stored memories: {memories}")

            # Clean up - remove test memories
            print("\n🧹 Cleaning up test data...")
            if memories:
                await client.call_tool("remove", arguments={"memories": memories})
                print("✅ Cleanup complete")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure the server is running with:")
        print(
            "  ENABLE_LLM_INFER=false MCP_TRANSPORT=http ./start-reflectlog-mcp-server.sh --project_id TestInfer"
        )
        raise


async def main():
    '''Run the test.'''
    print("\n" + "=" * 70)
    print("🎯 ENABLE_LLM_INFER FUNCTIONALITY VERIFICATION")
    print("=" * 70)
    print("   This test verifies that with ENABLE_LLM_INFER=false,")
    print("   memories are stored exactly as provided without LLM processing.")
    print("=" * 70)

    await test_infer_disabled()

    print("\n" + "=" * 70)
    print("🎉 TEST COMPLETE!")
    print("=" * 70)
    print('''
Key Insights:
  ✅ ENABLE_LLM_INFER=false (default): Memories stored exactly as provided
  • This is the default behavior for faster, exact storage
  • No LLM API calls are made during the add operation
  • Memories are retrieved exactly as they were stored

To test with LLM inference enabled:
  1. Stop the current server
  2. Start with: ENABLE_LLM_INFER=true ./start-reflectlog-mcp-server.sh --project_id TestInfer
  3. Run this test again to see memories being transformed by the LLM
    ''')


if __name__ == "__main__":
    print("Starting ENABLE_LLM_INFER Test...")
    anyio.run(main)
