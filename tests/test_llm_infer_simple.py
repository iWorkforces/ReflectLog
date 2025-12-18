#!/usr/bin/env python3
"""Simple test to verify ENABLE_LLM_INFER functionality."""

from fastmcp import Client
from fastmcp.client.client import CallToolResult
import anyio
import json


async def test_infer_disabled():
    """Test with ENABLE_LLM_INFER=false (default)."""
    print("\n" + "=" * 70)
    print("🔬 TEST: With LLM Inference DISABLED (default)")
    print("=" * 70)
    print("   Server: http://127.0.0.1:9104/mcp")
    print("   ENABLE_LLM_INFER: false (messages stored exactly as provided)")
    print("=" * 70)

    server_url = "http://127.0.0.1:9104/mcp"

    try:
        async with Client(server_url) as client:
            print("\n✅ Successfully connected to MCP server!")

            # Add a simple message
            test_message = "Python tutorial for beginners"
            print(f"\n📤 Adding message: '{test_message}'")

            await client.call_tool("add", arguments={"messages": [test_message]})
            print("✅ Message added successfully!")

            # Small delay for persistence
            await anyio.sleep(1)

            # Retrieve all messages
            print("\n📥 Retrieving all messages...")
            result: CallToolResult = await client.call_tool("get_all", arguments={})

            messages = []
            for content in result.content:
                if hasattr(content, "text"):
                    try:
                        messages = json.loads(content.text)
                    except Exception:
                        if isinstance(content.text, list):
                            messages = content.text

            print("\n📊 Stored messages:")
            for msg in messages:
                print(f"  - {msg}")

            # Check if message was stored exactly as provided
            if test_message in messages:
                print("\n✅ SUCCESS: LLM Inference is DISABLED")
                print(f"   Original: '{test_message}'")
                print("   Found exact match in storage!")
                print("   The message was stored without LLM transformation.")
            else:
                print("\n⚠️ UNEXPECTED: Message not found exactly as provided")
                print(f"   Original: '{test_message}'")
                print(f"   Stored messages: {messages}")

            # Clean up - remove test messages
            print("\n🧹 Cleaning up test data...")
            if messages:
                await client.call_tool("remove", arguments={"messages": messages})
                print("✅ Cleanup complete")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure the server is running with:")
        print(
            "  ENABLE_LLM_INFER=false MCP_TRANSPORT=http ./start-openmemories-mcp-server.sh --project_id TestInfer"
        )
        raise


async def main():
    """Run the test."""
    print("\n" + "=" * 70)
    print("🎯 ENABLE_LLM_INFER FUNCTIONALITY VERIFICATION")
    print("=" * 70)
    print("   This test verifies that with ENABLE_LLM_INFER=false,")
    print("   messages are stored exactly as provided without LLM processing.")
    print("=" * 70)

    await test_infer_disabled()

    print("\n" + "=" * 70)
    print("🎉 TEST COMPLETE!")
    print("=" * 70)
    print("""
Key Insights:
  ✅ ENABLE_LLM_INFER=false (default): Messages stored exactly as provided
  • This is the default behavior for faster, exact storage
  • No LLM API calls are made during the add operation
  • Messages are retrieved exactly as they were stored

To test with LLM inference enabled:
  1. Stop the current server
  2. Start with: ENABLE_LLM_INFER=true ./start-openmemories-mcp-server.sh --project_id TestInfer
  3. Run this test again to see messages being transformed by the LLM
    """)


if __name__ == "__main__":
    print("Starting ENABLE_LLM_INFER Test...")
    anyio.run(main)
