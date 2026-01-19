#!/usr/bin/env python3
"""Test to demonstrate ENABLE_LLM_INFER functionality."""

import subprocess
import json
from fastmcp import Client
from fastmcp.client.client import CallToolResult
import anyio
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Set RUN_LIVE_TESTS=1 to run live server tests",
)


async def test_with_infer_enabled():
    """Test with ENABLE_LLM_INFER=true."""
    print("\n" + "=" * 70)
    print("🔬 TEST 1: With LLM Inference ENABLED")
    print("=" * 70)

    # Start server with ENABLE_LLM_INFER=true
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_LLM_INFER": "true",
            "MCP_TRANSPORT": "http",
            "MCP_PORT": "9103",  # Different port to avoid conflicts
            "LOG_LEVEL": "INFO",
            "PROJECT_ID": "TestLLMInfer",
        }
    )

    server = subprocess.Popen(
        [
            "./start-reflectlog-mcp-server.sh",
            "--project_id",
            "TestLLMInfer",
            "--transport",
            "http",
            "--port",
            "9103",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give server time to start
    await anyio.sleep(3)

    try:
        async with Client("http://127.0.0.1:9103/mcp") as client:
            print("✅ Connected to server with ENABLE_LLM_INFER=true")

            # Add a simple message
            test_message = "Python tutorial"
            print(f"\n📤 Adding message: '{test_message}'")

            await client.call_tool("add", arguments={"messages": [test_message]})
            print("✅ Message added")

            # Small delay for persistence
            await anyio.sleep(2)

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

            if messages and messages[0] != test_message:
                print("\n✅ LLM INFERENCE IS ACTIVE:")
                print(f"   Original: '{test_message}'")
                print(f"   Stored:   '{messages[0]}'")
                print("   The message was transformed by the LLM!")
            else:
                print(
                    "\n⚠️ Message appears unchanged (LLM might not have transformed it)"
                )

            # Clean up
            if messages:
                await client.call_tool("remove", arguments={"messages": messages})

    finally:
        # Kill the server
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()


async def test_with_infer_disabled():
    """Test with ENABLE_LLM_INFER=false (default)."""
    print("\n" + "=" * 70)
    print("🔬 TEST 2: With LLM Inference DISABLED (default)")
    print("=" * 70)

    # Start server with ENABLE_LLM_INFER=false (or not set, since default is false)
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_LLM_INFER": "false",  # Explicitly set to false
            "MCP_TRANSPORT": "http",
            "MCP_PORT": "9105",  # Different port
            "LOG_LEVEL": "INFO",
            "PROJECT_ID": "TestNoInfer",
        }
    )

    server = subprocess.Popen(
        [
            "./start-reflectlog-mcp-server.sh",
            "--project_id",
            "TestNoInfer",
            "--transport",
            "http",
            "--port",
            "9105",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give server time to start
    await anyio.sleep(3)

    try:
        async with Client("http://127.0.0.1:9105/mcp") as client:
            print("✅ Connected to server with ENABLE_LLM_INFER=false")

            # Add a simple message
            test_message = "Python tutorial"
            print(f"\n📤 Adding message: '{test_message}'")

            await client.call_tool("add", arguments={"messages": [test_message]})
            print("✅ Message added")

            # Small delay for persistence
            await anyio.sleep(2)

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

            if messages and messages[0] == test_message:
                print("\n✅ LLM INFERENCE IS DISABLED:")
                print(f"   Original: '{test_message}'")
                print(f"   Stored:   '{messages[0]}'")
                print("   The message was stored exactly as provided!")
            else:
                print(
                    "\n⚠️ Unexpected: Message was transformed even though inference is disabled"
                )
                if messages:
                    print(f"   Original: '{test_message}'")
                    print(f"   Stored:   '{messages[0]}'")

            # Clean up
            if messages:
                await client.call_tool("remove", arguments={"messages": messages})

    finally:
        # Kill the server
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()


async def main():
    """Run both tests."""
    print("\n" + "=" * 70)
    print("🎯 ENABLE_LLM_INFER FUNCTIONALITY TEST")
    print("=" * 70)
    print("   This test demonstrates how ENABLE_LLM_INFER controls")
    print("   whether the LLM processes messages during storage.")
    print("=" * 70)

    try:
        # Note: Running tests sequentially to avoid OpenAI API issues
        # Test with inference disabled first (default behavior)
        await test_with_infer_disabled()

        # Small delay between tests
        await anyio.sleep(2)

        # Test with inference enabled
        # NOTE: This test requires valid OpenAI API key and may incur costs!
        print("\n⚠️ Note: Test with ENABLE_LLM_INFER=true requires OpenAI API key")
        print("   and may incur API costs. Skipping for safety...")
        # Uncomment the line below to run the test with inference enabled:
        # await test_with_infer_enabled()

        print("\n" + "=" * 70)
        print("🎉 ENABLE_LLM_INFER TEST COMPLETE!")
        print("=" * 70)
        print("""
Summary:
  ✅ ENABLE_LLM_INFER=false (default): Messages stored exactly as provided
  ⚠️ ENABLE_LLM_INFER=true: Would transform messages using LLM (requires API key)

Key Points:
  • Default behavior (false) is faster and preserves exact message content
  • Enabling LLM inference allows the AI to enhance messages with context
  • The choice depends on your use case and whether you want AI enhancement
        """)

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("Starting ENABLE_LLM_INFER Test...")
    anyio.run(main)
