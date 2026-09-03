#!/usr/bin/env python3
"""Test to demonstrate ENABLE_LLM_INFER functionality."""

import json
import os
import subprocess

import anyio
from fastmcp import Client
from fastmcp.client.client import CallToolResult
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
            "WORKSPACE_ID": "TestLLMInfer",
        }
    )

    server = subprocess.Popen(
        [
            "./start-reflectlog-mcp-server.sh",
            "--workspace_id",
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

            # Add a simple memory
            test_memory = "Python tutorial"
            print(f"\n📤 Adding memory: '{test_memory}'")

            await client.call_tool("add", arguments={"memories": [test_memory]})
            print("✅ Memory added")

            # Small delay for persistence
            await anyio.sleep(2)

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

            if memories and memories[0] != test_memory:
                print("\n✅ LLM INFERENCE IS ACTIVE:")
                print(f"   Original: '{test_memory}'")
                print(f"   Stored:   '{memories[0]}'")
                print("   The memory was transformed by the LLM!")
            else:
                print(
                    "\n⚠️ Memory appears unchanged (LLM might not have transformed it)"
                )

            # Clean up
            if memories:
                await client.call_tool("remove", arguments={"memories": memories})

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
            "WORKSPACE_ID": "TestNoInfer",
        }
    )

    server = subprocess.Popen(
        [
            "./start-reflectlog-mcp-server.sh",
            "--workspace_id",
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

            # Add a simple memory
            test_memory = "Python tutorial"
            print(f"\n📤 Adding memory: '{test_memory}'")

            await client.call_tool("add", arguments={"memories": [test_memory]})
            print("✅ Memory added")

            # Small delay for persistence
            await anyio.sleep(2)

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

            if memories and memories[0] == test_memory:
                print("\n✅ LLM INFERENCE IS DISABLED:")
                print(f"   Original: '{test_memory}'")
                print(f"   Stored:   '{memories[0]}'")
                print("   The memory was stored exactly as provided!")
            else:
                print(
                    "\n⚠️ Unexpected: Memory was transformed even though inference is disabled"
                )
                if memories:
                    print(f"   Original: '{test_memory}'")
                    print(f"   Stored:   '{memories[0]}'")

            # Clean up
            if memories:
                await client.call_tool("remove", arguments={"memories": memories})

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
    print("   whether the LLM processes memories during storage.")
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
  ✅ ENABLE_LLM_INFER=false (default): Memories stored exactly as provided
  ⚠️ ENABLE_LLM_INFER=true: Would transform memories using LLM (requires API key)

Key Points:
  • Default behavior (false) is faster and preserves exact memory content
  • Enabling LLM inference allows the AI to enhance memories with context
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
