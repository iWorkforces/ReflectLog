'''Test to demonstrate verbose logging in the MCP server.'''

import os

import anyio
from fastmcp import Client
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Set RUN_LIVE_TESTS=1 to run live server tests",
)


async def test_verbose_logging():
    '''Test with a small dataset to see verbose logging.'''

    print("\n" + "=" * 70)
    print("Testing Verbose Logging")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    async with Client(server_url) as client:
        print("\n1️⃣  Testing ADD (will log each memory being added)")
        print("-" * 70)

        memories = [
            "Test memory 1",
            "Test memory 2",
            "Test memory 3",
        ]

        await client.call_tool("add", arguments={"memories": memories})
        print("✓ Add completed - check server logs for detailed per-memory logging")

        await anyio.sleep(1)

        print("\n2️⃣  Testing GET_ALL (will log each memory retrieved)")
        print("-" * 70)

        await client.call_tool("get_all")
        print("✓ Get all completed - check server logs for each memory retrieved")

        await anyio.sleep(1)

        print("\n3️⃣  Testing SEARCH (will log semantic candidates and filtering)")
        print("-" * 70)

        await client.call_tool("search", arguments={"query": "Test"})
        print("✓ Search completed - check server logs for:")
        print("  - Semantic search candidates")
        print("  - Substring filtering details")
        print("  - Match/no-match for each candidate")

        await anyio.sleep(1)

        print("\n4️⃣  Testing REMOVE (will log search, matching, and deletion)")
        print("-" * 70)

        await client.call_tool("remove", arguments={"memories": ["Test memory 2"]})
        print("✓ Remove completed - check server logs for:")
        print("  - Semantic search for candidates")
        print("  - Each candidate found")
        print("  - Exact match filtering")
        print("  - Deletion of matched items")

        print("\n" + "=" * 70)
        print("✓ All tests completed!")
        print("=" * 70)
        print(
            "\nCheck the server logs to see detailed verbose logging for each operation."
        )
        print("The server logs every:")
        print("  • Memory being added with progress [1/3], [2/3], etc.")
        print("  • Memory retrieved with full content preview")
        print("  • Search candidate found and filtering result")
        print("  • Deletion operation with memory ID and status")


if __name__ == "__main__":
    anyio.run(test_verbose_logging)
