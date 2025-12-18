"""Test to demonstrate verbose logging in the MCP server."""

from fastmcp import Client
import anyio


async def test_verbose_logging():
    """Test with a small dataset to see verbose logging."""

    print("\n" + "=" * 70)
    print("Testing Verbose Logging")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    async with Client(server_url) as client:
        print("\n1️⃣  Testing ADD (will log each message being added)")
        print("-" * 70)

        messages = [
            "Test message 1",
            "Test message 2",
            "Test message 3",
        ]

        await client.call_tool("add", arguments={"messages": messages})
        print("✓ Add completed - check server logs for detailed per-message logging")

        await anyio.sleep(1)

        print("\n2️⃣  Testing GET_ALL (will log each message retrieved)")
        print("-" * 70)

        await client.call_tool("get_all")
        print("✓ Get all completed - check server logs for each message retrieved")

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

        await client.call_tool("remove", arguments={"messages": ["Test message 2"]})
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
        print("  • Message being added with progress [1/3], [2/3], etc.")
        print("  • Message retrieved with full content preview")
        print("  • Search candidate found and filtering result")
        print("  • Deletion operation with memory ID and status")


if __name__ == "__main__":
    anyio.run(test_verbose_logging)
