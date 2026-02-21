# https://github.com/jlowin/fastmcp?tab=readme-ov-file#mcp-clients
# https://github.com/StevenBtw/uv-docs-mcp/blob/main/src/uv_docs/server.py
import anyio
from fastmcp import Client
from fastmcp.client.client import CallToolResult


async def main():
    # Connect via stdio to a local script
    async with Client("http://127.0.0.1:9103/mcp") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        result: CallToolResult = await client.call_tool(
            "get_all",
        )
        print(f"Result: {result.content}")


if __name__ == "__main__":
    anyio.run(main)
