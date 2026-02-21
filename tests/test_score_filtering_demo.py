#!/usr/bin/env python3
"""Test to demonstrate score filtering with 5 relevant documents where only 2 pass threshold."""

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


async def test_score_filtering_demo():
    """Test that demonstrates score filtering behavior."""

    print("\n" + "=" * 70)
    print("🎯 SCORE FILTERING DEMONSTRATION TEST")
    print("=" * 70)
    print("   Scenario: 5 semantically relevant documents")
    print("   Expected: Only 2-3 documents pass score > 0.8 threshold")
    print("   Server: http://127.0.0.1:9103/mcp")
    print("=" * 70)

    server_url = "http://127.0.0.1:9103/mcp"

    try:
        async with Client(server_url) as client:
            print("\n✅ Successfully connected to MCP server!")

            # Create memories with varying levels of relevance to "Python machine learning"
            # These are designed to have different semantic similarity scores
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
            print("📝 STEP 1: Adding test memories to server")
            print("=" * 70)
            for i, mem in enumerate(memories, 1):
                print(f"  {i}. {mem[:80]}{'...' if len(mem) > 80 else ''}")

            print("\n📤 Calling 'add' tool...")
            _: CallToolResult = await client.call_tool(
                "add", arguments={"memories": memories}
            )
            print("✅ Memories added successfully!")

            # Small delay for persistence
            await anyio.sleep(2)

            print("\n" + "=" * 70)
            print("🔍 STEP 2: Searching for 'Python machine learning'")
            print("=" * 70)
            print("   This query should match multiple documents semantically")
            print("   But only 2-3 should score > 0.8 threshold")
            print("=" * 70)

            search_query = "Python machine learning"
            print(f"\n📤 Calling 'search' tool with query='{search_query}'...")
            search_result: CallToolResult = await client.call_tool(
                "search", arguments={"query": search_query}
            )

            print("\n✅ Search completed!")
            search_results = []
            for content in search_result.content:
                if hasattr(content, "text"):
                    try:
                        search_results = json.loads(content.text)
                    except Exception:
                        if isinstance(content.text, list):
                            search_results = content.text

            print("\n📊 Results after score filtering:")
            if search_results:
                for i, mem in enumerate(search_results, 1):
                    print(f"  {i}. {mem[:80]}{'...' if len(mem) > 80 else ''}")
                print(
                    f"\n✅ {len(search_results)} document(s) passed the score threshold"
                )
            else:
                print("  No documents passed the score threshold")

            # Let's also test with a different query
            print("\n" + "=" * 70)
            print("🔍 STEP 3: Searching for 'Python programming'")
            print("=" * 70)

            search_query2 = "Python programming"
            print(f"\n📤 Calling 'search' tool with query='{search_query2}'...")
            search_result2: CallToolResult = await client.call_tool(
                "search", arguments={"query": search_query2}
            )

            search_results2 = []
            for content in search_result2.content:
                if hasattr(content, "text"):
                    try:
                        search_results2 = json.loads(content.text)
                    except Exception:
                        if isinstance(content.text, list):
                            search_results2 = content.text

            print("\n📊 Results after score filtering:")
            if search_results2:
                for i, mem in enumerate(search_results2, 1):
                    print(f"  {i}. {mem[:80]}{'...' if len(mem) > 80 else ''}")
                print(
                    f"\n✅ {len(search_results2)} document(s) passed the score threshold"
                )
            else:
                print("  No documents passed the score threshold")

            print("\n" + "=" * 70)
            print("🎉 SCORE FILTERING DEMONSTRATION COMPLETE!")
            print("=" * 70)
            print(f"""
Summary:
  ✅ Added {len(memories)} test messages with varying relevance
  ✅ Search for 'Python machine learning': {len(search_results)} results passed threshold
  ✅ Search for 'Python programming': {len(search_results2)} results passed threshold

Key Insights:
  • The system filters out semantically related but low-confidence matches
  • Only high-scoring documents (> 0.8) are returned
  • This prevents false positives and ensures high precision
  • Check server logs to see the actual scores and filtering decisions
            """)

            # Clean up - remove test memories
            print("\n🧹 Cleaning up test data...")
            for mem in memories:
                try:
                    await client.call_tool("remove", arguments={"memories": [mem]})
                except Exception:
                    pass  # Ignore errors during cleanup
            print("✅ Cleanup complete")

    except Exception as e:
        import traceback

        print(f"\n❌ Error during test: {e}")
        print("\nDetailed error:")
        traceback.print_exc()
        print("\nMake sure the server is running:")
        print("  ./start-reflectlog-mcp-server.sh --project_id ReflectLogMCP")
        raise


if __name__ == "__main__":
    print("Starting Score Filtering Demonstration Test...")
    anyio.run(test_score_filtering_demo)
