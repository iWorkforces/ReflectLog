"""Search tool implementation for CCMemoriesMCP Server."""

import time
from typing import Annotated, List

from pydantic import Field

from ..exceptions import SearchError
from ..utils import truncate_message
from .base import BaseTool


class SearchTool(BaseTool):
    """Tool for semantic search in memory storage."""

    def get_name(self) -> str:
        """Get the tool name."""
        return "search"

    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • search(query: str) -> list[str]\n"
            "      Hybrid semantic + full-text search. Finds semantically similar\n"
            "      messages using vector embeddings (limit: configurable, default 5)."
        )

    def get_handler(self):
        """Get the async tool handler function."""

        async def search(
            query: Annotated[
                str,
                Field(
                    min_length=1,
                    description="Search query for semantic matching",
                ),
            ],
        ) -> List[str]:
            """Search for messages using semantic matching (async).

            This tool performs hybrid semantic + full-text search to find messages
            conceptually similar to the query:
            1. Hybrid search: Combines semantic vector search (USearch) with
               full-text search (Tantivy) using RRF fusion.
            2. Score filtering: Filters out results with fusion scores below
               the threshold.

            This approach returns conceptually related content based on both
            semantic similarity and keyword matching.

            Note: Calls async MemoryManager.search() directly, which internally
            runs parallel searches using asyncio.gather(). This enables concurrent
            tool calls without blocking the event loop.

            Args:
                query: Non-empty search query string (min_length=1 enforced by Pydantic).

            Returns:
                List of semantically similar messages ordered by relevance.
                Returns empty list [] if no matches found.

            Raises:
                ValueError: If query is empty string (enforced by Pydantic).
                SearchError: If search operation fails.

            Examples:
                >>> add(["Python tutorial", "JavaScript guide", "Programming basics"])
                >>> search("Python")
                ["Python tutorial", "Programming basics"]  # Semantically related
                >>> search("coding")
                ["Programming basics", "Python tutorial", "JavaScript guide"]
                >>> search("Ruby")
                []  # No semantically similar content
            """
            try:
                start_time = time.time()
                self.log_invocation("search", query=query)

                # Log search header
                self.logger.info(
                    "=" * 60,
                    extra={"tool": "search", "section": "header"},
                )
                self.logger.info(
                    "SEARCH OPERATION",
                    extra={"tool": "search"},
                )
                self.logger.info(
                    f'   Query: "{query}"',
                    extra={"tool": "search", "query": query},
                )
                self.logger.info(
                    f"   Settings: limit={self.config.search_limit}",
                    extra={
                        "tool": "search",
                        "limit": self.config.search_limit,
                    },
                )

                # Perform async hybrid search directly (search is now async)
                search_start = time.time()
                similar_messages = await self.memory.search(
                    query,
                    limit=self.config.search_limit,
                )
                search_duration = (time.time() - search_start) * 1000  # ms

                # Log final results
                if similar_messages:
                    self.logger.info(
                        "=" * 60,
                        extra={"tool": "search", "section": "results"},
                    )
                    self.logger.info(
                        f"FINAL RESULTS ({len(similar_messages)} message(s)):",
                        extra={
                            "tool": "search",
                            "query": query,
                            "result_count": len(similar_messages),
                        },
                    )

                    # Log each result with preview
                    for idx, message in enumerate(similar_messages, 1):
                        preview = truncate_message(message, max_length=70)
                        self.logger.info(
                            f"   [{idx}] {preview}",
                            extra={
                                "tool": "search",
                                "query": query,
                                "result_index": idx,
                                "total_results": len(similar_messages),
                            },
                        )
                else:
                    self.logger.info(
                        "=" * 60,
                        extra={"tool": "search", "section": "results"},
                    )
                    self.logger.info(
                        "FINAL RESULTS: No matching messages found",
                        extra={
                            "tool": "search",
                            "query": query,
                            "result_count": 0,
                        },
                    )

                # Log timing
                total_duration = (time.time() - start_time) * 1000  # ms
                self.logger.info(
                    f"Search completed in {search_duration:.0f}ms (total: {total_duration:.0f}ms)",
                    extra={
                        "tool": "search",
                        "search_duration_ms": search_duration,
                        "total_duration_ms": total_duration,
                    },
                )
                self.logger.info(
                    "=" * 60,
                    extra={"tool": "search", "section": "footer"},
                )

                self.log_completion(
                    "search", query=query, result_count=len(similar_messages)
                )

                return similar_messages

            except Exception as e:
                self.log_error("search", e, query=query)
                raise SearchError(f"Failed to search memory store: {e}") from e

        return search
