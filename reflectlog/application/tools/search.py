"""Search tool implementation for ReflectLog Server."""

from typing import Annotated, override

from pydantic import Field

from reflectlog.core.exceptions import SearchError

from .base import BaseTool


class SearchTool(BaseTool):
    """Tool for semantic search in memory storage."""

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return "search"

    @override
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • search(query: str) -> list[str]\n"
            "      Hybrid semantic + full-text search. Finds semantically similar\n"
            "      memories using vector embeddings (limit: configurable, default 5)."
        )

    @override
    def get_handler(self):
        """Get the async tool handler function."""

        async def search(
            query: Annotated[
                str,
                Field(
                    min_length=1,
                    max_length=1000,
                    description="Search query for semantic matching",
                ),
            ],
        ) -> list[str]:
            """Search for memories using semantic matching (async).

            This tool performs hybrid semantic + full-text search to find memories
            conceptually similar to the query:
            1. Hybrid search: Combines semantic vector search (USearch) with
               full-text search (Tantivy) using RRF fusion.
            2. Score filtering: Filters out results with fusion scores below
               the threshold.

            This approach returns conceptually related content based on both
            semantic similarity and keyword matching.

            Note: Calls async MemoryManager.search() directly. Backend USearch
            and Tantivy work runs on worker threads so concurrent MCP calls are
            not stalled by one slow search. Cancelling this tool call does not
            abort native work already running in a worker thread.

            Args:
                query: Non-empty search query string (min_length=1 enforced by Pydantic).

            Returns:
                List of semantically similar memories ordered by relevance.
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
                self.log_invocation("search", query_length=len(query))

                start_time = self._log_operation_header("search", "SEARCH OPERATION")
                if self.config.log_search_results_verbose:
                    self.logger.info(
                        f"   Query length: {len(query)}",
                        extra={"tool": "search", "query_length": len(query)},
                    )

                # Perform async hybrid search directly (search is now async)
                similar_memories = await self.memory.search(
                    query,
                    limit=self.config.search_limit,
                )

                if self.config.log_search_results_verbose:
                    if similar_memories:
                        self.logger.info(
                            f"FINAL RESULTS ({len(similar_memories)} memory(ies)):",
                            extra={
                                "tool": "search",
                                "result_count": len(similar_memories),
                            },
                        )
                        self.logger.info(
                            f"   Returning {len(similar_memories)} result(s)",
                            extra={
                                "tool": "search",
                                "result_count": len(similar_memories),
                            },
                        )
                    else:
                        self.logger.info(
                            "FINAL RESULTS: No matching memories found",
                            extra={"tool": "search", "result_count": 0},
                        )

                _ = self._log_operation_footer("search", start_time)

                self.log_completion(
                    "search",
                    query_length=len(query),
                    result_count=len(similar_memories),
                )

                return similar_memories

            except Exception as e:
                self._raise_tool_error(
                    "search",
                    e,
                    error_cls=SearchError,
                    message="Failed to search memory store",
                    query_length=len(query),
                )

        return search
