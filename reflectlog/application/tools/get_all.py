"""Get all tool implementation for ReflectLog Server."""

from typing import override

from asyncer import asyncify

from reflectlog.core.exceptions import StorageError

from .base import BaseTool


class GetAllTool(BaseTool):
    """Tool for retrieving all memories from memory storage."""

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return "get_all"

    @override
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • get_all(limit: int | None = None, offset: int = 0) -> list[str]\n"
            "      Retrieve stored memories with an optional page. Default cap 1000."
        )

    @override
    def get_handler(self):
        """Get the async tool handler function."""

        async def get_all(
            limit: int | None = None, offset: int = 0
        ) -> list[str]:
            """Retrieve all stored memories (async).

            Returns all memories currently in the memory store. Order is determined
            by the underlying storage mechanism (typically insertion order).
            If no memories have been added, returns an empty list.

            Note: Uses asyncify() to bridge to sync MemoryManager.get_all() without
            blocking the event loop. This enables concurrent tool calls.

            Returns:
                List of all memories currently in storage. Returns empty list [] if no
                memories have been stored.

            Raises:
                StorageError: If retrieval operation fails.

            Examples:
                >>> add(["Message 1", "Message 2"])
                >>> get_all()
                ["Message 1", "Message 2"]
            """
            try:
                self.log_invocation("get_all")

                start = max(0, offset)
                page_size = self.config.get_all_limit if limit is None else max(0, limit)
                page_size = min(page_size, self.config.get_all_limit)
                page = await asyncify(self.memory.get_all)(
                    limit=page_size, offset=start
                )
                if len(page) == page_size:
                    self.logger.warning(
                        "get_all may be truncated",
                        extra={
                            "tool": "get_all",
                            "returned": len(page),
                            "offset": start,
                            "limit": page_size,
                        },
                    )
                self.log_completion("get_all", count=len(page))
                return page

            except Exception as e:
                self.log_error("get_all", e)
                raise StorageError(
                    f"Failed to retrieve memories from memory store: {e}"
                ) from e

        return get_all
