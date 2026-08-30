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
            "    • get_all(limit: int | None = None, offset: int = 0) -> dict\n"
            "      Page stored memories. Default cap 1000. Returns memories, "
            "total, offset, limit, truncated."
        )

    @override
    def get_handler(self):
        """Get the async tool handler function."""

        async def get_all(
            limit: int | None = None, offset: int = 0
        ) -> dict[str, object]:
            """Page stored memories (async).

            Default page size is Config.get_all_limit (1000). Returns paging
            metadata so clients can fetch further pages.

            Raises:
                StorageError: If retrieval operation fails.
            """
            try:
                self.log_invocation("get_all")

                start = max(0, offset)
                page_size = self.config.get_all_limit if limit is None else max(0, limit)
                page_size = min(page_size, self.config.get_all_limit)
                page = await asyncify(self.memory.get_all)(
                    limit=page_size, offset=start
                )
                total = self.memory.count()
                truncated = start + len(page) < total
                if truncated:
                    self.logger.warning(
                        "get_all truncated",
                        extra={
                            "tool": "get_all",
                            "returned": len(page),
                            "total": total,
                            "offset": start,
                            "limit": page_size,
                        },
                    )
                self.log_completion("get_all", count=len(page), total=total)
                return {
                    "memories": page,
                    "total": total,
                    "offset": start,
                    "limit": page_size,
                    "truncated": truncated,
                }

            except Exception as e:
                self.log_error("get_all", e)
                raise StorageError(
                    f"Failed to retrieve memories from memory store: {e}"
                ) from e

        return get_all
