"""Get all tool implementation for ReflectLog Server."""

from typing import TYPE_CHECKING, override

from asyncer import asyncify

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class GetAllTool(BaseTool):
    """Tool for retrieving all memories from memory storage."""

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return ToolName.GET_ALL

    @override
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • get_all(limit: int | None = None, offset: int = 0) -> dict\n"
            "      Page stored memories. Default cap 1000. Returns memories, "
            "total, offset, limit, truncated."
        )

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
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
                self.log_invocation(ToolName.GET_ALL)

                start = max(0, offset)
                page_size = (
                    self.config.get_all_limit if limit is None else max(0, limit)
                )
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
                self.log_completion(ToolName.GET_ALL, count=len(page), total=total)
                return {
                    "memories": page,
                    "total": total,
                    "offset": start,
                    "limit": page_size,
                    "truncated": truncated,
                }

            except Exception as e:
                self.log_error(ToolName.GET_ALL, e)
                raise StorageError(
                    f"Failed to retrieve memories from memory store: {e}"
                ) from e

        return get_all
