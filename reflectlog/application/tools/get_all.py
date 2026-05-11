"""Get all tool implementation for ReflectLogMCP Server."""

from typing import override

from asyncer import asyncify

from reflectlog.core.exceptions import StorageError

from ..utils.validation import truncate_memory
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
            "    • get_all() -> list[str]\n"
            "      Retrieve all stored memories. Returns empty list if none stored."
        )

    @override
    def get_handler(self):
        """Get the async tool handler function."""

        async def get_all() -> list[str]:
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

                # Retrieve all memories via asyncify (non-blocking)
                memories = await asyncify(self.memory.get_all)()

                # Log each retrieved memory
                for idx, memory in enumerate(memories, 1):
                    self.logger.info(
                        f"[{idx}/{len(memories)}] Memory: {truncate_memory(memory)}",
                        extra={
                            "tool": "get_all",
                            "memory_index": idx,
                            "total_memories": len(memories),
                            "memory_length": len(memory),
                        },
                    )

                self.log_completion("get_all", count=len(memories))

                # Return copy to prevent external mutation
                return memories.copy()

            except Exception as e:
                self.log_error("get_all", e)
                raise StorageError(
                    f"Failed to retrieve memories from memory store: {e}"
                ) from e

        return get_all
