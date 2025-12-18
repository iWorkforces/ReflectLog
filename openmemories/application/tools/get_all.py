"""Get all tool implementation for CCMemoriesMCP Server."""

from typing import List

from asyncer import asyncify

from ..exceptions import StorageError
from ..utils import truncate_message
from .base import BaseTool


class GetAllTool(BaseTool):
    """Tool for retrieving all messages from memory storage."""

    def get_name(self) -> str:
        """Get the tool name."""
        return "get_all"

    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • get_all() -> list[str]\n"
            "      Retrieve all stored messages. Returns empty list if none stored."
        )

    def get_handler(self):
        """Get the async tool handler function."""

        async def get_all() -> List[str]:
            """Retrieve all stored messages (async).

            Returns all messages currently in the memory store. Order is determined
            by the underlying storage mechanism (typically insertion order).
            If no messages have been added, returns an empty list.

            Note: Uses asyncify() to bridge to sync MemoryManager.get_all() without
            blocking the event loop. This enables concurrent tool calls.

            Returns:
                List of all messages currently in storage. Returns empty list [] if no
                messages have been stored.

            Raises:
                StorageError: If retrieval operation fails.

            Examples:
                >>> add(["Message 1", "Message 2"])
                >>> get_all()
                ["Message 1", "Message 2"]
            """
            try:
                self.log_invocation("get_all")

                # Retrieve all messages via asyncify (non-blocking)
                messages = await asyncify(self.memory.get_all)()

                # Log each retrieved message
                for idx, message in enumerate(messages, 1):
                    self.logger.info(
                        f"[{idx}/{len(messages)}] Message: {truncate_message(message)}",
                        extra={
                            "tool": "get_all",
                            "message_index": idx,
                            "total_messages": len(messages),
                            "message_length": len(message),
                        },
                    )

                self.log_completion("get_all", count=len(messages))

                # Return copy to prevent external mutation
                return messages.copy()

            except Exception as e:
                self.log_error("get_all", e)
                raise StorageError(
                    f"Failed to retrieve messages from memory store: {e}"
                ) from e

        return get_all
