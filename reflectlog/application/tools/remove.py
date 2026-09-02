"""Remove tool implementation for ReflectLog Server."""

from typing import override

from asyncer import asyncify

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from ..utils.validation import validate_remove_batch, validate_remove_memories
from .base import BaseTool


class RemoveTool(BaseTool):
    """Tool for removing memories from memory storage."""

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return ToolName.REMOVE

    @override
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • remove(memories: list[str])\n"
            "      Remove memories using exact string matching (case-sensitive).\n"
            "      Uses USearch (source of truth) with Python-level exact matching.\n"
            "      Removes all occurrences of each memory. Silently ignores non-existent memories."
        )

    @override
    def get_handler(self):
        """Get the async tool handler function."""

        async def remove(memories: list[str]) -> None:
            """Remove memories from the memory store using exact string matching (async).

            This tool removes memories that exactly match the provided strings.
            Uses semantic search to find candidates, then filters for exact matches
            to ensure only the intended memories are deleted. If a memory appears
            multiple times, all exact occurrences are removed. Memories not found
            are silently ignored.

            Note: Uses asyncify() to bridge to sync MemoryManager methods without
            blocking the event loop. This enables concurrent tool calls.

            Performance Note: Memories are processed sequentially (O(n) where n is
            the number of memories). For bulk removals (100+ memories), consider
            batching into smaller groups or using a dedicated bulk operation if
            available in the future.

            Args:
                memories: List of memory strings to remove from storage. Can be empty
                    (treated as no-op). Memories not found in the store are ignored.
                    Memories are matched using exact string equality (case-sensitive).

            Returns:
                None. Memories are removed successfully if no error is raised.

            Raises:
                StorageError: If deletion operation fails.

            Examples:
                >>> add(["Hello", "World", "Hello"])
                >>> remove(["Hello"])
                >>> get_all()
                ["World"]
                >>> remove(["NonExistent"])  # No error, silently ignored
            """
            # Handle empty list gracefully (no-op)
            if not memories:
                self.log_invocation(ToolName.REMOVE, count=0)
                self.logger.info("Remove called with empty list, skipping")
                return

            unique_memories = list(dict.fromkeys(memories))
            is_valid, error_msg = validate_remove_memories(unique_memories)
            if is_valid:
                is_valid, error_msg = validate_remove_batch(
                    unique_memories,
                    self.config.max_add_batch,
                    self.config.max_add_chars,
                )
            if not is_valid:
                raise ValueError(f"Invalid memory: {error_msg}")

            self.log_invocation(
                ToolName.REMOVE,
                requested_count=len(unique_memories),
                search_limit=self.config.remove_search_limit,
            )

            self.logger.info(
                f"Attempting to remove {len(unique_memories)} unique memory(ies)",
                extra={
                    "tool": "remove",
                    "count": len(unique_memories),
                },
            )

            self.logger.info(
                f"  {len(unique_memories)} removal target(s)",
                extra={
                    "tool": "remove",
                    "total_memories": len(unique_memories),
                },
            )
            actual_removed = 0
            memories_not_found: list[str] = []

            try:
                deleted = await asyncify(self.memory.delete_memories)(unique_memories)
                deleted_set = set(deleted)
                actual_removed = len(deleted_set)
                memories_not_found = [
                    f"index:{idx}"
                    for idx, memory in enumerate(unique_memories)
                    if memory not in deleted_set
                ]

                # Log final summary
                self.logger.info(
                    f"Removal complete: {actual_removed} total occurrence(s) removed, "
                    f"{len(memories_not_found)} memory(ies) not found",
                    extra={
                        "tool": "remove",
                        "requested_count": len(memories),
                        "actual_removed": actual_removed,
                        "not_found_count": len(memories_not_found),
                    },
                )

                if actual_removed > 0:
                    self.logger.info(
                        f"  Successfully deleted {actual_removed} memory occurrence(s)",
                        extra={"tool": "remove", "actual_removed": actual_removed},
                    )

                if memories_not_found:
                    self.logger.info(
                        f"  {len(memories_not_found)} memory(ies) were not found",
                        extra={
                            "tool": "remove",
                            "not_found_count": len(memories_not_found),
                            "not_found_previews": memories_not_found,
                        },
                    )

                self.log_completion(
                    "remove",
                    requested=len(memories),
                    removed=actual_removed,
                    not_found=len(memories_not_found),
                )

            except TypeError:
                raise
            except Exception as e:
                self._raise_tool_error(
                    "remove",
                    e,
                    error_cls=StorageError,
                    message="Failed to remove memories from memory store",
                    requested_count=len(memories),
                    actual_removed=actual_removed,
                )

        return remove
