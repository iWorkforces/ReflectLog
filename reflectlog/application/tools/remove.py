"""Remove tool implementation for ReflectLogMCP Server."""

from typing import Any, override

from asyncer import asyncify

from ..exceptions import StorageError
from ..utils import truncate_memory
from .base import BaseTool


class RemoveTool(BaseTool):
    """Tool for removing memories from memory storage."""

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return "remove"

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
                self.log_invocation("remove", count=0)
                self.logger.info("Remove called with empty list, skipping")
                return

            self.log_invocation(
                "remove",
                requested_count=len(memories),
                search_limit=self.config.remove_search_limit,
            )

            self.logger.info(
                f"Attempting to remove {len(memories)} unique memory(ies)",
                extra={
                    "tool": "remove",
                    "count": len(memories),
                },
            )

            for idx, memory in enumerate(memories, 1):
                self.logger.info(
                    f"  [{idx}/{len(memories)}] Target: {truncate_memory(memory)}",
                    extra={
                        "tool": "remove",
                        "memory_index": idx,
                        "total_memories": len(memories),
                    },
                )

            actual_removed = 0
            memories_not_found: list[str] = []

            try:
                for memory_idx, memory in enumerate(memories, 1):
                    removed_count = await self._remove_single_memory_async(
                        memory, memory_idx, len(memories)
                    )

                    if removed_count == 0:
                        memories_not_found.append(truncate_memory(memory, 50))
                    else:
                        actual_removed += removed_count

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

            except Exception as e:
                self.log_error(
                    "remove",
                    e,
                    requested_count=len(memories),
                    actual_removed=actual_removed,
                )
                raise StorageError(
                    f"Failed to remove memories from memory store: {e}"
                ) from e

        return remove

    async def _remove_single_memory_async(
        self, memory: str, memory_idx: int, total_memories: int
    ) -> int:
        """Remove a single memory and all its occurrences (async).

        Args:
            memory: The memory to remove.
            memory_idx: Current memory index (for logging).
            total_memories: Total number of memories being removed (for logging).

        Returns:
            Number of occurrences removed.
        """
        self.logger.info(
            f"[{memory_idx}/{total_memories}] Searching for memory: {truncate_memory(memory)}",
            extra={
                "tool": "remove",
                "memory_index": memory_idx,
                "total_memories": total_memories,
            },
        )

        # Search for candidates via asyncify (non-blocking)
        candidates: list[dict[str, Any]] = await asyncify(
            self.memory.search_for_removal
        )(memory)

        self.logger.info(
            f"[{memory_idx}/{total_memories}] Found {len(candidates)} semantic candidate(s)",
            extra={
                "tool": "remove",
                "memory_index": memory_idx,
                "candidates_found": len(candidates),
            },
        )

        # Filter for exact matches FIRST - exact matches are definitively correct
        # regardless of score (scores can vary due to embedding quirks)
        self.logger.info(
            f"[{memory_idx}/{total_memories}] Filtering for exact matches...",
            extra={
                "tool": "remove",
                "memory_index": memory_idx,
            },
        )

        exact_matches = [item for item in candidates if item["memory"] == memory]

        # Log candidates with scores for debugging (only if no exact match found)
        if not exact_matches and candidates:
            self._log_removal_candidates(candidates, memory_idx, total_memories)

        if not exact_matches:
            self.logger.info(
                f"[{memory_idx}/{total_memories}] No exact match found",
                extra={
                    "tool": "remove",
                    "memory_index": memory_idx,
                },
            )
            return 0

        self.logger.info(
            f"[{memory_idx}/{total_memories}] Found {len(exact_matches)} exact match(es)",
            extra={
                "tool": "remove",
                "memory_index": memory_idx,
                "exact_matches": len(exact_matches),
            },
        )

        for match_idx, exact_match in enumerate(exact_matches, 1):
            self.logger.info(
                f"[{memory_idx}/{total_memories}] Deleting occurrence {match_idx}/{len(exact_matches)}",
                extra={
                    "tool": "remove",
                    "memory_index": memory_idx,
                    "match_index": match_idx,
                },
            )

            delete_by_memory = getattr(
                self.memory,
                "delete_by_memory",
                None,
            )
            if delete_by_memory is None:
                delete_by_memory = self.memory.delete_by_message
            _ = await asyncify(delete_by_memory)(exact_match["memory"])

            self.logger.info(
                f"[{memory_idx}/{total_memories}] Deleted occurrence {match_idx}",
                extra={
                    "tool": "remove",
                    "memory_index": memory_idx,
                    "match_index": match_idx,
                },
            )

        self.logger.info(
            f"[{memory_idx}/{total_memories}] Removed {len(exact_matches)} occurrence(s)",
            extra={
                "tool": "remove",
                "memory_index": memory_idx,
                "occurrences_removed": len(exact_matches),
            },
        )

        return len(exact_matches)

    def _log_removal_candidates(
        self,
        candidates: list[dict[str, Any]],
        memory_idx: int,
        total_memories: int,
    ) -> None:
        """Log removal candidates for debugging.

        Args:
            candidates: List of candidate records.
            memory_idx: Current memory index.
            total_memories: Total number of memories being removed.
        """
        self.logger.info(
            f"[{memory_idx}/{total_memories}] Found {len(candidates)} candidate(s)",
            extra={
                "tool": "remove",
                "memory_index": memory_idx,
                "candidates_count": len(candidates),
            },
        )

        # Sort by score for visualization
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("score", 0.0), reverse=True
        )

        for idx, candidate in enumerate(sorted_candidates[:3], 1):
            score = candidate.get("score", 0.0)
            preview = truncate_memory(candidate.get("memory", ""), max_length=50)

            self.logger.info(
                f"[{memory_idx}/{total_memories}]   [{idx}] Score: {score:.4f} → {preview}",
                extra={
                    "tool": "remove",
                    "memory_index": memory_idx,
                    "candidate_index": idx,
                    "score": score,
                },
            )
