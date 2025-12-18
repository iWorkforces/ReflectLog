"""Remove tool implementation for CCMemoriesMCP Server."""

from typing import Any, Dict, List

from asyncer import asyncify

from ..exceptions import StorageError
from ..utils import truncate_message
from .base import BaseTool


class RemoveTool(BaseTool):
    """Tool for removing messages from memory storage."""

    def get_name(self) -> str:
        """Get the tool name."""
        return "remove"

    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • remove(messages: list[str])\n"
            "      Remove messages using exact string matching (case-sensitive).\n"
            "      Uses USearch (source of truth) with Python-level exact matching.\n"
            "      Removes all occurrences of each message. Silently ignores non-existent messages."
        )

    def get_handler(self):
        """Get the async tool handler function."""

        async def remove(messages: List[str]) -> None:
            """Remove messages from the message store using exact string matching (async).

            This tool removes messages that exactly match the provided strings.
            Uses semantic search to find candidates, then filters for exact matches
            to ensure only the intended messages are deleted. If a message appears
            multiple times, all exact occurrences are removed. Messages not found
            are silently ignored.

            Note: Uses asyncify() to bridge to sync MemoryManager methods without
            blocking the event loop. This enables concurrent tool calls.

            Args:
                messages: List of message strings to remove from storage. Can be empty
                    (treated as no-op). Messages not found in the store are ignored.
                    Messages are matched using exact string equality (case-sensitive).

            Returns:
                None. Messages are removed successfully if no error is raised.

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
            if not messages:
                self.log_invocation("remove", count=0)
                self.logger.info("Remove called with empty list, skipping")
                return

            # Log invocation
            self.log_invocation(
                "remove",
                requested_count=len(messages),
                search_limit=self.config.remove_search_limit,
            )

            self.logger.info(
                f"Attempting to remove {len(messages)} unique message(s)",
                extra={
                    "tool": "remove",
                    "count": len(messages),
                },
            )

            # Log each message to be removed
            for idx, msg in enumerate(messages, 1):
                self.logger.info(
                    f"  [{idx}/{len(messages)}] Target: {truncate_message(msg)}",
                    extra={
                        "tool": "remove",
                        "message_index": idx,
                        "total_messages": len(messages),
                    },
                )

            actual_removed = 0
            messages_not_found: List[str] = []

            try:
                for msg_idx, message in enumerate(messages, 1):
                    removed_count = await self._remove_single_message_async(
                        message, msg_idx, len(messages)
                    )

                    if removed_count == 0:
                        messages_not_found.append(truncate_message(message, 50))
                    else:
                        actual_removed += removed_count

                # Log final summary
                self.logger.info(
                    f"Removal complete: {actual_removed} total occurrence(s) removed, "
                    f"{len(messages_not_found)} message(s) not found",
                    extra={
                        "tool": "remove",
                        "requested_count": len(messages),
                        "actual_removed": actual_removed,
                        "not_found_count": len(messages_not_found),
                    },
                )

                if actual_removed > 0:
                    self.logger.info(
                        f"  Successfully deleted {actual_removed} message occurrence(s)",
                        extra={"tool": "remove", "actual_removed": actual_removed},
                    )

                if messages_not_found:
                    self.logger.info(
                        f"  {len(messages_not_found)} message(s) were not found",
                        extra={
                            "tool": "remove",
                            "not_found_count": len(messages_not_found),
                            "not_found_previews": messages_not_found,
                        },
                    )

                self.log_completion(
                    "remove",
                    requested=len(messages),
                    removed=actual_removed,
                    not_found=len(messages_not_found),
                )

            except Exception as e:
                self.log_error(
                    "remove",
                    e,
                    requested_count=len(messages),
                    actual_removed=actual_removed,
                )
                raise StorageError(
                    f"Failed to remove messages from memory store: {e}"
                ) from e

        return remove

    async def _remove_single_message_async(
        self, message: str, msg_idx: int, total_messages: int
    ) -> int:
        """Remove a single message and all its occurrences (async).

        Args:
            message: The message to remove.
            msg_idx: Current message index (for logging).
            total_messages: Total number of messages being removed (for logging).

        Returns:
            Number of occurrences removed.
        """
        self.logger.info(
            f"[{msg_idx}/{total_messages}] Searching for message: {truncate_message(message)}",
            extra={
                "tool": "remove",
                "message_index": msg_idx,
                "total_messages": total_messages,
            },
        )

        # Search for candidates via asyncify (non-blocking)
        candidates: List[Dict[str, Any]] = await asyncify(
            self.memory.search_for_removal
        )(message)

        self.logger.info(
            f"[{msg_idx}/{total_messages}] Found {len(candidates)} semantic candidate(s)",
            extra={
                "tool": "remove",
                "message_index": msg_idx,
                "candidates_found": len(candidates),
            },
        )

        # Filter for exact matches FIRST - exact matches are definitively correct
        # regardless of score (scores can vary due to embedding quirks)
        self.logger.info(
            f"[{msg_idx}/{total_messages}] Filtering for exact matches...",
            extra={
                "tool": "remove",
                "message_index": msg_idx,
            },
        )

        exact_matches = [item for item in candidates if item["memory"] == message]

        # Log candidates with scores for debugging (only if no exact match found)
        if not exact_matches and candidates:
            self._log_removal_candidates(candidates, msg_idx, total_messages)

        if not exact_matches:
            self.logger.info(
                f"[{msg_idx}/{total_messages}] No exact match found",
                extra={
                    "tool": "remove",
                    "message_index": msg_idx,
                },
            )
            return 0

        self.logger.info(
            f"[{msg_idx}/{total_messages}] Found {len(exact_matches)} exact match(es)",
            extra={
                "tool": "remove",
                "message_index": msg_idx,
                "exact_matches": len(exact_matches),
            },
        )

        # Delete all exact matches via asyncify using message content
        for match_idx, exact_match in enumerate(exact_matches, 1):
            self.logger.info(
                f"[{msg_idx}/{total_messages}] Deleting occurrence {match_idx}/{len(exact_matches)}",
                extra={
                    "tool": "remove",
                    "message_index": msg_idx,
                    "match_index": match_idx,
                },
            )

            # Use delete_by_message which properly handles hybrid storage deletion
            await asyncify(self.memory.delete_by_message)(exact_match["memory"])

            self.logger.info(
                f"[{msg_idx}/{total_messages}] Deleted occurrence {match_idx}",
                extra={
                    "tool": "remove",
                    "message_index": msg_idx,
                    "match_index": match_idx,
                },
            )

        self.logger.info(
            f"[{msg_idx}/{total_messages}] Removed {len(exact_matches)} occurrence(s)",
            extra={
                "tool": "remove",
                "message_index": msg_idx,
                "occurrences_removed": len(exact_matches),
            },
        )

        return len(exact_matches)

    def _log_removal_candidates(
        self, candidates: list, msg_idx: int, total_messages: int
    ) -> None:
        """Log removal candidates for debugging.

        Args:
            candidates: List of candidate records.
            msg_idx: Current message index.
            total_messages: Total number of messages being removed.
        """
        self.logger.info(
            f"[{msg_idx}/{total_messages}] Found {len(candidates)} candidate(s)",
            extra={
                "tool": "remove",
                "message_index": msg_idx,
                "candidates_count": len(candidates),
            },
        )

        # Sort by score for visualization
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("score", 0.0), reverse=True
        )

        for idx, candidate in enumerate(sorted_candidates[:3], 1):
            score = candidate.get("score", 0.0)
            preview = truncate_message(candidate.get("memory", ""), max_length=50)

            self.logger.info(
                f"[{msg_idx}/{total_messages}]   [{idx}] Score: {score:.4f} → {preview}",
                extra={
                    "tool": "remove",
                    "message_index": msg_idx,
                    "candidate_index": idx,
                    "score": score,
                },
            )
