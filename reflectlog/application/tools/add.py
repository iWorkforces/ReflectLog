"""Add tool implementation for ReflectLogMCP Server."""

import time
from typing import List

from ..exceptions import StorageError
from ..utils import truncate_message, validate_messages
from .base import BaseTool

LOG_MESSAGE_PREVIEW_LIMIT = 20


class AddTool(BaseTool):
    """Tool for adding messages to memory storage."""

    def get_name(self) -> str:
        """Get the tool name."""
        return "add"

    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • add(messages: list[str])\n"
            "      Add messages with semantic embeddings. Empty lists are no-op.\n"
            "      Messages must be 1-30720 characters, non-whitespace."
        )

    def get_handler(self):
        """Get the async tool handler function."""

        async def add(messages: List[str]) -> None:
            """Add messages to the message store with parallel processing (async).

            This tool stores one or more text messages in the memory store using
            semantic embeddings. Messages are processed concurrently for improved
            performance when adding multiple messages. Messages are stored with
            vector embeddings and will be available via get_all() or search().

            Note: Uses async add_messages_async() for concurrent processing of
            batch additions without blocking the event loop.

            Args:
                messages: List of message strings to add to storage. Empty list
                    is treated as no-op. Each message must be 1-30720 characters
                    and contain non-whitespace content.

            Returns:
                None. Messages are stored successfully if no error is raised.

            Raises:
                ValueError: If any message is invalid (empty, too long, whitespace-only).
                RuntimeError: If storage operation fails.

            Examples:
                >>> add(["Hello", "World"])  # Processed in parallel
                >>> add(["Meeting notes: Discussed API design"])
                >>> add([])  # No-op, no error
            """
            # Handle empty list gracefully (no-op, no error)
            if not messages:
                self.log_invocation("add", count=0)
                self.logger.info("Add called with empty list, nothing to store")
                return

            # Validate messages
            is_valid, error_msg = validate_messages(
                messages, self.config.min_message_length, self.config.max_message_length
            )

            if not is_valid:
                self.log_error("add", ValueError(error_msg), count=len(messages))
                raise ValueError(f"Invalid message: {error_msg}")

            # Log invocation with detailed info
            start_time = time.time()
            self.log_invocation("add", count=len(messages))

            # Log operation header
            total_chars = sum(len(m) for m in messages)
            self.logger.info(
                "=" * 60,
                extra={"tool": "add", "section": "header"},
            )
            self.logger.info(
                f"ADD OPERATION: Storing {len(messages)} message(s) "
                f"({total_chars:,} total characters)",
                extra={
                    "tool": "add",
                    "message_count": len(messages),
                    "total_characters": total_chars,
                    "hybrid_mode": self.config.enable_hybrid_search,
                },
            )

            # Log message previews (throttled for large batches)
            log_limit = min(len(messages), LOG_MESSAGE_PREVIEW_LIMIT)
            for idx, message in enumerate(messages[:log_limit], 1):
                preview = truncate_message(message, max_length=80)
                self.logger.info(
                    f"  Message {idx}/{len(messages)} ({len(message):,} chars): {preview}",
                    extra={
                        "tool": "add",
                        "message_index": idx,
                        "message_length": len(message),
                    },
                )
            if len(messages) > log_limit:
                self.logger.info(
                    f"  ... {len(messages) - log_limit} more message(s) omitted from logs",
                    extra={
                        "tool": "add",
                        "omitted_count": len(messages) - log_limit,
                        "message_count": len(messages),
                    },
                )

            # Store messages using async method for better concurrency
            try:
                result = await self.memory.add_messages_async(messages)

                # Log completion summary with timing
                duration = (time.time() - start_time) * 1000  # ms
                stored_count = result.stored_count
                skipped_count = result.skipped_count
                replaced_count = result.replaced_count

                self.logger.info(
                    "─" * 50,
                    extra={"tool": "add", "section": "summary"},
                )

                # Build summary message
                summary_parts = []
                if stored_count > 0:
                    summary_parts.append(f"{stored_count} stored")
                if replaced_count > 0:
                    summary_parts.append(f"{replaced_count} replaced")
                if skipped_count > 0:
                    summary_parts.append(f"{skipped_count} skipped (duplicates)")

                if summary_parts:
                    summary = ", ".join(summary_parts)
                    self.logger.info(
                        f"ADD SUMMARY: {summary}",
                        extra={
                            "tool": "add",
                            "stored_count": stored_count,
                            "replaced_count": replaced_count,
                            "skipped_count": skipped_count,
                        },
                    )
                else:
                    self.logger.info(
                        f"ADD COMPLETE: All {stored_count} message(s) stored successfully",
                        extra={
                            "tool": "add",
                            "stored_count": stored_count,
                        },
                    )

                # Log replacement details if any
                for replacement in result.replacements:
                    self.logger.info(
                        f"  Replaced: '{replacement.old_memory[:50]}...' → "
                        f"'{replacement.new_memory[:50]}...' (confidence: {replacement.confidence:.2f})",
                        extra={
                            "tool": "add",
                            "action": "replacement",
                            "confidence": replacement.confidence,
                            "reason": replacement.reason,
                        },
                    )

                avg_time = duration / len(messages) if messages else 0
                self.logger.info(
                    f"Completed in {duration:.0f}ms ({avg_time:.0f}ms/message avg)",
                    extra={
                        "tool": "add",
                        "duration_ms": duration,
                        "avg_ms_per_message": avg_time,
                    },
                )
                self.logger.info(
                    "=" * 60,
                    extra={"tool": "add", "section": "footer"},
                )

                self.log_completion("add", requested=len(messages), stored=stored_count)

            except Exception as e:
                self.log_error("add", e, count=len(messages))
                raise StorageError(
                    f"Failed to add messages to memory store: {e}"
                ) from e

        return add
