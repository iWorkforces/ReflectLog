"""Message validation utilities for CCMemoriesMCP Server."""

from typing import Any, Optional


def validate_messages(
    messages: list[Any], min_length: int, max_length: int
) -> tuple[bool, Optional[str]]:
    """Validate message list for add/remove operations.

    Args:
        messages: List of messages to validate.
        min_length: Minimum allowed message length in characters.
        max_length: Maximum allowed message length in characters.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is None.
        If invalid, error_message contains the validation error.
    """
    if not messages:
        # Empty list is a valid no-op for add/remove operations.
        return True, None

    for i, msg in enumerate(messages):
        # Check type
        if not isinstance(msg, str):
            return False, f"Message at index {i} is not a string: {type(msg)}"

        # Check minimum length
        if len(msg) < min_length:
            return (
                False,
                f"Message at index {i} is too short (min: {min_length} chars)",
            )

        # Check maximum length
        if len(msg) > max_length:
            return (
                False,
                f"Message at index {i} is too long (max: {max_length} chars)",
            )

        # Check for whitespace-only content
        if not msg.strip():
            return False, f"Message at index {i} contains only whitespace"

    return True, None


def truncate_message(message: str, max_length: int = 100) -> str:
    """Truncate a message for display purposes.

    Args:
        message: The message to truncate.
        max_length: Maximum length for display.

    Returns:
        Truncated message with ellipsis if needed.
    """
    if len(message) <= max_length:
        return message
    return f"{message[:max_length]}..."
