"""Security utilities for input validation.

This module provides security validation functions that are safe
to use across all layers (utility, infrastructure, application).
"""

import re


def validate_workspace_id(workspace_id: str) -> str:
    """Validate workspace_id to prevent path traversal attacks.

    This function validates that a workspace_id contains only safe characters
    and does not contain path traversal patterns like "../" that could allow
    writing files outside the intended directory.

    Args:
        workspace_id: The workspace identifier to validate.

    Returns:
        The validated workspace_id (lowercased).

    Raises:
        ValidationError: If workspace_id contains invalid characters or path patterns.

    Example:
        >>> validate_workspace_id("my-project-123")
        'my-project-123'
        >>> validate_workspace_id("../../../etc")
        ValidationError: Invalid workspace_id: ../../../etc
    """
    from reflectlog.core.exceptions import ValidationError

    if not workspace_id:
        raise ValidationError("workspace_id cannot be empty")

    # Normalize to lowercase first (before other checks)
    workspace_id = workspace_id.lower()

    # Check for path traversal patterns
    if ".." in workspace_id or workspace_id.startswith("/"):
        raise ValidationError(f"Invalid workspace_id: {workspace_id}")

    # Allow only alphanumeric, hyphen, underscore, and dot
    if not re.match(r"^[a-zA-Z0-9_.-]+$", workspace_id):
        raise ValidationError(
            f"workspace_id contains invalid characters: {workspace_id}. "
            "Only alphanumeric, hyphens, underscores, and dots are allowed."
        )

    # Length limit (prevent excessively long workspace IDs)
    if len(workspace_id) > 128:
        raise ValidationError(
            f"workspace_id too long (max 128 characters): {len(workspace_id)}"
        )

    return workspace_id
