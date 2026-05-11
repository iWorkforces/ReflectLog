"""Security utilities for input validation.

This module provides security validation functions that are safe
to use across all layers (utility, infrastructure, application).
"""

import re


def validate_project_id(project_id: str) -> str:
    """Validate project_id to prevent path traversal attacks.

    This function validates that a project_id contains only safe characters
    and does not contain path traversal patterns like "../" that could allow
    writing files outside the intended directory.

    Args:
        project_id: The project identifier to validate.

    Returns:
        The validated project_id (lowercased).

    Raises:
        ValidationError: If project_id contains invalid characters or path patterns.

    Example:
        >>> validate_project_id("my-project-123")
        'my-project-123'
        >>> validate_project_id("../../../etc")
        ValidationError: Invalid project_id: ../../../etc
    """
    from reflectlog.core.exceptions import ValidationError

    if not project_id:
        raise ValidationError("project_id cannot be empty")

    # Normalize to lowercase first (before other checks)
    project_id = project_id.lower()

    # Check for path traversal patterns
    if ".." in project_id or project_id.startswith("/"):
        raise ValidationError(f"Invalid project_id: {project_id}")

    # Allow only alphanumeric, hyphen, underscore, and dot
    if not re.match(r"^[a-zA-Z0-9_.-]+$", project_id):
        raise ValidationError(
            f"project_id contains invalid characters: {project_id}. "
            "Only alphanumeric, hyphens, underscores, and dots are allowed."
        )

    # Length limit (prevent excessively long project IDs)
    if len(project_id) > 128:
        raise ValidationError(
            f"project_id too long (max 128 characters): {len(project_id)}"
        )

    return project_id
