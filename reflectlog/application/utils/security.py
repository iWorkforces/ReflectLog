"""Security utilities for ReflectLog Server."""

import re
from typing import Any, cast


class SecretString:
    """A wrapper for sensitive strings that prevents accidental exposure in logs.

    This class wraps sensitive values like API keys to ensure they are not
    accidentally logged or displayed. The actual value is only accessible
    via the `get_secret_value()` method.

    Example:
        >>> secret = SecretString("sk-abc123")
        >>> print(secret)
        ***REDACTED***
        >>> secret.get_secret_value()
        'sk-abc123'
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        """Initialize with a secret value.

        Args:
            value: The sensitive string to protect.
        """
        self._value = value

    def get_secret_value(self) -> str:
        """Get the actual secret value.

        Returns:
            The underlying secret string.
        """
        return self._value

    def __str__(self) -> str:
        """Return redacted representation for printing."""
        return "***REDACTED***"

    def __repr__(self) -> str:
        """Return redacted representation for debugging."""
        return "***REDACTED***"

    def __bool__(self) -> bool:
        """Return True if the secret value is non-empty."""
        return bool(self._value)

    def __len__(self) -> int:
        """Return the length of the secret value."""
        return len(self._value)

    def __eq__(self, other: object) -> bool:
        """Compare SecretString instances for equality.

        Args:
            other: Another object to compare with.

        Returns:
            True if the other object is a SecretString with the same value,
            False otherwise.
        """
        if not isinstance(other, SecretString):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        """Return hash of the secret value.

        Returns:
            Hash of the underlying secret string.
        """
        return hash(self._value)


# Patterns for sensitive data detection
_SENSITIVE_PATTERNS = [
    # API keys - expanded coverage
    (re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE), "[API_KEY_REDACTED]"),
    (
        re.compile(r"sk-or-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
        "[OPENROUTER_KEY_REDACTED]",
    ),
    (
        re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
        "[ANTHROPIC_KEY_REDACTED]",
    ),
    (
        re.compile(r"sk-proj-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
        "[OPENAI_PROJECT_KEY_REDACTED]",
    ),
    # Generic API key patterns
    (
        re.compile(r"api[_-]?key[\"']?\s*[:=]\s*[\"']?[a-zA-Z0-9_-]+", re.IGNORECASE),
        "[API_KEY_REDACTED]",
    ),
    (
        re.compile(r"apikey[\"']?\s*[:=]\s*[\"']?[a-zA-Z0-9_-]+", re.IGNORECASE),
        "[API_KEY_REDACTED]",
    ),
    # Bearer tokens
    (
        re.compile(r"bearer\s+[a-zA-Z0-9_.-]{20,}", re.IGNORECASE),
        "[BEARER_TOKEN_REDACTED]",
    ),
    (
        re.compile(r"Bearer\s+[a-zA-Z0-9_.-]{20,}", re.IGNORECASE),
        "[BEARER_TOKEN_REDACTED]",
    ),
    # GitHub tokens
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}", re.IGNORECASE), "[GITHUB_TOKEN_REDACTED]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36,}", re.IGNORECASE), "[GITHUB_TOKEN_REDACTED]"),
    (re.compile(r"ghu_[a-zA-Z0-9]{36,}", re.IGNORECASE), "[GITHUB_TOKEN_REDACTED]"),
    (re.compile(r"ghs_[a-zA-Z0-9]{36,}", re.IGNORECASE), "[GITHUB_TOKEN_REDACTED]"),
    (re.compile(r"ghr_[a-zA-Z0-9]{36,}", re.IGNORECASE), "[GITHUB_TOKEN_REDACTED]"),
    # Slack tokens
    (
        re.compile(r"xox[baprs]-[0-9]{10,}-[0-9]+", re.IGNORECASE),
        "[SLACK_TOKEN_REDACTED]",
    ),
    # Passwords
    (
        re.compile(r"password[\"']?\s*[:=]\s*[\"']?[^\s\"']{8,}", re.IGNORECASE),
        "[PASSWORD_REDACTED]",
    ),
    (
        re.compile(r"passwd[\"']?\s*[:=]\s*[\"']?[^\s\"']{8,}", re.IGNORECASE),
        "[PASSWORD_REDACTED]",
    ),
    # Email addresses - RFC 5322 compliant-ish pattern for redaction
    # Note: Full RFC 5322 compliance requires extremely complex regex.
    # This practical pattern covers common formats while being performant.
    (
        re.compile(
            r"[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}"
        ),
        "[EMAIL_REDACTED]",
    ),
    # URL query parameters with sensitive data
    (
        re.compile(
            r"[?&](api[_-]?key|apikey|token|password|secret)=[^&\\s\"']+", re.IGNORECASE
        ),
        "[URL_PARAM_REDACTED]",
    ),
    (
        re.compile(
            r"[?&](api[_-]?key|apikey|token|password|secret)=\"[^\"]*\"", re.IGNORECASE
        ),
        "[URL_PARAM_REDACTED]",
    ),
]

# Additional patterns specifically for URL parameter redaction
_URL_PARAM_PATTERNS = [
    (
        re.compile(
            r"[?&](api[_-]?key|apikey|token|password|secret)=[^&\\s\"']+", re.IGNORECASE
        ),
        "[URL_PARAM_REDACTED]",
    ),
    (
        re.compile(
            r"[?&](api[_-]?key|apikey|token|password|secret)=\"[^\"]*\"", re.IGNORECASE
        ),
        "[URL_PARAM_REDACTED]",
    ),
]


def sanitize_for_logging(
    value: Any,
    max_length: int = 200,
    redact_sensitive: bool = True,
) -> str:
    """Sanitize a value for safe logging.

    This function prepares values for logging by:
    1. Converting to string
    2. Truncating to max_length
    3. Optionally redacting sensitive patterns (API keys, passwords, etc.)

    Args:
        value: The value to sanitize (will be converted to string).
        max_length: Maximum length before truncation (default: 200).
        redact_sensitive: Whether to redact sensitive patterns (default: True).

    Returns:
        A sanitized string safe for logging.

    Example:
        >>> sanitize_for_logging("My API key is sk-abc123def456")
        'My API key is [API_KEY_REDACTED]'
        >>> sanitize_for_logging("x" * 500, max_length=100)
        'xxxx...xxxx (truncated, original length: 500)'
    """
    # Convert to string
    text = str(value) if value is not None else ""

    # Redact sensitive patterns
    if redact_sensitive:
        for pattern, replacement in _SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)

    # Truncate if needed
    if len(text) > max_length:
        # Show beginning and end for context
        half = (max_length - 20) // 2
        text = f"{text[:half]}...{text[-half:]} (truncated, original length: {len(str(value))})"

    return text


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


def redact_dict_secrets(
    data: dict[str, Any],
    secret_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Create a copy of a dictionary with secret values redacted.

    This is useful for safely logging configuration dictionaries that may
    contain sensitive values.

    Recursively handles nested dictionaries, lists, and tuples.

    Args:
        data: The dictionary to redact.
        secret_keys: Set of key names to redact (case-insensitive).
            Defaults to common secret key names.

    Returns:
        A new dictionary with secret values replaced by "[REDACTED]".

    Example:
        >>> config = {"api_key": "sk-secret", "model": "gpt-4"}
        >>> redact_dict_secrets(config)
        {'api_key': '[REDACTED]', 'model': 'gpt-4'}
    """
    if secret_keys is None:
        secret_keys = {
            "api_key",
            "apikey",
            "api-key",
            "secret",
            "password",
            "token",
            "key",
            "openrouter_api_key",
            "openai_api_key",
            "anthropic_auth_token",
        }

    secret_keys_lower = {k.lower() for k in secret_keys}

    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in secret_keys_lower:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_dict_secrets(cast(dict[str, Any], value), secret_keys)
        elif isinstance(value, list):
            # Handle lists - recurse on dict elements
            result[key] = [
                redact_dict_secrets(cast(dict[str, Any], item), secret_keys)
                if isinstance(item, dict)
                else item
                for item in cast(list[Any], value)
            ]
        elif isinstance(value, tuple):
            # Handle tuples - recurse on dict elements, preserve tuple type
            result[key] = tuple(
                redact_dict_secrets(cast(dict[str, Any], item), secret_keys)
                if isinstance(item, dict)
                else item
                for item in cast(tuple[Any, ...], value)
            )
        elif isinstance(value, SecretString):
            result[key] = str(value)  # Returns "***REDACTED***"
        else:
            result[key] = value

    return result
