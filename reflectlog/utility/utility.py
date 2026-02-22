"""CCOAuth2 Utility - Cross-platform Anthropic API key retrieval.

This module provides functions to retrieve Anthropic API keys from
environment variables and platform-specific credential stores.
"""

import os
import platform
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from .platforms import get_platform_retriever
from .types import OAUTH_TOKEN_PREFIX, ApiKeyResult

__all__ = [
    "AssistantMessage",
    "ClaudeAgentOptions",
    "ResultMessage",
    "TextBlock",
    "generate_content",
    "get_anthropic_api_key",
    "get_claude_code_api_key",
    "init_credentials",
]


def get_claude_code_api_key(verbose: bool = False) -> str | None:
    """Retrieve Claude Code API key from system keychain/credential store.

    Platform-specific retrieval:
    - macOS: Uses `security find-generic-password`
    - Windows: Uses PowerShell `Get-StoredCredential`
    - Linux: Checks config files, falls back to `secret-tool`

    Args:
        verbose: If True, print debug messages to stderr.

    Returns:
        The API key string if found, None otherwise.

    Note:
        Returns None silently on any error (matching TypeScript behavior).
    """
    if verbose:
        print(f"Retrieving from {platform.system()} credential store.", file=sys.stderr)

    try:
        retriever = get_platform_retriever()
        if retriever is None:
            return None

        return retriever.get_credential()

    except Exception:
        return None


def get_anthropic_api_key() -> ApiKeyResult | None:
    """Get API key from environment variable or Claude Code keychain.

    Checks in order:
    1. ANTHROPIC_API_KEY environment variable
    2. Platform-specific credential store (keychain/credential manager)

    Returns:
        Dictionary with 'api_key' and 'source' keys if found, None otherwise.
        Source is 'env' for environment variable, 'claude-code' for keychain.

    Example:
        >>> result = get_anthropic_api_key()
        >>> if result:
        ...     print(f"Key from {result['source']}: {result['api_key'][:20]}...")
    """
    # Check environment variable first
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return {"api_key": env_key, "source": "env"}

    # Fall back to Claude Code keychain
    claude_code_key = get_claude_code_api_key()
    if claude_code_key:
        return {"api_key": claude_code_key, "source": "claude-code"}

    return None


def init_credentials(verbose: bool = True) -> str | None:
    """Initialize OAuth credentials from environment or keychain.

    Checks for ANTHROPIC_AUTH_TOKEN environment variable first,
    then falls back to keychain retrieval. If an OAuth token is found
    in the keychain, it's automatically set in the environment.

    Args:
        verbose: If True, print status messages to stdout.

    Returns:
        The OAuth token if found, None otherwise.

    Example:
        >>> from ccoauth2 import init_credentials
        >>> token = init_credentials()
        >>> # Token is now available in os.environ["ANTHROPIC_AUTH_TOKEN"]
    """
    oauth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")

    if not oauth_token:
        # Try to retrieve from keychain directly
        keychain_token = get_claude_code_api_key()

        if keychain_token and keychain_token.startswith(OAUTH_TOKEN_PREFIX):
            oauth_token = keychain_token
            os.environ["ANTHROPIC_AUTH_TOKEN"] = oauth_token
            if verbose:
                print(
                    f"ANTHROPIC_AUTH_TOKEN: {oauth_token[:20]}... "
                    "(retrieved from keychain)"
                )
        elif keychain_token:
            # It's a regular API key, not an OAuth token
            if verbose:
                print(
                    "Found API key in keychain, but it is not an "
                    "OAuth token (sk-ant-oat01-*)."
                )
                print(
                    "OAuth tokens are only available with Claude Pro/Max subscriptions."
                )
                print("SDK will use stored credentials.\n")
        else:
            if verbose:
                print(
                    "ANTHROPIC_AUTH_TOKEN: <not set> (SDK will use stored credentials)"
                )
    else:
        if verbose:
            print(f"ANTHROPIC_AUTH_TOKEN: {oauth_token[:20]}...")

    return oauth_token


async def generate_content(
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
) -> str:
    """Generate content using Claude Agent SDK with OAuth authentication.

    This function uses the Claude Agent SDK which properly handles OAuth
    authentication through Claude Code. It's the recommended way to use
    OAuth tokens with the Anthropic API.

    Args:
        prompt: The prompt to send to Claude.
        model: Optional model to use (e.g., 'claude-sonnet-4-5-20250929').
        system_prompt: Optional system prompt.
        allowed_tools: Optional list of allowed tools (empty = no tools).

    Returns:
        The generated content as a string.

    Raises:
        RuntimeError: If the query fails.

    Example:
        >>> import asyncio
        >>> result = asyncio.run(generate_content("Say hello"))
        >>> print(result)
    """
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools or [],
        permission_mode="bypassPermissions",
    )

    result_text = ""

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result_text += block.text
        elif message.is_error:
            raise RuntimeError(f"Query failed: {message.subtype}")

    return result_text.strip()
