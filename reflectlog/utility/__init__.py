"""CCOAuth2 - Cross-platform Anthropic API key retrieval.

This package provides utilities for retrieving Anthropic API keys from
environment variables and platform-specific credential stores (keychain,
credential manager, etc.).
"""

__version__ = "0.1.0"

from .types import OAUTH_TOKEN_PREFIX, TOKEN_PREFIX, ApiKeyResult
from .utility import (
    generate_content,
    get_anthropic_api_key,
    get_claude_code_api_key,
    init_credentials,
)

__all__ = [
    "generate_content",
    "get_anthropic_api_key",
    "get_claude_code_api_key",
    "init_credentials",
    "ApiKeyResult",
    "TOKEN_PREFIX",
    "OAUTH_TOKEN_PREFIX",
]
