"""Type definitions for CCOAuth2."""

from typing import Literal, TypedDict

# Token prefix constants
TOKEN_PREFIX = "sk-ant-"
OAUTH_TOKEN_PREFIX = "sk-ant-oat01-"
API_KEY_PREFIX = "sk-ant-api"

# Credential service name (must match Claude Code)
SERVICE_NAME = "Claude Code-credentials"


class ApiKeyResult(TypedDict):
    """Result from get_anthropic_api_key()."""

    api_key: str
    source: Literal["env", "claude-code"]
