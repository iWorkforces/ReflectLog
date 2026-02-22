"""Base class for platform-specific credential retrieval."""

from abc import ABC, abstractmethod
import json
from typing import Any, cast

from ..types import SERVICE_NAME, TOKEN_PREFIX


class CredentialRetriever(ABC):
    """Abstract base class for platform-specific credential retrieval."""

    service_name: str = SERVICE_NAME
    token_prefix: str = TOKEN_PREFIX

    @abstractmethod
    def get_credential(self) -> str | None:
        """Retrieve credential from platform-specific store.

        Returns:
            The API key/token if found, None otherwise.
        """

    def parse_credential(self, raw: str) -> str | None:
        """Parse credential from raw string (JSON or plain text).

        Handles three formats:
        1. OAuth JSON: {"claudeAiOauth": {"accessToken": "sk-ant-..."}}
        2. Legacy JSON: {"apiKey": "sk-ant-..."}
        3. Raw token: "sk-ant-..."

        Args:
            raw: Raw credential string from keychain/credential store.

        Returns:
            Parsed API key if valid, None otherwise.
        """
        raw = raw.strip()
        if not raw:
            return None

        if raw.startswith("{"):
            try:
                raw_json: Any = json.loads(raw)
                if isinstance(raw_json, dict):
                    parsed: dict[str, Any] = cast(dict[str, Any], raw_json)

                    oauth_data = parsed.get("claudeAiOauth")
                    if isinstance(oauth_data, dict):
                        typed_oauth: dict[str, Any] = cast(dict[str, Any], oauth_data)
                        access_token = typed_oauth.get("accessToken")
                        if (
                            access_token
                            and isinstance(access_token, str)
                            and access_token.startswith(self.token_prefix)
                        ):
                            return access_token

                    api_key = parsed.get("apiKey")
                    if (
                        api_key
                        and isinstance(api_key, str)
                        and api_key.startswith(self.token_prefix)
                    ):
                        return api_key

            except json.JSONDecodeError:
                pass

        if raw.startswith(self.token_prefix):
            return raw

        return None
