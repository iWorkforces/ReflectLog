"""Linux credential retrieval via config files and secret-tool."""

import json
import os
from pathlib import Path
import subprocess
from typing import Any, cast

from .base import CredentialRetriever


class LinuxCredentialRetriever(CredentialRetriever):
    """Retrieve credentials from Linux config files or GNOME Keyring."""

    def get_credential(self) -> str | None:
        """Retrieve credential from Linux config files or secret-tool.

        Checks config files first, then falls back to secret-tool.

        Returns:
            The API key if found and valid, None otherwise.
        """
        # First, check config files
        config_credential = self._check_config_files()
        if config_credential:
            return config_credential

        # Fall back to secret-tool
        return self._get_from_secret_tool()

    def _get_config_paths(self) -> list[Path]:
        """Get list of potential credential file paths."""
        home = Path.home()
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))

        return [
            home / ".claude" / "credentials",
            home / ".config" / "claude-code" / "credentials",
            Path(xdg_config) / "claude-code" / "credentials",
        ]

    def _check_config_files(self) -> str | None:
        """Check credential files in standard locations."""
        for config_path in self._get_config_paths():
            if config_path.exists():
                try:
                    content = config_path.read_text().strip()

                    # Handle raw API key
                    if content.startswith(self.token_prefix):
                        return content

                    # Handle JSON formats
                    try:
                        raw_parsed: Any = json.loads(content)
                        if isinstance(raw_parsed, dict):
                            parsed: dict[str, Any] = cast("dict[str, Any]", raw_parsed)

                            # OAuth JSON format
                            oauth_data = parsed.get("claudeAiOauth")
                            if isinstance(oauth_data, dict):
                                typed_oauth: dict[str, Any] = cast(
                                    "dict[str, Any]", oauth_data
                                )
                                oauth_token = typed_oauth.get("accessToken")
                                if (
                                    oauth_token
                                    and isinstance(oauth_token, str)
                                    and oauth_token.startswith(self.token_prefix)
                                ):
                                    return oauth_token

                            # Legacy apiKey format
                            api_key = parsed.get("apiKey")
                            if (
                                api_key
                                and isinstance(api_key, str)
                                and api_key.startswith(self.token_prefix)
                            ):
                                return api_key

                    except json.JSONDecodeError:
                        pass

                except OSError:
                    continue

        return None

    def _get_from_secret_tool(self) -> str | None:
        """Retrieve from GNOME Keyring via secret-tool."""
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["secret-tool", "lookup", "service", self.service_name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return None

            return self.parse_credential(result.stdout)

        except subprocess.SubprocessError, OSError, FileNotFoundError:
            return None
