"""Windows credential retrieval via Credential Manager."""

import subprocess

from .base import CredentialRetriever


class WindowsCredentialRetriever(CredentialRetriever):
    """Retrieve credentials from Windows Credential Manager using PowerShell."""

    def get_credential(self) -> str | None:
        """Retrieve credential from Windows Credential Manager.

        Uses PowerShell Get-StoredCredential command.

        Returns:
            The API key if found and valid, None otherwise.
        """
        try:
            ps_command = (
                f'$cred = Get-StoredCredential -Target "{self.service_name}" '
                "-ErrorAction SilentlyContinue; "
                "if ($cred) { $cred.GetNetworkCredential().Password }"
            )

            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            return self.parse_credential(result.stdout)

        except (subprocess.SubprocessError, OSError):
            return None
