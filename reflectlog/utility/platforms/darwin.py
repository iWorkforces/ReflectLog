'''macOS (Darwin) credential retrieval via Keychain.'''

import subprocess

from .base import CredentialRetriever


class DarwinCredentialRetriever(CredentialRetriever):
    '''Retrieve credentials from macOS Keychain using security command.'''

    def get_credential(self) -> str | None:
        '''Retrieve credential from macOS Keychain.

        Uses: security find-generic-password -s "Claude Code-credentials" -w

        Returns:
            The API key if found and valid, None otherwise.
        '''
        try:
            result = subprocess.run(
                [
                    'security',
                    'find-generic-password',
                    '-s',
                    self.service_name,
                    '-w',
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return None

            return self.parse_credential(result.stdout)

        except subprocess.SubprocessError, OSError:
            return None
