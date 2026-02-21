"""Platform-specific credential retrieval."""

import platform

from .base import CredentialRetriever


def get_platform_retriever() -> CredentialRetriever | None:
    """Get the appropriate credential retriever for the current platform.

    Returns:
        Platform-specific CredentialRetriever instance, or None if unsupported.
    """
    system = platform.system()

    match system:
        case "Darwin":
            from .darwin import DarwinCredentialRetriever

            return DarwinCredentialRetriever()
        case "Windows":
            from .windows import WindowsCredentialRetriever

            return WindowsCredentialRetriever()
        case "Linux":
            from .linux import LinuxCredentialRetriever

            return LinuxCredentialRetriever()
        case _:
            return None


__all__ = ["CredentialRetriever", "get_platform_retriever"]
