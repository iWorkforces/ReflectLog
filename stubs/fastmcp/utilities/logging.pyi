"""Type stubs for fastmcp.utilities.logging module."""

import logging

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance configured for FastMCP.

    Args:
        name: The name of the logger.

    Returns:
        A configured logging.Logger instance.
    """
    ...

__all__ = ["get_logger"]
