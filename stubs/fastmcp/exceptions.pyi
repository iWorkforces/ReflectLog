"""Type stubs for fastmcp.exceptions."""


class ToolError(Exception):
    """MCP tool error surfaced to the client."""

    def __init__(self, message: str = "") -> None: ...


__all__ = ["ToolError"]
