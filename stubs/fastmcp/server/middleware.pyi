"""Type stubs for fastmcp.server.middleware."""

from collections.abc import Callable

class Middleware:
    """FastMCP request middleware base class."""

    def __init__(self) -> None: ...
    async def on_request(
        self, context: object, call_next: Callable[..., object]
    ) -> object: ...

__all__ = ["Middleware"]
