"""Type stubs for fastmcp.server.dependencies."""


class _HttpHeaders:
    def get(self, key: str, default: str = "") -> str: ...


class _HttpRequest:
    headers: _HttpHeaders


def get_http_request() -> _HttpRequest: ...


__all__ = ["get_http_request"]
