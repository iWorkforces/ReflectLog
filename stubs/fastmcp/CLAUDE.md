# stubs/fastmcp/

Type stubs for the fastmcp library - MCP server framework.

## Purpose

Type stubs (`*.pyi`) provide type hints for fastmcp without requiring runtime dependencies. These stubs enable:

- IDE autocompletion for fastmcp APIs
- Type checking with mypy/pyright
- Better developer experience when working with MCP tools

## Structure

```
fastmcp/
├── __init__.pyi          # Main fastmcp exports
├── client/               # Client-related type stubs
└── utilities/            # Utility type stubs
```

## Related

- fastmcp documentation: https://github.com/jlowin/fastmcp
- Implementation in codebase: `reflectlog/application/mcp_server.py`
