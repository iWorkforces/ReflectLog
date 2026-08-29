# Agent Guidelines for reflectlog/application/tools/

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW
MCP tool implementations providing external interface to memory system. Each tool inherits BaseTool.

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|-------|
| Add tool | tools/add.py | 3-phase add pipeline |
| GetAll tool | tools/get_all.py | Returns all messages |
| Search tool | tools/search.py | RRF fusion + reranking |
| Remove tool | tools/remove.py | `delete_memories` → `list[str]`; not-found = set difference |
| Health check | tools/health_check.py | Read-only; no reconcile |
| Tool base | tools/base.py | Abstract base class |
| Registration | mcp_server.py | AVAILABLE_TOOL_CLASSES dict |

## ANTI-PATTERNS
- Never bypass tool layer for direct engine access
- Never log message content in production
- Never skip input validation
- Never use synchronous LLM calls in add pipeline
- Never assume both search engines return same result count

## NOTES

Tools validate input, emit redacted structured metadata, delegate to `MemoryManager`, and translate errors. Keep engine, pipeline, and locking logic outside this package.

Tool registration is owned by `application/mcp_server.py`.
