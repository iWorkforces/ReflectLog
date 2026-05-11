# Agent Guidelines for reflectlog/application/tools/

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

## OVERVIEW
MCP tool implementations providing external interface to memory system. Each tool inherits BaseTool.

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|-------|
| Add tool | tools/add.py | 3-phase add pipeline |
| GetAll tool | tools/get_all.py | Returns all messages |
| Search tool | tools/search.py | RRF fusion + reranking |
| Remove tool | tools/remove.py | Exact match deletion |
| Health check | tools/health_check.py | Status reporting |
| Tool base | tools/base.py | Abstract base class |
| Registration | mcp_server.py | AVAILABLE_TOOL_CLASSES dict |

## ANTI-PATTERNS
- Never bypass tool layer for direct engine access
- Never log message content in production
- Never skip input validation
- Never use synchronous LLM calls in add pipeline
- Never assume both search engines return same result count
