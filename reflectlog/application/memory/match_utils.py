"""Shared utilities for exact-match checks and Tantivy query escaping."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine

from reflectlog.application.types import ISemanticSearchEngine
from reflectlog.core.logging import IStructuredLogger


def escape_tantivy_query(query: str) -> str:
    """Escape special characters for Tantivy query syntax.

    Tantivy uses Lucene-style query syntax where certain characters have
    special meaning. This function escapes them to prevent query injection.
    """
    special_chars = r'+-&|!(){}[]^"~*?:\/'
    escaped: list[str] = []
    for char in query:
        if char in special_chars:
            escaped.append(f"\\{char}")
        else:
            escaped.append(char)
    return "".join(escaped)


def has_exact_match(
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    project_id: str,
    content: str,
    logger: IStructuredLogger | None,
) -> bool:
    """Check whether the exact content already exists in storage.

    Uses Tantivy for fast exact phrase matching when available,
    falling back to direct database lookup otherwise.
    """
    if tantivy_engine is not None:
        try:
            escaped_query = escape_tantivy_query(content)
            results = tantivy_engine.search(
                f'"{escaped_query}"',
                project_id,
                limit=5,
            )
            has_match = any(msg == content for msg, _ in results)
            if has_match and logger:
                logger.debug(
                    "Tantivy found exact duplicate",
                    extra={"project_id": project_id},
                )
            return has_match
        except Exception as e:
            if logger:
                logger.warning(
                    "Tantivy duplicate check failed; falling back to database lookup",
                    extra={"project_id": project_id, "error": str(e)},
                )

    try:
        msg_id = semantic_engine.get_id_by_content(project_id, content)
        if msg_id is not None:
            if logger:
                logger.debug(
                    "Database lookup found exact duplicate",
                    extra={"project_id": project_id, "msg_id": msg_id},
                )
            return True
        return False
    except Exception as e:
        if logger:
            logger.warning(
                "Duplicate detection failed; proceeding without deduplication",
                extra={
                    "project_id": project_id,
                    "error": str(e),
                },
            )
        return False
