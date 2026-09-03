"""Shared utilities for exact-match checks."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflectlog.core.logging import IStructuredLogger
    from reflectlog.core.types import ISemanticSearchEngine
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine


def has_exact_match(
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    workspace_id: str,
    content: str,
    logger: IStructuredLogger | None,
) -> bool:
    """Check whether the exact content already exists in storage.

    Uses the unique SQLite ``(workspace_id, content)`` index. Tantivy is
    not consulted: ``en_stem`` cannot do exact identity and a phrase query
    of the full memory is more expensive than the indexed lookup.

    ``tantivy_engine`` is accepted for call-site compatibility.
    """
    _ = tantivy_engine
    try:
        msg_id = semantic_engine.get_id_by_content(workspace_id, content)
        if msg_id is not None:
            if logger:
                logger.debug(
                    "Database lookup found exact duplicate",
                    extra={"workspace_id": workspace_id, "msg_id": msg_id},
                )
            return True
        return False
    except Exception as e:
        if logger:
            logger.warning(
                "Duplicate detection failed; proceeding without deduplication",
                extra={
                    "workspace_id": workspace_id,
                    "error": str(e),
                },
            )
        return False
