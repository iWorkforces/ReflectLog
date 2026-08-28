"""Restart-safe reconciliation of unfinished smart replacements.

SQLite archive + transition rows are one local transaction. USearch and
Tantivy commits are independent. Recovery *attempts* to converge leftover
intent; it can be skipped, and it will not mark a row complete while
SQLite or hybrid Tantivy still disagree.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import os
from typing import TYPE_CHECKING, cast

from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import (
    IArchiveMemoryStore,
    ISemanticSearchEngine,
    ReplacementTransition,
)

if TYPE_CHECKING:
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine


def reconcile_pending_replacements(
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    write_lock: AbstractContextManager[object],
    lock: AbstractContextManager[object] | None = None,
    logger: IStructuredLogger,
) -> int:
    """Finish pending replacements using the semantic store as source of truth.

    Acquires ``write_lock`` before ``lock`` when both are provided.

    Returns:
        Number of pending transitions that were applied.
    """
    store = _recovery_store(semantic_engine, logger)
    if store is None:
        return 0

    pending = _pending_rows(store.list_pending_transitions())
    if not pending:
        return 0

    semantic_engine.ensure_initialized()
    if tantivy_engine is not None:
        tantivy_engine.ensure_initialized()

    completed = 0
    inner_lock = lock if lock is not None else nullcontext()
    with write_lock, inner_lock:
        for transition in _pending_rows(store.list_pending_transitions()):
            apply_pending_transition(
                transition,
                semantic_engine=semantic_engine,
                tantivy_engine=tantivy_engine,
                logger=logger,
            )
            completed += 1

    if completed:
        logger.info(
            "Reconciled unfinished replacement transitions",
            extra={"reconciled_count": completed},
        )
    return completed


def apply_pending_transition(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    logger: IStructuredLogger,
) -> None:
    """Converge both indexes to the replacement recorded in ``transition``."""
    _remove_recorded_old(transition, semantic_engine, tantivy_engine)
    _ensure_replacement_present(
        transition,
        semantic_engine=semantic_engine,
        tantivy_engine=tantivy_engine,
    )

    if tantivy_engine is not None:
        tantivy_engine.commit()
    semantic_engine.commit()

    if not replacement_converged(
        transition, semantic_engine=semantic_engine, tantivy_engine=tantivy_engine
    ):
        logger.warning(
            "Replacement transition not complete; indexes have not converged",
            extra={
                "transition_id": transition.id,
                "old_memory_id": transition.old_memory_id,
                "workspace_id": transition.workspace_id,
            },
        )
        return

    semantic_engine.memory_store.complete_replacement_transition(transition.id)
    logger.info(
        "Applied pending replacement transition",
        extra={
            "transition_id": transition.id,
            "archive_id": transition.archive_id,
            "old_memory_id": transition.old_memory_id,
            "workspace_id": transition.workspace_id,
        },
    )


def replacement_converged(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> bool:
    """Return True when SQLite has the replacement and hybrid Tantivy agrees."""
    new_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if new_id is None:
        return False
    if not _vector_present(semantic_engine, new_id):
        return False

    if tantivy_engine is None:
        return True
    if not _tantivy_has(tantivy_engine, transition.workspace_id, transition.new_content):
        return False
    return not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.old_content
    )


def _remove_recorded_old(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> None:
    """Delete the recorded old id; tombstone Tantivy only if that text is not live."""
    semantic_engine.delete(memory_id=str(transition.old_memory_id))
    if tantivy_engine is None:
        return

    current_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.old_content
    )
    if current_id is not None and current_id != transition.old_memory_id:
        return
    _ = tantivy_engine.delete(transition.workspace_id, transition.old_content)


def _recovery_store(
    semantic_engine: ISemanticSearchEngine,
    logger: IStructuredLogger,
) -> IArchiveMemoryStore | None:
    """Return a transition store when pending rows can be listed."""
    store = getattr(semantic_engine, "memory_store", None)
    list_pending = getattr(store, "list_pending_transitions", None)
    if store is None or not callable(list_pending):
        logger.warning(
            "Skipping replacement recovery; memory store cannot list transitions",
            extra={"store_type": type(store).__name__},
        )
        return None

    db_path = getattr(store, "db_path", None)
    if isinstance(db_path, str) and db_path and not os.path.exists(db_path):
        return None
    return cast(IArchiveMemoryStore, store)


def _ensure_replacement_present(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> None:
    """Insert the replacement when SQLite/USearch or Tantivy still lacks it."""
    existing_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if existing_id is None:
        semantic_engine.add(
            workspace_id=transition.workspace_id,
            content=transition.new_content,
            infer=False,
        )
    else:
        _reindex_if_vector_missing(semantic_engine, existing_id, transition)

    if tantivy_engine is None:
        return
    if not _tantivy_has(tantivy_engine, transition.workspace_id, transition.new_content):
        tantivy_engine.add(transition.workspace_id, transition.new_content)


def _reindex_if_vector_missing(
    semantic_engine: ISemanticSearchEngine,
    existing_id: int,
    transition: ReplacementTransition,
) -> None:
    """Re-add a SQLite row whose USearch vector was not committed."""
    if _vector_present(semantic_engine, existing_id):
        return

    semantic_engine.delete(memory_id=str(existing_id))
    semantic_engine.add(
        workspace_id=transition.workspace_id,
        content=transition.new_content,
        infer=False,
    )


def _vector_present(semantic_engine: ISemanticSearchEngine, memory_id: int) -> bool:
    """Return False only when a real index is missing the id."""
    index = getattr(semantic_engine, "index", None)
    if index is None:
        return True
    try:
        return memory_id in index
    except TypeError:
        return False


def _pending_rows(raw: object) -> list[ReplacementTransition]:
    """Accept only real transition rows from list_pending_transitions()."""
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, ReplacementTransition)]


def _tantivy_has(engine: TantivyEngine, workspace_id: str, content: str) -> bool:
    """Return True when exact-match results include ``content``."""
    finder = getattr(engine, "find_by_exact_match", None)
    if not callable(finder):
        return False
    matches = finder(workspace_id, content)
    if not isinstance(matches, list):
        return False
    hits: list[str] = []
    for match in matches:
        if isinstance(match, str):
            hits.append(match)
    return content in hits
