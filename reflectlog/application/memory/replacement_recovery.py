"""Restart-safe reconciliation of unfinished smart replacements.

SQLite archive + transition rows are one local transaction. USearch and
Tantivy commits are independent. This module applies leftover intent so
a crash between those commits cannot leave two active versions or drop
both.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import os
from typing import TYPE_CHECKING

from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import ISemanticSearchEngine, ReplacementTransition
from reflectlog.infrastructure.memory_store import MemoryStore

if TYPE_CHECKING:
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine


def reconcile_pending_replacements(
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    write_lock: AbstractContextManager[object],
    lock: AbstractContextManager[object],
    logger: IStructuredLogger,
) -> int:
    """Finish pending replacements using the semantic store as source of truth.

    Acquires ``write_lock`` before ``lock``. Idempotent: a completed
    transition, missing old memory, or already-present replacement is a
    no-op besides marking the row complete.

    Returns:
        Number of pending transitions that were applied.
    """
    store = _recovery_store(semantic_engine)
    if store is None:
        return 0

    pending = store.list_pending_transitions()
    if not pending:
        return 0

    semantic_engine.ensure_initialized()
    if tantivy_engine is not None:
        tantivy_engine.ensure_initialized()

    completed = 0
    with write_lock, lock:
        for transition in store.list_pending_transitions():
            apply_pending_transition(
                transition,
                semantic_engine=semantic_engine,
                tantivy_engine=tantivy_engine,
                logger=logger,
            )
            completed += 1

    if completed and logger:
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
    semantic_engine.delete(memory_id=str(transition.old_memory_id))
    if tantivy_engine is not None:
        _ = tantivy_engine.delete(transition.project_id, transition.old_content)

    _ensure_replacement_present(
        transition,
        semantic_engine=semantic_engine,
        tantivy_engine=tantivy_engine,
    )

    if tantivy_engine is not None:
        tantivy_engine.commit()
    semantic_engine.commit()
    semantic_engine.memory_store.complete_replacement_transition(transition.id)

    if logger:
        logger.info(
            "Applied pending replacement transition",
            extra={
                "transition_id": transition.id,
                "archive_id": transition.archive_id,
                "old_memory_id": transition.old_memory_id,
                "project_id": transition.project_id,
            },
        )


def _recovery_store(
    semantic_engine: ISemanticSearchEngine,
) -> MemoryStore | None:
    """Return the SQLite store only when a database already exists.

    Avoids creating an empty memories.db during first-time startup.
    Mock engines used in unit tests are skipped.
    """
    store = getattr(semantic_engine, "memory_store", None)
    if not isinstance(store, MemoryStore):
        return None

    if not store.db_path or not os.path.exists(store.db_path):
        return None
    return store


def _ensure_replacement_present(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> None:
    """Insert the replacement when SQLite/USearch or Tantivy still lacks it."""
    existing_id = semantic_engine.get_id_by_content(
        transition.project_id, transition.new_content
    )
    if existing_id is None:
        semantic_engine.add(
            project_id=transition.project_id,
            content=transition.new_content,
            infer=False,
        )
    else:
        _reindex_if_vector_missing(semantic_engine, existing_id, transition)

    if tantivy_engine is None:
        return
    matches = tantivy_engine.find_by_exact_match(
        transition.project_id, transition.new_content
    )
    if not matches:
        tantivy_engine.add(transition.project_id, transition.new_content)


def _reindex_if_vector_missing(
    semantic_engine: ISemanticSearchEngine,
    existing_id: int,
    transition: ReplacementTransition,
) -> None:
    """Re-add a SQLite row whose USearch vector was not committed."""
    index = getattr(semantic_engine, "index", None)
    if index is None:
        return
    try:
        vector_missing = existing_id not in index
    except TypeError:
        return
    if not vector_missing:
        return

    semantic_engine.delete(memory_id=str(existing_id))
    semantic_engine.add(
        project_id=transition.project_id,
        content=transition.new_content,
        infer=False,
    )
