"""Restart-safe reconciliation of unfinished smart replacements.

SQLite archive + transition rows are one local transaction. USearch and
Tantivy commits are independent. Recovery *attempts* to converge leftover
intent; it can be skipped, and it will not mark a row complete while
SQLite or hybrid Tantivy still disagree.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
import os
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from reflectlog.core.enums import SearchComponent, TransitionKind
from reflectlog.core.exceptions import InitializationError, StorageError
from reflectlog.core.storage_coordination import IStorageCoordinator, LeaseMode
from reflectlog.core.types import (
    IArchiveMemoryStore,
    IMemorySnapshotReader,
    ISemanticSearchEngine,
    MemoryStoreSnapshot,
    ReplacementTransition,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from reflectlog.core.logging import IStructuredLogger
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine


@runtime_checkable
class _RefreshableEngine(Protocol):
    def refresh(self) -> None: ...


def _refresh_engine(engine: object) -> None:
    if isinstance(engine, _RefreshableEngine):
        engine.refresh()


@contextmanager
def _refusal_of(
    component: SearchComponent,
    on_refusal: Callable[[SearchComponent, InitializationError], None] | None,
) -> Generator[None]:
    """Report an ``InitializationError`` from one engine call, then re-raise it.

    Attribution is by call site: wrap exactly one engine's call so the refusal
    is reported once, against the component that raised it.
    """
    try:
        yield
    except InitializationError as refusal:
        if on_refusal is not None:
            on_refusal(component, refusal)
        raise


def _complete_converged_intent(
    store: IArchiveMemoryStore,
    transition_id: int,
    *,
    coordinator: IStorageCoordinator | None,
    workspace_id: str,
    hook: Callable[[str], None] | None,
    leftover_add_content: str = "",
) -> None:
    """Publish generation, then complete leftover ADDs and the durable intent."""
    if coordinator is not None and workspace_id:
        if hook is not None:
            hook("before_generation")
        generation = coordinator.read_generation(workspace_id)
        coordinator.publish_generation(workspace_id, generation + 1)
        if hook is not None:
            hook("after_generation")
    if leftover_add_content:
        _complete_pending_adds_of(
            store,
            workspace_id=workspace_id,
            content=leftover_add_content,
        )
    if hook is not None:
        hook("before_intent")
    store.complete_replacement_transition(transition_id)
    if hook is not None:
        hook("after_intent")


def reconcile_pending_replacements(
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    write_lock: AbstractContextManager[object],
    lock: AbstractContextManager[object] | None = None,
    logger: IStructuredLogger,
    coordinator: IStorageCoordinator | None = None,
    workspace_id: str = "",
    orchestration_hook: Callable[[str], None] | None = None,
    on_refusal: Callable[[SearchComponent, InitializationError], None] | None = None,
) -> int:
    """Finish pending replacements using the semantic store as source of truth.

    Acquires ``write_lock`` before ``lock`` when both are provided.

    ``on_refusal`` is called once, before the error is re-raised unchanged,
    when an engine refuses to open or verify: with the semantic or tantivy
    component for the engine call that raised, or ``SEMANTIC`` for a refusal
    raised while converging a row (the tantivy engine opened successfully
    above and only refuses at construction, so that is the vector index).
    Callers that do not count refusals leave it unset.

    Returns:
        Number of pending transitions that were marked complete.
    """
    store = _recovery_store(semantic_engine, logger)
    if store is None:
        return 0

    # Listing through the writable store would switch an older database to WAL
    # and migrate its schema before the vector index has been verified, so a
    # workspace the drift check refuses would still be modified. Look at the
    # journal read-only first. Only a real snapshot is trusted; an engine
    # without a reader takes the original path.
    if isinstance(semantic_engine, IMemorySnapshotReader):
        try:
            snapshot = _real_snapshot(semantic_engine.read_memory_snapshot())
        except StorageError:
            # The journal cannot be read without recovery (a crashed
            # transaction's hot journal, a damaged file). Verify the index
            # before the writable store can touch the file: a refusal then
            # leaves it as found, and an empty or new index passes through.
            with _refusal_of(SearchComponent.SEMANTIC, on_refusal):
                semantic_engine.ensure_initialized()
        else:
            if snapshot is not None:
                if (
                    not snapshot.pending_transitions
                    and not snapshot.unrecognized_pending_count
                ):
                    return 0
                # Opens the index and refuses drift before the writable store
                # opens.
                with _refusal_of(SearchComponent.SEMANTIC, on_refusal):
                    semantic_engine.ensure_initialized()

    pending = _pending_rows(store.list_pending_transitions())
    if not pending:
        return 0

    with _refusal_of(SearchComponent.SEMANTIC, on_refusal):
        semantic_engine.ensure_initialized()
    if tantivy_engine is not None:
        with _refusal_of(SearchComponent.TANTIVY, on_refusal):
            tantivy_engine.ensure_initialized()

    precomputed = _precompute_add_vectors(pending, semantic_engine, logger)

    completed = 0
    inner_lock = lock if lock is not None else nullcontext()
    lease = (
        coordinator.acquire(workspace_id, LeaseMode.EXCLUSIVE)
        if coordinator is not None and workspace_id
        else nullcontext()
    )
    with lease, write_lock, inner_lock:
        with _refusal_of(SearchComponent.SEMANTIC, on_refusal):
            semantic_engine.ensure_initialized()
        if tantivy_engine is not None:
            with _refusal_of(SearchComponent.TANTIVY, on_refusal):
                tantivy_engine.ensure_initialized()
        with _refusal_of(SearchComponent.SEMANTIC, on_refusal):
            _refresh_engine(semantic_engine)
        if tantivy_engine is not None:
            with _refusal_of(SearchComponent.TANTIVY, on_refusal):
                _refresh_engine(tantivy_engine)
        snapshot = _pending_rows(store.list_pending_transitions())
        for transition in snapshot:
            try:
                if not store.is_pending_transition(transition.id):
                    continue
                if apply_pending_transition(
                    transition,
                    semantic_engine=semantic_engine,
                    tantivy_engine=tantivy_engine,
                    logger=logger,
                    precomputed_vectors=precomputed,
                    coordinator=coordinator,
                    orchestration_hook=orchestration_hook,
                ):
                    completed += 1
            except InitializationError as refusal:
                if on_refusal is not None:
                    on_refusal(SearchComponent.SEMANTIC, refusal)
                raise
            except Exception as exc:
                logger.error(
                    "Skipping pending replacement after recovery error",
                    extra={
                        "transition_id": transition.id,
                        "error": str(exc),
                    },
                )
        if snapshot:
            # Recovery must leave the ids consistent. Completed intents now
            # authorize nothing and any still-pending one keeps only its narrow
            # allowances. No lease or lock is taken here: all are held already.
            with _refusal_of(SearchComponent.SEMANTIC, on_refusal):
                semantic_engine.verify_index_integrity()

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
    precomputed_vectors: dict[str, list[float]] | None = None,
    coordinator: IStorageCoordinator | None = None,
    orchestration_hook: Callable[[str], None] | None = None,
) -> bool:
    """Converge both indexes to the replacement recorded in ``transition``.

    Returns:
        True when the transition was marked complete.
    """
    store = semantic_engine.memory_store
    if not store.is_pending_transition(transition.id):
        return False

    if transition.kind == TransitionKind.ADD:
        return _apply_pending_add(
            transition,
            semantic_engine=semantic_engine,
            tantivy_engine=tantivy_engine,
            logger=logger,
            precomputed_vectors=precomputed_vectors,
            coordinator=coordinator,
            orchestration_hook=orchestration_hook,
        )
    if transition.kind == TransitionKind.DELETE:
        return _apply_pending_delete(
            transition,
            semantic_engine=semantic_engine,
            tantivy_engine=tantivy_engine,
            logger=logger,
            coordinator=coordinator,
            orchestration_hook=orchestration_hook,
        )

    if _later_intent_exists(
        store, transition, kind=TransitionKind.DELETE, content=transition.new_content
    ):
        _remove_recorded_old(transition, semantic_engine, tantivy_engine)
        if tantivy_engine is not None:
            tantivy_engine.commit()
        semantic_engine.commit()
        if not _delete_converged(
            transition,
            semantic_engine,
            tantivy_engine,
            later_add=False,
        ):
            logger.warning(
                "Superseded replace not complete; old text still live",
                extra={"transition_id": transition.id},
            )
            return False
        _complete_converged_intent(
            store,
            transition.id,
            coordinator=coordinator,
            workspace_id=transition.workspace_id,
            hook=orchestration_hook,
        )
        logger.info(
            "Completed replace intent superseded by a later delete or replace",
            extra={"transition_id": transition.id},
        )
        return True

    replacement_live = _ensure_replacement_present(
        transition,
        semantic_engine=semantic_engine,
        tantivy_engine=tantivy_engine,
        precomputed_vectors=precomputed_vectors,
    )
    if not replacement_live:
        return False
    _remove_recorded_old(transition, semantic_engine, tantivy_engine)

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
        return False

    _complete_converged_intent(
        store,
        transition.id,
        coordinator=coordinator,
        workspace_id=transition.workspace_id,
        hook=orchestration_hook,
        leftover_add_content=transition.old_content,
    )
    logger.info(
        "Applied pending replacement transition",
        extra={
            "transition_id": transition.id,
            "archive_id": transition.archive_id,
            "old_memory_id": transition.old_memory_id,
            "workspace_id": transition.workspace_id,
        },
    )
    return True


def replacement_converged(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> bool:
    """Return True when NEW is live, the recorded OLD id is gone, and Tantivy agrees."""
    new_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if new_id is None:
        return False
    if not _vector_present(semantic_engine, new_id):
        return False
    if not _old_id_gone(transition, semantic_engine):
        return False

    if tantivy_engine is None:
        return True
    if not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.new_content
    ):
        return False
    if not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.old_content
    ):
        return True
    return _old_text_live_under_new_id(transition, semantic_engine)


def _old_id_gone(
    transition: ReplacementTransition, semantic_engine: ISemanticSearchEngine
) -> bool:
    """Return True when the recorded old id is gone from SQLite and USearch."""
    if _sqlite_id_for(transition, semantic_engine, transition.old_content) == (
        transition.old_memory_id
    ):
        return False
    return _vector_absent(semantic_engine, transition.old_memory_id)


def _old_text_live_under_new_id(
    transition: ReplacementTransition, semantic_engine: ISemanticSearchEngine
) -> bool:
    """Return True when old text is stored under a different live id."""
    current_id = _sqlite_id_for(transition, semantic_engine, transition.old_content)
    return current_id is not None and current_id != transition.old_memory_id


def _sqlite_id_for(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    content: str,
) -> int | None:
    """Look up the live SQLite id for ``content`` in this transition's workspace."""
    return semantic_engine.get_id_by_content(transition.workspace_id, content)


def _apply_pending_add(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    logger: IStructuredLogger,
    precomputed_vectors: dict[str, list[float]] | None = None,
    coordinator: IStorageCoordinator | None = None,
    orchestration_hook: Callable[[str], None] | None = None,
) -> bool:
    """Ensure NEW content exists unless a later delete/replace of that text won."""
    store = semantic_engine.memory_store
    if _later_intent_exists(
        store, transition, kind=TransitionKind.DELETE, content=transition.new_content
    ):
        _complete_converged_intent(
            store,
            transition.id,
            coordinator=coordinator,
            workspace_id=transition.workspace_id,
            hook=orchestration_hook,
        )
        logger.info(
            "Completed add intent superseded by a later delete or replace",
            extra={"transition_id": transition.id},
        )
        return True

    existing_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    vector = (
        None
        if precomputed_vectors is None
        else precomputed_vectors.get(transition.new_content)
    )
    if existing_id is None:
        if precomputed_vectors is not None and vector is None:
            logger.warning(
                "Add intent not complete; precomputed vector missing",
                extra={"transition_id": transition.id},
            )
            return False
        _insert_recovered_add(semantic_engine, transition, vector=vector)
    else:
        if (
            precomputed_vectors is not None
            and not _vector_present(semantic_engine, existing_id)
            and vector is None
        ):
            logger.warning(
                "Add intent not complete; precomputed vector missing",
                extra={"transition_id": transition.id},
            )
            return False
        _reindex_if_vector_missing(
            semantic_engine,
            existing_id,
            transition,
            vector=vector,
        )

    if tantivy_engine is not None and not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.new_content
    ):
        tantivy_engine.add(transition.workspace_id, transition.new_content)

    if tantivy_engine is not None:
        tantivy_engine.commit()
    semantic_engine.commit()

    if not _add_converged(transition, semantic_engine, tantivy_engine):
        logger.warning(
            "Add intent not complete; indexes have not converged",
            extra={"transition_id": transition.id},
        )
        return False
    _complete_converged_intent(
        store,
        transition.id,
        coordinator=coordinator,
        workspace_id=transition.workspace_id,
        hook=orchestration_hook,
    )
    return True


def _apply_pending_delete(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    logger: IStructuredLogger,
    coordinator: IStorageCoordinator | None = None,
    orchestration_hook: Callable[[str], None] | None = None,
) -> bool:
    """Remove the recorded old id; do not wipe a later re-add of the same text."""
    store = semantic_engine.memory_store
    later_add = _later_intent_exists(
        store, transition, kind=TransitionKind.ADD, content=transition.old_content
    )
    semantic_engine.delete(memory_id=str(transition.old_memory_id))
    if (
        tantivy_engine is not None
        and not later_add
        and not _old_text_live_under_new_id(transition, semantic_engine)
    ):
        _ = tantivy_engine.delete(
            transition.workspace_id,
            transition.old_content,
            verify_exists=False,
        )

    if tantivy_engine is not None:
        tantivy_engine.commit()
    semantic_engine.commit()

    if not _delete_converged(
        transition,
        semantic_engine,
        tantivy_engine,
        later_add=later_add,
    ):
        logger.warning(
            "Delete intent not complete; old id or Tantivy copy still present",
            extra={"transition_id": transition.id},
        )
        return False
    _complete_converged_intent(
        store,
        transition.id,
        coordinator=coordinator,
        workspace_id=transition.workspace_id,
        hook=orchestration_hook,
    )
    return True


def _add_converged(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> bool:
    """Return True when the added content is live in every required backend."""
    new_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if new_id is None or not _vector_present(semantic_engine, new_id):
        return False
    if tantivy_engine is None:
        return True
    return _tantivy_has(tantivy_engine, transition.workspace_id, transition.new_content)


def _delete_converged(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    *,
    later_add: bool,
) -> bool:
    """Return True when the recorded old id is gone and Tantivy agrees."""
    if not _old_id_gone(transition, semantic_engine):
        return False
    if tantivy_engine is None or later_add:
        return True
    if _old_text_live_under_new_id(transition, semantic_engine):
        return True
    return not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.old_content
    )


def _complete_pending_adds_of(
    store: IArchiveMemoryStore,
    *,
    workspace_id: str,
    content: str,
) -> None:
    """Complete leftover pending ADD rows for text that a replace just removed.

    Snapshot-visible leftover ADDs of the old text (including ``id`` greater
    than the replace) must not be applied after this replace converges.
    A genuine later ADD is journaled only after this replace is complete.
    """
    if not content:
        return
    for row in store.list_pending_transitions():
        if row.workspace_id != workspace_id:
            continue
        if row.kind != TransitionKind.ADD:
            continue
        if row.new_content != content:
            continue
        store.complete_replacement_transition(row.id)


def _later_intent_exists(
    store: IArchiveMemoryStore,
    transition: ReplacementTransition,
    *,
    kind: TransitionKind,
    content: str,
) -> bool:
    return store.has_later_intent(
        workspace_id=transition.workspace_id,
        kind=kind,
        content=content,
        after_id=transition.id,
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

    if _old_text_live_under_new_id(transition, semantic_engine):
        return
    _ = tantivy_engine.delete(
        transition.workspace_id,
        transition.old_content,
        verify_exists=False,
    )


def _real_snapshot(raw: object) -> MemoryStoreSnapshot | None:
    """Accept only a real snapshot, like ``_pending_rows`` for journal rows.

    The snapshot is filtered to the engine's workspace while
    ``list_pending_transitions`` lists every workspace; a database file belongs
    to exactly one workspace, so the two agree.
    """
    return raw if isinstance(raw, MemoryStoreSnapshot) else None


def _recovery_store(
    semantic_engine: ISemanticSearchEngine,
    logger: IStructuredLogger,
) -> IArchiveMemoryStore | None:
    """Return a transition store when pending rows can be listed."""
    store = semantic_engine.memory_store
    if store.db_path and not os.path.exists(store.db_path):
        return None
    return store


def _ensure_replacement_present(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    precomputed_vectors: dict[str, list[float]] | None = None,
) -> bool:
    """Ensure the replacement has both SQLite identity and a live vector."""
    existing_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    vector = (
        None
        if precomputed_vectors is None
        else precomputed_vectors.get(transition.new_content)
    )
    if existing_id is None:
        if precomputed_vectors is not None and vector is None:
            return False
        _insert_recovered_add(semantic_engine, transition, vector=vector)
    elif not _vector_present(semantic_engine, existing_id):
        if precomputed_vectors is not None and vector is None:
            return False
        _reindex_if_vector_missing(
            semantic_engine, existing_id, transition, vector=vector
        )

    replacement_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if replacement_id is None or not _vector_present(semantic_engine, replacement_id):
        return False
    if tantivy_engine is None:
        return True
    if not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.new_content
    ):
        tantivy_engine.add(transition.workspace_id, transition.new_content)
    return True


def _reindex_if_vector_missing(
    semantic_engine: ISemanticSearchEngine,
    existing_id: int,
    transition: ReplacementTransition,
    vector: list[float] | None = None,
) -> None:
    """Re-add a SQLite row whose USearch vector was not committed."""
    if _vector_present(semantic_engine, existing_id):
        return

    semantic_engine.delete(memory_id=str(existing_id))
    if vector is not None:
        _ = semantic_engine.add_batch(
            transition.workspace_id,
            [transition.new_content],
            infer=False,
            vectors=[vector],
        )
        return
    semantic_engine.add(
        workspace_id=transition.workspace_id,
        content=transition.new_content,
        infer=False,
    )


def _vector_present(semantic_engine: ISemanticSearchEngine, memory_id: int) -> bool:
    """Return True only when a real index contains ``memory_id``."""
    return semantic_engine.contains_id(memory_id) is True


def _vector_absent(semantic_engine: ISemanticSearchEngine, memory_id: int) -> bool:
    """Return True only when a real index is missing ``memory_id``."""
    return semantic_engine.contains_id(memory_id) is False


def _precompute_add_vectors(
    pending: list[ReplacementTransition],
    semantic_engine: ISemanticSearchEngine,
    logger: IStructuredLogger,
) -> dict[str, list[float]]:
    """Embed missing add/replace text outside the write lock."""
    embedder = semantic_engine.embedder
    needed: list[str] = []
    seen: set[str] = set()
    for transition in pending:
        if transition.kind not in {TransitionKind.ADD, TransitionKind.REPLACE}:
            continue
        content = transition.new_content
        if not content or content in seen:
            continue
        existing_id = semantic_engine.get_id_by_content(
            transition.workspace_id, content
        )
        if existing_id is not None and _vector_present(semantic_engine, existing_id):
            continue
        seen.add(content)
        needed.append(content)
    if not needed:
        return {}

    try:
        raw_vectors = embedder.embed_documents(needed)
    except Exception as exc:
        logger.warning(
            "Pre-embed for recovery add failed; leaving intent pending",
            extra={"error": str(exc)},
        )
        return {}
    if len(raw_vectors) != len(needed):
        logger.warning(
            "Pre-embed for recovery add failed; leaving intent pending",
            extra={"error": "Embedding batch size mismatch for recovery add"},
        )
        return {}

    vectors: dict[str, list[float]] = {}
    for content, raw in zip(needed, raw_vectors, strict=True):
        converted = _as_floats(raw)
        if converted is None:
            logger.warning(
                "Pre-embed for recovery add failed; leaving intent pending",
                extra={"error": "Embedding batch contained an empty vector"},
            )
            return {}
        vectors[content] = converted
    return vectors


def _insert_recovered_add(
    semantic_engine: ISemanticSearchEngine,
    transition: ReplacementTransition,
    *,
    vector: list[float] | None,
) -> None:
    """Insert recovered content, preferring a precomputed vector."""
    if vector is not None:
        _ = semantic_engine.add_batch(
            transition.workspace_id,
            [transition.new_content],
            infer=False,
            vectors=[vector],
        )
        return
    semantic_engine.add(
        workspace_id=transition.workspace_id,
        content=transition.new_content,
        infer=False,
    )


def _pending_rows(raw: object) -> list[ReplacementTransition]:
    """Accept only real transition rows from list_pending_transitions()."""
    rows: list[ReplacementTransition] = []
    for item in _as_objects(raw):
        if isinstance(item, ReplacementTransition):
            rows.append(item)
    return rows


def _as_objects(raw: object) -> list[object]:
    """Treat a dynamic list result as ``list[object]``."""
    if not isinstance(raw, list):
        return []
    return cast("list[object]", raw)


def _as_floats(raw: object) -> list[float] | None:
    """Return a float list when ``raw`` is a non-empty sequence of numbers."""
    items = _as_objects(raw)
    if not items:
        return None
    values: list[float] = []
    for item in items:
        if not isinstance(item, (int, float)):
            return None
        values.append(float(item))
    return values


def _tantivy_has(engine: TantivyEngine, workspace_id: str, content: str) -> bool:
    """Return True when exact-match results include ``content``."""
    return content in engine.find_by_exact_match(workspace_id, content)
