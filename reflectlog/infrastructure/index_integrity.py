"""Pure id-set comparison between SQLite memory rows and live vector keys.

The vector key of a memory is its SQLite autoincrement id. A workspace is
consistent when the ids of its SQLite rows (``S``) equal the live keys of the
vector index (``V``). ``M = S - V`` are rows without a vector and
``O = V - S`` are vectors without a row.

Crash states that restart recovery repairs legitimately leave ``M`` or ``O``
non-empty, so pending journal rows authorise a *bounded* set of identities:

==========  ======================================  =============================
Kind        Allowed in ``M``                        Allowed in ``O``
==========  ======================================  =============================
ADD         current SQLite id whose content is     none (the ``old_memory_id``
            ``new_content``                         sentinel ``0`` never matches)
DELETE      ``old_memory_id`` (> 0) while its       ``old_memory_id`` (> 0)
            live row still has ``old_content``
REPLACE     current id of ``new_content`` plus     ``old_memory_id`` (> 0) only
            ``old_memory_id`` as for DELETE
==========  ======================================  =============================

Only pending rows of the same workspace authorise anything. A non-empty
journal is never a blanket exemption: every other difference is unexplained.
Later-write-wins replay stays with restart recovery; this module only decides
whether a difference is accounted for.

Nothing here touches the filesystem, SQLite, or a vector index, and nothing
here formats memory text: reports and messages carry ids and counts only.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reflectlog.core.enums import TransitionKind, TransitionStatus

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from reflectlog.core.types import ReplacementTransition

INDEX_DRIFT_OPERATOR_ACTION = (
    "Restore a consistent backup or rebuild this workspace offline."
)
MAX_REPORTED_IDS = 10


@dataclass(frozen=True)
class IndexIntegrityReport:
    """Outcome of comparing SQLite row ids with live vector keys.

    Attributes:
        sqlite_count: Number of SQLite rows in the workspace.
        vector_count: Number of live vector keys.
        missing_ids: SQLite ids without a vector (``S - V``).
        orphan_ids: Vector keys without a SQLite row (``V - S``).
        allowed_missing_ids: Subset of ``missing_ids`` a pending row accounts for.
        allowed_orphan_ids: Subset of ``orphan_ids`` a pending row accounts for.
    """

    sqlite_count: int
    vector_count: int
    missing_ids: frozenset[int]
    orphan_ids: frozenset[int]
    allowed_missing_ids: frozenset[int]
    allowed_orphan_ids: frozenset[int]

    @property
    def unexplained_missing_ids(self) -> frozenset[int]:
        """SQLite ids without a vector that no pending row accounts for."""
        return self.missing_ids - self.allowed_missing_ids

    @property
    def unexplained_orphan_ids(self) -> frozenset[int]:
        """Vector keys without a row that no pending row accounts for."""
        return self.orphan_ids - self.allowed_orphan_ids

    @property
    def is_consistent(self) -> bool:
        """True when every difference is accounted for by a pending row."""
        return not self.unexplained_missing_ids and not self.unexplained_orphan_ids


def evaluate_index_integrity(
    *,
    workspace_id: str,
    sqlite_rows: Mapping[int, str],
    vector_keys: Iterable[int],
    pending: Iterable[ReplacementTransition],
) -> IndexIntegrityReport:
    """Compare SQLite row ids with vector keys under the pending-row allowances.

    Args:
        workspace_id: Workspace the rows and journal belong to.
        sqlite_rows: Mapping of every SQLite memory id of the workspace to its
            content. Used for matching only.
        vector_keys: Live keys of the vector index.
        pending: Journal rows. Completed, cross-workspace, and unknown-kind
            rows are ignored; only pending rows of ``workspace_id`` authorise
            a difference.

    Returns:
        A frozen report with the raw and the authorised difference sets.
    """
    sqlite_ids = frozenset(sqlite_rows)
    index_ids = frozenset(vector_keys)
    missing = sqlite_ids - index_ids
    orphans = index_ids - sqlite_ids

    id_by_content = {content: row_id for row_id, content in sqlite_rows.items()}
    allowed_missing: set[int] = set()
    allowed_orphan: set[int] = set()
    for row in pending:
        if row.status != TransitionStatus.PENDING:
            continue
        if row.workspace_id != workspace_id:
            continue
        match row.kind:
            case TransitionKind.ADD:
                _allow_new_content(row, id_by_content, allowed_missing)
            case TransitionKind.DELETE:
                _allow_old_id(row, sqlite_rows, allowed_missing, allowed_orphan)
            case TransitionKind.REPLACE:
                _allow_new_content(row, id_by_content, allowed_missing)
                _allow_old_id(row, sqlite_rows, allowed_missing, allowed_orphan)

    return IndexIntegrityReport(
        sqlite_count=len(sqlite_ids),
        vector_count=len(index_ids),
        missing_ids=missing,
        orphan_ids=orphans,
        allowed_missing_ids=missing & allowed_missing,
        allowed_orphan_ids=orphans & allowed_orphan,
    )


def _allow_new_content(
    row: ReplacementTransition,
    id_by_content: Mapping[str, int],
    allowed_missing: set[int],
) -> None:
    """Authorise the current SQLite id holding the row's ``new_content``."""
    current = id_by_content.get(row.new_content)
    if current is not None:
        allowed_missing.add(current)


def _allow_old_id(
    row: ReplacementTransition,
    sqlite_rows: Mapping[int, str],
    allowed_missing: set[int],
    allowed_orphan: set[int],
) -> None:
    """Authorise the recorded old id; the sentinel ``0`` and negatives never match."""
    if row.old_memory_id <= 0:
        return
    allowed_orphan.add(row.old_memory_id)
    if sqlite_rows.get(row.old_memory_id) == row.old_content:
        allowed_missing.add(row.old_memory_id)


def index_drift_message(report: IndexIntegrityReport) -> str:
    """Build the operator-facing refusal text from a report.

    Contains counts, a bounded sorted sample of ids, and the restore-or-rebuild
    sentence. It never contains memory text.
    """
    parts = [
        "USearch vector index and SQLite memory rows disagree "
        f"({report.sqlite_count} rows, {report.vector_count} vectors)"
    ]
    parts.append(_describe_ids("rows without a vector", report.unexplained_missing_ids))
    parts.append(_describe_ids("vectors without a row", report.unexplained_orphan_ids))
    detail = "; ".join(part for part in parts if part)
    return (
        f"{detail}. No pending journal entry accounts for the difference. "
        f"{INDEX_DRIFT_OPERATOR_ACTION}"
    )


def _describe_ids(label: str, ids: frozenset[int]) -> str:
    if not ids:
        return ""
    ordered = sorted(ids)
    sample = ", ".join(str(item) for item in ordered[:MAX_REPORTED_IDS])
    extra = len(ordered) - MAX_REPORTED_IDS
    suffix = f", ... (+{extra} more)" if extra > 0 else ""
    return f"{len(ordered)} {label} (ids: {sample}{suffix})"
