"""Unit tests for the pure SQLite-versus-vector-key allowance logic."""

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from reflectlog.core.enums import TransitionKind, TransitionStatus
from reflectlog.core.types import ReplacementTransition
from reflectlog.infrastructure.index_integrity import (
    INDEX_DRIFT_OPERATOR_ACTION,
    MAX_REPORTED_IDS,
    IndexIntegrityReport,
    evaluate_index_integrity,
    index_drift_message,
)

WS = "ws"
SECRET = "SECRET-MEMORY-TEXT"


def _row(
    kind: TransitionKind,
    *,
    old_id: int = 0,
    old: str = "",
    new: str = "",
    status: TransitionStatus = TransitionStatus.PENDING,
    workspace: str = WS,
    row_id: int = 1,
) -> ReplacementTransition:
    return ReplacementTransition(
        id=row_id,
        workspace_id=workspace,
        old_memory_id=old_id,
        old_content=old,
        new_content=new,
        archive_id=0,
        reason="test",
        confidence=1.0,
        status=status,
        kind=kind,
    )


def _add(
    new: str,
    *,
    status: TransitionStatus = TransitionStatus.PENDING,
    workspace: str = WS,
    row_id: int = 1,
) -> ReplacementTransition:
    return _row(
        TransitionKind.ADD,
        new=new,
        status=status,
        workspace=workspace,
        row_id=row_id,
    )


def _delete(old_id: int, old: str, *, row_id: int = 1) -> ReplacementTransition:
    return _row(TransitionKind.DELETE, old_id=old_id, old=old, row_id=row_id)


def _replace(
    old_id: int, old: str, new: str, *, row_id: int = 1
) -> ReplacementTransition:
    return _row(TransitionKind.REPLACE, old_id=old_id, old=old, new=new, row_id=row_id)


@dataclass(frozen=True)
class Case:
    sqlite_rows: Mapping[int, str]
    vector_keys: frozenset[int]
    pending: tuple[ReplacementTransition, ...] = ()
    missing: frozenset[int] = frozenset()
    orphans: frozenset[int] = frozenset()
    allowed_missing: frozenset[int] = frozenset()
    allowed_orphans: frozenset[int] = frozenset()
    consistent: bool = False


def _ids(*values: int) -> frozenset[int]:
    return frozenset(values)


CASES: dict[str, Case] = {
    "empty_both_sides": Case({}, _ids(), consistent=True),
    "ids_agree": Case({1: "a", 2: "b"}, _ids(1, 2), consistent=True),
    "missing_vector_no_journal": Case({1: "a", 2: "b"}, _ids(1), missing=_ids(2)),
    "orphan_vector_no_journal": Case({1: "a"}, _ids(1, 5), orphans=_ids(5)),
    "equal_count_swapped_ids": Case(
        {1: "a", 2: "b"}, _ids(1, 3), missing=_ids(2), orphans=_ids(3)
    ),
    "all_vectors_missing": Case({1: "a"}, _ids(), missing=_ids(1)),
    "all_rows_missing": Case({}, _ids(4), orphans=_ids(4)),
    # ADD
    "add_authorises_current_id_of_new_content": Case(
        {1: "a", 2: "new"},
        _ids(1),
        (_add("new"),),
        missing=_ids(2),
        allowed_missing=_ids(2),
        consistent=True,
    ),
    "add_does_not_authorise_other_missing_id": Case(
        {1: "a", 2: "new", 3: "unrelated"},
        _ids(1),
        (_add("new"),),
        missing=_ids(2, 3),
        allowed_missing=_ids(2),
    ),
    "add_never_authorises_an_orphan": Case(
        {1: "a", 2: "new"},
        _ids(1, 2, 9),
        (_add("new"),),
        orphans=_ids(9),
    ),
    "add_sentinel_zero_never_authorises_vector_zero": Case(
        {1: "a"},
        _ids(0, 1),
        (_add("a"),),
        orphans=_ids(0),
    ),
    "add_for_text_without_a_row_authorises_nothing": Case(
        {1: "a", 2: "b"},
        _ids(1),
        (_add("not stored"),),
        missing=_ids(2),
    ),
    # DELETE
    "delete_authorises_recorded_old_id_orphan": Case(
        {1: "a"},
        _ids(1, 5),
        (_delete(5, "gone"),),
        orphans=_ids(5),
        allowed_orphans=_ids(5),
        consistent=True,
    ),
    "delete_authorises_missing_when_live_row_matches_old_content": Case(
        {1: "a", 5: "gone"},
        _ids(1),
        (_delete(5, "gone"),),
        missing=_ids(5),
        allowed_missing=_ids(5),
        consistent=True,
    ),
    "delete_missing_not_authorised_when_live_row_differs": Case(
        {1: "a", 5: "something else"},
        _ids(1),
        (_delete(5, "gone"),),
        missing=_ids(5),
    ),
    "delete_old_text_readded_under_other_id_is_not_authorised": Case(
        {1: "a", 9: "gone"},
        _ids(1, 5),
        (_delete(5, "gone"),),
        missing=_ids(9),
        orphans=_ids(5),
        allowed_orphans=_ids(5),
    ),
    "delete_does_not_authorise_other_orphan": Case(
        {1: "a"},
        _ids(1, 5, 6),
        (_delete(5, "gone"),),
        orphans=_ids(5, 6),
        allowed_orphans=_ids(5),
    ),
    "delete_sentinel_zero_authorises_nothing": Case(
        {1: "a"},
        _ids(0, 1),
        (_delete(0, "ghost"),),
        orphans=_ids(0),
    ),
    "delete_negative_id_authorises_nothing": Case(
        {1: "a"},
        _ids(1, 7),
        (_delete(-3, "ghost"),),
        orphans=_ids(7),
    ),
    # REPLACE
    "replace_authorises_new_content_missing_and_old_orphan": Case(
        {1: "a", 2: "new"},
        _ids(1, 7),
        (_replace(7, "old", "new"),),
        missing=_ids(2),
        orphans=_ids(7),
        allowed_missing=_ids(2),
        allowed_orphans=_ids(7),
        consistent=True,
    ),
    "replace_authorises_old_missing_when_live_row_matches": Case(
        {1: "a", 7: "old", 8: "new"},
        _ids(1),
        (_replace(7, "old", "new"),),
        missing=_ids(7, 8),
        allowed_missing=_ids(7, 8),
        consistent=True,
    ),
    "replace_does_not_authorise_an_arbitrary_new_orphan": Case(
        {1: "a", 2: "new"},
        _ids(1, 2, 7, 99),
        (_replace(7, "old", "new"),),
        orphans=_ids(7, 99),
        allowed_orphans=_ids(7),
    ),
    "replace_old_missing_not_authorised_when_live_row_differs": Case(
        {1: "a", 7: "edited", 8: "new"},
        _ids(1, 8),
        (_replace(7, "old", "new"),),
        missing=_ids(7),
    ),
    # Journal filtering
    "completed_row_authorises_nothing": Case(
        {1: "a", 2: "new"},
        _ids(1),
        (_add("new", status=TransitionStatus.COMPLETED),),
        missing=_ids(2),
    ),
    "cross_workspace_row_authorises_nothing": Case(
        {1: "a", 2: "new"},
        _ids(1),
        (_add("new", workspace="other"),),
        missing=_ids(2),
    ),
    "pending_journal_cannot_mask_unrelated_drift": Case(
        {1: "a", 2: "new", 3: "unrelated"},
        _ids(1, 42),
        (_add("new"), _delete(5, "gone")),
        missing=_ids(2, 3),
        orphans=_ids(42),
        allowed_missing=_ids(2),
    ),
    "several_rows_union_their_allowances": Case(
        {1: "a", 2: "n1", 3: "n2"},
        _ids(1, 5, 6),
        (_add("n1"), _replace(5, "o", "n2", row_id=2), _delete(6, "d", row_id=3)),
        missing=_ids(2, 3),
        orphans=_ids(5, 6),
        allowed_missing=_ids(2, 3),
        allowed_orphans=_ids(5, 6),
        consistent=True,
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_allowance_table(name: str) -> None:
    case = CASES[name]

    report = evaluate_index_integrity(
        workspace_id=WS,
        sqlite_rows=case.sqlite_rows,
        vector_keys=case.vector_keys,
        pending=case.pending,
    )

    assert report.missing_ids == case.missing
    assert report.orphan_ids == case.orphans
    assert report.allowed_missing_ids == case.allowed_missing
    assert report.allowed_orphan_ids == case.allowed_orphans
    assert report.unexplained_missing_ids == case.missing - case.allowed_missing
    assert report.unexplained_orphan_ids == case.orphans - case.allowed_orphans
    assert report.is_consistent is case.consistent
    assert report.sqlite_count == len(case.sqlite_rows)
    assert report.vector_count == len(case.vector_keys)


def test_equal_count_swap_is_detected_even_though_totals_match() -> None:
    report = evaluate_index_integrity(
        workspace_id=WS,
        sqlite_rows={1: "a", 2: "b"},
        vector_keys=[1, 3],
        pending=[],
    )

    assert report.sqlite_count == report.vector_count == 2
    assert not report.is_consistent


def test_vector_keys_may_be_any_iterable_of_ints() -> None:
    report = evaluate_index_integrity(
        workspace_id=WS,
        sqlite_rows={1: "a", 2: "b"},
        vector_keys=iter([2, 1]),
        pending=iter([]),
    )

    assert report.is_consistent


def _drift_report(missing: int = 0, orphans: int = 0) -> IndexIntegrityReport:
    rows = {row_id: f"{SECRET}-{row_id}" for row_id in range(1, missing + 1)}
    keys = [1000 + offset for offset in range(orphans)]
    return evaluate_index_integrity(
        workspace_id=WS, sqlite_rows=rows, vector_keys=keys, pending=[]
    )


def test_message_carries_operator_sentence_counts_and_ids() -> None:
    message = index_drift_message(_drift_report(missing=2, orphans=1))

    assert INDEX_DRIFT_OPERATOR_ACTION == (
        "Restore a consistent backup or rebuild this workspace offline."
    )
    assert INDEX_DRIFT_OPERATOR_ACTION in message
    assert "2 rows without a vector (ids: 1, 2)" in message
    assert "1 vectors without a row (ids: 1000)" in message
    assert "2 rows, 1 vectors" in message


def test_message_never_contains_memory_text() -> None:
    report = _drift_report(missing=3, orphans=2)

    assert SECRET not in index_drift_message(report)
    assert SECRET not in repr(report)


def test_message_samples_a_bounded_number_of_ids() -> None:
    report = _drift_report(missing=MAX_REPORTED_IDS + 5)

    message = index_drift_message(report)

    assert f"{MAX_REPORTED_IDS + 5} rows without a vector" in message
    assert "(+5 more)" in message
    assert str(MAX_REPORTED_IDS + 5) not in message.split("ids:")[1].split(")")[0]


def test_message_lists_only_unexplained_ids() -> None:
    report = evaluate_index_integrity(
        workspace_id=WS,
        sqlite_rows={1: "a", 2: "new", 3: "other"},
        vector_keys=[1],
        pending=[_add("new")],
    )

    message = index_drift_message(report)

    assert "1 rows without a vector (ids: 3)" in message
